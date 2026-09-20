"""The invariants a fuzzer checks (audit priority 9).

A fuzzer without oracles only finds crashes, and crashes are the *easy* class of
bug.  The interesting failures in a compiler are the ones where it produces an
answer that is wrong, or refuses a program with a diagnostic that does not
describe the real problem, or behaves differently when run twice.  Each check
below names one such property and says which part of the specification asks for
it.

The distinction that makes this work at all: `GamaError` and `GamaRuntimeFault`
are the language's own error channel, so a program raising one has been *handled*.
Anything else that escapes the public API is a bug in the toolchain, whatever it
happens to be called.
"""

from __future__ import annotations

import io
import os
import random
import re
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..diagnostics import GamaError, GamaRuntimeFault
from ..driver import (compile_file, compile_source, execute, find_entry,
                      program_grants)
from ..runtime.context import Context

CODE_RE = re.compile(r"^[EW]-[a-z0-9][a-z0-9-]*$")


@dataclass
class Violation:
    """One invariant that did not hold."""

    invariant: str
    severity: str          # crash | wrong | inconsistent | noise
    detail: str
    source: str = ""
    origin: str = ""
    #: How the case was produced, so a report can say whether generation or
    #: mutation found it.
    kind: str = ""
    signature: str = ""
    traceback: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = f"{self.invariant}:{self.detail[:120]}"

    def render(self) -> List[str]:
        head = f"[{self.severity}] {self.invariant}"
        if self.origin:
            head += f"  ({self.origin})"
        out = [head, f"    {self.detail}"]
        if self.kind:
            out.append(f"    produced by: {self.kind}")
        if self.traceback:
            out.append("    traceback:")
            for line in self.traceback.strip().splitlines()[-6:]:
                out.append("      " + line)
        return out


def _is_handled(exc: BaseException) -> bool:
    """Whether an exception is the language's own error channel."""
    return isinstance(exc, (GamaError, GamaRuntimeFault))


def _tb(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc,
                                              exc.__traceback__))


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_compiles_or_explains(source: str, path: str = "<fuzz>",
                               profile: str = "strict",
                               opt_level: int = 1,
                               origin: str = "",
                               kind: str = "") -> List[Violation]:
    """The compiler must compile a program or explain why.  Nothing else.

    This is the single most valuable check here, and it is not about
    correctness of the answer: it is that `compile_source` is a total function
    over byte strings.  Any input at all -- truncated, binary, adversarial --
    must come back as a Compilation with a verdict.
    """
    out: List[Violation] = []
    try:
        compilation = compile_source(source, path, profile=profile,
                                     opt_level=opt_level)
    except BaseException as exc:                     # noqa: BLE001
        if _is_handled(exc):
            # A GamaError escaping the driver is still a bug: the driver's job
            # is to catch it and turn it into a diagnostic.
            out.append(Violation(
                "compiles-or-explains", "crash",
                f"the driver let its own error type escape instead of turning "
                f"it into a diagnostic: {type(exc).__name__}: {exc}",
                source, origin, kind, traceback=_tb(exc)))
        else:
            out.append(Violation(
                "compiles-or-explains", "crash",
                f"{type(exc).__name__}: {exc}", source, origin, kind,
                traceback=_tb(exc)))
        return out

    if not compilation.ok:
        diagnostics = list(compilation.bag.diagnostics) if compilation.bag \
            else []
        if not diagnostics:
            out.append(Violation(
                "compiles-or-explains", "wrong",
                "the program was rejected with no diagnostic at all, so the "
                "user has nothing to act on",
                source, origin, kind))
        if not compilation.stopped_at:
            out.append(Violation(
                "compiles-or-explains", "inconsistent",
                "the compilation failed but did not record which stage stopped",
                source, origin, kind))
        out.extend(_check_diagnostics(diagnostics, source, origin, kind))

    if compilation.ok:
        out.extend(_check_models(compilation, source, origin, kind))
    return out


def _check_diagnostics(diagnostics: Sequence[Any], source: str,
                       origin: str, kind: str) -> List[Violation]:
    """Every diagnostic must be actionable: a code, a message, and usually a place."""
    out: List[Violation] = []
    for diagnostic in diagnostics:
        code = getattr(diagnostic, "code", "") or ""
        message = getattr(diagnostic, "message", "") or ""
        if not code:
            out.append(Violation(
                "diagnostic-has-a-code", "noise",
                f"a diagnostic with no code: {message[:80]!r}",
                source, origin, kind))
        elif not CODE_RE.match(code):
            out.append(Violation(
                "diagnostic-code-is-well-formed", "noise",
                f"code {code!r} does not match `E-`/`W-` plus lowercase words, "
                f"so it cannot be filtered or looked up",
                source, origin, kind))
        if not message.strip():
            out.append(Violation(
                "diagnostic-has-a-message", "noise",
                f"diagnostic {code!r} has an empty message",
                source, origin, kind))
        if len(message) > 4000:
            out.append(Violation(
                "diagnostic-message-is-bounded", "noise",
                f"diagnostic {code!r} has a {len(message)}-character message, "
                f"which usually means it echoed the input back",
                source, origin, kind))
    return out


def _check_models(compilation: Any, source: str, origin: str,
                  kind: str) -> List[Violation]:
    """A program that compiled must satisfy the models v0.4 built."""
    out: List[Violation] = []
    memory = getattr(compilation, "core_memory", None)
    if memory is not None:
        violations = list(getattr(memory, "violations", []) or [])
        if violations:
            out.append(Violation(
                "memory-model-is-consistent", "wrong",
                f"a program that compiled violates its own memory model: "
                f"{violations[:3]}", source, origin, kind))
    recovery = getattr(compilation, "core_recovery", None)
    if recovery is not None:
        problems = list(getattr(recovery, "problems", []) or [])
        if problems:
            out.append(Violation(
                "recovery-policy-is-valid", "wrong",
                f"a program that compiled has an invalid recovery policy: "
                f"{problems[:2]}", source, origin, kind))
    return out


#: How many VM instructions a fuzz case may execute before the runtime faults
#: with StepLimitExceeded.
#:
#: The language already bounds execution -- `Context.max_steps` -- but its
#: default is fifty million, which in CPython is minutes of wall clock for one
#: case.  A fuzzer that inherits that default does not merely run slowly: a
#: single generated `while` loop hangs the whole campaign, and the wall-clock
#: check below never gets a chance to report it because it only runs once
#: `execute` returns.  So every execution here asks for a small budget instead,
#: which turns an unbounded loop into a classified fault in milliseconds.
#: Measured, not guessed: the largest of the sixteen shipped examples executes
#: 6,048 instructions (train_linear_model), and most finish in the hundreds.
#: Two hundred thousand is about thirty-three times that, so a legitimate
#: program is never cut short while a runaway loop costs well under a second
#: instead of minutes.  A program that genuinely needs more is reporting
#: something worth looking at, and StepLimitExceeded says so by name.
FUZZ_MAX_STEPS = 200_000


def check_file_path_is_total(source: str, path: str = "<fuzz>",
                             origin: str = "",
                             kind: str = "") -> List[Violation]:
    """A *file* of arbitrary bytes must reach a verdict, never a traceback.

    Every other check here hands ``compile_source`` a ``str`` that has already
    decoded.  That is precisely why a source file which was not UTF-8 could
    reach the user as a raw ``UnicodeDecodeError`` traceback while every
    campaign reported green: the fuzzer was not testing the path the command
    line actually takes.  This check writes bytes to disk and calls
    ``compile_file`` -- the same function ``ggc check`` and ``ggc run`` call --
    so the file path is covered by the invariant, not just the in-memory one.

    The cases are the ways a file stops being text: a bad byte inside otherwise
    valid source, arbitrary binary, a path that is not there, and a path that
    is a directory.  None of them may raise.
    """
    out: List[Violation] = []
    encoded = source.encode("utf-8", "surrogatepass")
    cut = len(encoded) // 2
    cases = {
        "a valid file with one byte that is not UTF-8": (
            encoded[:cut] + b"\x9e\xfe" + encoded[cut:]),
        "arbitrary binary": bytes(range(256)),
        "an empty file": b"",
    }
    with tempfile.TemporaryDirectory(prefix="ggfuzz-") as tmp:
        for description, payload in cases.items():
            candidate = os.path.join(tmp, "case.gg")
            try:
                with open(candidate, "wb") as handle:
                    handle.write(payload)
            except OSError as exc:                  # pragma: no cover
                out.append(Violation(
                    "file-path-is-total", "crash",
                    f"the harness could not write its own test file: {exc}",
                    source, origin, kind))
                return out
            out.extend(_totality(candidate, description, source, origin, kind))
        # a path that does not exist, and a path that is a directory
        out.extend(_totality(os.path.join(tmp, "absent.gg"),
                             "a path that does not exist",
                             source, origin, kind))
        out.extend(_totality(tmp, "a path that is a directory",
                             source, origin, kind))
    return out


def _totality(candidate: str, description: str, source: str, origin: str,
              kind: str) -> List[Violation]:
    """``compile_file`` must return a Compilation for this path, always."""
    try:
        compilation = compile_file(candidate)
    except BaseException as exc:                    # noqa: BLE001
        return [Violation(
            "file-path-is-total", "crash",
            f"reading {description} raised instead of producing a "
            f"diagnostic: {type(exc).__name__}: {exc}",
            source, origin, kind, traceback=_tb(exc))]
    if not compilation.ok and not compilation.bag.diagnostics:
        return [Violation(
            "file-path-is-total", "wrong",
            f"{description} was rejected with no diagnostic at all",
            source, origin, kind)]
    return []


def check_runs_or_faults(source: str, path: str = "<fuzz>",
                         origin: str = "", kind: str = "",
                         timeout: float = 10.0,
               max_steps: int = FUZZ_MAX_STEPS) -> List[Violation]:
    """A compiled program must run to a value or report a classified fault."""
    out: List[Violation] = []
    try:
        compilation = compile_source(source, path, profile="standard")
    except BaseException as exc:                     # noqa: BLE001
        if not _is_handled(exc):
            out.append(Violation("compiles-or-explains", "crash",
                                 f"{type(exc).__name__}: {exc}", source,
                                 origin, kind, traceback=_tb(exc)))
        return out
    if not compilation.ok or compilation.program is None:
        return out

    started = time.perf_counter()
    buffer = io.StringIO()
    try:
        context = Context(grants=program_grants(compilation), stdout=buffer,
                          seed=0, deterministic=True, max_steps=max_steps)
        execution = execute(compilation, entry=find_entry(compilation) or "main",
                            context=context, grants=())
    except BaseException as exc:                     # noqa: BLE001
        if _is_handled(exc):
            out.append(Violation(
                "runs-or-faults", "crash",
                f"a runtime fault escaped `execute` instead of being reported "
                f"on the result: {type(exc).__name__}: {exc}",
                source, origin, kind, traceback=_tb(exc)))
        else:
            out.append(Violation(
                "runs-or-faults", "crash",
                f"{type(exc).__name__}: {exc}", source, origin, kind,
                traceback=_tb(exc)))
        return out

    if time.perf_counter() - started > timeout:
        out.append(Violation(
            "runs-in-bounded-time", "wrong",
            f"the program ran for more than {timeout:.0f}s; the core forbids "
            f"unbounded loops, so this is either a v0.1 program or a bug",
            source, origin, kind))

    fault = execution.fault
    if fault is not None:
        if not _is_handled(fault):
            out.append(Violation(
                "faults-are-classified", "wrong",
                f"the program failed with {type(fault).__name__}, which is not "
                f"one of the language's fault types: {fault}",
                source, origin, kind))
        elif not str(fault).strip():
            out.append(Violation(
                "faults-say-what-happened", "noise",
                f"a {type(fault).__name__} with an empty message",
                source, origin, kind))
        return out

    # It ran.  The audit chain, if the program produced one, must still verify:
    # a chain that does not verify is worse than no chain.
    audit = getattr(context, "audit", None)
    if audit is not None and getattr(audit, "records", None):
        try:
            valid, problems = audit.verify()
        except BaseException as exc:                 # noqa: BLE001
            out.append(Violation("audit-chain-verifies", "crash",
                                 f"{type(exc).__name__}: {exc}", source,
                                 origin, kind, traceback=_tb(exc)))
            return out
        if not valid:
            out.append(Violation(
                "audit-chain-verifies", "wrong",
                f"the program's own audit chain does not verify: {problems[:3]}",
                source, origin, kind))
    return out


def check_deterministic(source: str, path: str = "<fuzz>",
                        origin: str = "", kind: str = "",
               max_steps: int = FUZZ_MAX_STEPS) -> List[Violation]:
    """Two runs of the same program must be indistinguishable.

    Spec section 1.3 leads with reproducibility, so this is not a nicety: a
    program whose audit hash changes between runs cannot be reproduced, which
    is the property the language is sold on.
    """
    out: List[Violation] = []
    results: List[Tuple[str, str]] = []
    for _ in range(2):
        try:
            compilation = compile_source(source, path, profile="standard")
            if not compilation.ok or compilation.program is None:
                return out
            buffer = io.StringIO()
            context = Context(grants=program_grants(compilation), stdout=buffer,
                              seed=0, deterministic=True, max_steps=max_steps)
            execute(compilation, entry=find_entry(compilation) or "main",
                    context=context, grants=())
            digest = ""
            audit = getattr(context, "audit", None)
            if audit is not None:
                digest = str(getattr(audit, "chain_hash", "")
                             or getattr(audit, "head", ""))
            results.append((buffer.getvalue(), digest))
        except BaseException as exc:                 # noqa: BLE001
            if _is_handled(exc):
                # The language's own error channel: another check owns it, and
                # this one has nothing to say.
                return out
            # An *internal* error is different.  Swallowing it here would report
            # "no difference" for a compiler that fell over, which is the one
            # answer this check must never give.  Validating the fuzzer by
            # injecting bugs is what turned this up.
            out.append(Violation(
                "runs-are-deterministic", "crash",
                f"an internal error escaped instead of being reported: "
                f"{type(exc).__name__}: {exc}", source, origin, kind,
                traceback=_tb(exc)))
            return out

    if len(results) == 2 and results[0] != results[1]:
        first, second = results
        detail = "stdout differs" if first[0] != second[0] else \
            "the audit chain hash differs"
        out.append(Violation(
            "runs-are-deterministic", "wrong",
            f"{detail} between two runs of the same program with the same seed",
            source, origin, kind))
    return out


def check_optimizer_preserves_behavior(source: str, path: str = "<fuzz>",
                                       origin: str = "",
                                       kind: str = "",
               max_steps: int = FUZZ_MAX_STEPS) -> List[Violation]:
    """`-O0` and `-O2` must produce the same output.

    Spec section 23 promises the optimization tiers are behaviour-preserving.
    A tier that changes what a program prints is not an optimization.
    """
    out: List[Violation] = []
    outputs: Dict[int, str] = {}
    for level in (0, 2):
        try:
            compilation = compile_source(source, path, profile="standard",
                                         opt_level=level)
            if not compilation.ok or compilation.program is None:
                return out
            buffer = io.StringIO()
            context = Context(grants=program_grants(compilation), stdout=buffer,
                              seed=0, deterministic=True, max_steps=max_steps)
            execute(compilation, entry=find_entry(compilation) or "main",
                    context=context, grants=())
            outputs[level] = buffer.getvalue()
        except BaseException as exc:                 # noqa: BLE001
            if _is_handled(exc):
                # The language's own error channel: another check owns it, and
                # this one has nothing to say.
                return out
            # An *internal* error is different.  Swallowing it here would report
            # "no difference" for a compiler that fell over, which is the one
            # answer this check must never give.  Validating the fuzzer by
            # injecting bugs is what turned this up.
            out.append(Violation(
                "optimization-preserves-behavior", "crash",
                f"an internal error escaped instead of being reported: "
                f"{type(exc).__name__}: {exc}", source, origin, kind,
                traceback=_tb(exc)))
            return out

    if 0 in outputs and 2 in outputs and outputs[0] != outputs[2]:
        out.append(Violation(
            "optimization-preserves-behavior", "wrong",
            f"-O0 printed {outputs[0][:80]!r} and -O2 printed "
            f"{outputs[2][:80]!r}", source, origin, kind))
    return out


#: Every check, with what it costs.  The engine picks by budget.
CHECKS: Tuple[Tuple[str, Callable[..., List[Violation]]], ...] = (
    ("compiles-or-explains", check_compiles_or_explains),
    ("runs-or-faults", check_runs_or_faults),
    ("runs-are-deterministic", check_deterministic),
    ("optimization-preserves-behavior", check_optimizer_preserves_behavior),
)


#: What the front end did with an input.  Compiled and rejected are both fine
#: answers; the invariant is that one of them happens, and that a rejection is
#: explained.  The distinction matters for reporting: a campaign that says "9
#: rejected" is describing the generator, not a defect.
VERDICT_COMPILED = "compiled"
VERDICT_REJECTED = "rejected"
VERDICT_CRASHED = "crashed"


@dataclass
class RunResult:
    """Everything one input produced: its violations and what happened to it.

    `verdict` is separate from `violations` on purpose.  A rejected program is
    not a violation, and counting rejections as failures -- or counting a clean
    run as a compiled program -- hides which of the two a campaign is measuring.
    """

    verdict: str = VERDICT_REJECTED
    violations: List[Violation] = field(default_factory=list)


def run_checks(source: str, *, path: str = "<fuzz>", origin: str = "",
               kind: str = "", deep: bool = True,
               timeout: float = 10.0,
               max_steps: int = FUZZ_MAX_STEPS) -> RunResult:
    """Run every applicable check over one input."""
    out: List[Violation] = []
    out.extend(check_compiles_or_explains(source, path, origin=origin,
                                          kind=kind))
    out.extend(check_file_path_is_total(source, path, origin=origin,
                                        kind=kind))

    # Did the front end accept the program?  Recomputed here rather than
    # inferred from the violations, because "no violation" also covers a program
    # that was correctly rejected.
    verdict = VERDICT_REJECTED
    if any(v.invariant == "compiles-or-explains" and v.severity == "crash"
           for v in out):
        verdict = VERDICT_CRASHED
    else:
        try:
            if compile_source(source, path).ok:
                verdict = VERDICT_COMPILED
        except BaseException:                      # noqa: BLE001
            verdict = VERDICT_CRASHED

    if not deep:
        return RunResult(verdict, out)
    if verdict == VERDICT_COMPILED:
        # The remaining checks only mean something for a program that compiled.
        out.extend(check_runs_or_faults(source, path, origin=origin, kind=kind,
                                        timeout=timeout, max_steps=max_steps))
        out.extend(check_deterministic(source, path, origin=origin, kind=kind,
                                       max_steps=max_steps))
        out.extend(check_optimizer_preserves_behavior(source, path,
                                                      origin=origin, kind=kind,
                                                      max_steps=max_steps))
    return RunResult(verdict, out)
