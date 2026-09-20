"""Differential testing (audit priority 7): does the backend agree with the VM?

A backend is only worth having if it computes the same thing as the reference
interpreter.  "It compiled" is not evidence, and neither is "it looked right".
So the two are run on the same program and their observable behaviour compared.

What is compared, and why those things:

* **stdout, byte for byte.**  This is the program's visible result.  A single
  differing digit is a bug in the backend, not a rounding difference to be
  waved away -- which is why the C runtime reproduces Python's shortest
  round-trip float repr instead of using ``%g``.
* **exit status.**  Success versus fault.
* **fault kind.**  ``[DivideByZero]`` versus ``[IntegerOverflow]`` is the
  difference between two different diagnoses.  The fault *message* is not
  compared: the interpreter can print an arbitrary-precision integer that the
  native runtime, computing in 64 bits, cannot.  Where that happens the kinds
  still agree, and the limitation is documented rather than hidden.

What is deliberately not compared: timing, memory, and the audit chain.  The
first two are not semantics, and comparing them would invite a performance
claim this repository does not make.  The audit chain is not yet implemented in
the native runtime, so any program that audits is refused by the support
analysis before it gets here.
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..driver import compile_source, execute, find_entry, program_grants
from ..runtime.context import Context
from . import cgen, native

FAULT_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")


@dataclass
class InterpreterResult:
    ran: bool = False
    stdout: str = ""
    exit_status: int = 0
    fault_kind: str = ""
    fault_text: str = ""
    compile_error: str = ""
    elapsed_ms: float = 0.0

    @property
    def comparable(self) -> bool:
        return not self.compile_error


def run_interpreter(source: str, path: str = "<program>",
                    grants: Sequence[str] = (),
                    entry: Optional[str] = None,
                    timeout: float = 30.0) -> InterpreterResult:
    """Run a program on the reference interpreter and record what it did."""
    started = time.perf_counter()
    compilation = compile_source(source, path, grants=tuple(grants))
    if not compilation.ok or compilation.program is None:
        codes = []
        for diag in compilation.bag.diagnostics if compilation.bag else []:
            if getattr(diag, "code", ""):
                codes.append(diag.code)
        return InterpreterResult(
            compile_error=", ".join(codes) or "the program did not compile",
            elapsed_ms=(time.perf_counter() - started) * 1000.0)

    buffer = io.StringIO()
    context = Context(grants=program_grants(compilation, tuple(grants)),
                      stdout=buffer, deterministic=True, seed=0)
    name = entry or find_entry(compilation) or "main"
    execution = execute(compilation, entry=name, context=context, grants=())
    elapsed = (time.perf_counter() - started) * 1000.0

    result = InterpreterResult(ran=True, stdout=buffer.getvalue(),
                               elapsed_ms=elapsed)
    fault = execution.fault
    if fault is None:
        result.exit_status = 0
        return result
    result.exit_status = 3
    text = str(fault)
    result.fault_text = text
    kind = type(fault).__name__
    # Faults carry their own classification where they have one, which is what
    # the CLI prints; fall back to the exception type name.
    own = getattr(fault, "kind", None)
    if isinstance(own, str) and own:
        kind = own
    result.fault_kind = kind
    return result


@dataclass
class Comparison:
    """One program, both machines, and whether they agreed."""

    path: str
    #: `agreed`, `refused`, `diverged`, `build-failed`, `compile-error`
    outcome: str = "diverged"
    reasons: List[str] = field(default_factory=list)
    interpreter: Optional[InterpreterResult] = None
    native_stdout: str = ""
    native_status: int = 0
    native_fault_kind: str = ""
    native_stderr: str = ""
    native_elapsed_ms: float = 0.0
    interpreter_elapsed_ms: float = 0.0
    #: Where stdout first differed, for a readable report.
    first_difference: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome in ("agreed", "refused")

    def render(self) -> List[str]:
        out = [f"{self.path}: {self.outcome}"]
        for reason in self.reasons[:6]:
            out.append(f"    {reason}")
        if self.first_difference:
            out.append(f"    first difference: {self.first_difference}")
        return out


def _first_difference(a: str, b: str) -> str:
    lines_a = a.splitlines()
    lines_b = b.splitlines()
    for index in range(max(len(lines_a), len(lines_b))):
        left = lines_a[index] if index < len(lines_a) else "<missing>"
        right = lines_b[index] if index < len(lines_b) else "<missing>"
        if left != right:
            return f"line {index + 1}: interpreter {left!r} vs native {right!r}"
    if len(a) != len(b):
        return "trailing whitespace differs"
    return ""


def compare(source: str, path: str = "<program>",
            grants: Sequence[str] = (),
            build_dir: str = native.DEFAULT_BUILD_DIR,
            keep_c: bool = True,
            entry: Optional[str] = None) -> Comparison:
    """Build a program natively, run both, and report whether they agree."""
    result = Comparison(path=path)

    interp = run_interpreter(source, path, grants, entry)
    result.interpreter = interp
    result.interpreter_elapsed_ms = interp.elapsed_ms
    if not interp.comparable:
        result.outcome = "compile-error"
        result.reasons.append(f"the program does not compile: {interp.compile_error}")
        return result

    compilation = compile_source(source, path, grants=tuple(grants))
    build = native.build(compilation.program, path, build_dir=build_dir,
                         keep_c=keep_c, entry=entry)

    if build.refused:
        result.outcome = "refused"
        result.reasons = [p.render() for p in build.problems]
        return result
    if not build.ok:
        result.outcome = "build-failed"
        result.reasons = build.render()
        return result

    run_result = native.run(build.exe_path)
    result.native_stdout = run_result.stdout
    result.native_status = run_result.returncode
    result.native_stderr = run_result.stderr
    result.native_elapsed_ms = run_result.elapsed_ms
    match = FAULT_RE.search(run_result.stderr)
    result.native_fault_kind = match.group(1) if match else ""

    if run_result.timed_out:
        result.outcome = "diverged"
        result.reasons.append("the native binary did not finish in time")
        return result

    differences: List[str] = []
    if run_result.stdout != interp.stdout:
        differences.append("stdout differs")
        result.first_difference = _first_difference(interp.stdout,
                                                    run_result.stdout)
    if run_result.returncode != interp.exit_status:
        differences.append(
            f"exit status differs: interpreter {interp.exit_status}, "
            f"native {run_result.returncode}")
    if interp.fault_kind and result.native_fault_kind \
            and interp.fault_kind != result.native_fault_kind:
        differences.append(
            f"fault kind differs: interpreter {interp.fault_kind}, "
            f"native {result.native_fault_kind}")
    if interp.fault_kind and not result.native_fault_kind:
        differences.append(
            f"the interpreter faulted with {interp.fault_kind} and the native "
            f"binary did not")
    if result.native_fault_kind and not interp.fault_kind:
        differences.append(
            f"the native binary faulted with {result.native_fault_kind} and "
            f"the interpreter did not")

    if differences:
        result.outcome = "diverged"
        result.reasons = differences
    else:
        result.outcome = "agreed"
    return result


def compare_file(path: str, **kwargs: Any) -> Comparison:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    kwargs.setdefault("path", path)
    return compare(source, **kwargs)


@dataclass
class Corpus:
    """The result of differential testing over many programs."""

    comparisons: List[Comparison] = field(default_factory=list)

    @property
    def agreed(self) -> List[Comparison]:
        return [c for c in self.comparisons if c.outcome == "agreed"]

    @property
    def refused(self) -> List[Comparison]:
        return [c for c in self.comparisons if c.outcome == "refused"]

    @property
    def diverged(self) -> List[Comparison]:
        return [c for c in self.comparisons
                if c.outcome in ("diverged", "build-failed")]

    @property
    def broken(self) -> List[Comparison]:
        return [c for c in self.comparisons if c.outcome == "compile-error"]

    def summary(self) -> str:
        return (f"{len(self.agreed)} agreed, {len(self.refused)} refused by the "
                f"backend, {len(self.diverged)} diverged, "
                f"{len(self.broken)} did not compile")

    def render(self, *, include_refused: bool = False) -> List[str]:
        out: List[str] = []
        for comparison in self.comparisons:
            if comparison.outcome == "agreed":
                continue
            if comparison.outcome == "refused" and not include_refused:
                continue
            out.extend(comparison.render())
        if not out:
            out.append("every program the backend accepted agreed with the "
                       "reference interpreter")
        out.append("")
        out.append(self.summary())
        return out


def compare_many(paths: Sequence[str], build_dir: str = ".ggbuild",
                 **kwargs: Any) -> Corpus:
    corpus = Corpus()
    for path in paths:
        corpus.comparisons.append(
            compare_file(path, build_dir=build_dir, **kwargs))
    return corpus
