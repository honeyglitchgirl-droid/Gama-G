"""Shared helpers for the Gama-G test suite.

The suite deliberately uses only the standard library, so it runs on any
checkout with ``python3 -m unittest discover -s tests`` and needs no
third-party packages installed.
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILER_DIR = os.path.join(REPO_ROOT, "compiler")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
SPEC_PATH = os.path.join(
    REPO_ROOT, "Gama-G_v1.0_Production_Specification.txt")

if COMPILER_DIR not in sys.path:
    sys.path.insert(0, COMPILER_DIR)

from gamag.driver import (compile_source, execute,  # noqa: E402
                          find_entry, program_grants)
from gamag.runtime.context import Context  # noqa: E402


@dataclass
class Outcome:
    """The result of compiling (and maybe running) a snippet."""

    source: str
    compilation: Any
    execution: Any = None
    output: str = ""
    context: Optional[Context] = None

    # ---- compilation -------------------------------------------------
    @property
    def compiled(self) -> bool:
        return self.compilation.ok

    @property
    def diagnostics(self) -> List[Any]:
        return list(self.compilation.bag.diagnostics)

    def codes(self) -> List[str]:
        """Diagnostic codes, e.g. ``["E-arg-type", "E-no-member"]``."""
        return [d.code for d in self.diagnostics if d.code]

    def messages(self) -> str:
        """Messages with their help text, which is where the fix-it lives."""
        parts = []
        for d in self.diagnostics:
            parts.append(d.message)
            if getattr(d, "help_text", None):
                parts.append("  help: " + d.help_text)
        return "\n".join(parts)

    def has_code(self, code: str) -> bool:
        return code in self.codes()

    def stopped_at(self) -> Optional[str]:
        return self.compilation.stopped_at

    # ---- execution ---------------------------------------------------
    @property
    def ran(self) -> bool:
        return self.execution is not None and self.execution.ok

    @property
    def value(self) -> Any:
        return self.execution.value if self.execution else None

    @property
    def fault(self) -> Optional[BaseException]:
        return self.execution.fault if self.execution else None

    def fault_kind(self) -> Optional[str]:
        return getattr(self.fault, "kind", None)

    # ---- assertions --------------------------------------------------
    def assert_compiled(self, testcase: Any) -> "Outcome":
        if not self.compiled:
            testcase.fail(
                "expected the program to compile, but it did not:\n"
                + self.messages() + "\n--- source ---\n" + self.source)
        return self

    def assert_ran(self, testcase: Any) -> "Outcome":
        self.assert_compiled(testcase)
        if not self.ran:
            testcase.fail(
                f"expected the program to run, but it faulted: "
                f"{self.fault_kind()}: {self.fault}\n--- source ---\n"
                + self.source)
        return self

    def assert_output_contains(self, testcase: Any, *needles: str) -> "Outcome":
        self.assert_ran(testcase)
        for needle in needles:
            if needle not in self.output:
                testcase.fail(
                    f"expected {needle!r} in the output; got:\n{self.output}")
        return self

    def assert_rejected(self, testcase: Any, code: Optional[str] = None,
                        phase: Optional[str] = None) -> "Outcome":
        """The program must fail to compile, ideally for the stated reason."""
        if self.compiled:
            testcase.fail(
                "expected a compile-time rejection, but the program compiled:\n"
                + self.source)
        if code is not None and not self.has_code(code):
            testcase.fail(
                f"expected diagnostic {code!r}, got {self.codes()}:\n"
                + self.messages() + "\n--- source ---\n" + self.source)
        if phase is not None and self.stopped_at() != phase:
            testcase.fail(
                f"expected the pipeline to stop at {phase!r}, stopped at "
                f"{self.stopped_at()!r}")
        return self

    def assert_faulted(self, testcase: Any, kind: Optional[str] = None) -> "Outcome":
        """The program must compile and then fault at run time."""
        self.assert_compiled(testcase)
        if self.execution is None or self.execution.ok:
            testcase.fail(
                "expected a runtime fault, but the program completed:\n"
                + self.output + "\n--- source ---\n" + self.source)
        if kind is not None and self.fault_kind() != kind:
            testcase.fail(
                f"expected fault kind {kind!r}, got {self.fault_kind()!r}: "
                f"{self.fault}")
        return self


def compile_only(source: str, *, profile: str = "strict", opt_level: int = 1,
                 grants: Tuple[str, ...] = (), path: str = "<test>") -> Outcome:
    """Compile without running."""
    compilation = compile_source(source, path, profile=profile,
                                 opt_level=opt_level, grants=grants)
    return Outcome(source=source, compilation=compilation)


def run(source: str, *, entry: str = "main", profile: str = "standard",
        opt_level: int = 1, grants: Tuple[str, ...] = (),
        path: str = "<test>", seed: int = 0,
        deterministic: bool = True) -> Outcome:
    """Compile and run, capturing the program's output and context.

    Deterministic by default, as `ggc run` is, so the audit chain and the
    RNG stream are reproducible across runs (spec section 1.3).
    """
    compilation = compile_source(source, path, profile=profile,
                                 opt_level=opt_level, grants=grants)
    if not compilation.ok:
        return Outcome(source=source, compilation=compilation)
    buffer = io.StringIO()
    # The module's own `grant` header is part of the program; without merging
    # it in, every example that declares capabilities would fail here for a
    # reason that has nothing to do with the language.
    context = Context(grants=program_grants(compilation, grants),
                      stdout=buffer, seed=seed, deterministic=deterministic)
    name = find_entry(compilation, preferred=entry)
    execution = execute(compilation, entry=name, context=context, grants=grants)
    return Outcome(source=source, compilation=compilation,
                   execution=execution, output=buffer.getvalue(),
                   context=context)


def run_file(path: str, **kwargs: Any) -> Outcome:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    kwargs.setdefault("path", os.path.relpath(path, REPO_ROOT))
    return run(source, **kwargs)


def example(name: str) -> str:
    return os.path.join(EXAMPLES_DIR, name)


def example_names() -> List[str]:
    """The v0.1 reference-surface examples, at the top of `examples/`."""
    return sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".gg"))


def core_example_names() -> List[str]:
    """The v0.2 core examples, under `examples/core/`.

    Returned with their subdirectory attached so that :func:`example` resolves
    them the same way it resolves the top-level ones.
    """
    directory = os.path.join(EXAMPLES_DIR, "core")
    if not os.path.isdir(directory):
        return []
    return sorted("core/" + f for f in os.listdir(directory)
                  if f.endswith(".gg"))


def all_example_names() -> List[str]:
    """Every shipped example, in both dialects.

    The generic example checks -- comment syntax, "does it say how to run it",
    "does it compile under strict" -- are properties of *being an example*, not
    of either language surface, so they are run across both.
    """
    return example_names() + core_example_names()


def spec_lines() -> List[str]:
    with open(SPEC_PATH, "r", encoding="utf-8") as handle:
        return handle.read().split("\n")


def spec_block(first: int, last: int) -> str:
    """Extract a code example from the specification.

    The blueprint indents its examples by four spaces.  Line numbers are
    1-based and inclusive, matching what ``sed -n '140,147p'`` would show, so
    a failing test can be checked against the document by hand.
    """
    lines = spec_lines()
    out = []
    for line in lines[first - 1:last]:
        if line.startswith("    "):
            out.append(line[4:])
        else:
            out.append(line.strip() if line.strip() else "")
    return "\n".join(out).strip() + "\n"


def spec_block_from(first: int, limit: int = 40) -> str:
    """The code example beginning at line `first`, to the end of its indent.

    The blueprint indents examples by four spaces and returns to prose at
    column zero, so the extent can be found rather than hard-coded; a test
    then keeps working if lines are inserted above it.
    """
    lines = spec_lines()
    out: List[str] = []
    for line in lines[first - 1:first - 1 + limit]:
        if line.startswith("    "):
            out.append(line[4:])
        elif not line.strip():
            if out and not out[-1].strip():
                break
            out.append("")
        else:
            break
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def spec_section(number: str) -> Optional[int]:
    """The line number of a section heading such as ``17. POLICY ENGINE``.

    The blueprint also contains a numbered list of target domains near the
    top ("3. Corporate and enterprise systems"), so a heading is recognised
    by its number *and* its upper-case title rather than by number alone.
    """
    for index, line in enumerate(spec_lines(), 1):
        text = line.strip()
        if not text.startswith(number) or text[len(number):len(number) + 1] != ".":
            continue
        title = text[len(number) + 1:].strip()
        if title and title == title.upper():
            return index
    return None


@dataclass
class SpecList:
    """A bulleted enumeration from the specification (types, effects, ...)."""

    name: str
    first: int
    last: int
    items: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, name: str, first: int, last: int) -> "SpecList":
        items = [line.strip() for line in spec_lines()[first - 1:last]
                 if line.strip()]
        return cls(name=name, first=first, last=last, items=items)
