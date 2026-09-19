"""Compilation driver: source to running program.

Wires the pipeline described in spec sections 20-23::

    source -> tokens -> AST -> typed AST -> GIR -> optimized GIR -> GAEM

Each phase either produces a result or records diagnostics; the driver never
runs a later phase on input an earlier phase rejected, because a malformed
typed AST would make later diagnostics meaningless.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import ast_nodes as A
from .diagnostics import (DiagnosticBag, GamaError, Phase, Severity, SourcePos)
from .gir.builder import build_program
from .gir.ir import GProgram
from .gir.optimizer import OptimizationReport, optimize
from .lexer import tokenize
from .parser import Parser
from .runtime.context import Context
from .runtime.vm import VM
from .semantic.checker import Checker, check_module

GIR_VERSION = "1.0"


@dataclass
class PhaseTimings:
    lex: float = 0.0
    parse: float = 0.0
    check: float = 0.0
    lower: float = 0.0
    optimize: float = 0.0
    execute: float = 0.0

    def total(self) -> float:
        return (self.lex + self.parse + self.check + self.lower
                + self.optimize)

    def as_rows(self) -> List[Tuple[str, float]]:
        return [("lex", self.lex), ("parse", self.parse),
                ("type check", self.check), ("lower to GIR", self.lower),
                ("optimize", self.optimize), ("execute", self.execute)]


@dataclass
class Compilation:
    """Everything one compilation produced, for tooling and diagnostics."""

    path: str = "<source>"
    source: str = ""
    profile: str = "strict"
    opt_level: int = 1
    tokens: List[Any] = field(default_factory=list)
    module: Optional[A.Module] = None
    checker: Optional[Checker] = None
    bag: DiagnosticBag = field(default_factory=DiagnosticBag)
    program: Optional[GProgram] = None
    optimization: Optional[OptimizationReport] = None
    timings: PhaseTimings = field(default_factory=PhaseTimings)
    stopped_at: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.program is not None and self.bag.ok

    @property
    def errors(self) -> List[Any]:
        return self.bag.errors

    @property
    def warnings(self) -> List[Any]:
        return self.bag.warnings

    def diagnostics_text(self, color: bool = False) -> str:
        return self.bag.render_all(self.source, color)

    def required_capabilities(self) -> List[str]:
        if self.checker is None:
            return []
        caps: set = set()
        for info in self.checker.functions.values():
            caps |= set(info.caps)
        return sorted(caps)

    def declared_effects(self) -> List[str]:
        if self.checker is None:
            return []
        effects: set = set()
        for info in self.checker.functions.values():
            effects |= set(info.effects)
        return sorted(effects)


def compile_source(source: str, path: str = "<source>", *,
                   profile: str = "strict", opt_level: int = 1,
                   grants: Sequence[str] = (),
                   emit_gir: bool = True) -> Compilation:
    """Run every compile-time phase.  Never raises for source-level errors."""
    c = Compilation(path=path, source=source, profile=profile,
                    opt_level=opt_level)
    base = os.path.splitext(os.path.basename(path))[0] or "main"

    t0 = time.perf_counter()
    try:
        c.tokens = tokenize(source, path)
    except GamaError as exc:
        c.bag.add(exc.diagnostic)
        c.stopped_at = "lex"
        return c
    c.timings.lex = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        c.module = Parser(c.tokens, path, source).parse_module()
    except GamaError as exc:
        c.bag.add(exc.diagnostic)
        c.stopped_at = "parse"
        return c
    c.timings.parse = time.perf_counter() - t0
    if grants:
        c.module.grants = tuple(dict.fromkeys(
            tuple(c.module.grants) + tuple(grants)))

    t0 = time.perf_counter()
    checker, bag = check_module(c.module, source, profile=profile)
    c.checker = checker
    for diag in bag.diagnostics:
        c.bag.add(diag)
    c.timings.check = time.perf_counter() - t0
    if not bag.ok:
        c.stopped_at = "check"
        return c

    if not emit_gir:
        return c

    t0 = time.perf_counter()
    try:
        c.program = build_program(c.module, checker)
    except GamaError as exc:
        c.bag.add(exc.diagnostic)
        c.stopped_at = "lower"
        return c
    except Exception as exc:                    # noqa: BLE001
        c.bag.error(f"internal compiler error while lowering to GIR: {exc}",
                    phase=Phase.GIR, code="E-ice")
        c.stopped_at = "lower"
        return c
    c.timings.lower = time.perf_counter() - t0

    if opt_level > 0:
        t0 = time.perf_counter()
        c.optimization = optimize(c.program, opt_level,
                                  deterministic=(profile == "strict"))
        c.timings.optimize = time.perf_counter() - t0

    return c


def compile_file(path: str, **kwargs: Any) -> Compilation:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return compile_source(source, path, **kwargs)


@dataclass
class Execution:
    value: Any = None
    context: Optional[Context] = None
    vm: Optional[VM] = None
    fault: Optional[BaseException] = None
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        return self.fault is None


def execute(compilation: Compilation, *, entry: str = "main",
            args: Sequence[Any] = (), context: Optional[Context] = None,
            grants: Sequence[str] = ()) -> Execution:
    """Run a compiled program on the GAEM reference interpreter."""
    if compilation.program is None:
        raise ValueError("cannot execute a compilation that produced no GIR")
    # Deterministic by default, matching `ggc run`.  Spec section 1.3 makes
    # reproducibility the promise the language is sold on, so tying it to the
    # strict profile meant a library caller using the default profile got
    # wall-clock timestamps and random event ids in the audit chain -- and
    # therefore a different chain hash on every run.  Non-determinism is the
    # explicit opt-out: pass your own Context, or `--lenient-runtime`.
    ctx = context or Context(
        deterministic=True,
        grants=set(grants) | set(compilation.checker.grants
                                 if compilation.checker else ()),
        program_version=GIR_VERSION)
    vm = VM(compilation.program, ctx, compilation.checker)
    result = Execution(context=ctx, vm=vm)
    t0 = time.perf_counter()
    try:
        result.value = vm.run(entry, args)
    except BaseException as exc:                # noqa: BLE001
        result.fault = exc
    result.duration = time.perf_counter() - t0
    compilation.timings.execute = result.duration
    return result


def run_source(source: str, path: str = "<source>", *, entry: str = "main",
               args: Sequence[Any] = (), profile: str = "strict",
               opt_level: int = 1, grants: Sequence[str] = (),
               context: Optional[Context] = None
               ) -> Tuple[Compilation, Execution]:
    compilation = compile_source(source, path, profile=profile,
                                 opt_level=opt_level, grants=grants)
    if not compilation.ok:
        return compilation, Execution(fault=None)
    return compilation, execute(compilation, entry=entry, args=args,
                                context=context, grants=grants)


def find_entry(compilation: Compilation, preferred: str = "main") -> Optional[str]:
    """Pick an entry point: `main` if present, else a `main()`-like component."""
    if compilation.program is None:
        return None
    if preferred in compilation.program.functions:
        return preferred
    if "<main>" in compilation.program.functions:
        return "<main>"
    return None
