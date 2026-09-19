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
from .core import graph as core_graph
from .core import mir as CoreMIR
from .core import native as core_native
from .core.parser import CoreParser, CoreSyntax
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
    # The v0.2 core front end (audit report sections 16-18).  `dialect` is
    # "core" when the source opened with `gama core <version>`; `core` and
    # `core_graph` then hold the program as written and the execution graph the
    # compiler derived from it.  `module` is always the elaborated form that the
    # rest of the pipeline consumes.
    dialect: str = "v0.1"
    core_syntax: Optional[CoreSyntax] = None
    core_model: Optional[CoreMIR.SemanticModel] = None

    @property
    def core_graph(self) -> Optional[CoreMIR.OperationGraph]:
        """The derived operation graph, when this is a core program."""
        return self.core_model.operations if self.core_model else None

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


def is_core_dialect(tokens: Sequence[Any]) -> bool:
    """True when the token stream opens with the `gama core <version>` pragma.

    Detection happens on tokens rather than text so that comments and blank
    lines before the pragma do not matter, and so a program cannot be mistaken
    for core just because the word appears in a string.
    """
    from .tokens import TokenKind
    seen = 0
    for tok in tokens:
        if tok.kind in (TokenKind.NEWLINE, TokenKind.SEPARATOR,
                        TokenKind.INDENT, TokenKind.DEDENT):
            continue
        if seen == 0:
            if not (tok.kind is TokenKind.IDENT and tok.text == "gama"):
                return False
        elif seen == 1:
            return tok.kind is TokenKind.IDENT and tok.text == "core"
        seen += 1
        if seen > 1:
            break
    return False


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

    if is_core_dialect(c.tokens):
        return _compile_core(c, source, profile=profile,
                             opt_level=opt_level, emit_gir=emit_gir)

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


def _compile_core(c: "Compilation", source: str, *, profile: str,
                  opt_level: int, emit_gir: bool) -> Compilation:
    """Compile a core program natively.

    The path is source -> semantic model -> GIR.  Nothing here builds the older
    language's abstract syntax, so `let`, `if`, `while` and `match` are never an
    intermediate representation of a core program -- which was the audit's
    central structural finding on the previous version.

    The older front end is still used for the *type lattice* and the *standard
    library signatures*.  Neither is a language model, and having two competing
    definitions of what `F64` means would be a defect rather than independence.
    """
    c.dialect = "core"
    t0 = time.perf_counter()
    try:
        c.core_syntax = CoreParser(c.tokens, c.path, source).parse()
    except GamaError as exc:
        c.bag.add(exc.diagnostic)
        c.stopped_at = "parse"
        return c
    c.timings.parse = time.perf_counter() - t0

    # The model *is* the meaning of a core program: relationships are derived
    # from `uses`/`yields` and validated before anything is lowered.
    t0 = time.perf_counter()
    model_bag = DiagnosticBag()
    c.core_model = core_graph.build(c.core_syntax, model_bag)
    for diag in model_bag.diagnostics:
        c.bag.add(diag)
    if not model_bag.ok:
        c.stopped_at = "graph"
        return c

    # The checker is kept, not discarded: it holds the resolved type of every
    # binding and every expression, which is exactly what lowering needs.
    # Building a second checker would lower against an empty environment.
    checker = core_native.check(c.core_model, model_bag)
    for diag in model_bag.diagnostics:
        if diag not in c.bag.diagnostics:
            c.bag.add(diag)
    c.timings.check = time.perf_counter() - t0
    if not model_bag.ok:
        c.stopped_at = "check"
        return c
    if not emit_gir:
        return c

    t0 = time.perf_counter()
    lower_bag = DiagnosticBag()
    try:
        c.program = core_native.lower(c.core_model, checker, lower_bag)
    except GamaError as exc:
        c.bag.add(exc.diagnostic)
        c.stopped_at = "lower"
        return c
    except Exception as exc:                        # noqa: BLE001
        c.bag.error(f"internal compiler error while lowering the core program "
                    f"to GIR: {exc}", phase=Phase.GIR, code="E-ice")
        c.stopped_at = "lower"
        return c
    for diag in lower_bag.diagnostics:
        c.bag.add(diag)
    if not lower_bag.ok:
        c.program = None
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
