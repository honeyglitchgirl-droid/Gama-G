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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import ast_nodes as A
from .core import capability as core_capability
from .core import graph as core_graph
from .core import memory as core_memory
from .core import mir as CoreMIR
from .core import native as core_native
from .core import recovery as core_recovery
from .core.parser import CoreParser, CoreSyntax
from .diagnostics import (Diagnostic, DiagnosticBag, GamaError, Phase,
                          Severity, SourcePos)
from .gir.builder import build_program
from .gir.ir import GProgram
from .gir.optimizer import OptimizationReport, optimize
from .lexer import tokenize
from .nesting import E_NESTING_CODE, ast_depth_limit, deepest
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
    # The core front end (first audit sections 16-18, second audit section 13).
    # `dialect` is "core" when the source opened with `gama core <version>`, and
    # the fields below then hold the program as written, the semantic model the
    # compiler derived from it, and the two models that give that derivation its
    # formal meaning.  `module` stays None for a core program: nothing is
    # elaborated into the older language's AST any more.
    dialect: str = "v0.1"
    core_syntax: Optional[CoreSyntax] = None
    core_model: Optional[CoreMIR.SemanticModel] = None
    #: ownership, extents, releases and the proofs about them (audit priority 4)
    core_memory: Optional[core_memory.MemoryModel] = None
    #: what the program holds and what it demands (audit priority 5)
    core_capabilities: Optional[core_capability.CapabilityPolicy] = None
    #: the declared escalation policy, if any (audit priority 6)
    core_recovery: Optional[core_recovery.RecoveryPolicy] = None

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


def depth_diagnostic(root: Any) -> Optional[Diagnostic]:
    """The diagnostic for an AST too deep for the recursive phases, if any.

    Flat source can still be a deep tree -- ``1+1+1`` repeated five thousand
    times parses iteratively and then recurses in the checker -- so this is
    measured over the completed tree rather than counted while parsing, and
    the walk is iterative so that measuring cannot itself overflow.
    """
    limit = ast_depth_limit()
    depth, offender = deepest(root, limit)
    if offender is None:
        return None
    return Diagnostic(
        Severity.ERROR, Phase.PARSE,
        f"expression is nested deeper than {limit} levels (found {depth})",
        pos=getattr(offender, "pos", None), code=E_NESTING_CODE,
        help_text=(
            "later phases are recursive, so the bound protects the host "
            "stack; simplify the expression, or raise "
            "sys.setrecursionlimit for generated input"))


def bounded_by_depth(c: "Compilation", root: Any) -> bool:
    """Whether an AST is shallow enough for every recursive consumer."""
    diag = depth_diagnostic(root)
    if diag is None:
        return True
    c.bag.add(diag)
    c.stopped_at = "parse"
    return False


def compile_source(source: str, path: str = "<source>", *,
                   profile: str = "strict", opt_level: int = 1,
                   grants: Sequence[str] = (),
                   emit_gir: bool = True) -> Compilation:
    """Run every compile-time phase.  Never raises for source-level errors.

    That claim is about *every* input, and it used to be false for two of
    them: a file that was not UTF-8, and nesting deep enough to exhaust the
    host stack.  Decoding is handled where the file is read
    (:func:`read_source`), nesting is bounded in :mod:`gamag.nesting`, and
    this wrapper is the backstop for whatever those bounds did not foresee --
    so that running out of stack stays a diagnostic, not a traceback.
    """
    try:
        return _compile_source(source, path, profile=profile,
                               opt_level=opt_level, grants=grants,
                               emit_gir=emit_gir)
    except RecursionError:
        c = Compilation(path=path, source=source, profile=profile,
                        opt_level=opt_level)
        c.bag.error(
            "input nests too deeply to compile on this host's stack",
            phase=Phase.PARSE, code=E_NESTING_CODE,
            help_text=("this is the backstop behind the depth bound; "
                       "the source is beyond what this host can carry"))
        c.stopped_at = "parse"
        return c


def _compile_source(source: str, path: str = "<source>", *,
                    profile: str = "strict", opt_level: int = 1,
                    grants: Sequence[str] = (),
                    emit_gir: bool = True) -> Compilation:
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
    if not bounded_by_depth(c, c.module):
        return c
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
    if not bounded_by_depth(c, c.core_syntax):
        return c

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

    # `fn` helpers written in this core file, compiled first so the core
    # checker can resolve a call to one.  They go through the v0.1 parser, the
    # v0.1 checker and the v0.1 lowering -- the same pipeline a v0.1 module
    # uses -- and the GIR they produce is merged into this module below.
    helpers = _Helpers()
    if c.core_syntax.functions:
        helpers = _compile_helpers(c, source, profile)
        for diag in helpers.diagnostics:
            if diag not in c.bag.diagnostics:
                c.bag.add(diag)
        if not helpers.ok:
            c.stopped_at = "helpers"
            return c

    # The checker is kept, not discarded: it holds the resolved type of every
    # binding and every expression, which is exactly what lowering needs.
    # Building a second checker would lower against an empty environment.
    checker = core_native.check(c.core_model, model_bag, helpers.signatures)
    for diag in model_bag.diagnostics:
        if diag not in c.bag.diagnostics:
            c.bag.add(diag)

    # The two formal models. They are built whether or not GIR is being emitted,
    # because `ggc graph` and `ggc memory` report on them without lowering, and
    # because a model that only exists on the way to codegen is not a model.
    c.core_capabilities = core_capability.CapabilityPolicy.of(
        c.core_model.intent.authority)
    core_capability.all_demands(c.core_model, checker.secret_bindings(),
                                c.core_capabilities)
    c.core_recovery = core_recovery.policy_of(c.core_model.intent)
    c.core_memory = core_memory.build(c.core_model, checker=checker)
    for violation in c.core_memory.violations:
        model_bag.error(f"the memory model is inconsistent: {violation}",
                        phase=Phase.GIR, code="E-memory-model",
                        help_text="this is a compiler bug rather than a problem "
                                  "with the program; please report it")
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
        c.program = core_native.lower(c.core_model, checker, lower_bag,
                                      memory=c.core_memory)
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

    # `fn` helpers written in a core file.  They are compiled by the v0.1 front
    # end -- the same parser, the same checker, the same lowering -- and merged
    # into the one GIR module the core produced.  That is what makes the two
    # surfaces one language: the declaration families share a file, a pipeline
    # and an output, rather than being two dialects the driver chooses between.
    if helpers.program is not None:
        _merge_programs(c.program, helpers.program)

    if opt_level > 0:
        t0 = time.perf_counter()
        c.optimization = optimize(c.program, opt_level,
                                  deterministic=(profile == "strict"))
        c.timings.optimize = time.perf_counter() - t0
    return c


@dataclass
class HelperSignature:
    """A helper's parameters and result, for the core checker to resolve against."""

    params: List[Any] = field(default_factory=list)   # [(name, type), ...]
    returns: Any = None


@dataclass
class _Helpers:
    """The result of compiling a core program's `fn` declarations."""

    ok: bool = True
    program: Optional[Any] = None
    diagnostics: List[Any] = field(default_factory=list)
    signatures: Dict[str, HelperSignature] = field(default_factory=dict)


def _compile_helpers(c: "Compilation", source: str,
                     profile: str) -> _Helpers:
    """Compile the `fn` declarations captured from a core program.

    The tokens carry their original positions, so a diagnostic inside a helper
    points at the helper in the user's file rather than at a synthesized one.
    That is the reason the capture keeps tokens instead of text.
    """
    from .core.parser import CoreFunction
    from .gir.builder import build_program
    from .lexer import tokenize
    from .parser import Parser
    from .semantic.checker import check_module
    from .semantic import types as T
    from .tokens import Token, TokenKind

    out = _Helpers()
    # No `if c.program is None` guard here.  Helpers are compiled *before* the
    # core is lowered, so the module does not exist yet -- the guard was left
    # over from when they were compiled afterwards, and it silently returned an
    # empty result, which then made every helper call an unknown call.
    tokens: List[Any] = []
    for function in c.core_syntax.functions:
        if tokens:
            # A separator between declarations, so the parser sees them as
            # separate top-level items rather than one run-on declaration.
            last = tokens[-1]
            tokens.append(Token(kind=TokenKind.NEWLINE, text="\n",
                                pos=last.pos, end=last.pos, value=None))
        tokens.extend(function.tokens)

    # A token stream without an EOF ends the parser in an IndexError rather than
    # at the end of the module, so the terminator is part of the contract.
    if tokens:
        last = tokens[-1]
        tokens.append(Token(kind=TokenKind.EOF, text="", pos=last.pos,
                            end=last.pos, value=None))

    try:
        module = Parser(tokens, c.path, source).parse_module()
    except GamaError as exc:
        out.diagnostics.append(exc.diagnostic)
        out.ok = False
        return out

    depth_issue = depth_diagnostic(module)
    if depth_issue is not None:
        out.diagnostics.append(depth_issue)
        out.ok = False
        return out

    checker, bag = check_module(module, source, profile=profile)
    out.diagnostics.extend(bag.diagnostics)
    if not bag.ok:
        out.ok = False
        return out

    # The signatures the core checker needs.  Taken from the checked module
    # rather than re-derived, so there is one answer to "what type is this
    # parameter" and it is the checker's.
    for name, info in getattr(checker, "functions", {}).items():
        fn_type = getattr(info, "fn_type", None)
        if fn_type is None:
            continue
        names = getattr(fn_type, "param_names", ()) or ()
        out.signatures[name] = HelperSignature(
            params=[(names[i] if i < len(names) else f"arg{i}", param)
                    for i, param in enumerate(fn_type.params)],
            returns=fn_type.ret)

    try:
        program = build_program(module, checker)
    except GamaError as exc:
        out.diagnostics.append(exc.diagnostic)
        out.ok = False
        return out
    except Exception as exc:                        # noqa: BLE001
        bag.error(f"internal compiler error while lowering a helper function: "
                  f"{exc}", phase=Phase.GIR, code="E-ice")
        out.diagnostics.append(bag.diagnostics[-1])
        out.ok = False
        return out
    out.program = program
    return out


def _merge_programs(core: Any, helpers: Any) -> None:
    """Fold the helpers' functions into the core program's module.

    A name that already exists in the core module is left alone and reported by
    the core checker rather than silently overwritten: two functions with one
    name is a program the user did not write.
    """
    if core is None or helpers is None:
        return
    for name, function in helpers.functions.items():
        core.functions.setdefault(name, function)
    core.grants = tuple(dict.fromkeys(tuple(core.grants) + tuple(helpers.grants)))


def read_source(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read a source file as UTF-8 text.

    Returns ``(source, None)`` or ``(None, reason)``.  Source files are text,
    and a compiler that is handed bytes it cannot decode has to say so: this
    used to escape as a ``UnicodeDecodeError`` traceback, which is both a crash
    and a lie -- it tells the user the compiler broke, when in fact the file
    is not text.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read(), None
    except UnicodeDecodeError as exc:
        offset = exc.start
        bad = exc.object[offset:offset + 1]
        shown = f"0x{bad[0]:02x}" if bad else "an invalid byte"
        return None, (
            f"{path} is not valid UTF-8 text: byte {shown} at offset "
            f"{offset} does not decode. Gama-G source must be UTF-8; "
            f"convert the file (for example `iconv -f latin1 -t utf-8`)")
    except OSError as exc:
        return None, f"cannot read {path}: {exc.strerror or exc}"


def _unreadable(path: str, reason: str) -> Compilation:
    """A compilation carrying one error, for a file that could not be read."""
    c = Compilation(path=path)
    c.bag.error(reason, phase=Phase.LEX, code="E-source-unreadable",
                help_text="the compiler reads source files as UTF-8 text")
    c.stopped_at = "lex"
    return c


def compile_file(path: str, **kwargs: Any) -> Compilation:
    source, reason = read_source(path)
    if source is None:
        return _unreadable(path, reason)
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
        grants=program_grants(compilation, grants),
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


def declared_grants(compilation: "Compilation") -> Set[str]:
    """The capabilities the *program itself* asks for.

    A ``grant`` header in the older dialect, or ``authority`` on a core intent,
    which the lowerer writes into :attr:`GProgram.grants`.  This is a request,
    not authority: spec section 12 says "no ambient filesystem access", and a
    program that could confer a capability on itself by writing it down would
    have exactly that.

    Keeping the request separate from the grant is what lets a deployment decide.
    `ggc run` chooses to honour these (the user chose to run this program);
    an embedder that calls `execute` gets only what it passes.
    """
    declared: Set[str] = set()
    if compilation.program is not None:
        declared |= set(compilation.program.grants)
    if compilation.checker is not None:
        declared |= set(compilation.checker.grants)
    return declared


def program_grants(compilation: "Compilation",
                   extra: Sequence[str] = ()) -> Set[str]:
    """The authority a run of this compilation actually has.

    One source: what the caller supplied.  A capability the program merely
    declared is *not* included, because including it made the whole capability
    system decorative -- a program could read any file it liked by writing
    ``grant FileRead`` at the top, which is the ambient authority spec section 12
    forbids by name.

    A caller that has decided to trust a program's declarations can say so
    explicitly, and `ggc run` does exactly that, in one place, where the decision
    is visible:

        grants = set(args.grant)
        if not args.strict_authority:
            grants |= declared_grants(compilation)

    The checker's grants are not included either.  They are the same
    declarations seen from the older front end, so including them would reopen
    the hole this closes.
    """
    return set(extra)


def find_entry(compilation: Compilation, preferred: str = "main") -> Optional[str]:
    """Pick an entry point: `main` if present, else a `main()`-like component."""
    if compilation.program is None:
        return None
    if preferred in compilation.program.functions:
        return preferred
    if "<main>" in compilation.program.functions:
        return "<main>"
    return None
