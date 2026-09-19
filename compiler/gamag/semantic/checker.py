"""Semantic analysis for Gama-G (spec section 22, steps 3-7).

Runs, in the specification's order:
    3. name resolution
    4. type checking
    5. effect checking
    6. capability checking
    7. ownership analysis (the secret-lifecycle and mutability subset)

Diagnostics accumulate rather than aborting on the first problem, so
``ggc check`` can report a whole file the way an editor needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import ast_nodes as A
from ..diagnostics import Diagnostic, DiagnosticBag, Phase, Severity, SourcePos
from ..std import library as L
from . import types as T

# Builtins allowed to receive a `secret` value (spec section 8).
SECRET_SAFE = L.SECRET_SAFE

# Effects that a `deterministic` function must not perform (spec 1.3).
NONDETERMINISTIC_EFFECTS = {"io", "network", "storage"}

ASSIGN_OPS = {"=": None, "+=": "+", "-=": "-", "*=": "*", "/=": "/"}

NUMERIC_OPS = {"+", "-", "*", "/", "%", "**"}


@dataclass
class Symbol:
    name: str
    type: T.Type
    kind: str = "var"        # var|param|fn|record|enum|variant|module|builtin|model|service|pipeline|policy|agent|const
    mutable: bool = False
    secret: bool = False
    decl: Any = None
    pos: Optional[SourcePos] = None
    effects: Tuple[str, ...] = ()
    caps: Tuple[str, ...] = ()
    builtin: Optional[L.Builtin] = None

    @property
    def is_value(self) -> bool:
        return self.kind in ("var", "param", "const", "variant")


class Scope:
    def __init__(self, parent: Optional["Scope"] = None, name: str = ""):
        self.parent = parent
        self.name = name
        self.symbols: Dict[str, Symbol] = {}

    def define(self, sym: Symbol) -> Optional[Symbol]:
        previous = self.symbols.get(sym.name)
        self.symbols[sym.name] = sym
        return previous

    def lookup(self, name: str) -> Optional[Symbol]:
        scope: Optional[Scope] = self
        while scope is not None:
            sym = scope.symbols.get(name)
            if sym is not None:
                return sym
            scope = scope.parent
        return None

    def child(self, name: str = "") -> "Scope":
        return Scope(self, name)


@dataclass
class FunctionInfo:
    """Everything later phases need about one checked function."""

    name: str
    decl: Any
    fn_type: T.FnType
    effects: Set[str] = field(default_factory=set)
    declared_effects: List[str] = field(default_factory=list)
    caps: Set[str] = field(default_factory=set)
    deterministic: bool = False
    scope: Optional[Scope] = None
    returns: bool = False
    kind: str = "fn"


class Checker:
    def __init__(self, module: A.Module, source: str = "",
                 profile: str = "strict"):
        self.module = module
        self.source = source
        self.profile = profile          # strict | standard | lenient
        self.bag = DiagnosticBag()
        self.globals = Scope(name="<module>")
        self.records: Dict[str, T.RecordType] = {}
        self.enums: Dict[str, T.EnumType] = {}
        self.variants: Dict[str, Tuple[str, T.EnumType]] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.models: Dict[str, A.ModelDecl] = {}
        self.services: Dict[str, A.ServiceDecl] = {}
        self.pipelines: Dict[str, A.PipelineDecl] = {}
        self.policies: Dict[str, A.PolicyDecl] = {}
        self.agents: Dict[str, A.AgentDecl] = {}
        self.faults: Dict[str, A.FaultDecl] = {}
        self.transactions: Dict[str, A.TransactionDecl] = {}
        self.tests: List[A.TestDecl] = []
        self.grants: Set[str] = set(module.grants)
        # ids of NamePat nodes that denote an enum variant rather than a
        # binding; the GIR builder reads this to emit a tag test.
        self.variant_patterns: Set[int] = set()
        # Suppresses diagnostics during speculative inference (see
        # _declare_parallel_outputs).
        self._quiet = False
        self.imports: Set[str] = set()
        self.current_fn: Optional[FunctionInfo] = None
        self.used_results: Set[int] = set()
        self.secret_flows: List[Tuple[SourcePos, str]] = []
        # name/member resolution results, keyed by AST node identity, so the
        # GIR builder does not have to repeat scope resolution.
        self.resolved: Dict[int, Any] = {}

        # Domain types provided by the standard library (spec section 16).
        for name, ty in L.const_types.items():
            if isinstance(ty, T.RecordType):
                self.records[name] = ty
            elif isinstance(ty, T.EnumType):
                self.enums[name] = ty

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def error(self, message: str, pos: Optional[SourcePos] = None,
              phase: Phase = Phase.TYPE, code: Optional[str] = None,
              help_text: Optional[str] = None) -> None:
        if self._quiet:
            return
        self.bag.error(message, pos, phase=phase, code=code, help_text=help_text)

    def warn(self, message: str, pos: Optional[SourcePos] = None,
             phase: Phase = Phase.TYPE, code: Optional[str] = None,
             help_text: Optional[str] = None) -> None:
        if self._quiet or self.profile == "lenient":
            return
        self.bag.warning(message, pos, phase=phase, code=code,
                         help_text=help_text)

    def strict_error(self, message: str, pos=None, **kw) -> None:
        """An error in strict profiles, a warning otherwise."""
        if self.profile == "strict":
            self.error(message, pos, **kw)
        else:
            self.warn(message, pos, **kw)

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def check(self) -> DiagnosticBag:
        self.register_prelude()
        self.collect_declarations()
        self.resolve_grants()
        # Top-level statements are checked *first* and their bindings are
        # published at module scope, so a function body can refer to
        # module-level state such as the `secret patientRecord` binding of
        # spec section 8.  Signatures were already collected above, so the
        # top level can still call functions declared anywhere in the module.
        self.check_top_level()
        for decl in self.module.decls:
            self.check_decl(decl)
        return self.bag

    def register_prelude(self) -> None:
        for name, b in L.prelude_symbols().items():
            self.globals.define(Symbol(
                name=name, type=b.fn_type(), kind="builtin", builtin=b,
                effects=b.effects, caps=b.caps))
        for name, c in L.CONSTANTS.items():
            short = name.split(".")[-1]
            self.globals.define(Symbol(name=short, type=c.type, kind="const",
                                       pos=None))
        for name, ty in L.MODULE_TYPE_NAMES.items():
            self.globals.define(Symbol(name=name, type=ty, kind="module"))
        for name, rec in self.records.items():
            self.globals.define(Symbol(name=name, type=rec, kind="record"))
        for name, en in self.enums.items():
            self.globals.define(Symbol(name=name, type=en, kind="enum"))
            for variant in en.variants:
                vtype = (en if not variant.params else
                         T.FnType(tuple(t for _, t in variant.params), en,
                                  param_names=tuple(n for n, _ in variant.params),
                                  name=variant.name))
                self.globals.define(Symbol(name=variant.name, type=vtype,
                                           kind="variant"))
                self.variants[variant.name] = (name, en)

    def resolve_grants(self) -> None:
        for imp in self.module.imports:
            self.imports.add(imp.path)
        for cap in sorted(self.grants):
            if cap not in L.KNOWN_CAPABILITIES and cap not in (
                    "Read", "Write", "*"):
                self.warn(f"unknown capability `{cap}`",
                          phase=Phase.CAPABILITY, code="E-cap-unknown",
                          help_text="known capabilities: "
                                    + ", ".join(sorted(L.KNOWN_CAPABILITIES)))

    # ------------------------------------------------------------------
    # declaration collection (pass 1: names before bodies)
    # ------------------------------------------------------------------
    def collect_declarations(self) -> None:
        for decl in self.module.decls:
            if isinstance(decl, A.RecordDecl):
                fields = tuple((f.name, self.resolve_type(f.type))
                               for f in decl.fields)
                rec = T.RecordType(decl.name, fields)
                self.records[decl.name] = rec
                self.globals.define(Symbol(decl.name, rec, "record", decl=decl,
                                           pos=decl.pos))
            elif isinstance(decl, A.EnumDecl):
                variants = tuple(
                    T.EnumVariantInfo(v.name,
                                      tuple((p.name, self.resolve_type(p.type))
                                            for p in v.params))
                    for v in decl.variants)
                en = T.EnumType(decl.name, variants)
                self.enums[decl.name] = en
                self.globals.define(Symbol(decl.name, en, "enum", decl=decl,
                                           pos=decl.pos))
                for v in decl.variants:
                    vinfo = next(x for x in variants if x.name == v.name)
                    vtype = (en if not vinfo.params else
                             T.FnType(tuple(t for _, t in vinfo.params), en,
                                      param_names=tuple(n for n, _ in vinfo.params),
                                      name=v.name))
                    self.globals.define(Symbol(v.name, vtype, "variant",
                                               decl=decl, pos=v.pos))
                    self.variants[v.name] = (decl.name, en)

        # Functions, models and pipelines next, so bodies can refer forward.
        for decl in self.module.decls:
            if isinstance(decl, A.FnDecl):
                self.register_function(decl)
            elif isinstance(decl, A.PipelineDecl):
                self.register_pipeline(decl)
            elif isinstance(decl, A.ModelDecl):
                self.models[decl.name] = decl
                self.globals.define(Symbol(decl.name, T.ANY, "model",
                                           decl=decl, pos=decl.pos))
                for method in decl.methods:
                    method.name = f"{decl.name}.{method.name or 'predict'}"
                    self.register_function(method, owner=decl.name)
            elif isinstance(decl, A.ServiceDecl):
                if decl.name != "<module>":
                    self.services[decl.name] = decl
                    self.globals.define(Symbol(decl.name, T.ANY, "service",
                                               decl=decl, pos=decl.pos))
            elif isinstance(decl, A.PolicyDecl):
                self.policies[decl.name] = decl
                self.globals.define(Symbol(decl.name, T.ANY, "policy",
                                           decl=decl, pos=decl.pos))
            elif isinstance(decl, A.AgentDecl):
                self.agents[decl.name] = decl
                self.globals.define(Symbol(decl.name, T.ANY, "agent",
                                           decl=decl, pos=decl.pos))
            elif isinstance(decl, A.FaultDecl):
                self.faults[decl.name] = decl
                self.globals.define(Symbol(decl.name, T.ANY, "fault",
                                           decl=decl, pos=decl.pos))
            elif isinstance(decl, A.TransactionDecl):
                self.transactions[decl.name] = decl
                self.globals.define(Symbol(decl.name, T.ANY, "transaction",
                                           decl=decl, pos=decl.pos))
            elif isinstance(decl, A.TestDecl):
                self.tests.append(decl)
                self.globals.define(Symbol(f"test:{decl.name}", T.UNIT,
                                           "test", decl=decl, pos=decl.pos))

    def register_function(self, decl: A.FnDecl, owner: str = "") -> FunctionInfo:
        params = tuple(self.resolve_type(p.type) if p.type else T.ANY
                       for p in decl.params)
        ret = self.resolve_type(decl.ret) if decl.ret else T.UNIT
        caps: Set[str] = set()
        for p in decl.params:
            if p.type and p.type.caps:
                caps.update(p.type.caps)
        ftype = T.FnType(params, ret,
                         param_names=tuple(p.name for p in decl.params),
                         effects=tuple(decl.effects),
                         generics=tuple(decl.generics), name=decl.name)
        info = FunctionInfo(name=decl.name, decl=decl, fn_type=ftype,
                            declared_effects=list(decl.effects), caps=caps,
                            deterministic=decl.deterministic,
                            kind="method" if owner else "fn")
        self.functions[decl.name] = info
        self.globals.define(Symbol(decl.name, ftype, "fn", decl=decl,
                                   pos=decl.pos, effects=tuple(decl.effects),
                                   caps=tuple(caps)))
        return info

    def register_pipeline(self, decl: A.PipelineDecl) -> FunctionInfo:
        params = tuple(self.resolve_type(p.type) if p.type else T.ANY
                       for p in decl.params)
        for extra in decl.inputs:
            params = params + (self.resolve_type(extra.type) if extra.type else T.ANY,)
        ret = self.resolve_type(decl.ret) if decl.ret else T.ANY
        ftype = T.FnType(params, ret, name=decl.name,
                         effects=tuple(decl.effects))
        info = FunctionInfo(name=decl.name, decl=decl, fn_type=ftype,
                            declared_effects=list(decl.effects), kind="pipeline")
        self.functions[decl.name] = info
        self.pipelines[decl.name] = decl
        self.globals.define(Symbol(decl.name, ftype, "pipeline", decl=decl,
                                   pos=decl.pos))
        return info

    # ------------------------------------------------------------------
    # type resolution
    # ------------------------------------------------------------------
    def resolve_type(self, ref: Optional[A.TypeRef]) -> T.Type:
        if ref is None:
            return T.ANY
        shape: Optional[Tuple[Any, ...]] = None
        args: List[T.Type] = []
        for arg in ref.args:
            if isinstance(arg, A.ShapeLit):
                shape = tuple(arg.dims)
            else:
                args.append(self.resolve_type(arg))

        name = ref.name
        base_name = name.split(".")[-1]

        if ref.caps:
            return T.CapabilityType(name, tuple(ref.caps))
        if base_name in T.PRIMITIVES and not args:
            return T.PRIMITIVES[base_name]
        if base_name in T.GENERIC_ARITY and args:
            expected = T.GENERIC_ARITY[base_name]
            if expected is not None and len(args) != expected:
                self.error(
                    f"`{base_name}` expects {expected} type argument(s) but "
                    f"got {len(args)}", ref.pos, code="E-type-arity",
                    help_text=f"e.g. `{base_name}<"
                              + ", ".join(["..."] * expected) + ">")
                return T.ERROR
            try:
                return T.build_generic(base_name, args, shape)
            except KeyError:
                pass
        if base_name in self.records:
            return self.records[base_name]
        if base_name in self.enums:
            return self.enums[base_name]
        if shape is not None and args:
            return T.TensorType(args[0], shape)
        if "." in name:
            return T.NamedType(name, tuple(args))
        self.strict_error(
            f"unknown type `{name}`", ref.pos, code="E-type-unknown",
            help_text="declare it with `record`/`enum`, or use a built-in "
                      "type: " + ", ".join(sorted(
                          k for k in T.PRIMITIVES if k[0].isupper())[:8]) + ", ...")
        return T.NamedType(name, tuple(args))

    # ------------------------------------------------------------------
    # declaration checking (pass 2)
    # ------------------------------------------------------------------
    def check_decl(self, decl: A.Decl) -> None:
        if isinstance(decl, A.FnDecl):
            self.check_function(decl)
        elif isinstance(decl, A.PipelineDecl):
            self.check_pipeline(decl)
        elif isinstance(decl, A.ModelDecl):
            self.check_model(decl)
        elif isinstance(decl, A.ServiceDecl):
            self.check_service(decl)
        elif isinstance(decl, A.PolicyDecl):
            self.check_policy(decl)
        elif isinstance(decl, A.AgentDecl):
            self.check_agent(decl)
        elif isinstance(decl, A.FaultDecl):
            self.check_fault(decl)
        elif isinstance(decl, A.TransactionDecl):
            self.check_transaction(decl)
        elif isinstance(decl, A.TestDecl):
            self.check_test(decl)
        elif isinstance(decl, (A.RecordDecl, A.EnumDecl)):
            for f in getattr(decl, "fields", []):
                self.resolve_type(f.type)
        elif isinstance(decl, A.ServiceDecl):
            pass

    def _fn_scope(self, decl) -> Scope:
        scope = self.globals.child(f"fn:{decl.name}")
        params = decl.params
        if isinstance(decl, A.PipelineDecl):
            params = list(decl.params) + [
                A.Param(pos=i.pos, name=i.name, type=i.type)
                for i in decl.inputs]
        ftype = self.functions[decl.name].fn_type
        for param, ptype in zip(params, ftype.params):
            secret = isinstance(ptype, T.SecretType)
            scope.define(Symbol(param.name, ptype, "param",
                                mutable=False, secret=secret, pos=param.pos))
        return scope

    def check_function(self, decl: A.FnDecl) -> None:
        info = self.functions.get(decl.name)
        if info is None:
            info = self.register_function(decl)
        scope = self._fn_scope(decl)
        info.scope = scope
        previous = self.current_fn
        self.current_fn = info
        # Spec section 12: authority arrives as a parameter, never from the
        # environment.  `store: PatientStore[Read]` puts PatientRead in scope
        # for this body only, and disappears when the body ends.
        previous_grants = self.grants
        self.grants = previous_grants | self._parameter_capabilities(decl)
        body = decl.body or A.Block(stmts=[])
        self.check_block(body, scope)
        self.check_contracts(decl, info, scope)
        self.check_return_coverage(decl, info)
        self.check_effects(decl, info)
        self.grants = previous_grants
        self.current_fn = previous

    def _parameter_capabilities(self, decl: A.FnDecl) -> Set[str]:
        caps: Set[str] = set()
        for param in decl.params or ():
            ty = self.resolve_type(param.type) if param.type is not None \
                else T.ANY
            if isinstance(ty, T.CapabilityType):
                caps.update(L.expand_capability(ty.base, ty.caps))
        return caps

    def check_contracts(self, decl: A.FnDecl, info: FunctionInfo,
                        scope: Scope) -> None:
        """Runtime contracts (spec section 27) are type-checked here."""
        for contract in decl.contracts:
            cscope = scope.child("contract")
            if contract.kind == "ensures":
                cscope.define(Symbol("result", info.fn_type.ret, "param"))
            ty = self.infer(contract.expr, cscope)
            if not isinstance(ty, (T.BoolType, T.AnyType, T.ErrorType,
                                   T.NeverType)):
                self.error(
                    f"`{contract.kind}` must be a Bool expression, found "
                    f"{ty.render()}", contract.expr.pos, code="E-contract-type",
                    help_text="contracts are predicates, e.g. "
                              "`requires weight > 0`")

    def check_return_coverage(self, decl: A.FnDecl, info: FunctionInfo) -> None:
        ret = info.fn_type.ret
        if isinstance(ret, (T.UnitType, T.NeverType)):
            return
        if not info.returns:
            self.error(
                f"function `{decl.name}` declares a return type of "
                f"{ret.render()} but never returns a value",
                decl.pos, code="E-missing-return",
                help_text="add a `return` expression on every path")

    def check_effects(self, decl: A.FnDecl, info: FunctionInfo) -> None:
        """Effect checking (spec section 7)."""
        inferred = info.effects
        if "pure" in info.declared_effects:
            leaked = sorted(inferred - {"pure"})
            if leaked:
                self.error(
                    f"function `{decl.name}` is declared `pure` but performs "
                    f"{', '.join(leaked)}",
                    decl.pos, phase=Phase.EFFECT, code="E-effect-pure",
                    help_text="spec section 7: a pure function cannot silently "
                              "perform network access or modify persistent "
                              "state; remove the `pure` marker or remove the "
                              "effect")
        if info.deterministic:
            bad = sorted(inferred & NONDETERMINISTIC_EFFECTS)
            if bad:
                self.error(
                    f"function `{decl.name}` is declared `deterministic` but "
                    f"performs {', '.join(bad)}",
                    decl.pos, phase=Phase.EFFECT, code="E-effect-determinism",
                    help_text="spec section 1.3 requires reproducible results; "
                              "time, I/O and unseeded entropy break that")
        # An undeclared effect that the body performs is worth reporting so
        # signatures stay honest.
        undeclared = sorted(inferred - set(info.declared_effects) - {"pure"})
        if undeclared and info.declared_effects:
            self.warn(
                f"function `{decl.name}` performs "
                f"{', '.join(undeclared)} but does not declare it",
                decl.pos, phase=Phase.EFFECT, code="W-effect-undeclared",
                help_text="add the effect to the function body, e.g. a line "
                          "reading `" + ", ".join(undeclared) + "`")

    def check_pipeline(self, decl: A.PipelineDecl) -> None:
        info = self.functions[decl.name]
        scope = self._fn_scope(decl)
        info.scope = scope
        previous = self.current_fn
        self.current_fn = info
        for st in (decl.body.stmts if decl.body else []):
            self.check_stmt(st, scope)
        self.check_effects(decl, info)
        self.current_fn = previous

    def check_model(self, decl: A.ModelDecl) -> None:
        for io_decl in decl.inputs + decl.outputs:
            self.resolve_type(io_decl.type)
        for method in decl.methods:
            if method.name in self.functions:
                self.check_function(method)

    def check_service(self, decl: A.ServiceDecl) -> None:
        scope = self.globals.child(f"service:{decl.name}")
        info = FunctionInfo(name=f"{decl.name}.<protect>", decl=decl,
                            fn_type=T.FnType((), T.UNIT, name=decl.name),
                            kind="service")
        previous = self.current_fn
        self.current_fn = info
        if decl.protect:
            self.check_block(decl.protect, scope)
        for cp in decl.checkpoints:
            self.check_checkpoint(cp, scope)
        for step in decl.recover:
            if step.action == "restore" and "checkpoint" not in step.target \
                    and step.target:
                self.warn(f"recovery step `{step.raw}` has an unrecognised "
                          f"target `{step.target}`", decl.pos,
                          phase=Phase.TYPE, code="W-recovery-target")
        self.current_fn = previous
        if decl.audit_all:
            info.effects.add("audit")

    def check_agent(self, decl: A.AgentDecl) -> None:
        scope = self.globals.child(f"agent:{decl.name}")
        for handler in decl.handlers:
            hscope = scope.child(f"on:{handler.event}")
            hscope.define(Symbol(handler.event, T.ANY, "param",
                                 pos=handler.pos))
            info = FunctionInfo(name=f"{decl.name}.on.{handler.event}",
                                decl=decl, fn_type=T.FnType((), T.UNIT),
                                kind="agent")
            previous = self.current_fn
            self.current_fn = info
            if handler.body:
                self.check_block(handler.body, hscope)
            self.current_fn = previous

    def check_fault(self, decl: A.FaultDecl) -> None:
        scope = self.globals.child(f"fault:{decl.name}")
        for handler in decl.handlers:
            if handler.body:
                self.check_block(handler.body, scope)

    def check_transaction(self, decl: A.TransactionDecl) -> None:
        scope = self.globals.child(f"transaction:{decl.name}")
        info = FunctionInfo(name=decl.name, decl=decl,
                            fn_type=T.FnType((), T.UNIT), kind="transaction")
        previous = self.current_fn
        self.current_fn = info
        if decl.body:
            self.check_block(decl.body, scope)
        self.current_fn = previous

    def check_policy(self, decl: A.PolicyDecl) -> None:
        scope = self.globals.child(f"policy:{decl.name}")
        for rule in decl.rules:
            if isinstance(rule, A.PolicyRule) and rule.expr is not None:
                self.infer(rule.expr, scope)
            elif isinstance(rule, A.Require) and rule.expr is not None:
                self.infer(rule.expr, scope)

    def check_test(self, decl: A.TestDecl) -> None:
        scope = self.globals.child(f"test:{decl.name}")
        if decl.category not in L.TEST_CATEGORIES if hasattr(L, "TEST_CATEGORIES") \
                else False:
            pass
        info = FunctionInfo(name=f"test:{decl.name}", decl=decl,
                            fn_type=T.FnType((), T.UNIT), kind="test")
        previous = self.current_fn
        self.current_fn = info
        if decl.body:
            self.check_block(decl.body, scope)
        self.current_fn = previous

    def check_top_level(self) -> None:
        scope = self.globals.child("<main>")
        info = FunctionInfo(name="<main>", decl=None,
                            fn_type=T.FnType((), T.UNIT), kind="main")
        self.functions["<main>"] = info
        previous = self.current_fn
        self.current_fn = info
        self.main_scope = scope
        for st in self.module.top_level:
            if isinstance(st, A.LetDecl):
                # Module-level binding: visible to every function below.
                self.stmt_LetDecl(st, self.globals)
                continue
            self.check_stmt(st, scope)
        self.current_fn = previous
        info.scope = scope

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------
    def check_block(self, block: A.Block, scope: Scope) -> Scope:
        inner = scope.child("block")
        for st in block.stmts:
            self.check_stmt(st, inner)
        return inner

    def check_stmt(self, st: A.Stmt, scope: Scope) -> None:
        method = getattr(self, f"stmt_{type(st).__name__}", None)
        if method is None:
            self.error(f"unsupported statement `{type(st).__name__}`", st.pos,
                       code="E-unsupported")
            return
        method(st, scope)

    def stmt_Block(self, st: A.Block, scope: Scope) -> None:
        self.check_block(st, scope)

    def stmt_LetDecl(self, st: A.LetDecl, scope: Scope) -> None:
        declared = self.resolve_type(st.type) if st.type else None
        value_type = None
        if st.value is not None:
            value_type = self.infer(st.value, scope)
        if declared is not None and value_type is not None:
            if not value_type.assignable_to(declared):
                self.error(
                    f"cannot initialise `{st.name}` of type "
                    f"{declared.render()} with a value of type "
                    f"{value_type.render()}", st.pos, code="E-type-mismatch",
                    help_text=_conversion_hint(value_type, declared))
                final = declared
            else:
                final = declared
        elif declared is not None:
            final = declared
        else:
            final = value_type or T.ANY
        if st.secret:
            final = T.SecretType(final)
        self.resolved[id(st)] = final
        previous = scope.define(Symbol(
            st.name, final, "var", mutable=st.mutable, secret=st.secret,
            pos=st.pos, decl=st))
        if previous is not None and previous.kind in ("var", "param"):
            self.warn(
                f"`{st.name}` shadows an existing binding", st.pos,
                code="W-shadow",
                help_text="use `var` and assignment to mutate the existing "
                          "binding instead of redeclaring it")
        if st.secret:
            self.bag.add(Diagnostic(
                Severity.NOTE, Phase.OWNERSHIP,
                f"`{st.name}` is under secret lifecycle control", st.pos,
                help_text="spec section 8: it cannot be printed, serialised "
                          "or converted to Text"))

    def stmt_Assign(self, st: A.Assign, scope: Scope) -> None:
        target_type = self.infer(st.target, scope, assign_target=True)
        value_type = self.infer(st.value, scope)
        if st.op != "=":
            op = ASSIGN_OPS[st.op]
            if not _numeric_op_applies(op, target_type, value_type):
                self.error(
                    f"`{st.op}` cannot apply to {target_type.render()} and "
                    f"{value_type.render()}", st.pos, code="E-binop-type")
        elif not value_type.assignable_to(target_type):
            self.error(
                f"cannot assign {value_type.render()} to "
                f"{target_type.render()}", st.pos, code="E-type-mismatch",
                help_text=_conversion_hint(value_type, target_type))
        if isinstance(st.target, A.Name):
            sym = scope.lookup(st.target.id)
            if sym is not None and not sym.mutable:
                self.error(
                    f"cannot assign to `{st.target.id}`: it is immutable",
                    st.pos, phase=Phase.OWNERSHIP, code="E-immutable",
                    help_text="declare it with `var` instead of `let` "
                              "(spec section 4: values are immutable by "
                              "default)")
            if sym is not None and sym.secret:
                self.warn("assigning into a secret binding keeps it secret",
                          st.pos, phase=Phase.OWNERSHIP, code="W-secret")

    def stmt_ExprStmt(self, st: A.ExprStmt, scope: Scope) -> None:
        ty = self.infer(st.expr, scope)
        if isinstance(ty, T.ResultType):
            self.used_results.discard(id(st.expr))
            self.bag.add(Diagnostic(
                Severity.WARNING if self.profile != "strict" else Severity.ERROR,
                Phase.TYPE,
                "this expression produces a Result which is discarded",
                st.pos, code="E-unused-result",
                help_text="spec section 6: match on the Result, `unwrap` it "
                          "deliberately, or assign it to a binding"))
        elif isinstance(ty, T.OptionType):
            self.warn("this expression produces an Option which is discarded",
                      st.pos, code="W-unused-option")

    def stmt_If(self, st: A.If, scope: Scope) -> None:
        cond = self.infer(st.cond, scope)
        if not isinstance(cond, (T.BoolType, T.AnyType, T.ErrorType)):
            self.error(
                f"`if` requires a Bool condition, found {cond.render()}",
                st.cond.pos, code="E-cond-type",
                help_text="Gama-G does not treat other types as truthy; "
                          "write an explicit comparison")
        if st.then_body:
            self.check_block(st.then_body, scope)
        if st.else_body:
            self.check_block(st.else_body, scope)

    def stmt_While(self, st: A.While, scope: Scope) -> None:
        cond = self.infer(st.cond, scope)
        if not isinstance(cond, (T.BoolType, T.AnyType, T.ErrorType)):
            self.error(f"`while` requires a Bool condition, found "
                       f"{cond.render()}", st.cond.pos, code="E-cond-type")
        if st.body:
            self.check_block(st.body, scope)

    def stmt_For(self, st: A.For, scope: Scope) -> None:
        it = self.infer(st.iter, scope)
        elem = _iterable_element(it, st.iter.pos, self)
        inner = scope.child("for")
        inner.define(Symbol(st.var, elem, "var", mutable=False, pos=st.pos))
        if st.body:
            self.check_block(st.body, inner)

    def stmt_ForAll(self, st: A.ForAll, scope: Scope) -> None:
        """Property quantifier (spec section 26)."""
        domain_type = T.ListType(T.I64)
        if st.domain is not None:
            dt = self.infer(st.domain, scope)
            domain_type = dt
        elem = _iterable_element(domain_type, st.pos, self)
        inner = scope.child("for-all")
        inner.define(Symbol(st.var, elem, "var", pos=st.pos))
        if st.where is not None:
            wt = self.infer(st.where, inner)
            if not isinstance(wt, (T.BoolType, T.AnyType, T.ErrorType)):
                self.error("`where` must be a Bool predicate", st.where.pos,
                           code="E-cond-type")
        if st.body:
            self.check_block(st.body, inner)

    def stmt_Return(self, st: A.Return, scope: Scope) -> None:
        info = self.current_fn
        value_type = T.UNIT if st.value is None else self.infer(st.value, scope)
        if info is not None:
            info.returns = True
            expected = info.fn_type.ret
            if st.value is None and not isinstance(expected,
                                                    (T.UnitType, T.AnyType,
                                                     T.ErrorType, T.NeverType)):
                self.error(
                    f"`return` without a value inside a function declared to "
                    f"return {expected.render()}", st.pos,
                    code="E-return-type")
            elif st.value is not None and not value_type.assignable_to(expected):
                self.error(
                    f"expected a return value of type {expected.render()}, "
                    f"found {value_type.render()}", st.pos,
                    code="E-return-type",
                    help_text=_conversion_hint(value_type, expected))

    def stmt_Break(self, st: A.Break, scope: Scope) -> None:
        if not self._in_loop:
            self.error("`break` outside a loop", st.pos, code="E-break")

    def stmt_Continue(self, st: A.Continue, scope: Scope) -> None:
        if not self._in_loop:
            self.error("`continue` outside a loop", st.pos, code="E-continue")

    _in_loop = False

    def stmt_Match(self, st: A.Match, scope: Scope) -> None:
        subject = self.infer(st.subject, scope)
        covered: Set[str] = set()
        has_wildcard = False
        for arm in st.arms:
            ascope = scope.child("arm")
            bindings = self.check_pattern(arm.pattern, subject, ascope, st.pos)
            covered.update(bindings.get("tags", set()))
            if bindings.get("wildcard"):
                has_wildcard = True
            if arm.guard is not None:
                gt = self.infer(arm.guard, ascope)
                if not isinstance(gt, (T.BoolType, T.AnyType, T.ErrorType)):
                    self.error("a match guard must be Bool", arm.guard.pos,
                               code="E-cond-type")
                has_wildcard = has_wildcard  # guards do not add coverage
            for body_stmt in arm.body:
                self.check_stmt(body_stmt, ascope)
        self.check_exhaustive(st, subject, covered, has_wildcard)

    def check_pattern(self, pat: A.Pattern, subject: T.Type, scope: Scope,
                      pos: SourcePos) -> Dict[str, Any]:
        out: Dict[str, Any] = {"tags": set(), "wildcard": False}
        if isinstance(pat, A.WildcardPat):
            out["wildcard"] = True
            return out
        if isinstance(pat, A.NamePat):
            # A bare identifier in a pattern is genuinely ambiguous: it is
            # either a binding (`x`) or a nullary enum variant
            # (`DivideByZero`).  The parser cannot tell them apart without
            # semantic information, so the checker resolves it here: a name
            # that denotes a variant tests the tag and binds nothing.
            if pat.name in self.variants:
                enum_name, enum = self.variants[pat.name]
                if isinstance(subject, T.EnumType) \
                        and subject.name != enum_name:
                    self.error(
                        f"pattern `{pat.name}` belongs to enum {enum_name} but "
                        f"the subject is {subject.name}", pat.pos,
                        code="E-pattern-type")
                out["tags"].add(pat.name)
                self.variant_patterns.add(id(pat))
                return out
            if pat.name == "none" and isinstance(
                    subject, (T.OptionType, T.AnyType, T.ErrorType)):
                out["tags"].add("none")
                self.variant_patterns.add(id(pat))
                return out
            scope.define(Symbol(pat.name, subject, "var", pos=pat.pos))
            return out
        if isinstance(pat, A.LiteralPat):
            lit = _literal_type(pat.value, pat.lit_kind)
            if not lit.assignable_to(subject) and not isinstance(
                    subject, (T.AnyType, T.ErrorType)):
                self.error(
                    f"pattern of type {lit.render()} cannot match "
                    f"{subject.render()}", pat.pos, code="E-pattern-type")
            return out
        if isinstance(pat, A.CtorPat):
            tag = pat.tag
            if tag in ("ok", "fail"):
                if not isinstance(subject, (T.ResultType, T.AnyType, T.ErrorType)):
                    self.error(
                        f"pattern `{tag}(...)` requires a Result subject, "
                        f"found {subject.render()}", pat.pos,
                        code="E-pattern-type",
                        help_text="spec section 6: `ok`/`fail` match "
                                  "Result<T,E> values")
                else:
                    out["tags"].add(tag)
                    inner = (subject.ok if tag == "ok" else subject.err) \
                        if isinstance(subject, T.ResultType) else T.ANY
                    if pat.args:
                        for sub in pat.args:
                            sub_out = self.check_pattern(sub, inner, scope, pos)
                            out["tags"].update(sub_out["tags"])
                            out["wildcard"] = out["wildcard"] or sub_out["wildcard"]
                return out
            if tag in ("some", "none", "value", "err"):
                if tag in ("some", "none"):
                    if not isinstance(subject, (T.OptionType, T.AnyType,
                                                T.ErrorType)):
                        self.error(
                            f"pattern `{tag}` requires an Option subject, "
                            f"found {subject.render()}", pat.pos,
                            code="E-pattern-type")
                    else:
                        out["tags"].add(tag)
                        inner = subject.inner if isinstance(subject, T.OptionType) \
                            else T.ANY
                        for sub in pat.args:
                            sub_out = self.check_pattern(sub, inner, scope, pos)
                            out["tags"].update(sub_out["tags"])
                    return out
                inner = T.ANY
                for sub in pat.args:
                    self.check_pattern(sub, inner, scope, pos)
                return out
            if tag in self.variants:
                enum_name, enum = self.variants[tag]
                if isinstance(subject, T.EnumType) and subject.name != enum_name:
                    self.error(
                        f"pattern `{tag}` belongs to enum {enum_name} but the "
                        f"subject is {subject.name}", pat.pos,
                        code="E-pattern-type")
                out["tags"].add(tag)
                info = next((v for v in enum.variants if v.name == tag), None)
                if info:
                    if len(info.params) != len(pat.args):
                        self.error(
                            f"variant `{tag}` takes {len(info.params)} "
                            f"argument(s), pattern has {len(pat.args)}",
                            pat.pos, code="E-pattern-arity")
                    for sub, (_, ptype) in zip(pat.args, info.params):
                        self.check_pattern(sub, ptype, scope, pos)
                return out
            out["tags"].add(tag)
            for sub in pat.args:
                self.check_pattern(sub, T.ANY, scope, pos)
            return out
        self.error(f"unsupported pattern `{type(pat).__name__}`", pat.pos,
                   code="E-unsupported")
        return out

    def check_exhaustive(self, st: A.Match, subject: T.Type,
                         covered: Set[str], has_wildcard: bool) -> None:
        """Warn when a match can fail to cover its subject (spec section 6)."""
        if has_wildcard:
            return
        required: Optional[Set[str]] = None
        if isinstance(subject, T.ResultType):
            required = {"ok", "fail"}
        elif isinstance(subject, T.OptionType):
            required = {"some", "none"}
        elif isinstance(subject, T.EnumType):
            required = {v.name for v in subject.variants}
        if required is None:
            return
        missing = sorted(required - covered)
        if missing:
            self.error(
                f"`match` is not exhaustive: missing {', '.join(missing)}",
                st.pos, code="E-match-exhaustive",
                help_text="add the missing arm(s) or a `_ =>` fallback; "
                          "spec section 6 requires explicit handling")

    def stmt_Parallel(self, st: A.Parallel, scope: Scope) -> None:
        # Spec section 7 lists pure/io/network/storage/crypto/model/medical/
        # audit/unsafe as the effect vocabulary.  Concurrency is a scheduling
        # property the compiler derives from the operation graph (section 3),
        # not an effect a function declares, so a `parallel` region adds none.
        inner = scope.child("parallel")
        for sym_name, sym in scope.symbols.items():
            inner.symbols.setdefault(sym_name, sym)
        if st.body:
            self._declare_parallel_outputs(st.body.stmts, inner)
            for body_stmt in st.body.stmts:
                self.check_stmt(body_stmt, inner)
        for name, sym in inner.symbols.items():
            if name not in scope.symbols:
                scope.define(sym)

    def _declare_parallel_outputs(self, stmts: List[A.Stmt],
                                  inner: Scope) -> None:
        """Bind the names a parallel region writes before checking its body.

        Spec section 9C writes::

            parallel
                a = computeA()
                b = computeB()
                c = computeC(a, b)

        Those names are *outputs* of the region, not pre-existing variables,
        and they remain visible afterwards.  Declaring them up front, in
        program order, is also what lets `c` see `a` and `b`.  Types come
        from the right-hand side; that inference is run silently because the
        body is checked properly immediately afterwards and the diagnostics
        would otherwise be reported twice.
        """
        previous = self._quiet
        self._quiet = True
        try:
            for st in stmts:
                if not isinstance(st, A.Assign) or st.op != "=":
                    continue
                if not isinstance(st.target, A.Name):
                    continue
                name = st.target.id
                if inner.lookup(name) is not None:
                    continue
                ty = self.infer(st.value, inner)
                inner.define(Symbol(name, ty, "var", mutable=True))
        finally:
            self._quiet = previous

    def stmt_AuditRecord(self, st: A.AuditRecord, scope: Scope) -> None:
        if self.current_fn is not None:
            self.current_fn.effects.add("audit")
        seen: Set[str] = set()
        for key, expr in st.fields:
            if key in seen:
                self.error(f"duplicate audit field `{key}`", expr.pos,
                           code="E-audit-duplicate")
            seen.add(key)
            ty = self.infer(expr, scope)
            if isinstance(ty, T.SecretType):
                self.error(
                    f"audit field `{key}` would record a secret value",
                    expr.pos, phase=Phase.OWNERSHIP, code="E-secret-leak",
                    help_text="spec section 8: use `secrets.redact` or "
                              "`secrets.fingerprint` instead")
        if "action" not in seen:
            self.warn("audit record has no `action` field", st.pos,
                      phase=Phase.TYPE, code="W-audit-action",
                      help_text="spec section 13 lists actor, action, object "
                                "and reason as the expected fields")

    def stmt_Require(self, st: A.Require, scope: Scope) -> None:
        ty = self.infer(st.expr, scope)
        if not isinstance(ty, (T.BoolType, T.AnyType, T.ErrorType)):
            self.error(
                f"`require` expects a Bool condition, found {ty.render()}",
                st.expr.pos, code="E-cond-type",
                help_text="spec sections 15 and 27: contracts are predicates")

    def stmt_Assert(self, st: A.Assert, scope: Scope) -> None:
        ty = self.infer(st.expr, scope)
        if not isinstance(ty, (T.BoolType, T.AnyType, T.ErrorType)):
            self.error(f"`assert` expects a Bool condition, found "
                       f"{ty.render()}", st.expr.pos, code="E-cond-type")

    def stmt_CheckpointStmt(self, st: A.CheckpointStmt, scope: Scope) -> None:
        self.check_checkpoint(st, scope)

    def check_checkpoint(self, st: A.CheckpointStmt, scope: Scope) -> None:
        if st.mode == "every" and st.interval_seconds is not None:
            if st.interval_seconds <= 0:
                self.error("checkpoint interval must be positive", st.pos,
                           code="E-checkpoint")
        if st.mode == "at" and not st.boundary:
            self.error("`checkpoint at` needs a boundary name", st.pos,
                       code="E-checkpoint",
                       help_text="e.g. `checkpoint at transaction boundary`")

    def stmt_EffectDecl(self, st: A.EffectDecl, scope: Scope) -> None:
        for eff in st.effects:
            if eff not in ("pure", "io", "network", "storage", "crypto",
                           "model", "medical", "audit", "unsafe",
                           "deterministic"):
                self.error(f"unknown effect `{eff}`", st.pos,
                           phase=Phase.EFFECT, code="E-effect-unknown",
                           help_text="spec section 7 effects: pure, io, "
                                     "network, storage, crypto, model, "
                                     "medical, audit, unsafe")
            if self.current_fn is not None:
                self.current_fn.declared_effects.append(eff)

    def stmt_RecoveryStep(self, st: A.RecoveryStep, scope: Scope) -> None:
        if st.action not in ("retry", "restore", "restart", "replay", "alert",
                             "reconnect", "failover", "escalate", "reset"):
            self.error(f"unknown recovery action `{st.action}`", st.pos,
                       code="E-recovery-action",
                       help_text="spec section 10 levels: retry, reset, "
                                 "restore checkpoint, restart, failover, "
                                 "escalate")

    def stmt_PolicyRule(self, st: A.PolicyRule, scope: Scope) -> None:
        if st.expr is not None:
            self.infer(st.expr, scope)

    def stmt_AuditDirective(self, st: A.AuditDirective, scope: Scope) -> None:
        if self.current_fn is not None:
            self.current_fn.effects.add("audit")

    def stmt_StageDirective(self, st: A.StageDirective, scope: Scope) -> None:
        for arg in st.args:
            self.infer(arg, scope)
        if st.using is not None:
            self.infer(st.using, scope)

    def stmt_Section(self, st: A.Section, scope: Scope) -> None:
        if st.body:
            self.check_block(st.body, scope)

    def stmt_HandlerDecl(self, st: A.HandlerDecl, scope: Scope) -> None:
        inner = scope.child(f"on:{st.event}")
        inner.define(Symbol(st.event, T.ANY, "param", pos=st.pos))
        if st.body:
            self.check_block(st.body, inner)

    def stmt_IODirective(self, st: A.IODirective, scope: Scope) -> None:
        ty = self.resolve_type(st.type)
        scope.define(Symbol(st.name, ty, "param", pos=st.pos))

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def infer(self, expr: Optional[A.Expr], scope: Scope,
              assign_target: bool = False) -> T.Type:
        if expr is None:
            return T.UNIT
        method = getattr(self, f"expr_{type(expr).__name__}", None)
        if method is None:
            self.error(f"unsupported expression `{type(expr).__name__}`",
                       expr.pos, code="E-unsupported")
            expr.inferred = T.ERROR
            return T.ERROR
        ty = method(expr, scope, assign_target)
        expr.inferred = ty
        return ty

    def expr_Literal(self, e: A.Literal, scope: Scope, at: bool) -> T.Type:
        return _literal_type(e.value, e.lit_kind)

    def expr_Name(self, e: A.Name, scope: Scope, at: bool) -> T.Type:
        sym = scope.lookup(e.id)
        if sym is not None:
            self.resolved[id(e)] = sym
        if sym is None:
            self.strict_error(
                f"cannot find `{e.id}` in this scope", e.pos,
                phase=Phase.RESOLVE, code="E-unresolved-name",
                help_text=self._name_hint(e.id))
            return T.ERROR
        if sym.kind == "module":
            return sym.type
        return sym.type

    def _name_hint(self, name: str) -> Optional[str]:
        candidates = set(self.globals.symbols) | set(L.PRELUDE)
        near = [c for c in candidates if _close(c, name)][:3]
        if near:
            return "did you mean " + ", ".join(f"`{c}`" for c in near) + "?"
        if name in L.UNIMPLEMENTED_MODULES:
            return f"module `{name}` is not implemented in v0.1: " \
                   f"{L.UNIMPLEMENTED_MODULES[name]}"
        return None

    def expr_Binary(self, e: A.Binary, scope: Scope, at: bool) -> T.Type:
        left_raw = self.infer(e.left, scope)
        right_raw = self.infer(e.right, scope)
        # Spec section 6 forbids implicit *unsafe* conversion, but an integer
        # literal such as the `0` in `weight > 0` carries no precision to
        # lose: it simply denotes a value of whatever numeric type it meets.
        # Without this, comparing any F64 against a literal would be an error.
        left = _literal_numeric(left_raw, e.left, right_raw)
        right = _literal_numeric(right_raw, e.right, left_raw)
        op = e.op
        if op in ("and", "or"):
            for side, ty in (("left", left), ("right", right)):
                if not isinstance(ty, (T.BoolType, T.AnyType, T.ErrorType)):
                    self.error(
                        f"`{op}` requires Bool operands, but the {side} "
                        f"operand is {ty.render()}", e.pos, code="E-binop-type")
            return T.BOOL
        if op in ("==", "!="):
            if not _comparable(left, right):
                self.error(
                    f"cannot compare {left.render()} with {right.render()}",
                    e.pos, code="E-binop-type",
                    help_text=_conversion_hint(left, right))
            return T.BOOL
        if op in ("<", ">", "<=", ">="):
            if not _ordered(left, right):
                self.error(
                    f"`{op}` is not defined for {left.render()} and "
                    f"{right.render()}", e.pos, code="E-binop-type",
                    help_text="ordering is defined for numbers, Text and "
                              "Duration")
            return T.BOOL
        if op in NUMERIC_OPS:
            if op == "+" and _is_text(left) and _is_text(right):
                return T.TEXT
            if op == "+" and isinstance(left, T.ListType) and \
                    isinstance(right, T.ListType):
                joined = T.unify(left.elem, right.elem)
                return T.ListType(joined)
            if op == "*" and _is_text(left) and isinstance(right, T.IntType):
                return T.TEXT
            if op == "*" and isinstance(right, T.TextType) and \
                    isinstance(left, T.IntType):
                return T.TEXT
            if isinstance(left, T.DurationType) or isinstance(right, T.DurationType):
                other = right if isinstance(left, T.DurationType) else left
                if op in ("+", "-") and isinstance(other, T.DurationType):
                    return T.DURATION
                if op == "*" and isinstance(other, (T.IntType, T.FloatType)):
                    return T.DURATION
                if op == "/" and isinstance(left, T.DurationType) and \
                        isinstance(right, (T.IntType, T.FloatType)):
                    return T.DURATION
            if not left.is_numeric or not right.is_numeric:
                self.error(
                    f"`{op}` cannot apply to {left.render()} and "
                    f"{right.render()}", e.pos, code="E-binop-type",
                    help_text=_conversion_hint(left, right))
                return T.ERROR
            joined = T.unify(left, right)
            if isinstance(joined, T.ErrorType):
                self.error(
                    f"`{op}` requires operands of the same numeric family, "
                    f"got {left.render()} and {right.render()}", e.pos,
                    code="E-binop-type",
                    help_text=_conversion_hint(left, right))
                return T.ERROR
            if op == "/":
                if isinstance(joined, T.FloatType):
                    return joined
                return joined          # integer division stays integral
            return joined
        self.error(f"unknown operator `{op}`", e.pos, code="E-binop-type")
        return T.ERROR

    def expr_Unary(self, e: A.Unary, scope: Scope, at: bool) -> T.Type:
        operand = self.infer(e.operand, scope)
        if e.op in ("!", "not"):
            if not isinstance(operand, (T.BoolType, T.AnyType, T.ErrorType)):
                self.error(f"`{e.op}` requires a Bool operand, found "
                           f"{operand.render()}", e.pos, code="E-unop-type")
            return T.BOOL
        if e.op == "-":
            if not operand.is_numeric:
                self.error(f"unary `-` requires a numeric operand, found "
                           f"{operand.render()}", e.pos, code="E-unop-type")
            return operand
        self.error(f"unknown unary operator `{e.op}`", e.pos, code="E-unop-type")
        return T.ERROR

    def expr_RangeExpr(self, e: A.RangeExpr, scope: Scope, at: bool) -> T.Type:
        lo = self.infer(e.start, scope)
        hi = self.infer(e.end, scope)
        if isinstance(lo, T.ErrorType) or isinstance(hi, T.ErrorType):
            return T.ERROR
        if isinstance(lo, T.SecretType) or isinstance(hi, T.SecretType):
            self.error(
                "range bounds cannot be secret values", e.pos,
                code="E-secret-leak",
                help_text="a range materialises its bounds into a list, which "
                          "would expose the secret")
            return T.ERROR
        elem = T.unify(lo, hi)
        if isinstance(elem, T.ErrorType):
            self.error(
                f"range bounds have incompatible types {lo.render()} and "
                f"{hi.render()}", e.pos, code="E-range-type")
            return T.ERROR
        if not isinstance(elem, (T.IntType, T.FloatType)):
            self.error(
                f"range bounds must be numeric, but {lo.render()} and "
                f"{hi.render()} are not", e.pos, code="E-range-type",
                help_text="`a..b` iterates from a up to (but not including) b; "
                          "`a..=b` includes b")
            return T.ERROR
        return T.ListType(elem)

    def expr_Call(self, e: A.Call, scope: Scope, at: bool) -> T.Type:
        callee_type = self.infer(e.callee, scope)
        arg_types = [self.infer(a, scope) for a in e.args]

        builtin = self._builtin_of(e.callee, scope)
        if builtin is not None:
            return self.check_builtin_call(builtin, arg_types, e)

        if isinstance(callee_type, T.FnType):
            return self.check_fn_call(callee_type, arg_types, e)
        if isinstance(callee_type, (T.AnyType, T.ErrorType)):
            return T.ANY
        if isinstance(callee_type, T.NamedType):
            return T.ANY
        self.error(
            f"{callee_type.render()} is not callable", e.pos,
            code="E-not-callable",
            help_text="only functions, pipelines and enum variant "
                      "constructors can be called")
        return T.ERROR

    def _builtin_of(self, callee: A.Expr, scope: Scope) -> Optional[L.Builtin]:
        if isinstance(callee, A.Member) and isinstance(callee.obj, A.Name):
            return L.lookup_member(callee.obj.id, callee.attr)
        if isinstance(callee, A.Name):
            sym = scope.lookup(callee.id)
            if sym is not None and sym.builtin is not None:
                return sym.builtin
        return None

    def check_builtin_call(self, b: L.Builtin, arg_types: List[T.Type],
                           e: A.Call) -> T.Type:
        self._record_builtin_effects(b, e)
        self.check_capability_requirements(b.caps, e.pos, b.name)
        self.check_secret_arguments(b, arg_types, e)

        if b.variadic:
            minimum = b.min_args if b.min_args is not None else len(b.params)
            if len(arg_types) < minimum:
                self.error(
                    f"`{b.name}` expects at least {minimum} argument(s), "
                    f"got {len(arg_types)}", e.pos, code="E-arity")
        else:
            expected = len(b.params)
            if len(arg_types) != expected:
                self.error(
                    f"`{b.name}` expects {expected} argument(s), got "
                    f"{len(arg_types)}", e.pos, code="E-arity",
                    help_text=f"signature: {b.name}(" + ", ".join(b.params) + ")")
                return b.fn_type(arg_types).ret
        for i, (declared, actual) in enumerate(zip(b.argtypes, arg_types)):
            if declared is None:
                continue
            if not actual.assignable_to(declared):
                self.error(
                    f"argument {i + 1} of `{b.name}` expects "
                    f"{declared.render()}, found {actual.render()}",
                    e.args[i].pos if i < len(e.args) else e.pos,
                    code="E-arg-type",
                    help_text=_conversion_hint(actual, declared))
        ret = b.ret
        if b.infer is not None:
            try:
                ret = b.infer(arg_types)
            except Exception as exc:                          # noqa: BLE001
                self.error(f"cannot infer the result type of `{b.name}`: {exc}",
                           e.pos, code="E-infer")
                return T.ERROR
        return ret if ret is not None else T.ANY

    def _record_builtin_effects(self, b: L.Builtin, e: A.Call) -> None:
        if self.current_fn is None:
            return
        for eff in b.effects:
            if eff != "pure":
                self.current_fn.effects.add(eff)
        if b.name.startswith(("medical.", "model.")):
            self.current_fn.effects.add(
                "medical" if b.name.startswith("medical.") else "model")

    def check_fn_call(self, ftype: T.FnType, arg_types: List[T.Type],
                      e: A.Call) -> T.Type:
        if len(arg_types) != len(ftype.params):
            self.error(
                f"`{ftype.name}` expects {len(ftype.params)} argument(s), got "
                f"{len(arg_types)}", e.pos, code="E-arity",
                help_text=f"signature: {ftype.name}("
                          + ", ".join(ftype.param_names) + ")")
            return ftype.ret
        for i, (expected, actual) in enumerate(zip(ftype.params, arg_types)):
            if not actual.assignable_to(expected):
                label = ftype.param_names[i] if i < len(ftype.param_names) \
                    else str(i + 1)
                self.error(
                    f"argument `{label}` of `{ftype.name}` expects "
                    f"{expected.render()}, found {actual.render()}",
                    e.args[i].pos if i < len(e.args) else e.pos,
                    code="E-arg-type",
                    help_text=_conversion_hint(actual, expected))
        if self.current_fn is not None:
            for eff in ftype.effects:
                # `pure` marks the absence of effects; propagating it would
                # wrongly taint callers (spec section 7).
                if eff != "pure":
                    self.current_fn.effects.add(eff)
        target = self.functions.get(ftype.name)
        if target is not None:
            if self.current_fn is not None:
                self.current_fn.caps.update(target.caps)
        return ftype.ret

    def check_capability_requirements(self, caps: Sequence[str],
                                      pos: SourcePos, what: str) -> None:
        """Capability checking (spec section 12): least privilege by default."""
        for cap in caps:
            if cap in self.grants or "*" in self.grants:
                continue
            if self.current_fn is not None:
                self.current_fn.caps.add(cap)
            self.error(
                f"`{what}` requires the `{cap}` capability, which this module "
                f"does not grant", pos, phase=Phase.CAPABILITY,
                code="E-capability-missing",
                help_text=f"add `grant {cap}` to the module header; spec "
                          f"section 12 gives programs no ambient access")

    def check_secret_arguments(self, b: L.Builtin, arg_types: List[T.Type],
                               e: A.Call) -> None:
        """Ownership/lifecycle analysis for secrets (spec section 8)."""
        if b.name in SECRET_SAFE:
            return
        for i, ty in enumerate(arg_types):
            if isinstance(ty, T.SecretType):
                self.error(
                    f"a secret value cannot be passed to `{b.name}`",
                    e.args[i].pos if i < len(e.args) else e.pos,
                    phase=Phase.OWNERSHIP, code="E-secret-leak",
                    help_text="spec section 8 restricts secrets from logging, "
                              "serialisation and conversion to Text; use "
                              "`secrets.redact`, `secrets.fingerprint` or an "
                              "audited `secrets.expose`")

    def expr_Member(self, e: A.Member, scope: Scope, at: bool) -> T.Type:
        obj_type = self.infer(e.obj, scope)
        if isinstance(obj_type, T.ModuleType):
            member = L.lookup_member(obj_type.name, e.attr)
            if member is None:
                available = L.MODULES.get(obj_type.name, [])
                near = [n.split(".")[-1] for n in available
                        if _close(n.split(".")[-1], e.attr)][:3]
                self.error(
                    f"module `{obj_type.name}` has no member `{e.attr}`",
                    e.pos, phase=Phase.RESOLVE, code="E-no-member",
                    help_text=("did you mean "
                               + ", ".join(f"`{n}`" for n in near) + "?")
                    if near else None)
                return T.ERROR
            self.resolved[id(e)] = member
            if isinstance(member, L.Constant):
                return member.type
            return member.fn_type()
        if isinstance(obj_type, T.RecordType):
            fields = obj_type.field_map()
            if e.attr in fields:
                return fields[e.attr]
            self.error(
                f"record `{obj_type.name}` has no field `{e.attr}`", e.pos,
                code="E-no-member",
                help_text="available fields: "
                          + ", ".join(sorted(fields)) if fields else None)
            return T.ERROR
        if isinstance(obj_type, T.CapabilityType):
            if e.attr in ("base", "caps", "token"):
                return T.TEXT if e.attr != "caps" else T.ListType(T.TEXT)
            return T.ANY
        if isinstance(obj_type, T.SecretType):
            self.error(
                f"cannot access `.{e.attr}` on a secret value", e.pos,
                phase=Phase.OWNERSHIP, code="E-secret-leak",
                help_text="secrets expose no fields; use `secrets.expose` "
                          "with a recorded reason")
            return T.ERROR
        known = _member_table(obj_type)
        if known is not None:
            if e.attr in known:
                return known[e.attr]
            self.error(
                f"{obj_type.render()} has no member `{e.attr}`", e.pos,
                code="E-no-member",
                help_text="available members: " + ", ".join(sorted(known)))
            return T.ERROR
        if isinstance(obj_type, (T.AnyType, T.ErrorType, T.NamedType)):
            return T.ANY
        if isinstance(obj_type, T.EnumType):
            for variant in obj_type.variants:
                if variant.name == e.attr:
                    return obj_type
            self.error(f"enum `{obj_type.name}` has no variant `{e.attr}`",
                       e.pos, code="E-no-member")
            return T.ERROR
        self.error(
            f"cannot access `.{e.attr}` on a value of type "
            f"{obj_type.render()}", e.pos, code="E-no-member")
        return T.ERROR

    def expr_Index(self, e: A.Index, scope: Scope, at: bool) -> T.Type:
        obj = self.infer(e.obj, scope)
        index = self.infer(e.index, scope)
        if isinstance(obj, T.ListType):
            if not isinstance(index, (T.IntType, T.AnyType, T.ErrorType)):
                self.error("a List index must be an integer, found "
                           f"{index.render()}", e.index.pos, code="E-index-type")
            return obj.elem
        if isinstance(obj, T.MapType):
            if not index.assignable_to(obj.key) and not isinstance(
                    index, (T.AnyType, T.ErrorType)):
                self.error(
                    f"this Map is keyed by {obj.key.render()}, but the index "
                    f"is {index.render()}", e.index.pos, code="E-index-type",
                    help_text="use `collections.map_get` for a checked lookup "
                              "returning Option")
            return obj.value
        if isinstance(obj, T.TextType):
            return T.CHAR
        if isinstance(obj, T.TensorType):
            if obj.shape is not None and len(obj.shape) == 1:
                return obj.elem
            if obj.shape is not None:
                return T.TensorType(obj.elem, obj.shape[1:])
            return T.ANY
        if isinstance(obj, T.TupleType):
            if isinstance(index, T.IntType) and isinstance(e.index, A.Literal) \
                    and isinstance(e.index.value, int):
                i = e.index.value
                if 0 <= i < len(obj.items):
                    return obj.items[i]
                self.error(f"tuple index {i} is out of range", e.index.pos,
                           code="E-index-range")
                return T.ERROR
            return T.ANY
        if isinstance(obj, (T.AnyType, T.ErrorType, T.NamedType)):
            return T.ANY
        self.error(f"{obj.render()} cannot be indexed", e.pos,
                   code="E-index-type",
                   help_text="indexing applies to List, Map, Text, Tensor "
                             "and Tuple")
        return T.ERROR

    def expr_ListLit(self, e: A.ListLit, scope: Scope, at: bool) -> T.Type:
        if not e.items:
            return T.ListType(T.ANY)
        types = [self.infer(i, scope) for i in e.items]
        joined = types[0]
        for ty in types[1:]:
            joined = T.unify(joined, ty)
        if isinstance(joined, T.ErrorType):
            self.error(
                "a List literal must have a single element type, found "
                + ", ".join(sorted({t.render() for t in types})),
                e.pos, code="E-heterogeneous",
                help_text="Gama-G Lists are homogeneous; use a Tuple for "
                          "mixed types")
            return T.ERROR
        return T.ListType(joined)

    def expr_SetLit(self, e: A.SetLit, scope: Scope, at: bool) -> T.Type:
        if not e.items:
            return T.SetType(T.ANY)
        types = [self.infer(i, scope) for i in e.items]
        joined = types[0]
        for ty in types[1:]:
            joined = T.unify(joined, ty)
        if isinstance(joined, T.ErrorType):
            self.error("a Set literal must have a single element type", e.pos,
                       code="E-heterogeneous")
            return T.ERROR
        return T.SetType(joined)

    def expr_MapLit(self, e: A.MapLit, scope: Scope, at: bool) -> T.Type:
        if not e.entries:
            return T.MapType(T.TEXT, T.ANY)
        keys = [self.infer(k, scope) for k, _ in e.entries]
        values = [self.infer(v, scope) for _, v in e.entries]
        k = keys[0]
        for tk in keys[1:]:
            k = T.unify(k, tk)
        v = values[0]
        for tv in values[1:]:
            v = T.unify(v, tv)
        if isinstance(k, T.ErrorType):
            self.error("a Map literal must have a single key type", e.pos,
                       code="E-heterogeneous")
            return T.ERROR
        if isinstance(v, T.ErrorType):
            self.error("a Map literal must have a single value type", e.pos,
                       code="E-heterogeneous")
            return T.ERROR
        return T.MapType(k, v)

    def expr_TupleLit(self, e: A.TupleLit, scope: Scope, at: bool) -> T.Type:
        return T.TupleType(tuple(self.infer(i, scope) for i in e.items))

    def expr_Construct(self, e: A.Construct, scope: Scope, at: bool) -> T.Type:
        args = [self.infer(a, scope) for a in e.args]
        if e.tag == "ok":
            err = self._expected_result_err(scope)
            return T.ResultType(args[0] if args else T.ANY, err)
        if e.tag in ("fail", "err"):
            ok = self._expected_result_ok(scope)
            return T.ResultType(ok, args[0] if args else T.ANY)
        if e.tag == "some":
            return T.OptionType(args[0] if args else T.ANY)
        if e.tag == "none":
            return T.OptionType(T.ANY)
        if e.tag in self.variants:
            enum_name, enum = self.variants[e.tag]
            info = next((v for v in enum.variants if v.name == e.tag), None)
            if info and len(info.params) != len(args):
                self.error(
                    f"variant `{e.tag}` takes {len(info.params)} argument(s), "
                    f"got {len(args)}", e.pos, code="E-arity")
            return enum
        return T.ANY

    def _expected_result_err(self, scope: Scope) -> T.Type:
        info = self.current_fn
        if info is not None and isinstance(info.fn_type.ret, T.ResultType):
            return info.fn_type.ret.err
        return T.ANY

    def _expected_result_ok(self, scope: Scope) -> T.Type:
        info = self.current_fn
        if info is not None and isinstance(info.fn_type.ret, T.ResultType):
            return info.fn_type.ret.ok
        return T.ANY

    def expr_Cast(self, e: A.Cast, scope: Scope, at: bool) -> T.Type:
        src = self.infer(e.expr, scope)
        target = self.resolve_type(e.target)
        if isinstance(src, (T.AnyType, T.ErrorType)) or isinstance(
                target, (T.AnyType, T.ErrorType)):
            return target
        legal = src.assignable_to(target) or target.assignable_to(src)
        if not legal and not _cast_allowed(src, target):
            self.error(
                f"cannot convert {src.render()} to {target.render()}",
                e.pos, code="E-cast",
                help_text="spec section 6 forbids implicit unsafe conversion; "
                          "use an explicit conversion function such as "
                          "`int`, `float` or `to_text`")
        return target

    def expr_RecordLit(self, e: A.RecordLit, scope: Scope, at: bool) -> T.Type:
        rec = self.records.get(e.name or "")
        field_types = [(k, self.infer(v, scope)) for k, v in e.fields]
        if rec is None:
            self.strict_error(
                f"unknown record type `{e.name}`", e.pos,
                phase=Phase.RESOLVE, code="E-unknown-record",
                help_text="declare it with `record " + (e.name or "Name") + "`")
            return T.ERROR
        expected = rec.field_map()
        given = dict(field_types)
        for name in sorted(set(expected) - set(given)):
            self.error(f"record `{e.name}` is missing field `{name}`", e.pos,
                       code="E-record-field",
                       help_text=f"expected type {expected[name].render()}")
        for name in sorted(set(given) - set(expected)):
            self.error(f"record `{e.name}` has no field `{name}`", e.pos,
                       code="E-record-field",
                       help_text="available fields: " + ", ".join(sorted(expected)))
        for name, ty in field_types:
            if name in expected and not ty.assignable_to(expected[name]):
                self.error(
                    f"field `{name}` of `{e.name}` expects "
                    f"{expected[name].render()}, found {ty.render()}",
                    e.pos, code="E-record-field",
                    help_text=_conversion_hint(ty, expected[name]))
        return rec


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _literal_type(value: Any, kind: str) -> T.Type:
    if kind == "int":
        return T.I64
    if kind == "float":
        return T.F64
    if kind == "string":
        return T.TEXT
    if kind == "char":
        return T.CHAR
    if kind == "bool":
        return T.BOOL
    if kind == "duration":
        return T.DURATION
    return T.UNIT


def _is_text(ty: T.Type) -> bool:
    return isinstance(ty, T.TextType)


def _numeric_op_applies(op: str, left: T.Type, right: T.Type) -> bool:
    if op == "+" and (_is_text(left) and _is_text(right)):
        return True
    return left.is_numeric and right.is_numeric and \
        not isinstance(T.unify(left, right), T.ErrorType)


def _comparable(a: T.Type, b: T.Type) -> bool:
    if isinstance(a, (T.AnyType, T.ErrorType)) or \
            isinstance(b, (T.AnyType, T.ErrorType)):
        return True
    if a.assignable_to(b) or b.assignable_to(a):
        return True
    return False


def _ordered(a: T.Type, b: T.Type) -> bool:
    if isinstance(a, (T.AnyType, T.ErrorType)) or \
            isinstance(b, (T.AnyType, T.ErrorType)):
        return True
    if isinstance(a, T.BoolType) or isinstance(b, T.BoolType):
        return False
    if a.is_numeric and b.is_numeric:
        return not isinstance(T.unify(a, b), T.ErrorType)
    if isinstance(a, T.TextType) and isinstance(b, T.TextType):
        return True
    if isinstance(a, T.DurationType) and isinstance(b, T.DurationType):
        return True
    return False


def _cast_allowed(src: T.Type, dst: T.Type) -> bool:
    if src.is_numeric and dst.is_numeric:
        return True
    if isinstance(src, T.CharType) and isinstance(dst, T.IntType):
        return True
    if isinstance(src, T.IntType) and isinstance(dst, T.CharType):
        return True
    if isinstance(src, T.TextType) and isinstance(dst, T.BytesType):
        return True
    if isinstance(src, T.BytesType) and isinstance(dst, T.TextType):
        return True
    if isinstance(src, T.SecretType):
        return False
    return False


def _iterable_element(ty: T.Type, pos: SourcePos, checker: Checker) -> T.Type:
    if isinstance(ty, T.ListType):
        return ty.elem
    if isinstance(ty, T.SetType):
        return ty.elem
    if isinstance(ty, T.MapType):
        return T.TupleType((ty.key, ty.value))
    if isinstance(ty, T.TextType):
        return T.CHAR
    if isinstance(ty, T.TensorType):
        return T.ANY
    if isinstance(ty, (T.AnyType, T.ErrorType, T.NamedType)):
        return T.ANY
    checker.error(f"{ty.render()} is not iterable", pos, code="E-not-iterable",
                  help_text="`for` iterates over List, Set, Map, Text and "
                            "Tensor values")
    return T.ERROR


def _literal_numeric(ty: T.Type, node: A.Expr, other: T.Type) -> T.Type:
    """Give an integer literal the numeric type of the operand it meets."""
    if isinstance(node, A.Literal) and node.lit_kind == "int" \
            and isinstance(other, (T.FloatType, T.DecimalType)):
        return other
    return ty


def _member_table(ty: T.Type) -> Optional[Dict[str, T.Type]]:
    """Static members of the built-in structured types."""
    if isinstance(ty, T.OptionType):
        return {"some": T.BOOL, "value": ty.inner, "is_some": T.BOOL}
    if isinstance(ty, T.ResultType):
        return {"ok": T.BOOL, "value": ty.ok, "error": ty.err}
    if isinstance(ty, T.TensorType):
        return {"shape": T.ListType(T.I64), "rank": T.I64, "size": T.I64,
                "dtype": T.TEXT, "data": T.ListType(ty.elem)}
    if isinstance(ty, T.TextType):
        return {"length": T.I64, "len": T.I64}
    if isinstance(ty, T.ListType):
        return {"length": T.I64, "size": T.I64, "len": T.I64}
    if isinstance(ty, T.SetType):
        return {"size": T.I64, "len": T.I64}
    if isinstance(ty, T.MapType):
        return {"size": T.I64, "len": T.I64, "keys": T.ListType(ty.key),
                "values": T.ListType(ty.value)}
    if isinstance(ty, T.DurationType):
        return {"seconds": T.F64, "ms": T.F64}
    if isinstance(ty, T.InstantType):
        return {"epoch": T.F64}
    if isinstance(ty, T.UUIDType):
        return {"text": T.TEXT}
    if isinstance(ty, T.URIType):
        return {"text": T.TEXT}
    return None


def _conversion_hint(src: T.Type, dst: T.Type) -> Optional[str]:
    if src.is_numeric and dst.is_numeric:
        fn = {True: "float", False: "int"}[isinstance(dst, T.FloatType)]
        return (f"Gama-G performs no implicit numeric conversion; write "
                f"`{fn}(x)` explicitly")
    if isinstance(src, T.TextType) and dst.is_numeric:
        return "parse the Text first, e.g. `text.parse_int(s)`"
    if src.is_numeric and isinstance(dst, T.TextType):
        return "convert with `to_text(x)`"
    if isinstance(src, T.OptionType):
        return "unwrap the Option first, e.g. `unwrap(x)` or `or_default(x, d)`"
    if isinstance(src, T.ResultType):
        return "handle the Result first with `match` or `unwrap`"
    return None


def _close(a: str, b: str) -> bool:
    if a == b:
        return True
    al, bl = a.lower(), b.lower()
    if al == bl:
        return True
    if len(al) > 3 and len(bl) > 3 and (al.startswith(bl[:3])
                                        or bl.startswith(al[:3])):
        return True
    if abs(len(al) - len(bl)) <= 2:
        diffs = sum(1 for x, y in zip(al, bl) if x != y)
        return diffs <= 2 and abs(len(al) - len(bl)) + diffs <= 3
    return False


def check_module(module: A.Module, source: str = "",
                 profile: str = "strict") -> Tuple[Checker, DiagnosticBag]:
    checker = Checker(module, source, profile)
    bag = checker.check()
    return checker, bag
