"""Typed-AST node definitions for Gama-G (spec section 22, steps 2-3).

Every node carries a source position so that all later phases -- name
resolution, type checking, effect checking, capability checking, ownership
analysis and GIR generation -- can produce precise diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .diagnostics import SourcePos


@dataclass
class Node:
    pos: SourcePos = field(default_factory=SourcePos, compare=False)
    end: Optional[SourcePos] = field(default=None, compare=False)

    @property
    def kind(self) -> str:
        return type(self).__name__


# ----------------------------------------------------------------------
# Type references (syntactic, pre-resolution)
# ----------------------------------------------------------------------
@dataclass
class TypeRef(Node):
    """A syntactic type: ``Result<F64, MathError>``, ``PatientStore[Read]``."""

    name: str = ""
    args: List["TypeArg"] = field(default_factory=list)
    caps: List[str] = field(default_factory=list)   # capability qualifiers

    def render(self) -> str:
        out = self.name
        if self.args:
            out += "<" + ", ".join(render_type_arg(a) for a in self.args) + ">"
        if self.caps:
            out += "[" + ", ".join(self.caps) + "]"
        return out


@dataclass
class ShapeLit(Node):
    """A tensor shape argument such as ``[1,224,224,3]`` or ``[N]``."""

    dims: List[Union[int, str]] = field(default_factory=list)

    def render(self) -> str:
        return "[" + ", ".join(str(d) for d in self.dims) + "]"


TypeArg = Union[TypeRef, ShapeLit]


def render_type_arg(arg: TypeArg) -> str:
    return arg.render()


# ----------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------
@dataclass
class Expr(Node):
    inferred: Optional[object] = field(default=None, compare=False)


@dataclass
class Literal(Expr):
    value: object = None
    lit_kind: str = "int"      # int|float|string|char|bool|none|duration


@dataclass
class Name(Expr):
    id: str = ""


@dataclass
class Binary(Expr):
    op: str = ""
    left: Optional[Expr] = None
    right: Optional[Expr] = None


@dataclass
class RangeExpr(Expr):
    """`a..b` (exclusive) or `a..=b` (inclusive).

    Not present in specification v1.0; added because integer iteration is
    otherwise inexpressible.  See docs/IMPLEMENTATION.md.
    """

    start: Optional[Expr] = None
    end: Optional[Expr] = None
    inclusive: bool = False


@dataclass
class Unary(Expr):
    op: str = ""
    operand: Optional[Expr] = None


@dataclass
class Call(Expr):
    callee: Optional[Expr] = None
    args: List[Expr] = field(default_factory=list)


@dataclass
class Member(Expr):
    obj: Optional[Expr] = None
    attr: str = ""


@dataclass
class Index(Expr):
    obj: Optional[Expr] = None
    index: Optional[Expr] = None


@dataclass
class ListLit(Expr):
    items: List[Expr] = field(default_factory=list)


@dataclass
class SetLit(Expr):
    items: List[Expr] = field(default_factory=list)


@dataclass
class MapLit(Expr):
    entries: List[tuple] = field(default_factory=list)   # (key_expr, value_expr)


@dataclass
class TupleLit(Expr):
    items: List[Expr] = field(default_factory=list)


@dataclass
class Construct(Expr):
    """``ok(v)`` / ``fail(e)`` / ``some(v)`` / ``none`` constructors."""

    tag: str = ""
    args: List[Expr] = field(default_factory=list)


@dataclass
class Cast(Expr):
    expr: Optional[Expr] = None
    target: Optional[TypeRef] = None


@dataclass
class RecordLit(Expr):
    """``Patient { name: "x", age: 3 }``."""

    name: Optional[str] = None
    fields: List[tuple] = field(default_factory=list)


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------
@dataclass
class Pattern(Node):
    pass


@dataclass
class WildcardPat(Pattern):
    pass


@dataclass
class NamePat(Pattern):
    name: str = ""


@dataclass
class LiteralPat(Pattern):
    value: object = None
    lit_kind: str = "int"


@dataclass
class CtorPat(Pattern):
    tag: str = ""
    args: List[Pattern] = field(default_factory=list)


@dataclass
class MatchArm(Node):
    pattern: Optional[Pattern] = None
    body: List["Stmt"] = field(default_factory=list)
    guard: Optional[Expr] = None


# ----------------------------------------------------------------------
# Statements
# ----------------------------------------------------------------------
@dataclass
class Stmt(Node):
    pass


@dataclass
class Block(Stmt):
    stmts: List[Stmt] = field(default_factory=list)
    braced: bool = False


@dataclass
class LetDecl(Stmt):
    name: str = ""
    type: Optional[TypeRef] = None
    value: Optional[Expr] = None
    mutable: bool = False
    secret: bool = False


@dataclass
class Assign(Stmt):
    target: Optional[Expr] = None
    op: str = "="
    value: Optional[Expr] = None


@dataclass
class ExprStmt(Stmt):
    expr: Optional[Expr] = None


@dataclass
class If(Stmt):
    cond: Optional[Expr] = None
    then_body: Optional[Block] = None
    else_body: Optional[Block] = None


@dataclass
class While(Stmt):
    cond: Optional[Expr] = None
    body: Optional[Block] = None


@dataclass
class For(Stmt):
    var: str = ""
    iter: Optional[Expr] = None
    body: Optional[Block] = None


@dataclass
class ForAll(Stmt):
    """Property-test quantifier: ``for all x where x != 0`` (spec section 26)."""

    var: str = ""
    domain: Optional[Expr] = None
    where: Optional[Expr] = None
    body: Optional[Block] = None
    samples: int = 100


@dataclass
class Return(Stmt):
    value: Optional[Expr] = None


@dataclass
class Break(Stmt):
    pass


@dataclass
class Continue(Stmt):
    pass


@dataclass
class Match(Stmt):
    subject: Optional[Expr] = None
    arms: List[MatchArm] = field(default_factory=list)


@dataclass
class Parallel(Stmt):
    """Structured data-parallel region (spec sections 3, 9C)."""

    body: Optional[Block] = None


@dataclass
class AuditRecord(Stmt):
    """``audit.record { actor: ..., action: ... }`` (spec section 13)."""

    fields: List[tuple] = field(default_factory=list)   # (key, expr)


@dataclass
class Require(Stmt):
    """Runtime contract / AI-safety gate (spec sections 15, 27)."""

    expr: Optional[Expr] = None
    message: Optional[str] = None


@dataclass
class Assert(Stmt):
    expr: Optional[Expr] = None
    message: Optional[str] = None


@dataclass
class CheckpointStmt(Stmt):
    """``checkpoint every 5s`` / ``checkpoint at transaction boundary``."""

    mode: str = "every"          # every | at
    interval_seconds: Optional[float] = None
    boundary: Optional[str] = None
    raw: str = ""


@dataclass
class EffectDecl(Stmt):
    """A bare effect modifier line such as ``pure`` (spec section 7)."""

    effects: List[str] = field(default_factory=list)


@dataclass
class RecoveryStep(Stmt):
    """One line of a ``recover`` block (spec section 10)."""

    action: str = ""             # retry|restore|restart|replay|alert|reconnect|failover|escalate
    count: Optional[int] = None
    target: str = ""             # e.g. "checkpoint", "operator", "safe events"
    raw: str = ""


@dataclass
class PolicyRule(Stmt):
    kind: str = ""               # allow|deny|require|audit
    expr: Optional[Expr] = None
    phrase: str = ""


@dataclass
class StageDirective(Stmt):
    """An operation-graph stage inside a ``pipeline`` (spec section 3)."""

    stage: str = ""              # normalize|clean|extract|predict|features|audit
    args: List[Expr] = field(default_factory=list)
    using: Optional[Expr] = None
    target: Optional[str] = None


@dataclass
class HandlerDecl(Stmt):
    """``on alert`` inside an ``agent`` or ``fault`` (spec sections 9B, 19)."""

    event: str = ""
    body: Optional[Block] = None
    steps: List[RecoveryStep] = field(default_factory=list)


@dataclass
class IODirective(Stmt):
    """``input patient: Medical.Patient`` / ``output risk: F32``."""

    direction: str = "input"
    name: str = ""
    type: Optional[TypeRef] = None


# ----------------------------------------------------------------------
# Top-level declarations
# ----------------------------------------------------------------------
@dataclass
class Decl(Node):
    name: str = ""


@dataclass
class Param(Node):
    name: str = ""
    type: Optional[TypeRef] = None


@dataclass
class Contract(Node):
    kind: str = "requires"       # requires | ensures
    expr: Optional[Expr] = None
    # The predicate exactly as written, so a violation can quote it.
    text: str = ""


@dataclass
class FnDecl(Decl):
    params: List[Param] = field(default_factory=list)
    ret: Optional[TypeRef] = None
    generics: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    contracts: List[Contract] = field(default_factory=list)
    body: Optional[Block] = None
    deterministic: bool = False
    unsafe: bool = False


@dataclass
class PipelineDecl(Decl):
    params: List[Param] = field(default_factory=list)
    ret: Optional[TypeRef] = None
    inputs: List[IODirective] = field(default_factory=list)
    outputs: List[IODirective] = field(default_factory=list)
    body: Optional[Block] = None
    effects: List[str] = field(default_factory=list)


@dataclass
class ServiceDecl(Decl):
    protect: Optional[Block] = None
    recover: List[RecoveryStep] = field(default_factory=list)
    checkpoints: List[CheckpointStmt] = field(default_factory=list)
    audit_all: bool = False
    body: Optional[Block] = None


@dataclass
class AgentDecl(Decl):
    handlers: List[HandlerDecl] = field(default_factory=list)
    body: Optional[Block] = None


@dataclass
class FaultDecl(Decl):
    handlers: List[HandlerDecl] = field(default_factory=list)
    body: Optional[Block] = None


@dataclass
class PolicyDecl(Decl):
    rules: List[PolicyRule] = field(default_factory=list)
    body: Optional[Block] = None


@dataclass
class TransactionDecl(Decl):
    body: Optional[Block] = None


@dataclass
class ModelDecl(Decl):
    inputs: List[IODirective] = field(default_factory=list)
    outputs: List[IODirective] = field(default_factory=list)
    methods: List[FnDecl] = field(default_factory=list)
    body: Optional[Block] = None


@dataclass
class RecordDecl(Decl):
    fields: List[Param] = field(default_factory=list)


@dataclass
class EnumVariant(Node):
    name: str = ""
    params: List[Param] = field(default_factory=list)


@dataclass
class EnumDecl(Decl):
    variants: List[EnumVariant] = field(default_factory=list)


@dataclass
class TestDecl(Decl):
    category: str = "unit"
    body: Optional[Block] = None


@dataclass
class GrantDecl(Decl):
    caps: List[str] = field(default_factory=list)


@dataclass
class ImportDecl(Decl):
    path: str = ""
    alias: Optional[str] = None


@dataclass
class ModuleDecl(Decl):
    pass


@dataclass
class Module(Node):
    """A whole compilation unit."""

    decls: List[Decl] = field(default_factory=list)
    top_level: List[Stmt] = field(default_factory=list)   # implicit `main`
    grants: List[str] = field(default_factory=list)
    imports: List[ImportDecl] = field(default_factory=list)
    module_name: Optional[str] = None
    filename: str = "<input>"


@dataclass
class Section(Stmt):
    """A named sub-block such as ``protect`` / ``recover`` (spec section 10)."""

    name: str = ""
    body: Optional[Block] = None
    steps: List[RecoveryStep] = field(default_factory=list)


@dataclass
class AuditDirective(Stmt):
    """``audit``, ``audit all recovery``, ``audit required`` (spec 13/17/40)."""

    phrase: str = ""
