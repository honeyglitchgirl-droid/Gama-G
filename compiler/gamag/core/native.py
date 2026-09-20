"""Native checking and direct lowering -- v0.3 Priority 2.

This module compiles the semantic model in :mod:`gamag.core.mir` straight to
GIR.  It does not build the older language's abstract syntax on the way, so
`let`, `if`, `while` and `match` never appear as an intermediate representation
of a core program.  That was the audit's central structural finding on v0.2:
*"Do not first convert Gama-G into let, if, while, match, conventional function
calls.  Instead, represent Gama-G's own semantic concepts directly."*

What lowering means here
------------------------
GIR is Gama-G's own intermediate representation, and it is a register machine
over basic blocks.  Emitting jumps and blocks is not a return to the older
language's model -- every compiler lowers to something, and the audit says so
explicitly.  What matters is *which* concepts survive the descent:

* a node's **identity** survives: every block is labelled with the operation
  that produced it, so a backend or a reader can see the graph in the IR;
* the **derived graph itself** survives, recorded in GIR's own
  operation-graph metadata (``parallel_tasks``) with each node's reads, writes
  and dependencies -- which is what makes the levels available to a backend
  that wants to run one in parallel;
* a **constraint** survives as ``Op.REQUIRE`` carrying its verbatim text, and a
  **guard** survives as ``Op.JUMP_IF`` over the guard's own block;
* a **bound** survives as a comparison against a literal and an ``Op.FAULT``
  whose kind is ``RefinementDiverged`` -- not as a call to a panic function;
* an **exhaustiveness failure** survives as ``Op.FAULT`` /
  ``Op.MATCH_FAIL``, which are GIR terminators, rather than as a synthesised
  arm of someone else's match construct.

What is deliberately reused
---------------------------
:mod:`gamag.semantic.types` for the type lattice, :mod:`gamag.std.library` for
builtin signatures and effects, and :mod:`gamag.gir.ir` for the IR itself. None
of those is a language model. Reimplementing them would create two competing
definitions of what `F64` means, which is a defect rather than independence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..diagnostics import DiagnosticBag, Phase, SourcePos
from ..gir.ir import (BasicBlock, GFunction, GProgram, Instr, Operand, Op,
                      ParallelTask, RecoveryPlan, Slot)
from ..semantic import types as T
from ..std import library as L
from . import capability as CAP
from . import memory as MEM
from . import recovery as REC
from . import mir as M

# Fault kinds the core can raise.  These are the core's own vocabulary: they
# name what went wrong in the program's terms, not in a host language's.
FAULT_NO_ALTERNATIVE = "NoActiveAlternative"
FAULT_DIVERGED = "RefinementDiverged"
FAULT_UNRESOLVED = "UnresolvedDispatch"


# ==========================================================================
# types
# ==========================================================================
def resolve_type(ref: Optional[M.MType], bag: DiagnosticBag,
                 what: str) -> T.Type:
    """Resolve a type as written into the shared type lattice."""
    if ref is None:
        return T.ANY
    base = T.PRIMITIVES.get(ref.name)
    if base is None and not ref.args:
        bag.error(f"`{ref.name}` is not a type the core knows", ref.pos,
                  phase=Phase.TYPE, code="E-unknown-type",
                  help_text=f"{what}; the core's primitive types are "
                            + ", ".join(sorted(
                                k for k in T.PRIMITIVES if k == k.capitalize()
                                or k.isupper()))
                  )
        return T.ERROR
    if ref.args:
        arity = T.GENERIC_ARITY.get(ref.name)
        if arity is None:
            bag.error(f"`{ref.name}` does not take type arguments", ref.pos,
                      phase=Phase.TYPE, code="E-unknown-type")
            return T.ERROR
        if isinstance(arity, int) and len(ref.args) != arity:
            bag.error(f"`{ref.name}` takes {arity} type argument(s), found "
                      f"{len(ref.args)}", ref.pos, phase=Phase.TYPE,
                      code="E-type-arity")
            return T.ERROR
        if isinstance(arity, tuple) and not (arity[0] <= len(ref.args)
                                             <= arity[1]):
            bag.error(f"`{ref.name}` takes {arity[0]} to {arity[1]} type "
                      f"arguments, found {len(ref.args)}", ref.pos,
                      phase=Phase.TYPE, code="E-type-arity")
            return T.ERROR
        args = [resolve_type(a, bag, what) for a in ref.args]
        base = T.build_generic(ref.name, args)
    if ref.secret:
        return T.SecretType(base)
    return base


def element_type(ty: T.Type) -> T.Type:
    """The element type of a collection, or ANY if it is not one."""
    if isinstance(ty, T.SecretType):
        return element_type(ty.inner)
    elem = getattr(ty, "elem", None)
    return elem if isinstance(elem, T.Type) else T.ANY


def unwrap(ty: T.Type) -> T.Type:
    return ty.inner if isinstance(ty, T.SecretType) else ty


def is_numeric(ty: T.Type) -> bool:
    return isinstance(unwrap(ty), (T.IntType, T.FloatType))


def is_secret(ty: T.Type) -> bool:
    return isinstance(ty, T.SecretType)


# ==========================================================================
# checking
# ==========================================================================
class NativeChecker:
    """Types, effects and secret flow for the semantic model.

    The environment maps a binding to its type.  Bindings are produced once, so
    there is no scope stack and no shadowing: the environment *is* the graph's
    producer table with types attached.
    """

    def __init__(self, model: M.SemanticModel, bag: DiagnosticBag):
        self.m = model
        self.bag = bag
        self.env: Dict[str, T.Type] = {}
        self.producer: Dict[str, M.OpNode] = {}
        self.node_effects: Dict[str, Set[str]] = {}
        # Expressions whose type has been settled.  Kept separately from the
        # type itself because ANY is a legitimate answer, not an absence of one.
        self.resolved: Set[int] = set()

    # ------------------------------------------------------------------
    def error(self, message: str, pos=None, code: str = "E-type",
              help_text: Optional[str] = None,
              phase: Phase = Phase.TYPE) -> None:
        self.bag.error(message, pos, phase=phase, code=code,
                       help_text=help_text)

    # ------------------------------------------------------------------
    def check(self) -> None:
        graph = self.m.operations
        # inputs first: everything else may refer to them
        for name, node in graph.inputs.items():
            type_ = resolve_type(node.type, self.bag,
                                 f"source `{name}` has no usable type")
            if node.secret and not is_secret(type_):
                type_ = T.SecretType(type_)
            self.env[name] = type_
            self.producer[name] = node
            if node.origin is not None:
                # A secret source's `from` is the point where the secret enters
                # the program.  Requiring that value to already be secret would
                # make `source secret x : T from e` unwritable, and the marking
                # is what the declaration does, not what the origin carries.
                self._check_value(node, node.origin, self.env[name],
                                  f"the origin of source `{name}`",
                                  secret_ok=True)
        for name, node in graph.resources.items():
            type_ = resolve_type(node.type, self.bag,
                                 f"state `{name}` has no usable type")
            if node.secret and not is_secret(type_):
                type_ = T.SecretType(type_)
            self.env[name] = type_
            self.producer[name] = node
            if node.initial is not None:
                self._check_value(node, node.initial, self.env[name],
                                f"the initial value of state `{name}`")
        # then the graph in derived order, so a node sees only real producers
        for name in graph.order():
            node = graph.nodes[name]
            self._check_node(node)

    # ------------------------------------------------------------------
    def _check_node(self, node: M.OpNode) -> None:
        declared = T.ANY
        if node.kind != "transition":
            if node.type is None:
                self.error(f"`{node.name}` yields `{node.produces}` without a "
                           f"type", node.pos, code="E-no-yields-type",
                           help_text="the graph needs a type for every binding "
                                     "so that consumers can be checked")
            declared = resolve_type(node.type, self.bag,
                                   f"`{node.name}` yields an untyped binding")
            if node.secret and not is_secret(declared):
                declared = T.SecretType(declared)

        if node.kind == "fanout":
            # a fan-out's declared type is its *element* type; the binding it
            # produces is a collection of those
            declared = self._check_fanout(node, declared)  # type: ignore[arg-type]
        elif node.kind == "dispatch":
            self._check_dispatch(node, declared)    # type: ignore[arg-type]
        elif node.kind == "refine":
            self._check_refine(node, declared)      # type: ignore[arg-type]
        else:
            if node.value is not None:              # type: ignore[attr-defined]
                self._check_value(node, node.value, declared,  # type: ignore[arg-type]
                                  f"what `{node.name}` computes")

        if node.kind != "transition":
            self.env[node.produces] = declared
            self.producer[node.produces] = node

        # Predicates are checked after the binding exists, because `holds`
        # constrains what this node produces -- that is the point of it.
        for hold in node.holds:
            self._check_predicate(node, hold)
        if node.when is not None:
            self._check_predicate(node, node.when)
        stop = getattr(node, "stop", None)
        if stop is not None:
            self._check_predicate(node, stop)
        self._check_effect(node)

    def _check_fanout(self, node: M.FanOutNode, declared: T.Type) -> T.Type:
        if node.collection is None:
            return
        collection_type = self.infer(node.collection)
        item_type = element_type(collection_type)
        outer = dict(self.env)
        if node.item:
            self.env[node.item] = item_type
        element_declared = unwrap(declared)
        if isinstance(element_declared, T.ListType):
            element_declared = element_declared.elem
        if node.value is not None:
            self._check_value(node, node.value, element_declared,
                              f"what `{node.name}` computes for each "
                              f"`{node.item}`", secret_ok=True)
        if node.when is not None:
            self._check_predicate(node, node.when)
        self.env = outer
        # the binding a fan-out produces is a collection of the declared
        # element; return the corrected type so the caller publishes it
        if not isinstance(unwrap(declared), T.ListType):
            node.type = M.MType(name="List",
                                args=[node.type] if node.type else [],
                                pos=node.pos)
            declared = T.ListType(element_declared)
            if node.secret:
                declared = T.SecretType(declared)
        return declared

    def _check_dispatch(self, node: M.DispatchNode, declared: T.Type) -> None:
        if node.subject is None:
            return
        self.infer(node.subject)
        outer = dict(self.env)
        for pattern, value in node.cases:
            for binding in M.pattern_bindings(pattern):
                self.env[binding] = T.ANY
            self._check_value(node, value, declared,
                              f"the `{pattern.text}` alternative of "
                              f"`{node.name}`")
            self.env = dict(outer)

    def _check_refine(self, node: M.RefinementNode, declared: T.Type) -> None:
        # the binding exists from `starts` onwards, so `repeats` and `until`
        # may refer to it: that is what makes a refinement a recurrence
        if node.start is not None:
            self._check_value(node, node.start, declared,
                              f"where `{node.name}` starts")
        self.env[node.produces] = declared
        self.producer[node.produces] = node
        if node.step is not None:
            self._check_value(node, node.step, declared,
                              f"what `{node.name}` repeats")

    # ------------------------------------------------------------------
    def _check_value(self, node: M.OpNode, expr: M.MExpr, declared: T.Type,
                     what: str, secret_ok: bool = False) -> None:
        actual = self.infer(expr)
        if isinstance(actual, T.ErrorType) or isinstance(declared, T.ErrorType):
            return
        if isinstance(actual, T.AnyType) or isinstance(declared, T.AnyType):
            return
        # Declaring a binding secret when its value is not one is a
        # *strengthening*, and the secret rule below only ever fires the other
        # way round -- so compare the two with the marking set aside.
        if is_secret(declared):
            actual_cmp, declared_cmp = unwrap(actual), unwrap(declared)
        else:
            actual_cmp, declared_cmp = actual, declared
        if not actual_cmp.assignable_to(declared_cmp):
            self.error(
                f"{what} has type {actual.render()} but `{node.produces}` is "
                f"declared {declared.render()}", expr.pos,
                code="E-yield-type",
                help_text="the core does not convert between types silently; "
                          "write the conversion, e.g. `float(x)`")
        if is_secret(actual) and not is_secret(declared) and not secret_ok:
            self.error(
                f"{what} carries a secret but `{node.produces}` is not declared "
                f"secret", expr.pos, phase=Phase.OWNERSHIP,
                code="E-secret-escape",
                help_text=f"write `yields secret {node.produces} : ...` so the "
                          f"secret keeps propagating, or redact it first")

    def _check_predicate(self, node: M.OpNode, constraint: M.Constraint
                         ) -> None:
        if constraint.expr is None:
            return
        actual = self.infer(constraint.expr)
        if isinstance(actual, (T.AnyType, T.ErrorType)):
            return
        if not isinstance(unwrap(actual), T.BoolType):
            self.error(
                f"`{constraint.kind} {constraint.text}` on `{node.name}` has "
                f"type {actual.render()}, but a {constraint.kind} must be a "
                f"question with a yes or no answer", constraint.expr.pos,
                code="E-constraint-type")

    # ------------------------------------------------------------------
    def _check_effect(self, node: M.OpNode) -> None:
        """A node's declared effect must cover what it actually does."""
        used: Set[str] = set()
        for expr in M.node_expressions(node):
            used |= self._effects_of(expr)
        self.node_effects[node.name] = used
        if not used:
            return
        declared = set(node.effects) or {"pure"}
        missing = sorted(used - declared)
        if missing:
            written = "`" + "`, `".join(sorted(declared)) + "`"
            self.error(
                f"`{node.name}` declares effect {written} but calls "
                f"{'`' + '`, `'.join(missing) + '`'} work", node.pos,
                phase=Phase.EFFECT, code="E-effect-undeclared",
                help_text="effects are declared, not inferred: an operation "
                          "that does something must say so, and `effect` takes a "
                          "list -- write `effect crypto, audit`")

    def _effects_of(self, expr: Optional[M.MExpr],
                    out: Optional[Set[str]] = None) -> Set[str]:
        found: Set[str] = out if out is not None else set()
        if expr is None:
            return found
        if isinstance(expr, M.MCall):
            builtin = L.BUILTINS.get(self._builtin_key(expr))
            if builtin is not None:
                found.update(builtin.effects)
        for value in vars(expr).values():
            if isinstance(value, M.MExpr):
                self._effects_of(value, found)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, M.MExpr):
                        self._effects_of(item, found)
        return found

    def _builtin_key(self, call: M.MCall) -> str:
        return f"{call.module}.{call.name}" if call.module else call.name

    def secret_bindings(self) -> Set[str]:
        """Every binding whose *resolved* type is secret.

        Not the same as the bindings whose declaration says `secret`. A value
        becomes secret by propagation -- anything computed from a secret carries
        the marking -- and the rules about where a secret may go have to see those
        too, or the propagation the checker just proved would stop mattering the
        moment the capability rules were derived.
        """
        return {name for name, type_ in self.env.items() if is_secret(type_)}

    # ------------------------------------------------------------------
    def infer(self, expr: Optional[M.MExpr]) -> T.Type:
        if expr is None:
            return T.ANY
        if id(expr) in self.resolved:
            return expr.type
        ty = self._infer(expr)
        expr.type = ty
        self.resolved.add(id(expr))
        return ty

    def _infer(self, expr: M.MExpr) -> T.Type:
        if isinstance(expr, M.MLit):
            return {"int": T.I64, "float": T.F64, "string": T.TEXT,
                    "char": T.CHAR, "bool": T.BOOL,
                    "unit": T.UNIT}.get(expr.kind, T.ANY)

        if isinstance(expr, M.MRef):
            if expr.binding not in self.env:
                self.error(f"`{expr.binding}` is not a binding this intent "
                           f"produces", expr.pos,
                           code="E-unresolved-relationship",
                           help_text="a reference must reach a `source`, a "
                                     "`state` or an operation's `yields`")
                return T.ERROR
            return self.env[expr.binding]

        if isinstance(expr, M.MItems):
            types = [self.infer(i) for i in expr.items]
            types = [t for t in types if not isinstance(t, T.ErrorType)]
            if not types:
                return T.ListType(T.ANY)
            joined = types[0]
            for other in types[1:]:
                joined = T.unify(joined, other)
            return T.ListType(joined)

        if isinstance(expr, M.MUn):
            operand = self.infer(expr.operand)
            if isinstance(operand, T.ErrorType):
                return T.ERROR
            if expr.op == "not":
                return T.BOOL
            if not is_numeric(operand):
                self.error(f"unary `{expr.op}` needs a number, found "
                           f"{operand.render()}", expr.pos, code="E-unop-type")
                return T.ERROR
            return operand

        if isinstance(expr, M.MBin):
            return self._infer_binary(expr)

        if isinstance(expr, M.MIndex):
            obj = self.infer(expr.obj)
            self.infer(expr.index)
            if isinstance(obj, T.ErrorType):
                return T.ERROR
            inner = unwrap(obj)
            if isinstance(inner, T.ListType):
                return inner.elem
            if isinstance(inner, T.MapType):
                return inner.value
            if isinstance(inner, T.TextType):
                return T.TEXT
            return T.ANY

        if isinstance(expr, M.MField):
            self.infer(expr.obj)
            # The core declares no record types yet, so a field access cannot be
            # resolved statically.  Returning ANY is honest; inventing a type
            # would not be.
            return T.ANY

        if isinstance(expr, M.MCall):
            return self._infer_call(expr)

        return T.ANY

    def _infer_binary(self, expr: M.MBin) -> T.Type:
        left = self.infer(expr.left)
        right = self.infer(expr.right)
        if isinstance(left, T.ErrorType) or isinstance(right, T.ErrorType):
            return T.ERROR
        op = expr.op
        secret = is_secret(left) or is_secret(right)
        plain_left, plain_right = unwrap(left), unwrap(right)

        if op in ("and", "or"):
            return T.BOOL
        if op in ("==", "!=", "<", ">", "<=", ">="):
            if op in ("<", ">", "<=", ">="):
                ordered = (is_numeric(left) and is_numeric(right)) or \
                    (isinstance(plain_left, T.TextType)
                     and isinstance(plain_right, T.TextType))
                if not ordered and not (isinstance(plain_left, T.AnyType)
                                        or isinstance(plain_right, T.AnyType)):
                    self.error(
                        f"`{op}` needs two numbers or two texts, found "
                        f"{plain_left.render()} and {plain_right.render()}",
                        expr.pos, code="E-binop-type")
                    return T.ERROR
            return T.SecretType(T.BOOL) if secret else T.BOOL

        # arithmetic
        if isinstance(plain_left, T.TextType) and \
                isinstance(plain_right, T.TextType) and op == "+":
            result: T.Type = T.TEXT
        elif isinstance(plain_left, T.ListType) and \
                isinstance(plain_right, T.ListType) and op == "+":
            result = T.ListType(T.unify(plain_left.elem, plain_right.elem))
        elif is_numeric(left) and is_numeric(right):
            if isinstance(plain_left, T.FloatType) or \
                    isinstance(plain_right, T.FloatType):
                result = T.F64
            elif op == "/":
                # integer division stays integer in Gama-G; `math.div` is the
                # explicit floating form
                result = T.I64
            else:
                result = T.I64
        elif isinstance(plain_left, T.AnyType) or isinstance(plain_right,
                                                            T.AnyType):
            return T.ANY
        else:
            self.error(
                f"`{op}` is not defined for {plain_left.render()} and "
                f"{plain_right.render()}", expr.pos, code="E-binop-type",
                help_text="the core does not convert between types silently")
            return T.ERROR
        return T.SecretType(result) if secret else result

    def _infer_call(self, expr: M.MCall) -> T.Type:
        key = self._builtin_key(expr)
        builtin = L.BUILTINS.get(key)
        if builtin is None:
            self.error(f"`{key}` is not an operation the standard library "
                       f"provides", expr.pos, code="E-unknown-call",
                       help_text="list what exists with "
                                 "`ggc explain builtins"
                                 + (f" --module {expr.module}`"
                                    if expr.module else "`"))
            return T.ERROR
        arg_types = [self.infer(a) for a in expr.args]
        if builtin.variadic:
            minimum = builtin.min_args if builtin.min_args is not None else 1
            if len(arg_types) < minimum:
                self.error(f"`{key}` needs at least {minimum} argument(s), "
                           f"found {len(arg_types)}", expr.pos,
                           code="E-arity")
                return T.ERROR
        elif len(arg_types) != len(builtin.params):
            self.error(f"`{key}` takes {len(builtin.params)} argument(s) "
                       f"({', '.join(builtin.params)}), found "
                       f"{len(arg_types)}", expr.pos, code="E-arity")
            return T.ERROR
        for index, want in enumerate(builtin.argtypes):
            if want is None or index >= len(arg_types):
                continue
            got = arg_types[index]
            if not isinstance(got, (T.AnyType, T.ErrorType)) and \
                    not got.assignable_to(want):
                self.error(f"argument {index + 1} of `{key}` must be "
                           f"{want.render()}, found {got.render()}",
                           expr.args[index].pos, code="E-arg-type")
        if any(isinstance(a, T.ErrorType) for a in arg_types):
            return T.ERROR
        if builtin.infer is not None:
            try:
                return builtin.infer(list(arg_types))
            except Exception:                       # noqa: BLE001
                return builtin.ret or T.ANY
        return builtin.ret or T.ANY


def check(model: M.SemanticModel, bag: DiagnosticBag) -> NativeChecker:
    """Public entry point: check the model, recording types on its nodes."""
    checker = NativeChecker(model, bag)
    checker.check()
    return checker


# ==========================================================================
# lowering: semantic model -> GIR, directly
# ==========================================================================
class Lowerer:
    """Emits GIR for the semantic model.

    Every basic block is labelled with the operation that produced it, and the
    derived graph is written into GIR's own operation-graph metadata, so the
    graph survives the descent into a register machine instead of being
    flattened away on the way down.
    """

    def __init__(self, model: M.SemanticModel, checker: NativeChecker,
                 bag: DiagnosticBag,
                 memory: Optional[MEM.MemoryModel] = None):
        self.m = model
        self.c = checker
        self.bag = bag
        self.memory = memory
        self.intent_name = model.intent.name or "Intent"
        self.prog = GProgram(name=self.intent_name,
                             grants=list(model.intent.authority))
        self.fn: Optional[GFunction] = None
        self.cur: Optional[BasicBlock] = None
        self.slots: Dict[str, int] = {}
        self.blocks_made = 0
        self.instrs_made = 0
        self.temps_made = 0
        # The recovery policy the intent declared, if any. When it exists the
        # intent lowers to a protected region around its own work rather than to
        # a bare sequence, which is what "direct recovery representation" means:
        # the policy reaches the machine as a policy.
        self.policy = REC.policy_of(model.intent)
        # node name -> the capabilities its calls actually demand, derived by the
        # graph builder from the standard library's own metadata
        self.demands: Dict[str, List[M.DerivedDemand]] = {}
        for demand in model.authority.derived:
            self.demands.setdefault(demand.node, []).append(demand)

    # ------------------------------------------------------------------
    # IR helpers
    # ------------------------------------------------------------------
    def new_block(self, label: str = "") -> BasicBlock:
        block = BasicBlock(id=f"b{self.blocks_made}", label=label)
        self.blocks_made += 1
        assert self.fn is not None
        self.fn.blocks.append(block)
        return block

    def start(self, label: str = "entry") -> BasicBlock:
        self.cur = self.new_block(label)
        assert self.fn is not None
        self.fn.entry = self.cur.id
        return self.cur

    def emit(self, op: Op, args: Sequence[Operand] = (), dst: int = -1,
             type_: T.Type = T.ANY, meta: Optional[Dict[str, Any]] = None,
             pos: Optional[SourcePos] = None) -> Instr:
        assert self.cur is not None, "emit before start()"
        instr = Instr(op=op, args=list(args), dst=dst, type=type_,
                      meta=dict(meta or {}), pos=pos, id=self.instrs_made)
        self.instrs_made += 1
        self.cur.instrs.append(instr)
        return instr

    def temp(self, type_: T.Type = T.ANY) -> int:
        assert self.fn is not None
        name = f"t{self.temps_made}"
        self.temps_made += 1
        return self.fn.new_slot(name, type_, "temp")

    def const(self, value: Any, type_: T.Type = T.ANY) -> Operand:
        return Operand(kind="const", value=value, type=type_)

    def slot_op(self, index: int) -> Operand:
        assert self.fn is not None
        type_ = self.fn.slots[index].type if 0 <= index < len(self.fn.slots) \
            else T.ANY
        return Operand(kind="slot", index=index, type=type_)

    def binding(self, name: str, type_: T.Type, *, mutable: bool = False,
                kind: str = "local", pos=None) -> int:
        """Allocate the slot a binding lives in, and remember it by name.

        When the memory model says this binding's extent does not overlap another
        binding of the same type, they share a slot. That is deterministic
        destruction showing up in the IR rather than only in a report: the release
        point was derived from the graph, so the slot is genuinely free, and no
        collector had to decide anything.
        """
        assert self.fn is not None
        if name in self.slots:
            return self.slots[name]
        if self.memory is not None:
            record = self.memory.bindings.get(name)
            if record is not None and record.shares_with:
                shared = self.slots.get(record.shares_with)
                if shared is not None:
                    self.slots[name] = shared
                    return shared
        index = self.fn.new_slot(name, type_, kind, mutable=mutable,
                                 secret=is_secret(type_))
        self.slots[name] = index
        return index

    def guard_secret(self, name: str, slot: int, pos=None) -> None:
        """Mark a secret binding at the point it comes into existence.

        Spec section 8 asks for stronger lifecycle controls on secrets, and
        section 12 for them not to reach ordinary logging. GIR has an instruction
        for the first half -- `SECRET_GUARD` -- so the marking is something the
        machine does at a known point rather than something the value happens to
        carry. Emitting it here, at every definition of a secret binding, is what
        makes that point known.
        """
        if not is_secret(self.c.env.get(name, T.ANY)):
            return
        self.emit(Op.SECRET_GUARD, [self.slot_op(slot)], dst=-1,
                  meta={"binding": name}, pos=pos)
        assert self.fn is not None
        self.fn.secret_points += 1

    def capability_checks(self, node: M.OpNode) -> None:
        """Emit a runtime check for every capability this node's calls demand.

        The compile-time check already proved these are covered by the intent's
        authority. The runtime check is emitted anyway, for the same reason a
        proven-exhaustive selection still keeps its `NoActiveAlternative` fault:
        a proof is a reason to trust the program, not a reason to remove the
        boundary. A capability that is only checked at compile time is not
        enforced against a hand-edited IR or a future backend that reorders it.
        """
        for demand in self.demands.get(node.name, ()):
            for name in demand.alternatives:
                self.emit(Op.CAP_CHECK, [], dst=-1,
                          meta={"capability": name,
                                "what": f"{node.name}: {demand.origin}"},
                          pos=node.pos)
                assert self.fn is not None
                self.fn.cap_points += 1

    def goto(self, block: BasicBlock) -> None:
        self.emit(Op.JUMP, meta={"target": block.id})
        self.cur = block

    def fault(self, kind: str, message: str, pos=None) -> None:
        self.emit(Op.FAULT, [], dst=-1,
                  meta={"kind": kind, "message": message}, pos=pos)

    # ------------------------------------------------------------------
    def lower(self) -> GProgram:
        if self.policy.steps:
            # The work moves into its own function so that the intent can wrap it
            # in a protected region. The runtime calls the protected body with no
            # knowledge of what it does, which is the point: the policy is the
            # intent's, and the body is just the work.
            work = f"{self.intent_name}.work"
            self._intent_function(work, kind="intent")
            self._protected_function(work)
        else:
            self._intent_function()
        self._main_function()
        return self.prog

    def _protected_function(self, work_name: str) -> None:
        """An intent with a declared `recover` policy, as a protected region.

        Spec section 10 makes recovery a language capability and section 18 makes
        a failure before commit a recovery state rather than a half-applied
        change. GIR already has the instruction for both, and the runtime already
        implements the six levels -- so the policy the program declared is handed
        to the machine as a policy, not re-encoded as a fault message and a jump.

        Two things this makes true that a bare `FAULT` could not:

        * a checkpoint is captured *before* the work runs, so a `restore` step
          returns to recorded state instead of inventing some;
        * every recovery action taken is audited, because the region is entered
          with `audit_all`. That is section 10's "every recovery action should be
          observable and auditable", discharged by the lowering rather than left
          to the program to remember.
        """
        graph = self.m.operations
        effects = self._effects()
        self.fn = GFunction(
            name=self.intent_name, kind="service",
            ret=self._outcome_type(), effects=effects,
            caps=tuple(self.m.intent.authority),
            deterministic=not ({"io", "network", "unsafe"} & set(effects)))
        self.fn.recovery = RecoveryPlan(steps=self.policy.to_steps(),
                                       audit_all=True,
                                       level_names=dict(REC.LEVEL_NAMES))
        self.fn.recovery_points += 1
        self.slots = {}
        self.blocks_made = self.instrs_made = self.temps_made = 0
        self.start("protected")

        args: List[Operand] = []
        for name, node in graph.inputs.items():
            type_ = self.c.env.get(name, T.ANY)
            slot = self.binding(name, type_, kind="param")
            assert self.fn is not None
            self.fn.params.append(slot)
            self.fn.param_names.append(name)
            if is_secret(type_):
                self.fn.secret_params.append(slot)
                self.guard_secret(name, slot, node.pos)
            args.append(self.slot_op(slot))

        outcome = self.temp(T.ANY)
        self.emit(Op.PROTECTED, args, dst=outcome,
                  meta={"protect": work_name,
                        "steps": self.policy.to_steps(),
                        "checkpoints": list(self.policy.checkpoints),
                        "audit_all": True,
                        "component": self.intent_name},
                  pos=self.m.intent.pos)
        # The region's result is a RecoveryOutcome record; the intent's contract
        # is its outcome, so the value is taken back out of it. Whether recovery
        # was needed is visible in the audit trail, not in the return type --
        # changing the type on the failure path would make a caller handle two
        # shapes for one intent.
        value = self.temp(self._outcome_type())
        self.emit(Op.FIELD, [self.slot_op(outcome)], dst=value,
                  type_=self._outcome_type(), meta={"name": "value"},
                  pos=self.m.intent.pos)
        self.emit(Op.RETURN, [self.slot_op(value)],
                  type_=self._outcome_type(), pos=self.m.intent.pos)
        self.prog.add(self.fn)

    # ------------------------------------------------------------------
    def _effects(self) -> Tuple[str, ...]:
        found: List[str] = []
        for node in self.m.operations.nodes.values():
            for effect in node.effects:
                if effect and effect not in found:
                    found.append(effect)
            if node.trail and "audit" not in found:
                found.append("audit")
            for used in self.c.node_effects.get(node.name, ()):
                if used not in found:
                    found.append(used)
        for src in self.m.operations.inputs.values():
            if src.origin is not None:
                for used in self.c._effects_of(src.origin):
                    if used not in found:
                        found.append(used)
        if not found:
            return ("pure",)
        if "pure" in found and len(found) > 1:
            found.remove("pure")
        return tuple(found)

    def _outcome_type(self) -> T.Type:
        return self.c.env.get(self.m.operations.outcome, T.UNIT)

    # ------------------------------------------------------------------
    def _intent_function(self, name: Optional[str] = None,
                         kind: str = "intent") -> None:
        graph = self.m.operations
        effects = self._effects()
        self.fn = GFunction(
            name=name or self.intent_name, kind=kind,
            ret=self._outcome_type(), effects=effects,
            caps=tuple(self.m.intent.authority),
            deterministic=not ({"io", "network", "unsafe"} & set(effects)))
        self.slots = {}
        self.blocks_made = self.instrs_made = self.temps_made = 0
        self.start("intent")

        # inputs become parameters: an intent is callable with its sources
        for name, node in graph.inputs.items():
            type_ = self.c.env.get(name, T.ANY)
            slot = self.binding(name, type_, kind="param")
            assert self.fn is not None
            self.fn.params.append(slot)
            self.fn.param_names.append(name)
            if is_secret(type_):
                self.fn.secret_params.append(slot)
                # marked at the boundary it enters through, not at the first
                # place something happens to notice it
                self.guard_secret(name, slot, node.pos)

        self._resources()
        self._checkpoints()
        self._compute_phase()
        self._commit_phase()
        self._outcome()
        self._record_graph()
        self._finish()

    def _finish(self) -> None:
        assert self.fn is not None
        # A block with no terminator would be read as `return ()`, so every
        # path is closed explicitly rather than relying on fall-through.
        if self.cur is not None and self.cur.terminator is None:
            self.emit(Op.RETURN, [self.const(("unit",), T.UNIT)])
        self.prog.add(self.fn)

    def _resources(self) -> None:
        for name, state in self.m.operations.resources.items():
            type_ = self.c.env.get(name, T.ANY)
            slot = self.binding(name, type_, mutable=True)
            if state.initial is not None:
                value = self.expr(state.initial)
                self.emit(Op.COPY, [value], dst=slot, type_=type_,
                          pos=state.pos)
            self.guard_secret(name, slot, state.pos)

    def _checkpoints(self) -> None:
        """Record the checkpoints the intent declared, before any work runs.

        A `recover` policy may name a checkpoint only if one was declared, and a
        declaration is worth nothing unless the state is actually captured -- so
        the capture happens here, with the resources at their initial values.
        That is what makes `restore checkpoint admission` return to admission
        rather than to whatever happened to be recorded most recently.
        """
        for label in self.m.intent.checkpoints:
            self.emit(Op.CHECKPOINT, [], dst=-1,
                      meta={"mode": "at", "raw": label},
                      pos=self.m.intent.pos)
            assert self.fn is not None
            self.fn.recovery_points += 1

    def _compute_phase(self) -> None:
        graph = self.m.operations
        emitted: Set[str] = set()
        for name in graph.compute_order():
            node = graph.nodes[name]
            selection = graph.selections.get(node.produces)
            if selection is not None:
                if name != selection.members[-1] or node.produces in emitted:
                    continue
                emitted.add(node.produces)
                self._selection(selection)
                continue
            self._node(node)

    def _commit_phase(self) -> None:
        order = self.m.operations.commit_order()
        if not order:
            return
        name = self.m.intent.name or self.intent_name
        # Spec section 18: either the transitions take effect or they do not, and
        # a failure before the commit leaves a recovery state rather than a half
        # applied change. The core's two-phase split is already that shape --
        # every derivation first, every state change last -- so it is emitted as
        # a transaction instead of as a run of copies that happen to come at the
        # end. If a `holds` constraint fails in here, the runtime aborts the
        # transaction it was executing; see `Context.transaction_owner`.
        self.emit(Op.TRANSACTION, [], dst=-1,
                  meta={"action": "begin", "name": name},
                  pos=self.m.intent.pos)
        for node_name in order:
            self._node(self.m.operations.nodes[node_name])
        self.emit(Op.TRANSACTION, [], dst=-1,
                  meta={"action": "commit", "name": name},
                  pos=self.m.intent.pos)

    def _outcome(self) -> None:
        outcome = self.m.operations.outcome
        slot = self.slots.get(outcome)
        operand = self.slot_op(slot) if slot is not None else \
            self.const(("unit",), T.UNIT)
        self.emit(Op.RETURN, [operand], type_=self._outcome_type(),
                  pos=self.m.intent.pos)

    def _record_graph(self) -> None:
        """Write the derived graph into GIR's own operation-graph metadata.

        This is the part the elaboration path could not do: GIR has a place to
        say "these nodes are independent and these depend on those", and the
        derived levels are exactly that information.  A backend can read it
        without re-deriving anything.
        """
        assert self.fn is not None
        for name in self.m.operations.order():
            node = self.m.operations.nodes[name]
            reads = list(dict.fromkeys(
                list(node.consumes)
                + sorted(M.references(getattr(node, "value", None)))
                + sorted(M.references(getattr(node, "collection", None))
                         or [])
                + sorted(M.references(getattr(node, "subject", None)) or [])))
            self.fn.parallel_tasks.append(ParallelTask(
                name=name, function=self.intent_name, reads=reads,
                writes=[node.produces] if node.kind != "transition"
                else [node.state],           # type: ignore[attr-defined]
                depends_on=list(node.upstream)))
        self.fn.recovery_points = len(self.m.recovery.obligations)

    # ------------------------------------------------------------------
    # nodes
    # ------------------------------------------------------------------
    def _node(self, node: M.OpNode) -> None:
        label = f"{node.kind}:{node.name}"
        self.capability_checks(node)
        if node.kind == "refine":
            self._refine(node, label)       # type: ignore[arg-type]
        elif node.kind == "fanout":
            self._fanout(node, label)       # type: ignore[arg-type]
        elif node.kind == "dispatch":
            self._dispatch(node, label)     # type: ignore[arg-type]
        elif node.kind == "transition":
            self._transition(node, label)   # type: ignore[arg-type]
        else:
            self._compute(node, label)
        self._constraints(node)
        self._trail(node)
        self._guard_node(node)

    def _guard_node(self, node: M.OpNode) -> None:
        """Mark the binding a node produced, once the node is done with it.

        The guard goes *after* the node's own constraints and trail rather than
        before: within the node the value is still being computed and checked,
        and the marking is what protects it on its way to everything else.
        """
        name = node.produces or getattr(node, "state", "")
        slot = self.slots.get(name)
        if slot is not None:
            self.guard_secret(name, slot, node.pos)

    def _slot_for(self, node: M.OpNode, mutable: bool = False) -> int:
        type_ = self.c.env.get(node.produces, T.ANY)
        return self.binding(node.produces, type_, mutable=mutable,
                            pos=node.pos)

    def _compute(self, node: M.ComputeNode, label: str) -> None:
        slot = self._slot_for(node)
        if node.value is None:
            return
        value = self.expr(node.value)
        self.emit(Op.COPY, [value], dst=slot,
                  type_=self.c.env.get(node.produces, T.ANY), pos=node.pos)

    def _constraints(self, node: M.OpNode) -> None:
        for hold in node.holds:
            if hold.expr is None:
                continue
            condition = self.expr(hold.expr)
            self.emit(Op.REQUIRE, [condition], dst=-1,
                      meta={"message": f"holds `{hold.text}`"}, pos=hold.pos)

    def _trail(self, node: M.OpNode) -> None:
        """Record what the intent said it would record.

        A secret binding's *value* is never recorded: the audit trail says that
        the operation ran and what it was for, not what it computed. That is a
        property of the lowering, so it cannot be forgotten by a program.
        """
        if not node.trail:
            return
        keys = ["action", "intent", "record"]
        args: List[Operand] = [
            self.const(node.name, T.TEXT),
            self.const(self.m.intent.name, T.TEXT),
            self.const(node.trail, T.TEXT),
        ]
        type_ = self.c.env.get(node.produces)
        if node.produces and node.kind != "transition" and type_ is not None \
                and not is_secret(type_):
            keys.append("value")
            args.append(self.slot_op(self.slots[node.produces]))
        self.emit(Op.AUDIT, args, dst=-1, meta={"keys": keys}, pos=node.pos)
        assert self.fn is not None
        self.fn.audit_points += 1

    # ------------------------------------------------------------------
    def _selection(self, selection: M.Selection) -> None:
        """Guarded alternatives over one binding.

        Lowered as a chain of guarded blocks.  The fallback is `Op.FAULT` with
        the core's own kind -- a GIR terminator, not a call to a panic function,
        and not a synthesised arm of anyone else's match.  It stays even when
        the guards were proven complementary: a proof is a reason to trust the
        program, not a reason to remove the check.
        """
        graph = self.m.operations
        first = graph.nodes[selection.members[0]]
        slot = self._slot_for(first, mutable=True)
        join = self.new_block(f"select.{selection.binding}.join")
        for name in selection.members:
            node = graph.nodes[name]
            guard = self.expr(node.when.expr) if node.when and node.when.expr \
                else self.const(False, T.BOOL)
            taken = self.new_block(f"select.{name}")
            onward = self.new_block(f"select.{name}.next")
            self.emit(Op.JUMP_IF, [guard],
                      meta={"target": taken.id, "target_false": onward.id},
                      pos=node.pos)
            self.cur = taken
            value = self.expr(node.value) if getattr(node, "value", None) \
                is not None else self.const(("unit",), T.UNIT)
            self.emit(Op.COPY, [value], dst=slot,
                      type_=self.c.env.get(node.produces, T.ANY), pos=node.pos)
            self._constraints(node)
            self._trail(node)
            self.goto(join)
            self.cur = onward
        proof = " (the guards were proven complementary)" if \
            selection.proven_exhaustive else ""
        self.fault(
            FAULT_NO_ALTERNATIVE,
            f"nothing yields `{selection.binding}` -- "
            f"{len(selection.members)} guarded alternatives, none active"
            + proof, first.pos)
        self.cur = join
        for name in selection.members:
            self.capability_checks(graph.nodes[name])
        self.guard_secret(selection.binding, slot, first.pos)

    def _refine(self, node: M.RefinementNode, label: str) -> None:
        """Bounded recurrence.

        The bound is a literal in the IR and exceeding it is `Op.FAULT` with
        kind `RefinementDiverged`.  Nothing here can spin without a counter:
        the counter is part of the shape this method emits.
        """
        type_ = self.c.env.get(node.produces, T.ANY)
        slot = self.binding(node.produces, type_, mutable=True, pos=node.pos)
        if node.start is not None:
            self.emit(Op.COPY, [self.expr(node.start)], dst=slot,
                      type_=type_, pos=node.pos)
        rounds = self.temp(T.I64)
        self.emit(Op.CONST, [self.const(0, T.I64)], dst=rounds, type_=T.I64,
                  pos=node.pos)

        condition = self.new_block(f"refine.{node.name}.until")
        body = self.new_block(f"refine.{node.name}.round")
        diverged = self.new_block(f"refine.{node.name}.diverged")
        advance = self.new_block(f"refine.{node.name}.next")
        done = self.new_block(f"refine.{node.name}.done")

        self.goto(condition)
        stop = self.expr(node.stop.expr) if node.stop and node.stop.expr \
            else self.const(True, T.BOOL)
        self.emit(Op.JUMP_IF, [stop],
                  meta={"target": done.id, "target_false": body.id},
                  pos=node.pos)

        self.cur = body
        over = self.temp(T.BOOL)
        bound = int(node.bound) if node.bound is not None else 1
        self.emit(Op.BINOP,
                  [self.slot_op(rounds), self.const(bound, T.I64)],
                  dst=over, type_=T.BOOL, meta={"operator": ">="},
                  pos=node.pos)
        self.emit(Op.JUMP_IF, [self.slot_op(over)],
                  meta={"target": diverged.id, "target_false": advance.id},
                  pos=node.pos)

        self.cur = diverged
        stopping = node.stop.text if node.stop and node.stop.text else \
            "its stopping constraint"
        self.fault(FAULT_DIVERGED,
                   f"`{node.name}` did not satisfy `{stopping}` within "
                   f"{node.bound} rounds", node.pos)

        self.cur = advance
        if node.step is not None:
            self.emit(Op.COPY, [self.expr(node.step)], dst=slot, type_=type_,
                      pos=node.pos)
        incremented = self.temp(T.I64)
        self.emit(Op.BINOP, [self.slot_op(rounds), self.const(1, T.I64)],
                  dst=incremented, type_=T.I64, meta={"operator": "+"},
                  pos=node.pos)
        self.emit(Op.COPY, [self.slot_op(incremented)], dst=rounds,
                  type_=T.I64, pos=node.pos)
        self._constraints(node)
        self.goto(condition)
        self.cur = done

    def _fanout(self, node: M.FanOutNode, label: str) -> None:
        declared = self.c.env.get(node.produces, T.ListType(T.ANY))
        slot = self.binding(node.produces, declared, mutable=True,
                            pos=node.pos)
        self.emit(Op.MAKE_LIST, [], dst=slot, type_=declared, pos=node.pos)
        if node.collection is None:
            return
        collection = self.expr(node.collection)
        iterator = self.temp(T.ListType(T.ANY))
        self.emit(Op.BUILTIN, [collection], dst=iterator,
                  meta={"name": "__iter_list"}, pos=node.pos)
        index = self.temp(T.I64)
        self.emit(Op.CONST, [self.const(0, T.I64)], dst=index, type_=T.I64,
                  pos=node.pos)
        count = self.temp(T.I64)
        self.emit(Op.BUILTIN, [self.slot_op(iterator)], dst=count,
                  meta={"name": "len"}, pos=node.pos)
        item_slot = self.binding(node.item, element_type(
            self.c.infer(node.collection)), kind="local", pos=node.pos)

        condition = self.new_block(f"each.{node.name}.more")
        body = self.new_block(f"each.{node.name}.item")
        done = self.new_block(f"each.{node.name}.done")
        self.goto(condition)
        more = self.temp(T.BOOL)
        self.emit(Op.BINOP, [self.slot_op(index), self.slot_op(count)],
                  dst=more, type_=T.BOOL, meta={"operator": "<"}, pos=node.pos)
        self.emit(Op.JUMP_IF, [self.slot_op(more)],
                  meta={"target": body.id, "target_false": done.id},
                  pos=node.pos)

        self.cur = body
        self.emit(Op.INDEX, [self.slot_op(iterator), self.slot_op(index)],
                  dst=item_slot, pos=node.pos)
        if node.when is not None and node.when.expr is not None:
            keep = self.new_block(f"each.{node.name}.keep")
            skip = self.new_block(f"each.{node.name}.skip")
            guard = self.expr(node.when.expr)
            self.emit(Op.JUMP_IF, [guard],
                      meta={"target": keep.id, "target_false": skip.id},
                      pos=node.pos)
            self.cur = keep
            self._append(node, slot)
            self.goto(skip)
            self.cur = skip
        else:
            self._append(node, slot)
        stepped = self.temp(T.I64)
        self.emit(Op.BINOP, [self.slot_op(index), self.const(1, T.I64)],
                  dst=stepped, type_=T.I64, meta={"operator": "+"},
                  pos=node.pos)
        self.emit(Op.COPY, [self.slot_op(stepped)], dst=index, type_=T.I64,
                  pos=node.pos)
        self.goto(condition)
        self.cur = done
        self.slots.pop(node.item, None)

    def _append(self, node: M.FanOutNode, slot: int) -> None:
        element = self.expr(node.value) if node.value is not None else \
            self.const(("unit",), T.UNIT)
        single = self.temp(T.ListType(T.ANY))
        self.emit(Op.MAKE_LIST, [element], dst=single,
                  type_=T.ListType(T.ANY), pos=node.pos)
        joined = self.temp(unwrap(self.c.env.get(node.produces, T.ANY)))
        self.emit(Op.BINOP, [self.slot_op(slot), self.slot_op(single)],
                  dst=joined, meta={"operator": "+"}, pos=node.pos)
        self.emit(Op.COPY, [self.slot_op(joined)], dst=slot, pos=node.pos)

    def _dispatch(self, node: M.DispatchNode, label: str) -> None:
        type_ = self.c.env.get(node.produces, T.ANY)
        slot = self.binding(node.produces, type_, mutable=True, pos=node.pos)
        if node.subject is None:
            return
        subject = self.expr(node.subject)
        holder = self.temp(self.c.infer(node.subject))
        self.emit(Op.COPY, [subject], dst=holder, pos=node.pos)
        join = self.new_block(f"resolve.{node.name}.join")
        for pattern, value in node.cases:
            if isinstance(pattern, (M.PWild, M.PBind)):
                if isinstance(pattern, M.PBind) and pattern.binding:
                    self.slots[pattern.binding] = holder
                taken = self.new_block(f"resolve.{node.name}.default")
                self.goto(taken)
                self.emit(Op.COPY, [self.expr(value)], dst=slot, type_=type_,
                          pos=pattern.pos)
                self.goto(join)
                self.cur = self.new_block(f"resolve.{node.name}.unreachable")
                continue
            if isinstance(pattern, M.PTag):
                self.bag.error(
                    f"the native backend cannot yet dispatch on a constructed "
                    f"alternative `{pattern.text}`", pattern.pos,
                    phase=Phase.GIR, code="E-dispatch-unsupported",
                    help_text="dispatch on literals and a `_` catch-all is "
                              "supported; constructed alternatives are on the "
                              "roadmap")
                continue
            literal = M.MLit(pos=pattern.pos, value=pattern.value,
                             kind=pattern.kind)
            matches = self.temp(T.BOOL)
            self.emit(Op.BINOP,
                      [self.slot_op(holder), self.expr(literal)],
                      dst=matches, type_=T.BOOL, meta={"operator": "=="},
                      pos=pattern.pos)
            taken = self.new_block(f"resolve.{node.name}.{pattern.text}")
            onward = self.new_block(f"resolve.{node.name}.next")
            self.emit(Op.JUMP_IF, [self.slot_op(matches)],
                      meta={"target": taken.id, "target_false": onward.id},
                      pos=pattern.pos)
            self.cur = taken
            self.emit(Op.COPY, [self.expr(value)], dst=slot, type_=type_,
                      pos=pattern.pos)
            self.goto(join)
            self.cur = onward
        self.emit(Op.MATCH_FAIL, [self.slot_op(holder)], dst=-1,
                  pos=node.pos)
        self.cur = join

    def _transition(self, node: M.TransitionNode, label: str) -> None:
        state = node.state
        slot = self.slots.get(state)
        if slot is None or node.value is None:
            return
        value = self.expr(node.value)
        self.emit(Op.COPY, [value], dst=slot,
                  type_=self.c.env.get(state, T.ANY), pos=node.pos)

    # ------------------------------------------------------------------
    def _main_function(self) -> None:
        """A runnable entry point, when every source says where it comes from."""
        graph = self.m.operations
        if any(node.origin is None for node in graph.inputs.values()):
            return
        saved_fn, saved_cur = self.fn, self.cur
        saved_slots = self.slots
        saved_counters = (self.blocks_made, self.instrs_made, self.temps_made)
        effects = [e for e in self._effects() if e != "pure"]
        if "io" not in effects:
            effects.append("io")
        self.fn = GFunction(name="main", kind="fn", ret=T.UNIT,
                            effects=tuple(effects),
                            deterministic=self.fn.deterministic
                            if saved_fn else True)
        self.slots = {}
        self.blocks_made = self.instrs_made = self.temps_made = 0
        self.start("main")

        args: List[Operand] = []
        for name, node in graph.inputs.items():
            type_ = self.c.env.get(name, T.ANY)
            slot = self.binding(name, type_)
            self.emit(Op.COPY, [self.expr(node.origin)], dst=slot,
                      type_=type_, pos=node.pos)
            args.append(self.slot_op(slot))
        result = self.temp(self._outcome_type())
        self.emit(Op.CALL, args, dst=result, type_=self._outcome_type(),
                  meta={"callee": self.intent_name}, pos=self.m.intent.pos)
        shown = self.temp(T.UNIT)
        if is_secret(self._outcome_type()):
            # Spec section 12: secrets cannot be printed through ordinary
            # logging. The generated entry point therefore says that there is a
            # result and what it is called, not what it holds. Reaching `print`
            # with the value would demand `SecretExpose`, and a program that
            # wants that should ask for it in its own body rather than have the
            # compiler do it silently on the way out.
            self.emit(Op.BUILTIN,
                      [self.const(f"[secret {graph.outcome or 'outcome'}]",
                                  T.TEXT)],
                      dst=shown, type_=T.UNIT, meta={"name": "print"},
                      pos=self.m.intent.pos)
        else:
            self.emit(Op.BUILTIN, [self.slot_op(result)], dst=shown,
                      type_=T.UNIT, meta={"name": "print"},
                      pos=self.m.intent.pos)
        self._finish()

        self.fn, self.cur = saved_fn, saved_cur
        self.slots = saved_slots
        (self.blocks_made, self.instrs_made, self.temps_made) = saved_counters

    # ------------------------------------------------------------------
    def expr(self, node: Optional[M.MExpr]) -> Operand:
        if node is None:
            return self.const(("unit",), T.UNIT)
        if isinstance(node, M.MLit):
            if node.kind == "unit":
                return self.const(("unit",), T.UNIT)
            return self.const(node.value, self.c.infer(node))
        if isinstance(node, M.MRef):
            slot = self.slots.get(node.binding)
            if slot is None:
                self.bag.error(f"`{node.binding}` has no slot at this point in "
                               f"the graph", node.pos, phase=Phase.GIR,
                               code="E-ice")
                return self.const(None, T.ANY)
            return self.slot_op(slot)
        if isinstance(node, M.MItems):
            items = [self.expr(i) for i in node.items]
            dst = self.temp(self.c.infer(node))
            self.emit(Op.MAKE_LIST, items, dst=dst, type_=node.type,
                      pos=node.pos)
            return self.slot_op(dst)
        if isinstance(node, M.MBin):
            left = self.expr(node.left)
            right = self.expr(node.right)
            dst = self.temp(self.c.infer(node))
            self.emit(Op.BINOP, [left, right], dst=dst, type_=node.type,
                      meta={"operator": node.op}, pos=node.pos)
            return self.slot_op(dst)
        if isinstance(node, M.MUn):
            operand = self.expr(node.operand)
            dst = self.temp(self.c.infer(node))
            self.emit(Op.UNOP, [operand], dst=dst, type_=node.type,
                      meta={"operator": node.op}, pos=node.pos)
            return self.slot_op(dst)
        if isinstance(node, M.MCall):
            args = [self.expr(a) for a in node.args]
            key = self.c._builtin_key(node)
            dst = self.temp(self.c.infer(node))
            self.emit(Op.BUILTIN, args, dst=dst, type_=node.type,
                      meta={"name": key}, pos=node.pos)
            return self.slot_op(dst)
        if isinstance(node, M.MIndex):
            obj = self.expr(node.obj)
            index = self.expr(node.index)
            dst = self.temp(self.c.infer(node))
            self.emit(Op.INDEX, [obj, index], dst=dst, type_=node.type,
                      pos=node.pos)
            return self.slot_op(dst)
        if isinstance(node, M.MField):
            obj = self.expr(node.obj)
            dst = self.temp(T.ANY)
            self.emit(Op.FIELD, [obj], dst=dst, meta={"name": node.attr},
                      pos=node.pos)
            return self.slot_op(dst)
        return self.const(None, T.ANY)


def lower(model: M.SemanticModel, checker: NativeChecker,
          bag: DiagnosticBag,
          memory: Optional[MEM.MemoryModel] = None) -> GProgram:
    """Public entry point: semantic model -> GIR.

    ``memory`` is the model from :mod:`gamag.core.memory`. It is optional so that
    lowering still works without it, but when it is present the lowerer uses its
    extents to reuse slots, which is the part of the memory model that changes
    the emitted code rather than only describing it.
    """
    return Lowerer(model, checker, bag, memory=memory).lower()
