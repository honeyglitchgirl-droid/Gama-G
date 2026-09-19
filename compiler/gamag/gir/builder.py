"""Lowering from the typed AST to Gama IR (spec section 22, step 9).

Every language construct becomes explicit GIR: control flow becomes basic
blocks and jumps, ``match`` becomes a decision chain, contracts become
entry/epilogue checks, audit records become audit boundaries, and ``parallel``
regions become a real operation graph -- the builder performs read/write
dependence analysis over the region's statements and emits one task function
per statement together with the edges between them (spec sections 3 and 9C).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .. import ast_nodes as A
from ..diagnostics import GamaError, Diagnostic, Phase, Severity, SourcePos
from ..semantic import types as T
from ..semantic.checker import Checker, Symbol
from ..std import library as L
from .ir import (GFunction, GProgram, Instr, BasicBlock, Operand, Op,
                 ParallelTask, RecoveryPlan, Slot)

CONTROL_NAMES = {"ok", "fail", "some", "none", "true", "false"}


def _walk(node: Any):
    """Yield every AST node beneath ``node``."""
    if isinstance(node, A.Node):
        yield node
    for field_name in getattr(node, "__dataclass_fields__", {}):
        value = getattr(node, field_name, None)
        if isinstance(value, A.Node):
            yield from _walk(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, A.Node):
                    yield from _walk(item)
                elif isinstance(item, tuple):
                    for sub in item:
                        if isinstance(sub, A.Node):
                            yield from _walk(sub)


def names_read(node: Any) -> Set[str]:
    return {n.id for n in _walk(node) if isinstance(n, A.Name)}


def names_written(stmt: A.Stmt) -> Set[str]:
    out: Set[str] = set()
    for n in _walk(stmt):
        if isinstance(n, A.Assign) and isinstance(n.target, A.Name):
            out.add(n.target.id)
        elif isinstance(n, A.LetDecl):
            out.add(n.name)
        elif isinstance(n, A.For):
            out.add(n.var)
        elif isinstance(n, A.ForAll):
            out.add(n.var)
        elif isinstance(n, A.NamePat):
            out.add(n.name)
    return out


def names_declared(stmt: A.Stmt) -> Set[str]:
    out: Set[str] = set()
    for n in _walk(stmt):
        if isinstance(n, (A.LetDecl, A.For, A.ForAll, A.NamePat, A.IODirective)):
            out.add(n.name if hasattr(n, "name") else n.var)
    return out


class Const:
    """Encoded constant values understood by the VM."""

    @staticmethod
    def unit():
        return ("unit",)

    @staticmethod
    def duration(seconds: float, text: str = ""):
        return ("duration", seconds, text)

    @staticmethod
    def default(type_: Any):
        return ("default", type_)

    @staticmethod
    def variant(tag: str, enum: str):
        return ("variant", tag, enum)


class FunctionBuilder:
    """Builds one GIR function."""

    def __init__(self, program: GProgram, checker: Checker, name: str,
                 kind: str = "fn", ret: T.Type = T.UNIT,
                 effects: Sequence[str] = (), caps: Sequence[str] = (),
                 deterministic: bool = False,
                 contracts: Optional[List[Dict[str, Any]]] = None,
                 global_names: Optional[Set[str]] = None,
                 pos: Optional[SourcePos] = None):
        self.prog = program
        self.checker = checker
        self.fn = GFunction(name=name, kind=kind, ret=ret,
                            effects=tuple(effects), caps=tuple(caps),
                            deterministic=deterministic,
                            contracts=list(contracts or []))
        self.global_names: Set[str] = set(global_names or ())
        self.scopes: List[Dict[str, int]] = [{}]
        self.blocks_made = 0
        self.instrs_made = 0
        self.temps_made = 0
        self.cur: Optional[BasicBlock] = None
        self.pos = pos
        self.loop_depth = 0
        self.result_slot: Optional[int] = None
        self.epilogue: Optional[BasicBlock] = None
        self.needs_epilogue = any(c.get("kind") == "ensures"
                                  for c in self.fn.contracts)
        self.policy_ctx_slot: Optional[int] = None
        # How many parallel regions this function has lowered so far.
        self.parallel_regions = 0
        self.declared: Set[str] = set()
        self.loop_breaks: List[str] = []
        self.loop_continues: List[str] = []

    # ------------------------------------------------------------------
    # IR construction helpers
    # ------------------------------------------------------------------
    def new_block(self, label: str = "") -> BasicBlock:
        bid = f"b{self.blocks_made}"
        self.blocks_made += 1
        block = BasicBlock(id=bid, label=label)
        self.fn.blocks.append(block)
        return block

    def start(self) -> BasicBlock:
        self.cur = self.new_block("entry")
        self.fn.entry = self.cur.id
        return self.cur

    def set_block(self, block: BasicBlock) -> None:
        self.cur = block

    def emit(self, op: Op, args: Sequence[Operand] = (), dst: int = -1,
             type_: T.Type = T.ANY, meta: Optional[Dict[str, Any]] = None,
             pos: Optional[SourcePos] = None) -> Instr:
        assert self.cur is not None, "emit before start()"
        instr = Instr(op=op, args=list(args), dst=dst, type=type_,
                      meta=dict(meta or {}), pos=pos or self.pos,
                      id=self.instrs_made)
        self.instrs_made += 1
        self.cur.instrs.append(instr)
        return instr

    def temp(self, type_: T.Type = T.ANY) -> int:
        name = f"t{self.temps_made}"
        self.temps_made += 1
        return self.fn.new_slot(name, type_, "temp")

    def push_scope(self) -> None:
        self.scopes.append({})

    def pop_scope(self) -> None:
        if len(self.scopes) > 1:
            self.scopes.pop()

    def bind(self, name: str, slot: int) -> None:
        self.scopes[-1][name] = slot
        self.declared.add(name)

    def lookup(self, name: str) -> Optional[int]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def const(self, value: Any, type_: T.Type = T.ANY) -> Operand:
        return Operand(kind="const", value=value, type=type_)

    def slot_op(self, index: int, type_: T.Type = T.ANY) -> Operand:
        if 0 <= index < len(self.fn.slots):
            type_ = self.fn.slots[index].type
        return Operand(kind="slot", index=index, type=type_)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def expr(self, node: Optional[A.Expr]) -> Operand:
        if node is None:
            return self.const(Const.unit(), T.UNIT)
        method = getattr(self, f"e_{type(node).__name__}", None)
        if method is None:
            raise GamaError(Diagnostic(
                Severity.ERROR, Phase.GIR,
                f"cannot lower expression `{type(node).__name__}` to GIR",
                pos=node.pos))
        return method(node)

    def e_Literal(self, e: A.Literal) -> Operand:
        ty = e.inferred or T.ANY
        if e.lit_kind == "duration":
            return self.const(Const.duration(e.value["seconds"], e.value["unit"]),
                              T.DURATION)
        if e.lit_kind == "unit":
            return self.const(Const.unit(), T.UNIT)
        return self.const(e.value, ty)

    def e_RangeExpr(self, e: A.RangeExpr) -> Operand:
        lo = self.expr(e.start)
        hi = self.expr(e.end)
        ty = e.inferred or T.ListType(T.I64)
        dst = self.temp(ty)
        self.emit(Op.BUILTIN, [lo, hi, self.const(e.inclusive, T.BOOL)],
                  dst=dst, type_=ty, meta={"name": "__range"}, pos=e.pos)
        return self.slot_op(dst)

    def e_Name(self, e: A.Name) -> Operand:
        slot = self.lookup(e.id)
        if slot is not None:
            return self.slot_op(slot)
        if e.id in self.global_names or e.id in self.checker.functions:
            return Operand(kind="global", name=e.id,
                           type=(e.inferred or T.ANY))
        sym = self.checker.globals.lookup(e.id)
        if sym is not None and sym.kind == "variant":
            enum_name = self.checker.variants.get(e.id, ("", None))[0]
            return self.const(Const.variant(e.id, enum_name), sym.type)
        if sym is not None and sym.kind == "const":
            constant = L.CONSTANTS.get(f"math.{e.id}") or L.CONSTANTS.get(e.id)
            if constant is not None:
                return self.const(constant.value, constant.type)
        if sym is not None and sym.kind == "module":
            return Operand(kind="global", name=e.id, type=sym.type)
        if self.policy_ctx_slot is not None:
            # Inside a policy, free names are read from the decision context.
            dst = self.temp(T.ANY)
            self.emit(Op.BUILTIN,
                      [self.slot_op(self.policy_ctx_slot),
                       self.const(e.id, T.TEXT)], dst=dst,
                      type_=T.ANY, meta={"name": "__ctx_get",
                                        "source": "policy-context"},
                      pos=e.pos)
            return self.slot_op(dst)
        return Operand(kind="global", name=e.id, type=(e.inferred or T.ANY))

    def e_Binary(self, e: A.Binary) -> Operand:
        if e.op in ("and", "or"):
            return self.e_short_circuit(e)
        left = self.expr(e.left)
        right = self.expr(e.right)
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.BINOP, [left, right], dst=dst,
                  type_=(e.inferred or T.ANY),
                  meta={"operator": e.op}, pos=e.pos)
        return self.slot_op(dst)

    def e_short_circuit(self, e: A.Binary) -> Operand:
        """`and`/`or` must not evaluate their right operand unnecessarily."""
        dst = self.temp(T.BOOL)
        b_right = self.new_block(f"{e.op}.rhs")
        b_end = self.new_block(f"{e.op}.end")
        left = self.expr(e.left)
        self.emit(Op.COPY, [left], dst=dst, type_=T.BOOL, pos=e.pos)
        # `and` short-circuits when the left side is *false* (the result is
        # already false, so the right side need not run); `or` short-circuits
        # when the left side is *true*.  Whichever side does run, its value
        # becomes the result.
        if e.op == "and":
            on_true, on_false = b_right.id, b_end.id
        else:
            on_true, on_false = b_end.id, b_right.id
        self.emit(Op.JUMP_IF, [self.slot_op(dst)],
                  meta={"target": on_true, "target_false": on_false},
                  pos=e.pos)
        self.set_block(b_right)
        right = self.expr(e.right)
        self.emit(Op.COPY, [right], dst=dst, type_=T.BOOL, pos=e.pos)
        self.emit(Op.JUMP, meta={"target": b_end.id})
        self.set_block(b_end)
        return self.slot_op(dst)

    def e_Unary(self, e: A.Unary) -> Operand:
        operand = self.expr(e.operand)
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.UNOP, [operand], dst=dst, type_=(e.inferred or T.ANY),
                  meta={"operator": e.op}, pos=e.pos)
        return self.slot_op(dst)

    def e_Call(self, e: A.Call) -> Operand:
        args = [self.expr(a) for a in e.args]
        callee = e.callee

        if self.policy_ctx_slot is not None and isinstance(callee, A.Name) \
                and callee.id == "role" and len(args) == 1:
            # Spec section 17: `allow role("payroll_admin")` is decided
            # against the context the policy was given, never against
            # ambient process state -- otherwise a policy would not be a
            # pure function of its input and could not be tested.
            dst = self.temp(T.BOOL)
            self.emit(Op.BUILTIN,
                      [self.slot_op(self.policy_ctx_slot),
                       self.const("role", T.TEXT)],
                      dst=dst, type_=T.ANY, meta={"name": "__ctx_get"},
                      pos=e.pos)
            cmp_slot = self.temp(T.BOOL)
            self.emit(Op.BINOP, [self.slot_op(dst), args[0]], dst=cmp_slot,
                      type_=T.BOOL, meta={"operator": "=="}, pos=e.pos)
            return self.slot_op(cmp_slot)

        dst = self.temp(e.inferred or T.ANY)
        if isinstance(callee, A.Name):
            sym = self.checker.resolved.get(id(callee))
            name = sym.name if isinstance(sym, Symbol) else callee.id
            kind = sym.kind if isinstance(sym, Symbol) else None
            if kind == "variant":
                enum_name = self.checker.variants.get(callee.id, ("", None))[0]
                self.emit(Op.MAKE_VARIANT, args, dst=dst,
                          type_=(e.inferred or T.ANY),
                          meta={"tag": callee.id, "enum": enum_name}, pos=e.pos)
                return self.slot_op(dst)
            if kind == "record":
                fields = [f[0] for f in
                          (self.checker.records[callee.id].fields
                           if callee.id in self.checker.records else [])]
                pairs: List[Any] = []
                for fname, arg in zip(fields, args):
                    pairs.append(fname)
                    pairs.append(arg)
                self.emit(Op.MAKE_RECORD, pairs, dst=dst,
                          type_=(e.inferred or T.ANY),
                          meta={"record": callee.id}, pos=e.pos)
                return self.slot_op(dst)
            if kind in ("fn", "pipeline", "service", "agent", "policy",
                        "transaction", "model", "test"):
                target = callee.id
                if kind == "model":
                    target = f"{callee.id}.predict"
                if target in self.prog.functions or target in self.checker.functions:
                    self.emit(Op.CALL, args, dst=dst,
                              type_=(e.inferred or T.ANY),
                              meta={"callee": target}, pos=e.pos)
                    return self.slot_op(dst)
            if kind == "builtin" and isinstance(sym, Symbol) and sym.builtin:
                self.emit(Op.BUILTIN, args, dst=dst, type_=(e.inferred or T.ANY),
                          meta={"name": sym.builtin.name}, pos=e.pos)
                return self.slot_op(dst)
            operand = self.expr(callee)
            self.emit(Op.CALL_INDIRECT, [operand] + args, dst=dst,
                      type_=(e.inferred or T.ANY), meta={}, pos=e.pos)
            return self.slot_op(dst)

        if isinstance(callee, A.Member):
            variant = self._qualified_variant(callee.obj, callee.attr)
            if variant is not None:
                # `Route.Oral(250.0)`: same construction as the bare name.
                name, enum_name, _ = variant
                self.emit(Op.MAKE_VARIANT, args, dst=dst,
                          type_=(e.inferred or T.ANY),
                          meta={"tag": name, "enum": enum_name}, pos=e.pos)
                return self.slot_op(dst)
            obj_sym = self.checker.resolved.get(id(callee.obj)) \
                if isinstance(callee.obj, A.Name) else None
            if isinstance(obj_sym, Symbol) and obj_sym.kind == "module":
                self.emit(Op.BUILTIN, args, dst=dst, type_=(e.inferred or T.ANY),
                          meta={"name": f"{obj_sym.name}.{callee.attr}"},
                          pos=e.pos)
                return self.slot_op(dst)
            obj_op = self.expr(callee.obj)
            self.emit(Op.METHOD_CALL, [obj_op] + args, dst=dst,
                      type_=(e.inferred or T.ANY),
                      meta={"method": callee.attr}, pos=e.pos)
            return self.slot_op(dst)

        operand = self.expr(callee)
        self.emit(Op.CALL_INDIRECT, [operand] + args, dst=dst,
                  type_=(e.inferred or T.ANY), meta={}, pos=e.pos)
        return self.slot_op(dst)

    def e_Member(self, e: A.Member) -> Operand:
        resolved = self.checker.resolved.get(id(e))
        if isinstance(resolved, L.Builtin):
            # A builtin used as a value, e.g. passed to collections.map.
            return Operand(kind="builtin", name=resolved.name,
                           type=resolved.fn_type())
        if isinstance(resolved, L.Constant):
            return self.const(resolved.value, resolved.type)
        obj_sym = self.checker.resolved.get(id(e.obj)) \
            if isinstance(e.obj, A.Name) else None
        if isinstance(obj_sym, Symbol) and obj_sym.kind == "module":
            return Operand(kind="builtin", name=f"{obj_sym.name}.{e.attr}",
                           type=(e.inferred or T.ANY))
        variant = self._qualified_variant(e.obj, e.attr)
        if variant is not None:
            # `Colour.Red`: the enum name is a type, not a value, so it cannot
            # be lowered as a global.  The variant itself is already a symbol
            # under its bare name; qualify it to the same constant.
            name, enum_name, vtype = variant
            return self.const(Const.variant(name, enum_name), vtype)
        obj = self.expr(e.obj)
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.FIELD, [obj], dst=dst, type_=(e.inferred or T.ANY),
                  meta={"name": e.attr}, pos=e.pos)
        return self.slot_op(dst)

    def _qualified_variant(self, obj: Optional[A.Expr],
                           attr: str) -> Optional[Tuple[str, str, T.Type]]:
        """Resolve `EnumName.Variant` to (variant, enum, type), or None.

        Payload-less variants are values (`Colour.Red`) and variants with
        payloads are constructors (`Route.Oral(250)`); both are written
        qualified, and neither has a runtime value behind the enum name.
        """
        if not isinstance(obj, A.Name):
            return None
        obj_sym = self.checker.globals.lookup(obj.id)
        if obj_sym is None or obj_sym.kind != "enum":
            return None
        enum_name = self.checker.variants.get(attr, ("", None))[0]
        if enum_name != obj.id:
            return None
        variant_sym = self.checker.globals.lookup(attr)
        vtype = variant_sym.type if variant_sym is not None else T.ANY
        return attr, enum_name, vtype

    def e_Index(self, e: A.Index) -> Operand:
        obj = self.expr(e.obj)
        index = self.expr(e.index)
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.INDEX, [obj, index], dst=dst, type_=(e.inferred or T.ANY),
                  pos=e.pos)
        return self.slot_op(dst)

    def e_ListLit(self, e: A.ListLit) -> Operand:
        items = [self.expr(i) for i in e.items]
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.MAKE_LIST, items, dst=dst, type_=(e.inferred or T.ANY),
                  pos=e.pos)
        return self.slot_op(dst)

    def e_SetLit(self, e: A.SetLit) -> Operand:
        items = [self.expr(i) for i in e.items]
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.MAKE_SET, items, dst=dst, type_=(e.inferred or T.ANY),
                  pos=e.pos)
        return self.slot_op(dst)

    def e_TupleLit(self, e: A.TupleLit) -> Operand:
        items = [self.expr(i) for i in e.items]
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.MAKE_TUPLE, items, dst=dst, type_=(e.inferred or T.ANY),
                  pos=e.pos)
        return self.slot_op(dst)

    def e_MapLit(self, e: A.MapLit) -> Operand:
        pairs: List[Operand] = []
        for key, value in e.entries:
            pairs.append(self.expr(key))
            pairs.append(self.expr(value))
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.MAKE_MAP, pairs, dst=dst, type_=(e.inferred or T.ANY),
                  pos=e.pos)
        return self.slot_op(dst)

    def e_RecordLit(self, e: A.RecordLit) -> Operand:
        pairs: List[Operand] = []
        for key, value in e.fields:
            pairs.append(self.const(key, T.TEXT))
            pairs.append(self.expr(value))
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.MAKE_RECORD, pairs, dst=dst, type_=(e.inferred or T.ANY),
                  meta={"record": e.name}, pos=e.pos)
        return self.slot_op(dst)

    def e_Construct(self, e: A.Construct) -> Operand:
        args = [self.expr(a) for a in e.args]
        if e.tag in self.checker.variants:
            enum_name = self.checker.variants[e.tag][0]
            dst = self.temp(e.inferred or T.ANY)
            self.emit(Op.MAKE_VARIANT, args, dst=dst,
                      type_=(e.inferred or T.ANY),
                      meta={"tag": e.tag, "enum": enum_name}, pos=e.pos)
            return self.slot_op(dst)
        dst = self.temp(e.inferred or T.ANY)
        self.emit(Op.CONSTRUCT, args, dst=dst, type_=(e.inferred or T.ANY),
                  meta={"tag": e.tag}, pos=e.pos)
        return self.slot_op(dst)

    def e_Cast(self, e: A.Cast) -> Operand:
        operand = self.expr(e.expr)
        target = self.checker.resolve_type(e.target)
        dst = self.temp(target)
        self.emit(Op.CAST, [operand], dst=dst, type_=target,
                  meta={"target": target.render()}, pos=e.pos)
        return self.slot_op(dst)

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------
    def stmt(self, node: A.Stmt) -> None:
        method = getattr(self, f"s_{type(node).__name__}", None)
        if method is None:
            if isinstance(node, (A.EffectDecl, A.RecoveryStep, A.PolicyRule,
                                 A.AuditDirective)):
                return
            raise GamaError(Diagnostic(
                Severity.ERROR, Phase.GIR,
                f"cannot lower statement `{type(node).__name__}` to GIR",
                pos=node.pos))
        method(node)

    def s_Block(self, st: A.Block) -> None:
        self.push_scope()
        for s in st.stmts:
            self.stmt(s)
        self.pop_scope()

    def s_LetDecl(self, st: A.LetDecl) -> None:
        ty = self.checker.resolved.get(id(st))
        if not isinstance(ty, T.Type):
            sym = self.checker.globals.lookup(st.name)
            ty = sym.type if sym is not None else T.ANY
        if self.fn.kind == "main" and st.name in self.global_names:
            # A top-level binding lives in the module's global table, not in
            # a local slot of the initializer -- otherwise every other
            # function would read an unbound name.  `<main>` is the
            # initializer that gives it its value, in source order.
            if st.value is not None:
                value = self.expr(st.value)
            else:
                value = self.const(Const.default(ty), ty)
            self.emit(Op.STORE_GLOBAL, [value], dst=-1,
                      meta={"name": st.name, "secret": bool(st.secret),
                            "mutable": bool(st.mutable)}, pos=st.pos)
            return
        slot = self.fn.new_slot(st.name, ty, "local", mutable=st.mutable,
                                secret=st.secret)
        self.bind(st.name, slot)
        if st.value is not None:
            value = self.expr(st.value)
            self.emit(Op.COPY, [value], dst=slot, type_=ty, pos=st.pos)
            if st.secret:
                self.emit(Op.SECRET_GUARD, [self.slot_op(slot)], dst=-1,
                          meta={"name": st.name}, pos=st.pos)
        else:
            self.emit(Op.CONST, [self.const(Const.default(ty), ty)],
                      dst=slot, type_=ty, pos=st.pos)

    def _target_slot(self, target: A.Expr) -> Optional[int]:
        if isinstance(target, A.Name):
            return self.lookup(target.id)
        return None

    def s_Assign(self, st: A.Assign) -> None:
        value = self.expr(st.value)
        target = st.target
        if isinstance(target, A.Name):
            slot = self.lookup(target.id)
            if slot is None:
                # Assignment to a module-level binding.
                self.emit(Op.STORE_GLOBAL, [value], dst=-1,
                          meta={"name": target.id}, pos=st.pos)
                if st.op != "=":
                    pass
                return
            if st.op == "=":
                self.emit(Op.COPY, [value], dst=slot,
                          type_=self.fn.slots[slot].type, pos=st.pos)
            else:
                operator = {"+=": "+", "-=": "-", "*=": "*", "/=": "/"}[st.op]
                dst = self.temp(self.fn.slots[slot].type)
                self.emit(Op.BINOP, [self.slot_op(slot), value], dst=dst,
                          type_=self.fn.slots[slot].type,
                          meta={"operator": operator}, pos=st.pos)
                self.emit(Op.COPY, [self.slot_op(dst)], dst=slot, pos=st.pos)
            return
        if isinstance(target, A.Member):
            obj = self.expr(target.obj)
            self.emit(Op.SET_FIELD, [obj, value], dst=-1,
                      meta={"name": target.attr}, pos=st.pos)
            return
        if isinstance(target, A.Index):
            obj = self.expr(target.obj)
            index = self.expr(target.index)
            self.emit(Op.SET_INDEX, [obj, index, value], dst=-1, pos=st.pos)
            return
        self.expr(target)

    def s_ExprStmt(self, st: A.ExprStmt) -> None:
        self.expr(st.expr)

    def s_If(self, st: A.If) -> None:
        cond = self.expr(st.cond)
        b_then = self.new_block("if.then")
        b_else = self.new_block("if.else")
        b_end = self.new_block("if.end")
        self.emit(Op.JUMP_IF, [cond], meta={"target": b_then.id,
                                            "target_false": b_else.id},
                  pos=st.pos)
        self.set_block(b_then)
        if st.then_body:
            self.s_Block(st.then_body)
        self.emit(Op.JUMP, meta={"target": b_end.id})
        self.set_block(b_else)
        if st.else_body:
            self.s_Block(st.else_body)
        self.emit(Op.JUMP, meta={"target": b_end.id})
        self.set_block(b_end)

    def s_While(self, st: A.While) -> None:
        b_cond = self.new_block("while.cond")
        b_body = self.new_block("while.body")
        b_end = self.new_block("while.end")
        self.emit(Op.JUMP, meta={"target": b_cond.id})
        self.set_block(b_cond)
        cond = self.expr(st.cond)
        self.emit(Op.JUMP_IF, [cond], meta={"target": b_body.id,
                                            "target_false": b_end.id},
                  pos=st.pos)
        self.set_block(b_body)
        self.loop_depth += 1
        self.loop_breaks.append(b_end.id)
        self.loop_continues.append(b_cond.id)
        self.push_scope()
        if st.body:
            for s in st.body.stmts:
                self.stmt(s)
        self.pop_scope()
        self.loop_continues.pop()
        self.loop_breaks.pop()
        self.loop_depth -= 1
        self.emit(Op.JUMP, meta={"target": b_cond.id})
        self.set_block(b_end)

    def s_For(self, st: A.For) -> None:
        self.emit_for(st.var, st.iter, st.body, st.pos)

    def emit_for(self, var: str, iterable: A.Expr, body: Optional[A.Block],
                 pos: SourcePos) -> None:
        src = self.expr(iterable)
        it_slot = self.fn.new_slot(f"{var}__iter", T.ListType(T.ANY), "temp")
        self.emit(Op.BUILTIN, [src], dst=it_slot, meta={"name": "__iter_list"},
                  pos=pos)
        idx_slot = self.fn.new_slot(f"{var}__i", T.I64, "temp")
        self.emit(Op.CONST, [self.const(0, T.I64)], dst=idx_slot, type_=T.I64)
        len_slot = self.fn.new_slot(f"{var}__n", T.I64, "temp")
        self.emit(Op.BUILTIN, [self.slot_op(it_slot)], dst=len_slot,
                  meta={"name": "len"}, pos=pos)

        b_cond = self.new_block("for.cond")
        b_body = self.new_block("for.body")
        b_end = self.new_block("for.end")
        self.emit(Op.JUMP, meta={"target": b_cond.id})
        self.set_block(b_cond)
        cmp_slot = self.temp(T.BOOL)
        self.emit(Op.BINOP, [self.slot_op(idx_slot), self.slot_op(len_slot)],
                  dst=cmp_slot, type_=T.BOOL, meta={"operator": "<"}, pos=pos)
        self.emit(Op.JUMP_IF, [self.slot_op(cmp_slot)],
                  meta={"target": b_body.id, "target_false": b_end.id}, pos=pos)

        self.set_block(b_body)
        var_slot = self.fn.new_slot(var, T.ANY, "local")
        self.emit(Op.INDEX, [self.slot_op(it_slot), self.slot_op(idx_slot)],
                  dst=var_slot, pos=pos)
        self.push_scope()
        self.bind(var, var_slot)
        self.loop_depth += 1
        self.loop_breaks.append(b_end.id)
        self.loop_continues.append(b_cond.id)
        if body:
            for s in body.stmts:
                self.stmt(s)
        self.loop_continues.pop()
        self.loop_breaks.pop()
        self.loop_depth -= 1
        self.pop_scope()
        inc = self.temp(T.I64)
        self.emit(Op.BINOP, [self.slot_op(idx_slot), self.const(1, T.I64)],
                  dst=inc, type_=T.I64, meta={"operator": "+"}, pos=pos)
        self.emit(Op.COPY, [self.slot_op(inc)], dst=idx_slot, type_=T.I64)
        self.emit(Op.JUMP, meta={"target": b_cond.id})
        self.set_block(b_end)

    def s_ForAll(self, st: A.ForAll) -> None:
        """Property quantifier: lower to a deterministic sample loop (spec 26)."""
        domain = self.expr(st.domain) if st.domain is not None else \
            self.const(None, T.ANY)
        samples = self.fn.new_slot(f"{st.var}__samples", T.ListType(T.ANY), "temp")
        self.emit(Op.BUILTIN,
                  [domain, self.const(int(st.samples), T.I64),
                   self.const(0, T.I64)],
                  dst=samples, meta={"name": "__samples"}, pos=st.pos)
        sample_expr = A.Name(pos=st.pos, id="__samples_placeholder")
        # Reuse the ordinary for-loop lowering over the generated samples.
        src_slot = samples
        idx_slot = self.fn.new_slot(f"{st.var}__i", T.I64, "temp")
        self.emit(Op.CONST, [self.const(0, T.I64)], dst=idx_slot, type_=T.I64)
        len_slot = self.fn.new_slot(f"{st.var}__n", T.I64, "temp")
        self.emit(Op.BUILTIN, [self.slot_op(src_slot)], dst=len_slot,
                  meta={"name": "len"}, pos=st.pos)
        b_cond = self.new_block("forall.cond")
        b_body = self.new_block("forall.body")
        b_end = self.new_block("forall.end")
        self.emit(Op.JUMP, meta={"target": b_cond.id})
        self.set_block(b_cond)
        cmp_slot = self.temp(T.BOOL)
        self.emit(Op.BINOP, [self.slot_op(idx_slot), self.slot_op(len_slot)],
                  dst=cmp_slot, type_=T.BOOL, meta={"operator": "<"}, pos=st.pos)
        self.emit(Op.JUMP_IF, [self.slot_op(cmp_slot)],
                  meta={"target": b_body.id, "target_false": b_end.id}, pos=st.pos)
        self.set_block(b_body)
        var_slot = self.fn.new_slot(st.var, T.ANY, "local")
        self.emit(Op.INDEX, [self.slot_op(src_slot), self.slot_op(idx_slot)],
                  dst=var_slot, pos=st.pos)
        self.push_scope()
        self.bind(st.var, var_slot)
        if st.where is not None:
            cond = self.expr(st.where)
            b_inner = self.new_block("forall.where")
            b_skip = self.new_block("forall.skip")
            self.emit(Op.JUMP_IF, [cond], meta={"target": b_inner.id,
                                                "target_false": b_skip.id},
                      pos=st.pos)
            self.set_block(b_inner)
        if st.body:
            for s in st.body.stmts:
                self.stmt(s)
        if st.where is not None:
            self.emit(Op.JUMP, meta={"target": b_skip.id})
            self.set_block(b_skip)
        self.pop_scope()
        inc = self.temp(T.I64)
        self.emit(Op.BINOP, [self.slot_op(idx_slot), self.const(1, T.I64)],
                  dst=inc, type_=T.I64, meta={"operator": "+"}, pos=st.pos)
        self.emit(Op.COPY, [self.slot_op(inc)], dst=idx_slot, type_=T.I64)
        self.emit(Op.JUMP, meta={"target": b_cond.id})
        self.set_block(b_end)

    def s_Return(self, st: A.Return) -> None:
        value = self.expr(st.value) if st.value is not None else \
            self.const(Const.unit(), T.UNIT)
        if self.needs_epilogue:
            if self.result_slot is None:
                self.result_slot = self.fn.new_slot("result", self.fn.ret, "temp")
            self.emit(Op.COPY, [value], dst=self.result_slot,
                      type_=self.fn.ret, pos=st.pos)
            if self.epilogue is None:
                self.epilogue = self.new_block("epilogue")
            self.emit(Op.JUMP, meta={"target": self.epilogue.id}, pos=st.pos)
            self.set_block(self.new_block("after.return"))
        else:
            self.emit(Op.RETURN, [value], pos=st.pos)

    def s_Break(self, st: A.Break) -> None:
        target = self.loop_breaks[-1] if self.loop_breaks else None
        if target is None:
            self.emit(Op.RETURN, [self.const(Const.unit(), T.UNIT)], pos=st.pos)
        else:
            self.emit(Op.JUMP, meta={"target": target}, pos=st.pos)

    def s_Continue(self, st: A.Continue) -> None:
        target = self.loop_continues[-1] if self.loop_continues else None
        if target is not None:
            self.emit(Op.JUMP, meta={"target": target}, pos=st.pos)

    def s_Match(self, st: A.Match) -> None:
        """Lower `match` into a decision cascade (spec section 6).

        Each arm becomes a chain of test blocks followed by a binding block.
        Splitting the two matters: the discriminating test must branch *past*
        the payload extraction when it fails, so extraction has to live in a
        block the branch targets rather than in the block that performs the
        test.  Guards are evaluated after binding, so `ok(v) if v > 0` sees
        `v`, and a guard that fails falls through to the next arm.
        """
        subject = self.expr(st.subject)
        s_slot = self.temp(subject.type or T.ANY)
        self.emit(Op.COPY, [subject], dst=s_slot, pos=st.pos)
        subj = self.slot_op(s_slot)
        b_end = self.new_block("match.end")
        next_test = self.new_block("match.arm0")
        self.emit(Op.JUMP, meta={"target": next_test.id})

        for arm_index, arm in enumerate(st.arms):
            b_next = self.new_block(f"match.arm{arm_index + 1}")
            tests: List[Tuple[Any, ...]] = []
            binds: List[Tuple[str, Tuple[Any, ...]]] = []
            self._plan_pattern(arm.pattern, (), tests, binds)

            b_bind = self.new_block(f"match.bind{arm_index}")
            if not tests:
                self.set_block(next_test)
                self.emit(Op.JUMP, meta={"target": b_bind.id})
            else:
                # One block per test, so a passing test falls into the next
                # test and the final one falls into the binding block.
                blocks = [next_test] + [
                    self.new_block(f"match.arm{arm_index}.t{i}")
                    for i in range(1, len(tests))]
                targets = blocks[1:] + [b_bind]
                for block, test, target in zip(blocks, tests, targets):
                    self.set_block(block)
                    cond = self._emit_pattern_test(test, subj, arm.pos)
                    self.emit(Op.JUMP_IF, [self.slot_op(cond)],
                              meta={"target": target.id,
                                    "target_false": b_next.id}, pos=arm.pos)

            self.set_block(b_bind)
            self.push_scope()
            for name, path in binds:
                slot = self.fn.new_slot(name, T.ANY, "local")
                self.emit(Op.COPY, [self._materialise(subj, path, arm.pos)],
                          dst=slot, pos=arm.pos)
                self.bind(name, slot)
            if arm.guard is not None:
                guard = self.expr(arm.guard)
                b_guard_ok = self.new_block(f"match.guard{arm_index}")
                self.emit(Op.JUMP_IF, [guard],
                          meta={"target": b_guard_ok.id,
                                "target_false": b_next.id}, pos=arm.pos)
                self.set_block(b_guard_ok)
            for body_stmt in arm.body:
                self.stmt(body_stmt)
            self.pop_scope()
            self.emit(Op.JUMP, meta={"target": b_end.id})
            next_test = b_next

        self.set_block(next_test)
        self.emit(Op.MATCH_FAIL, [subj],
                  meta={"subject_type": (st.subject.inferred.render()
                                         if st.subject.inferred else "Any")},
                  pos=st.pos)
        self.set_block(b_end)

    def _plan_pattern(self, pat: A.Pattern, path: Tuple[Any, ...],
                      tests: List[Tuple[Any, ...]],
                      binds: List[Tuple[str, Tuple[Any, ...]]]) -> None:
        """Collect the tests and bindings a pattern needs, without emitting.

        ``path`` is the sequence of field/index steps from the match subject
        down to the value this pattern inspects, so nested patterns such as
        ``ok(some(x))`` produce paths like ``(("field", "value"),
        ("field", "value"))``.
        """
        if isinstance(pat, A.WildcardPat):
            return
        if isinstance(pat, A.NamePat):
            # The checker reclassifies a bare identifier that names an enum
            # variant: it tests the tag and binds nothing.
            if id(pat) in self.checker.variant_patterns:
                if pat.name == "none":
                    tests.append(("field_bool", path, "some", False))
                else:
                    tests.append(("tag", path, pat.name))
                return
            binds.append((pat.name, path))
            return
        if isinstance(pat, A.LiteralPat):
            tests.append(("eq", path, pat.value))
            return
        if isinstance(pat, A.CtorPat):
            tag = pat.tag
            if tag in ("ok", "fail"):
                tests.append(("field_bool", path, "ok", tag == "ok"))
                payload = "value" if tag == "ok" else "error"
                for sub in pat.args:
                    self._plan_pattern(sub, path + (("field", payload),),
                                       tests, binds)
                return
            if tag in ("some", "none"):
                tests.append(("field_bool", path, "some", tag == "some"))
                if tag == "some":
                    for sub in pat.args:
                        self._plan_pattern(sub, path + (("field", "value"),),
                                           tests, binds)
                return
            tests.append(("tag", path, tag))
            for index, sub in enumerate(pat.args):
                self._plan_pattern(sub, path + (("index", index),), tests, binds)
            return
        # Unknown pattern shape: bind it whole rather than silently dropping
        # the arm, so behaviour stays visible.
        binds.append((f"__pat{len(binds)}", path))

    def _emit_pattern_test(self, test: Tuple[Any, ...], subj: Operand,
                           pos: SourcePos) -> int:
        """Emit one discriminating test; return the slot holding its Bool."""
        kind = test[0]
        operand = self._materialise(subj, test[1], pos)
        if kind == "eq":
            cond = self.temp(T.BOOL)
            self.emit(Op.BINOP, [operand, self.const(test[2], T.ANY)],
                      dst=cond, type_=T.BOOL, meta={"operator": "=="}, pos=pos)
            return cond
        if kind == "field_bool":
            cond = self.temp(T.BOOL)
            self.emit(Op.FIELD, [operand], dst=cond,
                      meta={"name": test[2]}, pos=pos)
            if not test[3]:
                neg = self.temp(T.BOOL)
                self.emit(Op.UNOP, [self.slot_op(cond)], dst=neg, type_=T.BOOL,
                          meta={"operator": "not"}, pos=pos)
                return neg
            return cond
        # tag comparison against an enum variant
        tag_slot = self.temp(T.ANY)
        self.emit(Op.FIELD, [operand], dst=tag_slot, meta={"name": "tag"},
                  pos=pos)
        cond = self.temp(T.BOOL)
        self.emit(Op.BINOP, [self.slot_op(tag_slot), self.const(test[2], T.TEXT)],
                  dst=cond, type_=T.BOOL, meta={"operator": "=="}, pos=pos)
        return cond

    def _materialise(self, subj: Operand, path: Tuple[Any, ...],
                     pos: SourcePos) -> Operand:
        """Walk a field/index path from the subject, one instruction a step."""
        operand = subj
        for step in path:
            tmp = self.temp(T.ANY)
            if step[0] == "field":
                self.emit(Op.FIELD, [operand], dst=tmp,
                          meta={"name": step[1]}, pos=pos)
            else:
                self.emit(Op.INDEX, [operand, self.const(step[1], T.I64)],
                          dst=tmp, pos=pos)
            operand = self.slot_op(tmp)
        return operand

    def s_Parallel(self, st: A.Parallel) -> None:
        """Lower a parallel region into an operation graph (spec 3, 9C).

        Each statement becomes one task function.  Read/write dependence
        analysis over the region yields the edges; tasks with no edge between
        them may execute concurrently.  Names written by the region but not
        previously in scope are given a slot in the *enclosing* function, so
        the results land somewhere the rest of the program can read.
        """
        if st.body is None or not st.body.stmts:
            return
        statements = st.body.stmts
        enclosing: Dict[str, int] = {}
        for scope in self.scopes:
            enclosing.update(scope)

        # Give every name the region writes a home in the enclosing frame.
        for task_stmt in statements:
            for name in sorted(names_written(task_stmt)):
                if name in enclosing:
                    continue
                slot = self.fn.new_slot(name, T.ANY, "local", mutable=True)
                enclosing[name] = slot
                self.bind(name, slot)

        # A function may contain several parallel regions; each needs its own
        # namespace, or the task functions of one region would overwrite the
        # other's in the program's function table.
        region = self.parallel_regions
        self.parallel_regions += 1

        visible = set(enclosing)
        tasks: List[Dict[str, Any]] = []
        for index, task_stmt in enumerate(statements):
            declared = names_declared(task_stmt)
            reads = names_read(task_stmt) - declared
            writes = names_written(task_stmt)
            params = sorted((reads | writes) & visible)
            tasks.append({
                "index": index, "name": f"t{index}", "stmt": task_stmt,
                "reads": sorted(reads & visible),
                "writes": sorted(writes & visible),
                "params": params,
                "depends_on": [],
            })

        # RAW, WAR and WAW edges, always backwards in program order.
        for j, tj in enumerate(tasks):
            for i in range(j):
                ti = tasks[i]
                wi, wj = set(ti["writes"]), set(tj["writes"])
                rj = set(tj["reads"]) | wj
                ri = set(ti["reads"]) | wi
                if (wi & rj) or (wj & ri) or (wi & wj):
                    tj["depends_on"].append(ti["name"])

        for task in tasks:
            fname = f"{self.fn.name}$par{region}${task['name']}"
            fb = FunctionBuilder(self.prog, self.checker, fname, kind="task",
                                 ret=T.MapType(T.TEXT, T.ANY),
                                 global_names=self.global_names,
                                 pos=task["stmt"].pos)
            fb.start()
            for pname in task["params"]:
                pslot_index = enclosing.get(pname)
                ptype = (self.fn.slots[pslot_index].type
                         if pslot_index is not None else T.ANY)
                pslot = fb.fn.new_slot(pname, ptype, "param")
                fb.fn.params.append(pslot)
                fb.fn.param_names.append(pname)
                fb.bind(pname, pslot)
            fb.push_scope()
            fb.stmt(task["stmt"])
            fb.pop_scope()
            pairs: List[Operand] = []
            for wname in task["writes"]:
                pairs.append(fb.const(wname, T.TEXT))
                wslot = fb.lookup(wname)
                pairs.append(fb.slot_op(wslot) if wslot is not None
                             else fb.const(Const.unit(), T.UNIT))
            out_slot = fb.temp(T.MapType(T.TEXT, T.ANY))
            fb.emit(Op.MAKE_MAP, pairs, dst=out_slot,
                    type_=T.MapType(T.TEXT, T.ANY), pos=task["stmt"].pos)
            fb.emit(Op.RETURN, [fb.slot_op(out_slot)], pos=task["stmt"].pos)
            self.prog.add(fb.fn)
            self.fn.parallel_tasks.append(ParallelTask(
                name=task["name"], function=fname, reads=task["reads"],
                writes=task["writes"], depends_on=task["depends_on"]))

        meta_tasks = [{
            "index": t["index"],
            "name": t["name"],
            "function": f"{self.fn.name}$par{region}${t['name']}",
            "params": t["params"],
            "param_slots": [enclosing.get(p) for p in t["params"]],
            "writes": t["writes"],
            "write_slots": [enclosing.get(w) for w in t["writes"]],
            "depends_on": t["depends_on"],
        } for t in tasks]
        self.emit(Op.PARALLEL, [], dst=-1, meta={"tasks": meta_tasks},
                  pos=st.pos)

    def s_AuditRecord(self, st: A.AuditRecord) -> None:
        keys: List[str] = []
        ops: List[Operand] = []
        for key, expr in st.fields:
            keys.append(key)
            ops.append(self.expr(expr))
        self.emit(Op.AUDIT, ops, dst=-1, meta={"keys": keys}, pos=st.pos)
        self.fn.audit_points += 1

    def s_Require(self, st: A.Require) -> None:
        cond = self.expr(st.expr)
        self.emit(Op.REQUIRE, [cond], dst=-1,
                  meta={"message": st.message or ""}, pos=st.pos)

    def s_Assert(self, st: A.Assert) -> None:
        cond = self.expr(st.expr)
        self.emit(Op.ASSERT, [cond], dst=-1,
                  meta={"message": st.message or ""}, pos=st.pos)

    def s_CheckpointStmt(self, st: A.CheckpointStmt) -> None:
        self.emit(Op.CHECKPOINT, [], dst=-1,
                  meta={"mode": st.mode,
                        "interval": st.interval_seconds,
                        "boundary": st.boundary or "", "raw": st.raw},
                  pos=st.pos)
        self.fn.recovery_points += 1

    def s_Section(self, st: A.Section) -> None:
        if st.body:
            self.s_Block(st.body)

    def s_HandlerDecl(self, st: A.HandlerDecl) -> None:
        if st.body:
            self.s_Block(st.body)

    def s_IODirective(self, st: A.IODirective) -> None:
        if self.lookup(st.name) is None:
            ty = self.checker.resolve_type(st.type) if st.type else T.ANY
            slot = self.fn.new_slot(st.name, ty, "param")
            self.bind(st.name, slot)
            self.emit(Op.CONST, [self.const(Const.default(ty), ty)],
                      dst=slot, type_=ty, pos=st.pos)

    def s_StageDirective(self, st: A.StageDirective) -> None:
        args = [self.const(st.stage, T.TEXT)]
        args += [self.expr(a) for a in st.args]
        if st.using is not None:
            args.append(self.expr(st.using))
        dst = self.temp(T.UNIT)
        self.emit(Op.BUILTIN, args, dst=dst, meta={"name": "__stage"},
                  pos=st.pos)

    def s_AuditDirective(self, st: A.AuditDirective) -> None:
        dst = self.temp(T.UNIT)
        self.emit(Op.BUILTIN,
                  [self.const(st.phrase or "audit", T.TEXT)], dst=dst,
                  meta={"name": "audit.emit"}, pos=st.pos)
        self.fn.audit_points += 1

    def s_PolicyRule(self, st: A.PolicyRule) -> None:
        if st.expr is not None:
            self.expr(st.expr)

    # ------------------------------------------------------------------
    def finish(self) -> GFunction:
        if self.cur is not None and self.cur.terminator is None:
            self.emit(Op.RETURN, [self.const(Const.unit(), T.UNIT)])
        if self.needs_epilogue:
            if self.result_slot is None:
                self.result_slot = self.fn.new_slot("result", self.fn.ret, "temp")
            if self.epilogue is None:
                self.epilogue = self.new_block("epilogue")
            if self.cur is not None and self.cur.id != self.epilogue.id \
                    and self.cur.terminator is None:
                self.emit(Op.COPY, [self.const(Const.default(self.fn.ret),
                                               self.fn.ret)],
                          dst=self.result_slot, type_=self.fn.ret)
                self.emit(Op.JUMP, meta={"target": self.epilogue.id})
            self.set_block(self.epilogue)
            self.push_scope()
            self.bind("result", self.result_slot)
            for contract in self.fn.contracts:
                if contract.get("kind") != "ensures":
                    continue
                cond = self.expr(contract["expr"])
                self.emit(Op.CONTRACT, [cond], dst=-1,
                          meta={"kind": "ensures",
                                "text": contract.get("text", "")},
                          pos=contract.get("pos"))
            self.pop_scope()
            self.emit(Op.RETURN, [self.slot_op(self.result_slot)])
        return self.fn


class Builder:
    """Lowers a whole checked module into a GIR program."""

    def __init__(self, module: A.Module, checker: Checker):
        self.module = module
        self.checker = checker
        self.prog = GProgram(name=module.module_name or module.filename)
        self.progrants = list(module.grants)
        self.global_names: Set[str] = set()

    # ------------------------------------------------------------------
    def build(self) -> GProgram:
        self.collect_global_names()
        self.build_main()
        for decl in self.module.decls:
            self.build_decl(decl)
        self.prog.grants = self.progrants
        self.prog.records = {
            name: [(f, t.render()) for f, t in rec.fields]
            for name, rec in self.checker.records.items()}
        self.prog.enums = {
            name: [v.name for v in en.variants]
            for name, en in self.checker.enums.items()}
        self.prog.global_bindings = sorted(self.global_names)
        self.prog.metadata = {
            "source": self.module.filename,
            "profile": self.checker.profile,
            "spec_version": "1.0",
            "functions_by_kind": self._kind_counts(),
        }
        self.prog.tests = [f"test:{t.name}" for t in self.checker.tests]
        return self.prog

    def _kind_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for fn in self.prog.functions.values():
            counts[fn.kind] = counts.get(fn.kind, 0) + 1
        return counts

    def collect_global_names(self) -> None:
        for stmt in self.module.top_level:
            if isinstance(stmt, A.LetDecl):
                self.global_names.add(stmt.name)
        for decl in self.module.decls:
            if isinstance(decl, A.ServiceDecl) and decl.name == "<module>":
                continue
            self.global_names.add(decl.name)
            if isinstance(decl, A.ModelDecl):
                for method in decl.methods:
                    self.global_names.add(f"{decl.name}.{method.name.split('.')[-1]}")
            if isinstance(decl, A.AgentDecl):
                for handler in decl.handlers:
                    self.global_names.add(f"{decl.name}.on.{handler.event}")
        self.global_names.update(self.checker.variants)

    # ------------------------------------------------------------------
    def new_fn(self, name: str, kind: str = "fn", ret: T.Type = T.UNIT,
               effects: Sequence[str] = (), caps: Sequence[str] = (),
               deterministic: bool = False,
               contracts: Optional[List[Dict[str, Any]]] = None,
               pos: Optional[SourcePos] = None) -> FunctionBuilder:
        return FunctionBuilder(self.prog, self.checker, name, kind=kind,
                               ret=ret, effects=effects, caps=caps,
                               deterministic=deterministic,
                               contracts=contracts,
                               global_names=self.global_names, pos=pos)

    def build_main(self) -> None:
        fb = self.new_fn("<main>", kind="main", ret=T.UNIT,
                         pos=SourcePos(self.module.filename, 1, 1, 0))
        fb.start()
        info = self.checker.functions.get("<main>")
        if info is not None:
            fb.fn.effects = tuple(sorted(info.effects))
        for stmt in self.module.top_level:
            fb.stmt(stmt)
        self.prog.add(fb.finish())
        self.prog.entry = "<main>"

    def build_decl(self, decl: A.Decl) -> None:
        if isinstance(decl, A.FnDecl):
            self.build_function(decl)
        elif isinstance(decl, A.PipelineDecl):
            self.build_pipeline(decl)
        elif isinstance(decl, A.ModelDecl):
            self.build_model(decl)
        elif isinstance(decl, A.ServiceDecl):
            if decl.name != "<module>":
                self.build_service(decl)
        elif isinstance(decl, A.AgentDecl):
            self.build_agent(decl)
        elif isinstance(decl, A.FaultDecl):
            self.build_fault(decl)
        elif isinstance(decl, A.PolicyDecl):
            self.build_policy(decl)
        elif isinstance(decl, A.TransactionDecl):
            self.build_transaction(decl)
        elif isinstance(decl, A.TestDecl):
            self.build_test(decl)

    # ------------------------------------------------------------------
    def _contracts_of(self, decl: A.FnDecl) -> List[Dict[str, Any]]:
        return [{"kind": c.kind, "expr": c.expr, "pos": c.pos,
                 "text": c.kind} for c in decl.contracts]

    def build_function(self, decl: A.FnDecl) -> None:
        info = self.checker.functions[decl.name]
        fb = self.new_fn(decl.name, kind="fn", ret=info.fn_type.ret,
                         effects=sorted(info.effects),
                         caps=sorted(info.caps),
                         deterministic=decl.deterministic,
                         contracts=self._contracts_of(decl), pos=decl.pos)
        fb.start()
        for param, ptype in zip(decl.params, info.fn_type.params):
            secret = isinstance(ptype, T.SecretType)
            slot = fb.fn.new_slot(param.name, ptype, "param", secret=secret)
            fb.fn.params.append(slot)
            fb.fn.param_names.append(param.name)
            fb.bind(param.name, slot)
            if secret:
                fb.fn.secret_params.append(slot)
        for contract in decl.contracts:
            if contract.kind == "requires":
                cond = fb.expr(contract.expr)
                fb.emit(Op.CONTRACT, [cond], dst=-1,
                        meta={"kind": "requires",
                              "text": contract.text or "requires"},
                        pos=contract.pos)
        if decl.body:
            for stmt in decl.body.stmts:
                fb.stmt(stmt)
        self.prog.add(fb.finish())

    def build_pipeline(self, decl: A.PipelineDecl) -> None:
        info = self.checker.functions[decl.name]
        fb = self.new_fn(decl.name, kind="pipeline", ret=info.fn_type.ret,
                         effects=sorted(info.effects), pos=decl.pos)
        fb.start()
        params = list(decl.params) + [
            A.Param(pos=i.pos, name=i.name, type=i.type) for i in decl.inputs]
        for param, ptype in zip(params, info.fn_type.params):
            slot = fb.fn.new_slot(param.name, ptype, "param")
            fb.fn.params.append(slot)
            fb.fn.param_names.append(param.name)
            fb.bind(param.name, slot)
        for out in decl.outputs:
            ty = self.checker.resolve_type(out.type) if out.type else T.ANY
            slot = fb.fn.new_slot(out.name, ty, "local")
            fb.bind(out.name, slot)
        if decl.body:
            for stmt in decl.body.stmts:
                if isinstance(stmt, A.IODirective):
                    continue
                fb.stmt(stmt)
        self.prog.add(fb.finish())

    def build_model(self, decl: A.ModelDecl) -> None:
        for method in decl.methods:
            short = method.name.split(".")[-1]
            method.name = f"{decl.name}.{short}"
            if method.name not in self.checker.functions:
                self.checker.register_function(method)
            info = self.checker.functions[method.name]
            fb = self.new_fn(method.name, kind="method", ret=info.fn_type.ret,
                             effects=sorted(info.effects | {"model"}),
                             pos=method.pos)
            fb.start()
            for param, ptype in zip(method.params, info.fn_type.params):
                slot = fb.fn.new_slot(param.name, ptype, "param")
                fb.fn.params.append(slot)
                fb.fn.param_names.append(param.name)
                fb.bind(param.name, slot)
            for io_decl in decl.inputs:
                if fb.lookup(io_decl.name) is None:
                    ty = self.checker.resolve_type(io_decl.type) \
                        if io_decl.type else T.ANY
                    slot = fb.fn.new_slot(io_decl.name, ty, "local")
                    fb.bind(io_decl.name, slot)
            if method.body:
                for stmt in method.body.stmts:
                    fb.stmt(stmt)
            self.prog.add(fb.finish())

    def build_service(self, decl: A.ServiceDecl) -> None:
        protect_name = ""
        if decl.protect is not None:
            protect_name = f"{decl.name}$protect"
            fb = self.new_fn(protect_name, kind="protect", ret=T.ANY,
                             effects=("io",), pos=decl.pos)
            fb.start()
            statements = list(decl.protect.stmts)
            for position, stmt in enumerate(statements):
                # The value of a protected region is the value of its last
                # statement, so `publish()` at the end of `protect` is what
                # the service hands back to its caller.
                if position == len(statements) - 1 \
                        and isinstance(stmt, A.ExprStmt) \
                        and stmt.expr is not None:
                    slot = fb.temp(T.ANY)
                    fb.emit(Op.COPY, [fb.expr(stmt.expr)], dst=slot,
                            type_=T.ANY, pos=stmt.pos)
                    fb.emit(Op.RETURN, [fb.slot_op(slot)], pos=stmt.pos)
                else:
                    fb.stmt(stmt)
            self.prog.add(fb.finish())

        steps = [{"action": s.action, "count": s.count, "target": s.target,
                  "raw": s.raw} for s in decl.recover]
        checkpoints = [{"mode": c.mode, "interval": c.interval_seconds,
                        "boundary": c.boundary or "", "raw": c.raw}
                       for c in decl.checkpoints]

        entry = self.new_fn(decl.name, kind="service", ret=T.ANY,
                            effects=("io", "audit"), pos=decl.pos)
        entry.start()
        outcome_slot = entry.temp(T.ANY)
        entry.emit(Op.PROTECTED, [], dst=outcome_slot,
                   meta={"protect": protect_name, "steps": steps,
                         "checkpoints": checkpoints,
                         "audit_all": decl.audit_all,
                         "component": decl.name},
                   pos=decl.pos)
        entry.emit(Op.RETURN, [entry.slot_op(outcome_slot)], pos=decl.pos)
        entry.fn.recovery_points += 1
        if decl.audit_all:
            entry.fn.audit_points += 1
        self.prog.add(entry.finish())

    def build_agent(self, decl: A.AgentDecl) -> None:
        handler_names: Dict[str, str] = {}
        for handler in decl.handlers:
            hname = f"{decl.name}.on.{handler.event}"
            handler_names[handler.event] = hname
            fb = self.new_fn(hname, kind="handler", ret=T.UNIT, pos=handler.pos)
            fb.start()
            slot = fb.fn.new_slot(handler.event, T.ANY, "param")
            fb.fn.params.append(slot)
            fb.fn.param_names.append(handler.event)
            fb.bind(handler.event, slot)
            if handler.body:
                for stmt in handler.body.stmts:
                    fb.stmt(stmt)
            self.prog.add(fb.finish())

        dispatch = self.new_fn(decl.name, kind="agent", ret=T.UNIT,
                               pos=decl.pos)
        dispatch.start()
        ev = dispatch.fn.new_slot("event", T.TEXT, "param")
        pl = dispatch.fn.new_slot("payload", T.ANY, "param")
        dispatch.fn.params.extend([ev, pl])
        dispatch.fn.param_names.extend(["event", "payload"])
        dispatch.bind("event", ev)
        dispatch.bind("payload", pl)
        b_end = dispatch.new_block("agent.end")
        b_nomatch = dispatch.new_block("agent.nomatch")
        # Blocks are created before any of them is filled, so the entry block
        # can be given a terminator and each test can chain to the next one.
        blocks = {event: (dispatch.new_block(f"agent.test.{event}"),
                          dispatch.new_block(f"agent.call.{event}"))
                  for event in handler_names}
        order = list(handler_names)
        first = blocks[order[0]][0] if order else b_end
        dispatch.emit(Op.JUMP, meta={"target": first.id})

        for position, (event, hname) in enumerate(handler_names.items()):
            b_test, b_call = blocks[event]
            following = blocks[order[position + 1]][0] \
                if position + 1 < len(order) else b_nomatch
            dispatch.set_block(b_test)
            cmp_slot = dispatch.temp(T.BOOL)
            dispatch.emit(Op.BINOP,
                          [dispatch.slot_op(ev), dispatch.const(event, T.TEXT)],
                          dst=cmp_slot, type_=T.BOOL, meta={"operator": "=="},
                          pos=decl.pos)
            dispatch.emit(Op.JUMP_IF, [dispatch.slot_op(cmp_slot)],
                          meta={"target": b_call.id,
                                "target_false": following.id},
                          pos=decl.pos)
            dispatch.set_block(b_call)
            dst = dispatch.temp(T.ANY)
            dispatch.emit(Op.CALL, [dispatch.slot_op(pl)], dst=dst,
                          meta={"callee": hname}, pos=decl.pos)
            dispatch.emit(Op.JUMP, meta={"target": b_end.id})

        # Agents communicate by typed messages (spec section 9B), so a
        # message no handler covers is a defect rather than something to
        # drop silently.
        dispatch.set_block(b_nomatch)
        dispatch.emit(Op.FAULT, [],
                      meta={"kind": "UnhandledMessage",
                            "message": f"agent `{decl.name}` received an event "
                                       f"it has no handler for"},
                      pos=decl.pos)
        dispatch.set_block(b_end)
        dispatch.emit(Op.RETURN, [dispatch.const(Const.unit(), T.UNIT)])
        self.prog.add(dispatch.finish())

    def build_fault(self, decl: A.FaultDecl) -> None:
        steps: List[Dict[str, Any]] = []
        for handler in decl.handlers:
            for step in handler.steps:
                steps.append({"action": step.action, "count": step.count,
                              "target": step.target, "raw": step.raw,
                              "event": handler.event})
        fb = self.new_fn(decl.name, kind="fault", ret=T.ANY, pos=decl.pos)
        fb.start()
        fb.emit(Op.PROTECTED, [], dst=fb.temp(T.ANY),
                meta={"protect": "", "steps": steps, "checkpoints": [],
                      "audit_all": True, "component": decl.name},
                pos=decl.pos)
        fb.fn.recovery_points += 1
        self.prog.add(fb.finish())

    def build_policy(self, decl: A.PolicyDecl) -> None:
        fb = self.new_fn(decl.name, kind="policy",
                         ret=T.RecordType("PolicyDecision",
                                          (("allow", T.BOOL),
                                           ("reason", T.TEXT),
                                           ("matched", T.ListType(T.TEXT)),
                                           ("policy", T.TEXT))),
                         effects=("audit",), pos=decl.pos)
        fb.start()
        ctx_slot = fb.fn.new_slot("ctx", T.MapType(T.TEXT, T.ANY), "param")
        fb.fn.params.append(ctx_slot)
        fb.fn.param_names.append("ctx")
        fb.bind("ctx", ctx_slot)
        fb.policy_ctx_slot = ctx_slot

        rule_ops: List[Operand] = []
        for rule in decl.rules:
            if isinstance(rule, A.AuditDirective):
                pairs = [fb.const("kind", T.TEXT), fb.const("audit", T.TEXT),
                         fb.const("text", T.TEXT), fb.const(rule.phrase, T.TEXT)]
            else:
                kind = rule.kind if isinstance(rule, A.PolicyRule) else "require"
                value = (fb.expr(rule.expr)
                         if getattr(rule, "expr", None) is not None
                         else fb.const(True, T.BOOL))
                pairs = [fb.const("kind", T.TEXT), fb.const(kind, T.TEXT),
                         fb.const("value", T.ANY), value,
                         fb.const("text", T.TEXT),
                         fb.const(rule.phrase or kind, T.TEXT)]
            slot = fb.temp(T.MapType(T.TEXT, T.ANY))
            fb.emit(Op.MAKE_MAP, pairs, dst=slot,
                    type_=T.MapType(T.TEXT, T.ANY), pos=rule.pos)
            rule_ops.append(fb.slot_op(slot))
        list_slot = fb.temp(T.ListType(T.ANY))
        fb.emit(Op.MAKE_LIST, rule_ops, dst=list_slot,
                type_=T.ListType(T.ANY), pos=decl.pos)
        out = fb.temp(fb.fn.ret)
        fb.emit(Op.BUILTIN,
                  [fb.const(decl.name, T.TEXT), fb.slot_op(list_slot)],
                  dst=out, meta={"name": "__policy_rules"}, pos=decl.pos)
        fb.emit(Op.RETURN, [fb.slot_op(out)], pos=decl.pos)
        self.prog.add(fb.finish())

    def build_transaction(self, decl: A.TransactionDecl) -> None:
        fb = self.new_fn(decl.name, kind="transaction", ret=T.ANY,
                         effects=("audit", "storage"), pos=decl.pos)
        fb.start()
        tmp = fb.temp(T.UNIT)
        fb.emit(Op.BUILTIN, [fb.const(decl.name, T.TEXT)], dst=tmp,
                meta={"name": "__transaction_begin"}, pos=decl.pos)
        if decl.body:
            for stmt in decl.body.stmts:
                fb.stmt(stmt)
        tmp2 = fb.temp(T.UNIT)
        fb.emit(Op.BUILTIN, [fb.const(decl.name, T.TEXT)], dst=tmp2,
                meta={"name": "__transaction_commit"}, pos=decl.pos)
        fb.emit(Op.RETURN, [fb.const(Const.unit(), T.UNIT)])
        self.prog.add(fb.finish())

    def build_test(self, decl: A.TestDecl) -> None:
        name = f"test:{decl.name}"
        fb = self.new_fn(name, kind="test", ret=T.UNIT, pos=decl.pos)
        fb.start()
        if decl.body:
            for stmt in decl.body.stmts:
                fb.stmt(stmt)
        self.prog.add(fb.finish())
        self.prog.metadata.setdefault("test_categories", {})[name] = decl.category


def build_program(module: A.Module, checker: Checker) -> GProgram:
    return Builder(module, checker).build()
