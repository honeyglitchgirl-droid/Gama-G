"""Gama IR (GIR) -- the portable intermediate representation (spec 20, 21).

GIR is the stable representation between front-end language semantics and
machine-specific code generation.  Spec section 21 requires it to encode
typed operations, memory ownership, effects, capabilities, parallel regions,
tensor operations, recovery points, audit boundaries and deterministic
execution constraints -- and to have a versioned specification, which is
what ``GIR_VERSION`` and the ``version`` field on every program provide.

Design
------
GIR is a register machine over *slots* with an explicit control-flow graph of
basic blocks.  Slots are not in SSA form: a mutable variable keeps one slot
for its whole lifetime, which keeps the IR small and makes the reference
interpreter straightforward while still exposing a real CFG to the
optimizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..diagnostics import SourcePos
from ..semantic import types as T

GIR_VERSION = "1.0"


class Op(str, Enum):
    # data movement
    CONST = "const"
    COPY = "copy"
    LOAD_GLOBAL = "load_global"
    STORE_GLOBAL = "store_global"

    # arithmetic and logic
    BINOP = "binop"
    UNOP = "unop"
    CAST = "cast"

    # calls
    CALL = "call"                    # a function in this program
    CALL_INDIRECT = "call_indirect"  # a function held in a slot
    BUILTIN = "builtin"              # a standard-library function
    METHOD_CALL = "method_call"      # obj.method(args), resolved at runtime

    # aggregate construction
    MAKE_LIST = "make_list"
    MAKE_MAP = "make_map"
    MAKE_SET = "make_set"
    MAKE_TUPLE = "make_tuple"
    MAKE_RECORD = "make_record"
    MAKE_VARIANT = "make_variant"
    CONSTRUCT = "construct"          # ok/fail/some/none

    # aggregate access
    FIELD = "field"
    SET_FIELD = "set_field"
    INDEX = "index"
    SET_INDEX = "set_index"

    # tensor operations (spec section 14)
    TENSOR_OP = "tensor_op"

    # control flow
    JUMP = "jump"
    JUMP_IF = "jump_if"
    RETURN = "return"
    FAULT = "fault"
    MATCH_FAIL = "match_fail"

    # language-level boundaries
    AUDIT = "audit"                  # audit boundary (spec section 13)
    REQUIRE = "require"              # runtime contract (spec sections 15, 27)
    ASSERT = "assert"
    CONTRACT = "contract"            # requires/ensures check
    CAP_CHECK = "cap_check"          # capability boundary (spec section 12)
    CHECKPOINT = "checkpoint"        # recovery point (spec section 11)
    PARALLEL = "parallel"            # parallel region (spec sections 3, 9)
    PROTECTED = "protected"          # protect/recover region (spec section 10)
    POLICY = "policy"                # policy decision (spec section 17)
    AGENT_SEND = "agent_send"        # typed message to an agent (spec 9B)
    SECRET_GUARD = "secret_guard"    # secret lifecycle boundary (spec 8)
    TRANSACTION = "transaction"      # spec section 18
    FOR_STATEMENT = "for_statement"


TERMINATORS = {Op.JUMP, Op.JUMP_IF, Op.RETURN, Op.FAULT, Op.MATCH_FAIL}


@dataclass
class Operand:
    """An instruction operand: a slot, an immediate, a global or a builtin."""

    kind: str = "slot"           # slot | const | global | builtin
    index: int = -1
    value: Any = None
    name: str = ""
    type: T.Type = T.ANY

    def render(self, fn: Optional["GFunction"] = None) -> str:
        if self.kind == "slot":
            if fn is not None and 0 <= self.index < len(fn.slots):
                return f"%{fn.slots[self.index].name}"
            return f"%{self.index}"
        if self.kind == "const":
            return _render_const(self.value)
        if self.kind == "global":
            return f"@{self.name}"
        if self.kind == "builtin":
            return f"#{self.name}"
        return "?"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "type": self.type.render()}
        if self.kind == "slot":
            out["index"] = self.index
        elif self.kind == "const":
            out["value"] = self.value
        else:
            out["name"] = self.name
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Operand":
        return Operand(kind=d["kind"], index=d.get("index", -1),
                       value=d.get("value"), name=d.get("name", ""),
                       type=T.ANY)


def _render_const(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "()"
    return str(value)


@dataclass
class Instr:
    op: Op
    args: List[Operand] = field(default_factory=list)
    dst: int = -1
    type: T.Type = T.ANY
    meta: Dict[str, Any] = field(default_factory=dict)
    pos: Optional[SourcePos] = None
    id: int = -1
    # Filled in by the optimizer.
    dead: bool = False

    @property
    def is_terminator(self) -> bool:
        return self.op in TERMINATORS

    def render(self, fn: Optional["GFunction"] = None) -> str:
        parts = [self.op.value]
        target = ""
        if self.dst >= 0 and self.op not in TERMINATORS:
            slot = fn.slots[self.dst].name if fn and self.dst < len(fn.slots) \
                else str(self.dst)
            target = f"%{slot} = "
        extra = ""
        if self.op is Op.BINOP or self.op is Op.UNOP:
            extra = f" {self.meta.get('operator', '')}"
        elif self.op is Op.CONST:
            extra = ""
        for key in ("target", "target_false", "callee", "name", "method",
                    "tag", "record", "fields", "action", "message"):
            if key in self.meta:
                value = self.meta[key]
                extra += f" {key}={value!r}" if not isinstance(value, str) \
                    else f" {key}={value}"
        args = ", ".join(a.render(fn) for a in self.args)
        return f"{target}{parts[0]}{extra} {args}".rstrip()

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "op": self.op.value,
            "args": [a.to_dict() for a in self.args],
            "type": self.type.render(),
        }
        if self.dst >= 0:
            out["dst"] = self.dst
        if self.meta:
            out["meta"] = _jsonable(self.meta)
        if self.pos:
            out["pos"] = {"line": self.pos.line, "col": self.pos.col,
                          "file": self.pos.file}
        return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, T.Type):
        return value.render()
    if isinstance(value, SourcePos):
        return {"line": value.line, "col": value.col, "file": value.file}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass
class Slot:
    index: int
    name: str
    type: T.Type = T.ANY
    kind: str = "local"        # param | local | temp
    mutable: bool = False
    secret: bool = False


@dataclass
class BasicBlock:
    id: str
    instrs: List[Instr] = field(default_factory=list)
    label: str = ""

    @property
    def terminator(self) -> Optional[Instr]:
        return self.instrs[-1] if self.instrs and self.instrs[-1].is_terminator \
            else None

    def successors(self) -> List[str]:
        term = self.terminator
        if term is None:
            return []
        if term.op is Op.JUMP:
            return [term.meta["target"]]
        if term.op is Op.JUMP_IF:
            return [term.meta["target"], term.meta.get("target_false", "")]
        return []

    def render(self, fn: "GFunction") -> List[str]:
        head = f"{self.id}:" + (f"    ; {self.label}" if self.label else "")
        lines = [head]
        for instr in self.instrs:
            mark = "  " if not instr.dead else "# "
            lines.append(f"    {mark}{instr.render(fn)}")
        return lines


@dataclass
class ParallelTask:
    """One task inside a parallel region (spec sections 3 and 9)."""

    name: str
    function: str
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class RecoveryPlan:
    """A bounded recovery policy attached to a protected region (spec 10)."""

    steps: List[Dict[str, Any]] = field(default_factory=list)
    audit_all: bool = False
    level_names: Dict[int, str] = field(default_factory=dict)


@dataclass
class GFunction:
    name: str
    slots: List[Slot] = field(default_factory=list)
    blocks: List[BasicBlock] = field(default_factory=list)
    entry: str = "b0"
    params: List[int] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    ret: T.Type = T.UNIT
    effects: Tuple[str, ...] = ()
    caps: Tuple[str, ...] = ()
    deterministic: bool = False
    kind: str = "fn"
    contracts: List[Dict[str, Any]] = field(default_factory=list)
    secret_params: List[int] = field(default_factory=list)
    # Operation-graph metadata (spec section 3).
    parallel_tasks: List[ParallelTask] = field(default_factory=list)
    recovery: Optional[RecoveryPlan] = None
    audit_points: int = 0
    recovery_points: int = 0
    # boundaries the lowering made explicit, counted so that a tool can tell
    # whether a program has any at all rather than inferring it from the body
    secret_points: int = 0
    cap_points: int = 0

    def block(self, bid: str) -> Optional[BasicBlock]:
        for b in self.blocks:
            if b.id == bid:
                return b
        return None

    def new_slot(self, name: str, type_: T.Type = T.ANY, kind: str = "temp",
                 mutable: bool = False, secret: bool = False) -> int:
        index = len(self.slots)
        self.slots.append(Slot(index, name, type_, kind, mutable, secret))
        return index

    def slot_index(self, name: str) -> Optional[int]:
        for s in reversed(self.slots):
            if s.name == name:
                return s.index
        return None

    def render(self) -> str:
        header = [
            f"function {self.name} [{self.kind}]",
            f"  params   : " + ", ".join(
                f"%{self.slots[i].name}: {self.slots[i].type.render()}"
                for i in self.params),
            f"  returns  : {self.ret.render()}",
            f"  effects  : {', '.join(self.effects) or 'none'}",
            f"  caps     : {', '.join(self.caps) or 'none'}",
            f"  slots    : {len(self.slots)}",
            f"  blocks   : {len(self.blocks)}",
        ]
        if self.deterministic:
            header.append("  deterministic: yes")
        if self.contracts:
            header.append("  contracts: " + ", ".join(
                c["kind"] for c in self.contracts))
        if self.parallel_tasks:
            header.append(f"  operation graph: {len(self.parallel_tasks)} task(s)")
            for t in self.parallel_tasks:
                deps = ",".join(t.depends_on) or "-"
                header.append(f"    - {t.name}: reads={t.reads} "
                              f"writes={t.writes} depends_on={deps}")
        if self.recovery:
            header.append(f"  recovery plan: {len(self.recovery.steps)} step(s)")
        if self.cap_points:
            header.append(f"  capability boundaries: {self.cap_points}")
        if self.secret_points:
            header.append(f"  secret boundaries: {self.secret_points}")
        body: List[str] = []
        for b in self.blocks:
            body.extend(b.render(self))
        return "\n".join(header + body)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "entry": self.entry,
            "params": self.params, "param_names": self.param_names,
            "ret": self.ret.render(), "effects": list(self.effects),
            "caps": list(self.caps), "deterministic": self.deterministic,
            "contracts": self.contracts, "secret_params": self.secret_params,
            "slots": [{"index": s.index, "name": s.name, "type": s.type.render(),
                       "kind": s.kind, "mutable": s.mutable, "secret": s.secret}
                      for s in self.slots],
            "blocks": [{"id": b.id, "label": b.label,
                        "instrs": [i.to_dict() for i in b.instrs]}
                       for b in self.blocks],
            "parallel_tasks": [
                {"name": t.name, "function": t.function, "reads": t.reads,
                 "writes": t.writes, "depends_on": t.depends_on}
                for t in self.parallel_tasks],
            "recovery": ({"steps": self.recovery.steps,
                          "audit_all": self.recovery.audit_all}
                         if self.recovery else None),
            "audit_points": self.audit_points,
            "recovery_points": self.recovery_points,
        }


@dataclass
class GProgram:
    """A whole compiled module in GIR form."""

    version: str = GIR_VERSION
    name: str = "main"
    functions: Dict[str, GFunction] = field(default_factory=dict)
    entry: str = "<main>"
    grants: List[str] = field(default_factory=list)
    records: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    enums: Dict[str, List[str]] = field(default_factory=dict)
    global_bindings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tests: List[str] = field(default_factory=list)

    def add(self, fn: GFunction) -> GFunction:
        # Overwriting a function would silently redirect every call to it, so
        # a duplicate name is a compiler defect rather than something to
        # absorb.  Synthetic names (parallel tasks, protect bodies) are
        # namespaced by their producer.
        existing = self.functions.get(fn.name)
        if existing is not None and existing is not fn:
            raise AssertionError(
                f"GIR already contains a function named `{fn.name}`; "
                f"a later definition would silently replace it")
        self.functions[fn.name] = fn
        return fn

    def stats(self) -> Dict[str, Any]:
        instrs = sum(len(b.instrs) for f in self.functions.values()
                     for b in f.blocks)
        blocks = sum(len(f.blocks) for f in self.functions.values())
        live = sum(1 for f in self.functions.values() for b in f.blocks
                   for i in b.instrs if not i.dead)
        by_op: Dict[str, int] = {}
        for f in self.functions.values():
            for b in f.blocks:
                for i in b.instrs:
                    by_op[i.op.value] = by_op.get(i.op.value, 0) + 1
        return {
            "gir_version": self.version,
            "functions": len(self.functions),
            "blocks": blocks,
            "instructions": instrs,
            "live_instructions": live,
            "dead_instructions": instrs - live,
            "slots": sum(len(f.slots) for f in self.functions.values()),
            "audit_points": sum(f.audit_points for f in self.functions.values()),
            "recovery_points": sum(f.recovery_points
                                   for f in self.functions.values()),
            "parallel_tasks": sum(len(f.parallel_tasks)
                                  for f in self.functions.values()),
            "by_opcode": dict(sorted(by_op.items())),
        }

    def render(self, *, only: Optional[str] = None) -> str:
        parts = [
            f"; Gama IR (GIR) {self.version}",
            f"; module {self.name}",
            f"; grants: {', '.join(self.grants) or 'none'}",
        ]
        if self.records:
            parts.append("; records: " + ", ".join(
                f"{n}({len(f)})" for n, f in self.records.items()))
        if self.enums:
            parts.append("; enums: " + ", ".join(
                f"{n}[{len(v)}]" for n, v in self.enums.items()))
        parts.append("")
        order = [self.entry] if self.entry in self.functions else []
        order += [n for n in self.functions if n not in order]
        for name in order:
            if only and name != only:
                continue
            parts.append(self.functions[name].render())
            parts.append("")
        return "\n".join(parts)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "gir_version": self.version,
            "module": self.name,
            "entry": self.entry,
            "grants": self.grants,
            "records": self.records,
            "enums": self.enums,
            "global_bindings": self.global_bindings,
            "metadata": _jsonable(self.metadata),
            "tests": self.tests,
            "functions": {n: f.to_dict() for n, f in self.functions.items()},
            "stats": self.stats(),
        }, indent=indent, sort_keys=False, default=str)
