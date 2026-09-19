"""GIR optimization (spec section 23).

Priority order taken directly from the specification:
    1. Correctness
    2. Security
    3. Determinism when requested
    4. Latency/throughput
    5. Memory efficiency
    6. Code size

That ordering is enforced structurally, not just aspirationally: every pass
here only removes work it can prove has no observable effect.  Operations
that can fault -- division, indexing, field access, casts, anything that can
raise a capability or secret violation -- are never eliminated merely because
their result is unused, and constant folding is abandoned for any expression
that raises.  Security and audit boundaries (``AUDIT``, ``CAP_CHECK``,
``REQUIRE``, ``CHECKPOINT``, ``PROTECTED``) are treated as side-effecting by
construction.

Passes:
    unreachable-block elimination
    dead code after terminators
    constant folding
    copy propagation
    jump threading
    dead store elimination
    elementwise tensor fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..diagnostics import GamaRuntimeFault
from ..runtime import ops
from ..runtime.tensor import GTensor
from ..semantic import types as T
from .ir import GFunction, GProgram, Instr, Op

# Operations whose only effect is producing a value that cannot fault.
# Anything outside this set may raise, mutate state, or cross a security or
# audit boundary, and is therefore never dead-store eliminated.
PURE_VALUE_OPS = {
    Op.CONST, Op.MAKE_LIST, Op.MAKE_SET, Op.MAKE_TUPLE, Op.MAKE_RECORD,
    Op.MAKE_VARIANT, Op.CONSTRUCT,
}

# Side-effecting operations: never removed, never reordered.
EFFECT_OPS = {
    Op.CALL, Op.CALL_INDIRECT, Op.BUILTIN, Op.METHOD_CALL, Op.AUDIT,
    Op.REQUIRE, Op.ASSERT, Op.CONTRACT, Op.CAP_CHECK, Op.CHECKPOINT,
    Op.PARALLEL, Op.PROTECTED, Op.POLICY, Op.AGENT_SEND, Op.SET_FIELD,
    Op.SET_INDEX, Op.STORE_GLOBAL, Op.SECRET_GUARD, Op.TENSOR_OP, Op.FAULT,
    Op.MATCH_FAIL, Op.RETURN, Op.JUMP, Op.JUMP_IF,
}


@dataclass
class PassReport:
    name: str
    changed: int = 0
    detail: str = ""
    enabled: bool = True


@dataclass
class OptimizationReport:
    level: int = 0
    passes: List[PassReport] = field(default_factory=list)
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return (self.before.get("instructions", 0)
                - self.after.get("instructions", 0))

    def summary(self) -> str:
        lines = [f"optimization level: O{self.level}"]
        lines.append(
            f"instructions: {self.before.get('instructions', 0)} -> "
            f"{self.after.get('instructions', 0)} "
            f"({self.removed} eliminated)")
        for p in self.passes:
            if not p.enabled:
                continue
            lines.append(f"  {p.name:34s} {p.changed:4d}  {p.detail}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "before": self.before, "after": self.after,
            "removed": self.removed,
            "passes": [{"name": p.name, "changed": p.changed,
                        "detail": p.detail} for p in self.passes if p.enabled],
        }


class Optimizer:
    def __init__(self, level: int = 1, deterministic: bool = False):
        self.level = level
        self.deterministic = deterministic
        self.report = OptimizationReport(level=level)

    def run(self, program: GProgram) -> OptimizationReport:
        self.report.before = program.stats()
        if self.level <= 0:
            self.report.after = program.stats()
            for name in _PASS_NAMES:
                self.report.passes.append(
                    PassReport(name, 0, "skipped at O0", enabled=False))
            return self.report

        passes = [
            ("unreachable-blocks", self.pass_unreachable),
            ("dead-after-terminator", self.pass_dead_after_terminator),
            ("constant-folding", self.pass_constant_fold),
            ("copy-propagation", self.pass_copy_propagation),
            ("jump-threading", self.pass_jump_threading),
            ("dead-store-elimination", self.pass_dead_stores),
        ]
        if self.level >= 2:
            passes.append(("elementwise-tensor-fusion", self.pass_fusion))

        for _ in range(4):
            changed_any = False
            for name, fn in passes:
                changed = fn(program)
                changed_any = changed_any or changed > 0
                existing = next((p for p in self.report.passes
                                 if p.name == name), None)
                if existing is None:
                    self.report.passes.append(PassReport(name, changed))
                else:
                    existing.changed += changed
            if not changed_any:
                break

        self._prune(program)
        self.report.after = program.stats()
        self._annotate(program)
        return self.report

    # ------------------------------------------------------------------
    def _annotate(self, program: GProgram) -> None:
        for p in self.report.passes:
            if p.name == "constant-folding":
                p.detail = "expressions folded to constants"
            elif p.name == "unreachable-blocks":
                p.detail = "instructions in unreachable blocks removed"
            elif p.name == "dead-after-terminator":
                p.detail = "instructions after terminators removed"
            elif p.name == "copy-propagation":
                p.detail = "copies folded away"
            elif p.name == "jump-threading":
                p.detail = "branches resolved statically"
            elif p.name == "dead-store-elimination":
                p.detail = "unused pure results removed"
            elif p.name == "elementwise-tensor-fusion":
                p.detail = "tensor chains fused into single kernels"

    def _prune(self, program: GProgram) -> None:
        """Physically drop instructions and blocks marked dead."""
        for fn in program.functions.values():
            for block in fn.blocks:
                block.instrs = [i for i in block.instrs if not i.dead]
            live_ids = _reachable_ids(fn)
            fn.blocks = [b for b in fn.blocks if b.id in live_ids]

    # ------------------------------------------------------------------
    def pass_unreachable(self, program: GProgram) -> int:
        removed = 0
        for fn in program.functions.values():
            live = _reachable_ids(fn)
            for block in fn.blocks:
                if block.id in live:
                    continue
                for instr in block.instrs:
                    if not instr.dead:
                        instr.dead = True
                        removed += 1
        return removed

    def pass_dead_after_terminator(self, program: GProgram) -> int:
        removed = 0
        for fn in program.functions.values():
            for block in fn.blocks:
                seen_term = False
                for instr in block.instrs:
                    if seen_term and not instr.dead:
                        instr.dead = True
                        removed += 1
                    if instr.is_terminator:
                        seen_term = True
        return removed

    def pass_constant_fold(self, program: GProgram) -> int:
        folded = 0
        for fn in program.functions.values():
            for block in fn.blocks:
                for instr in block.instrs:
                    if instr.dead:
                        continue
                    if instr.op is Op.BINOP and len(instr.args) == 2:
                        a, b = instr.args
                        if a.kind != "const" or b.kind != "const":
                            continue
                        av, bv = _const_of(a.value), _const_of(b.value)
                        if av is _SKIP or bv is _SKIP:
                            continue
                        try:
                            result = ops.binop(instr.meta["operator"], av, bv)
                        except Exception:                  # noqa: BLE001
                            # Folding must not swallow a fault the program is
                            # entitled to raise at runtime.
                            continue
                        if isinstance(result, (GTensor, list, dict, set)):
                            continue
                        if not _foldable(result):
                            continue
                        instr.op = Op.CONST
                        instr.args = [_const_operand(result)]
                        instr.meta = {"folded": True}
                        folded += 1
                    elif instr.op is Op.UNOP and len(instr.args) == 1:
                        a = instr.args[0]
                        if a.kind != "const":
                            continue
                        av = _const_of(a.value)
                        if av is _SKIP:
                            continue
                        try:
                            result = ops.unop(instr.meta["operator"], av)
                        except Exception:                  # noqa: BLE001
                            continue
                        if not _foldable(result):
                            continue
                        instr.op = Op.CONST
                        instr.args = [_const_operand(result)]
                        instr.meta = {"folded": True}
                        folded += 1
        return folded

    def pass_copy_propagation(self, program: GProgram) -> int:
        changed = 0
        for fn in program.functions.values():
            for block in fn.blocks:
                for instr in block.instrs:
                    if instr.dead or instr.op is not Op.COPY or not instr.args:
                        continue
                    src = instr.args[0]
                    if src.kind == "const":
                        value = _const_of(src.value)
                        if value is _SKIP or not _foldable(value):
                            continue
                        # A copy into a sized-integer slot performs a range
                        # check; turning it into CONST would keep that check
                        # only if the literal is unchanged, which it is.
                        instr.op = Op.CONST
                        instr.args = [_const_operand(value)]
                        changed += 1
                    elif src.kind == "slot" and src.index == instr.dst:
                        instr.dead = True
                        changed += 1
        return changed

    def pass_jump_threading(self, program: GProgram) -> int:
        changed = 0
        for fn in program.functions.values():
            # JUMP -> block whose only instruction is another JUMP.
            for _ in range(3):
                progressed = False
                for block in fn.blocks:
                    term = block.terminator
                    if term is None:
                        continue
                    if term.op is Op.JUMP:
                        target = fn.block(term.meta["target"])
                        if target is not None and len(target.instrs) == 1 \
                                and target.instrs[0].op is Op.JUMP:
                            term.meta["target"] = target.instrs[0].meta["target"]
                            progressed = True
                            changed += 1
                    elif term.op is Op.JUMP_IF and term.args:
                        cond = term.args[0]
                        if cond.kind == "const":
                            value = _const_of(cond.value)
                            if isinstance(value, bool):
                                target = term.meta["target"] if value else \
                                    term.meta.get("target_false")
                                term.op = Op.JUMP
                                term.args = []
                                term.meta = {"target": target, "threaded": True}
                                progressed = True
                                changed += 1
                if not progressed:
                    break
        return changed

    def pass_dead_stores(self, program: GProgram) -> int:
        removed = 0
        for fn in program.functions.values():
            reads = _slot_reads(fn)
            for block in fn.blocks:
                for instr in block.instrs:
                    if instr.dead or instr.dst < 0:
                        continue
                    if instr.op not in PURE_VALUE_OPS:
                        continue
                    if instr.dst in reads:
                        continue
                    if instr.dst >= len(fn.slots):
                        continue
                    slot = fn.slots[instr.dst]
                    # A store into a sized integer slot performs a range
                    # check, which is observable; keep it.
                    if isinstance(slot.type, T.IntType):
                        continue
                    instr.dead = True
                    removed += 1
        return removed

    def pass_fusion(self, program: GProgram) -> int:
        """Fuse chains of elementwise tensor operations (spec 23).

        ``c = (a + b) * k`` over tensors allocates two intermediate tensors
        and walks the data twice.  When each intermediate is used exactly
        once, the chain collapses into one kernel that walks the data once
        and allocates once.
        """
        fused = 0
        for fn in program.functions.values():
            reads = _slot_reads(fn)
            for block in fn.blocks:
                index = 0
                while index < len(block.instrs):
                    instr = block.instrs[index]
                    if instr.dead or instr.op is not Op.BINOP \
                            or not isinstance(instr.type, T.TensorType) \
                            or instr.dst < 0:
                        index += 1
                        continue
                    chain = [instr]
                    slots = {instr.dst}
                    cursor = index + 1
                    while cursor < len(block.instrs):
                        nxt = block.instrs[cursor]
                        if nxt.dead or nxt.op is not Op.BINOP \
                                or not isinstance(nxt.type, T.TensorType) \
                                or nxt.dst < 0:
                            break
                        if not any(a.kind == "slot" and a.index in slots
                                   for a in nxt.args):
                            break
                        if reads.get(chain[-1].dst, 0) != 1:
                            break
                        chain.append(nxt)
                        slots.add(nxt.dst)
                        cursor += 1
                    if len(chain) < 2:
                        index += 1
                        continue
                    steps = []
                    for link in chain:
                        steps.append({
                            "operator": link.meta["operator"],
                            "args": [a.to_dict() for a in link.args],
                            "arg_kinds": [a.kind for a in link.args],
                            "arg_slots": [a.index for a in link.args],
                            "arg_values": [a.value for a in link.args],
                            "dst": link.dst,
                        })
                    head = chain[0]
                    head.op = Op.TENSOR_OP
                    head.meta = {"op": "fused_elementwise", "steps": steps,
                                 "final_dst": chain[-1].dst,
                                 "chain_length": len(chain)}
                    head.dst = chain[-1].dst
                    head.type = chain[-1].type
                    for link in chain[1:]:
                        link.dead = True
                    fused += len(chain) - 1
                    index = cursor
        return fused


class GTensorLike:
    """Marker used to refuse folding tensor constants into GIR."""


def _reachable_ids(fn: GFunction) -> Set[str]:
    live: Set[str] = set()
    stack = [fn.entry]
    while stack:
        bid = stack.pop()
        if bid in live:
            continue
        block = fn.block(bid)
        if block is None:
            continue
        live.add(bid)
        stack.extend(s for s in block.successors() if s)
    return live


def _slot_reads(fn: GFunction) -> Dict[int, int]:
    """Count how many times each slot is read anywhere in the function."""
    reads: Dict[int, int] = {}

    def bump(index: int) -> None:
        reads[index] = reads.get(index, 0) + 1

    for block in fn.blocks:
        for instr in block.instrs:
            if instr.dead:
                continue
            for operand in instr.args:
                if operand.kind == "slot":
                    bump(operand.index)
            # Parallel-region metadata references enclosing slots directly.
            for task in instr.meta.get("tasks", []) or []:
                for key in ("param_slots", "write_slots"):
                    for slot in task.get(key, []) or []:
                        if slot is not None and slot >= 0:
                            bump(slot)
    return reads


_SKIP = object()


def _const_of(value: Any) -> Any:
    """Decode a GIR constant, or _SKIP if it is not a plain value."""
    if type(value) is tuple and value and type(value[0]) is str:
        tag = value[0]
        if tag == "unit":
            return None
        if tag == "duration":
            return _SKIP
        if tag == "variant":
            return _SKIP
        if tag == "default":
            return _SKIP
        return _SKIP
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return _SKIP


def _foldable(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str)) or value is None


def _const_operand(value: Any):
    from .ir import Operand
    return Operand(kind="const", value=value,
                   type=T.BOOL if isinstance(value, bool)
                   else T.I64 if isinstance(value, int)
                   else T.F64 if isinstance(value, float)
                   else T.TEXT if isinstance(value, str) else T.UNIT)


_PASS_NAMES = [
    "unreachable-blocks", "dead-after-terminator", "constant-folding",
    "copy-propagation", "jump-threading", "dead-store-elimination",
    "elementwise-tensor-fusion",
]


def optimize(program: GProgram, level: int = 1,
             deterministic: bool = False) -> OptimizationReport:
    return Optimizer(level, deterministic).run(program)
