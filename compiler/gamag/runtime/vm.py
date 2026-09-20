"""The Gama Abstract Execution Model (GAEM) reference interpreter.

This is the "interpreter/reference backend" of spec section 20: it executes
optimised GIR directly.  Source semantics live above the hardware layer, so
nothing here depends on a particular CPU -- the same GIR could be handed to a
native or WebAssembly backend instead.

Runtime guarantees enforced here rather than only in the checker:
  * capability checks before any gated operation (spec section 12);
  * integer range checks for the sized integer types (spec section 5);
  * secret lifecycle barriers (spec section 8);
  * contract checks for ``requires``/``ensures`` (spec section 27);
  * bounded recovery with checkpoint restore (spec sections 10, 11);
  * audit emission at every audit boundary (spec section 13);
  * a step limit, so a runaway program cannot hang the toolchain.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

from ..diagnostics import (ContractViolation, CapabilityViolation,
                           GamaRuntimeFault, SecretLeak, SourcePos, TypeFault)
from ..nesting import vm_depth_limit
from ..gir.ir import GFunction, GProgram, Instr, Op
from ..semantic import types as T
from ..std import library as L
from .checkpoint import CheckpointRejected
from . import ops
from .context import Context
from .ops import default_for, values_equal
from .recovery import RecoveryEngine, RecoveryStepSpec
from .tensor import GTensor
from .values import (GCapability, GComponent, GDuration, GFunction as GFnValue,
                     GInstant, GOption, GRecord, GResult, GSecret, GUnit,
                     GUri, GUuid, GVariant, UNIT, canonical, display,
                     deep_copy_value, to_text, truthy, type_name)


class GModuleRef:
    """A standard-library namespace value, e.g. the ``math`` in ``math.pi``."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"<module {self.name}>"


class GBuiltinRef:
    """A first-class standard-library function value."""

    __slots__ = ("name", "builtin")

    def __init__(self, name: str, builtin: L.Builtin):
        self.name = name
        self.builtin = builtin

    def __repr__(self) -> str:
        return f"<builtin {self.name}>"


class _Jump(Exception):
    pass


class Frame:
    __slots__ = ("fn", "slots", "depth")

    def __init__(self, fn: GFunction, depth: int):
        self.fn = fn
        self.slots: List[Any] = [None] * len(fn.slots)
        self.depth = depth


# Method dispatch tables: `obj.method(args)` resolved by the runtime type of
# `obj`.  Keeping this explicit means an unknown method produces a helpful
# diagnostic listing what *is* available.
TENSOR_NATIVE = {
    "add", "sub", "mul", "div", "matmul", "dot", "relu", "sigmoid", "tanh",
    "exp", "neg", "softmax", "sum", "mean", "max", "min", "argmax",
    "reshape", "transpose", "clip", "allclose", "broadcast", "at", "copy",
    "flat",
}
TEXT_METHODS = {
    "upper": "text.upper", "lower": "text.lower", "split": "text.split",
    "trim": "text.trim", "contains": "text.contains",
    "starts_with": "text.starts_with", "ends_with": "text.ends_with",
    "replace": "text.replace", "length": "text.length",
    "char_at": "text.char_at", "slice": "text.slice", "repeat": "text.repeat",
    "index_of": "text.index_of", "is_empty": "text.is_empty",
    "lines": "text.lines", "pad_start": "text.pad_start",
    "pad_end": "text.pad_end", "format": "text.format",
    "parse_int": "text.parse_int", "parse_float": "text.parse_float",
    "join": "text.join",
}
from ..methods import (  # single source of truth, shared with the checker
    LIST_METHODS, MAP_METHODS, OPTION_METHODS, RESULT_METHODS, SET_METHODS,
    TENSOR_NATIVE, TEXT_METHODS,
)


class VM:
    #: Policy ceiling on Gama-G call depth.  The *effective* limit is lower
    #: whenever the host stack cannot carry this many frames, because the
    #: interpreter is recursive: each Gama-G call costs about
    #: ``nesting.FRAMES_PER_VM_CALL`` Python frames, and exhausting the host
    #: stack is a crash rather than a fault.  Raising the host's recursion
    #: limit raises this back towards the ceiling.
    MAX_DEPTH = 1500
    CHECKPOINT_POLL_INSTRS = 512

    def __init__(self, program: GProgram, context: Optional[Context] = None,
                 checker: Any = None):
        self.prog = program
        self.ctx = context or Context()
        self.checker = checker
        self.globals: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.depth = 0
        # Derived once, at construction, because the budget depends on how
        # much host stack is already in use and that only grows from here.
        self.max_depth = vm_depth_limit(self.MAX_DEPTH)
        self._armed_checkpoint: Optional[float] = None
        self._since_poll = 0
        # Whether the `<main>` module initializer has run yet.
        self._initialized = False
        self.dispatch = {
            Op.CONST: self.op_const, Op.COPY: self.op_copy,
            Op.LOAD_GLOBAL: self.op_load_global,
            Op.STORE_GLOBAL: self.op_store_global,
            Op.BINOP: self.op_binop, Op.UNOP: self.op_unop, Op.CAST: self.op_cast,
            Op.CALL: self.op_call, Op.CALL_INDIRECT: self.op_call_indirect,
            Op.BUILTIN: self.op_builtin, Op.METHOD_CALL: self.op_method_call,
            Op.MAKE_LIST: self.op_make_list, Op.MAKE_MAP: self.op_make_map,
            Op.MAKE_SET: self.op_make_set, Op.MAKE_TUPLE: self.op_make_tuple,
            Op.MAKE_RECORD: self.op_make_record,
            Op.MAKE_VARIANT: self.op_make_variant,
            Op.CONSTRUCT: self.op_construct,
            Op.FIELD: self.op_field, Op.SET_FIELD: self.op_set_field,
            Op.INDEX: self.op_index, Op.SET_INDEX: self.op_set_index,
            Op.TENSOR_OP: self.op_tensor_op,
            Op.AUDIT: self.op_audit, Op.REQUIRE: self.op_require,
            Op.ASSERT: self.op_assert, Op.CONTRACT: self.op_contract,
            Op.CAP_CHECK: self.op_cap_check,
            Op.CHECKPOINT: self.op_checkpoint,
            Op.PARALLEL: self.op_parallel, Op.PROTECTED: self.op_protected,
            Op.POLICY: self.op_policy, Op.AGENT_SEND: self.op_agent_send,
            Op.SECRET_GUARD: self.op_secret_guard,
            Op.TRANSACTION: self.op_transaction,
        }
        self.ctx._call_hook = self.call_value
        self.ctx.state_provider = self.capture_state
        self.ctx.state_applier = self.restore_state
        self._install_globals()

    # ------------------------------------------------------------------
    # global installation
    # ------------------------------------------------------------------
    def _install_globals(self) -> None:
        for name, fn in self.prog.functions.items():
            if fn.kind in ("fn", "pipeline", "method", "service", "agent",
                           "policy", "transaction", "test", "protect",
                           "handler"):
                self.globals[name] = GFnValue(name=name, gir=fn)
        for module in L.MODULE_TYPE_NAMES:
            self.globals[module] = GModuleRef(module)
        for name, constant in L.CONSTANTS.items():
            short = name.split(".")[-1]
            self.globals.setdefault(short, constant.value)

        if self.checker is not None:
            for enum_name, enum in self.checker.enums.items():
                for variant in enum.variants:
                    if variant.params:
                        continue
                    self.globals.setdefault(variant.name,
                                            GVariant(variant.name, enum_name))
            for name, model in self.checker.models.items():
                methods = {}
                for m in model.methods:
                    short = m.name.split(".")[-1]
                    full = f"{name}.{short}"
                    if full in self.prog.functions:
                        methods[short] = full
                self.globals[name] = GComponent(
                    kind="model", name=name, methods=methods,
                    meta={"inputs": [(i.name, i.type.render() if i.type else "Any")
                                     for i in model.inputs],
                          "outputs": [(o.name, o.type.render() if o.type else "Any")
                                      for o in model.outputs]})
            for name, svc in self.checker.services.items():
                if name == "<module>":
                    continue
                methods = {"run": name, "start": name}
                if f"{name}$protect" in self.prog.functions:
                    methods["protect"] = f"{name}$protect"
                self.globals[name] = GComponent(
                    kind="service", name=name, methods=methods,
                    meta={"recovery": [s.raw for s in svc.recover],
                          "checkpoints": [c.raw for c in svc.checkpoints],
                          "audit_all": svc.audit_all})
            for name, agent in self.checker.agents.items():
                methods = {"send": name, "dispatch": name}
                for h in agent.handlers:
                    full = f"{name}.on.{h.event}"
                    if full in self.prog.functions:
                        methods[h.event] = full
                self.globals[name] = GComponent(kind="agent", name=name,
                                                methods=methods)
            for name, policy in self.checker.policies.items():
                self.globals[name] = GComponent(kind="policy", name=name,
                                                methods={"evaluate": name,
                                                         "check": name})
            for name, tx in self.checker.transactions.items():
                self.globals[name] = GComponent(kind="transaction", name=name,
                                                methods={"run": name,
                                                         "commit": name})
            for name, fault in self.checker.faults.items():
                self.globals[name] = GComponent(kind="fault", name=name,
                                                methods={"run": name})

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        """Run the module initializer exactly once.

        Top-level bindings are lowered into a synthetic `<main>` function, so
        it must execute before any user entry point reads them.
        """
        if self._initialized:
            return
        self._initialized = True
        init = self.prog.functions.get("<main>")
        if init is not None:
            self.execute(init, [])

    def run(self, entry: str = "<main>", args: Sequence[Any] = ()) -> Any:
        if entry == "<main>":
            self._initialized = True
        else:
            self._ensure_initialized()
        fn = self.prog.functions.get(entry)
        if fn is None:
            raise GamaRuntimeFault("NoEntryPoint",
                                   f"the program has no `{entry}` function")
        return self.execute(fn, list(args))

    def call_function(self, name: str, args: Sequence[Any],
                      pos: Optional[SourcePos] = None) -> Any:
        fn = self.prog.functions.get(name)
        if fn is None:
            if name in self.globals and isinstance(self.globals[name], GFnValue):
                fn = self.globals[name].gir
            else:
                raise GamaRuntimeFault(
                    "UnknownFunction", f"no GIR function named `{name}`", pos)
        return self.execute(fn, list(args), pos)

    def call_value(self, value: Any, args: Sequence[Any]) -> Any:
        if isinstance(value, GFnValue):
            return self.execute(value.gir, list(args))
        if isinstance(value, GBuiltinRef):
            return self.invoke_builtin(value.builtin, list(args))
        if isinstance(value, GComponent):
            target = value.methods.get("run") or value.methods.get("evaluate") \
                or value.methods.get("send") or value.name
            return self.call_function(target, list(args))
        if isinstance(value, GVariant):
            return value
        raise TypeFault(
            f"{type_name(value)} is not callable",
            hint="only functions, pipelines, models, services, policies and "
                 "standard-library builtins can be called")

    def invoke_builtin(self, b: L.Builtin, args: List[Any]) -> Any:
        for cap in b.caps:
            self.ctx.require_capability(cap, what=b.name)
        return b.impl(self.ctx, *args)

    # ------------------------------------------------------------------
    def execute(self, fn: GFunction, args: List[Any],
                pos: Optional[SourcePos] = None) -> Any:
        with self.lock:
            self.depth += 1
        if self.depth > self.max_depth:
            self.depth -= 1
            raise GamaRuntimeFault(
                "StackOverflow",
                f"call depth exceeded {self.max_depth} frames",
                pos,
                context={"function": fn.name, "limit": self.max_depth},
                hint=("the limit is what this host's stack carries, not a "
                      "fixed language rule; rewrite the recursion "
                      "iteratively, or raise sys.setrecursionlimit"))
        frame = Frame(fn, self.depth)
        try:
            for i, slot in enumerate(fn.params):
                value = args[i] if i < len(args) else default_for(fn.slots[slot].type)
                self.store(frame, slot, value)
            if len(args) > len(fn.params):
                raise TypeFault(
                    f"`{fn.name}` takes {len(fn.params)} argument(s) but "
                    f"received {len(args)}", pos)
            self.ctx.stats.calls += 1
            bid = fn.entry
            steps = 0
            try:
                while bid is not None:
                    block = fn.block(bid)
                    if block is None:
                        raise GamaRuntimeFault(
                            "BadGIR", f"block `{bid}` is missing from `{fn.name}`")
                    next_bid, value, returned = self.run_block(frame, block)
                    if returned:
                        return self.coerce_return(value, fn)
                    bid = next_bid
                    steps += 1
                    if steps > 5_000_000:
                        raise GamaRuntimeFault(
                            "RunawayLoop",
                            f"`{fn.name}` exceeded the basic-block step budget")
                return UNIT
            except RecursionError:
                # The bound above is derived from the host stack and should
                # always fire first.  It cannot when the stack is shallower
                # than the host reports -- a worker thread, an embedder with a
                # deep stack of its own -- so the overflow is translated here
                # rather than escaping as an internal error.
                raise GamaRuntimeFault(
                    "StackOverflow",
                    f"call depth exceeded the {self.max_depth} frames this "
                    f"host stack can carry",
                    pos, context={"function": fn.name},
                    hint="rewrite the recursion iteratively, or run on a "
                         "host with a deeper stack")
            except GamaRuntimeFault as fault:
                # Spec section 18: a transaction that fails before it commits
                # must not leave its effects half applied.  Emitting
                # begin ... body ... commit leaves no branch to hang an abort on,
                # so this is the only place a failed transaction can be marked
                # aborted; otherwise it would stay "open" forever and the audit
                # trail would show a begin with no matching outcome.
                active = self.ctx.active_transaction
                if active is not None and self.ctx.transaction_owner == fn.name:
                    # the function that opened the still-active transaction is the
                    # one whose fault closes it
                    self.ctx.abort_transaction(
                        active, f"{fault.kind}: {fault.message}")
                    self.ctx.active_transaction = None
                    self.ctx.transaction_owner = None
                elif fn.kind == "transaction":
                    self.ctx.abort_transaction(
                        fn.name, f"{fault.kind}: {fault.message}")
                raise
        finally:
            with self.lock:
                self.depth -= 1

    def run_block(self, frame: Frame, block) -> Tuple[Optional[str], Any, bool]:
        for instr in block.instrs:
            if instr.dead:
                continue
            self.ctx.step()
            self._since_poll += 1
            if self._since_poll >= self.CHECKPOINT_POLL_INSTRS:
                self._since_poll = 0
                self._poll_armed_checkpoint()
            op = instr.op
            if op is Op.JUMP:
                return instr.meta["target"], None, False
            if op is Op.JUMP_IF:
                cond = truthy(self.resolve(frame, instr.args[0]))
                target = instr.meta["target"] if cond else \
                    instr.meta.get("target_false")
                return target, None, False
            if op is Op.RETURN:
                value = self.resolve(frame, instr.args[0]) if instr.args else UNIT
                return None, value, True
            if op is Op.FAULT:
                raise GamaRuntimeFault(
                    str(instr.meta.get("kind", "Fault")),
                    str(instr.meta.get("message", "")), instr.pos)
            if op is Op.MATCH_FAIL:
                value = self.resolve(frame, instr.args[0]) if instr.args else None
                raise GamaRuntimeFault(
                    "MatchError",
                    f"no match arm accepted a value of type {type_name(value)}",
                    instr.pos, context={"value": display(value)})
            handler = self.dispatch.get(op)
            if handler is None:
                raise GamaRuntimeFault("UnsupportedGIR",
                                       f"the reference interpreter cannot "
                                       f"execute `{op.value}`", instr.pos)
            handler(frame, instr)
        return None, UNIT, True

    # ------------------------------------------------------------------
    # operands and slots
    # ------------------------------------------------------------------
    def resolve(self, frame: Frame, operand) -> Any:
        kind = operand.kind
        if kind == "slot":
            value = frame.slots[operand.index]
            if value is None:
                slot = frame.fn.slots[operand.index]
                value = default_for(slot.type)
                frame.slots[operand.index] = value
            return value
        if kind == "const":
            return self.const_value(operand.value)
        if kind == "global":
            if operand.name not in self.globals:
                raise GamaRuntimeFault(
                    "UnboundGlobal",
                    f"`{operand.name}` is not bound; module-level bindings are "
                    f"initialised in source order, so a function cannot read "
                    f"one before its declaration has run",
                    None, context={"name": operand.name})
            return self.globals[operand.name]
        if kind == "builtin":
            b = L.BUILTINS.get(operand.name)
            if b is None:
                raise GamaRuntimeFault("UnknownBuiltin",
                                       f"no builtin named `{operand.name}`")
            return GBuiltinRef(operand.name, b)
        raise GamaRuntimeFault("BadGIR", f"unknown operand kind `{kind}`")

    def const_value(self, value: Any) -> Any:
        if type(value) is tuple and value and type(value[0]) is str:
            tag = value[0]
            if tag == "unit" and len(value) == 1:
                return UNIT
            if tag == "duration" and len(value) == 3:
                return GDuration(float(value[1]))
            if tag == "default" and len(value) == 2:
                return default_for(value[1])
            if tag == "variant" and len(value) == 3:
                return GVariant(value[1], value[2])
        return value

    def store(self, frame: Frame, slot: int, value: Any) -> None:
        ty = frame.fn.slots[slot].type
        if isinstance(ty, T.IntType) and isinstance(value, int) \
                and not isinstance(value, bool):
            lo, hi = ty.range
            if not lo <= value <= hi:
                raise GamaRuntimeFault(
                    "IntegerOverflow",
                    f"value {value} does not fit in {ty.render()} "
                    f"(range {lo} to {hi})",
                    context={"slot": frame.fn.slots[slot].name,
                             "type": ty.render(), "value": value})
        if isinstance(ty, T.SecretType) and not isinstance(value, GSecret):
            value = GSecret(value, frame.fn.slots[slot].name)
        frame.slots[slot] = value

    def coerce_return(self, value: Any, fn: GFunction) -> Any:
        if isinstance(fn.ret, T.IntType) and isinstance(value, int) \
                and not isinstance(value, bool):
            lo, hi = fn.ret.range
            if not lo <= value <= hi:
                raise GamaRuntimeFault(
                    "IntegerOverflow",
                    f"`{fn.name}` returned {value}, which does not fit in "
                    f"{fn.ret.render()}")
        return value

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------
    def capture_state(self) -> Dict[str, Any]:
        """The state a checkpoint records: module-level bindings."""
        out: Dict[str, Any] = {}
        for name, value in self.globals.items():
            if isinstance(value, (GFnValue, GModuleRef, GBuiltinRef,
                                  GComponent)):
                continue
            out[name] = deep_copy_value(value)
        return out

    def restore_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        for name, value in state.items():
            self.globals[name] = value
            applied[name] = value
        return applied

    def _poll_armed_checkpoint(self) -> None:
        if self._armed_checkpoint is None or self.ctx.deterministic:
            return
        if time.time() >= self._armed_checkpoint:
            interval = self._checkpoint_interval or 5.0
            self.ctx.capture_checkpoint("periodic")
            self._armed_checkpoint = time.time() + interval

    _checkpoint_interval: Optional[float] = None

    # ------------------------------------------------------------------
    # instruction implementations
    # ------------------------------------------------------------------
    def op_const(self, frame: Frame, instr: Instr) -> None:
        self.store(frame, instr.dst, self.const_value(instr.args[0].value))

    def op_copy(self, frame: Frame, instr: Instr) -> None:
        self.store(frame, instr.dst, self.resolve(frame, instr.args[0]))

    def op_load_global(self, frame: Frame, instr: Instr) -> None:
        self.store(frame, instr.dst, self.globals[instr.meta["name"]])

    def op_store_global(self, frame: Frame, instr: Instr) -> None:
        name = instr.meta["name"]
        value = self.resolve(frame, instr.args[0])
        # A top-level `secret` binding stays secret wherever it is read from.
        if instr.meta.get("secret") and not isinstance(value, GSecret):
            value = GSecret(value, name)
        self.globals[name] = value

    def op_binop(self, frame: Frame, instr: Instr) -> None:
        a = self.resolve(frame, instr.args[0])
        b = self.resolve(frame, instr.args[1])
        self.store(frame, instr.dst,
                   self.binop(instr.meta["operator"], a, b, instr.pos))

    def binop(self, op: str, a: Any, b: Any,
              pos: Optional[SourcePos] = None) -> Any:
        return ops.binop(op, a, b, pos)

    def op_unop(self, frame: Frame, instr: Instr) -> None:
        value = self.resolve(frame, instr.args[0])
        self.store(frame, instr.dst,
                   ops.unop(instr.meta["operator"], value, instr.pos))

    def op_cast(self, frame: Frame, instr: Instr) -> None:
        value = self.resolve(frame, instr.args[0])
        target = instr.type
        if isinstance(value, GSecret):
            raise SecretLeak("cannot cast a secret value", instr.pos)
        if isinstance(target, T.IntType):
            if isinstance(value, float) and not value.is_integer():
                raise TypeFault(
                    f"cannot cast {value} to {target.render()} without "
                    f"truncation", instr.pos,
                    hint="use `math.round`, `math.floor` or `math.ceil` first")
            result = int(value)
        elif isinstance(target, T.FloatType):
            result = float(value)
        elif isinstance(target, T.TextType):
            result = to_text(value)
        elif isinstance(target, T.BoolType):
            result = truthy(value)
        else:
            result = value
        self.store(frame, instr.dst, result)

    def op_call(self, frame: Frame, instr: Instr) -> None:
        args = [self.resolve(frame, a) for a in instr.args]
        value = self.call_function(instr.meta["callee"], args, instr.pos)
        if instr.dst >= 0:
            self.store(frame, instr.dst, value)

    def op_call_indirect(self, frame: Frame, instr: Instr) -> None:
        callee = self.resolve(frame, instr.args[0])
        args = [self.resolve(frame, a) for a in instr.args[1:]]
        value = self.call_value(callee, args)
        if instr.dst >= 0:
            self.store(frame, instr.dst, value)

    def op_builtin(self, frame: Frame, instr: Instr) -> None:
        name = instr.meta["name"]
        b = L.BUILTINS.get(name)
        if b is None:
            raise GamaRuntimeFault(
                "UnknownBuiltin", f"the standard library has no `{name}`",
                instr.pos)
        args = [self.resolve(frame, a) for a in instr.args]
        for arg in args:
            if isinstance(arg, GSecret) and name not in L.SECRET_SAFE:
                raise SecretLeak(
                    f"a secret value cannot be passed to `{name}`", instr.pos,
                    hint="use `secrets.redact` or `secrets.fingerprint`")
        value = self._invoke_with_capability_args(b, args, instr.pos)
        if instr.dst >= 0:
            self.store(frame, instr.dst, value)

    def _invoke_with_capability_args(self, b: Any, args: List[Any],
                                     pos: Optional[SourcePos] = None) -> Any:
        """Invoke a builtin, honouring capabilities passed as arguments.

        Spec section 12: "A function cannot access patient storage unless a
        valid capability exists."  At runtime the capability exists precisely
        when a GCapability handle is in scope at the call site, so a handle
        passed as an argument confers its authority for that one call.
        """
        if not b.caps:
            return self.invoke_builtin(b, args)
        conferred: Set[str] = set()
        for arg in args:
            if isinstance(arg, GCapability):
                conferred.update(L.expand_capability(arg.base, arg.caps))
                conferred.update(arg.caps)
        missing = [c for c in b.caps if c not in conferred]
        if not missing:
            extra = conferred - self.ctx.grants
            self.ctx.grants |= extra
            try:
                return self.invoke_builtin(b, args)
            finally:
                self.ctx.grants -= extra
        return self.invoke_builtin(b, args)

    def op_method_call(self, frame: Frame, instr: Instr) -> None:
        obj = self.resolve(frame, instr.args[0])
        args = [self.resolve(frame, a) for a in instr.args[1:]]
        method = instr.meta["method"]
        value = self.dispatch_method(obj, method, args, instr.pos)
        if instr.dst >= 0:
            self.store(frame, instr.dst, value)

    def dispatch_method(self, obj: Any, method: str, args: List[Any],
                        pos: Optional[SourcePos] = None) -> Any:
        if isinstance(obj, GModuleRef):
            member = L.lookup_member(obj.name, method)
            if isinstance(member, L.Builtin):
                return self.invoke_builtin(member, args)
            if isinstance(member, L.Constant):
                return member.value
            raise GamaRuntimeFault(
                "NoMember", f"module `{obj.name}` has no `{method}`", pos)
        if isinstance(obj, GComponent):
            target = obj.methods.get(method)
            if target is None:
                raise GamaRuntimeFault(
                    "NoMethod",
                    f"{obj.kind} `{obj.name}` has no method `{method}`", pos,
                    context={"available": sorted(obj.methods)})
            return self.call_function(target, args, pos)
        if isinstance(obj, GSecret):
            raise SecretLeak(f"cannot call `.{method}` on a secret value", pos)
        if isinstance(obj, GTensor):
            if method in TENSOR_NATIVE:
                return getattr(obj, method)(*args)
            if method == "to_list":
                return obj.to_nested()
            if method == "shape":
                return list(obj.shape)
            raise GamaRuntimeFault(
                "NoMethod", f"Tensor has no method `{method}`", pos,
                context={"available": sorted(TENSOR_NATIVE | {"to_list"})})
        if isinstance(obj, str):
            name = TEXT_METHODS.get(method)
            if name:
                return self.invoke_builtin(L.BUILTINS[name], [obj] + args)
        if isinstance(obj, list):
            name = LIST_METHODS.get(method)
            if name:
                return self.invoke_builtin(L.BUILTINS[name], [obj] + args)
        if isinstance(obj, dict):
            name = MAP_METHODS.get(method)
            if name:
                return self.invoke_builtin(L.BUILTINS[name], [obj] + args)
        if isinstance(obj, (set, frozenset)):
            name = SET_METHODS.get(method)
            if name:
                return self.invoke_builtin(L.BUILTINS[name], [obj] + args)
        if isinstance(obj, GOption):
            name = OPTION_METHODS.get(method)
            if name:
                return self.invoke_builtin(L.BUILTINS[name], [obj] + args)
        if isinstance(obj, GResult):
            name = RESULT_METHODS.get(method)
            if name:
                return self.invoke_builtin(L.BUILTINS[name], [obj] + args)
        if isinstance(obj, GDuration):
            if method in ("seconds", "ms"):
                return obj.seconds if method == "seconds" else obj.seconds * 1000
        if isinstance(obj, GInstant):
            if method == "epoch":
                return obj.epoch
        if isinstance(obj, (GUuid, GUri)):
            if method == "text":
                return obj.text
        if isinstance(obj, GCapability):
            return self.dispatch_capability(obj, method, args, pos)
        if isinstance(obj, GRecord):
            if method in obj.fields and callable(obj.fields[method]):
                return obj.fields[method](*args)
            raise GamaRuntimeFault(
                "NoMethod",
                f"record `{obj.name}` has no method `{method}`", pos,
                context={"fields": sorted(obj.fields)})
        if isinstance(obj, GBuiltinRef):
            return self.invoke_builtin(obj.builtin, args)
        if isinstance(obj, GFnValue):
            return self.call_value(obj, args)
        raise GamaRuntimeFault(
            "NoMethod",
            f"{type_name(obj)} has no method `{method}`", pos,
            context={"value": display(obj)})

    def dispatch_capability(self, cap: GCapability, method: str,
                            args: List[Any],
                            pos: Optional[SourcePos] = None) -> Any:
        if method == "record":
            cap_name = "AuditWrite" if "Write" in cap.caps else None
            if cap_name:
                self.ctx.require_capability(cap_name, what="audit record",
                                            pos=pos)
            action = to_text(args[0]) if args else "AUDIT_EVENT"
            rec = self.ctx.audit.record(action, authority=cap.base,
                                        **(args[1] if len(args) > 1
                                           and isinstance(args[1], dict) else {}))
            return rec.event_id
        if cap.resource is not None:
            resource = cap.resource
            if callable(resource):
                return resource(*args)
            if isinstance(resource, dict):
                if method in ("query", "read", "get"):
                    return resource.get(args[0]) if args else dict(resource)
                if method in ("keys",):
                    return list(resource.keys())
                return resource
            if isinstance(resource, list) and method in ("query", "read"):
                return list(resource)
        raise CapabilityViolation(
            f"capability `{cap.base}` has no backing resource for "
            f"`{method}`", pos,
            hint="mint the capability with `capabilities.open(base, caps, "
                 "resource)` so that operations have something to act on")

    def op_make_list(self, frame: Frame, instr: Instr) -> None:
        self.store(frame, instr.dst,
                   [self.resolve(frame, a) for a in instr.args])

    def op_make_set(self, frame: Frame, instr: Instr) -> None:
        items = [self.resolve(frame, a) for a in instr.args]
        try:
            self.store(frame, instr.dst, set(items))
        except TypeError:
            self.store(frame, instr.dst, {display(i) for i in items})

    def op_make_tuple(self, frame: Frame, instr: Instr) -> None:
        self.store(frame, instr.dst,
                   tuple(self.resolve(frame, a) for a in instr.args))

    def op_make_map(self, frame: Frame, instr: Instr) -> None:
        out: Dict[Any, Any] = {}
        for i in range(0, len(instr.args), 2):
            key = self.resolve(frame, instr.args[i])
            value = self.resolve(frame, instr.args[i + 1])
            if isinstance(key, GSecret):
                raise SecretLeak("a secret value cannot be a Map key", instr.pos)
            if isinstance(key, (list, dict, set, GTensor, GRecord)):
                key = display(key)
            out[key] = value
        self.store(frame, instr.dst, out)

    def op_make_record(self, frame: Frame, instr: Instr) -> None:
        name = instr.meta.get("record") or "Record"
        fields: Dict[str, Any] = {}
        for i in range(0, len(instr.args), 2):
            key = self.resolve(frame, instr.args[i])
            fields[to_text(key)] = self.resolve(frame, instr.args[i + 1])
        self.store(frame, instr.dst, GRecord(name, fields))

    def op_make_variant(self, frame: Frame, instr: Instr) -> None:
        args = tuple(self.resolve(frame, a) for a in instr.args)
        self.store(frame, instr.dst,
                   GVariant(instr.meta["tag"], instr.meta.get("enum", ""), args))

    def op_construct(self, frame: Frame, instr: Instr) -> None:
        tag = instr.meta["tag"]
        args = [self.resolve(frame, a) for a in instr.args]
        if tag == "ok":
            self.store(frame, instr.dst, GResult(True, args[0] if args else UNIT))
        elif tag in ("fail", "err"):
            self.store(frame, instr.dst, GResult(False, args[0] if args else UNIT))
        elif tag == "some":
            self.store(frame, instr.dst, GOption(True, args[0] if args else UNIT))
        elif tag == "none":
            self.store(frame, instr.dst, GOption(False))
        else:
            raise GamaRuntimeFault("BadGIR", f"unknown constructor `{tag}`",
                                   instr.pos)

    def op_field(self, frame: Frame, instr: Instr) -> None:
        obj = self.resolve(frame, instr.args[0])
        name = instr.meta["name"]
        self.store(frame, instr.dst, self.get_field(obj, name, instr.pos))

    def get_field(self, obj: Any, name: str,
                  pos: Optional[SourcePos] = None) -> Any:
        if isinstance(obj, GSecret):
            raise SecretLeak(f"cannot read field `.{name}` of a secret value",
                             pos, hint="use `secrets.expose` with a reason")
        if isinstance(obj, GResult):
            if name == "ok":
                return obj.ok
            if name == "value":
                return obj.unwrap()
            if name == "error":
                return obj.value
        if isinstance(obj, GOption):
            if name == "some":
                return obj.some
            if name == "value":
                return obj.unwrap()
            if name == "is_some":
                return obj.some
        if isinstance(obj, GVariant):
            if name == "tag":
                return obj.tag
            if name in ("args",):
                return list(obj.args)
        if isinstance(obj, GCapability):
            # A capability handle *is* access to its resource (spec section
            # 12), so field reads go through to the wrapped value -- but only
            # when the handle actually carries Read.
            if name in ("base", "caps", "resource", "token"):
                return getattr(obj, name)
            if not obj.grants("Read"):
                raise CapabilityViolation(
                    f"capability {obj.base} does not grant Read, so its "
                    f"fields cannot be read", pos,
                    context={"caps": list(obj.caps)})
            if obj.resource is None:
                raise GamaRuntimeFault(
                    "NoResource",
                    f"capability {obj.base} carries no resource to read "
                    f"`{name}` from", pos)
            return self.get_field(obj.resource, name, pos)
        if isinstance(obj, GRecord):
            if name in obj.fields:
                return obj.fields[name]
            raise GamaRuntimeFault(
                "NoField", f"record `{obj.name}` has no field `{name}`", pos,
                context={"fields": sorted(obj.fields)})
        if isinstance(obj, GComponent):
            if name in obj.meta:
                return obj.meta[name]
            if name == "name":
                return obj.name
            if name == "kind":
                return obj.kind
            if name in obj.methods:
                return GFnValue(name=obj.methods[name],
                                gir=self.prog.functions.get(obj.methods[name]))
            raise GamaRuntimeFault(
                "NoField", f"{obj.kind} `{obj.name}` has no field `{name}`",
                pos, context={"methods": sorted(obj.methods)})
        if isinstance(obj, GTensor):
            if name == "shape":
                return list(obj.shape)
            if name == "rank":
                return obj.rank
            if name == "size":
                return obj.size
            if name == "dtype":
                return obj.dtype
            if name == "data":
                return obj.flat()
        if isinstance(obj, str) and name in ("length", "size", "len"):
            return len(obj)
        if isinstance(obj, (list, dict, set)) \
                and name in ("length", "size", "len"):
            return len(obj)
        if isinstance(obj, dict) and name == "keys":
            return list(obj.keys())
        if isinstance(obj, dict) and name == "values":
            return list(obj.values())
        if isinstance(obj, GDuration):
            if name == "seconds":
                return obj.seconds
            if name == "ms":
                return obj.seconds * 1000.0
        if isinstance(obj, GInstant) and name == "epoch":
            return obj.epoch
        if isinstance(obj, (GUuid, GUri)) and name == "text":
            return obj.text
        if isinstance(obj, GCapability):
            if name == "base":
                return obj.base
            if name == "caps":
                return list(obj.caps)
            if name == "token":
                return obj.token
        if isinstance(obj, GModuleRef):
            member = L.lookup_member(obj.name, name)
            if isinstance(member, L.Constant):
                return member.value
            if isinstance(member, L.Builtin):
                return GBuiltinRef(f"{obj.name}.{name}", member)
            raise GamaRuntimeFault("NoMember",
                                   f"module `{obj.name}` has no `{name}`", pos)
        raise GamaRuntimeFault(
            "NoField", f"{type_name(obj)} has no field `{name}`", pos,
            context={"value": display(obj)})

    def op_set_field(self, frame: Frame, instr: Instr) -> None:
        obj = self.resolve(frame, instr.args[0])
        value = self.resolve(frame, instr.args[1])
        name = instr.meta["name"]
        if isinstance(obj, GSecret):
            raise SecretLeak("cannot mutate a secret value's field", instr.pos)
        if isinstance(obj, GRecord):
            obj.fields[name] = value
            return
        if isinstance(obj, dict):
            obj[name] = value
            return
        raise GamaRuntimeFault(
            "ImmutableField",
            f"cannot set `.{name}` on {type_name(obj)}", instr.pos,
            hint="Gama-G values are immutable by default (spec section 1.2); "
                 "construct a new record instead")

    def op_index(self, frame: Frame, instr: Instr) -> None:
        obj = self.resolve(frame, instr.args[0])
        index = self.resolve(frame, instr.args[1])
        self.store(frame, instr.dst, self.get_index(obj, index, instr.pos))

    def get_index(self, obj: Any, index: Any,
                  pos: Optional[SourcePos] = None) -> Any:
        if isinstance(obj, GSecret):
            raise SecretLeak("cannot index into a secret value", pos)
        if isinstance(obj, GCapability):
            if not obj.grants("Read"):
                raise CapabilityViolation(
                    f"capability {obj.base} does not grant Read, so it cannot "
                    f"be indexed", pos, context={"caps": list(obj.caps)})
            if obj.resource is None:
                raise GamaRuntimeFault(
                    "NoResource",
                    f"capability {obj.base} carries no resource to index", pos)
            return self.get_index(obj.resource, index, pos)
        if isinstance(obj, GVariant):
            # `NegativeInput(reason)` binds its payload positionally.
            i = int(index)
            if not -len(obj.args) <= i < len(obj.args):
                raise GamaRuntimeFault(
                    "IndexOutOfRange",
                    f"variant `{obj.tag}` carries {len(obj.args)} payload "
                    f"value(s); index {i} is out of range", pos)
            return obj.args[i]
        if isinstance(obj, GTensor):
            flat = obj.flat()
            i = int(index)
            if not -len(flat) <= i < len(flat):
                raise GamaRuntimeFault(
                    "IndexOutOfRange",
                    f"index {i} is out of range for a Tensor with "
                    f"{len(flat)} element(s)", pos)
            return flat[i]
        if isinstance(obj, list) or isinstance(obj, tuple):
            i = int(index)
            if not -len(obj) <= i < len(obj):
                raise GamaRuntimeFault(
                    "IndexOutOfRange",
                    f"index {i} is out of range for a List of length {len(obj)}",
                    pos)
            return obj[i]
        if isinstance(obj, str):
            i = int(index)
            if not -len(obj) <= i < len(obj):
                raise GamaRuntimeFault(
                    "IndexOutOfRange",
                    f"index {i} is out of range for Text of length {len(obj)}",
                    pos)
            return obj[i]
        if isinstance(obj, dict):
            key = display(index) if isinstance(index, (list, dict, GTensor)) \
                else index
            if key not in obj:
                raise GamaRuntimeFault(
                    "KeyNotFound",
                    f"Map has no key {display(key)}", pos,
                    hint="use `collections.map_get` for an Option-returning "
                         "lookup that cannot fault")
            return obj[key]
        if isinstance(obj, (set, frozenset)):
            return sorted(obj, key=repr)[int(index)]
        if isinstance(obj, GTensor):
            return obj.index1(int(index))
        if obj is None or isinstance(obj, GUnit):
            raise GamaRuntimeFault("KeyNotFound",
                                   "Map has no key " + display(index), pos)
        raise GamaRuntimeFault(
            "NotIndexable", f"{type_name(obj)} cannot be indexed", pos)

    def op_set_index(self, frame: Frame, instr: Instr) -> None:
        obj = self.resolve(frame, instr.args[0])
        index = self.resolve(frame, instr.args[1])
        value = self.resolve(frame, instr.args[2])
        if isinstance(obj, GSecret):
            raise SecretLeak("cannot mutate a secret value", instr.pos)
        if isinstance(obj, list):
            i = int(index)
            if not -len(obj) <= i < len(obj):
                raise GamaRuntimeFault(
                    "IndexOutOfRange",
                    f"index {i} is out of range for a List of length {len(obj)}",
                    instr.pos)
            obj[i] = value
            return
        if isinstance(obj, dict):
            obj[index] = value
            return
        if isinstance(obj, GTensor):
            obj.set_at([int(index)], value)
            return
        raise GamaRuntimeFault(
            "ImmutableIndex", f"cannot assign into {type_name(obj)}", instr.pos,
            hint="declare the binding with `var` and use a mutable "
                 "collection type")

    def op_tensor_op(self, frame: Frame, instr: Instr) -> None:
        args = [self.resolve(frame, a) for a in instr.args]
        op = instr.meta["op"]
        target = args[0]
        rest = args[1:]
        self.store(frame, instr.dst, getattr(target, op)(*rest))

    # ------------------------------------------------------------------
    # language-level boundaries
    # ------------------------------------------------------------------
    def op_audit(self, frame: Frame, instr: Instr) -> None:
        keys = instr.meta["keys"]
        values = [self.resolve(frame, a) for a in instr.args]
        fields: Dict[str, Any] = {}
        for key, value in zip(keys, values):
            if isinstance(value, GSecret):
                raise SecretLeak(
                    f"audit field `{key}` would record a secret value",
                    instr.pos,
                    hint="use `secrets.redact` or `secrets.fingerprint`")
            fields[key] = value
        action = to_text(fields.pop("action", "AUDIT_EVENT"))
        with self.lock:
            self.ctx.audit.record(
                action,
                actor=to_text(fields.pop("actor")) if "actor" in fields else None,
                object=to_text(fields.pop("object")) if "object" in fields else None,
                reason=to_text(fields.pop("reason")) if "reason" in fields else None,
                **fields)
            self.ctx.stats.audits += 1

    def op_require(self, frame: Frame, instr: Instr) -> None:
        value = self.resolve(frame, instr.args[0])
        if not truthy(value):
            message = instr.meta.get("message") or "a required condition failed"
            with self.lock:
                self.ctx.audit.record("REQUIRE_FAILED", level="security",
                                      reason=to_text(message))
            raise ContractViolation(to_text(message), instr.pos)

    def op_assert(self, frame: Frame, instr: Instr) -> None:
        value = self.resolve(frame, instr.args[0])
        if not truthy(value):
            message = instr.meta.get("message") or "assertion failed"
            raise GamaRuntimeFault("AssertionFailed", to_text(message),
                                   instr.pos)

    def op_contract(self, frame: Frame, instr: Instr) -> None:
        value = self.resolve(frame, instr.args[0])
        if not truthy(value):
            kind = instr.meta.get("kind", "requires")
            raise ContractViolation(
                f"{kind} contract violated: "
                f"{instr.meta.get('text') or 'condition did not hold'}",
                instr.pos)

    def op_cap_check(self, frame: Frame, instr: Instr) -> None:
        self.ctx.require_capability(instr.meta["capability"],
                                    what=instr.meta.get("what", ""),
                                    pos=instr.pos)

    def op_checkpoint(self, frame: Frame, instr: Instr) -> None:
        mode = instr.meta.get("mode", "at")
        label = instr.meta.get("raw") or "checkpoint"
        with self.lock:
            self.ctx.capture_checkpoint(label)
        if mode == "every" and instr.meta.get("interval"):
            self._checkpoint_interval = float(instr.meta["interval"])
            self._armed_checkpoint = time.time() + self._checkpoint_interval

    def op_secret_guard(self, frame: Frame, instr: Instr) -> None:
        operand = instr.args[0]
        if operand.kind != "slot":
            return
        value = frame.slots[operand.index]
        if not isinstance(value, GSecret):
            frame.slots[operand.index] = GSecret(
                value, frame.fn.slots[operand.index].name)

    def op_policy(self, frame: Frame, instr: Instr) -> None:
        rules = self.resolve(frame, instr.args[0])
        decision = self.ctx.evaluate_policy(instr.meta.get("name", ""), rules)
        if instr.dst >= 0:
            self.store(frame, instr.dst, decision)

    def op_agent_send(self, frame: Frame, instr: Instr) -> None:
        agent = to_text(self.resolve(frame, instr.args[0]))
        message = self.resolve(frame, instr.args[1])
        self.ctx.send_to_agent(agent, message)

    def op_transaction(self, frame: Frame, instr: Instr) -> None:
        action = instr.meta["action"]
        name = instr.meta.get("name", "")
        if action == "begin":
            self.ctx.begin_transaction(name)
            self.ctx.active_transaction = name
            self.ctx.transaction_owner = frame.fn.name
        elif action == "commit":
            self.ctx.commit_transaction(name)
            self.ctx.active_transaction = None
            self.ctx.transaction_owner = None
        elif action == "abort":
            self.ctx.abort_transaction(name, instr.meta.get("reason", ""))
            self.ctx.active_transaction = None
            self.ctx.transaction_owner = None

    # ------------------------------------------------------------------
    # parallel regions -- the operation graph (spec sections 3, 9C)
    # ------------------------------------------------------------------
    def op_parallel(self, frame: Frame, instr: Instr) -> None:
        """Execute a data-parallel operation graph (spec sections 3 and 9C).

        Dependence analysis in the builder produced a partial order over the
        region's tasks.  Tasks are grouped into topological levels; everything
        within a level is independent and runs concurrently, and a level's
        writes are committed before the next level reads them -- which is what
        makes `c = computeC(a, b)` see the values `a` and `b` produced.

        Writes within a level are applied in program order, so the region's
        final state does not depend on the schedule the graph happened to
        admit.  That is the determinism guarantee of spec section 1.3.
        """
        tasks = instr.meta["tasks"]
        if not tasks:
            return
        self.ctx.stats.parallel_regions += 1

        for level in _topological_levels(tasks):
            level = sorted(level, key=lambda t: t["index"])
            if len(level) == 1:
                produced = {level[0]["name"]: self._run_task(frame, level[0])}
            else:
                produced = {}
                with ThreadPoolExecutor(max_workers=min(len(level), 8),
                                        thread_name_prefix="gaem") as pool:
                    futures = [(task, pool.submit(self._run_task, frame, task))
                               for task in level]
                    for task, future in futures:
                        produced[task["name"]] = future.result()
            self._commit_writes(frame, level, produced)

        if instr.dst >= 0:
            self.store(frame, instr.dst, len(tasks))

    def _commit_writes(self, frame: Frame, tasks: List[Dict[str, Any]],
                       produced: Dict[str, Dict[str, Any]]) -> None:
        for task in tasks:
            values = produced.get(task["name"]) or {}
            for name, slot in zip(task["writes"], task["write_slots"]):
                if name not in values:
                    continue
                if slot is not None and slot >= 0:
                    self.store(frame, slot, values[name])
                else:
                    self.globals[name] = values[name]

    def _run_task(self, frame: Frame, task: Dict[str, Any]) -> Dict[str, Any]:
        self.ctx.stats.parallel_tasks += 1
        args = []
        for slot in task["param_slots"]:
            if slot is None or slot < 0:
                args.append(self.globals.get(task["params"][len(args)]))
            else:
                args.append(frame.slots[slot])
        value = self.call_function(task["function"], args)
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------
    # protected regions -- bounded recovery (spec section 10)
    # ------------------------------------------------------------------
    def op_protected(self, frame: Frame, instr: Instr) -> None:
        protect = instr.meta.get("protect") or ""
        steps = [RecoveryStepSpec(action=s["action"], count=s.get("count"),
                                  target=s.get("target", ""),
                                  raw=s.get("raw", s["action"]))
                 for s in instr.meta.get("steps", [])]
        component = instr.meta.get("component", "component")

        # Capture a checkpoint *before* entering the region so that
        # `restore checkpoint` has genuine recorded state to return to.
        # Spec section 10 forbids inventing state during recovery.
        with self.lock:
            self.ctx.capture_checkpoint(f"{component}-entry",
                                        authorization=f"service:{component}")

        # The region's operands are the arguments its protected work takes. The
        # older builder emits a protected region around a nullary service body and
        # so passes none, which is why this defaults to empty rather than being
        # required: a core intent's work takes its sources.
        protected_args = [self.resolve(frame, a) for a in instr.args]
        engine = RecoveryEngine(self.ctx)
        with self.lock:
            outcome = engine.run(
                steps,
                (lambda: self.call_function(protect, protected_args)) if protect
                else (lambda: UNIT),
                name=component)
        self.ctx.stats.recoveries += 1 if outcome.actions else 0
        if outcome.recovered:
            self.ctx.stats.recoveries_succeeded += 1
        record = GRecord("RecoveryOutcome", {
            "value": outcome.value,
            "recovered": outcome.recovered,
            "level": outcome.level,
            "level_name": outcome.level_name,
            "attempts": outcome.attempts,
            "escalated": outcome.escalated,
            "actions": [a.to_dict()["raw"] for a in outcome.actions],
            "final_error": outcome.final_error or "",
        })
        if not outcome.recovered and outcome.final_error:
            raise GamaRuntimeFault(
                "RecoveryExhausted",
                f"recovery policy for `{component}` was exhausted: "
                f"{outcome.final_error}", instr.pos,
                context={"outcome": outcome.to_dict()})
        if instr.dst >= 0:
            self.store(frame, instr.dst, record)


def _topological_levels(tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group tasks into levels that may run concurrently.

    Tasks in the same level have no dependence between them; level N+1 waits
    for level N.  A cyclic declaration is impossible here because edges only
    point backwards in program order, but the routine still guards against it.
    """
    by_name = {t["name"]: t for t in tasks}
    depth: Dict[str, int] = {}

    def resolve(name: str, stack: Tuple[str, ...]) -> int:
        if name in depth:
            return depth[name]
        if name in stack:
            return 0
        task = by_name.get(name)
        if task is None:
            return 0
        best = -1
        for dep in task.get("depends_on", []):
            best = max(best, resolve(dep, stack + (name,)))
        depth[name] = best + 1
        return depth[name]

    for task in tasks:
        resolve(task["name"], ())
    levels: List[List[Dict[str, Any]]] = []
    for task in tasks:
        level = depth.get(task["name"], 0)
        while len(levels) <= level:
            levels.append([])
        levels[level].append(task)
    return levels


def execute_program(program: GProgram, context: Optional[Context] = None,
                    checker: Any = None, entry: str = "<main>",
                    args: Sequence[Any] = ()) -> Tuple[Any, "VM"]:
    vm = VM(program, context, checker)
    result = vm.run(entry, args)
    return result, vm
