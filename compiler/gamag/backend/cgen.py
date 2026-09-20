"""The native CPU backend (audit priority 3): GIR to C to machine code.

The design constraint is not "generate C" -- that is the easy part.  It is that
the generated program must *agree with the reference interpreter*, because a
backend that is fast and occasionally wrong is worse than no backend.  So:

* `unsupported()` runs first and names every construct this backend cannot
  compile.  A program containing one is refused with a diagnostic rather than
  translated into something that quietly differs.  Nothing here guesses.
* Integer semantics follow `runtime/vm.py`, not C.  The interpreter computes
  exactly (Python ints are unbounded) and range-checks at *store* and *return*
  against the declared type.  C wraps.  So arithmetic uses the overflow-checking
  builtins and the range check is emitted where the interpreter puts it.
* Value formatting is delegated to the runtime, which reproduces
  `values.py::display` including Python's shortest-round-trip float repr.

What this backend deliberately does not yet compile is listed in `UNSUPPORTED`
and reported verbatim to the user: audit chains, capabilities, secrets,
transactions, recovery regions, tensors, autodiff and agents all need runtime
state that the C runtime does not carry yet.  They are named, not hidden.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..gir.ir import GFunction, GProgram, Instr, Op, Operand
from ..semantic import types as T

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt")
RUNTIME_SOURCES = ("gamag_rt.c",)

#: Builtins this backend can compile, because they are pure functions of their
#: arguments or write only to stdout.  Everything else needs runtime state the
#: C runtime does not carry yet, and is reported rather than approximated.
SUPPORTED_BUILTINS = frozenset({
    "print", "println", "print_raw", "eprint",
    "len", "range", "__range", "str", "float", "int", "bool", "abs",
    "to_text", "contains", "sorted", "min", "max", "sum",
    "__iter_list", "__iter_range", "__list_get",
    "math.abs", "math.round", "math.floor", "math.ceil", "math.sqrt",
    "math.pow", "math.clamp", "math.min", "math.max", "math.log", "math.exp",
    # Capability-gated.  These are the operations the capability system exists
    # for, so they are the ones that make native enforcement observable rather
    # than notional: without them this backend could only refuse whole
    # programs, which is not a security model.
    "io.read_file", "io.write_file", "io.append_file", "io.exists",
})

#: Ops this backend emits.  Anything outside this set is reported by name.
SUPPORTED_OPS = frozenset({
    Op.CONST, Op.COPY, Op.BINOP, Op.UNOP, Op.CAST, Op.CALL, Op.BUILTIN,
    Op.MAKE_LIST, Op.MAKE_TUPLE, Op.MAKE_MAP, Op.MAKE_RECORD, Op.MAKE_VARIANT,
    Op.CONSTRUCT, Op.FIELD, Op.INDEX, Op.SET_INDEX, Op.JUMP, Op.JUMP_IF,
    Op.RETURN, Op.FAULT, Op.MATCH_FAIL, Op.REQUIRE, Op.ASSERT,
    Op.LOAD_GLOBAL, Op.STORE_GLOBAL, Op.SECRET_GUARD, Op.CAP_CHECK,
})

#: Ops that are refused with a reason, because translating them would produce a
#: program that looks right and is not.
UNSUPPORTED_REASONS = {
    Op.AUDIT: "the audit chain (hash-chained, signed records) has no C "
              "implementation yet",
    Op.CHECKPOINT: "checkpoint capture and restore are not implemented natively",
    Op.TRANSACTION: "transaction begin/commit are not implemented natively",
    Op.PROTECTED: "recovery regions are not implemented natively",
    Op.PARALLEL: "the native backend is single-threaded",
    Op.POLICY: "policy evaluation is not implemented natively",
    Op.AGENT_SEND: "agents are not implemented natively",
    Op.CONTRACT: "contract checking is not implemented natively",
    Op.TENSOR_OP: "tensors are not implemented natively",
    Op.CALL_INDIRECT: "indirect calls are not implemented natively",
    Op.METHOD_CALL: "method dispatch is not implemented natively",
    Op.SET_FIELD: "record mutation is not implemented natively",
    Op.MAKE_SET: "sets are not implemented natively",
    Op.FOR_STATEMENT: "`for` loops are not implemented natively",
}


@dataclass
class Problem:
    """One thing this backend cannot compile, in the program's own terms."""

    what: str
    reason: str
    where: str = ""

    def render(self) -> str:
        at = f" at {self.where}" if self.where else ""
        return f"{self.what}{at} -- {self.reason}"


@dataclass
class Support:
    """The result of asking whether a program can be compiled natively."""

    problems: List[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _pos_of(instr: Instr) -> str:
    p = getattr(instr, "pos", None)
    if p is None:
        return ""
    return f"{getattr(p, 'path', '')}:{getattr(p, 'line', 0)}:{getattr(p, 'col', 0)}"


def _const_supported(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, tuple):
        if not value:
            return False
        tag = value[0]
        if tag == "unit":
            return True
        if tag == "default":
            return True
        if tag == "variant":
            return True
        return False
    if isinstance(value, (list, tuple)):
        return all(_const_supported(v) for v in value)
    return False


def _int_type_supported(ty: Any) -> bool:
    """Whether an integer type fits the runtime's exact 64-bit arithmetic."""
    if not isinstance(ty, T.IntType):
        return True
    lo, hi = ty.range
    return lo >= -(2 ** 63) and hi <= 2 ** 63 - 1


def unsupported(program: GProgram) -> Support:
    """Everything this backend cannot compile, found before any code is emitted.

    This runs first and its output is the diagnostic.  A backend that discovers
    an unsupported construct halfway through emission has already promised the
    user an executable.
    """
    out = Support()
    names = set(program.functions)

    for fname, fn in program.functions.items():
        for slot in fn.slots:
            if not _int_type_supported(slot.type):
                out.problems.append(Problem(
                    f"`{fname}` holds a `{slot.type.render()}` value",
                    "the native runtime computes integers exactly only within "
                    "64 bits, and this type's range exceeds that"))
        if not _int_type_supported(fn.ret):
            out.problems.append(Problem(
                f"`{fname}` returns `{fn.ret.render()}`",
                "the native runtime computes integers exactly only within "
                "64 bits, and this type's range exceeds that"))

        for block in fn.blocks:
            for instr in block.instrs:
                where = _pos_of(instr)
                if instr.op in UNSUPPORTED_REASONS:
                    out.problems.append(Problem(
                        f"`{instr.op.name.lower()}`", UNSUPPORTED_REASONS[instr.op],
                        where))
                    continue
                if instr.op not in SUPPORTED_OPS:
                    out.problems.append(Problem(
                        f"`{instr.op.name.lower()}`",
                        "not implemented by the native backend", where))
                    continue
                if instr.op == Op.BUILTIN:
                    name = instr.meta.get("name", "")
                    if name not in SUPPORTED_BUILTINS:
                        out.problems.append(Problem(
                            f"builtin `{name}`",
                            "needs runtime state the native backend does not "
                            "carry yet", where))
                if instr.op == Op.CALL:
                    callee = instr.meta.get("callee", "")
                    if callee not in names:
                        out.problems.append(Problem(
                            f"call to `{callee}`",
                            "no such function in this program", where))
                if instr.op == Op.CONST:
                    for arg in instr.args:
                        if not _const_supported(arg.value):
                            out.problems.append(Problem(
                                f"constant {arg.value!r}",
                                "not representable in the native runtime",
                                where))
                if instr.op == Op.FOR_STATEMENT:
                    out.problems.append(Problem(
                        "`for` loop",
                        "not implemented by the native backend", where))
    return out


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _c_ident(name: str) -> str:
    """A C identifier for a Gama-G name, reversibly enough to be readable."""
    out = re.sub(r"[^A-Za-z0-9]", "_", name)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "_unnamed"


def _c_string(text: str) -> str:
    out = ['"']
    for ch in text:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


class CGenerator:
    """Emit C for a program this backend has already agreed to compile."""

    def __init__(self, program: GProgram, source_path: str = "<program>",
                 entry: Optional[str] = None):
        self.program = program
        self.source_path = source_path
        self.lines: List[str] = []
        self.globals: Dict[str, Any] = {}
        self.entry = entry or pick_entry(program)

    def _declare_authority(self) -> None:
        """The capabilities the program *asks* for, as data.

        A request, not authority: `g_run` decides whether to honour it, and
        `--strict-authority` refuses it, because a program that could confer a
        capability on itself by writing it down has ambient authority under
        another name (spec section 12).
        """
        grants = list(getattr(self.program, "grants", ()) or ())
        self.emit("/* The capabilities this program declares it needs.  They are")
        self.emit("   a request: `g_run` decides whether to honour them. */")
        if grants:
            items = ", ".join(_c_string(g) for g in grants)
            self.emit(f"static const char *const g_declared_grants[] = {{{items}}};")
            self.emit(f"static const size_t g_declared_grants_n = {len(grants)};")
        else:
            self.emit("static const char *const g_declared_grants[] = {NULL};")
            self.emit("static const size_t g_declared_grants_n = 0;")
        self.emit()

    # ---- scaffolding ---------------------------------------------

    def emit(self, line: str = "") -> None:
        self.lines.append(line)

    def generate(self) -> str:
        self.emit(f"/* Generated by the Gama-G native backend (ggc native).")
        self.emit(f"   Source: {self.source_path}")
        self.emit(f"   Do not edit: this file is an artifact.  The semantics it")
        self.emit(f"   must match are in compiler/gamag/runtime/. */")
        self.emit()
        self.emit('#include "gamag_rt.h"')
        self.emit()

        self._declare_authority()
        self._declare_globals()
        for name, fn in self.program.functions.items():
            self.emit(self._signature(name, fn) + ";")
        self.emit()
        self.emit("int main(int argc, char **argv)")
        self.emit("{")
        self.emit("    return g_run(argc, argv, g_declared_grants,")
        self.emit("                 g_declared_grants_n);")
        self.emit("}")
        self.emit()
        self.emit("static int g_initialized = 0;")
        self.emit()
        self.emit("GValue g_main(int argc, char **argv)")
        self.emit("{")
        self.emit("    (void)argc; (void)argv;")
        # `runtime/vm.py::run` initialises the module (`<main>`) before calling
        # the entry point, exactly once.  Doing it in a different order would
        # change when top-level bindings exist, and the differential tester
        # would be comparing two different programs.
        initializer = self.program.entry or "<main>"
        if initializer in self.program.functions and initializer != self.entry:
            self.emit("    if (!g_initialized) {")
            self.emit("        g_initialized = 1;")
            self.emit(f"        {self._fn(initializer)}();")
            self.emit("    }")
        self.emit(f"    return {self._fn(self.entry)}();")
        self.emit("}")
        self.emit()

        for name, fn in self.program.functions.items():
            self._function(name, fn)
        return "\n".join(self.lines) + "\n"

    def _used_globals(self) -> List[str]:
        """The module bindings actually read or written by an instruction.

        `program.global_bindings` also names functions, enums and record types,
        which are not mutable state; declaring them would produce a wall of
        unused-variable warnings that hide the ones that matter.
        """
        used = set()
        for fn in self.program.functions.values():
            for block in fn.blocks:
                for instr in block.instrs:
                    if instr.op == Op.LOAD_GLOBAL:
                        for arg in instr.args:
                            if arg.kind == "global" and arg.name:
                                used.add(arg.name)
                    elif instr.op == Op.STORE_GLOBAL:
                        name = instr.meta.get("name")
                        if isinstance(name, str) and name:
                            used.add(name)
                        for arg in instr.args:
                            if arg.kind == "global" and arg.name:
                                used.add(arg.name)
                    else:
                        for arg in instr.args:
                            if arg.kind == "global" and arg.name:
                                used.add(arg.name)
        return sorted(used)

    def _declare_globals(self) -> None:
        """Module-level bindings become file-scope C variables."""
        for gname in self._used_globals():
            self.emit(f"static GValue g_global_{_c_ident(gname)} = "
                      f"{{GV_UNIT, {{0}}}};")
        if self._used_globals():
            self.emit()

    @staticmethod
    def _fn(name: str) -> str:
        """A C name for a Gama-G function.

        Prefixed unconditionally: a program is free to define `main`, `index` or
        `abort`, and a collision with C's own names is a bug in the generator
        that would otherwise surface as a confusing link error.
        """
        return "gf_" + _c_ident(name)

    @staticmethod
    def _jump_targets(fn: GFunction) -> set:
        """Block ids that some instruction jumps to."""
        out = set()
        for block in fn.blocks:
            for instr in block.instrs:
                for key in ("target", "target_false"):
                    value = instr.meta.get(key)
                    if isinstance(value, str) and value:
                        out.add(value)
        return out

    def _signature(self, name: str, fn: GFunction) -> str:
        params = fn.param_names or []
        args = ", ".join(f"GValue {_c_ident(p)}" for p in params) or "void"
        return f"static GValue {self._fn(name)}({args})"

    # ---- one function --------------------------------------------

    def _function(self, name: str, fn: GFunction) -> None:
        self.emit(self._signature(name, fn))
        self.emit("{")
        slots = fn.slots
        params = list(fn.param_names or [])
        decls: List[str] = []
        for index, slot in enumerate(slots):
            if index < len(params):
                decls.append(f"    GValue s{index} = {_c_ident(params[index])};")
            else:
                decls.append(f"    GValue s{index}; (void)s{index};")
        self.emit("\n".join(decls))
        # Slots the IR references beyond the declared list.
        self._slot_count = len(slots)
        self.emit()

        targets = self._jump_targets(fn)
        for index, block in enumerate(fn.blocks):
            if block.id in targets or index == 0 and block.id in targets:
                self.emit(f"  {block.id}: ;")
            elif block.id in targets:
                self.emit(f"  {block.id}: ;")
            for instr in block.instrs:
                self._instr(instr, fn)
        self.emit(f"    g_raise(\"BadGIR\", \"\", "
                  f"\"control fell off the end of `{name}`\");")
        self.emit("}")
        self.emit()

    # ---- operands -------------------------------------------------

    def _operand(self, op: Operand) -> str:
        if op.kind == "slot":
            return f"s{op.index}"
        if op.kind == "const":
            return self._const(op.value)
        if op.kind == "global":
            return f"g_global_{_c_ident(op.name)}"
        return "g_unit()"

    def _const(self, value: Any) -> str:
        if value is None:
            return "g_unit()"
        if isinstance(value, bool):
            return f"g_bool({1 if value else 0})"
        if isinstance(value, int):
            if value >= 2 ** 63 or value < -(2 ** 63):
                return "g_raise(\"BadGIR\", \"\", \"integer constant too large\"), g_unit()"
            if value == -(2 ** 63):
                return "g_int(-9223372036854775807LL - 1)"
            return f"g_int({value}LL)"
        if isinstance(value, float):
            return f"g_float({_c_float(value)})"
        if isinstance(value, str):
            return f"g_text({_c_string(value)})"
        if isinstance(value, tuple) and value:
            tag = value[0]
            if tag == "unit":
                return "g_unit()"
            if tag == "default":
                return self._default_for(value[1])
            if tag == "variant":
                return (f"g_variant(\"\", {_c_string(str(value[1]))}, 1, "
                        f"(const GValue[]){{g_unit()}})")
        if isinstance(value, (list, tuple)):
            items = ", ".join(self._const(v) for v in value) or ""
            n = len(value)
            if n == 0:
                return "g_list_new(0, NULL)"
            return f"g_list_new({n}, (const GValue[]){{{items}}})"
        return "g_unit()"

    def _default_for(self, ty: Any) -> str:
        if isinstance(ty, T.BoolType):
            return "g_bool(0)"
        if isinstance(ty, T.IntType):
            return "g_int(0)"
        if isinstance(ty, T.FloatType):
            return "g_float(0.0)"
        if isinstance(ty, T.TextType):
            return 'g_text("")'
        if isinstance(ty, T.ListType):
            return "g_list_new(0, NULL)"
        if isinstance(ty, T.MapType):
            return "g_map_new()"
        return "g_unit()"

    # ---- instructions ---------------------------------------------

    def _store(self, instr: Instr, fn: GFunction, expr: str) -> None:
        """Store into the destination slot, with the interpreter's range check."""
        dst = instr.dst
        if dst < 0:
            self.emit(f"    (void)({expr});")
            return
        self._ensure_slot(dst)
        ty = fn.slots[dst].type if dst < len(fn.slots) else None
        pos = _c_string(_pos_of(instr))
        if isinstance(ty, T.IntType):
            lo, hi = ty.range
            lo_c = "-9223372036854775807LL - 1" if lo == -(2 ** 63) else f"{lo}LL"
            hi_c = f"{hi}LL"
            name = _c_string(fn.slots[dst].name)
            self.emit(f"    s{dst} = g_store_int({expr}, {lo_c}, {hi_c}, "
                      f"{_c_string(ty.render())}, {name}, {pos});")
        else:
            self.emit(f"    s{dst} = {expr};")

    def _ensure_slot(self, index: int) -> None:
        if index >= self._slot_count:
            self.emit(f"    GValue s{index} = g_unit();")
            self._slot_count = index + 1

    def _args(self, instr: Instr) -> str:
        return ", ".join(self._operand(a) for a in instr.args)

    def _instr(self, instr: Instr, fn: GFunction) -> None:
        pos = _c_string(_pos_of(instr))
        op = instr.op

        if op == Op.CONST:
            self._store(instr, fn, self._operand(instr.args[0]))
        elif op == Op.COPY:
            self._store(instr, fn, self._operand(instr.args[0]))
        elif op == Op.BINOP:
            a, b = instr.args[0], instr.args[1]
            operator = _c_string(str(instr.meta.get("operator", "")))
            self._store(instr, fn,
                        f"g_binop({operator}, {self._operand(a)}, "
                        f"{self._operand(b)}, {pos})")
        elif op == Op.UNOP:
            operator = _c_string(str(instr.meta.get("operator", "")))
            self._store(instr, fn,
                        f"g_unop({operator}, {self._operand(instr.args[0])}, {pos})")
        elif op == Op.CAST:
            self._store(instr, fn,
                        f"g_cast({self._operand(instr.args[0])}, "
                        f"{_c_string(instr.type.render())}, {pos})")
        elif op == Op.CALL:
            callee = self._fn(str(instr.meta.get("callee", "")))
            self._store(instr, fn, f"{callee}({self._args(instr)})")
        elif op == Op.BUILTIN:
            self._store(instr, fn, self._builtin(instr, pos))
        elif op in (Op.MAKE_LIST, Op.MAKE_TUPLE):
            n = len(instr.args)
            ctor = "g_list_new" if op == Op.MAKE_LIST else "g_tuple_new"
            if n == 0:
                self._store(instr, fn, f"{ctor}(0, NULL)")
            else:
                items = ", ".join(self._operand(a) for a in instr.args)
                self._store(instr, fn,
                            f"{ctor}({n}, (const GValue[]){{{items}}})")
        elif op == Op.MAKE_MAP:
            pairs = ", ".join(self._operand(a) for a in instr.args)
            n = len(instr.args)
            self._store(instr, fn, f"g_map_literal({n}, (const GValue[]){{{pairs}}})"
                        if n else "g_map_new()")
        elif op == Op.MAKE_RECORD:
            self._record(instr, fn)
        elif op == Op.MAKE_VARIANT:
            tag = _c_string(str(instr.meta.get("tag", "")))
            enum = _c_string(str(instr.meta.get("enum", "")))
            n = len(instr.args)
            if n == 0:
                self._store(instr, fn, f"g_variant({enum}, {tag}, 0, NULL)")
            else:
                items = ", ".join(self._operand(a) for a in instr.args)
                self._store(instr, fn,
                            f"g_variant({enum}, {tag}, {n}, (const GValue[]){{{items}}})")
        elif op == Op.CONSTRUCT:
            self._construct(instr, fn)
        elif op == Op.FIELD:
            name = _c_string(str(instr.meta.get("name", "")))
            self._store(instr, fn,
                        f"g_field({self._operand(instr.args[0])}, {name}, {pos})")
        elif op == Op.INDEX:
            self._store(instr, fn,
                        f"g_index({self._operand(instr.args[0])}, "
                        f"{self._operand(instr.args[1])}, {pos})")
        elif op == Op.SET_INDEX:
            self.emit(f"    g_set_index({self._operand(instr.args[0])}, "
                      f"{self._operand(instr.args[1])}, "
                      f"{self._operand(instr.args[2])}, {pos});")
        elif op == Op.JUMP:
            self.emit(f"    goto {instr.meta.get('target', '')};")
        elif op == Op.JUMP_IF:
            cond = self._operand(instr.args[0])
            self.emit(f"    if (g_truthy({cond}, {pos})) "
                      f"goto {instr.meta.get('target', '')};")
            self.emit(f"    goto {instr.meta.get('target_false', '')};")
        elif op == Op.RETURN:
            expr = self._operand(instr.args[0]) if instr.args else "g_unit()"
            if isinstance(fn.ret, T.IntType):
                lo, hi = fn.ret.range
                lo_c = "-9223372036854775807LL - 1" if lo == -(2 ** 63) else f"{lo}LL"
                self.emit(f"    return g_return_int({expr}, {lo_c}, {hi}LL, "
                          f"{_c_string(fn.ret.render())}, {_c_string(fn.name)}, {pos});")
            else:
                self.emit(f"    return {expr};")
        elif op == Op.FAULT:
            kind = _c_string(str(instr.meta.get("kind", "Fault")))
            message = _c_string(str(instr.meta.get("message", "")))
            self.emit(f"    g_raise({kind}, {pos}, \"%s\", {message});")
        elif op == Op.MATCH_FAIL:
            subject = str(instr.meta.get("subject_type", "value"))
            self.emit(f"    g_raise(\"MatchError\", {pos}, "
                      f"\"no branch matched a value of %s\", "
                      f"g_type_name({self._operand(instr.args[0])}));")
            self.emit(f"    (void){_c_string(subject)};")
        elif op in (Op.REQUIRE, Op.ASSERT):
            self._require(instr, fn, pos)
        elif op == Op.LOAD_GLOBAL:
            self._store(instr, fn,
                        f"g_global_{_c_ident(str(instr.args[0].name))}")
        elif op == Op.STORE_GLOBAL:
            name = _c_ident(str(instr.meta.get("name", "")))
            self.emit(f"    g_global_{name} = {self._operand(instr.args[0])};")
        elif op == Op.CAP_CHECK:
            # The compile-time check already proved this demand is covered by
            # the program's authority.  The run-time check is emitted anyway,
            # for the reason the core lowerer gives when it emits the opcode: a
            # proof is a reason to trust the program, not a reason to remove
            # the boundary.  A capability checked only at compile time is not
            # enforced against a hand-edited IR or a reordered backend.
            capability = _c_string(str(instr.meta.get("capability", "")))
            what = _c_string(str(instr.meta.get("what", "")))
            self.emit(f"    g_cap_require({capability}, {what}, {pos});")
        elif op == Op.SECRET_GUARD:
            # The guard is a compile-time property that survives into the IR so
            # that it is visible; the runtime refusal lives in g_display.
            self.emit(f"    /* secret guard: {_pos_of(instr)} */")
        else:
            self.emit(f"    g_raise(\"BadGIR\", {pos}, "
                      f"\"the native backend cannot emit `%s`\", "
                      f"{_c_string(op.name)});")

    def _require(self, instr: Instr, fn: GFunction, pos: str) -> None:
        cond = self._operand(instr.args[0])
        text = instr.meta.get("text") or instr.meta.get("message") or ""
        kind = "ContractViolation" if instr.op == Op.REQUIRE else "AssertionFailed"
        label = _c_string(str(text) if text else instr.op.name.lower())
        self.emit(f"    if (!g_truthy({cond}, {pos}))")
        self.emit(f"        g_raise({_c_string(kind)}, {pos}, "
                  f"\"constraint does not hold: %s\", {label});")

    def _record(self, instr: Instr, fn: GFunction) -> None:
        names = instr.meta.get("fields") or []
        if not names and isinstance(instr.type, T.RecordType):
            names = list(getattr(instr.type, "fields", {}) or {})
        vals = ", ".join(self._operand(a) for a in instr.args)
        cname = ", ".join(_c_string(str(n)) for n in names) or ""
        n = len(instr.args)
        name = _c_string(getattr(instr.type, "name", "Record"))
        if n == 0:
            self._store(instr, fn, f"g_record({name}, 0, NULL, NULL)")
        else:
            self._store(instr, fn,
                        f"g_record({name}, {n}, (const char *[]){{{cname}}}, "
                        f"(const GValue[]){{{vals}}})")

    def _construct(self, instr: Instr, fn: GFunction) -> None:
        tag = str(instr.meta.get("tag", ""))
        ty = instr.type
        arg = self._operand(instr.args[0]) if instr.args else "g_unit()"
        if isinstance(ty, T.ResultType):
            self._store(instr, fn, f"g_result({1 if tag == 'ok' else 0}, {arg})")
        elif isinstance(ty, T.OptionType):
            self._store(instr, fn, f"g_option({1 if tag == 'some' else 0}, {arg})")
        else:
            enum = _c_string(str(getattr(ty, "name", "") or ""))
            if instr.args:
                self._store(instr, fn,
                            f"g_variant({enum}, {_c_string(tag)}, 2, "
                            f"(const GValue[]){{g_unit(), {arg}}})")
            else:
                self._store(instr, fn,
                            f"g_variant({enum}, {_c_string(tag)}, 1, "
                            f"(const GValue[]){{g_unit()}})")

    def _builtin(self, instr: Instr, pos: str) -> str:
        name = str(instr.meta.get("name", ""))
        args = [self._operand(a) for a in instr.args]
        n = len(args)
        call = ", ".join(args)

        if name in ("print", "println", "print_raw", "eprint"):
            # All four are variadic and space-join their arguments
            # (`std/library.py::_print`); taking only the first was a bug that
            # showed up as a line with its label and none of its values.
            target = {"print": "g_println_n", "println": "g_println_n",
                      "print_raw": "g_print_raw_n", "eprint": "g_eprint_n"}[name]
            if n == 0:
                return f"({target}(0, NULL), g_unit())"
            joined = ", ".join(args)
            return f"({target}({n}, (const GValue[]){{{joined}}}), g_unit())"
        if name == "len":
            return f"g_len({args[0]})"
        if name == "to_text" or name == "str":
            return f"g_text(g_to_text({args[0]}))"
        if name == "float":
            return f"g_to_float({args[0]}, {pos})"
        if name == "int":
            return f"g_to_int({args[0]}, {pos})"
        if name == "bool":
            return f"g_bool(g_truthy({args[0]}, {pos}))"
        if name == "io.read_file":
            return f"g_io_read_file({args[0]}, {pos})"
        if name == "io.write_file":
            return f"g_io_write_file({args[0]}, {args[1]}, {pos})"
        if name == "io.append_file":
            return f"g_io_append_file({args[0]}, {args[1]}, {pos})"
        if name == "io.exists":
            return f"g_io_exists({args[0]}, {pos})"
        if name == "abs" or name == "math.abs":
            return f"g_abs({args[0]}, {pos})"
        if name == "contains":
            return f"g_contains({args[0]}, {args[1]}, {pos})"
        if name in ("min", "math.min"):
            if n == 1:
                return f"g_min_of({args[0]}, {pos})"
            return f"g_min({args[0]}, {args[1]}, {pos})"
        if name in ("max", "math.max"):
            if n == 1:
                return f"g_max_of({args[0]}, {pos})"
            return f"g_max({args[0]}, {args[1]}, {pos})"
        if name == "sum":
            return f"g_sum({args[0]}, {pos})"
        if name == "sorted":
            return f"g_sorted({args[0]}, {pos})"
        if name == "range":
            if n == 0:
                return ('(g_raise("ValueError", ' + pos + ', '
                        '"`range` takes at least 1 argument"), g_unit())')
            return f"g_range({n}, (const GValue[]){{{call}}}, {pos})"
        if name == "__range":
            if n < 2:
                return ('(g_raise("BadGIR", ' + pos + ', '
                        '"`a..b` needs two bounds"), g_unit())')
            inclusive = f"g_truthy({args[2]}, {pos})" if n > 2 else "0"
            return f"g_range_span({args[0]}, {args[1]}, {inclusive}, {pos})"
        if name == "__iter_list":
            return f"{args[0]}"
        if name == "__list_get":
            return f"g_index({args[0]}, {args[1]}, {pos})"
        if name.startswith("math."):
            fn_name = {
                "math.round": "g_math_round", "math.floor": "g_math_floor",
                "math.ceil": "g_math_ceil", "math.sqrt": "g_math_sqrt",
                "math.pow": "g_math_pow", "math.clamp": "g_math_clamp",
                "math.log": "g_math_log", "math.exp": "g_math_exp",
            }.get(name)
            if fn_name:
                return f"{fn_name}({call}, {pos})"
        return f'(g_raise("BadGIR", {pos}, "the native backend cannot call ' \
               f'`%s`", {_c_string(name)}), g_unit())'


def _c_float(value: float) -> str:
    if value != value:
        return "NAN"
    if value == float("inf"):
        return "INFINITY"
    if value == float("-inf"):
        return "-INFINITY"
    return repr(value)


def pick_entry(program: GProgram, preferred: str = "main") -> str:
    """The same rule as `driver.find_entry`, so both paths agree."""
    if preferred in program.functions:
        return preferred
    if program.entry and program.entry in program.functions:
        return program.entry
    for name in program.functions:
        if name.endswith(".main") or name == "main":
            return name
    return program.entry or "main"


def generate_c(program: GProgram, source_path: str = "<program>",
               entry: Optional[str] = None) -> str:
    return CGenerator(program, source_path, entry).generate()
