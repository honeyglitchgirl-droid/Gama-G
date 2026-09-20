"""The WebAssembly backend (audit priority 13).

This module emits a real WebAssembly binary: LEB128-encoded, sectioned, with a
type section, function section, memory, exports and a code section.  It is
verified by **decoding it back** and by structural conformance checks, because
there is no WASM runtime in this repository's test environment.  That
limitation is stated here and in the CLI output rather than left for a user to
discover: a module this backend produces has never been executed by a WASM
interpreter in CI, so what is claimed is *conformance*, not *behaviour*.

Scope is deliberately narrow and stated by `unsupported()`, the same way the
native backend states its own: numeric computation and control flow over a
function whose parameters and result are numbers.  Anything needing the Gama-G
runtime -- text, lists, maps, records, the audit chain, capabilities, tensors --
is refused with a reason, because a WASM module that silently dropped a secret
guard would be worse than no module.

Why not reuse the C backend?  Because the point of priority 13 in the audit is
portability and a sandbox boundary, and routing WASM through C would mean the
artifact was a C program with extra steps.  This emits the binary format.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..gir.ir import GFunction, GProgram, Instr, Op
from ..semantic import types as T

MAGIC = b"\x00asm"
VERSION = b"\x01\x00\x00\x00"

# Section ids, from the WebAssembly specification's binary format.
S_CUSTOM, S_TYPE, S_IMPORT, S_FUNCTION, S_TABLE, S_MEMORY, S_GLOBAL, \
    S_EXPORT, S_START, S_ELEM, S_CODE, S_DATA = range(12)

# Value types.
I32, I64, F32, F64 = 0x7F, 0x7E, 0x7D, 0x7C

# Opcodes this backend emits.
OP_UNREACHABLE = 0x00
OP_NOP = 0x01
OP_BLOCK = 0x02
OP_LOOP = 0x03
OP_IF = 0x04
OP_ELSE = 0x05
OP_END = 0x0B
OP_BR = 0x0C
OP_BR_IF = 0x0D
OP_RETURN = 0x0F
OP_CALL = 0x10
OP_DROP = 0x1A
OP_SELECT = 0x1B
OP_LOCAL_GET = 0x20
OP_LOCAL_SET = 0x21
OP_LOCAL_TEE = 0x22
OP_GLOBAL_GET = 0x23
OP_I32_CONST = 0x41
OP_I64_CONST = 0x42
OP_F64_CONST = 0x44
OP_I32_EQZ = 0x45
OP_I32_EQ = 0x46
OP_I32_NE = 0x47
OP_I32_LT_S = 0x48
OP_I32_GT_S = 0x4A
OP_I32_LE_S = 0x4C
OP_I32_GE_S = 0x4E
OP_I64_ADD = 0x7C
OP_I64_SUB = 0x7D
OP_I64_MUL = 0x7E
OP_I64_DIV_S = 0x7F
OP_I64_REM_S = 0x81
OP_I64_EQ = 0x51
OP_I64_NE = 0x52
OP_I64_LT_S = 0x53
OP_I64_GT_S = 0x55
OP_I64_LE_S = 0x57
OP_I64_GE_S = 0x59
OP_F64_ADD = 0xA0
OP_F64_SUB = 0xA1
OP_F64_MUL = 0xA2
OP_F64_DIV = 0xA3
OP_F64_EQ = 0x61
OP_F64_NE = 0x62
OP_F64_LT = 0x63
OP_F64_GT = 0x64
OP_F64_LE = 0x65
OP_F64_GE = 0x66


# ---------------------------------------------------------------------------
# Encoding primitives
# ---------------------------------------------------------------------------

def uleb(value: int) -> bytes:
    """Unsigned LEB128."""
    if value < 0:
        raise ValueError(f"uleb128 cannot encode a negative value: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def sleb(value: int) -> bytes:
    """Signed LEB128."""
    out = bytearray()
    more = True
    while more:
        byte = value & 0x7F
        value >>= 7
        sign = byte & 0x40
        if (value == 0 and not sign) or (value == -1 and sign):
            more = False
            out.append(byte)
        else:
            out.append(byte | 0x80)
    return bytes(out)


def name(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return uleb(len(encoded)) + encoded


def section(section_id: int, payload: bytes) -> bytes:
    return bytes([section_id]) + uleb(len(payload)) + payload


def vector(items: Sequence[bytes]) -> bytes:
    return uleb(len(items)) + b"".join(items)


# ---------------------------------------------------------------------------
# Support analysis
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    what: str
    reason: str
    where: str = ""

    def render(self) -> str:
        at = f" at {self.where}" if self.where else ""
        return f"{self.what}{at} -- {self.reason}"


#: The GIR ops this backend can express in WASM.
#:
#: `JUMP` and `JUMP_IF` are absent on purpose.  WebAssembly has structured
#: control flow and no goto, so an arbitrary control-flow graph has to be
#: restructured into blocks and loops before it can be emitted.  That is a real
#: piece of work (a relooper) and it is not done here; emitting `unreachable` in
#: its place would produce a module that *validates* and then traps at run time,
#: which is the worst of the three options.  A program with control flow is
#: refused with that explanation instead.
SUPPORTED_OPS = frozenset({
    Op.CONST, Op.COPY, Op.BINOP, Op.UNOP, Op.RETURN, Op.CALL,
})

NUMERIC = (T.IntType, T.FloatType, T.BoolType)


def _pos_of(instr: Instr) -> str:
    p = getattr(instr, "pos", None)
    if p is None:
        return ""
    return f"{getattr(p, 'path', '')}:{getattr(p, 'line', 0)}:{getattr(p, 'col', 0)}"


def _is_numeric(ty: Any) -> bool:
    return isinstance(ty, NUMERIC)


@dataclass
class Analysis:
    """Which functions a module can carry, and why the others cannot.

    A Gama-G program usually has one function that is pure numeric computation
    (the intent's work) and one that performs I/O (the generated entry point,
    which prints the outcome).  Refusing the whole program because the entry
    point prints would be useless; silently dropping the entry point would be
    dishonest.  So both lists are reported and the caller decides.
    """

    compilable: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, List[Problem]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.compilable)

    @property
    def problems(self) -> List[Problem]:
        out: List[Problem] = []
        for fname, reasons in self.skipped:
            for reason in reasons:
                out.append(Problem(f"`{fname}`: {reason.what}", reason.reason,
                                   reason.where))
        return out

    def render(self) -> List[str]:
        out = [f"compilable: {', '.join(self.compilable) or 'none'}"]
        for fname, reasons in self.skipped:
            out.append(f"skipped `{fname}`:")
            for reason in reasons[:4]:
                out.append(f"    {reason.render()}")
        return out


def _function_problems(fname: str, fn: GFunction,
                       names: Sequence[str]) -> List[Problem]:
    problems: List[Problem] = []
    non_numeric = [s for s in fn.slots if not _is_numeric(s.type)]
    if non_numeric:
        problems.append(Problem(
            f"a `{non_numeric[0].type.render()}` value",
            "the WASM backend compiles numeric computation only; text, lists, "
            "maps and records need the Gama-G runtime, which a sandboxed "
            "module does not have"))
    if not _is_numeric(fn.ret) and not isinstance(fn.ret, T.UnitType):
        problems.append(Problem(
            f"a `{fn.ret.render()}` result",
            "the WASM backend compiles numeric results only"))
    for block in fn.blocks:
        for instr in block.instrs:
            where = _pos_of(instr)
            if instr.op not in SUPPORTED_OPS:
                reason = ("WebAssembly has structured control flow and no goto; "
                          "turning this into blocks and loops is not "
                          "implemented"
                          if instr.op in (Op.JUMP, Op.JUMP_IF) else
                          "not expressible in the WASM subset this backend emits")
                problems.append(Problem(
                    f"`{instr.op.name.lower()}`", reason, where))
                continue
            if instr.op == Op.BINOP:
                operator = str(instr.meta.get("operator", ""))
                if operator not in ("+", "-", "*", "/", "%", "<", ">", "<=", ">="):
                    problems.append(Problem(
                        f"operator `{operator}`",
                        "WebAssembly has no opcode for it, and this backend "
                        "does not synthesise one", where))
            if instr.op == Op.CALL:
                callee = str(instr.meta.get("callee", ""))
                if callee not in names:
                    problems.append(Problem(
                        f"call to `{callee}`", "no such function", where))
            if instr.op == Op.CONST:
                for arg in instr.args:
                    if isinstance(arg.value, str) or not isinstance(
                            arg.value, (int, float, bool)):
                        problems.append(Problem(
                            f"constant {arg.value!r}",
                            "the WASM backend emits numeric constants only",
                            where))
    return problems


def analyze(program: GProgram) -> Analysis:
    """Decide what a module can carry, before emitting anything."""
    analysis = Analysis()
    names = list(program.functions)
    for fname, fn in program.functions.items():
        problems = _function_problems(fname, fn, names)
        if problems:
            analysis.skipped.append((fname, problems))
        else:
            analysis.compilable.append(fname)
    return analysis


def unsupported(program: GProgram) -> List[Problem]:
    """Kept for symmetry with the native backend: problems, or an empty list.

    Unlike the native backend an empty result here does *not* mean the whole
    program was compiled -- use `analyze()` for that distinction.
    """
    analysis = analyze(program)
    if not analysis.compilable:
        return analysis.problems
    return []


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _wasm_type(ty: Any) -> int:
    if isinstance(ty, T.BoolType):
        return I32
    if isinstance(ty, T.IntType):
        return I64 if ty.bits > 32 else I32
    return F64


@dataclass
class WasmFunction:
    name: str
    params: List[int]
    results: List[int]
    locals_: List[int]
    body: bytes


class WasmGenerator:
    """Emit a WASM module for a program this backend has agreed to compile."""

    def __init__(self, program: GProgram, entry: Optional[str] = None):
        self.program = program
        self.entry = entry or _pick_entry(program)
        self.index: Dict[str, int] = {}
        self.functions: List[WasmFunction] = []

    def generate(self) -> bytes:
        exported = list(analyze(self.program).compilable)
        if not exported:
            raise ValueError(
                "no function in this program can be expressed in the WASM "
                "subset; call analyze() and report the reasons instead")
        for position, fname in enumerate(exported):
            self.index[fname] = position

        for fname in exported:
            self.functions.append(self._function(fname,
                                                 self.program.functions[fname]))

        types = self._type_section()
        type_indexes = self._assign_type_indexes(exported)

        out = bytearray()
        out += MAGIC + VERSION
        out += section(S_TYPE, vector(types))
        out += section(S_FUNCTION,
                       vector([uleb(type_indexes[f]) for f in exported]))
        out += section(S_MEMORY, vector([b"\x00" + uleb(1)]))
        exports = []
        for fname in exported:
            exports.append(name(fname) + b"\x00" + uleb(self.index[fname]))
        out += section(S_EXPORT, vector(exports))
        bodies = []
        for fn in self.functions:
            local_groups = _group(fn.locals_)
            payload = vector([uleb(n) + bytes([t]) for n, t in local_groups]) \
                + fn.body
            bodies.append(uleb(len(payload)) + payload)
        out += section(S_CODE, vector(bodies))
        return bytes(out)

    def _assign_type_indexes(self, exported: Sequence[str]) -> Dict[str, int]:
        seen: List[Tuple[bytes, int]] = []
        out: Dict[str, int] = {}
        for fname in exported:
            fn = self.program.functions[fname]
            key = self._type_key(fn)
            for existing, index in seen:
                if existing == key:
                    out[fname] = index
                    break
            else:
                out[fname] = len(seen)
                seen.append((key, len(seen)))
        return out

    def _type_key(self, fn: GFunction) -> bytes:
        params = [_wasm_type(s.type) for s in fn.slots[:len(fn.param_names or [])]]
        results = [] if isinstance(fn.ret, T.UnitType) else [_wasm_type(fn.ret)]
        return b"\x60" + vector([bytes([p]) for p in params]) \
            + vector([bytes([r]) for r in results])

    def _type_section(self) -> List[bytes]:
        keys: List[bytes] = []
        for fname in [n for n in self.program.functions
                      if n in self.index]:
            key = self._type_key(self.program.functions[fname])
            if key not in keys:
                keys.append(key)
        return keys

    def _function(self, fname: str, fn: GFunction) -> WasmFunction:
        params = [_wasm_type(s.type) for s in fn.slots[:len(fn.param_names or [])]]
        results = [] if isinstance(fn.ret, T.UnitType) else [_wasm_type(fn.ret)]
        # Every GIR slot becomes a WASM local; parameters are the first locals.
        locals_ = [_wasm_type(s.type) for s in fn.slots]
        body = bytearray()
        labels: Dict[str, int] = {}
        for block in fn.blocks:
            labels[block.id] = 0

        for block in fn.blocks:
            for instr in block.instrs:
                body += self._instr(instr, fn)
        body += bytes([OP_UNREACHABLE])
        return WasmFunction(name=fname, params=params, results=results,
                            locals_=locals_[len(params):], body=bytes(body))

    def _operand(self, arg: Any, fn: GFunction) -> bytes:
        if arg.kind == "slot":
            return bytes([OP_LOCAL_GET]) + uleb(arg.index)
        if arg.kind == "const":
            return self._const(arg.value, arg.type)
        return bytes([OP_I64_CONST]) + sleb(0)

    def _const(self, value: Any, ty: Any) -> bytes:
        if isinstance(value, bool):
            return bytes([OP_I32_CONST]) + sleb(1 if value else 0)
        if isinstance(value, int):
            if isinstance(ty, T.IntType) and ty.bits <= 32:
                return bytes([OP_I32_CONST]) + sleb(value)
            return bytes([OP_I64_CONST]) + sleb(value)
        if isinstance(value, float):
            return bytes([OP_F64_CONST]) + struct.pack("<d", value)
        return bytes([OP_I64_CONST]) + sleb(0)

    def _instr(self, instr: Instr, fn: GFunction) -> bytes:
        out = bytearray()
        op = instr.op
        if op == Op.CONST or op == Op.COPY:
            out += self._operand(instr.args[0], fn)
            out += bytes([OP_LOCAL_SET]) + uleb(instr.dst)
        elif op == Op.BINOP:
            operator = str(instr.meta.get("operator", ""))
            out += self._operand(instr.args[0], fn)
            out += self._operand(instr.args[1], fn)
            out += self._binop(operator, instr.args[0].type)
            out += bytes([OP_LOCAL_SET]) + uleb(instr.dst)
        elif op == Op.UNOP:
            operator = str(instr.meta.get("operator", ""))
            if operator == "-":
                # Negation is `0 - x`; the operand order matters, so 0 goes on
                # the stack first.
                out += self._const(0, instr.args[0].type)
                out += self._operand(instr.args[0], fn)
                out += self._binop("-", instr.args[0].type)
            elif operator in ("!", "not"):
                out += self._operand(instr.args[0], fn)
                out += bytes([OP_I32_EQZ])
            elif operator == "+":
                out += self._operand(instr.args[0], fn)
            else:
                return bytes([OP_UNREACHABLE])
            if instr.dst >= 0:
                out += bytes([OP_LOCAL_SET]) + uleb(instr.dst)
        elif op == Op.RETURN:
            if isinstance(fn.ret, T.UnitType):
                out += bytes([OP_RETURN])
            else:
                if not instr.args:
                    out += self._const(0, fn.ret)
                else:
                    out += self._operand(instr.args[0], fn)
                out += bytes([OP_RETURN])
        elif op == Op.CALL:
            callee = str(instr.meta.get("callee", ""))
            for arg in instr.args:
                out += self._operand(arg, fn)
            out += bytes([OP_CALL]) + uleb(self.index.get(callee, 0))
            if instr.dst >= 0:
                out += bytes([OP_LOCAL_SET]) + uleb(instr.dst)
        return bytes(out)

    def _binop(self, operator: str, ty: Any) -> bytes:
        if isinstance(ty, T.BoolType):
            ty = T.IntType(bits=32, signed=True)
        if isinstance(ty, T.IntType) and ty.bits <= 32:
            table = {"+": b"\x6a", "-": b"\x6b", "*": b"\x6c", "/": b"\x6d",
                     "%": b"\x6f", "==": bytes([OP_I32_EQ]),
                     "!=": bytes([OP_I32_NE]), "<": bytes([OP_I32_LT_S]),
                     ">": bytes([OP_I32_GT_S]), "<=": bytes([OP_I32_LE_S]),
                     ">=": bytes([OP_I32_GE_S])}
        elif isinstance(ty, T.IntType):
            table = {"+": bytes([OP_I64_ADD]), "-": bytes([OP_I64_SUB]),
                     "*": bytes([OP_I64_MUL]), "/": bytes([OP_I64_DIV_S]),
                     "%": bytes([OP_I64_REM_S]), "==": bytes([OP_I64_EQ]),
                     "!=": bytes([OP_I64_NE]), "<": bytes([OP_I64_LT_S]),
                     ">": bytes([OP_I64_GT_S]), "<=": bytes([OP_I64_LE_S]),
                     ">=": bytes([OP_I64_GE_S])}
        else:
            table = {"+": bytes([OP_F64_ADD]), "-": bytes([OP_F64_SUB]),
                     "*": bytes([OP_F64_MUL]), "/": bytes([OP_F64_DIV]),
                     "==": bytes([OP_F64_EQ]), "!=": bytes([OP_F64_NE]),
                     "<": bytes([OP_F64_LT]), ">": bytes([OP_F64_GT]),
                     "<=": bytes([OP_F64_LE]), ">=": bytes([OP_F64_GE])}
        # Every operator reaching here has been accepted by the support
        # analysis, so an absent entry is a bug in this module, not a limit to
        # report politely.
        return table[operator]


def _group(types: Sequence[int]) -> List[Tuple[int, int]]:
    """Run-length encode a local declaration list, as the format requires."""
    out: List[Tuple[int, int]] = []
    for ty in types:
        if out and out[-1][1] == ty:
            out[-1] = (out[-1][0] + 1, ty)
        else:
            out.append((1, ty))
    return out


def _pick_entry(program: GProgram, preferred: str = "main") -> str:
    if preferred in program.functions:
        return preferred
    return program.entry or preferred


def generate(program: GProgram, entry: Optional[str] = None) -> bytes:
    return WasmGenerator(program, entry).generate()


# ---------------------------------------------------------------------------
# Decoding -- how this backend is tested without a WASM runtime
# ---------------------------------------------------------------------------

@dataclass
class DecodedSection:
    id: int
    payload: bytes


@dataclass
class DecodedModule:
    magic: bytes = b""
    version: bytes = b""
    sections: List[DecodedSection] = field(default_factory=list)

    @property
    def valid_header(self) -> bool:
        return self.magic == MAGIC and self.version == VERSION

    def get(self, section_id: int) -> Optional[bytes]:
        for sec in self.sections:
            if sec.id == section_id:
                return sec.payload
        return None

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "valid_header": self.valid_header,
            "sections": [s.id for s in self.sections],
        }
        types = self.get(S_TYPE)
        if types is not None:
            out["type_count"], _ = _read_vec_len(types)
        funcs = self.get(S_FUNCTION)
        if funcs is not None:
            out["function_count"], _ = _read_vec_len(funcs)
        exports = self.get(S_EXPORT)
        if exports is not None:
            count, rest = _read_vec_len(exports)
            out["export_count"] = count
            names = []
            cursor = rest
            for _ in range(count):
                length, cursor = _read_uleb(cursor)
                names.append(cursor[:length].decode("utf-8", "replace"))
                cursor = cursor[length:]
                cursor = cursor[1:]                       # the export kind byte
                _index, cursor = _read_uleb(cursor)       # the function index
            out["exports"] = names
        code = self.get(S_CODE)
        if code is not None:
            out["code_count"], _ = _read_vec_len(code)
        return out


def _read_uleb(data: bytes, offset: int = 0) -> Tuple[int, bytes]:
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return result, data[offset:]


def _read_vec_len(data: bytes) -> Tuple[int, bytes]:
    return _read_uleb(data)


def decode(blob: bytes) -> DecodedModule:
    """Parse a module back into its sections.

    This is how the backend is checked without a WASM runtime: if the bytes
    decode into well-formed sections whose counts match what was emitted, the
    encoder is at least self-consistent and structurally conformant.
    """
    module = DecodedModule(magic=blob[:4], version=blob[4:8])
    offset = 8
    while offset < len(blob):
        section_id = blob[offset]
        offset += 1
        length, rest = _read_uleb(blob[offset:])
        consumed = len(blob) - offset - len(rest)
        offset += consumed
        module.sections.append(DecodedSection(id=section_id,
                                              payload=blob[offset:offset + length]))
        offset += length
    return module


def validate(blob: bytes) -> List[str]:
    """Structural conformance checks.  Not execution, and not claimed as such."""
    problems: List[str] = []
    module = decode(blob)
    if not module.valid_header:
        problems.append("the magic number or version is wrong")
    ids = [s.id for s in module.sections]
    if ids != sorted(ids):
        problems.append("sections are out of order, which the format forbids")
    if len(set(ids)) != len(ids):
        problems.append("a section appears twice, which the format forbids")
    for required in (S_TYPE, S_FUNCTION, S_EXPORT, S_CODE):
        if module.get(required) is None:
            problems.append(f"section {required} is missing")
    summary = module.summary()
    if summary.get("function_count") != summary.get("code_count"):
        problems.append("the function section and the code section disagree "
                        "about how many functions there are")
    return problems


# ---------------------------------------------------------------------------
# Disassembly -- verifying emitted code without a WASM runtime
# ---------------------------------------------------------------------------

OPCODE_NAMES: Dict[int, str] = {
    OP_UNREACHABLE: "unreachable", OP_NOP: "nop", OP_BLOCK: "block",
    OP_LOOP: "loop", OP_IF: "if", OP_ELSE: "else", OP_END: "end",
    OP_BR: "br", OP_BR_IF: "br_if", OP_RETURN: "return", OP_CALL: "call",
    OP_DROP: "drop", OP_SELECT: "select",
    OP_LOCAL_GET: "local.get", OP_LOCAL_SET: "local.set",
    OP_LOCAL_TEE: "local.tee", OP_GLOBAL_GET: "global.get",
    OP_I32_CONST: "i32.const", OP_I64_CONST: "i64.const",
    OP_F64_CONST: "f64.const", OP_I32_EQZ: "i32.eqz",
    OP_I32_EQ: "i32.eq", OP_I32_NE: "i32.ne", OP_I32_LT_S: "i32.lt_s",
    OP_I32_GT_S: "i32.gt_s", OP_I32_LE_S: "i32.le_s", OP_I32_GE_S: "i32.ge_s",
    OP_I64_ADD: "i64.add", OP_I64_SUB: "i64.sub", OP_I64_MUL: "i64.mul",
    OP_I64_DIV_S: "i64.div_s", OP_I64_REM_S: "i64.rem_s",
    OP_I64_EQ: "i64.eq", OP_I64_NE: "i64.ne", OP_I64_LT_S: "i64.lt_s",
    OP_I64_GT_S: "i64.gt_s", OP_I64_LE_S: "i64.le_s", OP_I64_GE_S: "i64.ge_s",
    OP_F64_ADD: "f64.add", OP_F64_SUB: "f64.sub", OP_F64_MUL: "f64.mul",
    OP_F64_DIV: "f64.div", OP_F64_EQ: "f64.eq", OP_F64_NE: "f64.ne",
    OP_F64_LT: "f64.lt", OP_F64_GT: "f64.gt", OP_F64_LE: "f64.le",
    OP_F64_GE: "f64.ge",
    0x6A: "i32.add", 0x6B: "i32.sub", 0x6C: "i32.mul", 0x6D: "i32.div_s",
    0x6F: "i32.rem_s",
}

#: Opcodes that carry an immediate, and how to read it.
IMMEDIATES: Dict[int, str] = {
    OP_LOCAL_GET: "uleb", OP_LOCAL_SET: "uleb", OP_LOCAL_TEE: "uleb",
    OP_GLOBAL_GET: "uleb", OP_CALL: "uleb", OP_BR: "uleb", OP_BR_IF: "uleb",
    OP_I32_CONST: "sleb", OP_I64_CONST: "sleb", OP_F64_CONST: "f64",
}


def _read_sleb(data: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
    if byte & 0x40:
        result -= 1 << shift
    return result, offset


def disassemble(body: bytes) -> List[str]:
    """Render a function body as mnemonic text.

    This is how the emitted code is checked in the absence of a WASM runtime:
    the instruction sequence can be read and compared against what the GIR said.
    It is a reader, not a validator of stack discipline.
    """
    out: List[str] = []
    offset = 0
    while offset < len(body):
        opcode = body[offset]
        offset += 1
        mnemonic = OPCODE_NAMES.get(opcode, f"unknown(0x{opcode:02x})")
        kind = IMMEDIATES.get(opcode)
        if kind == "uleb":
            value, offset = _read_uleb_at(body, offset)
            out.append(f"{mnemonic} {value}")
        elif kind == "sleb":
            value, offset = _read_sleb(body, offset)
            out.append(f"{mnemonic} {value}")
        elif kind == "f64":
            value = struct.unpack("<d", body[offset:offset + 8])[0]
            offset += 8
            out.append(f"{mnemonic} {value!r}")
        else:
            out.append(mnemonic)
    return out


def _read_uleb_at(data: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def disassemble_module(blob: bytes) -> Dict[str, List[str]]:
    """Every function body in a module, as mnemonics, keyed by export name."""
    module = decode(blob)
    code = module.get(S_CODE)
    exports = module.get(S_EXPORT)
    if code is None:
        return {}
    names: List[str] = []
    if exports is not None:
        count, cursor = _read_vec_len(exports)
        for _ in range(count):
            length, cursor = _read_uleb(cursor)
            names.append(cursor[:length].decode("utf-8", "replace"))
            cursor = cursor[length:]
            cursor = cursor[1:]
            _index, cursor = _read_uleb(cursor)

    count, cursor = _read_vec_len(code)
    out: Dict[str, List[str]] = {}
    for position in range(count):
        size, cursor = _read_uleb(cursor)
        body = cursor[:size]
        cursor = cursor[size:]
        # Skip the local declarations to reach the instruction stream.
        local_groups, offset = _read_uleb_at(body, 0)
        for _ in range(local_groups):
            _n, offset = _read_uleb_at(body, offset)
            offset += 1
        label = names[position] if position < len(names) else f"func{position}"
        out[label] = disassemble(body[offset:])
    return out
