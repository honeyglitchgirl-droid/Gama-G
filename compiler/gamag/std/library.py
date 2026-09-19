"""The Gama-G standard library (spec section 28).

Every builtin declares its parameter types, return type, effects (spec
section 7) and required capabilities (spec section 12).  Those declarations
are what make the effect checker and capability checker possible: a ``pure``
function cannot call ``io.read_file`` because the effect sets are
incompatible, and no function can call it at all unless ``FileRead`` was
granted.

Modules implemented here follow the specification's grouping -- core, text,
math, collections, time, io, crypto, secrets, audit, identity, policy,
tensor, autodiff, model, dataset, medical.  Modules the specification lists
but v0.1 does not implement (fhir, terminology, provenance, consent,
database, http, messaging, workflow, transaction, observability, accelerator,
process, concurrency) are reported honestly by ``ggc doc`` rather than being
silently stubbed.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import math as _math
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..diagnostics import GamaRuntimeFault, TypeFault
from ..runtime.context import KNOWN_CAPABILITIES
from ..runtime.tensor import GTensor, Tape, TapeNode, parameter as _parameter
from ..runtime.values import (GCapability, GDuration, GFunction, GInstant,
                              GOption, GRecord, GResult, GSecret, GUnit, GUuid,
                              GVariant, UNIT, canonical, display, to_text,
                              truthy, type_name, unwrap_secret)
from ..semantic import types as T


class BuiltinFault(GamaRuntimeFault):
    def __init__(self, message: str, **ctx):
        super().__init__("BuiltinFault", message, None, ctx)


@dataclass
class Builtin:
    """A standard-library function declaration plus its implementation."""

    name: str
    params: Tuple[str, ...]
    argtypes: Tuple[Optional[T.Type], ...] = ()
    ret: Optional[T.Type] = None
    infer: Optional[Callable[[List[T.Type]], T.Type]] = None
    impl: Optional[Callable[..., Any]] = None
    effects: Tuple[str, ...] = ()
    caps: Tuple[str, ...] = ()
    variadic: bool = False
    min_args: Optional[int] = None
    doc: str = ""

    @property
    def arity(self) -> Optional[int]:
        if self.variadic:
            return None
        return len(self.params)

    def fn_type(self, argtypes: Optional[List[T.Type]] = None) -> T.FnType:
        ret = self.ret if self.ret is not None else T.ANY
        if self.infer is not None and argtypes is not None:
            try:
                ret = self.infer(argtypes)
            except Exception:                            # noqa: BLE001
                ret = T.ANY
        declared = tuple(a if a is not None else T.ANY for a in self.argtypes)
        if self.variadic and argtypes:
            declared = tuple(argtypes)
        return T.FnType(declared, ret, param_names=self.params,
                        effects=self.effects, name=self.name)


@dataclass
class Constant:
    name: str
    type: T.Type
    value: Any
    doc: str = ""


BUILTINS: Dict[str, Builtin] = {}
CONSTANTS: Dict[str, Constant] = {}
MODULES: Dict[str, List[str]] = {}
PRELUDE: set = set()

# Modules the specification names but v0.1 does not implement.
UNIMPLEMENTED_MODULES: Dict[str, str] = {
    "fhir": "FHIR-compatible serialisation is a library/profile layer (spec 16)",
    "terminology": "terminology validation requires an external code system",
    "provenance": "planned for Phase 3 (spec 34)",
    "consent": "planned for Phase 3 (spec 34)",
    "database": "requires a native backend and driver model (Phase 3)",
    "http": "requires NetworkConnect capability plumbing (Phase 3)",
    "messaging": "planned for Phase 3",
    "workflow": "planned for Phase 3",
    "transaction": "the `transaction` block form is implemented in-language",
    "observability": "planned for Phase 3",
    "accelerator": "requires a GPU/accelerator backend (Phase 2/3)",
    "process": "requires ProcessSpawn capability plumbing",
    "concurrency": "structured `parallel` regions are implemented in-language",
    "train": "see `autodiff`, which provides the training primitives",
    "infer": "see `model.predict`",
    "identity_provider": "planned for Phase 3",
}


def reg(name: str, params: Sequence[str], *, ret: Optional[T.Type] = None,
        argtypes: Sequence[Optional[T.Type]] = (),
        infer: Optional[Callable[[List[T.Type]], T.Type]] = None,
        effects: Sequence[str] = (), caps: Sequence[str] = (),
        variadic: bool = False, min_args: Optional[int] = None,
        doc: str = "", prelude: bool = False,
        hidden: bool = False) -> Callable[[Callable], Callable]:
    """Decorator registering a builtin implementation."""

    def deco(f: Callable) -> Callable:
        b = Builtin(name=name, params=tuple(params),
                    argtypes=tuple(argtypes), ret=ret, infer=infer, impl=f,
                    effects=tuple(effects), caps=tuple(caps),
                    variadic=variadic, min_args=min_args, doc=doc)
        BUILTINS[name] = b
        if not hidden:
            module = name.split(".")[0] if "." in name else "core"
            MODULES.setdefault(module, []).append(name)
            if prelude or "." not in name:
                PRELUDE.add(name.split(".")[-1])
        return f
    return deco


def const(name: str, type_: T.Type, value: Any, doc: str = "") -> None:
    CONSTANTS[name] = Constant(name, type_, value, doc)
    module = name.split(".")[0]
    MODULES.setdefault(module, []).append(name)


# ======================================================================
# core  (spec section 28: Core)
# ======================================================================
def _flatten_args(args: Tuple[Any, ...]) -> List[str]:
    return [to_text(a) for a in args]


@reg("print", ("values",), ret=T.UNIT, effects=("io",), variadic=True,
     prelude=True,
     doc="Write values to standard output, space-separated, then a newline. "
         "Secret values are redacted rather than printed (spec section 12).")
def _print(ctx, *args):
    ctx.stdout.write(" ".join(_flatten_args(args)) + "\n")
    return UNIT


@reg("println", ("values",), ret=T.UNIT, effects=("io",), variadic=True,
     prelude=True, doc="Alias of `print`.")
def _println(ctx, *args):
    ctx.stdout.write(" ".join(_flatten_args(args)) + "\n")
    return UNIT


@reg("print_raw", ("values",), ret=T.UNIT, effects=("io",), variadic=True,
     doc="Write values to standard output with no terminating newline.")
def _print_raw(ctx, *args):
    ctx.stdout.write(" ".join(_flatten_args(args)))
    return UNIT


@reg("eprint", ("values",), ret=T.UNIT, effects=("io",), variadic=True,
     doc="Write values to standard error.")
def _eprint(ctx, *args):
    ctx.stderr.write(" ".join(_flatten_args(args)) + "\n")
    return UNIT


def _infer_len(argtypes):
    return T.I64


@reg("len", ("collection",), infer=lambda a: T.I64, prelude=True,
     doc="Length of a Text, List, Map, Set, Bytes or Tensor.")
def _len(ctx, value):
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, dict, set, frozenset, bytes, tuple)):
        return len(value)
    if isinstance(value, GTensor):
        return value.shape[0] if value.rank else 1
    if isinstance(value, GOption):
        return 1 if value.some else 0
    raise TypeFault(f"`len` does not apply to {type_name(value)}",
                    hint="len accepts Text, List, Map, Set, Bytes or Tensor")


@reg("range", ("args",), ret=T.ListType(T.I64), variadic=True, min_args=1,
     prelude=True, doc="range(n), range(a,b) or range(a,b,step) -> List<I64>.")
def _range(ctx, *args):
    if len(args) == 1:
        return list(range(int(args[0])))
    if len(args) == 2:
        return list(range(int(args[0]), int(args[1])))
    if len(args) == 3:
        step = int(args[2])
        if step == 0:
            raise BuiltinFault("`range` step cannot be zero")
        return list(range(int(args[0]), int(args[1]), step))
    raise BuiltinFault("`range` takes at most 3 arguments")


@reg("to_text", ("value",), ret=T.TEXT, prelude=True,
     doc="Explicit conversion to Text. Refuses secrets (spec section 8).")
def _to_text(ctx, value):
    return to_text(value)


@reg("str", ("value",), ret=T.TEXT, prelude=True, doc="Alias of `to_text`.")
def _str(ctx, value):
    return to_text(value)


def _numeric_cast(kind):
    def infer(argtypes):
        return kind
    return infer


@reg("int", ("value",), infer=lambda a: T.I64, prelude=True,
     doc="Convert to I64.")
def _int(ctx, value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise TypeFault(
                f"cannot convert {value} to I64 without truncation",
                hint="use `math.floor`, `math.ceil` or `math.round` first; "
                     "Gama-G performs no implicit unsafe conversion")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip(), 0)
        except ValueError:
            raise TypeFault(f"cannot parse {value!r} as an integer") from None
    if isinstance(value, GTensor) and value.size == 1:
        return int(value.data[0])
    if isinstance(value, GDuration):
        return int(value.seconds)
    raise TypeFault(f"cannot convert {type_name(value)} to I64")


@reg("float", ("value",), infer=lambda a: T.F64, prelude=True,
     doc="Convert to F64.")
def _float(ctx, value):
    if isinstance(value, (int, float, bool)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            raise TypeFault(f"cannot parse {value!r} as a float") from None
    if isinstance(value, GTensor) and value.size == 1:
        return float(value.data[0])
    if isinstance(value, GDuration):
        return float(value.seconds)
    raise TypeFault(f"cannot convert {type_name(value)} to F64")


@reg("bool", ("value",), ret=T.BOOL, prelude=True, doc="Convert to Bool.")
def _bool(ctx, value):
    return truthy(value)


def _infer_minmax(argtypes):
    if not argtypes:
        return T.ANY
    result = argtypes[0]
    for a in argtypes[1:]:
        result = T.unify(result, a)
    return result


@reg("min", ("a", "b"), infer=_infer_minmax, variadic=True, min_args=1,
     prelude=True, doc="Smallest of the arguments, or of a List.")
def _min(ctx, *args):
    values = args[0] if len(args) == 1 and isinstance(args[0], list) else args
    if isinstance(values, GTensor):
        return values.min()
    if not values:
        raise BuiltinFault("`min` requires at least one value")
    if any(isinstance(v, GSecret) for v in values):
        raise TypeFault("`min` cannot compare secret values")
    return min(values)


@reg("max", ("a", "b"), infer=_infer_minmax, variadic=True, min_args=1,
     prelude=True, doc="Largest of the arguments, or of a List.")
def _max(ctx, *args):
    values = args[0] if len(args) == 1 and isinstance(args[0], list) else args
    if isinstance(values, GTensor):
        return values.max()
    if not values:
        raise BuiltinFault("`max` requires at least one value")
    return max(values)


@reg("abs", ("x",), infer=lambda a: a[0] if a else T.ANY, prelude=True,
     doc="Absolute value.")
def _abs(ctx, value):
    return abs(value)


@reg("sum", ("values",), infer=lambda a: T.F64 if a and isinstance(
    a[0], T.ListType) and isinstance(a[0].elem, T.FloatType) else T.I64,
     prelude=True, doc="Sum of a List of numbers.")
def _sum(ctx, values):
    if isinstance(values, GTensor):
        return values.sum()
    return sum(values)


@reg("sorted", ("values",), infer=lambda a: a[0] if a else T.ANY, prelude=True,
     doc="A new sorted List.")
def _sorted(ctx, values, key=None, reverse=False):
    return sorted(values, key=key, reverse=bool(reverse))


@reg("reversed", ("values",), infer=lambda a: a[0] if a else T.ANY,
     prelude=True, doc="A new List in reverse order.")
def _reversed(ctx, values):
    return list(reversed(values))


@reg("enumerate", ("values",), infer=lambda a: T.ListType(T.TupleType(
    (T.I64, a[0].elem if a and isinstance(a[0], T.ListType) else T.ANY))),
     prelude=True, doc="List of (index, value) tuples.")
def _enumerate(ctx, values):
    return [(i, v) for i, v in enumerate(values)]


@reg("zip", ("a", "b"), infer=lambda a: T.ListType(T.TupleType(
    (a[0].elem if isinstance(a[0], T.ListType) else T.ANY,
     a[1].elem if len(a) > 1 and isinstance(a[1], T.ListType) else T.ANY))),
     prelude=True, doc="Pairwise List of tuples.")
def _zip(ctx, a, b):
    return [tuple(p) for p in zip(a, b)]


@reg("contains", ("haystack", "needle"), ret=T.BOOL, prelude=True,
     doc="Membership test for Text, List, Set, Map and Tensor-free collections.")
def _contains(ctx, haystack, needle):
    if isinstance(haystack, str):
        return needle in haystack
    if isinstance(haystack, dict):
        return needle in haystack
    if isinstance(haystack, (list, tuple, set, frozenset)):
        return any(_values_equal(x, needle) for x in haystack)
    raise TypeFault(f"`contains` does not apply to {type_name(haystack)}")


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, GTensor) or isinstance(b, GTensor):
        if isinstance(a, GTensor) and isinstance(b, GTensor):
            return a.equals(b)
        return False
    if isinstance(a, (GOption, GResult, GVariant, GRecord)) or \
            isinstance(b, (GOption, GResult, GVariant, GRecord)):
        return repr(a) == repr(b)
    return a == b


@reg("typeof", ("value",), ret=T.TEXT, prelude=True,
     doc="Runtime type name of a value.")
def _typeof(ctx, value):
    return type_name(value)


@reg("panic", ("message",), ret=T.NeverType(), effects=(), prelude=True,
     doc="Abort with an unrecoverable defect (spec section 19).")
def _panic(ctx, message, kind="Panic"):
    raise GamaRuntimeFault(str(kind), to_text(message))


@reg("exit", ("code",), ret=T.NeverType(), effects=("io",),
     argtypes=(T.I64,), doc="Terminate the program with an exit code.")
def _exit(ctx, code):
    raise SystemExit(int(code))


@reg("is_some", ("option",), ret=T.BOOL, prelude=True,
     doc="True when an Option holds a value.")
def _is_some(ctx, value):
    return isinstance(value, GOption) and value.some


@reg("is_none", ("option",), ret=T.BOOL, prelude=True,
     doc="True when an Option is none.")
def _is_none(ctx, value):
    return isinstance(value, GOption) and not value.some


@reg("is_ok", ("result",), ret=T.BOOL, prelude=True, doc="True for ok(..).")
def _is_ok(ctx, value):
    return isinstance(value, GResult) and value.ok


@reg("is_fail", ("result",), ret=T.BOOL, prelude=True, doc="True for fail(..).")
def _is_fail(ctx, value):
    return isinstance(value, GResult) and not value.ok


@reg("unwrap", ("value",), infer=lambda a: (
    a[0].inner if a and isinstance(a[0], T.OptionType)
    else a[0].ok if a and isinstance(a[0], T.ResultType) else T.ANY),
     prelude=True, doc="Extract an Option/Result payload or fault.")
def _unwrap(ctx, value):
    if isinstance(value, (GOption, GResult)):
        return value.unwrap()
    raise TypeFault(f"`unwrap` expects an Option or Result, got {type_name(value)}")


@reg("or_default", ("value", "default"), infer=lambda a: (
    a[0].inner if a and isinstance(a[0], T.OptionType) else T.ANY),
     prelude=True, doc="Option payload, or a default when none.")
def _or_default(ctx, value, default):
    if isinstance(value, GOption):
        return value.value if value.some else default
    if isinstance(value, GResult):
        return value.value if value.ok else default
    return value


@reg("assert_eq", ("left", "right", "message"), ret=T.UNIT, variadic=True,
     min_args=2, prelude=True, doc="Fault unless two values are equal.")
def _assert_eq(ctx, *args):
    left, right = args[0], args[1]
    message = args[2] if len(args) > 2 else None
    if not _values_equal(left, right):
        raise GamaRuntimeFault(
            "AssertionFailed",
            (to_text(message) + ": " if message else "")
            + f"expected {display(left)} == {display(right)}")
    return UNIT


@reg("error", ("message",), ret=T.ANY, prelude=True,
     doc="Construct a structured error value.")
def _error(ctx, message, kind="Error"):
    return GRecord("Error", {"kind": to_text(kind), "message": to_text(message)})


# ======================================================================
# math
# ======================================================================
for _name, _fn in (("sqrt", _math.sqrt), ("exp", _math.exp),
                   ("log", _math.log), ("log2", _math.log2),
                   ("log10", _math.log10), ("sin", _math.sin),
                   ("cos", _math.cos), ("tan", _math.tan),
                   ("asin", _math.asin), ("acos", _math.acos),
                   ("atan", _math.atan), ("sinh", _math.sinh),
                   ("cosh", _math.cosh), ("tanh", _math.tanh)):
    def _make(f):
        @reg(f"math.{f.__name__}", ("x",), ret=T.F64, argtypes=(T.F64,),
             effects=(), doc=f"Mathematical {f.__name__}.")
        def _impl(ctx, x, _f=f):
            try:
                return float(_f(x))
            except ValueError as exc:
                raise BuiltinFault(f"math.{_f.__name__}: {exc}") from None
        return _impl
    _make(_fn)


@reg("math.pow", ("base", "exp"), ret=T.F64, argtypes=(T.F64, T.F64),
     doc="Exponentiation.")
def _math_pow(ctx, base, exp):
    return float(_math.pow(base, exp))


@reg("math.atan2", ("y", "x"), ret=T.F64, argtypes=(T.F64, T.F64), doc="atan2.")
def _math_atan2(ctx, y, x):
    return float(_math.atan2(y, x))


@reg("math.hypot", ("a", "b"), ret=T.F64, argtypes=(T.F64, T.F64), doc="hypot.")
def _math_hypot(ctx, a, b):
    return float(_math.hypot(a, b))


@reg("math.floor", ("x",), ret=T.I64, argtypes=(T.F64,), doc="Round down.")
def _math_floor(ctx, x):
    return int(_math.floor(x))


@reg("math.ceil", ("x",), ret=T.I64, argtypes=(T.F64,), doc="Round up.")
def _math_ceil(ctx, x):
    return int(_math.ceil(x))


@reg("math.round", ("x",), ret=T.I64, argtypes=(T.F64,),
     doc="Round half away from zero.")
def _math_round(ctx, x):
    return int(_math.floor(x + 0.5)) if x >= 0 else int(_math.ceil(x - 0.5))


@reg("math.trunc", ("x",), ret=T.I64, argtypes=(T.F64,), doc="Truncate.")
def _math_trunc(ctx, x):
    return int(x)


@reg("math.signum", ("x",), ret=T.I64, doc="Sign as -1, 0 or 1.")
def _math_signum(ctx, x):
    return (x > 0) - (x < 0)


@reg("math.clamp", ("x", "lo", "hi"), infer=lambda a: a[0] if a else T.F64,
     doc="Clamp x into [lo, hi].")
def _math_clamp(ctx, x, lo, hi):
    return min(max(x, lo), hi)


@reg("math.div", ("a", "b"), ret=T.F64, argtypes=(T.F64, T.F64),
     doc="Exact division producing F64; faults on division by zero.")
def _math_div(ctx, a, b):
    if b == 0:
        raise BuiltinFault("division by zero", hint="return a Result instead "
                           "when zero is a reachable input (spec section 6)")
    return a / b


@reg("math.mod", ("a", "b"), ret=T.I64, argtypes=(T.I64, T.I64),
     doc="Integer modulo; faults on zero.")
def _math_mod(ctx, a, b):
    if b == 0:
        raise BuiltinFault("modulo by zero")
    return a % b


const("math.pi", T.F64, _math.pi, "The circle constant pi.")
const("math.e", T.F64, _math.e, "The base of natural logarithms.")
const("math.inf", T.F64, _math.inf, "Positive infinity.")
const("math.nan", T.F64, _math.nan, "Not a number.")
const("math.tau", T.F64, _math.tau, "2 * pi.")


# ======================================================================
# text
# ======================================================================
@reg("text.upper", ("s",), ret=T.TEXT, argtypes=(T.TEXT,), doc="Upper case.")
def _text_upper(ctx, s):
    return s.upper()


@reg("text.lower", ("s",), ret=T.TEXT, argtypes=(T.TEXT,), doc="Lower case.")
def _text_lower(ctx, s):
    return s.lower()


@reg("text.split", ("s", "sep"), ret=T.ListType(T.TEXT), variadic=True,
     min_args=1, doc="Split Text into a List.")
def _text_split(ctx, s, sep=None):
    return s.split(sep) if sep else s.split()


@reg("text.join", ("sep", "parts"), ret=T.TEXT,
     argtypes=(T.TEXT, T.ListType(T.TEXT)), doc="Join a List with a separator.")
def _text_join(ctx, sep, parts):
    return sep.join(to_text(p) for p in parts)


@reg("text.trim", ("s",), ret=T.TEXT, argtypes=(T.TEXT,), doc="Strip whitespace.")
def _text_trim(ctx, s):
    return s.strip()


@reg("text.contains", ("s", "needle"), ret=T.BOOL,
     argtypes=(T.TEXT, T.TEXT), doc="Substring test.")
def _text_contains(ctx, s, needle):
    return needle in s


@reg("text.starts_with", ("s", "prefix"), ret=T.BOOL,
     argtypes=(T.TEXT, T.TEXT), doc="Prefix test.")
def _text_starts_with(ctx, s, prefix):
    return s.startswith(prefix)


@reg("text.ends_with", ("s", "suffix"), ret=T.BOOL,
     argtypes=(T.TEXT, T.TEXT), doc="Suffix test.")
def _text_ends_with(ctx, s, suffix):
    return s.endswith(suffix)


@reg("text.replace", ("s", "old", "new"), ret=T.TEXT,
     argtypes=(T.TEXT, T.TEXT, T.TEXT), doc="Replace all occurrences.")
def _text_replace(ctx, s, old, new):
    return s.replace(old, new)


@reg("text.length", ("s",), ret=T.I64, argtypes=(T.TEXT,), doc="Character count.")
def _text_length(ctx, s):
    return len(s)


@reg("text.char_at", ("s", "i"), ret=T.CHAR, argtypes=(T.TEXT, T.I64),
     doc="Character at an index; faults if out of range.")
def _text_char_at(ctx, s, i):
    if not 0 <= i < len(s):
        raise BuiltinFault(f"index {i} out of range for Text of length {len(s)}")
    return s[i]


@reg("text.slice", ("s", "start", "end"), ret=T.TEXT, variadic=True,
     min_args=1, doc="Substring.")
def _text_slice(ctx, s, start=0, end=None):
    return s[start:] if end is None else s[start:end]


@reg("text.repeat", ("s", "n"), ret=T.TEXT, argtypes=(T.TEXT, T.I64),
     doc="Repeat Text n times.")
def _text_repeat(ctx, s, n):
    if n < 0:
        raise BuiltinFault("cannot repeat Text a negative number of times")
    return s * n


@reg("text.index_of", ("s", "needle"), ret=T.OptionType(T.I64),
     argtypes=(T.TEXT, T.TEXT), doc="First index of a substring, or none.")
def _text_index_of(ctx, s, needle):
    i = s.find(needle)
    return GOption(i >= 0, i if i >= 0 else None)


@reg("text.is_empty", ("s",), ret=T.BOOL, argtypes=(T.TEXT,), doc="Empty test.")
def _text_is_empty(ctx, s):
    return len(s) == 0


@reg("text.lines", ("s",), ret=T.ListType(T.TEXT), argtypes=(T.TEXT,),
     doc="Split into lines.")
def _text_lines(ctx, s):
    return s.splitlines()


@reg("text.pad_start", ("s", "width", "fill"), ret=T.TEXT, variadic=True,
     min_args=2, doc="Left-pad to a width.")
def _text_pad_start(ctx, s, width, fill=" "):
    return s.rjust(width, fill[0] if fill else " ")


@reg("text.pad_end", ("s", "width", "fill"), ret=T.TEXT, variadic=True,
     min_args=2, doc="Right-pad to a width.")
def _text_pad_end(ctx, s, width, fill=" "):
    return s.ljust(width, fill[0] if fill else " ")


@reg("text.format", ("template", "values"), ret=T.TEXT, variadic=True,
     min_args=1, effects=(), doc="Substitute {} placeholders positionally.")
def _text_format(ctx, template, *values):
    out = []
    idx = 0
    i = 0
    while i < len(template):
        if template.startswith("{}", i):
            if idx >= len(values):
                raise BuiltinFault(
                    f"text.format: placeholder {idx} has no argument")
            out.append(to_text(values[idx]))
            idx += 1
            i += 2
            continue
        out.append(template[i])
        i += 1
    return "".join(out)


@reg("text.parse_int", ("s",), ret=T.OptionType(T.I64), argtypes=(T.TEXT,),
     doc="Parse an integer, yielding none on failure.")
def _text_parse_int(ctx, s):
    try:
        return GOption(True, int(s.strip(), 0))
    except ValueError:
        return GOption(False)


@reg("text.parse_float", ("s",), ret=T.OptionType(T.F64), argtypes=(T.TEXT,),
     doc="Parse a float, yielding none on failure.")
def _text_parse_float(ctx, s):
    try:
        return GOption(True, float(s.strip()))
    except ValueError:
        return GOption(False)


# ======================================================================
# collections
# ======================================================================
@reg("collections.push", ("items", "value"), infer=lambda a: a[0] if a else T.ANY,
     doc="Append to a List, returning a new List.")
def _coll_push(ctx, items, value):
    return list(items) + [value]


@reg("collections.pop", ("items",), infer=lambda a: T.OptionType(
    a[0].elem if a and isinstance(a[0], T.ListType) else T.ANY),
     doc="Remove the last element, yielding none when empty.")
def _coll_pop(ctx, items):
    if not items:
        return GOption(False)
    return GOption(True, items[-1])


@reg("collections.get", ("items", "index"), infer=lambda a: T.OptionType(
    a[0].elem if a and isinstance(a[0], T.ListType) else T.ANY),
     doc="Safe indexed access, yielding none when out of range.")
def _coll_get(ctx, items, index):
    if isinstance(items, dict):
        return GOption(index in items, items.get(index))
    if -len(items) <= index < len(items):
        return GOption(True, items[index])
    return GOption(False)


@reg("collections.set_at", ("items", "index", "value"),
     infer=lambda a: a[0] if a else T.ANY, doc="Replace an element.")
def _coll_set_at(ctx, items, index, value):
    out = list(items)
    if not -len(out) <= index < len(out):
        raise BuiltinFault(f"index {index} out of range for List of length {len(out)}")
    out[index] = value
    return out


@reg("collections.insert", ("items", "index", "value"),
     infer=lambda a: a[0] if a else T.ANY, doc="Insert at an index.")
def _coll_insert(ctx, items, index, value):
    out = list(items)
    out.insert(index, value)
    return out


@reg("collections.remove", ("items", "index"), infer=lambda a: a[0] if a else T.ANY,
     doc="Remove the element at an index.")
def _coll_remove(ctx, items, index):
    out = list(items)
    del out[index]
    return out


def _call_fn(ctx, fnvalue, *args):
    return ctx.call_value(fnvalue, args)


@reg("collections.map", ("items", "f"), infer=lambda a: T.ListType(T.ANY),
     doc="Apply a function to every element.")
def _coll_map(ctx, items, f):
    return [_call_fn(ctx, f, x) for x in items]


@reg("collections.filter", ("items", "f"), infer=lambda a: a[0] if a else T.ANY,
     doc="Keep elements for which a predicate holds.")
def _coll_filter(ctx, items, f):
    return [x for x in items if truthy(_call_fn(ctx, f, x))]


@reg("collections.reduce", ("items", "init", "f"), infer=lambda a: T.ANY,
     doc="Fold a List with a binary function.")
def _coll_reduce(ctx, items, init, f):
    acc = init
    for x in items:
        acc = _call_fn(ctx, f, acc, x)
    return acc


@reg("collections.count", ("items", "value"), ret=T.I64, doc="Count equal elements.")
def _coll_count(ctx, items, value):
    return sum(1 for x in items if _values_equal(x, value))


@reg("collections.is_empty", ("items",), ret=T.BOOL, doc="Emptiness test.")
def _coll_is_empty(ctx, items):
    return len(items) == 0


@reg("collections.keys", ("m",), ret=T.ListType(T.ANY), doc="Keys of a Map.")
def _coll_keys(ctx, m):
    return list(m.keys())


@reg("collections.values", ("m",), ret=T.ListType(T.ANY), doc="Values of a Map.")
def _coll_values(ctx, m):
    return list(m.values())


@reg("collections.items", ("m",), ret=T.ListType(T.TupleType((T.ANY, T.ANY))),
     doc="(key, value) tuples of a Map.")
def _coll_items(ctx, m):
    return [(k, v) for k, v in m.items()]


@reg("collections.map_get", ("m", "key"), infer=lambda a: T.OptionType(T.ANY),
     doc="Map lookup yielding none when absent.")
def _coll_map_get(ctx, m, key):
    return GOption(key in m, m.get(key))


@reg("collections.map_set", ("m", "key", "value"), infer=lambda a: T.ANY,
     doc="Return a new Map with a key set.")
def _coll_map_set(ctx, m, key, value):
    out = dict(m)
    out[key] = value
    return out


@reg("collections.map_has", ("m", "key"), ret=T.BOOL, doc="Map membership test.")
def _coll_map_has(ctx, m, key):
    return key in m


@reg("collections.set_add", ("s", "value"), infer=lambda a: a[0] if a else T.ANY,
     doc="Return a new Set with a value added.")
def _coll_set_add(ctx, s, value):
    return set(s) | {value}


@reg("collections.set_has", ("s", "value"), ret=T.BOOL, doc="Set membership.")
def _coll_set_has(ctx, s, value):
    return value in s


@reg("collections.union", ("a", "b"), infer=lambda a: a[0] if a else T.ANY,
     doc="Set union.")
def _coll_union(ctx, a, b):
    return set(a) | set(b)


@reg("collections.intersect", ("a", "b"), infer=lambda a: a[0] if a else T.ANY,
     doc="Set intersection.")
def _coll_intersect(ctx, a, b):
    return set(a) & set(b)


@reg("collections.difference", ("a", "b"), infer=lambda a: a[0] if a else T.ANY,
     doc="Set difference.")
def _coll_difference(ctx, a, b):
    return set(a) - set(b)


# ======================================================================
# time
# ======================================================================
@reg("time.now", (), ret=T.INSTANT, effects=("io",),
     doc="Current wall-clock instant. Non-deterministic.")
def _time_now(ctx, ):
    return GInstant(ctx.now())


@reg("time.monotonic", (), ret=T.F64, effects=("io",), doc="Monotonic seconds.")
def _time_monotonic(ctx):
    return _time.monotonic()


@reg("time.sleep", ("seconds",), ret=T.UNIT, effects=("io",),
     doc="Suspend for a duration. Skipped in deterministic mode.")
def _time_sleep(ctx, seconds):
    secs = seconds.seconds if isinstance(seconds, GDuration) else float(seconds)
    if not ctx.deterministic:
        _time.sleep(secs)
    return UNIT


@reg("time.seconds", ("d",), ret=T.F64, argtypes=(T.DURATION,),
     doc="A Duration expressed in seconds.")
def _time_seconds(ctx, d):
    return float(d.seconds)


@reg("time.duration", ("seconds",), ret=T.DURATION, argtypes=(T.F64,),
     doc="Construct a Duration from seconds.")
def _time_duration(ctx, seconds):
    return GDuration(float(seconds))


@reg("time.since", ("instant",), ret=T.DURATION, effects=("io",),
     doc="Elapsed Duration since an Instant.")
def _time_since(ctx, instant):
    return GDuration(ctx.now() - instant.epoch)


@reg("time.format", ("instant",), ret=T.TEXT, effects=("io",),
     doc="ISO-8601 rendering of an Instant.")
def _time_format(ctx, instant):
    return _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime(instant.epoch))


# ======================================================================
# io  (capability gated -- spec section 12: no ambient filesystem access)
# ======================================================================
@reg("io.read_file", ("path",), ret=T.TEXT, effects=("io",),
     caps=("FileRead",), argtypes=(T.TEXT,),
     doc="Read a whole file. Requires the FileRead capability.")
def _io_read_file(ctx, path):
    ctx.require_capability("FileRead", what=f"io.read_file({path!r})")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise BuiltinFault(f"cannot read {path!r}: {exc.strerror or exc}") from None


@reg("io.write_file", ("path", "content"), ret=T.UNIT, effects=("io",),
     caps=("FileWrite",), argtypes=(T.TEXT, T.TEXT),
     doc="Write a file. Requires the FileWrite capability.")
def _io_write_file(ctx, path, content):
    ctx.require_capability("FileWrite", what=f"io.write_file({path!r})")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(to_text(content))
    except OSError as exc:
        raise BuiltinFault(f"cannot write {path!r}: {exc.strerror or exc}") from None
    return UNIT


@reg("io.append_file", ("path", "content"), ret=T.UNIT, effects=("io",),
     caps=("FileWrite",), argtypes=(T.TEXT, T.TEXT), doc="Append to a file.")
def _io_append_file(ctx, path, content):
    ctx.require_capability("FileWrite", what=f"io.append_file({path!r})")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(to_text(content))
    return UNIT


@reg("io.exists", ("path",), ret=T.BOOL, effects=("io",), caps=("FileRead",),
     argtypes=(T.TEXT,), doc="Test whether a path exists.")
def _io_exists(ctx, path):
    ctx.require_capability("FileRead", what=f"io.exists({path!r})")
    return os.path.exists(path)


@reg("io.read_line", (), ret=T.TEXT, effects=("io",),
     doc="Read one line from standard input.")
def _io_read_line(ctx):
    line = sys.stdin.readline()
    if not line:
        raise BuiltinFault("standard input is closed")
    return line.rstrip("\n")


# ======================================================================
# crypto
# ======================================================================
@reg("crypto.sha256", ("data",), ret=T.TEXT, effects=("crypto",),
     doc="SHA-256 hex digest of Text or Bytes.")
def _crypto_sha256(ctx, data):
    return hashlib.sha256(_as_bytes(data)).hexdigest()


@reg("crypto.sha512", ("data",), ret=T.TEXT, effects=("crypto",),
     doc="SHA-512 hex digest.")
def _crypto_sha512(ctx, data):
    return hashlib.sha512(_as_bytes(data)).hexdigest()


@reg("crypto.blake2b", ("data",), ret=T.TEXT, effects=("crypto",),
     doc="BLAKE2b hex digest.")
def _crypto_blake2b(ctx, data):
    return hashlib.blake2b(_as_bytes(data)).hexdigest()


def _as_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if hasattr(data, "data") and isinstance(getattr(data, "data"), bytes):
        return data.data
    return to_text(data).encode("utf-8")


@reg("crypto.hmac_sha256", ("key", "data"), ret=T.TEXT, effects=("crypto",),
     doc="HMAC-SHA256 hex digest.")
def _crypto_hmac(ctx, key, data):
    return _hmac.new(_as_bytes(key), _as_bytes(data), hashlib.sha256).hexdigest()


@reg("crypto.constant_time_eq", ("a", "b"), ret=T.BOOL, effects=("crypto",),
     doc="Timing-safe equality, for comparing digests and tokens.")
def _crypto_ct_eq(ctx, a, b):
    return _hmac.compare_digest(_as_bytes(a), _as_bytes(b))


@reg("crypto.sign", ("key", "message"), ret=T.TEXT, effects=("crypto",),
     caps=("CryptoSign",), doc="Sign a message. Requires CryptoSign.")
def _crypto_sign(ctx, key, message):
    ctx.require_capability("CryptoSign", what="crypto.sign")
    return _hmac.new(_as_bytes(key), _as_bytes(message), hashlib.sha256).hexdigest()


@reg("crypto.verify", ("key", "message", "signature"), ret=T.BOOL,
     effects=("crypto",), doc="Verify a crypto.sign signature.")
def _crypto_verify(ctx, key, message, signature):
    expected = _hmac.new(_as_bytes(key), _as_bytes(message),
                         hashlib.sha256).hexdigest()
    return _hmac.compare_digest(expected, to_text(signature))


@reg("crypto.hex", ("data",), ret=T.TEXT, effects=("crypto",),
     doc="Hexadecimal encoding of Bytes.")
def _crypto_hex(ctx, data):
    return _as_bytes(data).hex()


@reg("crypto.uuid", (), ret=T.UUID, effects=("crypto",),
     doc="A UUID. Deterministic (version-5 style) under a deterministic profile.")
def _crypto_uuid(ctx):
    if ctx.deterministic:
        digest = hashlib.sha256(str(ctx.now()).encode()).hexdigest()
        return GUuid(f"{digest[0:8]}-{digest[8:12]}-4{digest[12:15]}-"
                     f"8{digest[15:18]}-{digest[18:30]}")
    import uuid as _uuid
    return GUuid(str(_uuid.uuid4()))


@reg("crypto.random", (), ret=T.F64, effects=("crypto",),
     doc="Seeded pseudo-random float in [0,1). Reproducible under a seed.")
def _crypto_random(ctx):
    return ctx.random()


@reg("crypto.random_int", ("lo", "hi"), ret=T.I64, effects=("crypto",),
     doc="Seeded pseudo-random integer in [lo, hi].")
def _crypto_random_int(ctx, lo, hi):
    if hi < lo:
        raise BuiltinFault("crypto.random_int: hi must be >= lo")
    return ctx.randint(int(lo), int(hi))


# ======================================================================
# secrets  (spec section 8)
# ======================================================================
def _infer_secret(argtypes):
    """`secrets.wrap` yields a statically secret value.

    Typing the result as ordinary Any would mean the only thing standing
    between a secret and `print` is a runtime check, so the secret-ness is
    carried in the type and enforced statically (spec section 8).
    """
    inner = argtypes[0] if argtypes else T.ANY
    if isinstance(inner, T.SecretType):
        return inner
    return T.SecretType(inner)


def _infer_exposed(argtypes):
    inner = argtypes[0] if argtypes else T.ANY
    return inner.inner if isinstance(inner, T.SecretType) else inner


@reg("secrets.wrap", ("value", "label"), ret=T.SecretType(T.ANY),
     infer=_infer_secret, variadic=True, min_args=1,
     effects=("crypto",), doc="Place a value under secret lifecycle control.")
def _secrets_wrap(ctx, value, label="secret"):
    return GSecret(value, to_text(label))


@reg("secrets.expose", ("secret", "reason"), ret=T.ANY, infer=_infer_exposed,
     effects=("crypto", "audit"),
     caps=("SecretExpose",), doc="Audited reveal of a secret. Requires SecretExpose.")
def _secrets_expose(ctx, secret, reason=""):
    ctx.require_capability("SecretExpose", what="secrets.expose")
    if not isinstance(secret, GSecret):
        raise TypeFault("secrets.expose expects a secret value")
    ctx.audit.record("SECRET_EXPOSED", level="security",
                     label=secret.label, reason=to_text(reason) or None)
    return secret.expose(to_text(reason))


@reg("secrets.redact", ("value",), ret=T.TEXT, effects=("crypto",),
     doc="A non-revealing textual placeholder for a secret.")
def _secrets_redact(ctx, value):
    if isinstance(value, GSecret):
        return f"<redacted {value.label}>"
    return "<redacted>"


@reg("secrets.is_secret", ("value",), ret=T.BOOL, doc="Test for secret status.")
def _secrets_is_secret(ctx, value):
    return isinstance(value, GSecret)


@reg("secrets.fingerprint", ("secret",), ret=T.TEXT, effects=("crypto",),
     doc="A non-reversible fingerprint, safe to log.")
def _secrets_fingerprint(ctx, secret):
    inner = unwrap_secret(secret)
    return hashlib.sha256(_as_bytes(inner)).hexdigest()[:16]


# ======================================================================
# audit  (spec section 13)
# ======================================================================
@reg("audit.emit", ("action", "fields"), ret=T.TEXT, variadic=True, min_args=1,
     effects=("audit",), caps=("AuditWrite",),
     doc="Append a signed record to the audit chain.")
def _audit_emit(ctx, action, fields=None):
    ctx.require_capability("AuditWrite", what="audit.emit")
    extra = dict(fields) if isinstance(fields, dict) else {}
    rec = ctx.audit.record(to_text(action), **extra)
    return rec.event_id


@reg("audit.verify", (), ret=T.BOOL, effects=("audit",),
     doc="Re-validate the whole audit chain.")
def _audit_verify(ctx):
    ok, _ = ctx.audit.verify()
    return ok


@reg("audit.problems", (), ret=T.ListType(T.TEXT), effects=("audit",),
     doc="Concrete integrity problems found in the chain, if any.")
def _audit_problems(ctx):
    _, problems = ctx.audit.verify()
    return problems


@reg("audit.count", (), ret=T.I64, effects=("audit",), doc="Number of records.")
def _audit_count(ctx):
    return len(ctx.audit.records)


@reg("audit.head", (), ret=T.TEXT, effects=("audit",), doc="Latest record hash.")
def _audit_head(ctx):
    return ctx.audit.head_hash


@reg("audit.export", (), ret=T.TEXT, effects=("audit",),
     doc="The chain as newline-delimited JSON.")
def _audit_export(ctx):
    return ctx.audit.to_jsonl()


# ======================================================================
# identity / policy  (spec sections 12 and 17)
# ======================================================================
@reg("identity.actor", (), ret=T.TEXT, effects=("audit",),
     doc="The acting principal for audit attribution.")
def _identity_actor(ctx):
    return ctx.actor


@reg("identity.roles", (), ret=T.ListType(T.TEXT), effects=("audit",),
     doc="Roles held by the current actor.")
def _identity_roles(ctx):
    return list(getattr(ctx, "roles", []))


@reg("policy.role", ("name",), ret=T.BOOL, effects=("audit",),
     doc="True when the current actor holds a role (spec sections 17, 41).")
def _policy_role(ctx, name):
    return to_text(name) in list(getattr(ctx, "roles", []))


@reg("policy.evaluate", ("rules", "context"), ret=T.RecordType(
    "PolicyDecision", (("allow", T.BOOL), ("reason", T.TEXT),
                       ("matched", T.ListType(T.TEXT)))),
     effects=("audit",), doc="Explainable policy decision (spec section 17).")
def _policy_evaluate(ctx, rules, context=None):
    matched = []
    for rule in (rules or []):
        kind = rule.get("kind")
        expr = rule.get("expr")
        if kind == "deny" and truthy(expr):
            matched.append("deny")
            ctx.audit.record("POLICY_DENY", level="policy", reason=rule.get("text", ""))
            return GRecord("PolicyDecision", {"allow": False,
                                              "reason": rule.get("text", "denied"),
                                              "matched": matched})
        if kind == "allow" and truthy(expr):
            matched.append("allow")
    allow = "allow" in matched
    ctx.audit.record("POLICY_DECISION", level="policy", allow=allow,
                     matched=matched)
    return GRecord("PolicyDecision", {
        "allow": allow,
        "reason": "matched an allow rule" if allow else "no allow rule matched",
        "matched": matched})


@reg("policy.separation_of_duties", ("actor_role", "approver_role"), ret=T.BOOL,
     effects=("audit",), doc="True when two roles differ (spec section 41).")
def _policy_sod(ctx, a, b):
    return to_text(a) != to_text(b)


# ======================================================================
# tensor  (spec section 14)
# ======================================================================
def _tensor_type(argtypes, elem=T.F64):
    return T.TensorType(elem, None)


@reg("tensor.from_list", ("data", "dtype"), ret=T.TensorType(T.F64, None),
     variadic=True, min_args=1, doc="Build a Tensor from nested Lists.")
def _tensor_from_list(ctx, data, dtype="F64"):
    return GTensor.from_nested(data, to_text(dtype))


@reg("tensor.zeros", ("shape", "dtype"), ret=T.TensorType(T.F64, None),
     variadic=True, min_args=1, doc="A zero-filled Tensor.")
def _tensor_zeros(ctx, shape, dtype="F64"):
    return GTensor.zeros(list(shape), to_text(dtype))


@reg("tensor.ones", ("shape", "dtype"), ret=T.TensorType(T.F64, None),
     variadic=True, min_args=1, doc="A Tensor filled with ones.")
def _tensor_ones(ctx, shape, dtype="F64"):
    return GTensor.ones(list(shape), to_text(dtype))


@reg("tensor.full", ("shape", "value", "dtype"), ret=T.TensorType(T.F64, None),
     variadic=True, min_args=2, doc="A Tensor filled with a constant.")
def _tensor_full(ctx, shape, value, dtype="F64"):
    return GTensor.full(list(shape), value, to_text(dtype))


@reg("tensor.arange", ("n", "dtype"), ret=T.TensorType(T.F64, None),
     variadic=True, min_args=1, doc="0..n-1 as a rank-1 Tensor.")
def _tensor_arange(ctx, n, dtype="F64"):
    return GTensor.arange(int(n), to_text(dtype))


@reg("tensor.parameter", ("shape", "seed", "dtype"),
     ret=T.TensorType(T.F64, None), variadic=True, min_args=1,
     doc="Deterministically initialised parameter Tensor (spec section 1.3).")
def _tensor_parameter(ctx, shape, seed=0, dtype="F64"):
    return _parameter(list(shape), to_text(dtype), seed=int(seed))


@reg("tensor.shape", ("t",), ret=T.ListType(T.I64), doc="Shape as a List<I64>.")
def _tensor_shape(ctx, t):
    return list(t.shape)


@reg("tensor.rank", ("t",), ret=T.I64, doc="Number of dimensions.")
def _tensor_rank(ctx, t):
    return t.rank


@reg("tensor.size", ("t",), ret=T.I64, doc="Total element count.")
def _tensor_size(ctx, t):
    return t.size


@reg("tensor.dtype", ("t",), ret=T.TEXT, doc="Element type name.")
def _tensor_dtype(ctx, t):
    return t.dtype


@reg("tensor.reshape", ("t", "shape"), ret=T.TensorType(T.F64, None),
     doc="Reshape; -1 infers a dimension.")
def _tensor_reshape(ctx, t, shape):
    return t.reshape(list(shape))


@reg("tensor.transpose", ("t",), ret=T.TensorType(T.F64, None),
     doc="Transpose (rank 2, or reverse all axes).")
def _tensor_transpose(ctx, t):
    return t.transpose()


for _op, _method in (("add", "add"), ("sub", "sub"), ("mul", "mul"),
                     ("div", "div"), ("matmul", "matmul")):
    def _make_t(opname, method):
        @reg(f"tensor.{opname}", ("a", "b"), ret=T.TensorType(T.F64, None),
             effects=("model",), doc=f"Elementwise or linear `{opname}`.")
        def _impl(ctx, a, b, _m=method):
            return getattr(a, _m)(b)
        return _impl
    _make_t(_op, _method)

for _op, _method in (("relu", "relu"), ("sigmoid", "sigmoid"), ("tanh", "tanh"),
                     ("exp", "exp"), ("neg", "neg")):
    def _make_u(opname, method):
        @reg(f"tensor.{opname}", ("t",), ret=T.TensorType(T.F64, None),
             effects=("model",), doc=f"Elementwise `{opname}`.")
        def _impl(ctx, t, _m=method):
            return getattr(t, _m)()
        return _impl
    _make_u(_op, _method)


@reg("tensor.softmax", ("t", "axis"), ret=T.TensorType(T.F64, None),
     variadic=True, min_args=1, effects=("model",), doc="Softmax along an axis.")
def _tensor_softmax(ctx, t, axis=-1):
    return t.softmax(int(axis))


for _op in ("sum", "mean", "max", "min", "argmax"):
    def _make_r(opname):
        @reg(f"tensor.{opname}", ("t", "axis"), ret=T.TensorType(T.F64, None),
             variadic=True, min_args=1, doc=f"Reduce a Tensor with `{opname}`.")
        def _impl(ctx, t, axis=None, _o=opname):
            return getattr(t, _o)(None if axis is None else int(axis))
        return _impl
    _make_r(_op)


@reg("tensor.to_list", ("t",), ret=T.ListType(T.ANY), doc="Nested Lists.")
def _tensor_to_list(ctx, t):
    return t.to_nested()


@reg("tensor.clip", ("t", "lo", "hi"), ret=T.TensorType(T.F64, None),
     doc="Clamp every element into [lo, hi].")
def _tensor_clip(ctx, t, lo, hi):
    return t.clip(float(lo), float(hi))


@reg("tensor.allclose", ("a", "b", "tol"), ret=T.BOOL, variadic=True,
     min_args=2, doc="Elementwise equality within a tolerance.")
def _tensor_allclose(ctx, a, b, tol=1e-6):
    return a.allclose(b, float(tol))


@reg("tensor.at", ("t", "index"), ret=T.F64, doc="Element at a coordinate List.")
def _tensor_at(ctx, t, index):
    return t.at(list(index))


@reg("tensor.dot", ("a", "b"), ret=T.TensorType(T.F64, None), effects=("model",),
     doc="Dot product / matrix product.")
def _tensor_dot(ctx, a, b):
    return a.dot(b)


# ======================================================================
# autodiff  (spec section 14: automatic differentiation, training graphs)
# ======================================================================
_TAPES: Dict[int, Tape] = {}


def _tape(ctx) -> Tape:
    tape = getattr(ctx, "_autodiff_tape", None)
    if tape is None:
        tape = Tape()
        ctx._autodiff_tape = tape
    return tape


@reg("tensor.item", ("t",), ret=T.F64,
     doc="The single scalar in a rank-0 or one-element Tensor.")
def _tensor_item(ctx, t):
    tensor = t if isinstance(t, GTensor) else GTensor.from_nested(t)
    return tensor.item()


@reg("autodiff.begin", (), ret=T.UNIT, effects=("model",),
     doc="Start recording a training graph.")
def _ad_begin(ctx):
    ctx._autodiff_tape = Tape()
    _NODES.clear()
    return UNIT


def _wrap(ctx, t: GTensor, requires_grad: bool) -> GTensor:
    tape = _tape(ctx)
    node = tape.leaf(t, requires_grad)
    t._tape = tape
    _NODES[id(t)] = node
    return t


_NODES: Dict[int, TapeNode] = {}


def _node_of(t: GTensor) -> Optional[TapeNode]:
    return _NODES.get(id(t))


@reg("autodiff.parameter", ("t",), ret=T.TensorType(T.F64, None),
     effects=("model",), doc="Mark a Tensor as a trainable leaf.")
def _ad_parameter(ctx, t):
    return _wrap(ctx, t if isinstance(t, GTensor) else GTensor.from_nested(t), True)


@reg("autodiff.linear", ("x", "w", "b"), ret=T.TensorType(T.F64, None),
     effects=("model",), doc="Record x @ w + b on the training graph.")
def _ad_linear(ctx, x, w, b):
    tape = _tape(ctx)
    prod = x.matmul(w)
    out = prod.add(b)
    parents = [p for p in (_node_of(x), _node_of(w), _node_of(b)) if p]

    def backward(grad: GTensor, _x=x, _w=w, _prod=prod):
        if _node_of(_x) is not None:
            tape.accumulate(_node_of(_x), grad.matmul(_w.transpose()))
        if _node_of(_w) is not None:
            tape.accumulate(_node_of(_w), _x.transpose().matmul(grad))
        node = _node_of(b)
        if node is not None:
            # The bias is added to every row, so its gradient is the sum of
            # the incoming gradients over the batch axis.
            tape.accumulate(node, grad.sum(axis=0) if grad.rank == 2 else grad)

    node = tape.record(out, parents, backward, "linear")
    _NODES[id(out)] = node
    return out


@reg("autodiff.relu", ("t",), ret=T.TensorType(T.F64, None), effects=("model",),
     doc="Record a ReLU activation.")
def _ad_relu(ctx, t):
    tape = _tape(ctx)
    out = t.relu()
    parent = _node_of(t)
    mask = GTensor("F64", t.shape, [1.0 if v > 0 else 0.0 for v in t.data])

    def backward(grad: GTensor, _parent=parent):
        if _parent is not None:
            tape.accumulate(_parent, grad.mul(mask))

    node = tape.record(out, [parent] if parent else [], backward, "relu")
    _NODES[id(out)] = node
    return out


@reg("autodiff.mse", ("pred", "target"), ret=T.TensorType(T.F64, None),
     effects=("model",), doc="Mean squared error as a scalar Tensor.")
def _ad_mse(ctx, pred, target):
    tape = _tape(ctx)
    diff = pred.sub(target)
    sq = diff.mul(diff)
    out = sq.mean()
    parent = _node_of(pred)
    n = max(1, pred.size)

    def backward(grad: GTensor, _parent=parent, _diff=diff):
        if _parent is not None:
            g = grad.data[0] if grad.rank == 0 else grad.data[0]
            scale = 2.0 * g / n
            tape.accumulate(_parent, _diff.mul(scale))

    node = tape.record(out, [parent] if parent else [], backward, "mse")
    _NODES[id(out)] = node
    return out


@reg("autodiff.backward", ("loss",), ret=T.UNIT, effects=("model",),
     doc="Back-propagate from a scalar loss to every trainable leaf.")
def _ad_backward(ctx, loss):
    tape = _tape(ctx)
    node = _node_of(loss)
    if node is None:
        raise BuiltinFault(
            "autodiff.backward requires a loss produced by an autodiff operation",
            hint="build the loss with autodiff.linear / autodiff.relu / "
                 "autodiff.mse after autodiff.begin()")
    tape.backward(node)
    return UNIT


@reg("autodiff.grad", ("t",), ret=T.TensorType(T.F64, None), effects=("model",),
     doc="The gradient accumulated for a parameter Tensor.")
def _ad_grad(ctx, t):
    if t._grad is None:
        return GTensor.zeros(t.shape, "F64")
    return t._grad


@reg("autodiff.step", ("params", "lr"), ret=T.UNIT, effects=("model",),
     doc="One gradient-descent update; clears the accumulated gradients.")
def _ad_step(ctx, params, lr):
    for p in params:
        if p._grad is None:
            continue
        p.data = [v - float(lr) * g for v, g in zip(p.data, p._grad.data)]
    # Zero the gradients but keep the leaves registered: a training loop
    # reuses the same parameters on the next iteration.
    tape = getattr(ctx, "_autodiff_tape", None)
    if tape is not None:
        tape.reset()
    else:
        for p in params:
            p._grad = None
    return UNIT


# ======================================================================
# model  (spec section 14)
# ======================================================================
@dataclass
class GModel:
    name: str
    version: str
    weights: List[GTensor] = field(default_factory=list)
    layers: List[Tuple[int, int]] = field(default_factory=list)
    approved: bool = True

    def predict(self, x: GTensor) -> GTensor:
        out = x
        for i, w in enumerate(self.weights):
            out = out.matmul(w)
            if i < len(self.weights) - 1:
                out = out.relu()
        return out


@reg("model.load", ("name", "version", "shape"), ret=T.ANY, variadic=True,
     min_args=1, effects=("model", "storage"), caps=("ModelLoad",),
     doc="Load a model. Requires ModelLoad; deterministic under a seed.")
def _model_load(ctx, name, version="1.0.0", shape=None):
    ctx.require_capability("ModelLoad", what=f"model.load({name!r})")
    seed = int(hashlib.sha256(to_text(name).encode()).hexdigest()[:8], 16)
    model = GModel(name=to_text(name), version=to_text(version))
    if shape:
        dims = list(shape)
        for i in range(len(dims) - 1):
            model.layers.append((dims[i], dims[i + 1]))
            model.weights.append(_parameter([dims[i], dims[i + 1]], "F64",
                                            seed=seed + i))
    ctx.audit.record("MODEL_LOADED", level="medical", model=model.name,
                     version=model.version, layers=len(model.layers))
    return model


@reg("model.predict", ("m", "x"), ret=T.TensorType(T.F64, None),
     effects=("model",), doc="Run inference (spec section 14).")
def _model_predict(ctx, m, x):
    ctx.audit.record("MODEL_INFERENCE", level="model", model=m.name,
                     version=m.version)
    return m.predict(x)


@reg("model.version", ("m",), ret=T.TEXT, effects=("model",),
     doc="The loaded model's version, for `require model_version == approved`.")
def _model_version(ctx, m):
    return m.version


@reg("model.approved", ("m",), ret=T.BOOL, effects=("model",),
     doc="Whether the model is marked approved (spec section 15).")
def _model_approved(ctx, m):
    return bool(m.approved)


# ======================================================================
# dataset
# ======================================================================
@reg("dataset.from_list", ("rows",), ret=T.ANY, doc="Wrap rows as a Dataset.")
def _dataset_from_list(ctx, rows):
    return GRecord("Dataset", {"rows": list(rows), "size": len(rows)})


@reg("dataset.split", ("ds", "ratio", "seed"), ret=T.ListType(T.ANY),
     variadic=True, min_args=2, effects=("model",),
     doc="Deterministic train/test split (spec section 25).")
def _dataset_split(ctx, ds, ratio, seed=0):
    rows = list(ds.fields.get("rows", []))
    import random as _random
    rng = _random.Random(int(seed))
    shuffled = rows[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * float(ratio))
    train, test = shuffled[:cut], shuffled[cut:]
    return [GRecord("Dataset", {"rows": train, "size": len(train)}),
            GRecord("Dataset", {"rows": test, "size": len(test)})]


@reg("dataset.rows", ("ds",), ret=T.ListType(T.ANY), doc="The rows of a Dataset.")
def _dataset_rows(ctx, ds):
    return list(ds.fields.get("rows", []))


@reg("dataset.size", ("ds",), ret=T.I64, doc="Row count.")
def _dataset_size(ctx, ds):
    return int(ds.fields.get("size", 0))


@reg("dataset.batches", ("ds", "n"), ret=T.ListType(T.ANY), effects=("model",),
     doc="Split into batches of at most n rows.")
def _dataset_batches(ctx, ds, n):
    rows = list(ds.fields.get("rows", []))
    n = int(n)
    if n <= 0:
        raise BuiltinFault("batch size must be positive")
    return [rows[i:i + n] for i in range(0, len(rows), n)]


# ======================================================================
# medical  (spec section 16)
# ======================================================================
PATIENT_FIELDS = (("id", T.TEXT), ("name", T.TEXT), ("age", T.I64),
                  ("sex", T.TEXT), ("mrn", T.TEXT))
OBSERVATION_FIELDS = (("id", T.TEXT), ("patient_id", T.TEXT), ("code", T.TEXT),
                      ("value", T.F64), ("unit", T.TEXT), ("validated", T.BOOL))

const_types = {
    "Patient": T.RecordType("Patient", PATIENT_FIELDS),
    "Observation": T.RecordType("Observation", OBSERVATION_FIELDS),
    "Medication": T.RecordType("Medication", (("id", T.TEXT), ("code", T.TEXT),
                                              ("dose", T.F64), ("unit", T.TEXT))),
    "Encounter": T.RecordType("Encounter", (("id", T.TEXT), ("patient_id", T.TEXT),
                                            ("kind", T.TEXT))),
    "DiagnosticReport": T.RecordType("DiagnosticReport",
                                     (("id", T.TEXT), ("patient_id", T.TEXT),
                                      ("conclusion", T.TEXT))),
    "Risk": T.RecordType("Risk", (("score", T.F64), ("band", T.TEXT))),
    "MedicalError": T.EnumType("MedicalError", (
        T.EnumVariantInfo("DivideByZero"),
        T.EnumVariantInfo("Unvalidated"),
        T.EnumVariantInfo("NotApproved"),
        T.EnumVariantInfo("SchemaViolation", (("field", T.TEXT),)),
    )),
}


@reg("medical.patient", ("id", "name", "age", "sex", "mrn"),
     ret=T.RecordType("Patient", PATIENT_FIELDS), variadic=True, min_args=3,
     effects=("medical",), caps=("PatientWrite",),
     doc="Construct a typed Patient record. Requires PatientWrite.")
def _medical_patient(ctx, id, name, age, sex="unspecified", mrn=""):
    ctx.require_capability("PatientWrite", what="medical.patient")
    return GRecord("Patient", {"id": to_text(id), "name": to_text(name),
                               "age": int(age), "sex": to_text(sex),
                               "mrn": to_text(mrn)})


@reg("medical.observation", ("id", "patient_id", "code", "value", "unit"),
     ret=T.RecordType("Observation", OBSERVATION_FIELDS), variadic=True,
     min_args=4, effects=("medical",), caps=("PatientWrite",),
     doc="Construct a typed Observation record.")
def _medical_observation(ctx, id, patient_id, code, value, unit="",
                         validated=False):
    ctx.require_capability("PatientWrite", what="medical.observation")
    return GRecord("Observation", {
        "id": to_text(id), "patient_id": to_text(patient_id),
        "code": to_text(code), "value": float(value), "unit": to_text(unit),
        "validated": bool(validated)})


@reg("medical.validate", ("obs",), ret=T.ResultType(T.BOOL, T.TEXT),
     effects=("medical",), caps=("PatientRead",),
     doc="Schema validation of an Observation (spec section 16).")
def _medical_validate(ctx, obs):
    ctx.require_capability("PatientRead", what="medical.validate")
    problems = []
    if not obs.fields.get("patient_id"):
        problems.append("patient_id is required")
    if not obs.fields.get("code"):
        problems.append("code is required")
    value = obs.fields.get("value")
    if value is None:
        problems.append("value is required")
    if problems:
        return GResult(False, "; ".join(problems))
    return GResult(True, True)


@reg("medical.fhir_serialize", ("record",), ret=T.TEXT, effects=("medical", "io"),
     caps=("PatientRead",),
     doc="A FHIR-shaped JSON projection. Interoperability only: this does not "
         "confer compliance (spec sections 16 and 43).")
def _medical_fhir(ctx, record):
    ctx.require_capability("PatientRead", what="medical.fhir_serialize")
    resource = record.name if isinstance(record, GRecord) else "Resource"
    payload = {"resourceType": resource,
               **{k: canonical(v) for k, v in getattr(record, "fields", {}).items()}}
    ctx.audit.record("FHIR_SERIALIZED", level="medical", resource=resource)
    return json.dumps(payload, sort_keys=True)


@reg("medical.risk_band", ("score",), ret=T.TEXT, effects=("medical",),
     doc="Map a numeric risk score onto a band label.")
def _medical_risk_band(ctx, score):
    s = float(score)
    if s < 0.2:
        return "low"
    if s < 0.5:
        return "moderate"
    if s < 0.8:
        return "high"
    return "critical"


@reg("medical.human_review", ("result",), ret=T.BOOL, effects=("medical", "audit"),
     doc="Record that a human review gate was reached (spec section 15).")
def _medical_human_review(ctx, result):
    ctx.audit.record("HUMAN_REVIEW_REQUIRED", level="medical",
                     reason="spec section 15: AI output is data, not authority")
    return True


@reg("medical.minimize", ("record", "keep"), ret=T.ANY, effects=("medical",),
     caps=("PatientRead",), doc="Data minimisation: keep only listed fields.")
def _medical_minimize(ctx, record, keep):
    ctx.require_capability("PatientRead", what="medical.minimize")
    kept = {k: v for k, v in record.fields.items() if k in list(keep)}
    ctx.audit.record("DATA_MINIMIZED", level="medical",
                     dropped=len(record.fields) - len(kept))
    return GRecord(record.name, kept)


# ======================================================================
# capability construction (spec section 12)
# ======================================================================
@reg("capabilities.open", ("base", "caps", "resource"), ret=T.ANY,
     variadic=True, min_args=2, effects=("crypto",),
     doc="Mint a capability handle from a granted permission set. The runtime "
         "verifies the grant; programs cannot forge capabilities.")
def _cap_open(ctx, base, caps, resource=None):
    base = to_text(base)
    wanted = [to_text(c) for c in caps]
    for cap in wanted:
        ctx.require_capability(cap, what=f"capabilities.open({base}[{cap}])")
    rec = ctx.audit.record("CAPABILITY_ISSUED", level="security",
                           capability=base, permissions=wanted)
    return GCapability(base=base, caps=tuple(wanted), resource=resource,
                       token=rec.event_id)


@reg("capabilities.grants", ("cap", "permission"), ret=T.BOOL,
     effects=("crypto",), doc="Test whether a handle carries a permission.")
def _cap_grants(ctx, cap, permission):
    return cap.grants(to_text(permission))


# ----------------------------------------------------------------------
# lookup helpers used by the checker and the VM
# ----------------------------------------------------------------------
MODULE_TYPE_NAMES = {
    "core": T.ModuleType("core"), "math": T.ModuleType("math"),
    "text": T.ModuleType("text"), "collections": T.ModuleType("collections"),
    "time": T.ModuleType("time"), "io": T.ModuleType("io"),
    "crypto": T.ModuleType("crypto"), "secrets": T.ModuleType("secrets"),
    "audit": T.ModuleType("audit"), "identity": T.ModuleType("identity"),
    "policy": T.ModuleType("policy"), "tensor": T.ModuleType("tensor"),
    "autodiff": T.ModuleType("autodiff"), "model": T.ModuleType("model"),
    "dataset": T.ModuleType("dataset"), "medical": T.ModuleType("medical"),
    "capabilities": T.ModuleType("capabilities"),
}


def lookup(qualified: str) -> Optional[Builtin]:
    return BUILTINS.get(qualified)


# Builtins permitted to receive a `secret` value (spec section 8: secrets have
# stricter lifecycle controls, including restrictions on logging and conversion
# to ordinary Text).  Every other callable rejects them at compile time and
# again at runtime.
SECRET_SAFE = frozenset({
    "secrets.wrap", "secrets.expose", "secrets.redact", "secrets.is_secret",
    "secrets.fingerprint", "crypto.sha256", "crypto.sha512", "crypto.blake2b",
    "crypto.hmac_sha256", "crypto.sign", "crypto.constant_time_eq",
    "crypto.hex", "audit.emit", "typeof", "type_of",
})


_CAP_SUFFIXES = ("Store", "Access", "Handle", "Connection", "Client", "Service")


def expand_capability(base: str, caps: Sequence[str]) -> List[str]:
    """Map a capability-qualified type to concrete capability names.

    Spec section 12 spells authority as ``PatientStore[Read]`` but names the
    resulting capability ``PatientRead``.  This bridges the two so that a
    function parameter carrying a capability type actually confers authority.
    """
    domain = base
    for suffix in _CAP_SUFFIXES:
        if domain.endswith(suffix) and len(domain) > len(suffix):
            domain = domain[: -len(suffix)]
            break
    out: List[str] = []
    for cap in caps:
        if cap in ("Read", "Write", "Execute", "Sign", "Append"):
            expanded = f"{domain}{cap}"
        else:
            expanded = cap
        out.append(expanded)
        if expanded not in out:
            out.append(expanded)
    return out


def lookup_member(module: str, attr: str):
    """Resolve `module.attr` to a Builtin or Constant."""
    full = f"{module}.{attr}"
    if full in BUILTINS:
        return BUILTINS[full]
    if full in CONSTANTS:
        return CONSTANTS[full]
    return None


def prelude_symbols() -> Dict[str, Builtin]:
    """Unqualified names available in every module."""
    out: Dict[str, Builtin] = {}
    for name, b in BUILTINS.items():
        if "." not in name:
            out[name] = b
    return out


def module_names() -> List[str]:
    return sorted(MODULE_TYPE_NAMES)


def describe_module(module: str) -> Dict[str, Any]:
    names = sorted(n for n in MODULES.get(module, []))
    return {
        "module": module,
        "members": [
            {
                "name": n.split(".", 1)[-1] if "." in n else n,
                "kind": "constant" if n in CONSTANTS else "function",
                "signature": (f"{n.split('.', 1)[-1]}("
                              + ", ".join(BUILTINS[n].params) + ")"
                              + (f" -> {BUILTINS[n].ret.render()}"
                                 if BUILTINS[n].ret else ""))
                if n in BUILTINS else CONSTANTS[n].type.render(),
                "effects": list(BUILTINS[n].effects) if n in BUILTINS else [],
                "caps": list(BUILTINS[n].caps) if n in BUILTINS else [],
                "doc": (BUILTINS[n].doc if n in BUILTINS else CONSTANTS[n].doc),
            }
            for n in names
        ],
    }


# ======================================================================
# internal builtins used by the GIR lowering; not part of the public API
# ======================================================================
# A range is materialised rather than kept lazy: the reference interpreter has
# no streaming iterator protocol, and every consumer (`for`, `len`, indexing)
# wants random access anyway.  The cap below turns an accidental `1..10**12`
# into a clear diagnostic instead of an out-of-memory condition.
_MAX_RANGE = 10_000_000


@reg("__range", ("start", "end", "inclusive"), ret=T.ListType(T.ANY),
     hidden=True, doc="internal: materialise `a..b` / `a..=b`")
def _range(ctx, start, end, inclusive):
    if isinstance(start, bool) or isinstance(end, bool):
        raise TypeFault("range bounds must be numeric, not Bool")
    if isinstance(start, float) or isinstance(end, float):
        start = float(start)
        end = float(end)
        step = 1.0 if start <= end else -1.0
        limit = end + step if inclusive else end
        count = int(abs(limit - start)) + 1
        if count > _MAX_RANGE:
            raise GamaRuntimeFault(
                "RangeTooLarge",
                f"range would materialise {count} values, which exceeds the "
                f"limit of {_MAX_RANGE}",
                hint="iterate in chunks, or use a `while` loop with an "
                     "explicit counter")
        out = []
        v = start
        if step > 0:
            while v < limit:
                out.append(v)
                v += step
        else:
            while v > limit:
                out.append(v)
                v += step
        return out
    start = int(start)
    end = int(end)
    step = 1 if start <= end else -1
    limit = end + step if inclusive else end
    count = abs(limit - start)
    if count > _MAX_RANGE:
        raise GamaRuntimeFault(
            "RangeTooLarge",
            f"range would materialise {count} values, which exceeds the "
            f"limit of {_MAX_RANGE}",
            hint="iterate in chunks, or use a `while` loop with an explicit "
                 "counter")
    return list(range(start, limit, step))


@reg("__iter_list", ("value",), ret=T.ListType(T.ANY), hidden=True,
     variadic=True, min_args=1, doc="internal: normalise an iterable")
def _iter_list(ctx, value):
    if isinstance(value, str):
        return list(value)
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, GTensor):
        return [value.index1(i) for i in range(value.shape[0])] if value.rank else []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, GOption):
        return [value.value] if value.some else []
    raise TypeFault(f"cannot iterate over {type_name(value)}")


@reg("__stage", ("name", "args"), ret=T.UNIT, hidden=True, variadic=True,
     min_args=1, effects=("model",), doc="internal: operation-graph stage")
def _stage(ctx, name, *args):
    ctx.record_stage(to_text(name), list(args))
    return UNIT


@reg("__samples", ("domain", "count", "seed"), ret=T.ListType(T.ANY),
     hidden=True, variadic=True, min_args=2,
     doc="internal: deterministic sample domain for property tests")
def _samples(ctx, domain=None, count=100, seed=0):
    n = int(count)
    if isinstance(domain, (list, tuple)):
        if not domain:
            return []
        return [domain[i % len(domain)] for i in range(n)]
    if isinstance(domain, GTensor):
        flat = domain.flat()
        return [flat[i % len(flat)] for i in range(n)] if flat else []
    import random as _random
    rng = _random.Random(int(seed) ^ 0x5EED)
    out = []
    for i in range(n):
        # A spread of magnitudes and signs, deterministic under the seed so
        # that property tests reproduce (spec section 26).
        out.append(rng.randint(-1000, 1000))
    return out


@reg("role", ("name",), ret=T.BOOL, effects=("audit",), prelude=True,
     doc="True when the current actor holds a role (spec sections 17, 41).")
def _role_prelude(ctx, name):
    return to_text(name) in list(getattr(ctx, "roles", []))


@reg("__ctx_get", ("context", "key"), ret=T.ANY, hidden=True,
     doc="internal: read a policy decision context, None when absent")
def _ctx_get(ctx, context, key):
    if isinstance(context, dict):
        return context.get(key)
    return None


@reg("__default", ("type_name",), ret=T.ANY, hidden=True, variadic=True,
     min_args=1, doc="internal: default value for a type")
def _default(ctx, name=None):
    return None


@reg("__agent_send", ("agent", "message"), ret=T.UNIT, hidden=True,
     variadic=True, min_args=2, effects=("audit",),
     doc="internal: typed message to an isolated agent")
def _agent_send(ctx, agent, message):
    ctx.send_to_agent(to_text(agent), message)
    return UNIT


@reg("__policy_rules", ("name", "rules"), ret=T.ANY, hidden=True,
     variadic=True, min_args=1, effects=("audit",),
     doc="internal: evaluate a declared policy")
def _policy_rules(ctx, name, rules=None):
    return ctx.evaluate_policy(to_text(name), rules or [])


@reg("__transaction_begin", ("name",), ret=T.UNIT, hidden=True,
     effects=("audit", "storage"), doc="internal")
def _tx_begin(ctx, name):
    ctx.begin_transaction(to_text(name))
    return UNIT


@reg("__transaction_commit", ("name",), ret=T.UNIT, hidden=True,
     effects=("audit", "storage"), doc="internal")
def _tx_commit(ctx, name):
    ctx.commit_transaction(to_text(name))
    return UNIT


@reg("__transaction_abort", ("name", "reason"), ret=T.UNIT, hidden=True,
     effects=("audit", "storage"), doc="internal")
def _tx_abort(ctx, name, reason=""):
    ctx.abort_transaction(to_text(name), to_text(reason))
    return UNIT
