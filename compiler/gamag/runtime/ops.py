"""Shared evaluation semantics for GIR operations.

Both the reference interpreter and the compile-time optimizer evaluate
expressions, and they must agree exactly -- otherwise constant folding could
change program behaviour, which would violate the first optimisation priority
in spec section 23 ("Correctness").  This module is that single source of
truth.
"""

from __future__ import annotations

import math as _math
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..diagnostics import GamaRuntimeFault, SecretLeak, SourcePos, TypeFault
from ..semantic import types as T
from .tensor import GTensor
from .values import (GCapability, GComponent, GDuration, GInstant, GOption,
                     GRecord, GResult, GSecret, GUnit, GUri, GUuid, GVariant,
                     UNIT, deep_copy_value, display, to_text, truthy, type_name)


def default_for(ty: Any) -> Any:
    """The zero value of a type, for declarations without an initialiser."""
    if ty is None:
        return UNIT
    if isinstance(ty, T.SecretType):
        return GSecret(default_for(ty.inner), "secret")
    if isinstance(ty, T.BoolType):
        return False
    if isinstance(ty, T.IntType):
        return 0
    if isinstance(ty, T.FloatType):
        return 0.0
    if isinstance(ty, T.DecimalType):
        return 0
    if isinstance(ty, T.TextType):
        return ""
    if isinstance(ty, T.CharType):
        return "\0"
    if isinstance(ty, T.BytesType):
        return b""
    if isinstance(ty, T.ListType):
        return []
    if isinstance(ty, T.MapType):
        return {}
    if isinstance(ty, T.SetType):
        return set()
    if isinstance(ty, T.TupleType):
        return tuple(default_for(i) for i in ty.items)
    if isinstance(ty, T.OptionType):
        return GOption(False)
    if isinstance(ty, T.ResultType):
        return GResult(False, None)
    if isinstance(ty, T.TensorType):
        return GTensor.zeros(tuple(d for d in (ty.shape or (0,))
                                   if isinstance(d, int)),
                             ty.elem.render())
    if isinstance(ty, T.DurationType):
        return GDuration(0.0)
    if isinstance(ty, T.InstantType):
        return GInstant(0.0)
    if isinstance(ty, T.RecordType):
        return GRecord(ty.name, {k: default_for(v) for k, v in ty.fields})
    if isinstance(ty, T.UnitType) or isinstance(ty, T.NeverType):
        return UNIT
    return UNIT


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, GSecret) or isinstance(b, GSecret):
        raise SecretLeak("cannot compare secret values directly",
                         hint="compare `secrets.fingerprint` values instead")
    if isinstance(a, GTensor) or isinstance(b, GTensor):
        if isinstance(a, GTensor) and isinstance(b, GTensor):
            return a.equals(b)
        return False
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, GDuration) and isinstance(b, GDuration):
        return a.seconds == b.seconds
    if isinstance(a, GUnit) or isinstance(b, GUnit):
        return isinstance(a, GUnit) and isinstance(b, GUnit)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y)
                                        for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k])
                                            for k in a)
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return set(a) == set(b)
    if isinstance(a, (GOption, GResult)):
        return type(a) is type(b) and a.some == getattr(b, "some", None) \
            and values_equal(a.value, b.value)
    if isinstance(a, GVariant) or isinstance(b, GVariant):
        return (isinstance(a, GVariant) and isinstance(b, GVariant)
                and a.tag == b.tag and len(a.args) == len(b.args)
                and all(values_equal(x, y) for x, y in zip(a.args, b.args)))
    if isinstance(a, GRecord) or isinstance(b, GRecord):
        return (isinstance(a, GRecord) and isinstance(b, GRecord)
                and a.name == b.name and a.fields.keys() == b.fields.keys()
                and all(values_equal(a.fields[k], b.fields[k])
                        for k in a.fields))
    if isinstance(a, (GUuid, GUri)) or isinstance(b, (GUuid, GUri)):
        return getattr(a, "text", a) == getattr(b, "text", b)
    if isinstance(a, GCapability) or isinstance(b, GCapability):
        return a is b
    return a == b


def trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero (as in C, Rust and Java)."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def trunc_mod(a: int, b: int) -> int:
    r = abs(a) % abs(b)
    return -r if a < 0 else r




def compare(op: str, a: Any, b: Any, pos: Optional[SourcePos] = None) -> bool:
    if isinstance(a, GSecret) or isinstance(b, GSecret):
        raise SecretLeak("cannot order secret values", pos)
    if isinstance(a, GDuration) and isinstance(b, GDuration):
        a, b = a.seconds, b.seconds
    if isinstance(a, GInstant) and isinstance(b, GInstant):
        a, b = a.epoch, b.epoch
    if isinstance(a, bool) != isinstance(b, bool):
        raise TypeFault("cannot order a Bool against a non-Bool", pos)
    try:
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        return a >= b
    except TypeError:
        raise TypeFault(
            f"cannot order {type_name(a)} against {type_name(b)}", pos) from None


def tensor_binop(op: str, a: Any, b: Any,
                 pos: Optional[SourcePos] = None) -> Any:
    left = a if isinstance(a, GTensor) else GTensor.from_nested(a)
    right = b
    if op == "+":
        return left.add(right)
    if op == "-":
        return left.sub(right)
    if op == "*":
        return left.mul(right)
    if op == "/":
        return left.div(right)
    if op == "**":
        exponent = right.data[0] if isinstance(right, GTensor) else right
        return GTensor.from_nested([v ** exponent for v in left.data],
                                   left.dtype)
    raise TypeFault(f"operator `{op}` is not defined for Tensor operands", pos,
                    hint="use `tensor.matmul` for matrix products")


def duration_binop(op: str, a: Any, b: Any,
                   pos: Optional[SourcePos] = None) -> Any:
    left = a.seconds if isinstance(a, GDuration) else float(a)
    right = b.seconds if isinstance(b, GDuration) else float(b)
    if op == "+":
        return GDuration(left + right)
    if op == "-":
        return GDuration(left - right)
    if op == "*":
        return GDuration(left * right)
    if op == "/":
        if right == 0:
            raise GamaRuntimeFault("DivideByZero", "division by zero", pos)
        if isinstance(b, GDuration):
            return left / right
        return GDuration(left / right)
    raise TypeFault(f"operator `{op}` is not defined for Duration", pos)


def binop(op: str, a: Any, b: Any, pos: Optional[SourcePos] = None) -> Any:
    """Evaluate a GIR binary operation.  Total and side-effect free."""
    if op == "and":
        return truthy(a) and truthy(b)
    if op == "or":
        return truthy(a) or truthy(b)
    if op == "==":
        return values_equal(a, b)
    if op == "!=":
        return not values_equal(a, b)
    if op in ("<", ">", "<=", ">="):
        return compare(op, a, b, pos)

    if isinstance(a, GSecret) or isinstance(b, GSecret):
        raise SecretLeak("cannot perform arithmetic on a secret value", pos,
                         hint="use `secrets.expose` with a recorded reason")
    if isinstance(a, bool) or isinstance(b, bool):
        raise TypeFault(
            f"operator `{op}` is not defined for Bool operands", pos,
            hint="use `and`, `or` and `not` for boolean logic")
    if isinstance(a, GTensor) or isinstance(b, GTensor):
        return tensor_binop(op, a, b, pos)
    if isinstance(a, GDuration) or isinstance(b, GDuration):
        return duration_binop(op, a, b, pos)

    if op == "+":
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        if isinstance(a, list) and isinstance(b, list):
            return list(a) + list(b)
        if isinstance(a, str) or isinstance(b, str):
            raise TypeFault("`+` cannot combine Text with a non-Text value",
                            pos, hint="convert explicitly with `to_text(x)`")
    elif isinstance(a, str) or isinstance(b, str):
        raise TypeFault(f"operator `{op}` is not defined for Text operands",
                        pos, hint="only `+` concatenates Text")

    both_int = isinstance(a, int) and isinstance(b, int)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise GamaRuntimeFault(
                "DivideByZero", f"division by zero: {display(a)} / {display(b)}",
                pos)
        if both_int:
            return trunc_div(int(a), int(b))
        return a / b
    if op == "%":
        if b == 0:
            raise GamaRuntimeFault("DivideByZero", "modulo by zero", pos)
        if both_int:
            return trunc_mod(int(a), int(b))
        return _math.fmod(a, b)
    if op == "**":
        result = a ** b
        if both_int and isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    raise GamaRuntimeFault("BadGIR", f"unknown binary operator `{op}`", pos)


def unop(op: str, value: Any, pos: Optional[SourcePos] = None) -> Any:
    if op in ("!", "not"):
        return not truthy(value)
    if op == "-":
        if isinstance(value, GTensor):
            return value.neg()
        if isinstance(value, bool):
            raise TypeFault("unary `-` is not defined for Bool", pos)
        return -value
    if op == "+":
        return value
    raise GamaRuntimeFault("BadGIR", f"unknown unary operator `{op}`", pos)
