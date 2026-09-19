"""The Gama-G type system (spec sections 5 and 6).

Strong static typing with local inference, no implicit unsafe conversion,
explicit nullability through ``Option<T>``, algebraic data types, generics,
tensor shape checking and capability-qualified types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

ShapeDim = Union[int, str]


class Type:
    """Base class for all Gama-G types."""

    def render(self) -> str:
        return type(self).__name__

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"<{self.render()}>"

    # -- relations -----------------------------------------------------
    def assignable_to(self, target: "Type") -> bool:
        """May a value of ``self`` be used where ``target`` is expected?

        Spec section 6 forbids implicit *unsafe* conversion, so widening
        within a numeric family is permitted but nothing crosses families:
        an ``I64`` never silently becomes an ``F64``.
        """
        if isinstance(target, AnyType) or isinstance(self, AnyType):
            return True
        if isinstance(self, NeverType) or isinstance(target, NeverType):
            return True
        if isinstance(target, ErrorType) or isinstance(self, ErrorType):
            return True
        if self == target:
            return True
        if isinstance(self, IntType) and isinstance(target, IntType):
            return _int_widens(self, target)
        if isinstance(self, FloatType) and isinstance(target, FloatType):
            return self.bits <= target.bits
        if isinstance(self, IntType) and isinstance(target, DecimalType):
            return True
        if isinstance(self, ListType) and isinstance(target, ListType):
            return self.elem.assignable_to(target.elem)
        if isinstance(self, SetType) and isinstance(target, SetType):
            return self.elem.assignable_to(target.elem)
        if isinstance(self, MapType) and isinstance(target, MapType):
            return (self.key.assignable_to(target.key)
                    and self.value.assignable_to(target.value))
        if isinstance(self, TupleType) and isinstance(target, TupleType):
            return (len(self.items) == len(target.items)
                    and all(a.assignable_to(b)
                            for a, b in zip(self.items, target.items)))
        if isinstance(self, OptionType) and isinstance(target, OptionType):
            if isinstance(self.inner, NoneType_):
                return True
            return self.inner.assignable_to(target.inner)
        if isinstance(self, NoneType_):
            return isinstance(target, OptionType)
        if isinstance(self, ResultType) and isinstance(target, ResultType):
            return (self.ok.assignable_to(target.ok)
                    and self.err.assignable_to(target.err))
        if isinstance(self, TensorType) and isinstance(target, TensorType):
            return (self.elem.assignable_to(target.elem)
                    and _shape_compatible(self.shape, target.shape))
        if isinstance(self, CapabilityType) and isinstance(target, CapabilityType):
            return (self.base == target.base
                    and set(target.caps) <= set(self.caps))
        if isinstance(self, NamedType) and isinstance(target, NamedType):
            return self.name == target.name
        return False

    @property
    def is_numeric(self) -> bool:
        return isinstance(self, (IntType, FloatType, DecimalType))

    @property
    def is_truthy_testable(self) -> bool:
        return isinstance(self, BoolType)


def _int_widens(src: "IntType", dst: "IntType") -> bool:
    """Lossless integer widening only (spec 6: no implicit unsafe conversion)."""
    if src.signed == dst.signed:
        return src.bits <= dst.bits
    if not src.signed and dst.signed:
        return src.bits < dst.bits     # U32 -> I64 is lossless
    return False                        # signed -> unsigned is never implicit


def _shape_compatible(a: Optional[Tuple[ShapeDim, ...]],
                      b: Optional[Tuple[ShapeDim, ...]]) -> bool:
    if a is None or b is None:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, int) and isinstance(y, int) and x != y:
            return False
    return True


# ----------------------------------------------------------------------
# primitive types (spec section 5)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class UnitType(Type):
    def render(self) -> str:
        return "Unit"


@dataclass(frozen=True)
class BoolType(Type):
    def render(self) -> str:
        return "Bool"


@dataclass(frozen=True)
class IntType(Type):
    bits: int = 64
    signed: bool = True

    def render(self) -> str:
        return f"{'I' if self.signed else 'U'}{self.bits}"

    @property
    def range(self) -> Tuple[int, int]:
        if self.signed:
            return (-(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1)
        return (0, (1 << self.bits) - 1)


@dataclass(frozen=True)
class FloatType(Type):
    bits: int = 64

    def render(self) -> str:
        return f"F{self.bits}"


@dataclass(frozen=True)
class DecimalType(Type):
    def render(self) -> str:
        return "Decimal"


@dataclass(frozen=True)
class CharType(Type):
    def render(self) -> str:
        return "Char"


@dataclass(frozen=True)
class TextType(Type):
    def render(self) -> str:
        return "Text"


@dataclass(frozen=True)
class BytesType(Type):
    def render(self) -> str:
        return "Bytes"


@dataclass(frozen=True)
class DurationType(Type):
    def render(self) -> str:
        return "Duration"


@dataclass(frozen=True)
class InstantType(Type):
    def render(self) -> str:
        return "Instant"


@dataclass(frozen=True)
class UUIDType(Type):
    def render(self) -> str:
        return "UUID"


@dataclass(frozen=True)
class URIType(Type):
    def render(self) -> str:
        return "URI"


# ----------------------------------------------------------------------
# compound types (spec section 5)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ListType(Type):
    elem: Type

    def render(self) -> str:
        return f"List<{self.elem.render()}>"


@dataclass(frozen=True)
class MapType(Type):
    key: Type
    value: Type

    def render(self) -> str:
        return f"Map<{self.key.render()}, {self.value.render()}>"


@dataclass(frozen=True)
class SetType(Type):
    elem: Type

    def render(self) -> str:
        return f"Set<{self.elem.render()}>"


@dataclass(frozen=True)
class TupleType(Type):
    items: Tuple[Type, ...]

    def render(self) -> str:
        return "Tuple(" + ", ".join(t.render() for t in self.items) + ")"


@dataclass(frozen=True)
class NoneType_(Type):
    """The type of a bare ``none`` before it is unified with ``Option<T>``."""

    def render(self) -> str:
        return "None"


@dataclass(frozen=True)
class OptionType(Type):
    inner: Type

    def render(self) -> str:
        return f"Option<{self.inner.render()}>"


@dataclass(frozen=True)
class ResultType(Type):
    ok: Type
    err: Type

    def render(self) -> str:
        return f"Result<{self.ok.render()}, {self.err.render()}>"


@dataclass(frozen=True)
class TensorType(Type):
    """``Tensor<F32,[1,224,224,3]>`` (spec section 14)."""

    elem: Type
    shape: Optional[Tuple[ShapeDim, ...]] = None

    def render(self) -> str:
        if self.shape is None:
            return f"Tensor<{self.elem.render()}>"
        dims = ", ".join(str(d) for d in self.shape)
        return f"Tensor<{self.elem.render()}, [{dims}]>"

    @property
    def rank(self) -> Optional[int]:
        return None if self.shape is None else len(self.shape)

    @property
    def static(self) -> bool:
        return self.shape is not None and all(isinstance(d, int)
                                              for d in self.shape)


@dataclass(frozen=True)
class MatrixType(Type):
    elem: Type
    rows: Optional[int] = None
    cols: Optional[int] = None

    def render(self) -> str:
        if self.rows is None:
            return f"Matrix<{self.elem.render()}>"
        return f"Matrix<{self.elem.render()}, {self.rows}, {self.cols}>"


# ----------------------------------------------------------------------
# user-defined / structural types
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RecordType(Type):
    name: str
    fields: Tuple[Tuple[str, Type], ...] = ()
    secret: bool = False

    def render(self) -> str:
        return self.name

    def field_map(self) -> Dict[str, Type]:
        return dict(self.fields)


@dataclass(frozen=True)
class EnumVariantInfo:
    name: str
    params: Tuple[Tuple[str, Type], ...] = ()


@dataclass(frozen=True)
class EnumType(Type):
    name: str
    variants: Tuple[EnumVariantInfo, ...] = ()

    def render(self) -> str:
        return self.name


@dataclass(frozen=True)
class FnType(Type):
    params: Tuple[Type, ...]
    ret: Type
    param_names: Tuple[str, ...] = ()
    effects: Tuple[str, ...] = ()
    generics: Tuple[str, ...] = ()
    name: str = "<fn>"

    def render(self) -> str:
        args = ", ".join(p.render() for p in self.params)
        return f"fn({args}) -> {self.ret.render()}"


@dataclass(frozen=True)
class CapabilityType(Type):
    """``PatientStore[Read]`` -- a type qualified by capabilities (spec 12)."""

    base: str
    caps: Tuple[str, ...] = ()

    def render(self) -> str:
        if not self.caps:
            return self.base
        return f"{self.base}[{', '.join(self.caps)}]"


@dataclass(frozen=True)
class SecretType(Type):
    """A value under secret lifecycle control (spec section 8)."""

    inner: Type

    def render(self) -> str:
        return f"secret {self.inner.render()}"


@dataclass(frozen=True)
class NamedType(Type):
    """A nominal type referenced by name, resolved to a declaration if one
    exists in the module; otherwise opaque (matches only itself)."""

    name: str
    args: Tuple[Type, ...] = ()
    resolved: Optional[Type] = None

    def render(self) -> str:
        if self.args:
            return f"{self.name}<{', '.join(a.render() for a in self.args)}>"
        return self.name


@dataclass(frozen=True)
class ModuleType(Type):
    """A standard-library namespace such as `math` or `medical` (spec 28)."""

    name: str

    def render(self) -> str:
        return f"module {self.name}"


@dataclass(frozen=True)
class NeverType(Type):
    """The type of an expression that never produces a value (`panic`)."""

    def render(self) -> str:
        return "Never"


@dataclass(frozen=True)
class AnyType(Type):
    """Deliberately unchecked (FFI, unresolved externals, generics)."""

    reason: str = "any"

    def render(self) -> str:
        return "Any"


@dataclass(frozen=True)
class ErrorType(Type):
    """Poison type: suppresses cascading diagnostics after a first error."""

    def render(self) -> str:
        return "<error>"


# ----------------------------------------------------------------------
# singletons and construction helpers
# ----------------------------------------------------------------------
UNIT = UnitType()
BOOL = BoolType()
TEXT = TextType()
CHAR = CharType()
BYTES = BytesType()
DECIMAL = DecimalType()
DURATION = DurationType()
INSTANT = InstantType()
UUID = UUIDType()
URI = URIType()
ANY = AnyType()
ERROR = ErrorType()
NONE = NoneType_()

I8, I16, I32, I64, I128 = (IntType(b, True) for b in (8, 16, 32, 64, 128))
U8, U16, U32, U64, U128 = (IntType(b, False) for b in (8, 16, 32, 64, 128))
F16, F32, F64 = (FloatType(b) for b in (16, 32, 64))

PRIMITIVES: Dict[str, Type] = {
    "Bool": BOOL, "Unit": UNIT, "Text": TEXT, "Char": CHAR, "Bytes": BYTES,
    "Decimal": DECIMAL, "Duration": DURATION, "Instant": INSTANT,
    "UUID": UUID, "URI": URI,
    "I8": I8, "I16": I16, "I32": I32, "I64": I64, "I128": I128,
    "U8": U8, "U16": U16, "U32": U32, "U64": U64, "U128": U128,
    "F16": F16, "F32": F32, "F64": F64,
    # Common aliases used by the specification's examples.
    "Int": I64, "Float": F64, "String": TEXT, "None": UNIT,
}

GENERIC_ARITY = {
    "List": 1, "Set": 1, "Option": 1, "Map": 2, "Result": 2, "Tensor": 2,
    "Matrix": 3, "Tuple": None,
}


def build_generic(name: str, args: Sequence[Type],
                  shape: Optional[Tuple[ShapeDim, ...]] = None) -> Type:
    if name == "List":
        return ListType(args[0])
    if name == "Set":
        return SetType(args[0])
    if name == "Option":
        return OptionType(args[0])
    if name == "Map":
        return MapType(args[0], args[1])
    if name == "Result":
        return ResultType(args[0], args[1])
    if name == "Tensor":
        return TensorType(args[0], shape)
    if name == "Matrix":
        rows = args[1] if len(args) > 1 else None
        cols = args[2] if len(args) > 2 else None
        return MatrixType(args[0],
                          rows if isinstance(rows, int) else None,
                          cols if isinstance(cols, int) else None)
    if name == "Tuple":
        return TupleType(tuple(args))
    raise KeyError(name)


def unify(a: Type, b: Type) -> Type:
    """Join two inferred types, used for local type inference (spec 6).

    Returns ``ERROR`` when the types are genuinely incompatible so that the
    caller can report one precise diagnostic instead of cascading.
    """
    if isinstance(a, ErrorType):
        return b
    if isinstance(b, ErrorType):
        return a
    if a == b:
        return a
    if isinstance(a, AnyType):
        return b
    if isinstance(a, NeverType):
        return b
    if isinstance(b, NeverType):
        return a
    if isinstance(b, AnyType):
        return a
    if isinstance(a, NoneType_):
        return OptionType(b) if not isinstance(b, OptionType) else b
    if isinstance(b, NoneType_):
        return OptionType(a) if not isinstance(a, OptionType) else a
    if isinstance(a, OptionType) and not isinstance(b, OptionType):
        return OptionType(unify(a.inner, b)) if a.assignable_to(OptionType(b)) \
            else ERROR
    if isinstance(b, OptionType) and not isinstance(a, OptionType):
        return unify(b, a)
    if isinstance(a, IntType) and isinstance(b, IntType):
        return b if _int_widens(a, b) else (a if _int_widens(b, a) else ERROR)
    if isinstance(a, FloatType) and isinstance(b, FloatType):
        return a if a.bits >= b.bits else b
    if isinstance(a, NamedType) and isinstance(b, NamedType) and a.name == b.name:
        return a
    if a.assignable_to(b):
        return b
    if b.assignable_to(a):
        return a
    return ERROR


def default_int() -> Type:
    """Default type of an integer literal."""
    return I64


def default_float() -> Type:
    return F64
