"""Runtime value representation for the Gama-G reference runtime (GCR).

Primitives map directly onto host values (``bool``/``int``/``float``/``str``)
so that scalar code stays fast; structured values get explicit wrappers so
that the runtime can enforce the semantics the specification requires --
notably the secret lifecycle controls of section 8 and the tamper-evident
audit chain of section 13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..diagnostics import SecretLeak, TypeFault


class GamaValue:
    """Marker base class for all structured Gama-G values."""

    def reveal(self) -> Any:
        """The underlying host value, for internal use only."""
        return self


@dataclass
class GOption(GamaValue):
    """``Option<T>`` -- Gama-G's only null (spec section 6)."""

    some: bool
    value: Any = None

    def __repr__(self) -> str:
        return f"some({display(self.value)})" if self.some else "none"

    def unwrap(self) -> Any:
        if not self.some:
            raise TypeFault("called unwrap on none",
                            hint="match on the Option or use `or_default`")
        return self.value


@dataclass
class GResult(GamaValue):
    """``Result<T,E>`` -- the recoverable error channel (spec section 19)."""

    ok: bool
    value: Any = None

    def __repr__(self) -> str:
        tag = "ok" if self.ok else "fail"
        return f"{tag}({display(self.value)})"

    def unwrap(self) -> Any:
        if not self.ok:
            raise TypeFault(f"called unwrap on fail({display(self.value)})")
        return self.value


@dataclass
class GVariant(GamaValue):
    """An enum variant value, e.g. ``DivideByZero`` or ``InvalidInput(r)``."""

    tag: str
    enum: str = ""
    args: Tuple[Any, ...] = ()

    def __repr__(self) -> str:
        if not self.args:
            return self.tag
        return f"{self.tag}({', '.join(display(a) for a in self.args)})"


@dataclass
class GRecord(GamaValue):
    """A record value (spec section 5, compound types)."""

    name: str
    fields: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}: {display(v)}" for k, v in self.fields.items())
        return f"{self.name} {{ {inner} }}"


@dataclass
class GSecret(GamaValue):
    """A value under secret lifecycle control (spec section 8).

    Secrets cannot be printed, interpolated into ``Text``, serialised or
    logged through ordinary sinks.  The wrapper is the enforcement point:
    :func:`display` refuses to reveal it and :meth:`expose` is the single
    audited escape hatch.
    """

    inner: Any
    label: str = "secret"
    exposed: bool = False

    def __repr__(self) -> str:
        return f"<secret {self.label}>"

    def expose(self, reason: str = "") -> Any:
        """Audited reveal.  Callers must record why the secret was needed."""
        self.exposed = True
        return self.inner


@dataclass
class GCapability(GamaValue):
    """A runtime capability handle (spec section 12).

    A capability is unforgeable from inside the language: programs receive
    handles, they cannot construct them.  ``base`` names the resource kind
    (``PatientStore``, ``Database``, ``Audit``) and ``caps`` the permissions
    (``Read``, ``Write``).
    """

    base: str
    caps: Tuple[str, ...] = ()
    resource: Any = None
    token: str = ""

    def grants(self, cap: str) -> bool:
        return cap in self.caps or "*" in self.caps

    def __repr__(self) -> str:
        return f"{self.base}[{', '.join(self.caps)}]"


@dataclass
class GDuration(GamaValue):
    seconds: float

    def __repr__(self) -> str:
        return format_duration(self.seconds)


@dataclass
class GInstant(GamaValue):
    epoch: float

    def __repr__(self) -> str:
        return f"instant({self.epoch:.6f})"


@dataclass
class GUuid(GamaValue):
    text: str

    def __repr__(self) -> str:
        return self.text


@dataclass
class GUri(GamaValue):
    text: str

    def __repr__(self) -> str:
        return self.text


@dataclass
class GBytes(GamaValue):
    data: bytes

    def __repr__(self) -> str:
        return f"bytes({len(self.data)})"


@dataclass
class GFunction(GamaValue):
    """A first-class function value (closure)."""

    name: str
    decl: Any = None
    env: Dict[str, Any] = field(default_factory=dict)
    gir: Any = None

    def __repr__(self) -> str:
        return f"<fn {self.name}>"


@dataclass
class GComponent(GamaValue):
    """A first-class language component: model, service, agent, policy or
    transaction (spec sections 9, 10, 14, 17, 18).

    ``methods`` maps a source-level method name onto the name of the GIR
    function that implements it, so ``RiskModel.predict(x)`` and
    ``SensorGateway()`` dispatch through the same mechanism as any call.
    """

    kind: str
    name: str
    methods: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<{self.kind} {self.name}>"


@dataclass
class GAgentRef(GamaValue):
    name: str
    mailbox: List[Any] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<agent {self.name}>"


class GUnit:
    """The single ``Unit`` value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "()"

    def __bool__(self) -> bool:
        return True


UNIT = GUnit()


def format_duration(seconds: float) -> str:
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("m", 60.0), ("s", 1.0),
                       ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if seconds >= size or unit == "ns":
            value = seconds / size
            text = f"{value:g}"
            return f"{text}{unit}"
    return f"{seconds}s"


def is_secret(value: Any) -> bool:
    return isinstance(value, GSecret)


def unwrap_secret(value: Any) -> Any:
    return value.inner if isinstance(value, GSecret) else value


def display(value: Any, *, allow_secret: bool = False) -> str:
    """Render a value the way ``print`` and string conversion should.

    Refuses to reveal secrets (spec section 8: secrets have "restrictions on
    logging, serialization, and accidental conversion to ordinary Text").
    """
    if isinstance(value, GSecret):
        if not allow_secret:
            raise SecretLeak(
                f"cannot convert {value!r} to Text",
                hint="secrets cannot be printed, logged or serialised; use "
                     "`crypto.redact` or an explicit audited `expose`",
            )
        return display(value.inner, allow_secret=True)
    if value is None or isinstance(value, GUnit):
        return "()"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        if value.is_integer() and abs(value) < 1e16:
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        kind = value if isinstance(value, list) else list(value)
        return "[" + ", ".join(display(v, allow_secret=allow_secret)
                               for v in kind) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{display(k, allow_secret=allow_secret)}: "
            f"{display(v, allow_secret=allow_secret)}"
            for k, v in value.items()) + "}"
    if isinstance(value, (set, frozenset)):
        return "set{" + ", ".join(display(v, allow_secret=allow_secret)
                                  for v in sorted(value, key=repr)) + "}"
    if isinstance(value, GamaValue):
        return repr(value)
    return str(value)


def to_text(value: Any) -> str:
    """Explicit ``Text`` conversion; enforces the secret barrier."""
    return display(value, allow_secret=False)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, GOption):
        return value.some
    if isinstance(value, GResult):
        return value.ok
    if value is None or isinstance(value, GUnit):
        return False
    if isinstance(value, (int, float, str, list, dict, set)):
        return bool(value)
    raise TypeFault(
        f"cannot use {type_name(value)} as a Bool",
        hint="Gama-G conditions must be Bool; compare explicitly, e.g. "
             "`if x != 0` rather than `if x`",
    )


def type_name(value: Any) -> str:
    """Runtime type name, matching the static names of spec section 5."""
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int):
        return "I64"
    if isinstance(value, float):
        return "F64"
    if isinstance(value, str):
        return "Text" if len(value) != 1 else "Char"
    if isinstance(value, bytes):
        return "Bytes"
    if isinstance(value, GBytes):
        return "Bytes"
    if value is None or isinstance(value, GUnit):
        return "Unit"
    if isinstance(value, list):
        return "List"
    if isinstance(value, dict):
        return "Map"
    if isinstance(value, (set, frozenset)):
        return "Set"
    if isinstance(value, tuple):
        return "Tuple"
    if isinstance(value, GOption):
        return "Option"
    if isinstance(value, GResult):
        return "Result"
    if isinstance(value, GVariant):
        return value.enum or "Enum"
    if isinstance(value, GRecord):
        return value.name
    if isinstance(value, GSecret):
        return f"secret {type_name(value.inner)}"
    if isinstance(value, GCapability):
        return value.base
    if isinstance(value, GDuration):
        return "Duration"
    if isinstance(value, GInstant):
        return "Instant"
    if isinstance(value, GUuid):
        return "UUID"
    if isinstance(value, GUri):
        return "URI"
    if isinstance(value, GFunction):
        return "Fn"
    if isinstance(value, GAgentRef):
        return "Agent"
    from .tensor import GTensor
    if isinstance(value, GTensor):
        return "Tensor"
    return type(value).__name__


def deep_copy_value(value: Any) -> Any:
    """Structural copy used by checkpointing (spec section 11)."""
    if isinstance(value, list):
        return [deep_copy_value(v) for v in value]
    if isinstance(value, dict):
        return {k: deep_copy_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return set(value)
    if isinstance(value, GRecord):
        return GRecord(value.name, {k: deep_copy_value(v)
                                    for k, v in value.fields.items()})
    if isinstance(value, GOption):
        return GOption(value.some, deep_copy_value(value.value))
    if isinstance(value, GResult):
        return GResult(value.ok, deep_copy_value(value.value))
    if isinstance(value, GSecret):
        return GSecret(deep_copy_value(value.inner), value.label, value.exposed)
    from .tensor import GTensor
    if isinstance(value, GTensor):
        return value.copy()
    return value


def canonical(value: Any) -> Any:
    """A JSON-serialisable, deterministic projection of a value.

    Used for checkpoint and audit integrity hashing, where byte-stable
    serialisation is required (spec sections 11 and 13).
    """
    if isinstance(value, GSecret):
        return f"<secret {value.label}>"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, GUnit):
        return "()"
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, tuple):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(),
                                                        key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return [canonical(v) for v in sorted(value, key=repr)]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, GBytes):
        return value.data.hex()
    if isinstance(value, GOption):
        return {"some": canonical(value.value)} if value.some else {"none": True}
    if isinstance(value, GResult):
        return ({"ok": canonical(value.value)} if value.ok
                else {"fail": canonical(value.value)})
    if isinstance(value, GVariant):
        return {"variant": value.tag, "args": [canonical(a) for a in value.args]}
    if isinstance(value, GRecord):
        return {"record": value.name,
                "fields": {k: canonical(v) for k, v in sorted(value.fields.items())}}
    if isinstance(value, GDuration):
        return {"duration": value.seconds}
    if isinstance(value, GInstant):
        return {"instant": value.epoch}
    if isinstance(value, (GUuid, GUri)):
        return value.text
    if isinstance(value, GCapability):
        return {"capability": value.base, "caps": list(value.caps)}
    if isinstance(value, GFunction):
        return {"fn": value.name}
    from .tensor import GTensor
    if isinstance(value, GTensor):
        return {"tensor": {"dtype": value.dtype, "shape": list(value.shape),
                           "data": value.flat()}}
    return repr(value)
