"""Foreign function interface (audit priority 12, spec section 29).

Spec section 29 lists the interfaces Gama-G needs -- the C ABI first -- and then
imposes the rule that shapes this module:

    FFI must be explicitly marked unsafe/trusted where memory ownership cannot
    be verified by Gama-G.

The language already has the vocabulary for that: `unsafe` is a declared effect
and `ForeignCall` is a capability, so a foreign call has to be marked twice, in
the type system and in the authority the program was granted.  Neither is
optional, and neither is inferred.

Three limits are imposed deliberately, and each closes a hole rather than being
a gap:

* **A closed type vocabulary.** A foreign function's parameters and result must
  be built from i32, i64, f64 and C strings.  A `void *` is not expressible,
  because a pointer Gama-G cannot verify is exactly what the specification's
  rule is about -- allowing one would make the marker decorative.
* **No struct passing and no callbacks.** Both need a memory model Gama-G does
  not have for foreign code.  They are refused by name.
* **Every call is audited.** A foreign call is the point where the language
  stops being able to make promises, so it is the point where the record
  matters most: the record is written before the call, so a call that corrupts
  memory still leaves evidence that it happened.

`ctypes` is used rather than writing a C extension, so this works from a source
checkout with nothing to build.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..diagnostics import GamaRuntimeFault, TypeFault
from ..runtime.values import (GOption, GRecord, GResult, GUnit, UNIT, display,
                              to_text, type_name)
from ..semantic import types as T
from .library import reg

# ---------------------------------------------------------------------------
# The declared type vocabulary
# ---------------------------------------------------------------------------

#: The only C types a foreign signature may name, and what they mean.
#:
#: Anything absent is absent on purpose. A pointer type would let a program hold
#: a value whose lifetime Gama-G cannot check, which is the case spec section 29
#: says must be marked unsafe -- and marking it unsafe while still allowing it
#: unchecked would make the marking meaningless.
FOREIGN_TYPES: Dict[str, Tuple[Any, Any, str]] = {
    # name: (ctypes type, Gama-G type, description)
    "i32":   (ctypes.c_int32,  T.I32,  "a 32-bit signed integer"),
    "i64":   (ctypes.c_int64,  T.I64,  "a 64-bit signed integer"),
    "u32":   (ctypes.c_uint32, T.U32,  "a 32-bit unsigned integer"),
    "u64":   (ctypes.c_uint64, T.U64,  "a 64-bit unsigned integer"),
    "f32":   (ctypes.c_float,  T.F32,  "a 32-bit float"),
    "f64":   (ctypes.c_double, T.F64,  "a 64-bit float"),
    "text":  (ctypes.c_char_p, T.TEXT, "a NUL-terminated C string"),
    "bool":  (ctypes.c_int32,  T.BOOL, "an integer used as a truth value"),
    "unit":  (None,            T.UNIT, "no value"),
}

#: Types that are recognisably C but deliberately unsupported, with the reason.
REFUSED_TYPES: Dict[str, str] = {
    "ptr": "a raw pointer's lifetime cannot be checked by Gama-G, which is "
           "exactly the case spec section 29 says must be marked unsafe; "
           "allowing it here would make the marker decorative",
    "void*": "same as `ptr`",
    "struct": "passing a C struct needs a layout Gama-G cannot verify",
    "fn": "passing a C callback would let foreign code re-enter a Gama-G "
          "function outside its effect and capability checks",
}


class ForeignError(GamaRuntimeFault):
    """A foreign library, symbol or call failed."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__("ForeignCallError", message)
        if hint:
            self.hint = hint


# ---------------------------------------------------------------------------
# Handle registry
# ---------------------------------------------------------------------------

class Library:
    """One loaded foreign library and the symbols resolved from it."""

    def __init__(self, path: str, handle: Any):
        self.path = path
        self.handle = handle
        self.symbols: Dict[str, Any] = {}
        self.calls = 0

    def __repr__(self) -> str:
        return f"<foreign library {os.path.basename(self.path)}>"


class Binding:
    """One resolved symbol, with the signature it was declared with."""

    def __init__(self, library: Library, name: str,
                 argument_types: Sequence[str], result_type: str):
        self.library = library
        self.name = name
        self.argument_types = tuple(argument_types)
        self.result_type = result_type
        self.description = ""
        ctypes_args = [FOREIGN_TYPES[a][0] for a in self.argument_types]
        ctypes_result = FOREIGN_TYPES[result_type][0]
        self.function = getattr(library.handle, name)
        self.function.argtypes = ctypes_args
        self.function.restype = ctypes_result

    def __repr__(self) -> str:
        arguments = ", ".join(self.argument_types)
        return f"<foreign fn {self.name}({arguments}) -> {self.result_type}>"


#: Process-wide state.  A foreign library is a process resource, so this is
#: process state, deliberately: loading the same library twice must give the
#: same handle rather than two, because two would mean two sets of globals.
_LIBRARIES: Dict[str, Library] = {}


def _require_capability(ctx: Any, what: str) -> None:
    """Refuse unless `ForeignCall` was granted.

    Spec section 12: no ambient authority.  A program that can call into
    arbitrary native code can do anything the process can, so this is the
    broadest authority the language has and it is never implied.
    """
    if not ctx.has_cap("ForeignCall"):
        raise GamaRuntimeFault(
            "CapabilityViolation",
            f"{what} needs the `ForeignCall` capability, which was not granted",
            hint="foreign code runs outside every guarantee Gama-G makes; "
                 "grant ForeignCall explicitly, and only to programs that "
                 "declare the `unsafe` effect")


def _search_paths() -> List[str]:
    try:
        import sysconfig
        return [path for path in
                (sysconfig.get_config_var("LIBDIR"),
                 "/usr/lib", "/usr/local/lib", "/lib") if path]
    except Exception:                                  # pragma: no cover
        return []


# ---------------------------------------------------------------------------
# Builtins
# ---------------------------------------------------------------------------

@reg("ffi.libraries", (), ret=T.TEXT, effects=("unsafe",), caps=("ForeignCall",),
     doc="Names and descriptions of the foreign types a signature may use.")
def _ffi_libraries(ctx) -> str:
    _require_capability(ctx, "`ffi.libraries`")
    if not _LIBRARIES:
        return "no foreign library is loaded"
    return ", ".join(sorted(os.path.basename(lib.path)
                            for lib in _LIBRARIES.values()))


@reg("ffi.types", (), ret=T.TEXT, effects=("unsafe",), caps=("ForeignCall",),
     doc="The C types this interface accepts, and those it refuses.")
def _ffi_types(ctx) -> str:
    allowed = ", ".join(sorted(FOREIGN_TYPES))
    refused = "; ".join(f"{name} ({reason.split(',')[0]})"
                        for name, reason in sorted(REFUSED_TYPES.items()))
    return f"supported: {allowed}. refused: {refused}"


@reg("ffi.open", ("path",), ret=T.ANY, argtypes=(T.TEXT,),
     effects=("unsafe",), caps=("ForeignCall",),
     doc="Load a shared library. The path may be a name to look up or a path.")
def _ffi_open(ctx, path: str) -> GRecord:
    _require_capability(ctx, "`ffi.open`")
    resolved = path
    if os.sep not in path:
        found = ctypes.util.find_library(path)
        if found:
            resolved = found
    resolved = os.path.abspath(resolved) if os.sep in resolved else resolved
    if resolved in _LIBRARIES:
        library = _LIBRARIES[resolved]
    else:
        try:
            handle = ctypes.CDLL(resolved)
        except OSError as exc:
            hint = ""
            if os.sep not in path:
                hint = (f"looked for `{path}` as a library name; pass a full "
                        f"path if it is not in the loader's search path")
            raise ForeignError(
                f"cannot load {path!r}: {exc}", hint) from None
        library = Library(resolved, handle)
        _LIBRARIES[resolved] = library
    ctx.audit.record("FOREIGN_OPEN", object=resolved, level="security",
                     reason="the program requested a foreign library")
    return GRecord("ForeignLibrary", {"path": resolved})


@reg("ffi.bind", ("library", "name", "signature"), ret=T.ANY,
     argtypes=(T.ANY, T.TEXT, T.TEXT),
     effects=("unsafe",), caps=("ForeignCall",),
     doc="Resolve a symbol and declare its C signature, e.g. \"f64,f64->f64\".")
def _ffi_bind(ctx, library: GRecord, name: str, signature: str) -> GRecord:
    _require_capability(ctx, "`ffi.bind`")
    handle = _lookup_library(library)
    argument_types, result_type = _parse_signature(signature, name)

    def as_record() -> GRecord:
        return GRecord("ForeignBinding", {"library": handle.path, "name": name,
                                          "signature": signature})

    if name in handle.symbols:
        # Returning the cached `Binding` here was a bug: the cache holds this
        # module's internal object, and a program that binds the same symbol
        # twice -- or a second run in the same process, since libraries are
        # process state -- received something with no fields, and every later
        # call failed on it.  The value handed back is always a record.
        #
        # The record is written even on this path.  The event is "the program
        # asked for this symbol", which happened whether or not the dynamic
        # linker had to be consulted.
        ctx.audit.record("FOREIGN_BIND", object=f"{handle.path}:{name}",
                         level="security",
                         reason=f"signature {signature} (already resolved)")
        return as_record()

    if not hasattr(handle.handle, name):
        # A symbol that is absent is a typo or a version mismatch, and saying
        # which library failed matters when several are loaded.
        raise ForeignError(
            f"`{name}` is not exported by {os.path.basename(handle.path)}",
            hint="check the spelling, and that the library is the version the "
                 "signature was written for")
    try:
        binding = Binding(handle, name, argument_types, result_type)
    except (TypeError, ValueError) as exc:
        raise ForeignError(f"cannot bind `{name}`: {exc}") from None
    handle.symbols[name] = binding
    ctx.audit.record("FOREIGN_BIND", object=f"{handle.path}:{name}",
                     level="security", reason=f"signature {signature}")
    return as_record()


@reg("ffi.call", ("binding",), ret=T.ANY, variadic=True, min_args=1,
     effects=("unsafe",), caps=("ForeignCall",),
     doc="Call a bound foreign function with checked argument types.")
def _ffi_call(ctx, binding: GRecord, *args: Any) -> Any:
    _require_capability(ctx, "`ffi.call`")
    target = _lookup_binding(binding)
    if len(args) != len(target.argument_types):
        raise TypeFault(
            f"`{target.name}` takes {len(target.argument_types)} argument(s) "
            f"({', '.join(target.argument_types)}) but received {len(args)}")
    converted = [_to_c(value, kind, target.name, position)
                 for position, (value, kind)
                 in enumerate(zip(args, target.argument_types))]

    # Recorded *before* the call.  A foreign call is where the language stops
    # being able to make promises, so a call that corrupts memory must still
    # leave evidence that it happened; recording afterwards would lose exactly
    # the cases the record is for.
    ctx.audit.record("FOREIGN_CALL",
                     object=f"{target.library.path}:{target.name}",
                     level="security",
                     reason=f"arguments of type "
                            f"{', '.join(target.argument_types)}")
    target.library.calls += 1
    try:
        raw = target.function(*converted)
    except (ctypes.ArgumentError, TypeError, ValueError) as exc:
        raise ForeignError(f"calling `{target.name}` failed: {exc}") from None
    except OSError as exc:
        raise ForeignError(
            f"`{target.name}` failed: {exc}",
            "a foreign function that sets errno is reporting through it; the "
            "message is whatever the operating system said") from None
    return _from_c(raw, target.result_type, target.name)


@reg("ffi.close", ("library",), ret=T.UNIT,
     argtypes=(T.ANY,),
     effects=("unsafe",), caps=("ForeignCall",), doc="Release a library handle.")
def _ffi_close(ctx, library: GRecord) -> GUnit:
    _require_capability(ctx, "`ffi.close`")
    path = library.fields.get("path", "")
    if path in _LIBRARIES:
        # The binding objects keep the handle alive, so this only drops the
        # registry entry; unloading a library other bindings still point at
        # would be a use-after-free that Gama-G could not detect.
        del _LIBRARIES[path]
        ctx.audit.record("FOREIGN_CLOSE", object=path, level="security",
                         reason="released by the program")
    return UNIT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_library(record: GRecord) -> Library:
    path = record.fields.get("path", "")
    if path not in _LIBRARIES:
        raise ForeignError(
            f"the library handle for {path!r} is no longer valid",
            hint="it was closed, or this handle came from another process")
    return _LIBRARIES[path]


def _lookup_binding(record: GRecord) -> Binding:
    path = record.fields.get("library", "")
    name = record.fields.get("name", "")
    library = _LIBRARIES.get(path)
    if library is None or name not in library.symbols:
        raise ForeignError(
            f"the binding for `{name}` is no longer valid")
    return library.symbols[name]


def _parse_signature(signature: str, name: str) -> Tuple[List[str], str]:
    """`"f64,f64->f64"` -> (["f64", "f64"], "f64")."""
    text = signature.replace(" ", "")
    if "->" not in text:
        raise ForeignError(
            f"the signature for `{name}` is {signature!r}, which has no `->`",
            hint="signatures look like \"i64,i64->i64\" or \"->unit\"")
    left, right = text.split("->", 1)
    arguments = [part for part in left.split(",") if part]
    if not right:
        raise ForeignError(f"the signature for `{name}` has no result type")
    for kind in arguments + [right]:
        if kind in REFUSED_TYPES:
            raise ForeignError(
                f"the signature for `{name}` uses `{kind}`: "
                f"{REFUSED_TYPES[kind]}")
        if kind not in FOREIGN_TYPES:
            raise ForeignError(
                f"`{kind}` is not a C type this interface accepts",
                hint=f"supported types are {', '.join(sorted(FOREIGN_TYPES))}")
    return arguments, right


def _to_c(value: Any, kind: str, name: str, position: int) -> Any:
    if kind == "text":
        if not isinstance(value, str):
            raise TypeFault(
                f"argument {position + 1} of `{name}` is declared `text`, so it "
                f"must be Text, not {type_name(value)}")
        return value.encode("utf-8")
    if kind in ("f32", "f64"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeFault(
                f"argument {position + 1} of `{name}` is declared `{kind}`, so "
                f"it must be a number, not {type_name(value)}")
        return float(value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise TypeFault(
                f"argument {position + 1} of `{name}` is declared `bool`, so it "
                f"must be Bool, not {type_name(value)}")
        return 1 if value else 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeFault(
            f"argument {position + 1} of `{name}` is declared `{kind}`, so it "
            f"must be an integer, not {type_name(value)}")
    ctypes_type, declared, _text = FOREIGN_TYPES[kind]
    low, high = declared.range if hasattr(declared, "range") else (None, None)
    if low is not None and not low <= value <= high:
        # Passing an out-of-range value to C would truncate it silently, which
        # is the kind of difference that makes a foreign call unverifiable.
        raise TypeFault(
            f"argument {position + 1} of `{name}` is {value}, which does not "
            f"fit in {kind} (range {low} to {high})")
    return value


def _from_c(raw: Any, kind: str, name: str) -> Any:
    if kind == "unit":
        return UNIT
    if kind == "text":
        if raw is None:
            # A NULL char* is the C convention for "absent", and the language
            # has exactly one way to say that.
            return GOption(False, None)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", "replace")
        return str(raw)
    if kind == "bool":
        return bool(raw)
    if kind in ("f32", "f64"):
        return float(raw)
    return int(raw)


def loaded_libraries() -> Dict[str, Library]:
    """Loaded libraries, for `ggc doc` and tests."""
    return dict(_LIBRARIES)
