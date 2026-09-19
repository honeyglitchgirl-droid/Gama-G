"""The methods available on built-in types.

This module is the single source of truth shared by the type checker and the
virtual machine.  Keeping the two tables in one place means a method cannot be
callable at run time while being rejected at compile time (or the reverse),
which was a real defect: `semantic.checker._member_table` listed only the
property members, so `xs.push(v)` type-checked while `xs.append(v)` did not,
even though both worked at run time.

Each entry maps a method name to the builtin that implements it.  `None` means
the name is a synonym handled by the same implementation as another entry.
"""

from __future__ import annotations

from typing import Dict, Optional

# ------------------------------------------------------------------ lists
LIST_METHODS: Dict[str, Optional[str]] = {
    "push": "collections.push",
    "append": "collections.push",      # familiar spelling, same behaviour
    "pop": "collections.pop",
    "get": "collections.get",
    "set_at": "collections.set_at",
    "insert": "collections.insert",
    "remove": "collections.remove",
    "map": "collections.map",
    "filter": "collections.filter",
    "reduce": "collections.reduce",
    "count": "collections.count",
    "is_empty": "collections.is_empty",
    "sorted": "sorted",
    "reversed": "reversed",
    "contains": "contains",
    "len": "len",
    "length": "len",
    "size": "len",
    "sum": "sum",
    "min": "min",
    "max": "max",
}

# ------------------------------------------------------------------- maps
MAP_METHODS: Dict[str, Optional[str]] = {
    "get": "collections.map_get",
    "set": "collections.map_set",
    "has": "collections.map_has",
    "has_key": "collections.map_has",  # familiar spelling, same behaviour
    "keys": "collections.keys",
    "values": "collections.values",
    "items": "collections.items",
    "len": "len",
    "length": "len",
    "size": "len",
    "is_empty": "collections.is_empty",
    "contains": "contains",
}

# -------------------------------------------------------------------- sets
SET_METHODS: Dict[str, Optional[str]] = {
    "add": "collections.set_add",
    "has": "collections.set_has",
    "union": "collections.union",
    "intersect": "collections.intersect",
    "difference": "collections.difference",
    "len": "len",
    "length": "len",
    "size": "len",
    "is_empty": "collections.is_empty",
}

# ------------------------------------------------------------------ option
OPTION_METHODS: Dict[str, Optional[str]] = {
    "unwrap": "unwrap",
    "or_default": "or_default",
    "is_some": "is_some",
    "is_none": "is_none",
}

# ------------------------------------------------------------------ result
RESULT_METHODS: Dict[str, Optional[str]] = {
    "unwrap": "unwrap",
    "is_ok": "is_ok",
    "is_fail": "is_fail",
}

# -------------------------------------------------------------------- text
TEXT_METHODS: Dict[str, Optional[str]] = {
    "upper": "text.upper", "lower": "text.lower", "split": "text.split",
    "trim": "text.trim", "contains": "text.contains",
    "starts_with": "text.starts_with", "ends_with": "text.ends_with",
    "replace": "text.replace", "length": "text.length",
    "len": "text.length", "size": "text.length",
    "char_at": "text.char_at", "slice": "text.slice", "repeat": "text.repeat",
    "index_of": "text.index_of", "is_empty": "text.is_empty",
    "lines": "text.lines", "pad_start": "text.pad_start",
    "pad_end": "text.pad_end", "format": "text.format",
    "parse_int": "text.parse_int", "parse_float": "text.parse_float",
    "join": "text.join",
}

# ------------------------------------------------------------------- bytes
BYTES_METHODS: Dict[str, Optional[str]] = {
    "to_text": "bytes.to_text", "length": "bytes.length",
    "len": "bytes.length", "size": "bytes.length", "hex": "bytes.hex",
}

# ------------------------------------------------------------------ tensor
TENSOR_NATIVE = {
    "add", "sub", "mul", "div", "matmul", "dot", "relu", "sigmoid", "tanh",
    "exp", "neg", "softmax", "sum", "mean", "max", "min", "argmax",
    "reshape", "transpose", "clip", "allclose", "broadcast", "at", "copy",
    "flat",
}

# ------------------------------------------------------------------ secret
# A secret exposes only the operations that cannot leak its contents; printing
# or concatenating one is a compile error by design (spec section 8).
SECRET_METHODS: Dict[str, Optional[str]] = {
    "redact": "secrets.redact",
    "fingerprint": "secrets.fingerprint",
    "expose": "secrets.expose",
    "label": None,
}


# Methods that build and return a *new* collection rather than mutating the
# receiver.  Calling one as a bare statement is a silent no-op, so the checker
# rejects it instead of letting the program run and do nothing.
VALUE_RETURNING = {
    "push", "append", "pop", "insert", "remove", "set_at", "map", "filter",
    "reduce", "sorted", "reversed", "count", "set", "add", "union",
    "intersect", "difference",
}


def table_for(kind: str) -> Dict[str, Optional[str]]:
    """The method table for a built-in type kind."""
    return {
        "list": LIST_METHODS,
        "map": MAP_METHODS,
        "set": SET_METHODS,
        "option": OPTION_METHODS,
        "result": RESULT_METHODS,
        "text": TEXT_METHODS,
        "secret": SECRET_METHODS,
        "bytes": BYTES_METHODS,
    }[kind]
