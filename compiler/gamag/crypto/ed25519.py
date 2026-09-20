"""Ed25519 signatures, implemented from RFC 8032.

Why this exists rather than an import: spec section 30 wants packages to carry
cryptographic identity and to be signed, and section 12 lists "signed packages
preferred" among the security requirements.  Neither is provided by a keyed hash
-- anyone holding a shared key can forge with it, so it proves integrity and not
identity.  There is no cryptography library available to this toolchain, and a
signed-package mechanism that silently degraded to a symmetric MAC would be
claiming a property it does not have.

So this is the standard algorithm, written to the RFC, and tested against the
RFC's own test vectors (`tests/test_signing.py`).  That is the strongest evidence
available here: the vectors are published, they cover key generation, signing
and verification, and they are independent of this implementation.

**In production, use an audited library.**  This exists because the alternative
was a weaker guarantee or none.  It is not constant-time: signing and
verification take time that depends on the secret scalar, so it must not be used
where an attacker can measure it.  That limitation is real and is written here
rather than discovered by a user.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

#: Field modulus: p = 2**255 - 19
P = 2 ** 255 - 19

#: Group order: L = 2**252 + 27742317777372353535851937790883648493
L = 2 ** 252 + 27742317777372353535851937790883648493

#: The curve constant d = -121665/121666 mod p
D = (-121665 * pow(121666, P - 2, P)) % P

#: sqrt(-1) mod p, used when recovering x from y
SQRT_M1 = pow(2, (P - 1) // 4, P)


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


# ---------------------------------------------------------------------------
# Point arithmetic
#
# Affine coordinates, extended with a flag for the identity, exactly as the
# RFC's own reference implementation does it.  Slower than the extended
# coordinates a production library uses, and much easier to check against the
# specification.
# ---------------------------------------------------------------------------

Point = Tuple[int, int, int, int]   # (X, Y, Z, T) in extended coordinates


def _recover_x(y: int, sign: int) -> Optional[int]:
    """The x coordinate for a given y, or None if y is not on the curve."""
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(D * y * y + 1, P - 2, P) % P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * SQRT_M1 % P
    if (x * x - x2) % P != 0:
        return None
    if (x & 1) != sign:
        x = P - x
    return x


#: The base point B, as (x, y) with x even.
_BY = 4 * pow(5, P - 2, P) % P
_BX = _recover_x(_BY, 0)
if _BX is None:                                   # pragma: no cover
    raise AssertionError("the Ed25519 base point is not on the curve")
BASE = (_BX, _BY)


def _point_add(p: Point, q: Point) -> Point:
    """Addition in extended twisted Edwards coordinates."""
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = t1 * 2 * D * t2 % P
    dd = z1 * 2 * z2 % P
    e, f, g, h = b - a, dd - c, dd + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def _point_mul(scalar: int, point: Point) -> Point:
    if scalar == 0:
        return (0, 1, 1, 0)                       # the identity
    if scalar < 0:
        x, y, z, t = _point_mul(-scalar, point)
        return ((-x) % P, y, z, (-t) % P)
    result = (0, 1, 1, 0)
    addend = point
    while scalar:
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        scalar >>= 1
    return result


def _to_bytes(point: Point) -> bytes:
    """Encode a point: y with the low bit of x in the top bit."""
    x, y, z, _t = point
    zi = pow(z, P - 2, P)
    x = x * zi % P
    y = y * zi % P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) != 32:
        return None
    value = int.from_bytes(data, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y)


def _scalar_mod_l(data: bytes) -> int:
    return int.from_bytes(data, "little") % L


def _clamp(h: bytes) -> int:
    a = bytearray(h[:32])
    a[0] &= 248
    a[31] &= 127
    a[31] |= 64
    return int.from_bytes(bytes(a), "little")


# ---------------------------------------------------------------------------
# The public interface
# ---------------------------------------------------------------------------

def public_key(secret: bytes) -> bytes:
    """The 32-byte public key for a 32-byte secret key."""
    if len(secret) != 32:
        raise ValueError("an Ed25519 secret key is 32 bytes")
    digest = _sha512(secret)
    a = _clamp(digest)
    base = (BASE[0], BASE[1], 1, BASE[0] * BASE[1] % P)
    return _to_bytes(_point_mul(a, base))


def sign(secret: bytes, message: bytes) -> bytes:
    """A 64-byte signature (R || S) over `message`."""
    if len(secret) != 32:
        raise ValueError("an Ed25519 secret key is 32 bytes")
    digest = _sha512(secret)
    a = _clamp(digest)
    public = public_key(secret)
    r = _scalar_mod_l(_sha512(digest[32:] + message))
    base = (BASE[0], BASE[1], 1, BASE[0] * BASE[1] % P)
    r_point = _point_mul(r, base)
    encoded_r = _to_bytes(r_point)
    k = _scalar_mod_l(_sha512(encoded_r + public + message))
    s = (r + k * a) % L
    return encoded_r + s.to_bytes(32, "little")


def verify(public: bytes, message: bytes, signature: bytes,
           *, strict: bool = True) -> bool:
    """Whether `signature` is a valid signature over `message` under `public`.

    `strict` also rejects a non-canonical S (one at or above the group order),
    which RFC 8032 section 8.4 requires and which matters because a malleable
    signature is one an attacker can change while keeping it valid.
    """
    if len(public) != 32 or len(signature) != 64:
        return False
    try:
        point = _from_bytes(public)
        if point is None:
            return False
        a = (point[0], point[1], 1, point[0] * point[1] % P)

        encoded_r = signature[:32]
        s = int.from_bytes(signature[32:], "little")
        if strict and s >= L:
            return False

        r = _from_bytes(encoded_r)
        if r is None:
            return False
        r_point = (r[0], r[1], 1, r[0] * r[1] % P)

        k = _scalar_mod_l(_sha512(encoded_r + public + message))
        base = (BASE[0], BASE[1], 1, BASE[0] * BASE[1] % P)

        # [S]B == R + [k]A
        left = _to_bytes(_point_mul(s, base))
        right = _to_bytes(_point_add(r_point, _point_mul(k, a)))
        return left == right
    except Exception:                              # noqa: BLE001
        # A malformed key or signature is not a valid one.  Verification
        # returns a bool rather than raising, because a caller that has to
        # handle an exception here will eventually forget to.
        return False


# ---------------------------------------------------------------------------
# Encoding helpers, so signatures can live in the lock files as text
# ---------------------------------------------------------------------------

def to_hex(data: bytes) -> str:
    return data.hex()


def from_hex(text: str) -> bytes:
    return bytes.fromhex(text.strip())


def keypair_from_seed(seed: bytes) -> Tuple[bytes, bytes]:
    """Deterministic key generation from a seed, as in RFC 8032 section 5.1.5.

    Returning both halves together keeps the invariant that a secret key and a
    public key are derived from the same seed; there is no way to construct a
    mismatched pair through this function.
    """
    if len(seed) != 32:
        raise ValueError("a seed is 32 bytes")
    return seed, public_key(seed)
