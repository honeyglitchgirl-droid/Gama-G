"""Deciding exhaustiveness and exclusivity of `when` guards.

The core's selection is a set of operations yielding one binding, each saying
`when` it applies.  v0.2 proved one shape of this exhaustively: two guards
whose *text* was complementary (`g` and `not g`).  Beyond two alternatives --
or with guards that are complementary in *value* but not in spelling, like
`score >= 90`, `score >= 75 and score < 90`, `score < 75` -- the compiler
recorded the constraint as `unprovable` and kept a runtime fault.

That was the honest answer at the time: proving exhaustiveness over arbitrary
predicates is undecidable, and claiming otherwise is exactly the kind of claim
spec section 43 forbids.  It is no longer the whole answer.  For a decidable
fragment -- boolean combinations of comparisons of *one* integer binding
against integer constants -- the question "does the union of these regions
cover the type?" is arithmetic, not magic, and this module computes the
answer exactly, with finite unions of integer intervals.

What is proved:

* guards build from ``<  <=  >  >=  ==  !=`` over the one subject binding and
  integer literals (constants may be negative and may fold through ``+ - *``);
* combined with ``and``, ``or``, ``not``;
* a literal ``true`` guard covers everything and ``false`` nothing, so a
  catch-all written as a constant is understood;
* exclusivity: the pairwise intersections of the alternatives' regions.

What stays `unprovable`, and the runtime fault keeps its meaning there:

* anything over a *float* subject -- comparisons with NaN make "covers all
  reals" false, and pretending otherwise would be the bug;
* guards that read more than one binding, function calls, indexing, fields;
* subjects whose type is not a sized integer: with no type there are no
  bounds, and a proof needs them (``x >= -2**63`` covers I64 but not an
  unbounded mathematical integer).

The procedure is exact on its fragment, never an approximation: a proof is
recorded as `proven`, a give-up as `unprovable`, and no middle ground exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import mir as M

#: An interval is a closed integer range with optional infinite ends.
Interval = Tuple[Optional[int], Optional[int]]

#: The sized integer types and their bounds.  Floats are deliberately absent.
INT_BOUNDS = {
    "I8": (-128, 127), "I16": (-32768, 32767),
    "I32": (-2147483648, 2147483647),
    "I64": (-9223372036854775808, 9223372036854775807),
    "U8": (0, 255), "U16": (0, 65535), "U32": (0, 4294967295),
    "U64": (0, 18446744073709551615),
}

#: Union size beyond which a region gives up: a guard set that explodes to
#: this many intervals is not the common case, and a quadratic blow-up inside
#: a compiler on behalf of an exotic input is not worth the proof.
MAX_INTERVALS = 64

_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


@dataclass
class Proof:
    """The verdict, with the reasoning kept so a reader can see the shape."""
    exhaustive: bool
    exclusive: bool
    subject: str
    type_name: str
    note: str = ""


# ----------------------------------------------------------------------
# interval arithmetic over the integers
# ----------------------------------------------------------------------
def _merge(intervals: List[Interval]) -> Optional[List[Interval]]:
    """Sort and merge adjacent-or-overlapping closed integer intervals."""
    def lo_key(iv: Interval) -> int:
        return iv[0] if iv[0] is not None else -(1 << 200)

    ordered = sorted(intervals, key=lo_key)
    out: List[Interval] = []
    for lo, hi in ordered:
        if lo is not None and hi is not None and lo > hi:
            continue                       # empty interval
        if out:
            plo, phi = out[-1]
            touches = (phi is None or (lo is not None and lo <= phi + 1))
            if touches:
                if phi is None or hi is None:
                    new_hi = None          # something reaches +infinity
                else:
                    new_hi = max(phi, hi)
                out[-1] = (plo, new_hi)
                continue
        out.append((lo, hi))
    if len(out) > MAX_INTERVALS:
        return None
    return out


def _whole(a: Interval, b: Interval) -> bool:
    """True when interval `a` lies inside interval `b`."""
    if b[0] is not None and (a[0] is None or a[0] < b[0]):
        return False
    if b[1] is not None and (a[1] is None or a[1] > b[1]):
        return False
    return True


def _intersect(a: Interval, b: Interval) -> Optional[Interval]:
    lo = max(x for x in (a[0], b[0]) if x is not None) \
        if (a[0] is not None or b[0] is not None) else None
    hi = min(x for x in (a[1], b[1]) if x is not None) \
        if (a[1] is not None or b[1] is not None) else None
    if lo is not None and hi is not None and lo > hi:
        return None
    return (lo, hi)


def _intersect_lists(a: List[Interval], b: List[Interval]
                     ) -> Optional[List[Interval]]:
    out: List[Interval] = []
    for x in a:
        for y in b:
            piece = _intersect(x, y)
            if piece is not None:
                out.append(piece)
    return _merge(out)


def _union(*lists: List[Interval]) -> Optional[List[Interval]]:
    flat: List[Interval] = []
    for lst in lists:
        flat.extend(lst)
    return _merge(flat)


def _complement(a: List[Interval], bounds: Tuple[int, int]
                ) -> Optional[List[Interval]]:
    """Everything in `bounds` not covered by `a`.  The domain is the type, so
    a `!=` guard over I8 complements to exactly [-128,c-1] + [c+1,127]."""
    lo, hi = bounds
    out: List[Interval] = []
    cursor = lo
    for ilo, ihi in a:
        start = cursor if ilo is None else max(cursor, ilo)
        if start > cursor:
            out.append((cursor, start - 1))
        end = ihi if ihi is not None else hi
        cursor = end + 1
        if cursor > hi:
            cursor = hi + 1
            break
    if cursor <= hi:
        out.append((cursor, hi))
    # clip: the cursor walk already stays within bounds because `a` may
    # extend past them; re-clip each interval
    clipped: List[Interval] = []
    for ilo, ihi in out:
        piece = _intersect((ilo, ihi), bounds)
        if piece is not None:
            clipped.append(piece)
    return _merge(clipped)


def _covers_all(a: List[Interval], bounds: Tuple[int, int]) -> bool:
    """True when `a` covers every integer in `bounds`."""
    merged = _merge(a)
    if merged is None or not merged:
        return False
    lo, hi = bounds
    # one merged interval spanning the whole type covers it exactly
    return len(merged) == 1 and _whole((lo, hi), merged[0])


# ----------------------------------------------------------------------
# the same algebra, exposed: `contractproof.py` discharges promises with
# these, so there is one interval implementation in the compiler and not two
# ----------------------------------------------------------------------
def region_of(expr: Optional[M.MExpr]) -> Optional[Tuple[Optional[str],
                                                          List[Interval]]]:
    """The exact region of a predicate, or None outside the fragment."""
    return _region(expr)


def intersect(a: List[Interval], b: List[Interval]) -> Optional[List[Interval]]:
    """The intersection of two regions; None when this gives up (too many)."""
    return _intersect_lists(a, b)


def is_empty(region: List[Interval]) -> bool:
    """True when a region contains no integer at all."""
    merged = _merge(region)
    return merged is not None and not merged


def disjoint(a: List[Interval], b: List[Interval]) -> Optional[bool]:
    """True when `a` and `b` share no integer; None when this gave up."""
    inter = _intersect_lists(a, b)
    return None if inter is None else not inter


def within(a: List[Interval], b: List[Interval],
           bounds: Tuple[int, int]) -> Optional[bool]:
    """True when every integer of `a` is in `b`; None when this gave up.

    Complement is taken over `bounds`, because that is the whole domain a
    binding can hold: `x >= 0` does not contain `x > 0` over the integers, but
    over U8 both sides mean the same set, and a proof needs the type.
    """
    outside = _complement(b, bounds)
    if outside is None:
        return None
    inter = _intersect_lists(a, outside)
    return None if inter is None else not inter


def covers(a: List[Interval], bounds: Tuple[int, int]) -> bool:
    """True when `a` contains every integer in `bounds`."""
    return _covers_all(a, bounds)


# ----------------------------------------------------------------------
# expressions to regions
# ----------------------------------------------------------------------
def _lit_int(e: Optional[M.MExpr]) -> Optional[int]:
    """An integer constant, folding `+ - *` over literals and unary minus."""
    if isinstance(e, M.MLit) and e.kind == "int":
        return int(e.value)
    if isinstance(e, M.MUn) and e.op == "-":
        inner = _lit_int(e.operand)
        return None if inner is None else -inner
    if isinstance(e, M.MBin) and e.op in ("+", "-", "*"):
        a, b = _lit_int(e.left), _lit_int(e.right)
        if a is None or b is None:
            return None
        if e.op == "+":
            return a + b
        if e.op == "-":
            return a - b
        return a * b
    return None


def _atom(op: str, subject: str, value: int) -> List[Interval]:
    """The single-variable region of `x <op> value`."""
    if op == "<":
        return [(None, value - 1)]
    if op == "<=":
        return [(None, value)]
    if op == ">":
        return [(value + 1, None)]
    if op == ">=":
        return [(value, None)]
    if op == "==":
        return [(value, value)]
    if op == "!=":
        # expressed as a union to keep the algebra uniform; `_merge` keeps
        # it two intervals unless the type bound collapses one away
        return [(None, value - 1), (value + 1, None)]
    raise KeyError(op)


def _region(e: Optional[M.MExpr]
            ) -> Optional[Tuple[Optional[str], List[Interval]]]:
    """The exact region of a guard: (subject binding, intervals).

    `subject` None means the region is a constant -- `true` or `false` -- and
    it covers or covers nothing regardless of any binding.
    """
    if e is None:
        return None
    if isinstance(e, M.MLit) and e.kind == "bool":
        return (None, [(None, None)]) if e.value else (None, [])
    if isinstance(e, M.MUn) and e.op == "not":
        inner = _region(e.operand)
        if inner is None:
            return None
        subject, intervals = inner
        if not intervals:
            return (subject, [(None, None)])
        if intervals == [(None, None)]:
            return (subject, [])
        # complement over the unbounded integers; the top level re-checks
        # coverage against the type bounds
        merged = _merge(intervals)
        if merged is None:
            return None
        gaps: List[Interval] = []
        cursor: Optional[int] = None          # start of the current gap
        first = True
        for ilo, ihi in merged:
            if first:
                if ilo is not None:
                    gaps.append((None, ilo - 1))
                first = False
            else:
                # `_merge` guarantees a real gap between successive intervals
                gaps.append((cursor, ilo - 1 if ilo is not None else None))
            if ihi is None:
                return subject, _merge(gaps) or []
            cursor = ihi + 1
        gaps.append((cursor, None))           # trailing gap to +infinity
        result = _merge(gaps)
        return None if result is None else (subject, result)
    if isinstance(e, M.MBin):
        if e.op == "and":
            left, right = _region(e.left), _region(e.right)
            if left is None or right is None:
                return None
            sl, il = left
            sr, ir = right
            if sl is not None and sr is not None and sl != sr:
                return None          # two subjects: outside the fragment
            merged = _intersect_lists(il, ir)
            if merged is None:
                return None
            return sl if sl is not None else sr, merged
        if e.op == "or":
            left, right = _region(e.left), _region(e.right)
            if left is None or right is None:
                return None
            sl, il = left
            sr, ir = right
            if sl is not None and sr is not None and sl != sr:
                return None
            # a `true` arm (constant full region) makes the whole disjunction
            # constant-full only when the other arm is also constant; if the
            # other arm has a subject, the union is full *over* that subject,
            # which the empty-gap merge represents as [(-inf, +inf)] fine.
            merged = _union(il, ir)
            if merged is None:
                return None
            return sl if sl is not None else sr, merged
        if e.op in ("<", "<=", ">", ">=", "==", "!="):
            lk, rk = e.left, e.right
            if isinstance(lk, M.MRef) and _lit_int(rk) is not None:
                return lk.binding, _atom(e.op, lk.binding, _lit_int(rk))
            if isinstance(rk, M.MRef) and _lit_int(lk) is not None:
                flipped = _FLIP[e.op]
                return rk.binding, _atom(flipped, rk.binding, _lit_int(lk))
            lv, rv = _lit_int(lk), _lit_int(rk)
            if lv is not None and rv is not None:       # constant comparison
                truth = {
                    "<": lv < rv, "<=": lv <= rv, ">": lv > rv, ">=": lv >= rv,
                    "==": lv == rv, "!=": lv != rv,
                }[e.op]
                return (None, [(None, None)] if truth else [])
            return None
    return None


# ----------------------------------------------------------------------
# the top level: does a selection's guard set cover and separate?
# ----------------------------------------------------------------------
def prove_selection(guards: List[Optional[M.Constraint]],
                    type_of) -> Optional[Proof]:
    """Prove exhaustiveness and exclusivity of one selection's `when` guards.

    `guards` are the alternatives' `when` constraints in member order.
    `type_of` maps a binding name to its declared type name -- the prover
    asks for the *subject's* type, which is the binding the guards compare,
    not the binding the selection yields.  Returns None when the guard set
    is outside the decidable fragment -- the caller then records
    `unprovable`, exactly as before, and the runtime fault keeps covering
    the case.
    """
    if not guards:
        return None

    def declared(name: str) -> Optional[str]:
        if callable(type_of):
            return type_of(name)
        return type_of.get(name)

    subject: Optional[str] = None
    regions: List[List[Interval]] = []
    for g in guards:
        expr = getattr(g, "expr", None)
        region = _region(expr)
        if region is None:
            return None
        s, intervals = region
        if s is not None:
            if subject is not None and subject != s:
                return None
            subject = s
        regions.append(intervals)
    if subject is None:
        # every guard is a constant: `true`/`false` selections
        any_true = any(r == [(None, None)] for r in regions)
        pair_disjoint = all(not (r1 and r2 and _intersect_lists(r1, r2))
                            for i, r1 in enumerate(regions)
                            for r2 in regions[i + 1:])
        return Proof(exhaustive=any_true, exclusive=pair_disjoint,
                     subject="", type_name="",
                     note="constant guards")
    type_name = declared(subject)
    bounds = INT_BOUNDS.get(type_name or "")
    if bounds is None:
        return None          # not a sized integer: floats (NaN) and unknowns
    # normalise each region against the type bounds so coverage is exact
    clipped: List[List[Interval]] = []
    for r in regions:
        piece = _intersect_lists(r, [bounds])
        if piece is None:
            return None
        clipped.append(piece)
    union: List[Interval] = []
    for r in clipped:
        merged = _union(union, r)
        if merged is None:
            return None
        union = merged
    exhaustive = _covers_all(union, bounds)
    exclusive = True
    for i, a in enumerate(clipped):
        for b in clipped[i + 1:]:
            inter = _intersect_lists(a, b)
            if inter is None:
                return None
            if inter:
                exclusive = False
                break
        if not exclusive:
            break
    note = (f"exact interval partition over `{subject}` : {type_name}"
            if exhaustive and exclusive else
            ("coverage proved" if exhaustive else
             ("alternatives proved disjoint" if exclusive else
              "decided not exhaustive or exclusive")))
    return Proof(exhaustive=exhaustive, exclusive=exclusive,
                 subject=subject, type_name=type_name or "", note=note)
