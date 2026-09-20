"""Bounded recursion: the one input class that used to crash the toolchain.

A recursive-descent parser and a recursive tree walker both consume host stack
proportional to the *nesting* of their input.  Python's stack is finite, so a
deeply nested source file -- which any text editor can produce, and which a
fuzzer producing random bytes will produce -- could exhaust it.  What came out
was a raw ``RecursionError`` traceback, which breaks the toolchain's own rule
that no input may produce a Python traceback, and which is indistinguishable
from a compiler bug to whoever is holding the file.

The fix is a bound, not a bigger stack.  Two independent bounds are needed
because two different things can nest:

* **Syntactic nesting** -- ``((((1))))``, ``[[[[]]]]``, ``f(f(f(x)))``.  The
  parser descends about sixteen Python frames per level, so the parser itself
  is bounded: every place it descends into a nested construct goes through
  :meth:`BoundedRecursion.descend`.
* **Semantic depth** -- ``1+1+1+...`` five thousand times is *flat* source but
  a five-thousand-deep left-leaning tree, and every recursive consumer of the
  AST (the checker, the lowering pass, the interpreter) walks it.  The parser
  cannot see this, so it is bounded after parsing over the tree itself
  (:func:`deepest`).

Both limits are *derived from the host's stack budget* rather than hardcoded,
so a program that raises ``sys.setrecursionlimit`` -- for instance a build
running in a thread with a large stack -- gets correspondingly deeper input
accepted instead of a crash.  The policy ceilings in this module are the
upper bounds; the effective limits are computed below them.

Nothing here is a performance claim: the limits are chosen to sit well inside
the measured budget, which makes them conservative, not tight.
"""

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

# ----------------------------------------------------------------------
# policy ceilings
# ----------------------------------------------------------------------

#: Hardest ceiling on syntactic nesting, whatever the host stack allows.
#: Reached only when the caller has raised ``sys.setrecursionlimit`` enough
#: to earn it; the default host budget lands far below.
MAX_PARSE_NESTING = 96

#: Hardest ceiling on AST depth (long chains and deeply nested expressions).
MAX_AST_DEPTH = 256

#: The diagnostic code both bounds report.
E_NESTING_CODE = "E-nesting-too-deep"

# ----------------------------------------------------------------------
# host stack budget
# ----------------------------------------------------------------------

#: Python frames one syntactic nesting level costs: the parser's
#: expression-precedence chain (``parse_expr`` down to ``parse_primary``) plus
#: the guard frames.  Measured, then rounded up.
FRAMES_PER_NESTING_LEVEL = 16

#: Python frames one level of AST depth costs in the worst recursive consumer
#: (the checker's ``infer`` / ``expr_Binary`` pair, the interpreter's
#: expression walker).  Measured: a flat ``1+1+...`` chain compiles and runs at
#: 450 terms and overflows at 500, which is about two frames per level, so this
#: is the rounded-up figure and the resulting bound sits roughly 1.7x below the
#: point where the host actually ran out.  A bound that refuses input the host
#: could have carried is a cost; a bound that lets the host crash is a defect.
FRAMES_PER_AST_LEVEL = 3

#: Frames kept in hand so the diagnostic for exceeding the bound can itself be
#: constructed and raised -- a fault that cannot be reported is not a fix.
STACK_RESERVE = 160

#: Fallback limits for a host that cannot report its stack depth (not CPython).
_FALLBACK_LIMIT = 32


def python_frame_depth() -> int:
    """Frames currently on this thread's Python stack.

    ``sys._getframe`` is a CPython implementation detail; when it is absent the
    caller falls back to the conservative constant rather than guessing.
    """
    try:
        frame = sys._getframe()
    except (AttributeError, ValueError):        # pragma: no cover - not CPython
        return 0
    depth = 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def _budget(frames_per_level: int, ceiling: int,
            reserve: int = STACK_RESERVE) -> int:
    """How many levels of this kind the remaining host stack can carry."""
    if frames_per_level <= 0:                   # pragma: no cover - defensive
        return ceiling
    usable = sys.getrecursionlimit() - python_frame_depth() - reserve
    if usable <= 0:
        return _FALLBACK_LIMIT
    return max(8, min(ceiling, usable // frames_per_level))


def parse_nesting_limit() -> int:
    """Effective limit on syntactic nesting for the host as configured now."""
    return _budget(FRAMES_PER_NESTING_LEVEL, MAX_PARSE_NESTING)


def ast_depth_limit() -> int:
    """Effective limit on AST depth for the host as configured now."""
    return _budget(FRAMES_PER_AST_LEVEL, MAX_AST_DEPTH)


# ----------------------------------------------------------------------
# bounding the parsers
# ----------------------------------------------------------------------

class BoundedRecursion:
    """Mixin that gives a recursive-descent parser a nesting bound.

    Mixed into a parser to bound how deeply it may descend.  The counter is
    per-instance and unwound in ``finally``, so sibling expressions at the
    same level do not accumulate -- only genuine nesting does.
    """

    #: Current nesting depth; a class attribute so ``getattr`` is total.
    _nesting: int = 0

    def _nesting_limit(self) -> int:
        """The nesting budget for this parse, measured once and then fixed.

        Computed on first use, which is the shallowest point of the parse, and
        cached.  Recomputing it per descent does not work: the budget is
        derived from how much host stack is *left*, so measuring it forty
        levels down would report a shrinking limit and refuse a file that the
        outermost frame had just admitted.
        """
        limit = getattr(self, "_nesting_budget", None)
        if limit is None:
            limit = parse_nesting_limit()
            self._nesting_budget = limit
        return limit

    def descend(self, method, *args: Any, **kwargs: Any) -> Any:
        """Enter one nested construct, bounded.

        Used at the recursion sites rather than as a decorator, and that
        distinction matters: decorating every function in the precedence chain
        would count four or five units for one parenthesised subexpression,
        because ``(`` re-enters the whole chain.  Counting *descents* makes one
        unit mean one level of nesting, for every shape of input -- parens,
        calls, lists, unary runs, right-associative powers -- so the limit can
        be reasoned about, and reported to the user, in the terms the user
        wrote the program in.
        """
        depth = getattr(self, "_nesting", 0) + 1
        if depth > self._nesting_limit():
            raise self.nesting_fault()
        self._nesting = depth
        try:
            return method(*args, **kwargs)
        finally:
            self._nesting = depth - 1

    def nesting_fault(self):
        """The diagnostic raised when the bound is reached."""
        limit = self._nesting_limit()
        return self.error(                        # type: ignore[attr-defined]
            f"expression nests deeper than {limit} levels",
            help_text=(
                "the limit is derived from this host's stack budget, not a "
                "fixed language rule; split the expression, or raise "
                "sys.setrecursionlimit for deeper generated input"),
            code=E_NESTING_CODE)


# ----------------------------------------------------------------------
# measuring the tree
# ----------------------------------------------------------------------

#: Field names per class, cached: ``dataclasses.fields`` rebuilds its tuple on
#: every call, and this walk visits every node of the tree.
_FIELD_CACHE: Dict[type, Tuple[str, ...]] = {}

#: Give up measuring after this many nodes and report "not exceeded".  Only
#: reachable for a tree far larger than any real program, and failing open is
#: correct here because the caller keeps a ``RecursionError`` backstop.
MAX_VISITS = 2_000_000


def _field_names(cls: type) -> Tuple[str, ...]:
    names = _FIELD_CACHE.get(cls)
    if names is None:
        names = tuple(f.name for f in fields(cls))
        _FIELD_CACHE[cls] = names
    return names


def _children(obj: Any) -> Iterator[Any]:
    """The nodes directly under ``obj`` in an AST, whatever its shape."""
    if is_dataclass(obj) and not isinstance(obj, type):
        for name in _field_names(type(obj)):
            yield getattr(obj, name, None)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        yield from obj
    elif isinstance(obj, dict):
        yield from obj.values()


def deepest(root: Any, limit: Optional[int] = None
            ) -> Tuple[int, Optional[Any]]:
    """The depth of the deepest node under ``root``, iteratively.

    Returns ``(depth, offender)``.  ``offender`` is the first node found deeper
    than ``limit`` -- and the walk stops there, so a pathologically deep tree
    costs the limit, not the tree -- or ``None`` when the tree is within it.
    Iterative on purpose: measuring recursion with recursion is how this class
    of bug starts.
    """
    if limit is None:
        limit = ast_depth_limit()
    best = 0
    visits = 0
    stack: list = [(root, 1)]
    while stack:
        obj, depth = stack.pop()
        if depth > best:
            best = depth
        if depth > limit:
            return best, obj
        visits += 1
        if visits > MAX_VISITS:                 # pragma: no cover - huge input
            return best, None
        for child in _children(obj):
            if child is None or isinstance(child, (str, bytes, int, float,
                                                   bool)):
                continue                        # leaves cannot nest
            stack.append((child, depth + 1))
    return best, None
