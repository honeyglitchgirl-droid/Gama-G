"""Discharging a `holds` promise without running the program.

The core already separates a promise from how it is discharged: a
:class:`~gamag.core.mir.Constraint` carries `proven`, `runtime` or
`unprovable`, and :mod:`gamag.core.guardproof` decides that separation for
selection guards.  Before v1.3 a `holds` clause was never `proven`: the
compiler parsed it, kept its verbatim text, emitted an ``Op.REQUIRE`` and
waited.  A program that promised something its own constants made inevitable
-- `source reading : I64 from 72`, `computes reading`, `holds checked > 0` --
was checked at runtime like one that might break.

This module computes the other half.  It establishes two things and refuses
everything else, and the refusal is recorded with a reason rather than as a
shrug.

**Closed-form evaluation.**  A binding the graph gives exactly one value to --
a `source` with a literal origin, or an operation with a single producer whose
`computes` expression is built from literals and such bindings -- has a value
the compiler knows.  Evaluating a predicate over those values with the same
operators the runtime uses is not an approximation of the program, it *is* the
program's arithmetic, so the verdict is exact: the promise either holds or it
does not.  This is why floats are welcome here.  guardproof refuses a float
*subject* because "the region covers every float" is false in the presence of
NaN; evaluating `72.0 > 0.0` involves no such claim.  A NaN cannot enter,
because the only way to make one -- `0.0 / 0.0` -- is refused below.

One limit on that sentence, stated plainly: the verdict is about *this closed
program*.  A `source n : F64 from 2.0` is a value the program writes for
itself, and the reference toolchain offers no way to supply a source from
outside the file -- no `--input`, no environment read.  Should a toolchain ever
gain one, `_collect_facts` must stop folding that binding (the shape is
already there: a source with no origin is a parameter and yields no fact), and
until then a discharged promise is a claim about the text in front of the
compiler, which is what the note beside it says.

**Interval implication.**  When a value is not fixed, a promise about a single
sized-integer binding can still follow from what the graph does say: the
operation's `when` guard, and the binding's own type bounds.  `holds score >=
0` under `when score >= 75` is arithmetic, and the guard prover's finite
unions of integer intervals decide it exactly.  Floats are excluded again, for
guardproof's reason: bounding a real without reasoning about NaN is unsound.

What is refused, and stays a runtime check:

* `/ % **` -- integer `/` and `%` truncate through `ops.trunc_div` and
  `trunc_mod`, and a zero divisor faults.  Folding them would mean trusting a
  second copy of that arithmetic; the promise is checked where it is decided.
* calls, field reads, indexing, `.items` -- anything whose value this module
  cannot name without re-implementing the standard library.
* a binding with more than one producer, which is what a selection is, and a
  `state`, which a `transition` may alter;
* a `refine` binding, whose value changes every round: `starts` fixes the
  first one and the promise is about the last;
* predicates over two bindings that are not both constants.

A proof here does not remove an obligation.  `capability_checks` in
:mod:`gamag.core.native` states the reason, and it applies verbatim: "a proof
is a reason to trust the program, not a reason to remove the boundary."  The
``Op.REQUIRE`` for a discharged `holds` is emitted exactly as before, so a
hand-edited IR or a backend that reorders instructions still meets the check.

A refutation is reported as a warning, never an error, and that is a
deliberate limit.  A `holds` the compiler can prove false describes a program
that faults every time it runs, which is worth saying out loud -- but refusing
the build would replace the classified `ContractViolation` the specification
asks for with a compiler's opinion, and a program whose source is edited later
must not find that its guarantee quietly changed status.  So the prover
explains, and the runtime still enforces.

The export at the bottom turns what this module *cannot* decide into SMT-LIB2
text, for a solver somebody else runs.  Gama-G never invokes a solver: no
solver is required, assumed, or shipped.  An exported obligation is not a
finding until a solver answers `unsat` for it, and nothing here says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import guardproof as GP
from . import mir as M

#: The kinds whose binding holds exactly the value written for it.  `state` is
#: absent because a `transition` alters it, `refine` because the value changes
#: every round, and `fanout`/`dispatch` because the value is per item or per
#: case rather than per binding.
CONSTANT_KINDS = ("source", "compute")

#: The depth a fold may recurse to.  A contract clause deeper than this is not
#: a promise anyone wrote by hand, and a bound keeps the prover linear.
MAX_DEPTH = 32

_PROVEN = "proven"
_REFUTED = "refuted"
_UNDECIDED = "undecided"


@dataclass
class Fact:
    """The exact value a binding has, and what it cost to know that."""

    binding: str
    value: Any
    type_name: str = ""
    node: str = ""
    #: The one binding this value came from, when it came straight from a
    #: binding rather than being computed: `checked` = 72 because it *is*
    #: `reading`.  A note that stops at the number leaves the reader to find
    #: the source themselves.
    from_binding: str = ""
    #: True when the producing operation has a `when` guard: the value is
    #: fixed only on the paths where that guard is active, so the fact may be
    #: used to discharge that operation's own promise and nowhere else.
    conditional: bool = False


@dataclass
class Verdict:
    """What the prover concluded about one clause, and why.

    `generic` marks the fallback wording -- "I could not evaluate this" -- as
    opposed to a reason the prover actually established.  The report pass
    replaces a generic note with the sharper one `_blocked` finds and never
    overwrites a specific one, because the more precise obstacle is the more
    useful sentence.
    """

    status: str
    note: str
    values: Dict[str, Any] = field(default_factory=dict)
    generic: bool = False


class _Unknown:
    """The marker for "this expression has no value I can name"."""

    def __repr__(self) -> str:                                   # pragma: no cover
        return "<unknown>"


UNKNOWN = _Unknown()


# ----------------------------------------------------------------------
# the constants the graph fixes
# ----------------------------------------------------------------------
def shown(value: Any) -> str:
    """A constant as the program would have written it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(value)


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fits(type_name: str, value: Any) -> bool:
    """Whether `value` may be stored in a slot of `type_name`.

    The runtime checks the range on store (`vm.store` raises
    `IntegerOverflow`), not on arithmetic.  A folded constant that does not fit
    therefore describes a program that faults before its promise is ever
    tested, and no verdict about the promise is honest.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return True
    bounds = GP.INT_BOUNDS.get(type_name or "")
    if bounds is None:
        return True                      # not a sized integer: no range to break
    lo, hi = bounds
    return lo <= value <= hi


def _fold(expr: Optional[M.MExpr], facts: Dict[str, Fact],
          depth: int = 0) -> Any:
    """The exact value of a closed expression, or UNKNOWN.

    Every operator here mirrors `runtime.ops` on concrete values: `values_equal`
    for `==` (and so the Bool-versus-non-Bool rule that makes them unequal),
    `compare` for ordering, and Python's own `+ - *` for the arithmetic the
    runtime performs.  Operators are refused rather than approximated: an
    unknown is a finding, a wrong value is a bug.
    """
    if expr is None or depth > MAX_DEPTH:
        return UNKNOWN
    if isinstance(expr, M.MLit):
        if expr.kind in ("int", "float", "bool", "string"):
            return expr.value
        return UNKNOWN
    if isinstance(expr, M.MRef):
        fact = facts.get(expr.binding)
        return UNKNOWN if fact is None else fact.value
    if isinstance(expr, M.MUn):
        inner = _fold(expr.operand, facts, depth + 1)
        if isinstance(inner, _Unknown):
            return UNKNOWN
        if expr.op in ("not", "!"):
            return not inner if isinstance(inner, bool) else UNKNOWN
        if expr.op == "-":
            return -inner if _numeric(inner) else UNKNOWN
        if expr.op == "+":
            return inner
        return UNKNOWN
    if isinstance(expr, M.MBin):
        left = _fold(expr.left, facts, depth + 1)
        right = _fold(expr.right, facts, depth + 1)
        if isinstance(left, _Unknown) or isinstance(right, _Unknown):
            return UNKNOWN
        op = expr.op
        if op in ("and", "or"):
            if not (isinstance(left, bool) and isinstance(right, bool)):
                return UNKNOWN
            return left and right if op == "and" else left or right
        if op in ("==", "!="):
            # `values_equal` says a Bool never equals a non-Bool; anything
            # else here is compared as the runtime would compare it.
            if isinstance(left, bool) != isinstance(right, bool):
                equal = False
            elif type(left) is not type(right):
                return UNKNOWN           # int against float: not folded
            else:
                equal = left == right
            return equal if op == "==" else not equal
        if op in ("<", "<=", ">", ">="):
            if isinstance(left, bool) or isinstance(right, bool):
                return UNKNOWN           # the runtime faults: a Bool has no order
            if type(left) is not type(right):
                return UNKNOWN
            if not isinstance(left, (int, float, str)):
                return UNKNOWN
            return {
                "<": left < right, "<=": left <= right,
                ">": left > right, ">=": left >= right,
            }[op]
        if op == "+":
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, str) or isinstance(right, str):
                return UNKNOWN
            if _numeric(left) and _numeric(right):
                return left + right
            return UNKNOWN
        if op == "-" and _numeric(left) and _numeric(right):
            return left - right
        if op == "*" and _numeric(left) and _numeric(right):
            return left * right
        return UNKNOWN                   # `/ % **` and anything unknown
    return UNKNOWN


def _reason(expr: Optional[M.MExpr], facts: Dict[str, Fact],
            depth: int = 0) -> str:
    """One clause naming the first thing outside the fragment.

    The note a reader sees next to `checked at runtime`, so that "the compiler
    could not" and "the compiler would not" are distinguishable from outside.
    """
    if expr is None:
        return "no expression: the clause is text this compiler does not parse"
    if depth > MAX_DEPTH:
        return f"the clause is deeper than {MAX_DEPTH} levels"
    if isinstance(expr, M.MLit):
        return "" if expr.kind in ("int", "float", "bool", "string") \
            else f"a {expr.kind} literal"
    if isinstance(expr, M.MRef):
        if expr.binding in facts:
            return ""
        return f"the value of `{expr.binding}` is not fixed by the graph"
    if isinstance(expr, M.MUn):
        inner = _reason(expr.operand, facts, depth + 1)
        return inner or f"unary `{expr.op}` is not folded"
    if isinstance(expr, M.MBin):
        if expr.op in ("/", "%", "**"):
            return (f"`{expr.op}` is not folded: integer `{expr.op}` truncates "
                    f"and a zero divisor faults, so the value is the runtime's "
                    f"to give")
        left = _reason(expr.left, facts, depth + 1)
        right = _reason(expr.right, facts, depth + 1)
        for part in (left, right):
            if part:
                return part
        return f"`{expr.op}` over operands whose values are not both known"
    if isinstance(expr, M.MCall):
        name = f"{expr.module}.{expr.name}" if expr.module else expr.name
        return f"`{name}(...)` is a standard-library call: this prover does not"
    if isinstance(expr, M.MField):
        return f"a field read `{expr.attr}` is not modelled"
    if isinstance(expr, M.MIndex):
        return "indexing is not modelled"
    if isinstance(expr, M.MItems):
        return "`.items` is not modelled"
    return f"a {type(expr).__name__} expression is not modelled"


def _fact(binding: str, expr: Optional[M.MExpr], node: M.OpNode,
          facts: Dict[str, Fact]) -> Optional[Fact]:
    """The constant this expression fixes for `binding`, or None.

    A `source` with no `from` origin is a parameter of the intent and has no
    value to name.  A value that does not fit the declared type is refused:
    `vm.store` raises `IntegerOverflow` on the way in, so the operation faults
    before any promise about its result could be tested.
    """
    if expr is None or not binding:
        return None
    value = _fold(expr, facts)
    if isinstance(value, _Unknown):
        return None
    type_name = node.type.name if node.type is not None else ""
    if not _fits(type_name, value):
        return None
    origin = expr.binding if isinstance(expr, M.MRef) and expr.binding in facts \
        else ""
    return Fact(binding=binding, value=value, type_name=type_name,
                node=node.name, from_binding=origin,
                conditional=node.when is not None)


def _single_producer(graph: M.OperationGraph, binding: str) -> bool:
    """True when one operation, and only one, writes this binding.

    A selection's binding has one producer per arm, and a `state` has one per
    `transition` that alters it: neither has a value the graph fixes.
    """
    if binding in graph.selections:
        return False
    producers = set(graph.producers_of.get(binding) or ())
    return len(producers) <= 1


def _collect_facts(model: M.SemanticModel) -> Tuple[Dict[str, Fact],
                                                    Dict[str, Dict[str, Fact]]]:
    """Every binding the graph gives exactly one value to.

    Two environments come back.  `global` holds constants fixed by an ungarded
    `source` or `compute`, usable anywhere in the intent.  `per node` maps an
    operation to that global map plus the value *its own* `computes` writes: a
    promise checked at the end of an operation may rest on what that operation
    computed, even when the binding it names is a selection arm or a refined
    value whose next round differs.  Nothing else may read a node's private
    fact, which is the whole reason the two are kept apart.
    """
    graph = getattr(model, "operations", None)
    if graph is None:
        return {}, {}
    facts: Dict[str, Fact] = {}
    private: Dict[str, Dict[str, Fact]] = {}
    rank = {name: index for index, names in enumerate(graph.levels)
            for name in names}
    # `source` nodes are keyed by the binding they produce and sit outside the
    # derived levels; `state` nodes are excluded entirely, since a `transition`
    # alters them.
    nodes = list(graph.inputs.values()) + list(graph.nodes.values())
    for _round in range(len(nodes) + 1):        # acyclic, but bound the search
        progressed = False
        for node in sorted(nodes, key=lambda n: (rank.get(n.name, -1), n.name)):
            if node.kind == "source":
                fact = _fact(node.produces, node.origin, node, facts)  # type: ignore[attr-defined]
                if fact is not None and _single_producer(graph, node.produces):
                    facts[node.produces] = fact
                    progressed = True
                continue
            if node.kind != "compute" or node.produces in facts:
                continue
            fact = _fact(node.produces, node.value, node, facts)  # type: ignore[attr-defined]
            if fact is None:
                continue
            if _single_producer(graph, node.produces) and not fact.conditional:
                facts[node.produces] = fact
            # an operation's own promise may use its own value, guarded or not,
            # even when no other operation may
            private.setdefault(node.name, {})[node.produces] = fact
            progressed = True
        if not progressed:
            break
    per_node = {name: dict(facts, **extra) for name, extra in private.items()}
    return facts, per_node


def _refs(expr: Optional[M.MExpr], found: Optional[List[str]] = None,
          depth: int = 0) -> List[str]:
    """The bindings an expression reads."""
    out: List[str] = [] if found is None else found
    if expr is None or depth > MAX_DEPTH:
        return out
    if isinstance(expr, M.MRef):
        out.append(expr.binding)
        return out
    for attr in ("left", "right", "operand"):
        child = getattr(expr, attr, None)
        if child is not None:
            _refs(child, out, depth + 1)
    for arg in getattr(expr, "args", None) or []:
        _refs(arg, out, depth + 1)
    obj = getattr(expr, "obj", None)
    if obj is not None:
        _refs(obj, out, depth + 1)
    return out


def _blocked(model: M.SemanticModel, expr: M.MExpr,
             facts: Dict[str, Fact]) -> str:
    """Why a binding this clause reads has no fixed value, in one clause.

    The three reasons a core program meets are distinct faults of the program's
    own design, and a reader deserves to be told which one it is.
    """
    graph = model.operations
    for name in sorted(set(_refs(expr))):   # a lookup chain, not logic
        if name in graph.selections:
            return (f"`{name}` is yielded by several guarded operations, so "
                    f"the graph fixes no single value for it")
        node = graph.resources.get(name)
        if node is not None:
            return (f"`{name}` is a `state`, and a `transition` may alter it "
                    f"between the check and the promise")
        producers = [graph.nodes[p] for p in graph.producers_of.get(name, [])
                     if p in graph.nodes]
        if not producers and name in graph.inputs:
            producers = [graph.inputs[name]]
        for producer in producers:
            if producer.kind == "refine":
                return (f"`{name}` is refined: `starts` fixes its first value "
                        f"and `repeats` changes it every round, while the "
                        f"promise is about the value that ends the repetition")
            if producer.when is not None:
                return (f"`{name}` is written only when `{producer.when.text}` "
                        f"-- the guard guards this operation too, and this "
                        f"prover assumes only the operation's own `when`")
            written = getattr(producer, "value", None) \
                or getattr(producer, "origin", None)
            if written is None:
                return (f"`{name}` comes from outside the intent, so its value "
                        f"is not the program's to fix")
            why = _reason(written, facts)
            if why:
                return f"the value of `{name}` is not closed: {why}"
    return ""


# ----------------------------------------------------------------------
# the two proofs
# ----------------------------------------------------------------------
def _range_of(expr: Optional[M.MExpr], subject: str) -> Optional[Tuple[int, int]]:
    """The integer range a subject binding may hold, from its resolved type.

    Expressions carry the type the checker settled on, which is the authority
    a bounds argument needs: an `IntType` knows its range, and a `FloatType`
    must not be given one (guardproof's reason -- `NaN` breaks a covering
    argument -- applies unchanged here).
    """
    for node in _walk(expr):
        if isinstance(node, M.MRef) and node.binding == subject:
            resolved = getattr(node, "type", None)
            bounds = getattr(resolved, "range", None)
            if bounds is None or type(bounds) is not tuple:
                return None
            if type(resolved).__name__ != "IntType":
                return None        # floats and anything else: no bounds argument
            return int(bounds[0]), int(bounds[1])
    return None


def _walk(expr: Optional[M.MExpr], depth: int = 0):
    if expr is None or depth > MAX_DEPTH:
        return
    yield expr
    for attr in ("left", "right", "operand"):
        yield from _walk(getattr(expr, attr, None), depth + 1)
    for arg in getattr(expr, "args", None) or []:
        yield from _walk(arg, depth + 1)
    yield from _walk(getattr(expr, "obj", None), depth + 1)


FLOAT_TYPE_NAMES = ("F32", "F64", "Float")


def _is_float(type_name: str) -> bool:
    return (type_name or "") in FLOAT_TYPE_NAMES


def model_type_of(type_of: Optional[Callable[[str], str]],
                  binding: str) -> str:
    return type_of(binding) if type_of is not None else ""


def _float_subject(expr: Optional[M.MExpr], subject: str) -> bool:
    """True when the subject's resolved type is a float type.

    The reason is guardproof's and it is worth naming in the note: this prover
    will not claim a set of comparisons covers a float, because NaN defeats
    coverage.  It will happily *evaluate* a float expression, which is why the
    fold above takes floats and the interval argument below refuses them.
    """
    for node in _walk(expr):
        if isinstance(node, M.MRef) and node.binding == subject:
            return type(getattr(node, "type", None)).__name__ in (
                "FloatType", "F64Type", "F32Type")
    return False


def _type_of(model: M.SemanticModel, binding: str) -> str:
    graph = model.operations
    node = (graph.inputs.get(binding) or graph.resources.get(binding)
            or graph.nodes.get(binding))
    if node is not None and node.type is not None:
        return node.type.name
    producers = graph.producers_of.get(binding) or []
    names = {graph.nodes[p].type.name for p in producers
             if p in graph.nodes and graph.nodes[p].type is not None}
    return names.pop() if len(names) == 1 else ""


def prove(expr: Optional[M.MExpr], facts: Dict[str, Fact],
          *, assumption: Optional[M.MExpr] = None,
          type_of: Optional[Callable[[str], str]] = None,
          blocked: Optional[Callable[[M.MExpr], str]] = None) -> Verdict:
    """Discharge one clause: proven, refuted, or undecided with a reason.

    `assumption` is the operation's `when` guard -- a legitimate fact, because
    `native.Lowerer._constraints` emits the check inside the guarded block, so
    the promise is tested only where the guard holds.  Bounds come from the
    resolved type on the clause's own subject; `type_of` is the fallback for a
    reference whose type the checker never settled.
    """
    if expr is None:
        return Verdict(_UNDECIDED, "the clause carries no expression")

    def why(node: M.MExpr = expr) -> str:
        """The most specific reason available for this clause.

        `_reason` reads the clause; the caller's `blocked` hook may read the
        graph and explain a binding by how it is written, which is usually the
        truer sentence -- `dose <= 500.0` fails to be proved because of what
        `dose` is computed from, not because of anything in the comparison.
        """
        outer = blocked(node) if blocked is not None else ""
        return outer or _reason(node, facts)

    # 1. exact evaluation, whenever every leaf is fixed
    value = _fold(expr, facts)
    if not isinstance(value, _Unknown):
        if not isinstance(value, bool):
            return Verdict(_UNDECIDED,
                           "the clause is not a predicate: it evaluates to a "
                           f"{type(value).__name__}, not a Bool")
        chain = []
        for b in sorted(set(_refs(expr))):
            fact = facts.get(b)
            if fact is None:
                continue
            link = f"`{b}` = {shown(fact.value)}"
            if fact.from_binding:
                link += f", which is `{fact.from_binding}`"
            chain.append(link)
        tail = f" ({'; '.join(chain)})" if chain else ""
        if value:
            return Verdict(_PROVEN,
                           "evaluates to true from the constants the graph "
                           f"fixes{tail}",
                           {b: facts[b].value for b in sorted(_refs(expr))
                            if b in facts})
        return Verdict(_REFUTED,
                       "evaluates to false from the constants the graph fixes"
                       f"{tail}",
                       {b: facts[b].value for b in sorted(_refs(expr))
                        if b in facts})

    # a single float subject is worth naming before anything else: the region
    # grammar is integer-only, and a reader who is told "outside the fragment"
    # cannot tell that apart from the prover having nothing to say
    refs = sorted(set(_refs(expr)))
    only = refs[0] if len(refs) == 1 else ""
    float_subject = bool(only) and (
        _is_float(type_of(only) if type_of is not None else "")
        or _float_subject(expr, only))

    region = GP.region_of(expr)
    if region is None:
        reason = why()
        if float_subject:
            return Verdict(
                _UNDECIDED,
                (reason + ": " if reason else "")
                + f"no bounds argument is available for `{only}`, because a "
                  f"promise that covers the reals is false while NaN exists")
        return Verdict(_UNDECIDED,
                       reason or "the clause is outside the fragment this "
                                 "prover implements",
                       generic=not reason)
    subject, claimed = region
    if subject is None:
        return Verdict(_PROVEN if claimed else _REFUTED,
                       "the clause is a constant")
    bounds = _range_of(expr, subject)
    type_name = ""
    if bounds is None and type_of is not None:
        type_name = type_of(subject) or ""
        bounds = GP.INT_BOUNDS.get(type_name)
    if bounds is None:
        if float_subject:
            return Verdict(
                _UNDECIDED,
                f"`{subject}` : {type_name or 'a float type'} is not bounded -- "
                f"a promise that covers the reals is false while NaN exists, "
                f"and this prover will not bound a value it cannot order")
        return Verdict(_UNDECIDED,
                       why()
                       or (f"`{subject}` is not a sized integer type, so it has "
                           f"no bounds to reason over" if type_name else
                           f"the type of `{subject}` is not known here"),
                       generic=not why())
    whole = [list(bounds)]          # the whole type, as one interval
    known = whole
    if assumption is not None:
        given = GP.region_of(assumption)
        if given is None:
            return Verdict(_UNDECIDED,
                           "the clause is undecided and its `when` guard is "
                           "outside the fragment, so there is nothing to "
                           "assume from it")
        guard_subject, guard_region = given
        if guard_subject is not None and guard_subject != subject:
            return Verdict(_UNDECIDED,
                           f"the guard bounds `{guard_subject}` while the "
                           f"promise is about `{subject}`")
        narrowed = GP.intersect(whole, guard_region)
        if narrowed is None:
            return Verdict(_UNDECIDED, "the region bound was reached")
        known = narrowed
    # clip and merge the claim itself, so `within` sees a normalised region
    clipped_claim = GP.intersect(claimed, [list(bounds)])
    if clipped_claim is None:
        return Verdict(_UNDECIDED, "the region bound was reached")
    inside = GP.within(known, clipped_claim, bounds)
    if inside is None:
        return Verdict(_UNDECIDED, "the region bound was reached")
    if inside:
        return Verdict(_PROVEN,
                       f"exact interval argument: every `{subject}` : "
                       f"{type_name} that reaches this operation satisfies the "
                       f"clause"
                       + (" (the `when` guard is what establishes it)"
                          if assumption is not None else
                          " (the type's own range is what establishes it)"))
    apart = GP.disjoint(known, clipped_claim)
    if apart:
        lo, hi = bounds
        return Verdict(_REFUTED,
                       f"exact interval argument: no `{subject}` : {type_name} "
                       f"that reaches this operation satisfies the clause (the "
                       f"type spans {lo} to {hi}"
                       + (", and the `when` guard narrows it further"
                          if assumption is not None else "") + ")")
    return Verdict(_UNDECIDED,
                   why()
                   or (f"`{subject}` can reach values inside and outside the "
                       f"clause, so the promise is neither established nor "
                       f"contradicted here"), generic=not why())


# ----------------------------------------------------------------------
# the pass the compiler runs
# ----------------------------------------------------------------------
def discharge(model: M.SemanticModel,
              warn: Optional[Callable[..., None]] = None) -> Dict[str, int]:
    """Annotate every `holds` with its verdict.  Never refuses a build.

    `warn` is the caller's diagnostic hook, so a refutation reaches the same
    stream as every other warning.  The return value is a tally for the
    report layer: how much of a program's promises the compiler can vouch for.
    """
    tally = {"proven": 0, "refuted": 0, "undecided": 0, "unchecked": 0}
    graph = getattr(model, "operations", None)
    if graph is None:
        return tally
    facts, per_node = _collect_facts(model)
    seen = set()
    todo = list(graph.inputs.values()) + list(graph.nodes.values())
    for node in todo:
        if node.name in seen:
            continue
        seen.add(node.name)
        env = dict(facts)
        env.update(per_node.get(node.name, {}))
        for hold in node.holds:
            if hold.expr is None:
                hold.proof = "the clause carries no expression"
                tally["unchecked"] += 1
                continue
            verdict = prove(hold.expr, env,
                            assumption=node.when.expr if node.when else None,
                            type_of=lambda name: _type_of(model, name),
                            blocked=lambda clause: _blocked(model, clause,
                                                            facts))
            hold.proof = verdict.note
            if verdict.status == _PROVEN:
                hold.discharge = M.DISCHARGE_PROVEN
                tally["proven"] += 1
            elif verdict.status == _REFUTED:
                tally["refuted"] += 1
                if warn is not None:
                    warn(
                        f"`{node.name}` cannot satisfy `holds {hold.text}`",
                        hold.pos, code="W-contract-refuted",
                        help_text=(
                            f"{verdict.note}. The check stays in the program "
                            f"and will fault with `{hold.fault}` on every run "
                            f"that reaches this operation; fix the promise or "
                            f"the value that makes it impossible"))
            else:
                tally["undecided"] += 1
    return tally


# ----------------------------------------------------------------------
# the export: what this prover cannot decide, for a solver somebody else runs
# ----------------------------------------------------------------------
def _smt_term(expr: Optional[M.MExpr], bound: List[str]) -> Optional[str]:
    """Encode a clause as SMT-LIB2 over Ints and Bools, or None."""
    if expr is None:
        return None
    if isinstance(expr, M.MLit):
        if expr.kind == "int":
            return str(int(expr.value))
        if expr.kind == "bool":
            return "true" if expr.value else "false"
        return None                 # reals and strings: outside this encoding
    if isinstance(expr, M.MRef):
        if expr.binding not in bound:
            return None
        return expr.binding
    if isinstance(expr, M.MUn):
        inner = _smt_term(expr.operand, bound)
        if inner is None:
            return None
        if expr.op in ("not", "!"):
            return f"(not {inner})"
        if expr.op == "-":
            return f"(- {inner})"
        return None
    if isinstance(expr, M.MBin):
        left = _smt_term(expr.left, bound)
        right = _smt_term(expr.right, bound)
        if left is None or right is None:
            return None
        if expr.op in ("<", "<=", ">", ">=", "=", "==", "+", "-", "*"):
            op = "=" if expr.op == "==" else expr.op
            return f"({op} {left} {right})"
        if expr.op == "!=":
            return f"(not (= {left} {right}))"
        if expr.op in ("and", "or"):
            return f"({expr.op} {left} {right})"
        return None
    return None


def _smt_decl(model: M.SemanticModel, binding: str) -> Optional[List[str]]:
    """The declaration and range assumption for one sized-integer binding."""
    type_name = _type_of(model, binding)
    bounds = GP.INT_BOUNDS.get(type_name or "")
    if bounds is None:
        return None
    lo, hi = bounds
    return [f"(declare-fun {binding} () Int)",
            f"(assert (and (>= {binding} {lo}) (<= {binding} {hi})))"
            f"   ; {binding} : {type_name}"]


def to_smtlib2(model: M.SemanticModel) -> str:
    """Every `holds` this compiler could not settle, as SMT-LIB2.

    The shape is the standard verification condition: the declaration of each
    binding, the type's range, the operation's `when` guard as an assumption,
    and the negated clause.  ``unsat`` for such an obligation means the promise
    is established for every value that reaches the operation.

    Sized integers only.  A clause over floats, strings, calls or two bindings
    is emitted as a comment saying it was not encoded, because a solver handed
    a wrong encoding returns a wrong answer with confidence.  Nothing in this
    file runs a solver or claims one would agree.
    """
    graph = getattr(model, "operations", None)
    intent = model.intent.name if model.intent else "Intent"
    lines = [
        "; Gama-G verification conditions, generated by `ggc check --smt`.",
        f"; intent {intent}",
        ";",
        "; Each obligation below is `(assert (not <clause>))` under the facts",
        "; the graph fixes for that operation: its type's range and its own",
        "; `when` guard.  `unsat` means the clause holds for every value that",
        "; can reach the operation.",
        ";",
        "; The reference compiler never runs a solver, never requires one, and",
        "; records nothing in the program's own output from this file.  An",
        "; obligation exported here is an unanswered question, not a result.",
        ";",
        "(set-option :produce-models true)",
        "(set-logic QF_LIA)",
    ]
    facts, per_node = _collect_facts(model)
    emitted = 0
    for node in sorted(graph.nodes.values(), key=lambda n: n.name) if graph \
            else []:
        clauses = list(node.holds)
        for hold in clauses:
            if hold.expr is None or hold.discharge == M.DISCHARGE_PROVEN:
                continue
            names = [b for b in _refs(hold.expr) if b not in facts]
            decls: List[str] = []
            ok = True
            for name in sorted(set(names)):
                one = _smt_decl(model, name)
                if one is None:
                    ok = False
                    break
                decls.extend(one)
            where = f" on `{node.name}`" if node.name else ""
            body = _smt_term(hold.expr, sorted(set(names))) if ok else None
            if body is None:
                why = (_blocked(model, hold.expr, facts)
                       or f"outside the Int/Bool fragment over "
                          f"{', '.join(sorted(set(names)))}")
                lines += ["", f"; not encoded{where}: `{hold.text}` -- {why}"]
                continue
            emitted += 1
            lines += ["", f"(push 1)   ; holds `{hold.text}`{where}"]
            lines += [f"  {d}" for d in decls]
            if node.when is not None and node.when.expr is not None:
                given = _smt_term(node.when.expr, sorted(set(names)))
                if given is not None:
                    lines.append(f"  (assert {given})   ; when "
                                 f"`{node.when.text}`")
            lines.append(f"  (assert (not {body}))")
            lines.append("(check-sat)")
            lines.append("(pop 1)")
    lines += ["", f"; {emitted} obligation(s) exported; the rest are either",
              "; settled by the interval prover above or outside this encoding.",
              "(exit)"]
    return "\n".join(lines) + "\n"
