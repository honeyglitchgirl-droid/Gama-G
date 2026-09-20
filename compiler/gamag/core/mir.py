"""The Gama-G native semantic IR -- v0.3 Priority 1.

The audit's finding on v0.2 was precise and fair: the *front end* was original,
but it elaborated into the older language's abstract syntax, so `if`, `while`,
`match` and `let` reappeared as the intermediate representation. Its words:
*"Gama-G's frontend is more original than its current implementation
foundation"*, and *"Do not first convert Gama-G into let, if, while, match,
conventional function calls. Instead, represent Gama-G's own semantic concepts
directly."*

This module is that representation. It is the model the rest of the compiler
consumes, and it contains no statement, no assignment, no jump and no block --
those are machine concepts that belong in GIR, which is where this model lowers
to.

The five graphs
---------------
The audit's recommended architecture names five graphs between source and GIR.
They are not five passes over one structure; they are five *views* of one
model, each answering a different question, and each is separately inspectable:

``IntentGraph``      what the program is for, and what it must record
``OperationGraph``   what produces what, and therefore what runs when
``ConstraintGraph``  every promise, and whether it is proven or only checked
``AuthorityGraph``   every capability demand, and whether the intent meets it
``RecoveryGraph``    every bound, and every fault the program can classify

Keeping them separate is what makes the model auditable: a tool can ask "show
me every constraint that is only checked at runtime" or "show me every node
whose authority the intent does not hold" without re-deriving anything.

What this module deliberately reuses
------------------------------------
:mod:`gamag.semantic.types` (the type lattice) and :mod:`gamag.std.library`
(builtin signatures). Neither is a language model -- one is the domain of
types, the other a table of what the standard library provides. Reusing them is
not elaborating through v0.1, and reimplementing them would produce two
competing definitions of what ``F64`` means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..diagnostics import SourcePos
from ..semantic import types as T

# ==========================================================================
# expressions
# ==========================================================================
# A core expression is a value description, not a statement.  There is no
# assignment expression, no call-with-side-effects-as-statement, and no
# conditional expression: selection lives in the OperationGraph, where the
# compiler can reason about it, rather than inside a value.


@dataclass
class MExpr:
    """Base of every core expression."""

    pos: Optional[SourcePos] = None
    type: T.Type = T.ANY            # filled in by the native checker
    text: str = ""                  # verbatim source, kept for diagnostics


@dataclass
class MLit(MExpr):
    """A literal value."""

    value: Any = None
    kind: str = "int"               # int | float | string | bool | unit


@dataclass
class MRef(MExpr):
    """A reference to a binding produced elsewhere in the intent.

    Not a variable read: the binding has exactly one producer, so this is a
    reference to a relationship rather than to a memory location.
    """

    binding: str = ""


@dataclass
class MBin(MExpr):
    op: str = ""
    left: Optional[MExpr] = None
    right: Optional[MExpr] = None


@dataclass
class MUn(MExpr):
    op: str = ""
    operand: Optional[MExpr] = None


@dataclass
class MCall(MExpr):
    """A library operation: `math.clamp(x, 0.0, 500.0)` or `abs(x)`."""

    module: str = ""
    name: str = ""
    args: List[MExpr] = field(default_factory=list)


@dataclass
class MField(MExpr):
    obj: Optional[MExpr] = None
    attr: str = ""


@dataclass
class MIndex(MExpr):
    obj: Optional[MExpr] = None
    index: Optional[MExpr] = None


@dataclass
class MItems(MExpr):
    """A collection literal: `[3, 5, 8]`."""

    items: List[MExpr] = field(default_factory=list)


# ==========================================================================
# patterns (only `resolve` has them)
# ==========================================================================
@dataclass
class MPattern:
    pos: Optional[SourcePos] = None
    text: str = ""


@dataclass
class PWild(MPattern):
    pass


@dataclass
class PLit(MPattern):
    value: Any = None
    kind: str = "int"


@dataclass
class PBind(MPattern):
    """Binds whatever it matches; used as the subject of a nested test."""

    binding: str = ""


@dataclass
class PTag(MPattern):
    tag: str = ""
    args: List[MPattern] = field(default_factory=list)


# ==========================================================================
# types
# ==========================================================================
@dataclass
class MType:
    """A type as written, resolved to :mod:`semantic.types` by the checker."""

    name: str = ""
    args: List["MType"] = field(default_factory=list)
    secret: bool = False
    pos: Optional[SourcePos] = None
    resolved: T.Type = T.ANY

    def render(self) -> str:
        inner = self.name
        if self.args:
            inner += "<" + ", ".join(a.render() for a in self.args) + ">"
        return ("secret " + inner) if self.secret else inner


# ==========================================================================
# constraints
# ==========================================================================
# A constraint is a promise with a *discharge status*.  Separating the two is
# what lets the compiler -- and a reader -- tell a proven property from a
# runtime check, which is the distinction the audit insisted on.

DISCHARGE_PROVEN = "proven"          # established at compile time
DISCHARGE_RUNTIME = "runtime"        # checked while running, faults if broken
DISCHARGE_UNPROVABLE = "unprovable"  # known not to be provable here


@dataclass
class Constraint:
    """One promise the program makes."""

    kind: str = "holds"              # holds | when | until | exclusive
    text: str = ""                   # verbatim, so a fault can quote it
    expr: Optional[MExpr] = None
    node: str = ""                   # the operation that carries it
    binding: str = ""                # the binding it constrains, if any
    discharge: str = DISCHARGE_RUNTIME
    fault: str = "ContractViolation"
    pos: Optional[SourcePos] = None
    #: How the status was obtained, in one line: what the prover established,
    #: or the specific reason it declined.  `Selection.proof` is the same idea;
    #: a status a reader cannot question is not worth recording.
    proof: str = ""


# ==========================================================================
# operation-graph nodes
# ==========================================================================
@dataclass
class OpNode:
    """Base of every node in the operation graph."""

    name: str = ""
    kind: str = "compute"            # source | state | compute | refine
                                     # | fanout | dispatch | transition
    produces: str = ""               # the binding this node yields
    type: Optional[MType] = None
    secret: bool = False
    consumes: List[str] = field(default_factory=list)
    #: The effects this operation declares, in the order written.
    #:
    #: Plural because the specification's effect algebra is a set and the standard
    #: library's own operations declare more than one: `secrets.expose` is `crypto`
    #: *and* `audit`, `medical.fhir_serialize` is `medical` *and* `io`, `model.load`
    #: is `model` *and* `storage`. A node that could name only one effect could not
    #: call any of them, which would make the capabilities they demand impossible
    #: to exercise however correctly they were checked.
    effects: List[str] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)
    trail: str = ""
    holds: List[Constraint] = field(default_factory=list)
    when: Optional[Constraint] = None
    pos: Optional[SourcePos] = None
    # filled in by the graph builder
    level: int = 0
    phase: str = "compute"           # compute | commit
    upstream: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)

    @property
    def effect(self) -> str:
        """The primary declared effect, for messages and single-effect contexts."""
        return self.effects[0] if self.effects else ""


@dataclass
class SourceNode(OpNode):
    """An input from outside the intent."""

    origin: Optional[MExpr] = None   # `from <expr>`; None makes it a parameter


@dataclass
class StateNode(OpNode):
    """The only kind of thing that may change."""

    initial: Optional[MExpr] = None
    authority: List[str] = field(default_factory=list)


@dataclass
class ComputeNode(OpNode):
    value: Optional[MExpr] = None


@dataclass
class RefinementNode(OpNode):
    """Bounded repetition.  `bound` is never None: the grammar requires it."""

    start: Optional[MExpr] = None
    step: Optional[MExpr] = None
    stop: Optional[Constraint] = None
    # None means the program did not say.  Defaulting this to a number would
    # silently make an unbounded refinement look bounded, which is the one thing
    # the construct exists to prevent.
    bound: Optional[int] = None


@dataclass
class FanOutNode(OpNode):
    """Traversal.  `produces` is a collection of `element`."""

    collection: Optional[MExpr] = None
    item: str = ""
    element: Optional[MType] = None
    value: Optional[MExpr] = None


@dataclass
class DispatchNode(OpNode):
    """Exhaustive dispatch over a declared relationship."""

    subject: Optional[MExpr] = None
    cases: List[Tuple[MPattern, MExpr]] = field(default_factory=list)


@dataclass
class TransitionNode(OpNode):
    """The only way a state changes."""

    state: str = ""
    value: Optional[MExpr] = None


@dataclass
class Selection:
    """Several guarded nodes yielding one binding."""

    binding: str = ""
    members: List[str] = field(default_factory=list)
    proven_exclusive: bool = False
    proven_exhaustive: bool = False
    fallback_fault: str = "NoActiveAlternative"
    #: how the proof was obtained, when it was: the syntactic complement or
    #: the guard prover's exact interval decision, in one line
    proof: str = ""


# ==========================================================================
# the five graphs
# ==========================================================================
@dataclass
class IntentGraph:
    """What the program is for, what authority it holds, what it must record."""

    name: str = ""
    purpose: str = ""
    authority: List[str] = field(default_factory=list)
    trail: str = ""
    outcome: str = ""
    #: what to do when the intent's work fails, in escalation order
    recovery: List[RecoveryStep] = field(default_factory=list)
    #: the checkpoints a `restore` or `replay` step is allowed to return to.
    #: Declaring them is what makes spec section 10's "never silently invent
    #: state during recovery" checkable rather than aspirational.
    checkpoints: List[str] = field(default_factory=list)
    pos: Optional[SourcePos] = None

    def render(self) -> List[str]:
        lines = [f"intent {self.name}"]
        if self.purpose:
            lines.append(f"  purpose    {self.purpose}")
        if self.authority:
            lines.append(f"  authority  {', '.join(self.authority)}")
        if self.trail:
            lines.append(f"  trail      {self.trail}")
        if self.checkpoints:
            lines.append(f"  checkpoint {', '.join(self.checkpoints)}")
        lines.append(f"  outcome    {self.outcome}")
        return lines


@dataclass
class OperationGraph:
    """What produces what, and therefore what runs when.

    This is the graph the audit calls the project's strongest originality
    feature.  It is derived, never written: `levels` is an output of
    compilation.
    """

    nodes: Dict[str, OpNode] = field(default_factory=dict)
    levels: List[List[str]] = field(default_factory=list)
    producers_of: Dict[str, List[str]] = field(default_factory=dict)
    # the inverse index, filled on first use by `readers_of`
    reader_index: Dict[str, List[str]] = field(default_factory=dict)
    selections: Dict[str, Selection] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    # The nodes that feed the graph but are not scheduled by it: inputs come
    # from outside the intent, and a state's initial value is its producer.
    inputs: Dict[str, SourceNode] = field(default_factory=dict)
    resources: Dict[str, StateNode] = field(default_factory=dict)
    outcome: str = ""

    def order(self) -> List[str]:
        return [name for level in self.levels for name in level]

    def compute_order(self) -> List[str]:
        return [n for n in self.order() if self.nodes[n].phase == "compute"]

    def commit_order(self) -> List[str]:
        return [n for n in self.order() if self.nodes[n].phase == "commit"]

    def readers_of(self, binding: str) -> List[str]:
        """The nodes that read a binding: the inverse of `producers_of`.

        Ownership is only a fact if the borrowers are known, and the memory model
        needs them to compute where a binding's life ends.
        """
        if not self.reader_index:
            for name in (self.order() or list(self.nodes)):
                node = self.nodes.get(name)
                if node is None:
                    continue
                for read in reads_of(node):
                    self.reader_index.setdefault(read, []).append(name)
            for names in self.reader_index.values():
                names.sort()
        return list(self.reader_index.get(binding, []))

    def edges(self) -> List[Tuple[str, str]]:
        """Every derived dependency, as (producer, consumer)."""
        return [(up, name) for name in self.order()
                for up in self.nodes[name].upstream]

    def render(self) -> List[str]:
        lines = ["derived execution order (never written in the source):"]
        for index, level in enumerate(self.levels):
            phase = ""
            if any(self.nodes[n].phase == "commit" for n in level):
                phase = "   [commit]"
            lines.append(f"  level {index}: {', '.join(level)}{phase}")
        return lines


@dataclass
class ConstraintGraph:
    """Every promise, grouped by whether it is proven or only checked."""

    constraints: List[Constraint] = field(default_factory=list)

    def add(self, constraint: Constraint) -> Constraint:
        self.constraints.append(constraint)
        return constraint

    def proven(self) -> List[Constraint]:
        return [c for c in self.constraints if c.discharge == DISCHARGE_PROVEN]

    def runtime(self) -> List[Constraint]:
        return [c for c in self.constraints if c.discharge == DISCHARGE_RUNTIME]

    def unprovable(self) -> List[Constraint]:
        return [c for c in self.constraints
                if c.discharge == DISCHARGE_UNPROVABLE]

    def render(self) -> List[str]:
        lines = []
        for label, group in (("proven at compile time", self.proven()),
                             ("checked at runtime", self.runtime()),
                             ("not provable here", self.unprovable())):
            if not group:
                continue
            lines.append(f"  {label}:")
            for c in group:
                where = f" on {c.node}" if c.node else ""
                why = f"  [{c.proof}]" if c.proof else ""
                lines.append(f"    {c.kind} `{c.text}`{where}{why}")
        return lines


@dataclass
class AuthorityDemand:
    """One node's claim on the intent's authority."""

    node: str = ""
    needs: List[str] = field(default_factory=list)
    granted: bool = True
    missing: List[str] = field(default_factory=list)


@dataclass
class DerivedDemand:
    """A capability demand derived from what a node does, not what it declares.

    The distinction is the point of priority 5.  A declared `needs` is a promise
    the program makes about itself; a derived demand is what the operations it
    actually calls require, taken from the standard library's own metadata.  A
    program can under-declare, and only the second list catches that.
    """

    node: str = ""
    origin: str = ""
    alternatives: List[str] = field(default_factory=list)
    satisfied: bool = False
    covered_by: str = ""


@dataclass
class AuthorityGraph:
    """Every capability demand, and whether the intent actually holds it."""

    held: List[str] = field(default_factory=list)
    demands: List[AuthorityDemand] = field(default_factory=list)
    derived: List[DerivedDemand] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)

    def unmet(self) -> List[AuthorityDemand]:
        return [d for d in self.demands if not d.granted]

    def unmet_derived(self) -> List[DerivedDemand]:
        return [d for d in self.derived if not d.satisfied]

    def render(self) -> List[str]:
        lines = [f"  held by the intent: {', '.join(self.held) or '(none)'}"]
        for demand in self.demands:
            if not demand.needs:
                continue
            mark = "ok" if demand.granted else "UNMET"
            lines.append(f"    {demand.node} needs "
                         f"{', '.join(demand.needs)} [{mark}]")
        for demand in self.derived:
            mark = "ok" if demand.satisfied else "UNMET"
            covered = (f", covered by `{demand.covered_by}`"
                       if demand.covered_by else "")
            lines.append(f"    {demand.node} demands "
                         f"{' or '.join(demand.alternatives)} "
                         f"because {demand.origin} [{mark}{covered}]")
        if self.unknown:
            lines.append("    not in the capability vocabulary: "
                         + ", ".join(self.unknown))
        return lines


@dataclass
class RecoveryStep:
    """One line of an intent's `recover` policy.

    Spec section 10's own shape: an action, optionally bounded, optionally aimed
    at something.  ``restore checkpoint baseline`` is action ``restore``, target
    ``checkpoint baseline``.  The vocabulary of actions is the runtime's, so
    there is one list of levels in the project and not two.
    """

    action: str = ""
    count: Optional[int] = None
    target: str = ""
    raw: str = ""
    pos: Optional[SourcePos] = None


@dataclass
class RecoveryObligation:
    """One bounded repetition, or one classified fault the program can raise."""

    node: str = ""
    kind: str = "refinement"         # refinement | selection | transition
    bound: Optional[int] = None
    fault: str = ""
    reason: str = ""


@dataclass
class RecoveryGraph:
    """Every bound, and every fault the program can classify.

    Bounded recovery is one of the assets the audit says to build on.  This
    graph is where "bounded" stops being a claim and becomes an enumerable
    fact: every entry has a number attached or a fault name.
    """

    obligations: List[RecoveryObligation] = field(default_factory=list)
    #: the intent's declared escalation policy, and the checkpoints it may use
    policy: List[RecoveryStep] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)

    def faults(self) -> List[str]:
        return sorted({o.fault for o in self.obligations if o.fault})

    def render(self) -> List[str]:
        lines = []
        for o in self.obligations:
            if o.kind == "refinement":
                lines.append(f"  {o.node}: refinement bounded at {o.bound} "
                             f"rounds, else {o.fault}")
            else:
                lines.append(f"  {o.node}: {o.fault} ({o.reason})")
        if self.policy:
            lines.append("  declared policy, in escalation order:")
            for step in self.policy:
                bound = f" within {step.count} rounds" if step.count else ""
                target = f" {step.target}" if step.target else ""
                lines.append(f"    {step.action}{bound}{target}")
        if self.checkpoints:
            lines.append("  checkpoints: " + ", ".join(self.checkpoints))
        return lines


@dataclass
class SemanticModel:
    """The whole native semantic model of one core program."""

    intent: IntentGraph = field(default_factory=IntentGraph)
    operations: OperationGraph = field(default_factory=OperationGraph)
    constraints: ConstraintGraph = field(default_factory=ConstraintGraph)
    authority: AuthorityGraph = field(default_factory=AuthorityGraph)
    recovery: RecoveryGraph = field(default_factory=RecoveryGraph)
    version: str = ""
    filename: str = "<core>"

    # -- the OperationGraph is the view tools already know ----------------
    @property
    def nodes(self) -> Dict[str, OpNode]:
        return self.operations.nodes

    @property
    def levels(self) -> List[List[str]]:
        return self.operations.levels

    def render(self) -> str:
        parts: List[str] = []
        parts += self.intent.render()
        if self.operations.sources:
            parts.append("  sources    "
                         + ", ".join(self.operations.sources))
        if self.operations.states:
            parts.append("  states     " + ", ".join(self.operations.states))
        parts.append("")
        parts += self.operations.render()
        for binding, sel in sorted(self.operations.selections.items()):
            proof = []
            if sel.proven_exclusive:
                proof.append("mutually exclusive")
            if sel.proven_exhaustive:
                proof.append("exhaustive")
            verdict = ("proven " + " and ".join(proof)) if proof else \
                "not provable statically; a NoActiveAlternative check runs"
            if sel.proof and proof:
                verdict += f" ({sel.proof})"
            parts.append("")
            parts.append(f"selection of `{binding}` from "
                         f"{', '.join(sel.members)}: {verdict}")
        constraint_lines = self.constraints.render()
        if constraint_lines:
            parts.append("")
            parts.append("constraints:")
            parts += constraint_lines
        authority_lines = self.authority.render()
        if authority_lines:
            parts.append("")
            parts.append("authority:")
            parts += authority_lines
        recovery_lines = self.recovery.render()
        if recovery_lines:
            parts.append("")
            parts.append("recovery:")
            parts += recovery_lines
        return "\n".join(parts)


# ==========================================================================
# expression utilities
# ==========================================================================
def references(expr: Optional[MExpr], out: Optional[Set[str]] = None
               ) -> Set[str]:
    """Every binding an expression refers to.

    This is what makes the relationship check possible: the compiler compares
    the set returned here against what the node declared in `uses`, in both
    directions.
    """
    found: Set[str] = out if out is not None else set()
    if expr is None:
        return found
    stack: List[MExpr] = [expr]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if isinstance(node, MRef):
            found.add(node.binding)
        for value in vars(node).values():
            if isinstance(value, MExpr):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(v for v in value if isinstance(v, MExpr))
    return found


def pattern_bindings(pattern: MPattern) -> Set[str]:
    if isinstance(pattern, PBind):
        return {pattern.binding}
    if isinstance(pattern, PTag):
        out: Set[str] = set()
        for arg in pattern.args:
            out |= pattern_bindings(arg)
        return out
    return set()


def reads_of(node: OpNode) -> List[str]:
    """Every binding a node reads, in a stable order.

    The declared relationships plus the references actually present.  The checker
    requires the two to agree, so for a checked model this is normally just
    ``node.consumes``; walking the expressions as well means an unchecked model
    still gives a usable answer, which is what the memory model needs in order to
    compute an extent before it can be trusted.
    """
    found: Set[str] = set(node.consumes)
    for expr in node_expressions(node):
        references(expr, found)
    return sorted(found)


def node_expressions(node: OpNode) -> List[Optional[MExpr]]:
    """Every expression a node owns, for checking and reference collection."""
    out: List[Optional[MExpr]] = []
    for name in ("value", "origin", "initial", "start", "step", "collection",
                 "subject"):
        out.append(getattr(node, name, None))
    for constraint in list(node.holds) + ([node.when] if node.when else []):
        out.append(constraint.expr)
    stop = getattr(node, "stop", None)
    if stop is not None:
        out.append(stop.expr)
    for _pattern, value in getattr(node, "cases", []) or []:
        out.append(value)
    return out
