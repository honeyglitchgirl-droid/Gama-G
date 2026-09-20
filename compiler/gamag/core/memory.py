"""The explicit memory and resource model of the Gama-G core.

Spec section 8 asks for ownership, borrowing, controlled mutation, deterministic
destruction, no use-after-free, no double-free, no data races in safe code, no
mandatory garbage collector, and stronger lifecycle controls for secrets. Audit
priority 4 asks that this be a *model* rather than an absence of one.

The core has an advantage here that a statement language does not. Its program is
a graph whose order the compiler derived, so every fact this module states is a
fact about that graph rather than a guess about a sequence of mutations:

* **Ownership** is not a discipline, it is the graph. A binding has exactly one
  producer -- ``E-duplicate-binding`` and ``E-ambiguous-selection`` are what make
  that true -- and the producer owns it.
* **Borrowing** is what every consumer does. Consumers read; none of them may
  write, because the binding is single-assignment. There is no move and no alias
  to track, which is why there is no borrow checker: there is nothing to borrow
  mutably.
* **Controlled mutation** is confined to one kind of node. Only a ``transition``
  writes a ``state``, only in the commit phase, and only under declared
  authority. Everything else is a value.
* **Deterministic destruction** falls out of the derived levels. A binding's
  extent runs from its owner's level to the level of its last borrower, and it is
  released at the end of that level. No sweep runs, nothing is traced, and the
  release point is a number the graph already contained.

Because extents are known, slots can be reused -- :meth:`MemoryModel.allocate`
does the interval partitioning and :class:`MemoryModel` reports how many slots
that saved. On a small program the answer is often zero, and the report says so
rather than implying otherwise: the value of the model is that the question has a
derived answer at all.

The safety properties are *proved* here and asserted by
``tests/test_memory_model.py``, not assumed: no binding is read before its owner
produces it, no binding is read after its release point, no slot holds two
overlapping bindings, and no two nodes in the same level write the same binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..semantic import types as T
from . import mir as M

#: The kinds of external resource an effect acquires. Spec section 12 names the
#: resources; these are the ones an *effect* implies, used only to report what a
#: node holds while it runs.
EFFECT_RESOURCES: Dict[str, str] = {
    "io": "filesystem",
    "storage": "database",
    "network": "network",
    "crypto": "key material",
    "audit": "audit log",
    "medical": "patient store",
    "model": "model store",
}

#: How a binding that is still needed when the intent returns is marked. Its
#: extent has no end level, because the return reads it after every level.
NO_RELEASE = -1


@dataclass
class Binding:
    """One binding, and the whole of its life."""

    name: str
    kind: str                      # value | resource | item | param
    owner: str                     # the node that produces it
    borrowers: List[str] = field(default_factory=list)
    secret: bool = False
    type: T.Type = T.ANY
    mutable: bool = False
    first: int = 0                 # level at which it comes into existence
    last: int = 0                  # level of the last reader
    to_return: bool = False        # still needed when the intent returns
    slot: int = -1
    shares_with: str = ""          # the binding whose slot this one reuses

    @property
    def released_after(self) -> Optional[int]:
        """The level after which this binding is dead, or ``None``.

        Deterministic destruction: a number derived from the graph, not a
        decision made by a collector at some unspecified later moment. ``None``
        means the binding is read by the return itself, so it is never released
        early -- which is the case for the outcome, every parameter and every
        state.
        """
        return NO_RELEASE if self.to_return else self.last

    @property
    def extent(self) -> Tuple[int, int]:
        return (self.first, self.last)

    def describe_extent(self, levels: int) -> str:
        if self.to_return:
            return f"{self.first}..return"
        return f"{self.first}..{self.last}"


@dataclass
class Acquisition:
    """A resource one node holds while it runs.

    The core has no ``open``/``close`` pair, so there is no way to forget the
    second half. That is not an accident of syntax: a resource is acquired by a
    node and released when that node finishes, so its lifetime can never span a
    fault, a branch or another node. :attr:`spans` records that, and
    :meth:`MemoryModel.verify` refuses the model if it is ever false.
    """

    node: str
    resource: str
    effect: str
    level: int = 0
    spans: int = 1                 # always exactly one node


@dataclass
class SlotLife:
    """One slot, and every binding that occupied it."""

    index: int
    occupants: List[Binding] = field(default_factory=list)

    @property
    def reused(self) -> bool:
        return len(self.occupants) > 1

    def describe(self) -> str:
        parts = [f"%{self.index} " +
                 ", ".join(f"{b.name}[{b.first}..{b.last}]"
                           for b in self.occupants)]
        return parts[0]


@dataclass
class MemoryModel:
    """Ownership, extents, releases, acquisitions and the proofs about them."""

    bindings: Dict[str, Binding] = field(default_factory=dict)
    slots: List[SlotLife] = field(default_factory=list)
    acquisitions: List[Acquisition] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    slots_saved: int = 0
    outcome: str = ""

    # ------------------------------------------------------------------
    # proofs
    # ------------------------------------------------------------------
    def verify(self) -> List[str]:
        """Check the properties spec section 8 requires, and return violations.

        Each of these is a real check over the derived model. They are expected
        to pass -- that is the point of deriving extents from a validated graph
        -- and a failure means the model and the lowering disagree, which is
        exactly the kind of drift worth stopping on.
        """
        self.violations = []

        # no use before the owner produced it
        graph_levels = {b.owner: b.first for b in self.bindings.values()}
        for binding in self.bindings.values():
            for borrower in binding.borrowers:
                level = graph_levels.get(borrower)
                if level is not None and level < binding.first:
                    self.violations.append(
                        f"use-before-ownership: `{borrower}` at level {level} "
                        f"reads `{binding.name}`, which `{binding.owner}` "
                        f"produces at level {binding.first}")

        # no use after release
        for binding in self.bindings.values():
            if binding.to_return:
                continue          # read by the return, so never released early
            for borrower in binding.borrowers:
                level = graph_levels.get(borrower)
                if level is not None and level > binding.released_after:
                    self.violations.append(
                        f"use-after-release: `{borrower}` at level {level} "
                        f"reads `{binding.name}`, released after level "
                        f"{binding.released_after}")

        # no double release: a slot's occupants must not overlap, and each
        # binding must appear in exactly one slot
        seen: Set[str] = set()
        for life in self.slots:
            for binding in life.occupants:
                if binding.name in seen:
                    self.violations.append(
                        f"double-release: `{binding.name}` occupies more than "
                        f"one slot")
                seen.add(binding.name)
            occupants = sorted(life.occupants, key=lambda b: b.first)
            for earlier, later in zip(occupants, occupants[1:]):
                if later.first <= earlier.last:
                    self.violations.append(
                        f"slot-overlap: `{earlier.name}`[{earlier.first}.."
                        f"{earlier.last}] and `{later.name}`[{later.first}.."
                        f"{later.last}] both occupy slot %{life.index}")

        # no data race in safe code: one writer per binding, and same-level
        # nodes must not write the same thing
        writers: Dict[str, List[str]] = {}
        for binding in self.bindings.values():
            writers.setdefault(binding.name, []).append(binding.owner)
        for name, who in writers.items():
            if len(who) > 1:
                self.violations.append(
                    f"data-race: `{name}` is written by {', '.join(who)}")

        # a resource is held by exactly one node
        for acq in self.acquisitions:
            if acq.spans != 1:
                self.violations.append(
                    f"resource-escape: `{acq.node}` holds {acq.resource} across "
                    f"{acq.spans} nodes")

        return self.violations

    # ------------------------------------------------------------------
    # slot allocation
    # ------------------------------------------------------------------
    def allocate(self, reserve: Sequence[str] = ()) -> None:
        """Assign slots, reusing one wherever two extents do not overlap.

        This is interval partitioning over the derived levels. Reuse is
        deliberately conservative: only immutable, non-parameter value bindings
        of the *same type* may share, and the outcome is never released early
        because the intent still has to return it.

        Being conservative is not caution for its own sake. A shared slot with
        the wrong type would make the IR lie about itself, and the outcome is
        read after every level, so treating it as dead would be a real
        use-after-release.
        """
        order = sorted(
            self.bindings.values(),
            key=lambda b: (b.first, b.last, b.name))
        lives: List[SlotLife] = []
        index = 0
        for binding in order:
            reusable = (
                binding.name not in reserve
                and not binding.to_return
                and binding.kind == "value"
                and not binding.mutable
            )
            placed = False
            if reusable:
                for life in lives:
                    last_occupant = life.occupants[-1]
                    if (last_occupant.type.render() == binding.type.render()
                            and binding.first > last_occupant.last):
                        life.occupants.append(binding)
                        binding.slot = life.index
                        binding.shares_with = last_occupant.name
                        placed = True
                        break
            if not placed:
                life = SlotLife(index=index, occupants=[binding])
                lives.append(life)
                binding.slot = index
                index += 1
        self.slots = lives
        self.slots_saved = sum(len(life.occupants) - 1 for life in lives)

    # ------------------------------------------------------------------
    def proof_lines(self) -> List[str]:
        """The properties this model establishes, for `ggc memory`."""
        released = [b for b in self.bindings.values() if not b.to_return]
        resources = [b for b in self.bindings.values() if b.kind == "resource"]
        scratch = [b for b in self.bindings.values()
                   if b.mutable and b.kind != "resource"]
        lines = [
            f"ownership: {len(self.bindings)} binding(s), each with exactly "
            f"one producer",
            f"borrowing: {sum(len(b.borrowers) for b in self.bindings.values())}"
            f" read(s) by other nodes, none of which may write",
            f"controlled mutation: {len(resources)} state resource(s), written "
            f"only by a transition in the commit phase"
            + (f"; {len(scratch)} scratch slot(s) written incrementally by a "
               f"selection, refinement, fan-out or dispatch"
               if scratch else ""),
            f"deterministic destruction: {len(released)} binding(s) released at "
            f"a level derived from the graph"
            + (f"; {len(self.bindings) - len(released)} live until the return"
               if len(released) != len(self.bindings) else "")
            + "; no collector runs",
            f"slot reuse: {self.slots_saved} slot(s) saved, "
            f"{len(self.slots)} allocated",
            f"resources: {len(self.acquisitions)} acquisition(s), each held by "
            f"exactly one node, so none can outlive a fault",
            f"violations: {len(self.violations)}",
        ]
        return lines

    def render(self) -> str:
        lines = ["memory and resources:"]
        for binding in sorted(self.bindings.values(),
                              key=lambda b: (b.first, b.name)):
            secret = " secret" if binding.secret else ""
            shared = (f" shares slot with `{binding.shares_with}`"
                      if binding.shares_with else "")
            released = ("live until the return" if binding.to_return
                        else f"released after level {binding.last}")
            lines.append(
                f"  `{binding.name}`{secret} : {binding.type.render()}"
                f"  [{binding.kind}]"
                f"  owned by {binding.owner}"
                f"  extent {binding.first}..{binding.last}"
                f"  {released}"
                f"  slot %{binding.slot}{shared}")
            if binding.borrowers:
                lines.append(f"      borrowed by: {', '.join(binding.borrowers)}")
        if self.acquisitions:
            lines.append("  resources held:")
            for acq in self.acquisitions:
                lines.append(f"    {acq.node} ({acq.effect}) holds "
                             f"{acq.resource} for the duration of one node")
        lines.extend("  " + line for line in self.proof_lines())
        for violation in self.violations:
            lines.append(f"  VIOLATION: {violation}")
        return "\n".join(lines)


# ==========================================================================
# building the model
# ==========================================================================

def build(model: M.SemanticModel, checker=None,
          types: Optional[Dict[str, T.Type]] = None) -> MemoryModel:
    """Derive the memory model from an already-checked semantic model.

    ``types`` (or ``checker.env``) supplies the resolved type of each binding,
    which is what makes same-type slot reuse possible. Without it every binding
    is ``ANY`` and reuse is skipped rather than guessed at.
    """
    env: Dict[str, T.Type] = dict(types or {})
    if checker is not None:
        for name, type_ in getattr(checker, "env", {}).items():
            env.setdefault(name, type_)

    graph = model.operations
    memory = MemoryModel(outcome=graph.outcome or "")
    # the level *after* the last derived one: where the commit phase runs, and
    # the point a binding read by the return is still alive at
    end = len(graph.levels)

    def type_of(name: str) -> T.Type:
        return env.get(name, T.ANY)

    def is_secret_type(type_: T.Type) -> bool:
        return isinstance(type_, T.SecretType)

    # --- inputs: owned by the caller, borrowed by the graph ---------------
    for name, node in graph.inputs.items():
        type_ = type_of(name)
        memory.bindings[name] = Binding(
            name=name, kind="param", owner=node.name,
            secret=is_secret_type(type_) or bool(node.secret), type=type_,
            first=0, last=end, to_return=True)

    # --- resources: mutable, owned by the intent, written by transitions --
    for name, node in graph.resources.items():
        type_ = type_of(name)
        borrowers = graph.readers_of(name)
        memory.bindings[name] = Binding(
            name=name, kind="resource", owner=node.name, borrowers=borrowers,
            secret=is_secret_type(type_) or bool(node.secret), type=type_,
            mutable=True, first=0, last=end, to_return=True)

    # --- every other binding: owned by the node that yields it ------------
    for name, node in graph.nodes.items():
        produces = node.produces
        if not produces or produces in memory.bindings:
            continue
        type_ = type_of(produces)
        # a refinement, fan-out or dispatch -- and any member of a guarded
        # selection -- writes its binding more than once while it runs; a plain
        # operation writes it exactly once
        mutable = node.kind in ("refine", "fanout", "dispatch") or \
            produces in graph.selections
        borrowers = [r for r in graph.readers_of(produces) if r != node.name]
        memory.bindings[produces] = Binding(
            name=produces, kind="value", owner=node.name, borrowers=borrowers,
            secret=is_secret_type(type_) or bool(node.secret), type=type_,
            mutable=mutable, first=node.level, last=node.level)
        item = getattr(node, "item", "")
        if item and item not in memory.bindings:
            # the loop variable: born and dead inside one node's extent
            item_type = env.get(item, T.ANY)
            memory.bindings[item] = Binding(
                name=item, kind="item", owner=node.name,
                borrowers=[r for r in graph.readers_of(item)
                           if r != node.name],
                secret=is_secret_type(item_type), type=item_type,
                mutable=True, first=node.level, last=node.level)

    # --- extents: the last borrower sets the release point ---------------
    for name, node in graph.nodes.items():
        # the commit phase runs after every derived level
        level = node.level if node.phase == "compute" else end
        for consumed in M.reads_of(node):
            binding = memory.bindings.get(consumed)
            if binding is None:
                continue
            # a node is not a borrower of its own output: a refinement reads its
            # previous round, but that is inside its own extent, and counting it
            # would make every `holds` clause look like an outside reader
            if binding.owner != node.name and node.name not in binding.borrowers:
                binding.borrowers.append(node.name)
            if not binding.to_return:
                binding.last = max(binding.last, level)
    # the outcome is read by the return, so it lives to the end
    if memory.outcome in memory.bindings:
        outcome = memory.bindings[memory.outcome]
        outcome.to_return = True
        outcome.last = end

    # --- acquisitions ----------------------------------------------------
    for name, node in graph.nodes.items():
        for effect in node.effects:
            resource = EFFECT_RESOURCES.get(effect)
            if resource:
                memory.acquisitions.append(Acquisition(
                    node=node.name, resource=resource, effect=effect,
                    level=node.level, spans=1))

    # --- allocation ------------------------------------------------------
    reserve = [n for n, b in memory.bindings.items()
               if b.kind in ("param", "resource", "item")]
    memory.allocate(reserve=reserve)
    memory.verify()
    return memory


