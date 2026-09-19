"""Deriving the semantic model: the core's structural heart.

The programmer writes operations and the bindings they consume and produce.
Nothing else.  This module turns that into the five graphs of
:mod:`gamag.core.mir` and checks the properties that make them trustworthy.

**Relationships must be real.**  Every binding an operation actually refers to,
that this intent produces, must be declared in `uses`; and everything declared
must be real.  This is the check that makes derived order mean anything --
without it `uses` would be a comment that could drift from the code.

**The graph must be acyclic.**  A cycle means two operations each need the
other's result, which has no value. The error names the cycle.

**One binding, one producer -- unless every alternative is guarded.**  Where
the guards are syntactically complementary the model records the selection as
proven mutually exclusive and exhaustive. Where it cannot prove that, it says
so explicitly (`DISCHARGE_UNPROVABLE`) and the lowering keeps a classified
`NoActiveAlternative` fault rather than silently producing nothing.

**Two phases.**  Compute nodes produce bindings and are ordered purely by data.
Transitions commit changes to `state` resources afterwards. A compute node may
not depend on a transition: that would make derived order depend on an effect
rather than on data, which is the one thing this language refuses.

**Every repetition is bounded.**  `refine` requires `within`, so the
RecoveryGraph can enumerate a number for every loop the program contains.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from ..diagnostics import DiagnosticBag, Phase
from . import mir as M
from .parser import CoreSyntax

COMPUTE_KINDS = ("compute", "refine", "fanout", "dispatch")

# The effect algebra is the specification's own, and the audit lists the effect
# system among the assets worth building on rather than redesigning.
EFFECTS = frozenset({"pure", "io", "network", "storage", "crypto", "audit",
                     "medical", "model", "unsafe"})


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def is_complement(g1: str, g2: str) -> bool:
    """True when one guard is the syntactic negation of the other."""
    a, b = _norm(g1), _norm(g2)
    if not a or not b:
        return False
    return b in ("not(" + a + ")", "not" + a) or \
        a in ("not(" + b + ")", "not" + b)


class ModelBuilder:
    def __init__(self, syntax: CoreSyntax, bag: DiagnosticBag):
        self.s = syntax
        self.bag = bag
        self.model = M.SemanticModel(version=syntax.version,
                                     filename=syntax.filename)
        self.produced_by: Dict[str, str] = {}   # binding -> what produces it

    # ------------------------------------------------------------------
    def error(self, message: str, pos=None, code: str = "E-graph",
              help_text: Optional[str] = None) -> None:
        self.bag.error(message, pos, phase=Phase.TYPE, code=code,
                       help_text=help_text)

    def warn(self, message: str, pos=None, code: str = "W-graph",
             help_text: Optional[str] = None) -> None:
        self.bag.warning(message, pos, phase=Phase.TYPE, code=code,
                         help_text=help_text)

    # ------------------------------------------------------------------
    def build(self) -> M.SemanticModel:
        self.model.intent = self.s.intent
        self.model.intent.outcome = self.s.outcome
        self._collect()
        self._shapes()
        self._relationships()
        self._authority()
        if self.bag.ok:
            self._order()
        self._constraints()
        self._recovery()
        return self.model

    # ------------------------------------------------------------------
    def _collect(self) -> None:
        graph = self.model.operations
        for node in self.s.nodes:
            if node.kind == "source":
                if node.produces in self.produced_by:
                    self.error(f"`{node.produces}` is produced more than once",
                               node.pos, code="E-duplicate-binding")
                    continue
                self.produced_by[node.produces] = "source"
                graph.inputs[node.produces] = node      # type: ignore[index]
                graph.sources.append(node.produces)
                graph.producers_of.setdefault(node.produces, [])
            elif node.kind == "state":
                if node.produces in self.produced_by:
                    self.error(f"`{node.produces}` is produced more than once",
                               node.pos, code="E-duplicate-binding")
                    continue
                if node.initial is None:                # type: ignore[attr-defined]
                    self.error(
                        f"state `{node.produces}` has no `starts` value, so it "
                        f"would have no initial meaning", node.pos,
                        code="E-state-uninitialised")
                self.produced_by[node.produces] = "state"
                graph.resources[node.produces] = node   # type: ignore[index]
                graph.states.append(node.produces)
                graph.producers_of.setdefault(node.produces, [])
            elif node.kind == "transition":
                state = node.state                      # type: ignore[attr-defined]
                if not state:
                    self.error(f"transition `{node.name}` does not say which "
                               f"state it alters", node.pos,
                               code="E-transition-target")
                elif self.produced_by.get(state) != "state":
                    self.error(
                        f"transition `{node.name}` alters `{state}`, which is "
                        f"not a declared state", node.pos,
                        code="E-transition-target",
                        help_text="only `state` resources may change; bindings "
                                  "produced by operations are single-assignment")
                graph.nodes[node.name] = node
            else:
                if not node.produces:
                    self.error(
                        f"{node.kind} `{node.name}` does not `yield` a binding",
                        node.pos, code="E-no-yields",
                        help_text="every compute operation produces exactly one "
                                  "binding; that is how the graph gets its "
                                  "edges")
                    continue
                producers = graph.producers_of.setdefault(node.produces, [])
                if node.produces in self.produced_by and not producers:
                    self.error(
                        f"`{node.produces}` is already produced by a "
                        f"{self.produced_by[node.produces]} and cannot also be "
                        f"yielded by `{node.name}`", node.pos,
                        code="E-duplicate-binding")
                    continue
                self.produced_by.setdefault(node.produces, node.kind)
                producers.append(node.name)
                graph.nodes[node.name] = node

    # ------------------------------------------------------------------
    def _shapes(self) -> None:
        for node in self.s.nodes:
            kind = node.kind
            if kind == "compute" and node.value is None:  # type: ignore[attr-defined]
                self.error(f"operation `{node.name}` has no `computes` "
                           f"expression", node.pos, code="E-no-computes")
            if kind == "refine":
                # the model's field names, and the clause the programmer writes
                parts = (("start", "`starts`"), ("step", "`repeats`"),
                         ("stop", "`until`"))
                for field_name, clause_name in parts:
                    if getattr(node, field_name) is None:
                        self.error(
                            f"refine `{node.name}` is missing {clause_name}",
                            node.pos, code="E-refine-incomplete",
                            help_text="a refine needs `starts` (the first "
                                      "approximation), `repeats` (the next "
                                      "one), `until` (when to stop) and "
                                      "`within` (how long it may take)")
                if node.bound is None:                  # type: ignore[attr-defined]
                    self.error(
                        f"refine `{node.name}` has no `within` bound", node.pos,
                        code="E-unbounded-refinement",
                        help_text="an unbounded repetition cannot be written in "
                                  "Gama-G core; state how many rounds it may "
                                  "take, e.g. `within 50 rounds`")
                elif node.bound < 1:                    # type: ignore[attr-defined]
                    self.error(f"refine `{node.name}` must allow at least one "
                               f"round", node.pos,
                               code="E-unbounded-refinement")
            if kind == "fanout":
                if node.collection is None:             # type: ignore[attr-defined]
                    self.error(f"each `{node.name}` does not say what it is "
                               f"over", node.pos, code="E-each-incomplete")
                elif not node.item:                     # type: ignore[attr-defined]
                    self.error(f"each `{node.name}` does not name its item",
                               node.pos, code="E-each-incomplete",
                               help_text="write `over readings as reading`")
                if node.value is None:                  # type: ignore[attr-defined]
                    self.error(f"each `{node.name}` has no `computes` "
                               f"expression", node.pos, code="E-no-computes")
            if kind == "dispatch":
                if node.subject is None:                # type: ignore[attr-defined]
                    self.error(f"resolve `{node.name}` does not say what it "
                               f"resolves", node.pos,
                               code="E-resolve-incomplete")
                if not node.cases:                      # type: ignore[attr-defined]
                    self.error(f"resolve `{node.name}` has no `choose` "
                               f"alternatives", node.pos,
                               code="E-resolve-incomplete")
                elif not any(isinstance(p, (M.PWild, M.PBind))
                             for p, _ in node.cases):    # type: ignore[attr-defined]
                    self.error(
                        f"resolve `{node.name}` has no catch-all alternative, "
                        f"so some values would resolve to nothing", node.pos,
                        code="E-dispatch-not-exhaustive",
                        help_text="end the `choose` block with `_ => ...`")
            if kind == "transition" and node.value is None:  # type: ignore[attr-defined]
                self.error(f"transition `{node.name}` has no `computes` "
                           f"expression", node.pos, code="E-no-computes")
            if node.effect and node.effect not in EFFECTS:
                self.error(f"`{node.effect}` is not an effect", node.pos,
                           code="E-unknown-effect",
                           help_text="effects are: " + ", ".join(sorted(EFFECTS)))

    # ------------------------------------------------------------------
    def _relationships(self) -> None:
        """Declared relationships must match real ones, in both directions."""
        known = set(self.produced_by)
        for node in self.s.nodes:
            if node.kind in ("source", "state"):
                continue
            local: Set[str] = set()
            if node.kind == "fanout":
                local.add(node.item)                    # type: ignore[attr-defined]
            if node.kind == "dispatch":
                for pattern, _value in node.cases:      # type: ignore[attr-defined]
                    local |= M.pattern_bindings(pattern)

            referenced: Set[str] = set()
            for expr in M.node_expressions(node):
                referenced |= M.references(expr)
            referenced -= local
            # a node may constrain the binding it produces
            referenced.discard(node.produces)
            if node.kind == "transition":
                referenced.discard(node.state)          # type: ignore[attr-defined]

            real = referenced & known
            declared = set(node.consumes)
            # an `over` clause *is* the declaration of the relationship to the
            # collection being traversed or resolved
            subject = getattr(node, "collection", None) or \
                getattr(node, "subject", None)
            if subject is not None:
                declared |= M.references(subject) & known

            for name in sorted(real - declared):
                self.error(
                    f"`{node.name}` refers to `{name}` but does not declare it "
                    f"in `uses`", node.pos, code="E-undeclared-relationship",
                    help_text="the execution graph is derived from declared "
                              "relationships, so a real dependency must be "
                              "written down: add it to `uses`")
            for name in sorted(declared - real):
                if name in known:
                    self.warn(
                        f"`{node.name}` declares `{name}` in `uses` but never "
                        f"refers to it", node.pos,
                        code="W-unused-relationship",
                        help_text="an edge that is not real costs a dependency "
                                  "and can create a cycle; remove it")
                else:
                    self.error(
                        f"`{node.name}` uses `{name}`, which nothing produces",
                        node.pos, code="E-unresolved-relationship",
                        help_text="every relationship must connect to a "
                                  "`source`, a `state` or another operation's "
                                  "`yields`")

        if not self.s.outcome:
            self.error("the intent has no `outcome`, so it produces nothing",
                       self.s.intent.pos, code="E-no-outcome")
        elif self.s.outcome not in self.produced_by:
            self.error(f"outcome `{self.s.outcome}` is produced by nothing",
                       self.s.outcome_pos, code="E-unresolved-relationship")
        else:
            self.model.operations.outcome = self.s.outcome

        altered = {n.state for n in self.s.nodes        # type: ignore[attr-defined]
                   if n.kind == "transition"}
        for node in self.s.nodes:
            if node.kind in COMPUTE_KINDS:
                for name in node.consumes:
                    if name in altered:
                        self.error(
                            f"`{node.name}` uses `{name}`, which a transition "
                            f"alters; compute operations may only depend on "
                            f"data, never on a committed change", node.pos,
                            code="E-compute-after-commit")

    # ------------------------------------------------------------------
    def _authority(self) -> None:
        """Every capability demand must be met by the intent's authority."""
        held = set(self.model.intent.authority)
        graph = self.model.authority
        graph.held = list(self.model.intent.authority)
        for node in self.s.nodes:
            if not node.needs:
                continue
            missing = sorted(set(node.needs) - held)
            demand = M.AuthorityDemand(node=node.name, needs=list(node.needs),
                                       granted=not missing, missing=missing)
            graph.demands.append(demand)
            if missing:
                self.error(
                    f"`{node.name}` needs {'`' + '`, `'.join(missing) + '`'} "
                    f"but the intent holds "
                    f"{'`' + '`, `'.join(sorted(held)) + '`' if held else 'no authority'}",
                    node.pos, code="E-authority-unmet",
                    help_text="an operation may not require authority its "
                              "intent does not have; add it to the intent's "
                              "`authority` or remove the demand")
        for name, state in self.model.operations.resources.items():
            missing = sorted(set(state.authority) - held)
            if missing:
                self.error(
                    f"state `{name}` is guarded by "
                    f"{'`' + '`, `'.join(missing) + '`'}, which the intent does "
                    f"not hold", state.pos, code="E-authority-unmet")

    # ------------------------------------------------------------------
    def _order(self) -> None:
        graph = self.model.operations
        # Transitions are not scheduled by data: they form the commit phase.
        # Including them in the Kahn walk would place a transition twice -- once
        # where its inputs are ready and once in the commit phase.
        compute = {name: node for name, node in graph.nodes.items()
                   if node.phase == "compute"}
        deps: Dict[str, Set[str]] = {name: set() for name in compute}
        for name, node in compute.items():
            for binding in node.consumes:
                for producer in graph.producers_of.get(binding, []):
                    if producer != name and producer in compute:
                        deps[name].add(producer)

        cycle = _find_cycle(deps)
        if cycle:
            self.error(
                "the operation graph has a cycle: "
                + " -> ".join(cycle + [cycle[0]]),
                graph.nodes[cycle[0]].pos, code="E-graph-cycle",
                help_text="each operation in the cycle needs another's result, "
                          "so none of them has a value; break the cycle by "
                          "splitting an operation in two")
            return

        done: Set[str] = set()
        remaining = set(compute)
        while remaining:
            level = sorted(n for n in remaining if deps[n] <= done)
            if not level:                       # pragma: no cover - cycle caught
                break
            for name in level:
                compute[name].level = len(graph.levels)
                compute[name].upstream = sorted(deps[name])
                done.add(name)
                remaining.discard(name)
            graph.levels.append(level)
        for name, node in compute.items():
            node.downstream = sorted(o for o, d in deps.items() if name in d)

        # selections: several guarded producers of one binding
        for binding, producers in graph.producers_of.items():
            members = [graph.nodes[p] for p in producers if p in graph.nodes]
            if len(members) < 2:
                continue
            unguarded = [m for m in members if m.when is None]
            if unguarded:
                names = ", ".join(f"`{m.name}`" for m in unguarded)
                self.error(
                    f"`{binding}` is yielded by {len(members)} operations, but "
                    f"{names} {'has' if len(unguarded) == 1 else 'have'} no "
                    f"`when` guard, so the intent does not say which one "
                    f"applies", unguarded[0].pos, code="E-ambiguous-selection",
                    help_text="selection in Gama-G core is several guarded "
                              "operations yielding one binding; give every "
                              "alternative a `when`")
                continue
            selection = M.Selection(binding=binding,
                                    members=[m.name for m in members])
            if len(members) == 2 and is_complement(members[0].when.text,
                                                   members[1].when.text):
                selection.proven_exclusive = True
                selection.proven_exhaustive = True
            graph.selections[binding] = selection

        # transitions form the commit phase, after every derivation of data
        commit = [n for n in self.s.nodes if n.kind == "transition"]
        if commit:
            names = [n.name for n in commit]
            for index, node in enumerate(commit):
                node.level = len(graph.levels)
                # a transition reads data produced in the compute phase, and
                # any earlier transition that altered the same state
                data = sorted(p for binding in node.consumes
                              for p in graph.producers_of.get(binding, [])
                              if p in compute)
                earlier = [n for n in names[:index]
                           if graph.nodes[n].state == node.state]  # type: ignore[attr-defined]
                node.upstream = sorted(set(data) | set(earlier))
            graph.levels.append(names)

    # ------------------------------------------------------------------
    def _constraints(self) -> None:
        """Every promise, with its discharge status recorded honestly."""
        constraints = self.model.constraints
        for node in self.s.nodes:
            for hold in node.holds:
                constraints.add(hold)
            stop = getattr(node, "stop", None)
            if stop is not None:
                stop.fault = "RefinementDiverged"
                constraints.add(stop)
        for binding, selection in self.model.operations.selections.items():
            for member in selection.members:
                guard = self.model.operations.nodes[member].when
                if guard is not None:
                    constraints.add(guard)
            # the selection property itself is a constraint on the graph
            constraints.add(M.Constraint(
                kind="exclusive",
                text=f"exactly one of {', '.join(selection.members)} yields "
                     f"`{binding}`",
                node=", ".join(selection.members), binding=binding,
                discharge=(M.DISCHARGE_PROVEN
                           if selection.proven_exhaustive
                           and selection.proven_exclusive
                           else M.DISCHARGE_UNPROVABLE),
                fault=selection.fallback_fault,
                pos=self.model.operations.nodes[selection.members[0]].pos))

    # ------------------------------------------------------------------
    def _recovery(self) -> None:
        """Every bound, and every fault this program can classify."""
        recovery = self.model.recovery
        for node in self.s.nodes:
            if node.kind == "refine":
                recovery.obligations.append(M.RecoveryObligation(
                    node=node.name, kind="refinement", bound=node.bound,  # type: ignore[attr-defined]
                    fault="RefinementDiverged",
                    reason=(node.stop.text if node.stop else "")))  # type: ignore[attr-defined]
            for _hold in node.holds:
                recovery.obligations.append(M.RecoveryObligation(
                    node=node.name, kind="constraint",
                    fault="ContractViolation", reason=_hold.text))
        for binding, selection in self.model.operations.selections.items():
            if not (selection.proven_exhaustive and selection.proven_exclusive):
                recovery.obligations.append(M.RecoveryObligation(
                    node=", ".join(selection.members), kind="selection",
                    fault=selection.fallback_fault,
                    reason=f"{len(selection.members)} guards over `{binding}` "
                           f"were not proven complementary"))

    # ------------------------------------------------------------------
    # the old names, kept as views so existing tools and tests keep working
    @property
    def alternatives(self):
        return self.model.operations.selections


def _find_cycle(deps: Dict[str, Set[str]]) -> Optional[List[str]]:
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {n: WHITE for n in deps}
    path: List[str] = []

    def visit(node: str) -> Optional[List[str]]:
        colour[node] = GREY
        path.append(node)
        for nxt in sorted(deps[node]):
            if nxt not in colour:
                continue
            if colour[nxt] == GREY:
                return path[path.index(nxt):]
            if colour[nxt] == WHITE:
                found = visit(nxt)
                if found:
                    return found
        path.pop()
        colour[node] = BLACK
        return None

    for node in sorted(deps):
        if colour[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None


def build(syntax: CoreSyntax, bag: DiagnosticBag) -> M.SemanticModel:
    """Public entry point: derive and validate the semantic model."""
    return ModelBuilder(syntax, bag).build()
