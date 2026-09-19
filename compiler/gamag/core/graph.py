"""Relationship inference and validation: the core's semantic heart.

The programmer writes operations and the bindings they consume and produce.
Nothing else.  This module turns that into an execution graph and checks the
properties that make the graph trustworthy:

**Relationships must be real.**  Every name an operation actually refers to,
that names another binding in the intent, must be declared in `uses`.  A
dependency that exists in the expression but not in the declaration is an
error, and one that is declared but never used is a warning.  This is the check
that makes the derived order mean something: without it, `uses` would be
documentation that could drift from the code.

**The graph must be acyclic.**  A cycle means two operations each need the
other's result, which has no value.  The error names the cycle.

**One binding, one producer -- unless the alternatives are guarded.**  Several
operations may yield the same binding only if every one of them carries a
`when` guard.  Where the guards are syntactically complementary the compiler
records the selection as proven mutually exclusive and exhaustive; where it
cannot prove that, the elaborated code keeps a runtime check that faults with
"NoActiveAlternative" rather than silently producing nothing.

**Two phases.**  Compute operations produce bindings and are ordered purely by
data.  Transitions commit changes to `state` resources and run after the
compute phase.  A compute operation may not depend on a transition: that would
make the derived order depend on an effect rather than on data, which is the
one thing this language refuses to allow.

**Every traversal is bounded.**  `refine` requires `within`; there is no way to
write a repetition that has no limit.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .. import ast_nodes as A
from ..diagnostics import DiagnosticBag, Phase
from . import ast as C

COMPUTE_KINDS = ("operation", "refine", "each", "resolve")


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _is_complement(g1: str, g2: str) -> bool:
    """True when one guard is the syntactic negation of the other."""
    a, b = _norm(g1), _norm(g2)
    if not a or not b:
        return False
    return b in ("not(" + a + ")", "not" + a) or a in ("not(" + b + ")",
                                                       "not" + b)


def free_names(expr: Optional[A.Expr], out: Optional[Set[str]] = None
               ) -> Set[str]:
    """Every identifier an expression mentions.

    Deliberately shallow about meaning: names that turn out to be modules or
    builtins are filtered by the caller, which only cares about names that
    refer to bindings this intent actually declares.
    """
    found: Set[str] = out if out is not None else set()
    if expr is None:
        return found
    stack = [expr]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if isinstance(node, A.Name):
            found.add(node.id)
        for value in getattr(node, "__dict__", {}).values():
            if isinstance(value, A.Node):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(v for v in value if isinstance(v, A.Node))
    return found


class GraphBuilder:
    """Builds and validates the :class:`~gamag.core.ast.ExecutionGraph`."""

    def __init__(self, module: C.CoreModule, bag: DiagnosticBag):
        self.m = module
        self.bag = bag
        self.graph = C.ExecutionGraph()
        self.bindings: Dict[str, str] = {}      # binding -> producer kind
        self.transitions: List[C.OperationDecl] = []

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
    def build(self) -> C.ExecutionGraph:
        self._collect_bindings()
        self._validate_shapes()
        self._validate_relationships()
        if self.bag.ok:
            self._order()
        return self.graph

    # ------------------------------------------------------------------
    def _collect_bindings(self) -> None:
        g = self.graph
        for src in self.m.sources:
            if src.name in self.bindings:
                self.error(f"`{src.name}` is produced more than once", src.pos,
                           code="E-duplicate-binding")
            self.bindings[src.name] = "source"
            g.sources.append(src.name)
            g.producers_of.setdefault(src.name, [])

        for st in self.m.states:
            if st.name in self.bindings:
                self.error(f"`{st.name}` is produced more than once", st.pos,
                           code="E-duplicate-binding")
            if st.starts is None or st.starts.expr is None:
                self.error(f"state `{st.name}` has no `starts` value, so it "
                           f"would have no initial meaning", st.pos,
                           code="E-state-uninitialised")
            self.bindings[st.name] = "state"
            g.states.append(st.name)
            g.producers_of.setdefault(st.name, [])

        for op in self.m.operations:
            if op.kind == "transition":
                if not op.alters:
                    self.error(f"transition `{op.name}` does not say which "
                               f"state it alters", op.pos,
                               code="E-transition-target")
                elif op.alters not in self.bindings or \
                        self.bindings[op.alters] != "state":
                    self.error(
                        f"transition `{op.name}` alters `{op.alters}`, which is "
                        f"not a declared state", op.pos,
                        code="E-transition-target",
                        help_text="only `state` resources may change; bindings "
                                  "produced by operations are single-assignment")
                self.transitions.append(op)
                continue

            if not op.yields:
                self.error(f"{op.kind} `{op.name}` does not `yield` a binding",
                           op.pos, code="E-no-yields",
                           help_text="every compute operation produces exactly "
                                     "one binding; that is how the graph gets "
                                     "its edges")
                continue

            producers = g.producers_of.setdefault(op.yields, [])
            if op.yields in self.bindings and not producers:
                self.error(
                    f"`{op.yields}` is already produced by a "
                    f"{self.bindings[op.yields]} and cannot also be yielded by "
                    f"`{op.name}`", op.pos, code="E-duplicate-binding")
                continue
            self.bindings.setdefault(op.yields, op.kind)
            producers.append(op.name)
            g.nodes[op.name] = C.GraphNode(
                decl=op, produces=op.yields, consumes=list(op.uses),
                guarded=op.when is not None)

    # ------------------------------------------------------------------
    def _validate_shapes(self) -> None:
        for op in self.m.operations:
            kind = op.kind
            # `resolve` takes its value from `choose` and `refine` from
            # `repeats`; only a plain operation must say `computes`.
            if kind in COMPUTE_KINDS and op.computes is None and \
                    kind not in ("resolve", "refine"):
                self.error(f"{kind} `{op.name}` has no `computes` expression",
                           op.pos, code="E-no-computes")
            if kind == "refine":
                for clause_name in ("starts", "repeats", "until"):
                    if getattr(op, clause_name) is None:
                        self.error(
                            f"refine `{op.name}` is missing `{clause_name}`",
                            op.pos, code="E-refine-incomplete",
                            help_text="a refine needs `starts` (the first "
                                      "approximation), `repeats` (the next "
                                      "one), `until` (when to stop) and "
                                      "`within` (how long it may take)")
                if op.within is None:
                    self.error(
                        f"refine `{op.name}` has no `within` bound", op.pos,
                        code="E-unbounded-refinement",
                        help_text="an unbounded repetition cannot be written "
                                  "in Gama-G core; state how many rounds it "
                                  "may take, e.g. `within 50 rounds`")
                elif op.within < 1:
                    self.error(f"refine `{op.name}` must allow at least one "
                               f"round", op.pos, code="E-unbounded-refinement")
            if kind == "each":
                if op.over is None:
                    self.error(f"each `{op.name}` does not say what it is over",
                               op.pos, code="E-each-incomplete")
                elif not op.over.item:
                    self.error(
                        f"each `{op.name}` does not name its item", op.pos,
                        code="E-each-incomplete",
                        help_text="write `over readings as reading`")
            if kind == "resolve":
                if op.over is None:
                    self.error(f"resolve `{op.name}` does not say what it "
                               f"resolves", op.pos, code="E-resolve-incomplete")
                if not op.choices:
                    self.error(f"resolve `{op.name}` has no `choose` "
                               f"alternatives", op.pos,
                               code="E-resolve-incomplete")
            if kind == "transition" and op.computes is None:
                self.error(f"transition `{op.name}` has no `computes` "
                           f"expression", op.pos, code="E-no-computes")
            if op.effect and op.effect not in _EFFECTS:
                self.error(f"`{op.effect}` is not an effect", op.pos,
                           code="E-unknown-effect",
                           help_text="effects are: " + ", ".join(sorted(_EFFECTS)))

    # ------------------------------------------------------------------
    def _validate_relationships(self) -> None:
        """Declared relationships must match real ones."""
        known = set(self.bindings)
        for op in self.m.operations:
            local: Set[str] = set()
            if op.kind == "each" and op.over is not None:
                local.add(op.over.item)
            if op.kind == "resolve":
                for pattern, _value in op.choices:
                    local |= _pattern_names(pattern)

            referenced: Set[str] = set()
            for clause in [op.computes, op.when, op.starts, op.repeats,
                           op.until, op.over] + list(op.holds):
                if clause is None:
                    continue
                expr = clause.expr
                if op.kind == "each" and clause is op.over:
                    # the collection being traversed is external; the item is
                    # internal
                    referenced |= free_names(expr) - local
                else:
                    referenced |= free_names(expr)
            # `holds` may constrain the binding this operation produces
            own = {op.yields} if op.kind != "transition" else {op.alters}
            referenced -= local
            referenced -= own

            real = referenced & known
            declared = set(op.uses)
            # An `over` clause *is* the declaration of the relationship to the
            # collection being traversed or resolved, so it needs no `uses` too.
            if op.over is not None and op.over.expr is not None:
                declared |= free_names(op.over.expr) & known
            for name in sorted(real - declared):
                self.error(
                    f"`{op.name}` refers to `{name}` but does not declare it in "
                    f"`uses`", op.pos, code="E-undeclared-relationship",
                    help_text="the execution graph is derived from declared "
                              "relationships, so a real dependency must be "
                              "written down: add it to `uses`")
            for name in sorted(declared - real):
                if name in known:
                    self.warn(
                        f"`{op.name}` declares `{name}` in `uses` but never "
                        f"refers to it", op.pos, code="W-unused-relationship",
                        help_text="an edge that is not real costs a dependency "
                                  "and can create a cycle; remove it")
                else:
                    self.error(
                        f"`{op.name}` uses `{name}`, which nothing produces",
                        op.pos, code="E-unresolved-relationship",
                        help_text="every relationship must connect to a "
                                  "`source`, a `state` or another operation's "
                                  "`yields`")

        # the outcome must exist
        if self.m.outcome is None:
            self.error("the intent has no `outcome`, so it produces nothing",
                       self.m.pos, code="E-no-outcome")
        elif self.m.outcome.binding not in self.bindings:
            self.error(
                f"outcome `{self.m.outcome.binding}` is produced by nothing",
                self.m.outcome.pos, code="E-unresolved-relationship")
        else:
            self.graph.outcome = self.m.outcome.binding

        # compute phase must not depend on the commit phase
        for op in self.m.operations:
            if op.kind in COMPUTE_KINDS:
                for name in op.uses:
                    if any(t.alters == name for t in self.transitions):
                        self.error(
                            f"`{op.name}` uses `{name}`, which a transition "
                            f"alters; compute operations may only depend on "
                            f"data, never on a committed change", op.pos,
                            code="E-compute-after-commit")

    # ------------------------------------------------------------------
    def _order(self) -> None:
        g = self.graph
        # edges: producer operation -> consumer operation
        deps: Dict[str, Set[str]] = {name: set() for name in g.nodes}
        for name, node in g.nodes.items():
            for binding in node.consumes:
                for producer in g.producers_of.get(binding, []):
                    if producer != name and producer in g.nodes:
                        deps[name].add(producer)

        cycle = _find_cycle(deps)
        if cycle:
            first = g.nodes[cycle[0]].decl
            self.error(
                "the operation graph has a cycle: "
                + " -> ".join(cycle + [cycle[0]]), first.pos,
                code="E-graph-cycle",
                help_text="each operation in the cycle needs another's result, "
                          "so none of them has a value; break the cycle by "
                          "splitting an operation in two")
            return

        # Kahn by levels: everything whose dependencies are met runs together.
        done: Set[str] = set()
        remaining = set(g.nodes)
        while remaining:
            level = sorted(n for n in remaining if deps[n] <= done)
            if not level:                      # pragma: no cover - cycle caught
                break
            for name in level:
                g.nodes[name].level = len(g.levels)
                g.nodes[name].upstream = sorted(deps[name])
                done.add(name)
                remaining.discard(name)
            g.levels.append(level)
        for name, node in g.nodes.items():
            node.downstream = sorted(
                other for other, dep in deps.items() if name in dep)

        # selections: several guarded producers of one binding
        for binding, producers in g.producers_of.items():
            members = [g.nodes[p].decl for p in producers if p in g.nodes]
            if len(members) < 2:
                continue
            alt = C.Alternative(binding=binding, members=members)
            unguarded = [m for m in members if m.when is None]
            if unguarded:
                names = ", ".join(f"`{m.name}`" for m in unguarded)
                self.error(
                    f"`{binding}` is yielded by {len(members)} operations, but "
                    f"{names} "
                    f"{'has' if len(unguarded) == 1 else 'have'} no `when` "
                    f"guard, so the intent does not say which one applies",
                    unguarded[0].pos,
                    code="E-ambiguous-selection",
                    help_text="selection in Gama-G core is several guarded "
                              "operations yielding one binding; give every "
                              "alternative a `when`")
                continue
            if len(members) == 2:
                g1 = members[0].when.text
                g2 = members[1].when.text
                if _is_complement(g1, g2):
                    alt.proven_exclusive = True
                    alt.proven_exhaustive = True
            g.alternatives[binding] = alt

        # transitions are ordered after the compute phase, by their own data
        # dependencies and then by declaration order
        if self.transitions:
            commit: List[str] = []
            for index, tr in enumerate(self.transitions):
                commit.append(tr.name)
                g.nodes[tr.name] = C.GraphNode(
                    decl=tr, produces=tr.alters, consumes=list(tr.uses),
                    level=len(g.levels), upstream=sorted(
                        n for n in g.nodes if n in commit[:-1]
                        and g.nodes[n].produces == tr.alters))
            g.levels.append(commit)


# The effect algebra is the one the specification defines and the reference
# slice enforces: the core declares effects in the same vocabulary, so an
# effect written in the core means the same thing everywhere.
_EFFECTS = {"pure", "io", "network", "storage", "crypto", "audit",
            "medical", "model", "unsafe"}


def _pattern_names(pattern: A.Pattern) -> Set[str]:
    names: Set[str] = set()
    if isinstance(pattern, A.NamePat):
        names.add(pattern.name)
    elif isinstance(pattern, A.CtorPat):
        for arg in pattern.args or []:
            names |= _pattern_names(arg)
    return names


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


def build(module: C.CoreModule, bag: DiagnosticBag) -> C.ExecutionGraph:
    """Public entry point: derive and validate the execution graph."""
    return GraphBuilder(module, bag).build()
