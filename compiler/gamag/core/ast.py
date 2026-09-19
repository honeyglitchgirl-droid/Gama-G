"""The Gama-G v0.2 core language: its abstract syntax.

This is the language the audit report asked for in sections 16 to 18 -- a core
designed around Intent, Operation, Relationship, Constraint, Capability,
Effect, State, Recovery, Audit and the Execution Graph, rather than around
statements and functions.

The design inversion
--------------------
A conventional program is a *sequence of statements* that the compiler may
then analyse.  A Gama-G core program is a *set of operations* that declare what
they consume and what they produce; the compiler derives the execution order
from those declarations.  Order is an output of compilation, never an input.

That has consequences which are the point of the language:

* There is no assignment.  Every binding is produced by exactly one operation,
  so a binding has one definition and one lifetime.  `state` resources are the
  only things that change, and only through a declared `transition` under
  declared authority.
* There is no `return`.  An operation's value *is* the binding it yields.
* There are no unbounded loops.  `refine` repeats until a constraint holds
  `within` a mandatory bound; exceeding it is a classified fault.  A program
  that cannot be written as a hang cannot hang.
* There is no `if`/`else`.  Selection is several operations yielding the same
  binding under complementary `when` guards.  Because the alternatives are
  declarations rather than nested blocks, the compiler can -- and does -- check
  that they are mutually exclusive and exhaustive.
* There are no written edges.  The graph comes from matching `yields` to
  `uses`, so a dependency that is not real cannot be written down, and one that
  is real cannot be forgotten.

Expression *notation* (arithmetic, comparison, library calls) is retained
deliberately: those are mathematical symbols rather than another language's
control flow.  See docs/DESIGN_v0_2.md, which answers the audit report's ten
originality questions feature by feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .. import ast_nodes as A
from ..diagnostics import SourcePos


@dataclass
class Node:
    """Base of every core node.  Positions are carried into the elaborated
    v0.1 AST so diagnostics point at what the programmer actually wrote."""

    pos: Optional[SourcePos] = None


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
@dataclass
class IntentDecl(Node):
    """The program's contract with the world (audit report section 16)."""

    name: str = ""
    purpose: str = ""                 # free text, kept verbatim
    authority: List[str] = field(default_factory=list)
    trail: str = ""                   # what must be recorded, verbatim
    pos_name: Optional[SourcePos] = None


@dataclass
class SourceDecl(Node):
    """A binding supplied from outside the intent."""

    name: str = ""
    type: Optional[A.TypeRef] = None
    secret: bool = False
    # `source x : T from e` -- where the input comes from.  A source without
    # `from` is a parameter of the intent: the program is then a library, not a
    # runnable one.
    from_expr: Optional[A.Expr] = None


@dataclass
class Clause(Node):
    """One `keyword  value` line inside a declaration."""

    keyword: str = ""
    names: List[str] = field(default_factory=list)      # uses / needs / authority
    expr: Optional[A.Expr] = None                       # computes / holds / when
    text: str = ""                                      # trail / purpose
    binding: str = ""                                   # yields / alters name
    type: Optional[A.TypeRef] = None                    # yields / alters type
    secret: bool = False
    bound: Optional[int] = None                         # within <n>
    item: str = ""                                      # over <x> as <item>
    choices: List[Tuple[Any, Any]] = field(default_factory=list)  # choose


@dataclass
class OperationDecl(Node):
    """A node of the execution graph: consumes bindings, yields exactly one."""

    name: str = ""
    uses: List[str] = field(default_factory=list)
    yields: str = ""
    yields_type: Optional[A.TypeRef] = None
    yields_secret: bool = False
    effect: str = ""
    needs: List[str] = field(default_factory=list)
    holds: List[Clause] = field(default_factory=list)   # constraints
    when: Optional[Clause] = None                       # guard
    computes: Optional[Clause] = None
    trail: str = ""
    alters: str = ""                                    # transitions only
    # refine
    starts: Optional[Clause] = None
    repeats: Optional[Clause] = None
    until: Optional[Clause] = None
    within: Optional[int] = None
    # each
    over: Optional[Clause] = None
    # resolve
    choices: List[Tuple[A.Pattern, A.Expr]] = field(default_factory=list)
    kind: str = "operation"    # operation | refine | each | resolve | transition


@dataclass
class StateDecl(Node):
    """A resource that may change, under declared authority."""

    name: str = ""
    type: Optional[A.TypeRef] = None
    starts: Optional[Clause] = None
    authority: List[str] = field(default_factory=list)
    secret: bool = False


@dataclass
class OutcomeDecl(Node):
    """The binding the intent exists to produce."""

    binding: str = ""


@dataclass
class CoreModule(Node):
    """A whole core program."""

    version: str = ""
    intent: Optional[IntentDecl] = None
    sources: List[SourceDecl] = field(default_factory=list)
    states: List[StateDecl] = field(default_factory=list)
    operations: List[OperationDecl] = field(default_factory=list)
    outcome: Optional[OutcomeDecl] = None
    filename: str = "<core>"

    def producers(self) -> List[OperationDecl]:
        return list(self.operations)


# --------------------------------------------------------------------------
# the derived execution graph
# --------------------------------------------------------------------------
@dataclass
class GraphNode:
    """One node of the derived graph, with the relationships the compiler
    inferred from `uses` and `yields` -- never written by the programmer."""

    decl: OperationDecl
    produces: str = ""
    consumes: List[str] = field(default_factory=list)
    guarded: bool = False
    # filled in by graph.build
    level: int = 0
    upstream: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)


@dataclass
class Alternative:
    """Several guarded operations yielding one binding."""

    binding: str
    members: List[OperationDecl] = field(default_factory=list)
    proven_exhaustive: bool = False
    proven_exclusive: bool = False


@dataclass
class ExecutionGraph:
    """The derived graph: the program's actual meaning."""

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    levels: List[List[str]] = field(default_factory=list)
    alternatives: Dict[str, Alternative] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    outcome: str = ""
    # which binding each producer yields, including sources and states
    producers_of: Dict[str, List[str]] = field(default_factory=dict)

    def order(self) -> List[str]:
        """Operation names in derived execution order."""
        return [name for level in self.levels for name in level]

    def describe(self) -> str:
        """A human-readable rendering, for `ggc explain --graph`."""
        lines = []
        for index, level in enumerate(self.levels):
            lines.append(f"level {index}: " + ", ".join(level))
        for name, node in sorted(self.nodes.items()):
            upstream = ", ".join(node.upstream) or "(none)"
            lines.append(f"  {name} yields {node.produces} <- {upstream}")
        for binding, alt in sorted(self.alternatives.items()):
            flags = []
            if alt.proven_exclusive:
                flags.append("exclusive")
            if alt.proven_exhaustive:
                flags.append("exhaustive")
            lines.append(f"  selection {binding}: "
                         f"{len(alt.members)} alternatives"
                         + (f" [{', '.join(flags)}]" if flags else ""))
        return "\n".join(lines)
