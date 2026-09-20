"""Formal capability semantics for the Gama-G core.

Spec section 12 makes capabilities the security primitive, and section 8 gives
secrets "stronger lifecycle controls". Audit priority 5 asks that the semantics
be *formal* rather than a set-membership test, and it is right that v0.3's was
one: `needs` was compared to `authority` with `set.issubset`, which cannot say
what a capability *is*, when one covers another, or where the boundary of a
secret actually falls.

This module gives those three answers.

**What a capability is.** A pair of a resource and a permission over it
(:class:`Capability`). `PatientRead` is not an opaque token; it is the
permission `Read` over the resource `Patient`. That decomposition is what makes
the rest possible, because now two capabilities can be compared.

**When one covers another.** :meth:`Capability.grants`, driven by the explicit
implication table :data:`IMPLIES`. Coverage is a relation, not equality, so an
intent holding `PatientWrite` satisfies an operation demanding `PatientRead`,
and a whole-resource capability satisfies any permission on that resource.
Attenuation is the same relation read the other way: a program may narrow what
it holds and may never widen it, because nothing in the model can introduce a
capability the intent does not have. The intent's `authority` is a ceiling and
there is no operation that raises it.

**Where the boundary of a secret falls.** :data:`SECRET_BOUNDARIES` names the
ordinary renderers -- `print`, `to_text`, `str` -- through which spec section 12
says a secret must not pass. Reaching one with a secret value demands
`SecretExpose`. This is the rule that makes the secret lifecycle *explicit*
rather than a convention: the boundary is a named set of library operations, the
demand is derived, and the denial is a diagnostic.

Demands are derived from what a node actually calls, using the standard
library's own `caps` metadata, rather than from the node's declared `effect`.
That is deliberate. An effect is coarser than a capability -- `crypto` covers
hashing and fingerprinting as well as signing, and only signing has a capability
in the vocabulary -- so inferring `CryptoSign` from `effect crypto` would demand
authority from programs that never sign anything. Deriving from the calls made
is both more precise and more honest: it is the library's own statement of what
it needs, not a guess about what an effect usually means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..std import library as L
from . import mir as M

# ==========================================================================
# the algebra
# ==========================================================================

# The algebra itself -- :class:`Capability`, the permission implication table and
# the vocabulary -- lives in :mod:`gamag.capabilities`, because the runtime has to
# use the same relation the compiler does. What follows is the part that is
# specific to the core: how demands are derived from a node, and the boundary at
# which a secret leaves its lifecycle.
from ..capabilities import (Capability, IMPLIES, KNOWN_CAPABILITIES,  # noqa: F401
                            PERMISSIONS, covers, holder, known)  # noqa: F401

def held_names(policy: "CapabilityPolicy") -> List[str]:
    """The names a policy holds, for use in a diagnostic."""
    return [cap.name for cap in policy.held]


#: Library operations through which a secret leaves its lifecycle.
#:
#: Spec section 12: "secrets cannot be printed through ordinary logging", and
#: section 8: secrets have "restrictions on logging, serialization, and
#: accidental conversion to ordinary Text". These are the ordinary renderers.
#: Reaching one with a secret value demands ``SecretExpose`` -- the capability
#: the standard library's own ``secrets.expose`` declares.
SECRET_BOUNDARIES = frozenset({
    "print", "println", "print_raw", "eprint",
    "to_text", "str", "text.format", "text.join",
    "json.encode", "json.serialize", "bytes.from_text",
})

#: The capability a secret crossing :data:`SECRET_BOUNDARIES` demands.
SECRET_EXPOSE = "SecretExpose"


# ==========================================================================
# demands
# ==========================================================================

@dataclass
class Demand:
    """One capability an operation must be holding, and why.

    ``alternatives`` is a disjunction: satisfying any one of them satisfies the
    demand. It exists so that a demand can say "read *or* write access to this
    store" without pretending to know which the program needed.
    """

    node: str
    origin: str
    alternatives: Tuple[Capability, ...]
    pos: object = None
    secret: Optional[str] = None

    def satisfied_by(self, held: Sequence[Capability]) -> bool:
        return any(any(h.grants(a) for h in held) for a in self.alternatives)

    def missing_from(self, held: Sequence[Capability]) -> str:
        names = " or ".join(f"`{a.name}`" for a in self.alternatives)
        return f"{self.node} needs {names}"

    def describe(self) -> str:
        return f"{self.node}: {self.origin}"


@dataclass
class CapabilityPolicy:
    """What one intent holds, and the demands made against it.

    This is the object the checker reasons with. It is a policy rather than a
    set because it carries the relation: :meth:`covers` is not membership.
    """

    held: List[Capability] = field(default_factory=list)
    held_names: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    demands: List[Demand] = field(default_factory=list)

    @classmethod
    def of(cls, authority: Iterable[str]) -> "CapabilityPolicy":
        names = list(dict.fromkeys(authority))
        return cls(
            held=[Capability.parse(n) for n in names if known(n)],
            held_names=names,
            unknown=[n for n in names if not known(n)],
        )

    def covers(self, cap: Capability) -> bool:
        return any(h.grants(cap) for h in self.held)

    def covers_any(self, alternatives: Iterable[Capability]) -> bool:
        return any(self.covers(a) for a in alternatives)

    def unmet(self) -> List[Demand]:
        return [d for d in self.demands if not d.satisfied_by(self.held)]

    def render(self) -> str:
        lines = ["authority:"]
        if not self.held_names:
            lines.append("  held by the intent: (none)")
        for name in self.held_names:
            cap = Capability.parse(name)
            mark = "" if known(name) else "  <-- not in the vocabulary"
            lines.append(f"  {cap.describe()}{mark}")
        unmet = self.unmet()
        lines.append("  demands: " + (f"{len(self.demands)}"
                                      if self.demands else "none"))
        if unmet:
            lines.append("  NOT COVERED:")
            for demand in unmet:
                lines.append(f"    {demand.missing_from(self.held)}"
                             f"  ({demand.origin})")
        return "\n".join(lines)


# ==========================================================================
# deriving demands from a node
# ==========================================================================

def _walk(expr: Optional[M.MExpr]) -> Iterable[M.MExpr]:
    """Every subexpression, including the root."""
    stack: List[M.MExpr] = [expr] if expr is not None else []
    while stack:
        node = stack.pop()
        if node is None:
            continue
        yield node
        for value in vars(node).values():
            if isinstance(value, M.MExpr):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(v for v in value if isinstance(v, M.MExpr))


def _builtin_key(call: M.MCall) -> str:
    return f"{call.module}.{call.name}" if call.module else call.name


def demands_of(node: M.OpNode, secret_bindings: Set[str],
               type_of=None) -> List[Demand]:
    """The capabilities one operation demands, derived from what it does.

    Two sources, both grounded rather than inferred:

    * the standard library's own ``caps`` metadata for every operation called;
    * a secret value reaching :data:`SECRET_BOUNDARIES`.

    ``node.needs`` is *not* a source of demand -- it is the program's own claim,
    which is checked against the demands here rather than trusted.
    """
    found: List[Demand] = []
    seen: Set[Tuple[str, ...]] = set()

    def add(origin: str, names: Sequence[str], pos=None,
            secret: Optional[str] = None) -> None:
        key = (origin,) + tuple(names)
        if key in seen:
            return
        seen.add(key)
        found.append(Demand(node=node.name, origin=origin, pos=pos,
                            secret=secret,
                            alternatives=tuple(Capability.parse(n)
                                               for n in names)))

    for expr in M.node_expressions(node):
        for sub in _walk(expr):
            if not isinstance(sub, M.MCall):
                continue
            key = _builtin_key(sub)
            builtin = L.BUILTINS.get(key)
            if builtin is not None and builtin.caps:
                add(f"calls `{key}`", builtin.caps, sub.pos)
            if key in SECRET_BOUNDARIES:
                for arg in sub.args:
                    for name in M.references(arg):
                        if name in secret_bindings:
                            add(f"passes the secret `{name}` to `{key}`",
                                (SECRET_EXPOSE,), sub.pos, secret=name)
    return found


def all_demands(model: M.SemanticModel, secret_bindings: Set[str],
                policy: CapabilityPolicy) -> CapabilityPolicy:
    """Collect every demand in the program onto the policy."""
    for node in model.operations.nodes.values():
        if node.kind in ("source", "state"):
            # An input's origin can demand too: `source x : T from io.read_file(p)`
            origin = getattr(node, "origin", None)
            if origin is not None:
                for sub in _walk(origin):
                    if isinstance(sub, M.MCall):
                        key = _builtin_key(sub)
                        builtin = L.BUILTINS.get(key)
                        if builtin is not None and builtin.caps:
                            policy.demands.append(Demand(
                                node=node.name,
                                origin=f"its origin calls `{key}`",
                                alternatives=tuple(
                                    Capability.parse(n)
                                    for n in builtin.caps),
                                pos=sub.pos))
            continue
        policy.demands.extend(demands_of(node, secret_bindings))
    return policy
