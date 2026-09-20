"""The capability algebra, shared by the compiler and the runtime.

Spec section 12 makes capabilities the security primitive. This module is the
single definition of what a capability *is* and when one satisfies another, and
it deliberately depends on nothing inside the project: both the compile-time
checker (:mod:`gamag.core.capability`) and the running machine
(:meth:`gamag.runtime.context.Context.has_cap`) import it.

That sharing is the point. A compiler that reasons about capabilities with one
relation and a runtime that enforces them with another is two security models
wearing one name, and the gap between them is where a program that type-checked
gets denied -- or, worse, one that was denied runs. Coverage is defined once,
here, and both sides call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

# Capabilities named by spec section 12. This is the vocabulary; it lives here
# rather than in the runtime so that neither the compiler nor the machine owns
# it more than the other does.
KNOWN_CAPABILITIES = frozenset({
    "FileRead", "FileWrite", "NetworkConnect", "DatabaseRead", "DatabaseWrite",
    "PatientRead", "PatientWrite", "CryptoSign", "AuditWrite", "ProcessSpawn",
    "EnvironmentRead", "SecretExpose", "ModelLoad", "Network",
})


#: The permissions the vocabulary uses. A capability name is a resource followed
#: by one of these; `PatientRead` is `Patient` + `Read`.
PERMISSIONS: Tuple[str, ...] = (
    "Read", "Write", "Connect", "Sign", "Spawn", "Expose", "Load",
)

#: Which permissions entail which others, on the same resource.
#:
#: This is a decision, not a discovery, so it is written down with its reason
#: rather than buried in a comparison operator:
#:
#: * ``Write`` entails ``Read`` because spec section 12 models a capability as
#:   access to a store (``PatientStore[Read]`` versus ``PatientStore[Write]``),
#:   and writing a record requires reading it back.
#: * ``Expose`` entails ``Read`` because exposing a secret is reading it and
#:   then letting it out; the second cannot happen without the first.
#: * ``Load`` entails ``Read`` for the same reason.
#: * ``Connect``, ``Sign`` and ``Spawn`` are actions rather than accesses and
#:   entail nothing. Granting the ability to sign does not grant the ability to
#:   read, and pretending otherwise would widen every capability that names one.
IMPLIES: Dict[str, Tuple[str, ...]] = {
    "Write": ("Read",),
    "Expose": ("Read",),
    "Load": ("Read",),
    "Read": (),
    "Connect": (),
    "Sign": (),
    "Spawn": (),
}


@dataclass(frozen=True)
class Capability:
    """One permission over one resource.

    ``permission == ""`` means the whole resource, which is how the vocabulary's
    bare ``Network`` is read: unrestricted on that resource, and therefore
    covering every permission on it.
    """

    name: str
    resource: str
    permission: str

    @classmethod
    def parse(cls, name: str) -> "Capability":
        """Split a written capability into its resource and its permission.

        Two spellings are accepted, both from the specification. ``PatientRead``
        is a bare name, split on the longest matching permission suffix so that a
        resource ending in a permission word is still handled. ``PatientStore[Read]``
        is section 12's capability-qualified spelling of the same thing: the
        bracket holds the permission and the resource is what the store holds.
        """
        if name.endswith("]") and "[" in name:
            resource, _, rest = name.partition("[")
            permission = rest[:-1].strip()
            if resource.endswith("Store"):
                resource = resource[:-len("Store")]
            return cls(name=name, resource=resource, permission=permission)
        for permission in PERMISSIONS:
            if name.endswith(permission) and len(name) > len(permission):
                return cls(name=name, resource=name[:-len(permission)],
                           permission=permission)
        return cls(name=name, resource=name, permission="")

    def grants(self, other: "Capability") -> bool:
        """Whether holding ``self`` satisfies a demand for ``other``."""
        if self.resource != other.resource:
            return False
        if self.permission == "":          # unrestricted on this resource
            return True
        if other.permission == "":         # the demand is the whole resource
            return False
        if self.permission == other.permission:
            return True
        return other.permission in IMPLIES.get(self.permission, ())

    def attenuates(self, other: "Capability") -> bool:
        """Whether ``other`` is a narrowing of ``self``.

        Attenuation is coverage read backwards: ``PatientWrite`` attenuates to
        ``PatientRead``. A program may always attenuate what it holds; it may
        never do the reverse, because that would be amplification.
        """
        return self.grants(other)

    def describe(self) -> str:
        if not self.permission:
            return f"`{self.name}` (unrestricted access to {self.resource})"
        return f"`{self.name}` ({self.permission} on {self.resource})"


def known(name: str) -> bool:
    """Whether a written capability is one the vocabulary contains."""
    return name in KNOWN_CAPABILITIES


def held_names(policy: "CapabilityPolicy") -> List[str]:
    """The names a policy holds, for use in a diagnostic."""
    return [cap.name for cap in policy.held]


def covers(held: Iterable[str], wanted: str) -> bool:
    """Whether a set of held capability names satisfies a demand for one.

    This is the relation both the checker and the runtime use. ``"*"`` is the
    runtime's wildcard grant and covers everything, which is why it is handled
    here rather than in the caller: a wildcard that only the runtime understood
    would be a second semantics.
    """
    held = list(held)
    if "*" in held:
        return True
    target = Capability.parse(wanted)
    return any(Capability.parse(h).grants(target) for h in held)


def holder(held: Iterable[str], wanted: str) -> str:
    """Which held capability satisfies the demand, for a diagnostic."""
    target = Capability.parse(wanted)
    for name in held:
        if Capability.parse(name).grants(target):
            return name
    return ""
