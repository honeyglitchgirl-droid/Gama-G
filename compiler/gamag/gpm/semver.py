"""Semantic versioning (spec section 30: "semantic versioning", gpm).

Version comparisons in a package manager are a safety-critical function and they
are easy to get subtly wrong: `1.0.0-rc.2` sorts *below* `1.0.0`, a numeric
prerelease identifier sorts below an alphanumeric one, and a build tag is
ignored entirely when comparing.  Each of those has caused a real supply-chain
incident in some ecosystem, so they are implemented to the specification (SemVer
2.0.0 section 11) rather than to intuition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering
from typing import List, Optional, Sequence, Tuple

VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z\-.]+))?"
    r"(?:\+(?P<build>[0-9A-Za-z\-.]+))?$")

IDENT_RE = re.compile(r"^[0-9A-Za-z-]+$")


class VersionError(ValueError):
    """A version or constraint that cannot be parsed."""


@total_ordering
@dataclass(frozen=True)
class Version:
    """A semantic version, comparable by the rules of SemVer 2.0.0 section 11."""

    major: int
    minor: int
    patch: int
    prerelease: Tuple[str, ...] = ()
    build: Tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "Version":
        match = VERSION_RE.match(text.strip())
        if not match:
            raise VersionError(
                f"{text!r} is not a semantic version; versions look like "
                f"`1.2.3`, optionally with `-prerelease` and `+build`")
        pre = tuple(match.group("pre").split(".")) if match.group("pre") else ()
        build = (tuple(match.group("build").split("."))
                 if match.group("build") else ())
        for part in pre + build:
            if not IDENT_RE.match(part):
                raise VersionError(f"{text!r} has an invalid identifier {part!r}")
            if len(part) > 1 and part[0] == "0" and part.isdigit():
                raise VersionError(
                    f"{text!r} has a numeric identifier with a leading zero "
                    f"({part!r}), which SemVer forbids")
        return cls(int(match.group("major")), int(match.group("minor")),
                   int(match.group("patch")), pre, build)

    def __str__(self) -> str:
        text = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            text += "-" + ".".join(self.prerelease)
        if self.build:
            text += "+" + ".".join(self.build)
        return text

    def _precedence_key(self):
        # A version with a prerelease sorts below the same version without one:
        # 1.0.0-rc.1 < 1.0.0.  An empty tuple must therefore sort *higher*, which
        # is why the flag is inverted here.
        return (self.major, self.minor, self.patch,
                (1, ()) if not self.prerelease else (0, self._pre_tuple()))

    def _pre_tuple(self):
        out = []
        for part in self.prerelease:
            if part.isdigit():
                out.append((0, int(part), ""))
            else:
                out.append((1, 0, part))
        return tuple(out)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        # Build metadata is ignored when determining precedence.
        return self._precedence_key() == other._precedence_key()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        left, right = self._precedence_key(), other._precedence_key()
        # Compare prerelease tuples element-wise with the SemVer rule that a
        # shorter set of identifiers sorts lower when all shared ones are equal.
        if left[:3] != right[:3]:
            return left[:3] < right[:3]
        lflag, lpre = left[3]
        rflag, rpre = right[3]
        if lflag != rflag:
            return lflag < rflag
        if lpre == rpre:
            return False
        for a, b in zip(lpre, rpre):
            if a != b:
                return a < b
        return len(lpre) < len(rpre)

    def __hash__(self) -> int:
        return hash(self._precedence_key())

    @property
    def is_prerelease(self) -> bool:
        return bool(self.prerelease)

    def bump(self, part: str) -> "Version":
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise VersionError(f"cannot bump {part!r}")


@dataclass(frozen=True)
class Constraint:
    """A version requirement, as written in a package manifest."""

    #: One of: exact, range, caret, tilde
    kind: str
    version: Version
    #: For `range`, the operator that was written.
    operator: str = ""

    #: `^1.2.3` accepts >= 1.2.3 and < 2.0.0 -- but for 0.x it is stricter,
    #: because in 0.x every minor release may break.  Getting this wrong is how
    #: a resolver silently installs an incompatible version.
    @classmethod
    def parse(cls, text: str) -> "Constraint":
        raw = text.strip()
        if not raw or raw == "*":
            return cls("any", Version(0, 0, 0))
        if raw.startswith("^"):
            return cls("caret", Version.parse(raw[1:]))
        if raw.startswith("~"):
            body = raw[1:]
            # `~1.2` means >= 1.2.0 and < 1.3.0
            parts = body.split(".")
            version = Version.parse(body if len(parts) == 3
                                    else body + ".0" * (3 - len(parts)))
            return cls("tilde", version)
        for operator in (">=", "<=", ">", "<", "==", "="):
            if raw.startswith(operator):
                kind = "range" if operator not in ("==", "=") else "exact"
                return cls(kind, Version.parse(raw[len(operator):]),
                           operator=operator)
        return cls("exact", Version.parse(raw))

    def __str__(self) -> str:
        if self.kind == "any":
            return "*"
        if self.kind == "caret":
            return f"^{self.version}"
        if self.kind == "tilde":
            return f"~{self.version}"
        if self.kind == "exact":
            return str(self.version)
        return f"{self.operator}{self.version}"

    def accepts(self, version: Version, *, allow_prerelease: bool = False) -> bool:
        """Whether `version` satisfies this constraint.

        Prereleases are refused unless asked for, which is the convention every
        mature package manager follows: `^1.0.0` must not install `2.0.0-rc1`,
        and it must not install `1.5.0-beta` either.
        """
        if version.is_prerelease and not allow_prerelease:
            return False
        if self.kind == "any":
            return True
        if self.kind == "exact":
            return version == self.version
        if self.kind == "caret":
            lower = self.version
            if self.version.major != 0:
                upper = Version(self.version.major + 1, 0, 0)
            elif self.version.minor != 0:
                upper = Version(0, self.version.minor + 1, 0)
            else:
                upper = Version(0, 0, self.version.patch + 1)
            return lower <= version < upper
        if self.kind == "tilde":
            lower = self.version
            upper = Version(self.version.major, self.version.minor + 1, 0)
            return lower <= version < upper
        if self.operator == ">=":
            return version >= self.version
        if self.operator == "<=":
            return version <= self.version
        if self.operator == ">":
            return version > self.version
        if self.operator == "<":
            return version < self.version
        return False


def highest(versions: Sequence[Version],
            constraint: Optional[Constraint] = None,
            *, allow_prerelease: bool = False) -> Optional[Version]:
    """The highest version that satisfies `constraint`, or None."""
    candidates = [v for v in versions
                  if constraint is None
                  or constraint.accepts(v, allow_prerelease=allow_prerelease)]
    return max(candidates) if candidates else None
