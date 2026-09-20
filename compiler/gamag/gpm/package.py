"""Packages, registries, resolution and lock files (audit priority 11).

Spec section 30 asks gpm for semantic versioning, cryptographic package
identity, signed packages, lock files, reproducible dependency resolution,
vulnerability metadata and an offline cache.  This module implements those as
facts about files on disk, because that is the only form in which they can be
checked:

* A **package** is a directory with a `gama.pkg` manifest.  Its identity is the
  SHA-256 of its contents in a canonical order -- not a name and a version,
  which two different people can both publish.
* A **registry** is a directory of packages, which makes the offline cache and a
  private registry the same mechanism.  Reaching the network is deliberately not
  implemented: a resolver that can be tested without one is a resolver whose
  behaviour is reproducible, and spec section 1.3 wants reproducibility before
  convenience.
* A **lock file** records what was resolved, the content hash of each package,
  and optionally a signature, so an install can be verified without trusting the
  resolver that produced it.
* **Resolution is deterministic**: dependencies are visited in sorted order, and
  when several versions satisfy every constraint the highest wins.  Two runs,
  and two machines, produce the same lock file or a named conflict.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..crypto import ed25519
from .semver import Constraint, Version, VersionError

MANIFEST_NAME = "gama.pkg"

#: A safety bound on the resolution fixed point, so that a hypothetical
#: non-convergence becomes a diagnostic rather than a hang.
MAX_ROUNDS = 64
LOCK_NAME = "gama.lock"
ADVISORY_NAME = "advisories.json"

#: Files that are never part of a package's identity: they are either build
#: output or editor noise, and including them would make the hash depend on
#: things the author did not publish.
IGNORED_NAMES = frozenset({
    "__pycache__", ".git", ".ggbuild", ".DS_Store", LOCK_NAME,
})
IGNORED_SUFFIXES = (".pyc", ".pyo", ".swp", "~")


class PackageError(ValueError):
    """Something about a package, a registry or a resolution is wrong."""


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def package_files(directory: str) -> List[str]:
    """Every file that counts towards a package's contents, in sorted order.

    The manifest is excluded on purpose.  Including it creates a cycle: the
    signature covers the content hash, the signature is stored in the manifest,
    and writing the manifest therefore changes the very hash that was signed --
    so a signed package can never verify.  The manifest is covered separately,
    by `package_digest`, over a rendering that leaves the signature out.
    """
    out: List[str] = []
    for root, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_NAMES)
        for name in sorted(filenames):
            if name in IGNORED_NAMES or name == MANIFEST_NAME \
                    or name.endswith(IGNORED_SUFFIXES):
                continue
            out.append(os.path.relpath(os.path.join(root, name), directory))
    return sorted(out)


def content_hash(directory: str) -> str:
    """A hash of the package's *contents*, not of its name or its metadata.

    Each file contributes its relative path and its own hash, so renaming a file
    changes the identity even though the bytes are the same: two packages that
    differ only in a filename are different packages.
    """
    digest = hashlib.sha256()
    for relative in package_files(directory):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file(os.path.join(directory, relative)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_payload(directory: str) -> str:
    """The manifest as it is covered by a signature: without the signature.

    Everything else is included, so changing a dependency or a version changes
    the digest.  `signature` and `public_key` are excluded because they are the
    signature, and a signature cannot cover itself.
    """
    manifest = PackageManifest.read(directory)
    payload = json.loads(manifest.to_json())
    payload.pop("signature", None)
    payload.pop("public_key", None)
    return canonical_json(payload)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def package_digest(directory: str) -> str:
    """The package's identity: its contents *and* its declared metadata.

    This is what a signature covers and what a lock file records.  Both halves
    matter -- contents alone would let a dependency be edited away, metadata
    alone would let the code be replaced.
    """
    digest = hashlib.sha256()
    digest.update(content_hash(directory).encode())
    digest.update(b"\0")
    digest.update(manifest_payload(directory).encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

@dataclass
class PackageManifest:
    """What a `gama.pkg` file says."""

    name: str
    version: Version
    description: str = ""
    dependencies: Dict[str, Constraint] = field(default_factory=dict)
    entry: str = ""
    path: str = ""
    license: str = ""
    #: Optional detached signature over the content hash, and the key that made
    #: it.  Present means signed; absent is honest about being unsigned.
    signature: str = ""
    public_key: str = ""

    @classmethod
    def parse(cls, text: str, *, path: str = "") -> "PackageManifest":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PackageError(f"{path or 'the manifest'} is not valid JSON: "
                               f"{exc}") from None
        if not isinstance(payload, dict):
            raise PackageError(f"{path or 'the manifest'} is not an object")
        for required in ("name", "version"):
            if required not in payload:
                raise PackageError(
                    f"{path or 'the manifest'} has no `{required}`")
        try:
            version = Version.parse(str(payload["version"]))
        except VersionError as exc:
            raise PackageError(f"{path or 'the manifest'}: {exc}") from None
        dependencies: Dict[str, Constraint] = {}
        for name, constraint in (payload.get("dependencies") or {}).items():
            if not isinstance(name, str) or not name:
                raise PackageError(f"{path or 'the manifest'} has a dependency "
                                   f"with an invalid name: {name!r}")
            try:
                dependencies[name] = Constraint.parse(str(constraint))
            except VersionError as exc:
                raise PackageError(
                    f"{path or 'the manifest'}: dependency `{name}`: {exc}"
                ) from None
        return cls(name=payload["name"], version=version,
                   description=str(payload.get("description", "")),
                   dependencies=dependencies,
                   entry=str(payload.get("entry", "")),
                   path=path,
                   license=str(payload.get("license", "")),
                   signature=str(payload.get("signature", "")),
                   public_key=str(payload.get("public_key", "")))

    @classmethod
    def read(cls, directory: str) -> "PackageManifest":
        path = os.path.join(directory, MANIFEST_NAME)
        if not os.path.isfile(path):
            raise PackageError(f"{directory} has no {MANIFEST_NAME}")
        with open(path, "r", encoding="utf-8") as handle:
            return cls.parse(handle.read(), path=path)

    def to_json(self) -> str:
        payload: Dict[str, Any] = {
            "name": self.name,
            "version": str(self.version),
        }
        if self.description:
            payload["description"] = self.description
        if self.dependencies:
            payload["dependencies"] = {
                name: str(constraint)
                for name, constraint in sorted(self.dependencies.items())}
        if self.entry:
            payload["entry"] = self.entry
        if self.license:
            payload["license"] = self.license
        if self.signature:
            payload["signature"] = self.signature
            payload["public_key"] = self.public_key
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def sign(self, secret: bytes, directory: str) -> str:
        """Sign the package's identity, returning the digest that was signed.

        The key must be set before the digest is taken: `package_digest` excludes
        `public_key` from the payload, so it does not matter what it holds, but
        computing the digest first and the key second would make the digest
        depend on a field it deliberately ignores -- which is the mistake this
        docstring exists to prevent someone repeating.
        """
        self.public_key = ed25519.to_hex(ed25519.public_key(secret))
        digest = package_digest(directory)
        self.signature = ed25519.to_hex(ed25519.sign(secret, digest.encode()))
        return digest


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

@dataclass
class Registry:
    """A directory of packages, searched in order.

    Search paths make the offline cache and a private registry the same thing:
    a cache is a registry consulted first.  The order is part of the resolution
    result, so it is explicit rather than incidental.
    """

    roots: List[str] = field(default_factory=list)

    def candidates(self, name: str) -> List[Tuple[Version, str]]:
        """Every (version, directory) this registry offers for `name`."""
        out: List[Tuple[Version, str]] = []
        for root in self.roots:
            package_dir = os.path.join(root, name)
            if not os.path.isdir(package_dir):
                continue
            for entry in sorted(os.listdir(package_dir)):
                candidate = os.path.join(package_dir, entry)
                if not os.path.isdir(candidate):
                    continue
                try:
                    manifest = PackageManifest.read(candidate)
                except PackageError:
                    continue
                if manifest.name != name:
                    # A directory named `foo` holding a package called `bar` is
                    # a hijack attempt or a mistake; either way it is not a
                    # candidate.
                    continue
                out.append((manifest.version, candidate))
        return out

    def versions(self, name: str) -> List[Version]:
        return [version for version, _ in self.candidates(name)]

    def find(self, name: str, version: Version) -> Optional[str]:
        for candidate, directory in self.candidates(name):
            if candidate == version:
                return directory
        return None


def find_manifest(start: str = ".") -> Optional[str]:
    """The nearest `gama.pkg`, walking upwards."""
    current = os.path.abspath(start)
    while True:
        candidate = os.path.join(current, MANIFEST_NAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolved:
    """One package the resolver chose."""

    name: str
    version: Version
    directory: str
    content_hash: str
    signature: str = ""
    public_key: str = ""
    dependency_of: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "version": str(self.version),
            "content_hash": self.content_hash,
        }
        if self.signature:
            payload["signature"] = self.signature
            payload["public_key"] = self.public_key
        return payload


@dataclass
class Resolution:
    """The outcome: what was chosen, and every problem found."""

    resolved: Dict[str, Resolved] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    #: Constraints as they were applied, so a report can explain a choice.
    constraints: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def ordered(self) -> List[Resolved]:
        """Resolved packages, dependencies first, then by name.

        The order is total and derived only from the graph, so a lock file
        written on one machine is identical to one written on another.
        """
        out: List[Resolved] = []
        seen: Set[str] = set()

        def visit(name: str, stack: Set[str]) -> None:
            if name in seen or name in stack:
                return
            stack.add(name)
            entry = self.resolved.get(name)
            if entry is not None:
                for dependency in sorted(
                        _manifest_dependencies(entry.directory)):
                    visit(dependency, stack)
                seen.add(name)
                out.append(entry)
            stack.discard(name)

        for name in sorted(self.resolved):
            visit(name, set())
        return out


def _manifest_dependencies(directory: str) -> Dict[str, Constraint]:
    try:
        return PackageManifest.read(directory).dependencies
    except PackageError:
        return {}


def resolve(root_manifest: PackageManifest, registry: Registry,
            root_directory: str = "") -> Resolution:
    """Resolve `root_manifest`'s dependencies against `registry`.

    **The algorithm is a fixed point, not a single pass.**  A one-pass resolver
    that visits dependencies in sorted order picks `aa` before `b` has
    contributed its constraint when `aa` happens to sort first, and then keeps
    that choice even though `b` forbids it -- reporting success while installing
    a version a constraint excludes.  That bug was written, caught by a test
    with the alphabet reversed, and is the reason this loop exists.

    Each round derives the whole constraint set from the current choices and
    re-picks the highest satisfying version for every package.  The round after
    a change sees the constraints the new version brought with it, so a choice
    that has become unacceptable is replaced.  Iteration stops when nothing
    changes.

    Two deliberate limits, stated rather than discovered:

    * No backtracking.  The highest satisfying version is chosen for each
      package independently, so a graph that can only be satisfied by picking
      *below* the highest satisfying version of something will not resolve.  It
      reports the conflict and exits nonzero; it does not guess.  Failing closed
      is the right direction for a package manager, and a full solver is a
      larger piece of work than this.
    * `MAX_ROUNDS` bounds the loop.  A fixed point over a finite constraint set
      converges quickly, but a bound turns a hypothetical non-convergence into a
      diagnostic instead of a hang.
    """
    out = Resolution()
    candidates_cache: Dict[str, List[Tuple[Version, str]]] = {}
    manifest_cache: Dict[str, PackageManifest] = {}

    def candidates(name: str) -> List[Tuple[Version, str]]:
        if name not in candidates_cache:
            candidates_cache[name] = sorted(registry.candidates(name),
                                            key=lambda pair: pair[0],
                                            reverse=True)
        return candidates_cache[name]

    def manifest_of(directory: str) -> PackageManifest:
        if directory not in manifest_cache:
            manifest_cache[directory] = PackageManifest.read(directory)
        return manifest_cache[directory]

    #: name -> {(version, directory), ...} for the current round
    chosen: Dict[str, Tuple[Version, str]] = {}
    first_problems: List[str] = []

    for round_number in range(MAX_ROUNDS):
        # Derive the constraint set from the root plus the *current* choices, so
        # a constraint contributed by a version we no longer use disappears.
        wanted: Dict[str, List[Tuple[str, Constraint]]] = {}
        for name, constraint in sorted(root_manifest.dependencies.items()):
            wanted.setdefault(name, []).append(("(this package)", constraint))
        for name in sorted(chosen):
            version, directory = chosen[name]
            for child, constraint in sorted(
                    manifest_of(directory).dependencies.items()):
                wanted.setdefault(child, []).append((f"{name} {version}",
                                                     constraint))

        problems: List[str] = []
        new_choices: Dict[str, Tuple[Version, str]] = {}
        for name in sorted(wanted):
            constraints = wanted[name]
            available = candidates(name)
            if not available:
                problems.append(
                    f"no package named `{name}` in any registry "
                    f"({', '.join(registry.roots) or 'none configured'})")
                continue
            acceptable = [(version, directory) for version, directory in available
                          if all(constraint.accepts(version)
                                 for _why, constraint in constraints)]
            if not acceptable:
                asking = "; ".join(f"{constraint} from {why}"
                                   for why, constraint in constraints)
                listed = ", ".join(str(v) for v, _ in available) or "nothing"
                problems.append(
                    f"`{name}` is required as {asking}, but the registry only "
                    f"has {listed}")
                continue
            new_choices[name] = acceptable[0]      # candidates are sorted down

        if problems:
            # Report the first round's problems: a later round would be
            # describing a resolution that was already known to be broken.
            out.problems = first_problems or problems
            return out
        if new_choices == chosen and round_number > 0:
            break
        if not first_problems:
            first_problems = []
        chosen = new_choices
    else:
        out.problems.append(
            f"resolution did not settle after {MAX_ROUNDS} rounds; this is a "
            f"bug in gpm rather than a problem with the packages")
        return out

    for name in sorted(chosen):
        version, directory = chosen[name]
        manifest = manifest_of(directory)
        constraints = []
        for child_constraint in [root_manifest.dependencies.get(name)]:
            if child_constraint is not None:
                constraints.append(f"{child_constraint} from (this package)")
        for other in sorted(chosen):
            if other == name:
                continue
            _v, other_dir = chosen[other]
            constraint = manifest_of(other_dir).dependencies.get(name)
            if constraint is not None:
                constraints.append(
                    f"{constraint} from {other} {chosen[other][0]}")
        out.constraints[name] = constraints
        out.resolved[name] = Resolved(
            name=name, version=version, directory=directory,
            content_hash=package_digest(directory),
            signature=manifest.signature, public_key=manifest.public_key,
            dependency_of=[c.split(" from ")[-1] for c in constraints])
    return out


# ---------------------------------------------------------------------------
# Lock files
# ---------------------------------------------------------------------------

@dataclass
class Lock:
    """A `gama.lock`: exactly what was resolved, with hashes."""

    root: str = ""
    packages: List[Resolved] = field(default_factory=list)

    @classmethod
    def from_resolution(cls, resolution: Resolution, root: str = "") -> "Lock":
        return cls(root=root, packages=resolution.ordered())

    def to_json(self) -> str:
        return json.dumps({
            "lock_version": 1,
            "root": self.root,
            "packages": [entry.to_dict() for entry in self.packages],
        }, indent=2, sort_keys=True) + "\n"

    def write(self, directory: str) -> str:
        path = os.path.join(directory, LOCK_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
        return path

    @classmethod
    def parse(cls, text: str) -> "Lock":
        payload = json.loads(text)
        if payload.get("lock_version") != 1:
            raise PackageError(
                f"lock file version {payload.get('lock_version')!r} is not "
                f"understood; this toolchain writes version 1")
        packages = []
        for entry in payload.get("packages", []):
            try:
                packages.append(Resolved(
                    name=entry["name"],
                    version=Version.parse(entry["version"]),
                    directory="",
                    content_hash=entry.get("content_hash", ""),
                    signature=entry.get("signature", ""),
                    public_key=entry.get("public_key", "")))
            except (KeyError, VersionError) as exc:
                raise PackageError(f"malformed lock entry: {exc}") from None
        return cls(root=payload.get("root", ""), packages=packages)

    @classmethod
    def read(cls, directory: str) -> "Lock":
        path = os.path.join(directory, LOCK_NAME)
        if not os.path.isfile(path):
            raise PackageError(f"{directory} has no {LOCK_NAME}")
        with open(path, "r", encoding="utf-8") as handle:
            return cls.parse(handle.read())


@dataclass
class Verification:
    """Whether a lock file matches what is installed, and why not."""

    ok: bool = True
    problems: List[str] = field(default_factory=list)
    checked: int = 0
    verified_signatures: int = 0
    unsigned: int = 0

    def render(self) -> List[str]:
        out = [f"lock verification: {'ok' if self.ok else 'FAILED'} "
               f"({self.checked} package(s), {self.verified_signatures} signed, "
               f"{self.unsigned} unsigned)"]
        for problem in self.problems:
            out.append(f"  - {problem}")
        if self.ok and self.unsigned:
            out.append(f"  {self.unsigned} package(s) carry no signature: "
                       f"their integrity is verified against the lock file "
                       f"only, which is what the lock file is for")
        return out


def verify_lock(lock: Lock, registry: Registry) -> Verification:
    """Recompute every hash in a lock file and check the signatures.

    Both checks matter and they prove different things.  The hash proves the
    bytes on disk are the bytes that were resolved.  The signature proves who
    published them, and an unsigned package simply cannot make that claim --
    which is reported rather than glossed over.
    """
    out = Verification()
    for entry in lock.packages:
        out.checked += 1
        directory = registry.find(entry.name, entry.version)
        if directory is None:
            out.problems.append(
                f"`{entry.name}` {entry.version} is in the lock file but not "
                f"in the registry")
            continue
        actual = package_digest(directory)
        if actual != entry.content_hash:
            out.problems.append(
                f"`{entry.name}` {entry.version} has contents {actual[:16]} "
                f"but the lock file records {entry.content_hash[:16]}")
            continue
        if entry.signature:
            try:
                key = ed25519.from_hex(entry.public_key)
                signature = ed25519.from_hex(entry.signature)
            except ValueError:
                out.problems.append(
                    f"`{entry.name}` {entry.version} has a malformed signature "
                    f"or public key")
                continue
            if not ed25519.verify(key, entry.content_hash.encode(), signature):
                out.problems.append(
                    f"`{entry.name}` {entry.version} is signed, but the "
                    f"signature does not verify against the contents")
            else:
                out.verified_signatures += 1
        else:
            out.unsigned += 1
    out.ok = not out.problems
    return out


# ---------------------------------------------------------------------------
# Vulnerability metadata
# ---------------------------------------------------------------------------

@dataclass
class Advisory:
    """One published vulnerability affecting a range of versions."""

    identifier: str
    package: str
    affected: Constraint
    severity: str
    summary: str = ""

    @classmethod
    def parse(cls, payload: Dict[str, Any]) -> "Advisory":
        return cls(identifier=payload["id"], package=payload["package"],
                   affected=Constraint.parse(payload["affected"]),
                   severity=payload.get("severity", "unknown"),
                   summary=payload.get("summary", ""))


def load_advisories(directory: str) -> List[Advisory]:
    """Advisories from `advisories.json`, or none if the file is absent.

    Absent is not the same as "nothing is vulnerable", and the report says so:
    an audit against an empty database is evidence of nothing, and a tool that
    prints "no vulnerabilities" in that case is actively misleading.
    """
    path = os.path.join(directory, ADVISORY_NAME)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("advisories", payload) if isinstance(payload, dict) \
        else payload
    return [Advisory.parse(entry) for entry in entries]


@dataclass
class AuditReport:
    advisories: List[Advisory] = field(default_factory=list)
    hits: List[Tuple[Resolved, Advisory]] = field(default_factory=list)
    unknown_severity: int = 0

    def render(self) -> List[str]:
        out: List[str] = []
        if not self.advisories:
            out.append("audit: no advisory database was found, so nothing was "
                       "checked (this is not a clean bill of health)")
            return out
        count = len(self.advisories)
        out.append(f"audit: {count} "
                   f"{'advisory' if count == 1 else 'advisories'} against "
                   f"{len(self.hits)} match{'es' if len(self.hits) != 1 else ''}")
        for entry, advisory in self.hits:
            out.append(f"  [{advisory.severity.upper()}] {advisory.identifier} "
                       f"affects {entry.name} {entry.version}")
            if advisory.summary:
                out.append(f"      {advisory.summary}")
        if not self.hits:
            out.append("  no resolved package falls inside a published range")
        return out


def audit(lock: Lock, advisories: Sequence[Advisory]) -> AuditReport:
    report = AuditReport(advisories=list(advisories))
    for entry in lock.packages:
        for advisory in advisories:
            if advisory.package == entry.name and advisory.affected.accepts(
                    entry.version, allow_prerelease=True):
                report.hits.append((entry, advisory))
    return report
