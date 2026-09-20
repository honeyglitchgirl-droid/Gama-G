"""Reproducible builds and signed build manifests (audit priority 10).

Spec section 12 lists three requirements that belong together: packages should
be signed, dependencies verified for integrity, and builds reproducible.  This
module supplies the third and the machinery for the first two.

The interesting design question is what a manifest may contain.  Anything that
varies between two runs of the same build -- a timestamp, a temporary path, an
absolute path, the order a dictionary happened to iterate in -- destroys
reproducibility, and the failure is silent: the hashes simply differ and nobody
knows why.  So the manifest has a *fixed* set of fields, every one of them
derived from the inputs, and `reproducibility_report` checks the only thing that
matters: that building the same source twice gives the same manifest.

Signing is Ed25519 (`gamag.crypto.ed25519`).  A keyed hash would have been
shorter, but it proves possession of a shared secret rather than identity, and
spec section 30 asks for cryptographic package *identity*.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import GIR_VERSION, SPEC_VERSION, __version__
from ..crypto import ed25519

#: The manifest schema.  A verifier that does not know this version must refuse
#: rather than guess, which is why it is in the signed payload.
MANIFEST_VERSION = 1


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """JSON with sorted keys and no insignificant whitespace.

    The signature covers these bytes, so two implementations that agree on the
    data must agree on the encoding.  `sort_keys` is doing the real work: a
    dict's iteration order is not part of its meaning, and signing an unordered
    rendering would produce signatures that verify only sometimes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def compiler_fingerprint() -> Dict[str, str]:
    """What produced a build.

    The compiler's own sources are hashed rather than trusted from a version
    string: two checkouts can both say 1.0.0 and compile different programs, and
    the manifest is supposed to be evidence.
    """
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digest = hashlib.sha256()
    files = 0
    for root, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in ("__pycache__", ".ggbuild"))
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            relative = os.path.relpath(path, package_dir)
            digest.update(relative.encode("utf-8"))
            with open(path, "rb") as handle:
                digest.update(handle.read())
            files += 1
    return {
        "compiler_version": __version__,
        "gir_version": GIR_VERSION,
        "spec_version": SPEC_VERSION,
        "compiler_source_sha256": digest.hexdigest(),
        "compiler_source_files": str(files),
        "python": ".".join(str(p) for p in sys.version_info[:3]),
        "platform": platform.system().lower(),
    }


def c_toolchain_fingerprint(compiler: Optional[str] = None) -> Dict[str, str]:
    """The C compiler's identity, for builds that use the native backend."""
    from ..backend import native as native_backend

    cc = compiler or native_backend.find_c_compiler()
    if not cc:
        return {"cc": "", "cc_version": "none"}
    out: Dict[str, str] = {"cc": os.path.basename(cc), "cc_version": "unknown"}
    for flag in ("--version", "-v"):
        try:
            proc = subprocess.run([cc, flag], capture_output=True, text=True,
                                  timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (proc.stdout or proc.stderr or "").strip()
        if text:
            out["cc_version"] = text.splitlines()[0].strip()[:200]
            break
    return out


def program_fingerprint(program: Any) -> str:
    """A hash of the derived graph, which is what was actually compiled.

    Not the source: two different programs can lower to the same GIR, and the
    thing a backend consumes is the GIR.  Hashing the source would let a change
    in the front end go unnoticed in the manifest.
    """
    if hasattr(program, "to_json"):
        try:
            payload = program.to_json()
        except Exception:                          # noqa: BLE001
            payload = repr(program)
        if not isinstance(payload, str):
            payload = canonical_json(payload)
    else:
        payload = repr(program)
    return _sha256_text(payload)


def build_manifest(*, source_path: str, source_text: str, program: Any,
                   profile: str = "strict", opt_level: int = 1,
                   target: str = "interpreter",
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The manifest for one build.  Deterministic by construction."""
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "source": {
            "name": os.path.basename(source_path),
            "sha256": _sha256_text(source_text),
            "bytes": len(source_text.encode("utf-8")),
        },
        "program_sha256": program_fingerprint(program),
        "build": {"profile": profile, "opt_level": opt_level,
                  "target": target},
        "compiler": compiler_fingerprint(),
    }
    if target == "native":
        manifest["toolchain"] = c_toolchain_fingerprint()
    if extra:
        # Callers may add facts, but the manifest stays canonical: the extras are
        # nested under one key so they cannot collide with the schema.
        manifest["extra"] = extra
    return manifest


def manifest_digest(manifest: Dict[str, Any]) -> str:
    return _sha256_text(canonical_json(manifest))


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

@dataclass
class SignedManifest:
    """A manifest, its digest, and a signature over that digest."""

    manifest: Dict[str, Any]
    digest: str
    public_key: str = ""
    signature: str = ""
    signed: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "manifest": self.manifest,
            "digest": self.digest,
            "public_key": self.public_key,
            "signature": self.signature,
            "signed": self.signed,
        }, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "SignedManifest":
        payload = json.loads(text)
        return cls(manifest=payload["manifest"], digest=payload.get("digest", ""),
                   public_key=payload.get("public_key", ""),
                   signature=payload.get("signature", ""),
                   signed=bool(payload.get("signed", False)))


def sign_manifest(manifest: Dict[str, Any],
                  secret_key: Optional[bytes] = None) -> SignedManifest:
    """Sign a manifest, or return it unsigned when no key was supplied.

    Unsigned is a legitimate outcome and is recorded as such: a manifest that
    silently claimed to be signed when it was not is worse than an unsigned one.
    """
    digest = manifest_digest(manifest)
    if secret_key is None:
        return SignedManifest(manifest=manifest, digest=digest, signed=False)
    if len(secret_key) != 32:
        raise ValueError("an Ed25519 secret key is 32 bytes")
    return SignedManifest(
        manifest=manifest,
        digest=digest,
        public_key=ed25519.to_hex(ed25519.public_key(secret_key)),
        signature=ed25519.to_hex(
            ed25519.sign(secret_key, digest.encode("ascii"))),
        signed=True,
    )


def verify_manifest(signed: SignedManifest,
                    public_key: Optional[bytes] = None) -> List[str]:
    """Problems with a signed manifest, or an empty list.

    Two independent checks, because they catch different attacks: the digest
    catches a manifest whose contents were edited, and the signature catches a
    digest that was replaced along with them.
    """
    problems: List[str] = []
    if signed.digest and signed.digest != manifest_digest(signed.manifest):
        problems.append(
            f"the manifest's contents hash to "
            f"{manifest_digest(signed.manifest)[:16]} but it claims "
            f"{signed.digest[:16]}, so it was edited after signing")
    if not signed.signed:
        problems.append("the manifest is not signed")
        return problems
    key = public_key
    if key is None:
        if not signed.public_key:
            problems.append("the manifest is signed but carries no public key")
            return problems
        key = ed25519.from_hex(signed.public_key)
    if not signed.signature:
        problems.append("the manifest claims to be signed but has no signature")
        return problems
    if not ed25519.verify(key, signed.digest.encode("ascii"),
                          ed25519.from_hex(signed.signature)):
        problems.append(
            "the signature does not verify against the public key, so the "
            "manifest was not produced by the holder of the matching secret")
    return problems


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@dataclass
class Reproducibility:
    """Whether the same source produced the same manifest twice."""

    reproducible: bool = True
    first_digest: str = ""
    second_digest: str = ""
    differences: List[str] = field(default_factory=list)
    runs: int = 0
    elapsed_ms: float = 0.0

    def render(self) -> List[str]:
        out = [f"reproducibility: {'yes' if self.reproducible else 'NO'} "
               f"({self.runs} builds in {self.elapsed_ms:.0f} ms)"]
        if self.first_digest:
            out.append(f"  manifest digest: {self.first_digest}")
        for difference in self.differences:
            out.append(f"  - {difference}")
        if self.reproducible:
            out.append("  the same source produced the same manifest, which is "
                       "what a verifier needs")
        return out


def _diff_manifests(a: Dict[str, Any], b: Dict[str, Any],
                    path: str = "") -> List[str]:
    """Where two manifests differ, as readable paths."""
    out: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            where = f"{path}.{key}" if path else key
            if key not in a:
                out.append(f"{where}: only in the second manifest")
            elif key not in b:
                out.append(f"{where}: only in the first manifest")
            else:
                out.extend(_diff_manifests(a[key], b[key], where))
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def reproducibility_report(source_path: str, source_text: str,
                           build, runs: int = 2) -> Reproducibility:
    """Run `build` several times and compare the manifests it produces.

    `build` is a callable returning a manifest.  Passing it in keeps this
    function independent of which backend is being checked, which matters
    because the same question applies to the interpreter, the native backend
    and the package manager.
    """
    report = Reproducibility(runs=runs)
    digests: List[str] = []
    manifests: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(max(2, runs)):
        manifests.append(build())
        digests.append(manifest_digest(manifests[-1]))
    report.elapsed_ms = (time.perf_counter() - started) * 1000.0
    report.first_digest = digests[0]
    report.second_digest = digests[1]
    if len(set(digests)) != 1:
        report.reproducible = False
        report.differences = _diff_manifests(manifests[0], manifests[1])[:10]
    return report


def load_key(path: str) -> bytes:
    """Read a signing key from a file.

    The file holds 64 hex characters.  It is read as bytes and never logged;
    a key that reaches a log has to be rotated.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    key = ed25519.from_hex(text)
    if len(key) != 32:
        raise ValueError(f"{path} does not hold a 32-byte key")
    return key


def generate_key(path: str, *, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Create a signing key at `path`, returning (secret, public).

    The secret is written with owner-only permissions.  A world-readable
    signing key is not a signing key.
    """
    if seed is None:
        seed = os.urandom(32)
    secret, public = ed25519.keypair_from_seed(seed)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    # Create with restrictive permissions rather than fixing them afterwards,
    # so the key is never briefly world-readable.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(ed25519.to_hex(secret) + "\n")
    return secret, public
