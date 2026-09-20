"""Checkpointing and state restoration (spec section 11).

A checkpoint contains versioned state plus integrity metadata, and recovery
must validate schema, program version, state version, cryptographic
integrity and authorization context before the state is used again.

The binding constraint from spec section 10 is honoured here: "The runtime
must never silently invent medical or financial state during recovery."
Restoring therefore either returns genuinely recorded state or fails loudly.
There is no default-value path.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .values import canonical, deep_copy_value


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


@dataclass
class Checkpoint:
    id: str
    label: str
    created_at: float
    schema_version: str
    program_version: str
    state_version: int
    authorization: str
    state: Dict[str, Any] = field(default_factory=dict)
    integrity: str = ""
    parent: Optional[str] = None

    @staticmethod
    def digest(state: Dict[str, Any]) -> str:
        return hashlib.sha256(
            _canonical_json(canonical(state)).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "created_at": self.created_at,
            "schema_version": self.schema_version,
            "program_version": self.program_version,
            "state_version": self.state_version,
            "authorization": self.authorization,
            "state": canonical(self.state),
            "integrity": self.integrity, "parent": self.parent,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Checkpoint":
        state = d.get("state") or {}
        return Checkpoint(
            id=d["id"], label=d.get("label", ""), created_at=d.get("created_at", 0.0),
            schema_version=d.get("schema_version", ""),
            program_version=d.get("program_version", ""),
            state_version=int(d.get("state_version", 0)),
            authorization=d.get("authorization", ""),
            state=state, integrity=d.get("integrity", ""),
            parent=d.get("parent"),
        )


class CheckpointRejected(Exception):
    """Raised when a checkpoint fails validation and must not be used."""

    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


class CheckpointStore:
    """Versioned checkpoint history with integrity validation."""

    def __init__(self, *, program_version: str = "0.1.0",
                 schema_version: str = "1", deterministic: bool = False,
                 epoch: float = 0.0, keep: int = 32):
        self.program_version = program_version
        self.schema_version = schema_version
        self.deterministic = deterministic
        self.epoch = epoch
        self.keep = keep
        self.history: List[Checkpoint] = []
        self._clock = 0.0
        self.state_version = 0

    def _now(self) -> float:
        if self.deterministic:
            self._clock += 1.0
            return self.epoch + self._clock
        return time.time()

    def capture(self, state: Dict[str, Any], *, label: str = "",
                authorization: str = "runtime") -> Checkpoint:
        self.state_version += 1
        parent = self.history[-1].id if self.history else None
        cp = Checkpoint(
            id=f"cp-{len(self.history):06d}",
            label=label or f"checkpoint-{len(self.history)}",
            created_at=self._now(),
            schema_version=self.schema_version,
            program_version=self.program_version,
            state_version=self.state_version,
            authorization=authorization,
            state={k: deep_copy_value(v) for k, v in state.items()},
            parent=parent,
        )
        cp.integrity = Checkpoint.digest(cp.state)
        self.history.append(cp)
        if len(self.history) > self.keep:
            del self.history[0]
        return cp

    def latest(self) -> Optional[Checkpoint]:
        return self.history[-1] if self.history else None

    def with_label(self, label: str) -> Optional[Checkpoint]:
        """The most recent checkpoint recorded under ``label``, if any.

        A recovery policy that names a checkpoint has to get *that* checkpoint.
        Restoring the latest one instead would return the program to state it did
        not reason about, which is a quiet version of the invention spec section
        10 forbids -- the state is genuine, but it is not the state asked for.
        """
        if not label:
            return None
        for cp in reversed(self.history):
            if cp.label == label:
                return cp
        return None

    def get(self, cp_id: str) -> Optional[Checkpoint]:
        for cp in self.history:
            if cp.id == cp_id:
                return cp
        return None

    def validate(self, cp: Checkpoint, *, authorization: Optional[str] = None,
                 schema_version: Optional[str] = None,
                 program_version: Optional[str] = None,
                 max_state_version: Optional[int] = None) -> Tuple[bool, List[str]]:
        """Run every validation spec section 11 requires."""
        problems: List[str] = []
        if cp.schema_version != (schema_version or self.schema_version):
            problems.append(
                f"schema mismatch: checkpoint is schema "
                f"{cp.schema_version!r}, runtime expects "
                f"{schema_version or self.schema_version!r}")
        if cp.program_version != (program_version or self.program_version):
            problems.append(
                f"program version mismatch: checkpoint was written by "
                f"{cp.program_version!r}, running {program_version or self.program_version!r}")
        if max_state_version is not None and cp.state_version > max_state_version:
            problems.append(
                f"state version {cp.state_version} is newer than the permitted "
                f"maximum {max_state_version}")
        expected = Checkpoint.digest(cp.state)
        if cp.integrity != expected:
            problems.append(
                f"integrity check failed: stored digest {cp.integrity[:12]}... "
                f"does not match recomputed {expected[:12]}...")
        if authorization is not None and cp.authorization != authorization:
            problems.append(
                f"authorization context mismatch: checkpoint captured under "
                f"{cp.authorization!r}, requested under {authorization!r}")
        return (not problems), problems

    def restore(self, cp: Optional[Checkpoint] = None, *,
                authorization: Optional[str] = None) -> Dict[str, Any]:
        """Restore validated state, or raise rather than invent any.

        Spec section 10 forbids silently inventing state during recovery, so a
        missing or invalid checkpoint is a hard failure that escalates.
        """
        if cp is None:
            cp = self.latest()
        if cp is None:
            raise CheckpointRejected([
                "no checkpoint exists to restore; refusing to fabricate state "
                "(spec section 10 forbids inventing state during recovery)"])
        ok, problems = self.validate(cp, authorization=authorization)
        if not ok:
            raise CheckpointRejected(problems)
        return {k: deep_copy_value(v) for k, v in cp.state.items()}

    def to_jsonl(self) -> str:
        return "\n".join(_canonical_json(cp.to_dict()) for cp in self.history)

    @classmethod
    def from_jsonl(cls, text: str, **kwargs) -> "CheckpointStore":
        store = cls(**kwargs)
        for line in text.splitlines():
            line = line.strip()
            if line:
                store.history.append(Checkpoint.from_dict(json.loads(line)))
        if store.history:
            store.state_version = store.history[-1].state_version
        return store
