"""Tamper-evident audit log (spec section 13).

Each record carries the identity, policy, operation, timestamp and integrity
information required by spec section 1.7, and is hash-chained to its
predecessor so that modification of any historical record is detectable.

The specification is explicit that this is a *detective* control: "The
language itself cannot magically prevent corruption or fraud. It can provide
technical controls that make unauthorized modification detectable."  Nothing
here claims otherwise.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .values import canonical


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


@dataclass
class AuditRecord:
    """One link in the audit chain."""

    seq: int
    event_id: str
    actor: str
    authority: str
    action: str
    object: Optional[str]
    reason: Optional[str]
    timestamp: float
    program_version: str
    policy_version: str
    prev_hash: str
    fields: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    signature: str = ""
    level: str = "info"          # info | recovery | security | policy | medical

    def digest_payload(self) -> Dict[str, Any]:
        """Exactly the bytes that are hashed -- stable and complete."""
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "actor": self.actor,
            "authority": self.authority,
            "action": self.action,
            "object": self.object,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "program_version": self.program_version,
            "policy_version": self.policy_version,
            "prev_hash": self.prev_hash,
            "level": self.level,
            "fields": canonical(self.fields),
        }

    def to_dict(self) -> Dict[str, Any]:
        out = self.digest_payload()
        out["hash"] = self.hash
        out["signature"] = self.signature
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AuditRecord":
        rec = AuditRecord(
            seq=d["seq"], event_id=d["event_id"], actor=d["actor"],
            authority=d["authority"], action=d["action"], object=d.get("object"),
            reason=d.get("reason"), timestamp=d["timestamp"],
            program_version=d.get("program_version", ""),
            policy_version=d.get("policy_version", ""),
            prev_hash=d["prev_hash"], fields=d.get("fields") or {},
            hash=d.get("hash", ""), signature=d.get("signature", ""),
            level=d.get("level", "info"),
        )
        return rec


GENESIS_HASH = "0" * 64


class AuditLog:
    """An append-oriented, hash-chained, signed audit log."""

    def __init__(self, *, key: Optional[bytes] = None, actor: str = "system",
                 authority: str = "gama-g/runtime",
                 program_version: str = "0.1.0", policy_version: str = "0",
                 deterministic: bool = False, epoch: float = 0.0):
        self.records: List[AuditRecord] = []
        # A runtime-generated key signs the chain.  It is deliberately not
        # derivable from program text: signing keys belong to the deployment.
        self.key = key if key is not None else uuid.uuid4().bytes
        self.actor = actor
        self.authority = authority
        self.program_version = program_version
        self.policy_version = policy_version
        self.deterministic = deterministic
        self.epoch = epoch
        self._clock = 0.0
        # Parallel regions (spec section 9) can emit audit records from
        # several tasks; chaining must stay serialised to remain valid.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def now(self) -> float:
        if self.deterministic:
            # Reproducible timestamps: a synthetic monotonic clock so that two
            # runs of the same program produce byte-identical audit chains.
            self._clock += 1.0
            return self.epoch + self._clock
        return time.time()

    @property
    def head_hash(self) -> str:
        return self.records[-1].hash if self.records else GENESIS_HASH

    def __len__(self) -> int:
        return len(self.records)

    def compute_hash(self, rec: AuditRecord) -> str:
        return hashlib.sha256(
            _canonical_json(rec.digest_payload()).encode("utf-8")).hexdigest()

    def sign(self, digest: str) -> str:
        return hmac.new(self.key, digest.encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def record(self, action: str, *, actor: Optional[str] = None,
               object: Optional[str] = None, reason: Optional[str] = None,
               authority: Optional[str] = None, level: str = "info",
               **fields: Any) -> AuditRecord:
        with self._lock:
            return self._append(action, actor=actor, object=object,
                                reason=reason, authority=authority,
                                level=level, **fields)

    def _append(self, action: str, *, actor: Optional[str] = None,
                object: Optional[str] = None, reason: Optional[str] = None,
                authority: Optional[str] = None, level: str = "info",
                **fields: Any) -> AuditRecord:
        rec = AuditRecord(
            seq=len(self.records),
            event_id=self._new_event_id(),
            actor=actor or self.actor,
            authority=authority or self.authority,
            action=str(action),
            object=None if object is None else str(object),
            reason=None if reason is None else str(reason),
            timestamp=self.now(),
            program_version=self.program_version,
            policy_version=self.policy_version,
            prev_hash=self.head_hash,
            fields={k: v for k, v in fields.items() if v is not None},
            level=level,
        )
        rec.hash = self.compute_hash(rec)
        rec.signature = self.sign(rec.hash)
        self.records.append(rec)
        return rec

    def _new_event_id(self) -> str:
        if self.deterministic:
            return f"evt-{len(self.records):08d}"
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    def verify(self) -> Tuple[bool, List[str]]:
        """Re-validate the whole chain.

        Checks linkage, digest integrity and signatures, returning the list of
        concrete problems found.  This is what ``ggc audit verify`` runs.
        """
        problems: List[str] = []
        prev = GENESIS_HASH
        for i, rec in enumerate(self.records):
            if rec.seq != i:
                problems.append(f"record {i}: seq is {rec.seq}, expected {i}")
            if rec.prev_hash != prev:
                problems.append(
                    f"record {i} ({rec.action}): prev_hash "
                    f"{rec.prev_hash[:12]}... does not match the previous "
                    f"record hash {prev[:12]}... -- the chain is broken here")
            expected = self.compute_hash(rec)
            if rec.hash != expected:
                problems.append(
                    f"record {i} ({rec.action}): stored hash "
                    f"{rec.hash[:12]}... does not match the recomputed digest "
                    f"{expected[:12]}... -- the record contents were modified")
            if not hmac.compare_digest(rec.signature, self.sign(rec.hash)):
                problems.append(
                    f"record {i} ({rec.action}): signature does not verify "
                    f"-- the record was re-hashed with a different key")
            prev = rec.hash
        return (not problems), problems

    # ------------------------------------------------------------------
    def to_jsonl(self) -> str:
        return "\n".join(_canonical_json(r.to_dict()) for r in self.records)

    @classmethod
    def from_jsonl(cls, text: str, **kwargs) -> "AuditLog":
        log = cls(**kwargs)
        for line in text.splitlines():
            line = line.strip()
            if line:
                log.records.append(AuditRecord.from_dict(json.loads(line)))
        return log

    def summary(self) -> Dict[str, Any]:
        by_action: Dict[str, int] = {}
        by_level: Dict[str, int] = {}
        for rec in self.records:
            by_action[rec.action] = by_action.get(rec.action, 0) + 1
            by_level[rec.level] = by_level.get(rec.level, 0) + 1
        return {
            "records": len(self.records),
            "head_hash": self.head_hash,
            "by_action": by_action,
            "by_level": by_level,
            "verified": self.verify()[0],
        }


# ---------------------------------------------------------------------------
# offline verification -- what `ggc audit verify` runs
# ---------------------------------------------------------------------------
def verify_trail(text: str,
                 key: Optional[bytes] = None) -> Tuple[bool, List[str],
                                                        Dict[str, Any]]:
    """Verify a trail as written, without the runtime that produced it.

    Two checks never need a key and are therefore always run: the recomputed
    SHA-256 digest of every record, and the `prev_hash` linkage that makes the
    trail a chain rather than a list.  The HMAC signature binds the chain to
    the deployment's key, which deliberately does not live in the file; it is
    checked only when a key is supplied, and the summary says which mode ran,
    because "signatures not checked" is information a reviewer needs.

    Returns ``(ok, problems, summary)``.
    """
    log = AuditLog.from_jsonl(text)
    problems: List[str] = []
    prev = GENESIS_HASH
    for i, rec in enumerate(log.records):
        if rec.seq != i:
            problems.append(f"record {i}: seq is {rec.seq}, expected {i}")
        if rec.prev_hash != prev:
            problems.append(
                f"record {i} ({rec.action}): prev_hash "
                f"{rec.prev_hash[:12]}... does not match the previous record "
                f"hash {prev[:12]}... -- the chain is broken here")
        expected = log.compute_hash(rec)
        if rec.hash != expected:
            problems.append(
                f"record {i} ({rec.action}): stored hash "
                f"{rec.hash[:12]}... does not match the recomputed digest "
                f"{expected[:12]}... -- the record contents were modified")
        if key is not None:
            expected_sig = hmac.new(key, rec.hash.encode("utf-8"),
                                    hashlib.sha256).hexdigest()
            if not hmac.compare_digest(rec.signature, expected_sig):
                problems.append(
                    f"record {i} ({rec.action}): signature does not verify "
                    f"against the supplied key")
        prev = rec.hash
    by_action: Dict[str, int] = {}
    by_level: Dict[str, int] = {}
    for rec in log.records:
        by_action[rec.action] = by_action.get(rec.action, 0) + 1
        by_level[rec.level] = by_level.get(rec.level, 0) + 1
    summary = {
        "records": len(log.records),
        "head_hash": log.head_hash,
        "by_action": by_action,
        "by_level": by_level,
        "signatures_checked": key is not None,
        "first_timestamp": (log.records[0].timestamp if log.records else None),
        "last_timestamp": (log.records[-1].timestamp if log.records else None),
    }
    return (not problems), problems, summary
