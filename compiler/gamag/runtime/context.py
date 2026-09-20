"""The Gama-G Core Runtime (GCR) execution context.

Holds everything a running program is allowed to touch: the audit log, the
checkpoint store, the granted capability set (spec section 12), the
deterministic clock and RNG (spec section 1.3), the safe-event journal used
by recovery replay, and the operator-notification channel used at recovery
level 5.

Capabilities are the security boundary.  A program has *no* ambient
filesystem or network access: builtins that need one call
:meth:`Context.require_capability`, which raises unless the capability was
explicitly granted.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TextIO

from ..diagnostics import CapabilityViolation, GamaRuntimeFault
from .audit import AuditLog
from .checkpoint import CheckpointStore

# The capability vocabulary and the coverage relation both live in
# `gamag.capabilities`, shared with the compile-time checker. They are re-exported
# here because this module is where the runtime looks for them, and a second copy
# of the list would eventually disagree with the first.
from ..capabilities import KNOWN_CAPABILITIES, covers as _covers  # noqa: E402


@dataclass
class RuntimeStats:
    instructions: int = 0
    calls: int = 0
    audits: int = 0
    checkpoints: int = 0
    recoveries: int = 0
    recoveries_succeeded: int = 0
    parallel_regions: int = 0
    parallel_tasks: int = 0
    capability_denials: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instructions": self.instructions, "calls": self.calls,
            "audits": self.audits, "checkpoints": self.checkpoints,
            "recoveries": self.recoveries,
            "recoveries_succeeded": self.recoveries_succeeded,
            "parallel_regions": self.parallel_regions,
            "parallel_tasks": self.parallel_tasks,
            "capability_denials": self.capability_denials,
            "elapsed_seconds": round(self.elapsed(), 6),
        }


class Context:
    def __init__(self, *, grants: Optional[Set[str]] = None,
                 deterministic: bool = False, seed: int = 0,
                 program_version: str = "0.1.0", policy_version: str = "0",
                 actor: str = "program", stdout: Optional[TextIO] = None,
                 stderr: Optional[TextIO] = None,
                 audit_key: Optional[bytes] = None,
                 epoch: float = 0.0, max_steps: int = 50_000_000,
                 allow_ungranted: bool = False):
        self.grants: Set[str] = set(grants or ())
        self.deterministic = deterministic
        self.seed = seed
        self.program_version = program_version
        self.policy_version = policy_version
        self.actor = actor
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.epoch = epoch
        self.max_steps = max_steps
        # Escape hatch for experiments only; the checker reports its use.
        self.allow_ungranted = allow_ungranted

        self.audit = AuditLog(
            key=audit_key, actor=actor, program_version=program_version,
            policy_version=policy_version, deterministic=deterministic,
            epoch=epoch,
        )
        self.checkpoints = CheckpointStore(
            program_version=program_version, deterministic=deterministic,
            epoch=epoch,
        )
        self.rng = random.Random(seed)
        self.stats = RuntimeStats()
        self.journal: List[Dict[str, Any]] = []
        self.operator_alerts: List[Dict[str, Any]] = []
        self.components: Dict[str, int] = {}
        self.failovers: List[Dict[str, Any]] = []
        self.stages: List[Dict[str, Any]] = []
        self.agent_messages: List[Dict[str, Any]] = []
        self.transactions: Dict[str, Dict[str, Any]] = {}
        self.roles: List[str] = []
        self.active_transaction: Optional[str] = None
        # Which function opened the transaction that is still active. Without
        # this, a fault escaping a *callee* would abort a transaction belonging to
        # its caller, and a fault escaping the caller could not be attributed at
        # all -- spec section 18 needs the abort to land on the transaction that
        # actually failed.
        self.transaction_owner: Optional[str] = None
        self._call_hook: Optional[Callable[[Any, tuple], Any]] = None
        self._clock = 0.0

        # Installed by the VM: how to read/write critical program state.
        self.state_provider: Callable[[], Dict[str, Any]] = lambda: {}
        self.state_applier: Callable[[Dict[str, Any]], Dict[str, Any]] = \
            lambda s: dict(s)

        unknown = self.grants - KNOWN_CAPABILITIES
        self.unknown_grants = sorted(unknown)

    # ------------------------------------------------------------------
    # time and entropy (deterministic when requested, spec section 1.3)
    # ------------------------------------------------------------------
    def now(self) -> float:
        if self.deterministic:
            self._clock += 1.0
            return self.epoch + self._clock
        return time.time()

    def random(self) -> float:
        return self.rng.random()

    def randint(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    # ------------------------------------------------------------------
    # capabilities (spec section 12)
    # ------------------------------------------------------------------
    def has_cap(self, cap: str) -> bool:
        """Whether the granted set satisfies a demand for ``cap``.

        Coverage, not membership, and the same relation the compiler used when it
        accepted the program: an intent holding `PatientWrite` satisfies a demand
        for `PatientRead` at run time exactly as it did at compile time. A runtime
        that tested membership would deny programs the checker had accepted, and
        the denial would look like a bug in the program rather than a disagreement
        between two halves of the toolchain.
        """
        return _covers(self.grants, cap)

    def require_capability(self, cap: str, *, what: str = "",
                           pos: Any = None) -> None:
        if self.has_cap(cap):
            return
        if self.allow_ungranted:
            self.stats.capability_denials += 1
            self.audit.record(
                "CAPABILITY_UNGRANTED_BYPASSED", level="security",
                action_detail=what, capability=cap,
                reason="runtime started with --allow-ungranted; this is not a "
                       "secure profile")
            return
        self.stats.capability_denials += 1
        rec = self.audit.record(
            "CAPABILITY_DENIED", level="security", action_detail=what,
            capability=cap, reason="least privilege: capability not granted")
        raise CapabilityViolation(
            f"operation requires the `{cap}` capability, which was not granted"
            + (f" for {what}" if what else ""),
            pos,
            capability=cap, operation=what, audit_event=rec.event_id,
            hint=f"add `grant {cap}` to the module header, or pass "
                 f"--allow {cap} on the command line",
        )

    # ------------------------------------------------------------------
    # checkpointing (spec section 11)
    # ------------------------------------------------------------------
    def capture_checkpoint(self, label: str = "",
                           authorization: str = "runtime") -> Any:
        state = self.state_provider()
        cp = self.checkpoints.capture(state, label=label,
                                      authorization=authorization)
        self.stats.checkpoints += 1
        self.audit.record(
            "CHECKPOINT_CAPTURED", level="recovery", checkpoint=cp.id,
            label=cp.label, state_version=cp.state_version,
            integrity=cp.integrity, bindings=len(cp.state))
        return cp

    def apply_restored_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        applied = self.state_applier(state)
        self.audit.record("STATE_RESTORED", level="recovery",
                          bindings=len(applied))
        return applied

    # ------------------------------------------------------------------
    # recovery support (spec section 10)
    # ------------------------------------------------------------------
    def journal_safe_event(self, name: str, payload: Any = None) -> None:
        """Record an event that recovery is permitted to replay."""
        self.journal.append({"name": name, "payload": payload,
                             "at": self.now()})

    def replay_safe_events(self) -> List[Dict[str, Any]]:
        events = list(self.journal)
        for ev in events:
            self.audit.record("SAFE_EVENT_REPLAYED", level="recovery",
                              event=ev["name"])
        return events

    def restart_component(self, name: str) -> None:
        self.components[name] = self.components.get(name, 0) + 1
        self.audit.record("COMPONENT_RESTARTED", level="recovery",
                          component=name, restarts=self.components[name])

    def failover(self, name: str, target: str) -> None:
        entry = {"component": name, "target": target, "at": self.now()}
        self.failovers.append(entry)
        self.audit.record("FAILOVER", level="recovery", component=name,
                          target=target)

    def notify_operator(self, name: str, message: str) -> None:
        alert = {"component": name, "message": message, "at": self.now()}
        self.operator_alerts.append(alert)
        self.audit.record("OPERATOR_ALERT", level="recovery", component=name,
                          reason=message)

    # ------------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        self.stats.instructions += n
        if self.stats.instructions > self.max_steps:
            raise GamaRuntimeFault(
                "StepLimitExceeded",
                f"execution exceeded {self.max_steps} instructions",
                context={"limit": self.max_steps},
            )

    # ------------------------------------------------------------------
    # language-component hooks used by lowered GIR
    # ------------------------------------------------------------------
    def record_stage(self, name: str, args: List[Any]) -> None:
        """Record an operation-graph stage (spec section 3)."""
        self.stages.append({"name": name, "arity": len(args), "at": self.now()})

    def send_to_agent(self, agent: str, message: Any) -> None:
        self.agent_messages.append({"agent": agent, "message": message,
                                    "at": self.now()})
        self.audit.record("AGENT_MESSAGE", level="info", agent=agent)

    def evaluate_policy(self, name: str, rules: List[Any]) -> Any:
        from .values import GRecord
        matched: List[str] = []
        for rule in rules:
            kind = rule.get("kind")
            value = rule.get("value")
            text = rule.get("text", "")
            if kind == "audit":
                matched.append(f"audit:{text}")
                continue
            if kind == "deny" and bool(value):
                matched.append("deny")
                self.audit.record("POLICY_DENY", level="policy", policy=name,
                                  reason=text)
                return GRecord("PolicyDecision", {
                    "allow": False, "reason": text or "denied by policy",
                    "matched": matched, "policy": name})
            if kind == "allow" and bool(value):
                matched.append("allow")
            if kind == "require" and not bool(value):
                matched.append("require-failed")
                self.audit.record("POLICY_DENY", level="policy", policy=name,
                                  reason=text)
                return GRecord("PolicyDecision", {
                    "allow": False,
                    "reason": text or "a required condition did not hold",
                    "matched": matched, "policy": name})
        allow = "allow" in matched or (
            "require-failed" not in matched and not any(
                r.get("kind") == "allow" for r in rules))
        self.audit.record("POLICY_DECISION", level="policy", policy=name,
                          allow=allow, matched=matched)
        return GRecord("PolicyDecision", {
            "allow": allow,
            "reason": "matched an allow rule" if allow
            else "no allow rule matched",
            "matched": matched, "policy": name})

    def begin_transaction(self, name: str) -> None:
        self.transactions[name] = {"state": "open", "at": self.now()}
        self.audit.record("TRANSACTION_BEGIN", level="info", transaction=name)

    def commit_transaction(self, name: str) -> None:
        entry = self.transactions.setdefault(name, {})
        entry.update({"state": "committed", "committed_at": self.now()})
        self.audit.record("TRANSACTION_COMMIT", level="info", transaction=name)
        self.capture_checkpoint(f"{name}-commit", authorization=f"tx:{name}")

    def abort_transaction(self, name: str, reason: str = "") -> None:
        entry = self.transactions.setdefault(name, {})
        entry.update({"state": "aborted", "reason": reason,
                      "aborted_at": self.now()})
        self.audit.record("TRANSACTION_ABORT", level="recovery",
                          transaction=name, reason=reason or None)

    def call_value(self, value: Any, args: tuple) -> Any:
        """Invoke a first-class function value (set by the VM)."""
        if self._call_hook is None:
            raise GamaRuntimeFault(
                "NoCallable", "no interpreter is attached to this context")
        return self._call_hook(value, args)

    def describe(self) -> Dict[str, Any]:
        return {
            "grants": sorted(self.grants),
            "deterministic": self.deterministic,
            "program_version": self.program_version,
            "audit": self.audit.summary(),
            "checkpoints": len(self.checkpoints.history),
            "stats": self.stats.to_dict(),
            "operator_alerts": len(self.operator_alerts),
            "failovers": len(self.failovers),
            "stages": len(self.stages),
            "agent_messages": len(self.agent_messages),
            "transactions": len(self.transactions),
            "roles": list(self.roles),
        }
