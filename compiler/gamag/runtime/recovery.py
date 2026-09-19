"""Bounded recovery engine (spec section 10).

Self-healing in Gama-G is "a language/runtime capability, not an
unrestricted AI decision maker".  Recovery therefore follows an explicit,
ordered, bounded policy declared in source, escalates through the six
recovery levels defined by the specification, and emits an audit record for
every action taken -- "Every recovery action should be observable and
auditable."

Levels (spec section 10):
    LEVEL 0: local retry
    LEVEL 1: resource reset
    LEVEL 2: state checkpoint restore
    LEVEL 3: component restart
    LEVEL 4: failover
    LEVEL 5: operator escalation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .checkpoint import CheckpointRejected

LEVEL_RETRY = 0
LEVEL_RESET = 1
LEVEL_CHECKPOINT = 2
LEVEL_RESTART = 3
LEVEL_FAILOVER = 4
LEVEL_ESCALATE = 5

LEVEL_NAMES = {
    LEVEL_RETRY: "local retry",
    LEVEL_RESET: "resource reset",
    LEVEL_CHECKPOINT: "state checkpoint restore",
    LEVEL_RESTART: "component restart",
    LEVEL_FAILOVER: "failover",
    LEVEL_ESCALATE: "operator escalation",
}

# Maps a declared recovery action onto its recovery level.
ACTION_LEVELS: Dict[str, int] = {
    "retry": LEVEL_RETRY,
    "reset": LEVEL_RESET,
    "reconnect": LEVEL_RESET,
    "restore": LEVEL_CHECKPOINT,
    "replay": LEVEL_CHECKPOINT,
    "restart": LEVEL_RESTART,
    "failover": LEVEL_FAILOVER,
    "alert": LEVEL_ESCALATE,
    "escalate": LEVEL_ESCALATE,
}

# Actions that terminate the recovery attempt rather than retrying work.
TERMINAL_ACTIONS = {"alert", "escalate"}


@dataclass
class RecoveryStepSpec:
    """One parsed line of a ``recover`` block."""

    action: str
    count: Optional[int] = None
    target: str = ""
    raw: str = ""

    @property
    def level(self) -> int:
        return ACTION_LEVELS.get(self.action, LEVEL_RETRY)


@dataclass
class RecoveryAction:
    step: RecoveryStepSpec
    level: int
    attempted: bool = False
    succeeded: bool = False
    detail: str = ""
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.step.action,
            "raw": self.step.raw or self.step.action,
            "target": self.step.target,
            "count": self.step.count,
            "level": self.level,
            "level_name": LEVEL_NAMES.get(self.level, str(self.level)),
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "detail": self.detail,
            "event_id": self.event_id,
        }


@dataclass
class RecoveryOutcome:
    recovered: bool = False
    level: int = -1
    attempts: int = 0
    actions: List[RecoveryAction] = field(default_factory=list)
    final_error: Optional[str] = None
    value: Any = None
    escalated: bool = False

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.level, "none")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recovered": self.recovered,
            "level": self.level,
            "level_name": self.level_name,
            "attempts": self.attempts,
            "escalated": self.escalated,
            "final_error": self.final_error,
            "actions": [a.to_dict() for a in self.actions],
        }


class RecoveryEngine:
    """Executes a declared recovery policy against a protected region."""

    def __init__(self, context: Any):
        self.ctx = context

    def run(self, steps: List[RecoveryStepSpec], work: Callable[[], Any],
            *, name: str = "component",
            hooks: Optional[Dict[str, Callable[[], Any]]] = None) -> RecoveryOutcome:
        hooks = hooks or {}
        outcome = RecoveryOutcome()

        # First attempt of the protected region.
        outcome.attempts += 1
        try:
            outcome.value = work()
            outcome.recovered = True
            outcome.level = -1
            return outcome
        except CheckpointRejected as exc:
            last_error = "checkpoint rejected: " + "; ".join(exc.problems)
        except Exception as exc:                      # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

        for step in steps:
            action = RecoveryAction(step=step, level=step.level)
            outcome.actions.append(action)
            outcome.level = max(outcome.level, step.level)
            action.attempted = True

            detail, ok = self._apply(step, hooks, name)
            action.detail = detail
            action.succeeded = ok
            self._audit(name, step, action, ok)

            if step.action in TERMINAL_ACTIONS:
                outcome.escalated = True
                break

            if ok:
                attempts = step.count if step.count else 1
                for _ in range(max(1, attempts)):
                    outcome.attempts += 1
                    try:
                        outcome.value = work()
                        outcome.recovered = True
                        return outcome
                    except CheckpointRejected as exc:
                        last_error = "checkpoint rejected: " + "; ".join(exc.problems)
                    except Exception as exc:          # noqa: BLE001
                        last_error = f"{type(exc).__name__}: {exc}"

        # Policy exhausted without recovery: escalate to an operator.
        if not outcome.recovered:
            outcome.escalated = True
            outcome.level = max(outcome.level, LEVEL_ESCALATE)
            outcome.final_error = last_error
            esc = RecoveryStepSpec(action="escalate", target="operator",
                                   raw="escalate operator")
            act = RecoveryAction(step=esc, level=LEVEL_ESCALATE, attempted=True,
                                 succeeded=True,
                                 detail="recovery policy exhausted")
            outcome.actions.append(act)
            self._audit(name, esc, act, True,
                        extra={"final_error": last_error,
                               "exhausted": True})
        return outcome

    # ------------------------------------------------------------------
    def _apply(self, step: RecoveryStepSpec, hooks: Dict[str, Callable[[], Any]],
               name: str):
        """Perform one recovery action.  Returns (detail, succeeded)."""
        action = step.action
        hook = hooks.get(action)

        if action == "retry":
            return (f"retrying protected region up to {step.count or 1} time(s)",
                    True)

        if action in ("reset", "reconnect"):
            if hook is not None:
                hook()
            return (f"{action} completed for `{name}`", True)

        if action == "restore":
            # Never invent state: restore only genuinely recorded state.
            try:
                state = self.ctx.checkpoints.restore()
            except CheckpointRejected as exc:
                return ("refusing to restore: " + "; ".join(exc.problems), False)
            restored = self.ctx.apply_restored_state(state)
            return (f"restored {len(restored)} state binding(s) from checkpoint "
                    f"{self.ctx.checkpoints.latest().id}", True)

        if action == "replay":
            events = self.ctx.replay_safe_events()
            return (f"replayed {len(events)} journaled safe event(s)", True)

        if action == "restart":
            if hook is not None:
                hook()
            self.ctx.restart_component(name)
            return (f"component `{name}` restarted", True)

        if action == "failover":
            if hook is not None:
                hook()
            target = step.target or "standby"
            self.ctx.failover(name, target)
            return (f"failed over `{name}` to {target}", True)

        if action in ("alert", "escalate"):
            target = step.target or "operator"
            self.ctx.notify_operator(name, step.raw or action)
            return (f"operator notification raised ({target})", True)

        return (f"unknown recovery action `{action}`", False)

    def _audit(self, name: str, step: RecoveryStepSpec, action: RecoveryAction,
               ok: bool, extra: Optional[Dict[str, Any]] = None) -> None:
        fields = {
            "component": name,
            "recovery_action": step.raw or step.action,
            "level": action.level,
            "level_name": LEVEL_NAMES.get(action.level, str(action.level)),
            "succeeded": ok,
            "detail": action.detail,
        }
        if extra:
            fields.update(extra)
        rec = self.ctx.audit.record(
            "RECOVERY_ACTION", level="recovery",
            reason=f"bounded recovery of `{name}`", **fields)
        action.event_id = rec.event_id
