"""Recovery and transition semantics for the Gama-G core.

Spec section 10 defines six recovery levels and says two things that are easy to
write and hard to honour:

    The runtime must never silently invent medical or financial state during
    recovery.
    Every recovery action should be observable and auditable.

Section 18 adds that a failure before commit puts the work into a recovery state
rather than pretending it completed.

Audit priority 6 asks that transitions and recovery be represented *directly*.
In v0.3 they were not: a transition lowered to a ``COPY`` into a slot and
recovery lowered to a ``FAULT``, so by the time a backend saw the program, the
fact that a write was a state transition under authority -- and the fact that a
fault had a policy behind it -- had both been flattened into generic machine
steps.

GIR already has the instructions for both (``TRANSACTION``, ``PROTECTED``,
``CHECKPOINT``), and :mod:`gamag.runtime.recovery` already implements the six
levels. This module is the semantics that connects the core's syntax to them, and
the vocabulary is imported rather than restated: there is one list of recovery
levels in this project, and it lives in the runtime.

The two rules that carry the spec's requirements:

**A policy may only escalate.** :meth:`RecoveryPolicy.validate` refuses a policy
whose levels descend, because ``escalate`` followed by ``retry`` is not an
escalation policy -- it is a loop, and a loop in a recovery path is how a service
retries something that will never succeed.

**A restore must have something to restore.** ``restore`` and ``replay`` at level
2 return the program to recorded state. If the step names a checkpoint, that
checkpoint must have been declared with ``checkpoint``; a name that was never
declared is state the runtime would have to invent, which is exactly what section
10 forbids. A step with no named target resolves to the checkpoint the runtime
captures on entering the protected region, which is genuine recorded state, and
:meth:`RecoveryPolicy.resolves_to` says so explicitly rather than leaving it
implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..runtime.recovery import (ACTION_LEVELS, LEVEL_CHECKPOINT, LEVEL_NAMES,
                                LEVEL_RETRY)
from . import mir as M

#: The actions a policy may name. Imported from the runtime so that the core and
#: the machine that executes it cannot drift apart.
ACTIONS = frozenset(ACTION_LEVELS)

#: The step whose target the runtime supplies itself: it captures a checkpoint on
#: entering a protected region, so a bare `restore` has real recorded state to
#: return to.
ENTRY_CHECKPOINT = "{component}-entry"


def level_of(action: str) -> int:
    """The spec section 10 level of an action, or -1 if it is not one."""
    return ACTION_LEVELS.get(action, -1)


@dataclass
class RecoveryPolicy:
    """An intent's declared escalation policy, and what it is allowed to do.

    ``checkpoints`` are the ones the intent declared. A policy step that names a
    checkpoint outside that set is refused, because restoring to state that was
    never recorded is inventing it.
    """

    steps: List[M.RecoveryStep] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    component: str = ""
    problems: List[Tuple[str, Optional[object], str, str]] = field(
        default_factory=list)

    # ------------------------------------------------------------------
    def validate(self) -> List[Tuple[str, Optional[object], str, str]]:
        """Return ``(code, pos, message, help)`` for everything wrong here."""
        self.problems = []
        if not self.steps:
            return self.problems

        def problem(code: str, pos, message: str, help_text: str) -> None:
            self.problems.append((code, pos, message, help_text))

        last_level = -1
        last_action = ""
        for step in self.steps:
            level = level_of(step.action)
            if level < 0:
                problem(
                    "E-unknown-recovery", step.pos,
                    f"`{step.action}` is not a recovery action",
                    "the actions are the specification's own: "
                    + ", ".join(f"`{a}`" for a in sorted(ACTIONS)))
                continue
            if level < last_level:
                problem(
                    "E-recovery-unordered", step.pos,
                    f"`{step.action}` (level {level}, "
                    f"{LEVEL_NAMES[level]}) follows `{last_action}` "
                    f"(level {last_level}, {LEVEL_NAMES[last_level]})",
                    "a recovery policy escalates; it does not step back down. "
                    "Reordering these would let the program retry something it "
                    "had already escalated")
            last_level, last_action = level, step.action

            if level == LEVEL_CHECKPOINT:
                named = self._named_checkpoint(step)
                if named and named not in self.checkpoints:
                    problem(
                        "E-unknown-checkpoint", step.pos,
                        f"`{step.raw or step.action}` would restore to "
                        f"`{named}`, which this intent never recorded",
                        "spec section 10: the runtime must never silently invent "
                        "state during recovery. Declare it with a `checkpoint` "
                        "clause, or restore to the region entry point by "
                        "writing the action with no target"
                        + (f" -- this intent declares: "
                           f"{', '.join(self.checkpoints)}"
                           if self.checkpoints else ""))
        return self.problems

    def _named_checkpoint(self, step: M.RecoveryStep) -> str:
        """The checkpoint a step names, if it names one.

        ``restore checkpoint baseline`` names ``baseline``. A bare ``restore``
        names nothing and resolves to the region entry point.
        """
        words = (step.target or "").split()
        if not words:
            return ""
        if words[0] == "checkpoint" and len(words) > 1:
            return words[1]
        if len(words) == 1 and words[0] in self.checkpoints:
            return words[0]
        return ""

    def resolves_to(self, step: M.RecoveryStep) -> str:
        """What recorded state a level-2 step returns to, stated explicitly."""
        if level_of(step.action) != LEVEL_CHECKPOINT:
            return ""
        named = self._named_checkpoint(step)
        if named:
            return named
        return ENTRY_CHECKPOINT.replace(
            "{component}", self.component or "component")

    # ------------------------------------------------------------------
    def to_steps(self) -> List[Dict[str, object]]:
        """The step records the runtime's protected region expects.

        This is the direct representation priority 6 asks for: the policy the
        program declared becomes the policy the machine executes, with no
        intermediate encoding as a fault message or a jump target.
        """
        out: List[Dict[str, object]] = []
        for step in self.steps:
            if level_of(step.action) < 0:
                continue
            out.append({
                "action": step.action,
                "count": step.count,
                "target": self.resolves_to(step) or step.target,
                "raw": step.raw or step.action,
            })
        return out

    @property
    def highest_level(self) -> int:
        return max((level_of(s.action) for s in self.steps), default=-1)

    @property
    def bounded(self) -> bool:
        """Whether every step that can repeat has a bound.

        A ``retry`` with no count is unbounded, and unbounded recovery is the
        same defect as unbounded refinement wearing different clothes.
        """
        return all(step.count is not None
                   for step in self.steps
                   if level_of(step.action) == LEVEL_RETRY)

    def render(self) -> List[str]:
        if not self.steps:
            return []
        lines = ["  declared policy, in escalation order:"]
        for step in self.steps:
            level = level_of(step.action)
            name = LEVEL_NAMES.get(level, "not a recovery action")
            bound = f" within {step.count} rounds" if step.count else ""
            target = f" {step.target}" if step.target else ""
            lines.append(f"    level {level} ({name}): "
                         f"{step.action}{bound}{target}")
            # a bare `restore` resolves to the region entry point; saying so is
            # the difference between recorded state and invented state
            resolved = self.resolves_to(step)
            if resolved and not self._named_checkpoint(step):
                lines.append(f"        restores recorded state: {resolved}")
        return lines


def policy_of(intent: M.IntentGraph) -> RecoveryPolicy:
    """The policy an intent declares, with its component name filled in."""
    return RecoveryPolicy(steps=list(intent.recovery),
                          checkpoints=list(intent.checkpoints),
                          component=intent.name or "intent")
