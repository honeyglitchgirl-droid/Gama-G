"""The fuzzing campaign (audit priority 9).

The engine's job is small: produce inputs, run the oracles, and keep one
reproducer per *distinct* failure.  Keeping one per distinct failure is the part
that makes this usable -- a campaign over a broken invariant can produce
thousands of hits for one bug, and a report with ten thousand lines is a report
nobody reads.

Reproducers are written to disk with the exact source, the seed that produced
them and the violation they triggered, so a bug found at 3am by a campaign can
become a regression test in the morning instead of a story.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import generator, oracle


@dataclass
class Case:
    """One input the campaign tried."""

    source: str
    origin: str            # where it came from: a path, or "generated"
    kind: str              # how it was produced
    detail: str = ""       # what was done to it
    verdict: str = oracle.VERDICT_REJECTED
    violations: List[oracle.Violation] = field(default_factory=list)

    @property
    def compiled(self) -> bool:
        """Whether the front end accepted this input."""
        return self.verdict == oracle.VERDICT_COMPILED

    @property
    def clean(self) -> bool:
        """Whether every invariant held.  Not the same as `compiled`: a
        correctly rejected program is clean too."""
        return not self.violations


@dataclass
class Campaign:
    """The result of a run."""

    rounds: int = 0
    seed: int = 0
    cases: List[Case] = field(default_factory=list)
    violations: List[oracle.Violation] = field(default_factory=list)
    elapsed_ms: float = 0.0
    #: Cases the front end accepted.  Cases it correctly refused are counted
    #: separately: a refusal is an answer, not a failure, and mixing the two
    #: makes a report say less than the campaign knows.
    compiled_count: int = 0
    rejected_count: int = 0
    #: Cases where something escaped the handled channel -- the front end
    #: crashed, or an invariant reported a crash severity.
    crashed_count: int = 0
    #: Cases where every invariant held.
    clean_count: int = 0
    save_dir: str = ""

    #: One violation per distinct signature, which is what a report should show.
    distinct: Dict[str, oracle.Violation] = field(default_factory=dict)

    @property
    def by_severity(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for violation in self.violations:
            out[violation.severity] = out.get(violation.severity, 0) + 1
        return out

    @property
    def crashes(self) -> List[oracle.Violation]:
        return [v for v in self.distinct.values() if v.severity == "crash"]

    @property
    def wrong(self) -> List[oracle.Violation]:
        return [v for v in self.distinct.values()
                if v.severity in ("wrong", "inconsistent")]

    def summary(self) -> str:
        return (f"{self.rounds} rounds in {self.elapsed_ms:.0f} ms: "
                f"{self.compiled_count} compiled, "
                f"{self.rejected_count} rejected, "
                f"{self.clean_count} clean, "
                f"{len(self.violations)} violation(s) across "
                f"{len(self.distinct)} distinct failure(s)")

    def render(self, *, limit: int = 12) -> List[str]:
        out = ["fuzzing report", "  " + self.summary()]
        if self.by_severity:
            counts = ", ".join(f"{k}: {v}"
                               for k, v in sorted(self.by_severity.items()))
            out.append(f"  by severity -- {counts}")
        if not self.distinct:
            out.append("")
            out.append("  no invariant was broken")
            return out

        ordered = sorted(self.distinct.values(),
                         key=lambda v: ({"crash": 0, "wrong": 1,
                                         "inconsistent": 2}.get(v.severity, 3),
                                        v.invariant))
        for violation in ordered[:limit]:
            out.append("")
            out.extend("  " + line for line in violation.render())
            hits = sum(1 for v in self.violations
                       if v.signature == violation.signature)
            if hits > 1:
                out.append(f"    ({hits} inputs produced this same failure)")
        if len(ordered) > limit:
            out.append("")
            out.append(f"  ... and {len(ordered) - limit} more distinct failure(s)")
        if self.save_dir:
            out.append("")
            out.append(f"  reproducers written to {self.save_dir}")
        return out


def _save(case: Case, violation: oracle.Violation, directory: str,
          index: int) -> str:
    os.makedirs(directory, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                   for ch in violation.invariant)[:48]
    extension = ".gg" if case.source.strip() else ".txt"
    path = os.path.join(directory, f"{index:04d}-{safe}{extension}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"// reproducer for: {violation.invariant}\n")
        handle.write(f"// severity: {violation.severity}\n")
        handle.write(f"// detail: {violation.detail}\n")
        if case.origin:
            handle.write(f"// origin: {case.origin}\n")
        if case.detail:
            handle.write(f"// produced by: {case.kind} -- {case.detail}\n")
        handle.write("//\n")
        handle.write(case.source)
    return path


def run(rounds: int = 200, *, seed: int = 0,
        corpus_paths: Sequence[str] = (),
        deep: bool = True,
        save_dir: str = "",
        timeout: float = 10.0,
        max_steps: int = oracle.FUZZ_MAX_STEPS,
        mutation_bias: float = 0.5,
        progress: Optional[Callable[[int, int], None]] = None) -> Campaign:
    """Run a campaign.

    `mutation_bias` is the fraction of rounds spent corrupting a known-good
    program rather than generating one.  Almost-valid input finds different bugs
    from random input, and half-and-half is a reasonable default.
    """
    rng = random.Random(seed)
    campaign = Campaign(rounds=rounds, seed=seed, save_dir=save_dir)
    corpus = generator.corpus(list(corpus_paths))
    started = time.perf_counter()

    for index in range(rounds):
        if progress:
            progress(index, rounds)

        use_corpus = corpus and rng.random() < mutation_bias
        if use_corpus:
            origin, source = rng.choice(corpus)
            mutation = generator.mutate(source, rng, origin)
            case = Case(source=mutation.source, origin=origin,
                        kind=mutation.kind, detail=mutation.detail)
        else:
            made = generator.generate(rng)
            case = Case(source=made.source, origin="generated",
                        kind=made.kind, detail="; ".join(made.notes))

        try:
            result = oracle.run_checks(case.source, path=case.origin,
                                       origin=case.origin, kind=case.kind,
                                       deep=deep, timeout=timeout,
                                       max_steps=max_steps)
            case.verdict = result.verdict
            violations = result.violations
        except BaseException as exc:                 # noqa: BLE001
            # The oracle itself must be total.  If a check raised, that is a
            # finding, not a reason to stop the campaign.
            case.verdict = oracle.VERDICT_CRASHED
            violations = [oracle.Violation(
                "the-oracle-raised", "crash",
                f"{type(exc).__name__}: {exc}", case.source, case.origin,
                case.kind, traceback=oracle._tb(exc))]

        case.violations = violations
        campaign.cases.append(case)
        campaign.violations.extend(violations)
        for violation in violations:
            campaign.distinct.setdefault(violation.signature, violation)
        if case.verdict == oracle.VERDICT_COMPILED:
            campaign.compiled_count += 1
        elif case.verdict == oracle.VERDICT_REJECTED:
            campaign.rejected_count += 1
        if case.verdict == oracle.VERDICT_CRASHED or any(
                v.severity == "crash" for v in violations):
            campaign.crashed_count += 1
        if not violations:
            campaign.clean_count += 1

    campaign.elapsed_ms = (time.perf_counter() - started) * 1000.0

    if save_dir and campaign.distinct:
        seen: Dict[str, int] = {}
        for position, violation in enumerate(sorted(
                campaign.distinct.values(), key=lambda v: v.signature)):
            example = next((c for c in campaign.cases
                            if c.violations and
                            c.violations[0].signature == violation.signature),
                           None)
            if example is None:
                continue
            seen[violation.signature] = position
            _save(example, violation, save_dir, position)
    return campaign


def default_corpus(root: str) -> List[str]:
    """Every shipped example, which is the corpus worth mutating."""
    return [path for path, _ in generator.example_corpus(root)]
