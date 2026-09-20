"""Benchmarking (audit priority 8), under the constraint of spec section 43.

Section 43 forbids performance claims, and that forbids the usual shape of a
benchmark tool: one that prints a number and lets the reader infer "fast".  The
spec also says, of the runtime supervisor, that it should be able to "report
throughput, latency, scaling efficiency, and tail latency" -- so measuring is
wanted.  The resolution is that this module measures and reports, and never
concludes.

Three rules make that real rather than a disclaimer:

* **A number without its conditions is not a measurement.**  Every report carries
  the machine, the interpreter version, the optimisation level and the load, and
  `compare` refuses to compare two runs whose conditions differ.
* **One sample is an anecdote.**  Percentiles, not means: a mean hides exactly
  the tail that matters, and the spec asks for tail latency by name.
* **The work is counted, not timed away.**  Instruction counts come from the VM
  and do not depend on how busy the machine is, so they are reported alongside
  the timings and are the figures to trust when the two disagree.

Nothing here says what is fast, what got faster, or what is faster than what.
That is not modesty; it is the specification.
"""

from __future__ import annotations

import io
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..driver import compile_source, declared_grants, execute, find_entry
from ..runtime.context import Context

DISCLAIMER = (
    "These are measurements of this machine at this moment. Gama-G makes no "
    "performance claim: a number here describes one run, not a property of the "
    "language, and comparing it against another language or another machine is "
    "not supported by this tool."
)


@dataclass
class Stats:
    """A set of observations, described by their distribution."""

    samples: List[float] = field(default_factory=list)
    unit: str = "ms"

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def minimum(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def maximum(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples) if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    def percentile(self, fraction: float) -> float:
        """The value at `fraction` of the way through the sorted samples.

        Nearest-rank, which is what a tail latency figure should be: the value
        that `fraction` of the runs came in at or below.  Interpolating between
        two samples would invent a run that never happened.
        """
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = max(0, min(len(ordered) - 1,
                           int(round(fraction * len(ordered) + 0.5)) - 1))
        return ordered[index]

    def render(self, label: str) -> str:
        if not self.samples:
            return f"{label}: no samples"
        return (f"{label}: min {self.minimum:.3f} p50 {self.percentile(0.50):.3f} "
                f"p95 {self.percentile(0.95):.3f} "
                f"p99 {self.percentile(0.99):.3f} max {self.maximum:.3f} "
                f"mean {self.mean:.3f} sd {self.stdev:.3f} "
                f"({self.count} samples, {self.unit})")


@dataclass
class Conditions:
    """Everything a reader needs in order to know what a number describes."""

    machine: str = ""
    processor: str = ""
    python: str = ""
    gama_g: str = ""
    gir: str = ""
    profile: str = ""
    opt_level: int = 0
    target: str = ""

    @classmethod
    def current(cls, *, profile: str = "", opt_level: int = 0,
                target: str = "interpreter") -> "Conditions":
        from .. import GIR_VERSION, __version__
        return cls(
            machine=platform.machine(),
            processor=platform.processor() or platform.machine(),
            python=".".join(str(part) for part in sys.version_info[:3]),
            gama_g=__version__,
            gir=GIR_VERSION,
            profile=profile,
            opt_level=opt_level,
            target=target,
        )

    def key(self) -> Tuple:
        """The fields that must match before two runs can be compared."""
        return (self.machine, self.processor, self.python, self.gama_g,
                self.gir, self.profile, self.opt_level, self.target)

    def render(self) -> List[str]:
        return [
            f"conditions: {self.machine}, Python {self.python}, "
            f"Gama-G {self.gama_g} (GIR {self.gir})",
            f"            profile {self.profile or 'default'}, "
            f"-O{self.opt_level}, target {self.target}",
        ]


@dataclass
class Result:
    """One program, measured."""

    path: str
    wall: Stats = field(default_factory=Stats)
    instructions: Stats = field(default_factory=Stats)
    conditions: Conditions = field(default_factory=Conditions)
    compiled: bool = False
    faulted: str = ""
    output_bytes: int = 0
    problems: List[str] = field(default_factory=list)
    #: What the interquartile spread says about the timings: a wide one means
    #: the machine was busy and the timings should not be read closely.
    noise: str = ""

    def render(self) -> List[str]:
        out = [self.path]
        if not self.compiled:
            out.append("  did not compile" +
                       (f": {self.problems[0]}" if self.problems else ""))
            return out
        if self.faulted:
            out.append(f"  the program faults with {self.faulted}, so it is "
                       f"measuring the fault path")
        out.append("  " + self.wall.render("wall clock"))
        if self.instructions.samples:
            out.append("  " + self.instructions.render("instructions"))
        if self.noise:
            out.append(f"  {self.noise}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "compiled": self.compiled,
            "faulted": self.faulted,
            "output_bytes": self.output_bytes,
            "wall_ms": _stats_dict(self.wall),
            "instructions": _stats_dict(self.instructions),
            "conditions": {
                "machine": self.conditions.machine,
                "python": self.conditions.python,
                "gama_g": self.conditions.gama_g,
                "gir": self.conditions.gir,
                "profile": self.conditions.profile,
                "opt_level": self.conditions.opt_level,
                "target": self.conditions.target,
            },
            "problems": self.problems,
            "disclaimer": DISCLAIMER,
        }


def _stats_dict(stats: Stats) -> Dict[str, float]:
    if not stats.samples:
        return {}
    return {"count": stats.count, "min": stats.minimum,
            "p50": stats.percentile(0.50), "p95": stats.percentile(0.95),
            "p99": stats.percentile(0.99), "max": stats.maximum,
            "mean": stats.mean, "stdev": stats.stdev}


#: Below this many milliseconds, `perf_counter` resolution and process noise
#: dominate the measurement, and a wide spread says something about the clock
#: rather than about the machine's load.  Stated as a constant so the threshold
#: is visible rather than implied by a magic number in a comparison.
RESOLUTION_FLOOR_MS = 1.0


def _noise_note(stats: Stats) -> str:
    """Say when the timings should not be read closely.

    Two different situations get two different sentences, because they have
    different causes.  A wide spread around a sub-millisecond median is the
    timer's resolution, not a busy machine: reporting it as "the machine was
    busy" would be a wrong explanation of a real observation.  A wide spread
    around a slower median really does mean the load is in the number.
    """
    if stats.count < 3 or stats.median <= 0:
        return ""
    spread = (stats.percentile(0.95) - stats.percentile(0.50)) / stats.median
    if stats.median < RESOLUTION_FLOOR_MS:
        return (f"the median is under {RESOLUTION_FLOOR_MS:.0f} ms, so timer "
                f"resolution and per-run bookkeeping are a large part of these "
                f"figures; raise --repeats or measure a program that does more "
                f"work if the difference matters")
    if spread > 1.0:
        return (f"the p95 is {spread:.1f}x the median, so this machine was busy "
                f"and these timings describe the load as much as the program")
    return ""


def measure(source: str, path: str = "<program>", *, repeats: int = 7,
            warmup: int = 1, profile: str = "standard", opt_level: int = 1,
            grants: Sequence[str] = (), trust_declarations: bool = True,
            timeout: float = 300.0) -> Result:
    """Run one program `repeats` times and describe what happened.

    A warmup run is discarded by default.  The first run of anything in Python
    pays import and cache costs that later runs do not, and including it would
    describe the interpreter's startup rather than the program.
    """
    result = Result(path=path, conditions=Conditions.current(
        profile=profile, opt_level=opt_level, target="interpreter"))
    started = time.perf_counter()

    try:
        compilation = compile_source(source, path, profile=profile,
                                     opt_level=opt_level, grants=tuple(grants))
    except Exception as exc:                        # noqa: BLE001
        result.problems.append(f"{type(exc).__name__}: {exc}")
        return result
    if not compilation.ok or compilation.program is None:
        result.problems = [d.message for d in
                           (compilation.bag.diagnostics if compilation.bag else [])]
        return result
    result.compiled = True

    authority = set(grants)
    if trust_declarations:
        authority |= declared_grants(compilation)
    entry = find_entry(compilation) or "main"

    for run_index in range(max(0, warmup) + max(1, repeats)):
        if time.perf_counter() - started > timeout:
            result.problems.append(
                f"stopped after {timeout:.0f}s with "
                f"{run_index}/{warmup + repeats} runs complete")
            break
        buffer = io.StringIO()
        context = Context(grants=authority, stdout=buffer, seed=0,
                          deterministic=True)
        before = time.perf_counter()
        execution = execute(compilation, entry=entry, context=context, grants=())
        elapsed = (time.perf_counter() - before) * 1000.0
        if run_index < warmup:
            continue
        if execution.fault is not None and not result.faulted:
            result.faulted = type(execution.fault).__name__
        result.output_bytes = len(buffer.getvalue().encode("utf-8"))
        result.wall.samples.append(elapsed)
        result.instructions.samples.append(float(context.stats.instructions))
    result.instructions.unit = "instructions"

    result.noise = _noise_note(result.wall)
    return result


@dataclass
class Comparison:
    """Two measurements, and whether comparing them was legitimate."""

    first: Result
    second: Result
    comparable: bool = False
    reasons: List[str] = field(default_factory=list)

    def render(self) -> List[str]:
        out = ["comparison"]
        for result in (self.first, self.second):
            out.extend("  " + line for line in result.render())
            out.append("")
        if not self.comparable:
            out.append("  the two runs are not comparable:")
            for reason in self.reasons:
                out.append(f"    - {reason}")
            out.append("  no ratio is printed, because a ratio between "
                       "incomparable runs is a made-up number")
            return out
        left = self.first.wall.median
        right = self.second.wall.median
        if left > 0:
            out.append(f"  the second run's median wall clock is "
                       f"{right / left:.2f}x the first's")
        out.append("  this ratio describes these two runs on this machine; it "
                   "is not a property of either program")
        return out


def compare(first: Result, second: Result) -> Comparison:
    """Compare two measurements, refusing when the conditions differ."""
    comparison = Comparison(first=first, second=second)
    reasons: List[str] = []
    if first.conditions.key() != second.conditions.key():
        reasons.append(
            f"the conditions differ: {first.conditions.machine}/"
            f"{first.conditions.profile}/-O{first.conditions.opt_level}/"
            f"{first.conditions.target} versus "
            f"{second.conditions.machine}/{second.conditions.profile}/"
            f"-O{second.conditions.opt_level}/{second.conditions.target}")
    if not first.compiled or not second.compiled:
        reasons.append("a program that did not compile has no timing")
    if first.faulted != second.faulted:
        reasons.append(
            f"one run faults ({first.faulted or 'no'}) and the other "
            f"({second.faulted or 'no'}); a fault path is a different program")
    if first.wall.count < 3 or second.wall.count < 3:
        reasons.append("fewer than three samples on one side")
    comparison.reasons = reasons
    comparison.comparable = not reasons
    return comparison


@dataclass
class Report:
    """Every measurement from one invocation."""

    results: List[Result] = field(default_factory=list)

    def render(self) -> List[str]:
        out: List[str] = []
        if not self.results:
            out.append("nothing to report")
            return out
        out.extend(self.results[0].conditions.render())
        out.append("")
        for result in self.results:
            out.extend(result.render())
            out.append("")
        out.append(DISCLAIMER)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"results": [r.to_dict() for r in self.results]}


def bench_paths(paths: Sequence[str], **kwargs: Any) -> Report:
    report = Report()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        report.results.append(measure(source, path, **kwargs))
    return report


def bench_native(paths: Sequence[str], *, repeats: int = 7, warmup: int = 1,
                 build_dir: str = ".ggbuild", **kwargs: Any) -> Report:
    """Measure the native backend the same way, for programs it can compile.

    Kept separate from `bench_paths` rather than folded into it, because the two
    measure different things: the interpreter's wall clock includes the VM, and
    the native figure includes process startup.  Presenting them in one table
    would invite exactly the comparison section 43 forbids.
    """
    from ..backend import native as native_backend

    report = Report()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        compilation = compile_source(source, path,
                                     profile=kwargs.get("profile", "standard"))
        result = Result(path=f"{path} (native)",
                        conditions=Conditions.current(
                            profile=kwargs.get("profile", "standard"),
                            opt_level=kwargs.get("opt_level", 1),
                            target="native"))
        if not compilation.ok or compilation.program is None:
            result.problems = ["did not compile"]
            report.results.append(result)
            continue
        build = native_backend.build(compilation.program, path,
                                     build_dir=build_dir)
        if not build.ok:
            result.problems = [p.render() for p in build.problems] or \
                [build.stderr.splitlines()[0] if build.stderr else "build failed"]
            report.results.append(result)
            continue
        result.compiled = True
        for run_index in range(max(0, warmup) + max(1, repeats)):
            run = native_backend.run(build.exe_path)
            if run_index < warmup:
                continue
            if run.returncode != 0 and not result.faulted:
                result.faulted = f"exit {run.returncode}"
            result.output_bytes = len(run.stdout.encode("utf-8"))
            result.wall.samples.append(run.elapsed_ms)
        result.noise = _noise_note(result.wall)
        report.results.append(result)
    return report
