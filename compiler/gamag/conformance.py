"""Conformance checked against the specification, not against the compiler.

Every other test in this repository compares the toolchain with itself: the
interpreter against the native backend, a diagnostic against an expected code, a
behaviour against what it did last week.  That finds regressions and cannot find
a *misreading* -- if the compiler and the tests share a wrong belief about what
the specification says, they agree and both are wrong.

This module reads ``Gama-G_v1.0_Production_Specification.txt`` and checks the
toolchain against what it finds there:

* ``type-surface`` -- every type name in section 5's three lists has to be
  usable in a declaration, or be recorded as a known deviation with a reason.
* ``pipeline`` -- every stage in section 22 has to be either mapped to the part
  of the compiler that performs it, or recorded with the reason it is not.
* ``requirements`` -- every item in section 33's v1.0 checklist has to be either
  evidenced by a command that demonstrates it, or explicitly not claimed.  An
  item that quietly disappears from the checklist file is a failure, which is
  the point: the checklist is what stops a release from over-claiming.
* ``program`` / ``refused`` / ``fault`` -- behaviour the specification states
  (or the interpreter's documented fault model), with the section cited.
* ``deterministic`` -- the same source compiled twice produces the same output,
  which is spec section 1.3 and stage 14 of section 22.

The rule that makes this more than a second copy of the test suite: **a
deviation must be written down**.  A type the specification lists that the
compiler cannot express is a conformance failure *unless* an entry in
``conformance/deviations.json`` names it and explains it.  Silent gaps fail;
documented ones are reported and counted.  ``ggc conform --strict`` also fails
on the documented ones, for anyone who wants the tighter bar.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .backend import differential
from .semantic import types as T

#: Where the specification lives, and what it is called.  The suite is about a
#: *version* of the specification: a document that is not this one cannot be
#: checked with these expectations, and pretending otherwise would be the
#: loosest possible reading of "conformance".
SPEC_FILENAME = "Gama-G_v1.0_Production_Specification.txt"
SPEC_VERSION = "1.0"

#: The section headings the extractor needs.  Listed here rather than inline so
#: a failure names what went missing.
_REQUIRED_HEADINGS = ("Primitive types:", "Compound types:", "Domain types:",
                      "22. COMPILATION PIPELINE", "33. V1.0 PRODUCTION")


class SpecError(Exception):
    """The specification could not be read as the suite expects.

    Never caught into a pass: a suite that cannot find the document it checks
    against has checked nothing, and reporting that as success is the one
    failure mode worse than reporting a real one.
    """


@dataclass(frozen=True)
class SpecFacts:
    """What the specification states, extracted from its own text."""

    types: Dict[str, List[str]]          # section 5: group -> type names
    stages: List[str]                    # section 22: pipeline stages
    requirements: List[str]              # section 33: v1.0 checklist

    @property
    def all_types(self) -> List[str]:
        return [name for group in self.types.values() for name in group]


def _indented_block(lines: Sequence[str], header: str,
                    indent: int = 4) -> List[str]:
    """The indented lines under `header`, up to the first blank line."""
    out: List[str] = []
    started = False
    for line in lines:
        if line.strip() == header:
            started = True
            continue
        if not started:
            continue
        if not line.strip():
            if out:
                break
            continue
        if len(line) - len(line.lstrip()) >= indent:
            out.append(line.strip())
        else:
            break
    if not started:
        raise SpecError(f"the specification has no `{header}` block")
    if not out:
        raise SpecError(f"the `{header}` block is empty")
    return out


def _type_names(block: Sequence[str]) -> List[str]:
    """Type names from a section 5 list, joining declarations that wrap.

    `Tensor<T, Shape>` is written across two lines in the document, so the
    block is joined before the names are read out of it.  Reading the names
    rather than the lines is the difference between checking the specification
    and checking its line width.
    """
    joined = " ".join(block)
    names: List[str] = []
    for match in re.finditer(r"[A-Z][A-Za-z0-9]*\s*(<[^>]*>|\([^)]*\))?", joined):
        name = re.match(r"[A-Z][A-Za-z0-9]*", match.group(0))
        if name and name.group(0) not in names:
            names.append(name.group(0))
    return names


def _numbered_stages(lines: Sequence[str], heading: str) -> List[str]:
    out: List[str] = []
    for line in lines:
        if line.startswith("=========="):
            if out:
                break
            continue
        match = re.match(r"\s*(\d+)\.\s+(\S.*)", line)
        if match:
            out.append(match.group(2).strip())
    return out


def _checklist(lines: Sequence[str]) -> List[str]:
    """Section 33's `[ ]` items, with their wrapped continuation lines."""
    items: List[str] = []
    current: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("====="):
            if items:
                break
            continue
        if stripped.startswith("[ ]"):
            if current:
                items.append(current)
            current = stripped[3:].strip()
        elif current is not None and stripped:
            current += " " + stripped
    if current:
        items.append(current)
    return items


def extract(spec_text: str) -> SpecFacts:
    """Read the enumerations the suite checks out of the specification text."""
    for heading in _REQUIRED_HEADINGS:
        if heading not in spec_text:
            raise SpecError(
                f"the specification has no `{heading}`; this suite checks a "
                f"specific document and will not report on another one")
    lines = spec_text.splitlines()
    types = {
        group: _type_names(_indented_block(lines, f"{group} types:"))
        for group in ("Primitive", "Compound", "Domain")
    }
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("22. COMPILATION PIPELINE"))
    stages = _numbered_stages(lines[start + 1:], "22. COMPILATION PIPELINE")
    if len(stages) < 10:
        raise SpecError(f"section 22 lists {len(stages)} stages; expected the "
                        f"document's own list, not a shorter one")
    req_start = next(i for i, line in enumerate(lines)
                     if line.startswith("33. V1.0 PRODUCTION"))
    requirements = _checklist(lines[req_start:])
    if len(requirements) < 20:
        raise SpecError(f"section 33 lists {len(requirements)} requirements; "
                        f"expected at least 20")
    return SpecFacts(types=types, stages=stages, requirements=requirements)


def repo_root() -> str:
    """The checkout this module is running from.

    `compiler/gamag/conformance.py` is three levels below the repository root
    in a checkout.  In an installed wheel the same expression points at
    `site-packages`, which is why both lookups below also accept an explicit
    path and fall back to the working directory.
    """
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_spec_path(repo_root_path: Optional[str] = None) -> Optional[str]:
    """The specification to check against, or None if it is not to hand.

    Precedence: `$GAMA_SPEC`, the path given, then the file beside this
    checkout.  A wheel does not carry the specification -- it is a repository
    artifact -- so `ggc conform` from an installed copy needs `--spec`, and
    says so rather than passing.
    """
    root = repo_root_path or repo_root()
    for candidate in (os.environ.get("GAMA_SPEC"),
                      os.path.join(root, SPEC_FILENAME),
                      os.path.join(os.getcwd(), SPEC_FILENAME)):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# The suite on disk
# ---------------------------------------------------------------------------

def suite_dir(root: Optional[str] = None) -> str:
    """Where the claim files live.

    `$GAMA_CONFORMANCE` wins, then the checkout's own `conformance/`
    directory, then `./conformance` in the working directory, so an installed
    `ggc` can be pointed at a suite the user has.  The suite is deliberately
    *not* inside the wheel: it is written to be read next to the specification
    and the examples, and a copy inside an installed package would be one more
    thing to keep in step.
    """
    explicit = os.environ.get("GAMA_CONFORMANCE")
    if explicit:
        return explicit
    root = root or repo_root()
    candidate = os.path.join(root, "conformance")
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(os.getcwd(), "conformance")


@dataclass
class CaseResult:
    id: str
    spec: str
    kind: str
    claim: str
    ok: bool
    detail: str = ""
    #: Recorded, explained deviations that this case ran into.
    deviations: List[str] = field(default_factory=list)
    #: Requirements this case found answered but not evidenced (section 33).
    unmet: List[str] = field(default_factory=list)

    def render(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        line = f"  {mark} {self.id:<34} {self.spec:<9} {self.claim}"
        if self.detail:
            line += f"\n         {self.detail}"
        for deviation in self.deviations:
            line += f"\n         recorded deviation: {deviation}"
        return line


@dataclass
class Report:
    results: List[CaseResult]
    facts: Optional[SpecFacts] = None
    strict: bool = False
    #: Section 33 items that are not fully evidenced, so `--strict` can answer
    #: for them without re-reading the file the run was built from.
    unmet_requirements: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[CaseResult]:
        return [r for r in self.results if not r.ok]

    @property
    def strict_failures(self) -> List[str]:
        """Why `--strict` would fail, over and above a normal failure."""
        reasons = [f"{case}: {detail}" for case, detail in self.deviations]
        reasons += [f"section 33: {item}" for item in self.unmet_requirements]
        return reasons

    @property
    def deviations(self) -> List[Tuple[str, str]]:
        out = []
        for result in self.results:
            for deviation in result.deviations:
                out.append((result.id, deviation))
        return out

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_json(self) -> Dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "strict": self.strict,
            "ok": self.ok,
            "cases": [{"id": r.id, "spec": r.spec, "kind": r.kind,
                       "claim": r.claim, "ok": r.ok, "detail": r.detail,
                       "deviations": r.deviations} for r in self.results],
            "failures": [r.id for r in self.failures],
            "deviations": [{"case": c, "detail": d}
                           for c, d in self.deviations],
            "unmet_requirements": self.unmet_requirements,
        }

    def render(self) -> str:
        lines = [f"Gama-G conformance, specification {SPEC_VERSION}",
                 f"{len(self.results)} claims, {len(self.failures)} failed, "
                 f"{len(self.deviations)} recorded deviation(s)"]
        if self.facts is not None:
            lines.append(
                f"extracted from the specification: "
                f"{len(self.facts.all_types)} type names, "
                f"{len(self.facts.stages)} pipeline stages, "
                f"{len(self.facts.requirements)} v1.0 requirements")
        lines.append("")
        lines.extend(result.render() for result in self.results)
        if self.deviations:
            lines.append("")
            lines.append("Recorded deviations are not failures: each one is a "
                         "place the toolchain")
            lines.append("knowingly differs from the specification, and each "
                         "is written down with a")
            lines.append("reason in conformance/deviations.json.  A deviation "
                         "that is *not* recorded")
            lines.append("is a failure, and `--strict` fails on the recorded "
                         "ones too.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual claims
# ---------------------------------------------------------------------------

def _declaration_for(name: str) -> str:
    """A one-line program that uses `name` the way section 5 writes it.

    The generic groups are written as the specification writes them
    (`Matrix<T, Rows, Cols>`), not as this compiler prefers them: the point is
    to test the document, and the compiler's own spelling would be the compiler
    testing itself.
    """
    spelled = {"List": "List<I64>", "Set": "Set<I64>", "Map": "Map<Text, I64>",
               "Tuple": "Tuple(I64, Text)", "Option": "Option<I64>",
               "Result": "Result<I64, Text>", "Tensor": "Tensor<F64>",
               "Matrix": "Matrix<F64, 2, 2>"}
    return spelled.get(name, name)


def _accepts(annotation: str) -> Tuple[bool, str]:
    """Whether the compiler accepts a parameter of this type, and why not.

    `annotation` is usually a type as section 5 writes it, but a deviation's
    working alternative sometimes needs a declaration in front of it (`record
    Point` before `Point`), so a multi-line string is taken as a whole program.
    """
    if "\n" in annotation:
        source = annotation if annotation.endswith("\n") else annotation + "\n"
    else:
        source = f"fn takes(x: {annotation}) -> Unit\n    return\n"
    compilation = differential.compile_source(source, "<type-surface>")
    if compilation.ok:
        return True, ""
    for diagnostic in compilation.bag.diagnostics:
        code = getattr(diagnostic, "code", "") or "?"
        return False, f"{code}: {diagnostic.message}"
    return False, "refused with no diagnostic"


def _type_surface(facts: SpecFacts, deviations: Dict[str, Any]) -> CaseResult:
    recorded = {d["item"]: d for d in deviations.get("type-surface", [])}
    missing: List[str] = []
    present: List[str] = []
    seen_deviation: List[str] = []
    for group, names in facts.types.items():
        for name in names:
            ok, why = _accepts(_declaration_for(name))
            if ok:
                present.append(name)
                if name in recorded:
                    # The compiler accepts it *and* the file claims a
                    # deviation; one of the two is stale.
                    missing.append(
                        f"{name}: recorded as a deviation but the compiler "
                        f"accepts `{_declaration_for(name)}`")
                continue
            entry = recorded.get(name)
            if entry is None:
                missing.append(
                    f"{name} ({group}, section 5) cannot be used as a type "
                    f"name: {why}")
                continue
            alternative = entry.get("alternative")
            if alternative:
                accepted, alternative_why = _accepts(alternative)
                if not accepted:
                    missing.append(
                        f"{name}: the deviation records `{alternative}` as "
                        f"the working spelling, and it is refused: "
                        f"{alternative_why}")
                    continue
            seen_deviation.append(f"{name} -- {entry['reason']}")
    ok = not missing
    detail = ""
    if missing:
        detail = ("unrecorded deviations:\n           "
                  + "\n           ".join(missing))
    else:
        detail = (f"{len(present)} of {len(present) + len(seen_deviation)} "
                  f"names usable as written; the rest are recorded")
    return CaseResult("types-section-5", "§5", "type-surface",
                      "every type name section 5 lists can be declared",
                      ok, detail, seen_deviation)


def _pipeline(facts: SpecFacts, mapping: Dict[str, Any]) -> CaseResult:
    stages = mapping.get("stages", {})
    problems: List[str] = []
    mapped = 0
    for stage in facts.stages:
        entry = stages.get(stage)
        if entry is None:
            problems.append(f"`{stage}` (section 22) is not mapped and not "
                            f"recorded as unimplemented")
            continue
        status = entry.get("status")
        if status not in ("implemented", "partial", "not-implemented"):
            problems.append(f"`{stage}` has status {status!r}")
            continue
        if status != "implemented" and not entry.get("reason"):
            problems.append(f"`{stage}` is {status} with no reason given")
            continue
        mapped += 1
    # The other direction: a mapping entry for a stage the specification does
    # not have is a claim about a document that does not exist.
    extra = sorted(set(stages) - set(facts.stages))
    if extra:
        problems.append("mapped but not in section 22: " + ", ".join(extra))
    return CaseResult("pipeline-section-22", "§22", "pipeline",
                      "every compilation stage is implemented or explained",
                      not problems,
                      "\n           ".join(problems) if problems
                      else f"all {mapped} stages accounted for")


def _requirements(facts: SpecFacts, file: Dict[str, Any]) -> CaseResult:
    entries = file.get("requirements", {})
    problems: List[str] = []
    evidenced = 0
    not_claimed = 0
    unmet: List[str] = []
    for index, text in enumerate(facts.requirements, 1):
        entry = entries.get(str(index))
        if entry is None:
            problems.append(f"requirement {index} is not addressed: "
                            f"{text[:60]}")
            continue
        status = entry.get("status")
        if status == "evidenced":
            evidenced += 1
            if not entry.get("evidence"):
                problems.append(f"requirement {index} says `evidenced` with "
                                f"no evidence named")
            elif not entry.get("command"):
                problems.append(f"requirement {index} names evidence but no "
                                f"command a reader can run")
        elif status in ("partial", "not-claimed"):
            not_claimed += 1
            unmet.append(f"requirement {index} is `{status}`: {text[:60]}")
            if not entry.get("reason"):
                problems.append(f"requirement {index} is `{status}` with no "
                                f"reason -- an unexplained gap is the thing "
                                f"this checklist exists to prevent")
        else:
            problems.append(f"requirement {index} has status {status!r}; "
                            f"expected `evidenced`, `partial` or "
                            f"`not-claimed`")
    extra = sorted(set(entries) - {str(i) for i in range(
        1, len(facts.requirements) + 1)}, key=lambda k: (len(k), k))
    if extra:
        problems.append("addressed but not in section 33: "
                        + ", ".join(extra))
    result = CaseResult("requirements-section-33", "§33", "requirements",
                        "every v1.0 requirement is evidenced or not claimed",
                        not problems,
                        "\n           ".join(problems) if problems
                        else f"{evidenced} evidenced, {not_claimed} partial "
                             f"or not claimed")
    # Stashed on the result so `run` can put it in the report: `--strict` asks
    # whether anything is merely *not disproved*, which is a different question
    # from whether the checklist is answered.
    result.unmet = unmet          # type: ignore[attr-defined]
    return result


def _program(claim: Dict[str, Any], directory: str,
             spec_text: str) -> CaseResult:
    path = os.path.join(directory, claim["program"])
    source = _read(path)
    repeats = int(claim.get("repeat", 1))
    runs = [differential.run_interpreter(source, claim["program"])
            for _ in range(repeats)]
    result = runs[0]
    expected = claim.get("expect_stdout")
    problems: List[str] = []
    if repeats > 1:
        # A claim about reproducibility is only checked by running twice: one
        # run cannot show that the second would agree.
        outputs = {run.stdout for run in runs}
        if len(outputs) != 1:
            problems.append(f"{repeats} runs disagreed: "
                            f"{sorted(outputs)}")
    if result.compile_error:
        problems.append(f"did not compile: {result.compile_error}")
    else:
        if expected is not None and result.stdout.strip() != expected.strip():
            problems.append(f"printed {result.stdout.strip()!r}, expected "
                            f"{expected.strip()!r}")
        want_status = claim.get("expect_status", 0)
        if result.exit_status != want_status:
            problems.append(f"exited {result.exit_status}, expected "
                            f"{want_status} ({result.fault_kind})")
        want_fault = claim.get("expect_fault")
        if want_fault and result.fault_kind != want_fault:
            problems.append(f"fault kind was {result.fault_kind!r}, expected "
                            f"{want_fault!r}")
    kind = "fault" if claim.get("expect_fault") else "program"
    return CaseResult(claim["id"], claim["spec"], kind, claim["claim"],
                      not problems, "; ".join(problems))


def _refused(claim: Dict[str, Any], directory: str) -> CaseResult:
    path = os.path.join(directory, claim["program"])
    source = _read(path)
    compilation = differential.compile_source(source, claim["program"])
    problems: List[str] = []
    if compilation.ok:
        problems.append("was accepted; the specification requires it to be "
                        "refused")
    else:
        codes = [getattr(d, "code", "") for d in compilation.bag.diagnostics]
        wanted = claim.get("expect_code")
        if wanted and wanted not in codes:
            problems.append(f"refused with {codes or 'no code'}, expected "
                            f"{wanted!r}")
    return CaseResult(claim["id"], claim["spec"], "refused", claim["claim"],
                      not problems, "; ".join(problems))


def _deterministic(claim: Dict[str, Any], directory: str) -> CaseResult:
    """The same source compiled twice has to produce the same output.

    Section 1.3 asks for reproducible results and stage 14 of section 22 is
    reproducibility verification, so this is a claim about the compiler rather
    than a smoke test: two runs that differ mean the compiler is not a function
    of its input.
    """
    import hashlib
    path = os.path.join(directory, claim["program"])
    source = _read(path)
    digests = []
    problems: List[str] = []
    for _ in range(2):
        compilation = differential.compile_source(source, claim["program"])
        if not compilation.ok:
            problems.append("did not compile")
            break
        text = compilation.program.render()
        digests.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    if not problems and len(set(digests)) != 1:
        problems.append(f"two compilations differ: {digests}")
    return CaseResult(claim["id"], claim["spec"], "deterministic",
                      claim["claim"], not problems,
                      "; ".join(problems) or f"identical output ({digests[0][:12]})")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _load_json(path: str, what: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise SpecError(f"{what} is missing: {path}")
    with open(path, encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{what} is not valid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Running the suite
# ---------------------------------------------------------------------------

def run(spec_text: str, directory: Optional[str] = None,
        *, repo_root: Optional[str] = None, only: Optional[str] = None,
        strict: bool = False) -> Report:
    """Run every claim in the suite and report what happened.

    Raises `SpecError` if the specification or the suite cannot be read.  That
    is deliberate and is not a failure of the *toolchain*: it is the suite
    saying it could not check anything, which must never be reported as a pass.
    """
    facts = extract(spec_text)
    directory = directory or suite_dir(repo_root)
    suite = _load_json(os.path.join(directory, "claims.json"), "the claim list")
    claims = suite.get("claims", [])
    if not claims:
        raise SpecError("the claim list is empty; a suite with no claims "
                        "passes vacuously")
    deviations_file = _load_json(os.path.join(directory, "deviations.json"),
                                 "the deviation file")
    pipeline_file = _load_json(os.path.join(directory, "stage-mapping.json"),
                               "the stage mapping")
    requirements_file = _load_json(
        os.path.join(directory, "requirements.json"),
        "the requirement checklist")

    results: List[CaseResult] = []
    for claim in claims:
        if only and only not in claim["id"]:
            continue
        kind = claim["kind"]
        if kind == "type-surface":
            results.append(_type_surface(facts, deviations_file))
        elif kind == "pipeline":
            results.append(_pipeline(facts, pipeline_file))
        elif kind == "requirements":
            results.append(_requirements(facts, requirements_file))
        elif kind in ("program", "fault"):
            results.append(_program(claim, directory, spec_text))
        elif kind == "refused":
            results.append(_refused(claim, directory))
        elif kind == "deterministic":
            results.append(_deterministic(claim, directory))
        else:
            raise SpecError(f"claim {claim['id']} has unknown kind {kind!r}")

    unmet: List[str] = []
    for result in results:
        unmet.extend(getattr(result, "unmet", []))
    # `strict` does not change what passed: a recorded deviation is a
    # deviation, and calling it a failure would make `ok` mean two different
    # things depending on a flag.  It adds a second, stricter answer --
    # `strict_failures` -- for the caller that wants the production bar.
    return Report(results=results, facts=facts, strict=strict,
                  unmet_requirements=unmet)


def load_spec(path: Optional[str] = None) -> str:
    """Read the specification, or say why it cannot be checked."""
    resolved = path or default_spec_path()
    if not resolved:
        raise SpecError(
            f"cannot find {SPEC_FILENAME}; pass --spec PATH, or set GAMA_SPEC. "
            f"This suite checks the toolchain against that document and will "
            f"not report success without it")
    if not os.path.isfile(resolved):
        raise SpecError(f"no specification at {resolved}")
    return _read(resolved)
