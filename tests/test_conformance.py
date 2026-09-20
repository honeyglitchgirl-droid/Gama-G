"""The conformance suite has to be able to fail.

A suite that reads the specification is only worth having if it says something
the implementation's own tests cannot.  The tests here are therefore mostly
about the runner's *failure* modes: a type the compiler cannot express and
nobody recorded, a requirement that vanished from the checklist, a claim whose
expectation is wrong, a specification that is not the one the suite checks.

The last of those matters most.  `ggc conform` reporting success because it
could not find the document would be worse than any miscompile: it would be a
conformance claim with nothing behind it.  So the "cannot check" paths are
tested as paths, and they end in a non-zero status.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import support as S

from gamag import conformance


def suite_copy() -> str:
    """A writable copy of the shipped suite, for mutation tests."""
    target = tempfile.mkdtemp(prefix="conformance-")
    shutil.copytree(conformance.suite_dir(), target, dirs_exist_ok=True)
    return target


def rewrite(directory: str, name: str, mutate) -> None:
    path = os.path.join(directory, name)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    mutate(data)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


class TheShippedSuite(unittest.TestCase):
    """The suite as it ships, and the facts it reads out of the document."""

    @classmethod
    def setUpClass(cls):
        cls.spec = conformance.load_spec()
        cls.facts = conformance.extract(cls.spec)

    def test_the_specification_is_where_the_suite_expects_it(self):
        path = conformance.default_spec_path()
        self.assertIsNotNone(
            path, "the specification ships with the repository; without it "
                  "this suite is not a conformance suite")
        self.assertEqual(os.path.basename(path), conformance.SPEC_FILENAME)

    def test_the_extraction_finds_what_the_document_contains(self):
        """A parser that quietly finds nothing would make everything pass."""
        self.assertEqual(len(self.facts.stages), 14)
        self.assertEqual(len(self.facts.requirements), 20)
        self.assertEqual(sum(len(v) for v in self.facts.types.values()), 33)
        # Spot checks against the document itself, so a change to the
        # extraction that still returns *a* list of the right length is caught.
        self.assertIn("I128", self.facts.types["Primitive"])
        self.assertIn("Result", self.facts.types["Compound"])
        self.assertIn("Matrix", self.facts.types["Domain"])
        self.assertEqual(self.facts.stages[0], "Lexing")
        self.assertEqual(self.facts.stages[-1], "Reproducibility verification")
        self.assertIn("Medical functionality is explicitly scoped",
                      self.facts.requirements[19])

    def test_it_refuses_a_document_that_is_not_the_specification(self):
        with self.assertRaises(conformance.SpecError):
            conformance.extract("a file that is not the specification")

    def test_the_shipped_suite_passes(self):
        report = conformance.run(self.spec)
        self.assertTrue(report.ok, report.render())
        self.assertGreaterEqual(len(report.results), 10)

    def test_the_shipped_suite_records_its_deviations(self):
        """The deviations are the honest part, so they are pinned.

        If a deviation is fixed the runner notices and fails (the compiler
        accepts the spelling while the file still calls it a deviation), so
        this assertion cannot silently rot -- it fails in the other direction
        too, when a new one appears unrecorded.
        """
        report = conformance.run(self.spec)
        items = {detail.split(" -- ")[0] for _, detail in report.deviations}
        self.assertEqual(items, {"Record", "Enum", "Matrix"})

    def test_strict_reports_without_reclassifying_failures(self):
        """`ok` and `strict_failures` answer two different questions."""
        report = conformance.run(self.spec, strict=True)
        self.assertTrue(report.ok, "a recorded deviation is not a failure")
        self.assertTrue(report.strict_failures,
                        "--strict has to have something to say, or the flag "
                        "means nothing")
        kinds = " ".join(report.strict_failures)
        self.assertIn("Matrix", kinds)
        self.assertIn("requirement", kinds)

    def test_every_claim_cites_a_section(self):
        path = os.path.join(conformance.suite_dir(), "claims.json")
        with open(path, encoding="utf-8") as handle:
            claims = json.load(handle)["claims"]
        ids = [c["id"] for c in claims]
        self.assertEqual(len(ids), len(set(ids)), "claim ids must be unique")
        for claim in claims:
            with self.subTest(claim=claim["id"]):
                self.assertRegex(claim["spec"], r"^§\d",
                                 "a claim without a section is an opinion")
                self.assertIn("claim", claim)
                if claim["kind"] in ("program", "fault", "refused",
                                     "deterministic"):
                    program = os.path.join(conformance.suite_dir(),
                                           claim["program"])
                    self.assertTrue(os.path.isfile(program),
                                    f"{claim['program']} is missing")

    def test_the_requirements_checklist_answers_all_twenty(self):
        path = os.path.join(conformance.suite_dir(), "requirements.json")
        with open(path, encoding="utf-8") as handle:
            entries = json.load(handle)["requirements"]
        self.assertEqual(sorted(entries, key=int),
                         [str(i) for i in range(1, 21)])
        for index, entry in entries.items():
            with self.subTest(requirement=index):
                self.assertIn(entry["status"],
                              ("evidenced", "partial", "not-claimed"))
                if entry["status"] != "evidenced":
                    self.assertTrue(entry.get("reason"),
                                    "an unanswered gap is what this file is "
                                    "for")


class TheRunnerCanFail(unittest.TestCase):
    """Each mutation is a mistake the suite exists to catch."""

    @classmethod
    def setUpClass(cls):
        cls.spec = conformance.load_spec()

    def test_an_unrecorded_deviation_fails(self):
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "deviations.json",
                lambda d: d.__setitem__(
                    "type-surface",
                    [e for e in d["type-surface"] if e["item"] != "Matrix"]))
        report = conformance.run(self.spec, directory=directory)
        self.assertFalse(report.ok)
        self.assertIn("Matrix", report.failures[0].detail)

    def test_a_deviation_whose_alternative_does_not_work_fails(self):
        """A deviation is a claim about the compiler too, and it can be wrong."""
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "deviations.json",
                lambda d: [e.__setitem__("alternative", "Matrix<F64, 2, 2>")
                           for e in d["type-surface"] if e["item"] == "Matrix"])
        report = conformance.run(self.spec, directory=directory)
        self.assertFalse(report.ok)
        self.assertIn("working spelling", report.failures[0].detail)

    def test_a_stale_deviation_fails(self):
        """Recorded as broken, but the compiler accepts it: one of the two is
        wrong, and a suite that forgives that stops meaning anything."""
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "deviations.json",
                lambda d: d["type-surface"].append(
                    {"item": "I64", "spec": "§5", "reason": "not really"}))
        report = conformance.run(self.spec, directory=directory)
        self.assertFalse(report.ok)
        self.assertIn("recorded as a deviation", report.failures[0].detail)

    def test_a_dropped_requirement_fails(self):
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "requirements.json",
                lambda d: d["requirements"].pop("5"))
        report = conformance.run(self.spec, directory=directory)
        self.assertFalse(report.ok)
        self.assertIn("requirement 5", report.failures[0].detail)

    def test_an_unexplained_requirement_fails(self):
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "requirements.json",
                lambda d: d["requirements"].__setitem__(
                    "10", {"status": "not-claimed"}))
        report = conformance.run(self.spec, directory=directory)
        self.assertFalse(report.ok)
        self.assertIn("no reason", report.failures[0].detail)

    def test_a_stage_that_is_not_in_the_specification_fails(self):
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "stage-mapping.json",
                lambda d: d["stages"].__setitem__(
                    "Teleportation", {"status": "implemented"}))
        report = conformance.run(self.spec, directory=directory)
        self.assertFalse(report.ok)
        self.assertIn("not in section 22", report.failures[0].detail)

    def test_a_wrong_expectation_fails(self):
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)

        def break_it(data):
            for claim in data["claims"]:
                if claim["id"] == "syntax-examples-section-4":
                    claim["expect_stdout"] = "ok 6"
        rewrite(directory, "claims.json", break_it)
        report = conformance.run(self.spec, directory=directory)
        self.assertFalse(report.ok)
        failed = [r for r in report.failures
                  if r.id == "syntax-examples-section-4"]
        self.assertTrue(failed)
        self.assertIn("expected", failed[0].detail)

    def test_a_claim_with_no_cases_is_refused_not_passed(self):
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "claims.json", lambda d: d.__setitem__("claims", []))
        with self.assertRaises(conformance.SpecError):
            conformance.run(self.spec, directory=directory)

    def test_an_unknown_claim_kind_is_refused(self):
        directory = suite_copy()
        self.addCleanup(shutil.rmtree, directory, True)
        rewrite(directory, "claims.json",
                lambda d: d["claims"].append(
                    {"id": "invented", "kind": "vibes", "spec": "§1",
                     "claim": "it feels right"}))
        with self.assertRaises(conformance.SpecError):
            conformance.run(self.spec, directory=directory)

    def test_a_reproducibility_claim_runs_more_than_once(self):
        """One run cannot show that a second would agree."""
        path = os.path.join(conformance.suite_dir(), "claims.json")
        with open(path, encoding="utf-8") as handle:
            claims = json.load(handle)["claims"]
        repeat = [c for c in claims
                  if c["id"] == "deterministic-execution-is-reproducible"]
        self.assertEqual(len(repeat), 1)
        self.assertGreaterEqual(repeat[0].get("repeat", 1), 2)


class TheCommandLine(unittest.TestCase):

    def ggc(self, *args):
        proc = subprocess.run(
            [sys.executable, os.path.join(S.REPO_ROOT, "tools", "bin", "ggc")]
            + list(args), capture_output=True, text=True, timeout=300)
        self.assertNotIn("Traceback (most recent call last)",
                         proc.stdout + proc.stderr)
        return proc

    def test_it_runs_and_passes(self):
        proc = self.ggc("conform")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("specification 1.0", proc.stdout)
        self.assertIn("0 failed", proc.stdout)

    def test_it_answers_json(self):
        proc = self.ggc("conform", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["cases"]), len(payload["cases"]))
        self.assertTrue(payload["deviations"])
        self.assertIn("unmet_requirements", payload)

    def test_a_filter_runs_one_claim(self):
        proc = self.ggc("conform", "-k", "compilation-is-deterministic")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("1 claims", proc.stdout)

    def test_strict_fails_on_what_is_not_fully_evidenced(self):
        proc = self.ggc("conform", "--strict")
        self.assertEqual(proc.returncode, 4, proc.stdout + proc.stderr)
        self.assertIn("does not fully evidence", proc.stdout)

    def test_strict_with_a_filter_is_a_usage_error(self):
        proc = self.ggc("conform", "--strict", "-k", "types")
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)

    def test_a_missing_specification_is_an_error_not_a_pass(self):
        proc = self.ggc("conform", "--spec", "/nonexistent/spec.txt")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("cannot check conformance", proc.stderr)

    def test_a_foreign_document_is_an_error_not_a_pass(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("This is a specification in name only.\n")
            path = fh.name
        self.addCleanup(os.unlink, path)
        proc = self.ggc("conform", "--spec", path)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("cannot check conformance", proc.stderr)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
