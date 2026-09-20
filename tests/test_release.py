"""The release path, tested as a path.

A release is where a project makes claims it cannot take back: the classifier
in `pyproject.toml` says how much of this is believed finished, the version
numbers say what changed, and the attached conformance report says what was
checked.  Each of those is a file in this repository, so each of them can drift
away from the measurement that justified it.

These tests are mostly about that drift.  `docs/RELEASES.md` states the bar for
leaving Alpha; if the bar is met and nobody moved the classifier, or if the
classifier claims more than the conformance run supports, that is a defect in
the release, and it is the kind nobody notices until a user does.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import support as S

from gamag import conformance


def read(relative: str) -> str:
    with open(os.path.join(S.REPO_ROOT, relative), encoding="utf-8") as handle:
        return handle.read()


def read_path(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def release_notes(conformance_json: str, strict: str, **kwargs) -> str:
    """Run `tools/release_notes.py` and return what it wrote."""
    out = os.path.join(tempfile.mkdtemp(prefix="release-"), "notes.md")
    argv = [sys.executable,
            os.path.join(S.REPO_ROOT, "tools", "release_notes.py"),
            kwargs.pop("version", read("VERSION").strip()),
            "--conformance", conformance_json,
            "--strict", strict,
            "-o", out]
    for key, value in kwargs.items():
        argv += ["--" + key.replace("_", "-"), value]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:                       # pragma: no cover
        raise AssertionError(proc.stdout + proc.stderr)
    return read_path(out)


class TheNotesSayWhatWasChecked(unittest.TestCase):
    """Notes generated from the run, not written by hand beside it."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="release-src-")
        cls.report_path = os.path.join(cls.directory, "conformance.json")
        with open(cls.report_path, "w", encoding="utf-8") as handle:
            json.dump(conformance.run(conformance.load_spec()).as_json(),
                      handle)
        # A real strict log, so the notes have something to quote.
        proc = subprocess.run(
            [sys.executable, os.path.join(S.REPO_ROOT, "tools", "bin", "ggc"),
             "conform", "--strict"],
            capture_output=True, text=True, timeout=600)
        cls.strict_path = os.path.join(cls.directory, "strict.txt")
        with open(cls.strict_path, "w", encoding="utf-8") as handle:
            handle.write(proc.stdout)

    def test_the_notes_carry_the_conformance_summary(self):
        notes = release_notes(self.report_path, self.strict_path)
        self.assertIn("Gama-G", notes)
        self.assertIn("claims against specification", notes)

    def test_the_notes_reproduce_what_is_not_evidenced(self):
        notes = release_notes(self.report_path, self.strict_path)
        self.assertIn("does not fully evidence", notes)
        # The list itself, not a reference to it: a release that says "see CI"
        # will outlive the CI log.
        self.assertIn("types-section-5", notes)

    def test_the_notes_say_how_to_check_the_release(self):
        notes = release_notes(self.report_path, self.strict_path)
        self.assertIn("git checkout v", notes)
        self.assertIn("ggc conform", notes)

    def test_a_failed_claim_is_not_dressed_up(self):
        broken = json.loads(read_path(self.report_path))
        broken["ok"] = False
        broken["failures"] = ["types-section-5"]
        with open(os.path.join(self.directory, "bad.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(broken, handle)
        notes = release_notes(os.path.join(self.directory, "bad.json"),
                              self.strict_path)
        self.assertIn("FAILED", notes)
        self.assertNotIn("no failures", notes)

    def test_a_missing_report_is_admitted_rather_than_omitted(self):
        notes = release_notes(os.path.join(self.directory, "absent.json"),
                              os.path.join(self.directory, "absent.txt"))
        self.assertIn("makes no statement about conformance", notes)
        self.assertIn("not reproduced here", notes)

    def test_an_unsigned_manifest_is_reported_as_unsigned(self):
        """A signed-looking manifest with no key would be a fake provenance."""
        path = os.path.join(self.directory, "unsigned.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"digest": "abc123", "signed": False}, handle)
        notes = release_notes(self.report_path, self.strict_path,
                              manifest=path)
        self.assertIn("unsigned", notes)
        self.assertIn("abc123", notes)
        self.assertNotIn("is signed with", notes)

    def test_a_signed_manifest_names_the_key(self):
        path = os.path.join(self.directory, "signed.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"digest": "abc123", "signed": True,
                       "public_key": "deadbeefcafe"}, handle)
        notes = release_notes(self.report_path, self.strict_path,
                              manifest=path)
        self.assertIn("deadbeefcafe", notes)
        self.assertIn("reproducible", notes)

    def test_a_missing_manifest_is_admitted_rather_than_omitted(self):
        notes = release_notes(self.report_path, self.strict_path,
                              manifest=os.path.join(self.directory, "no.json"))
        self.assertIn("no reproducibility statement", notes)

    def test_checksums_are_included_when_there_are_any(self):
        sums = os.path.join(self.directory, "SHA256SUMS")
        with open(sums, "w", encoding="utf-8") as handle:
            handle.write("deadbeef  gama_g-1.2.0-py3-none-any.whl\n")
        notes = release_notes(self.report_path, self.strict_path,
                              checksums=sums)
        self.assertIn("deadbeef", notes)


class ThePolicyMatchesTheMeasurement(unittest.TestCase):
    """`docs/RELEASES.md` states a bar; this checks the bar is the real one."""

    @classmethod
    def setUpClass(cls):
        cls.doc = read("docs/RELEASES.md")
        report = conformance.run(conformance.load_spec())
        cls.strict_count = len(report.strict_failures)
        cls.version = read("VERSION").strip()

    def test_the_document_counts_what_strict_actually_lists(self):
        self.assertIn(f"lists {self.strict_count} things", self.doc,
                      "docs/RELEASES.md and `ggc conform --strict` disagree "
                      "about the distance to the production bar")

    def test_alpha_is_the_truth_while_strict_is_not_clean(self):
        """The classifier may not claim more than the run supports.

        Both directions matter.  Claiming Beta while `--strict` fails is the
        over-claim the whole conformance exercise exists to prevent; sitting on
        Alpha after the bar is met is the same claim being wrong in the other
        direction, and it is the failure mode this test is most likely to catch
        because it happens when someone does the work and forgets the label.
        """
        classifier = re.search(r'Development Status :: [\d.]+ - (\w+)',
                               read("pyproject.toml"))
        self.assertIsNotNone(classifier, "no development status in pyproject")
        status = classifier.group(1)
        if self.strict_count:
            self.assertEqual(
                status, "Alpha",
                f"`ggc conform --strict` lists {self.strict_count} things not "
                f"fully evidenced, so the classifier cannot say {status}")
        else:                                        # pragma: no cover
            self.assertNotEqual(
                status, "Alpha",
                "`ggc conform --strict` is clean, so this is no longer Alpha")

    def test_the_document_names_the_version_it_describes(self):
        self.assertIn(self.version, self.doc,
                      "docs/RELEASES.md should say what version it describes")

    def test_the_production_bar_is_stated_as_a_command(self):
        self.assertIn("`ggc conform --strict` exits 0", self.doc,
                      "the bar for leaving Alpha has to be checkable, which "
                      "means it has to be a command")


class TheDocumentsCountWhatTheSuiteHas(unittest.TestCase):
    """Test counts are claims, and claims drift.

    Both the README and `docs/PRODUCTION_GAPS.md` state how many tests pass.
    Those two numbers were already stale when this test was written -- 591 while
    the suite ran 620 -- which is what happens to a number nobody checks.  So
    the documents are compared with the suite, discovered exactly as CI
    discovers it.
    """

    @classmethod
    def setUpClass(cls):
        tests_dir = os.path.join(S.REPO_ROOT, "tests")
        suite = unittest.TestLoader().discover(tests_dir,
                                               top_level_dir=tests_dir)
        cls.total = suite.countTestCases()

    def test_the_readme_states_the_count_the_suite_has(self):
        readme = read("README.md")
        self.assertIn(f"**{self.total} tests pass**", readme,
                      f"the suite has {self.total} tests; README.md says "
                      f"something else")

    def test_the_gaps_document_states_the_count_the_suite_has(self):
        gaps = read("docs/PRODUCTION_GAPS.md")
        self.assertIn(f"# {self.total} tests", gaps,
                      f"the suite has {self.total} tests; "
                      f"docs/PRODUCTION_GAPS.md says something else")


class TheWorkflowRunsWhatTheDocumentsSaid(unittest.TestCase):
    """The workflow, the script, and the policy have to agree.

    There is no YAML parser in the standard library, and adding a dependency to
    read one file is not worth it — so these are textual checks.  They are
    aimed at the failures that actually happen: a flag renamed in the script
    but not in the workflow, a job that publishes before the suite has run, a
    release whose tag disagrees with its own version.
    """

    @classmethod
    def setUpClass(cls):
        cls.workflow = read(".github/workflows/release.yml")
        cls.script = read("tools/release_notes.py")

    def test_publishing_waits_for_conformance(self):
        needs = re.search(r"needs: \[([^\]]+)\]|needs: (\w+)", self.workflow)
        self.assertIsNotNone(needs, "no job dependency in the release workflow")
        self.assertIn("conformance", needs.group(0),
                      "the release must not publish before the suite has run")

    def test_the_tag_has_to_match_the_version(self):
        self.assertIn("does not match VERSION", self.workflow)

    def test_the_workflow_passes_the_script_the_flags_it_accepts(self):
        # Line continuations first: the invocation is four lines long, and a
        # pattern that stops at the first newline checks almost nothing.  It
        # did, before this was fixed -- the test passed while the workflow
        # carried a flag the script has never accepted.
        joined = re.sub(r"\\\n\s*", " ", self.workflow)
        calls = [line.strip() for line in joined.splitlines()
                 if "release_notes.py" in line]
        self.assertTrue(calls, "the workflow does not call release_notes.py")
        used = set(re.findall(r"--[a-z-]+", " ".join(calls)))
        used |= set(re.findall(r"(?<![\w-])-[a-zA-Z](?![\w-])",
                               " ".join(calls)))
        accepted = set(re.findall(r'"(-{1,2}[a-zA-Z-]+)"', self.script))
        unknown = used - accepted
        self.assertEqual(unknown, set(),
                         f"release.yml passes flags the script does not "
                         f"accept: {sorted(unknown)}")
        # And the other direction, so a documented flag cannot be dropped from
        # the workflow without the notes silently losing half their content.
        for required in ("--conformance", "--strict", "--checksums",
                         "--manifest"):
            self.assertIn(required, used,
                          f"the release notes should be built with {required}")

    def test_a_signing_key_comes_from_a_secret_and_has_no_stand_in(self):
        """No fallback key: an unsigned manifest must be able to stay unsigned."""
        self.assertIn("RELEASE_SIGNING_KEY", self.workflow)
        self.assertIn("unsigned", self.workflow)
        self.assertNotRegex(
            self.workflow, r"ggc manifest[^\n]*--sign\b",
            "`--sign` with no key generates a throwaway one, which signs the "
            "release with a key nobody can name; sign only with the secret")

    def test_the_wheel_is_installed_and_made_to_compile_something(self):
        # The C runtime is not a Python module and setuptools drops it unless
        # it is declared; running the native backend from the installed copy is
        # the only place that omission shows up.
        self.assertIn("ggc native", self.workflow)
        self.assertIn("pip install dist/*.whl", self.workflow)


class TheManifestCanBeChecked(unittest.TestCase):
    """`build-manifest.json` is in the release, so it has to be checkable."""

    def test_a_manifest_is_reproducible_and_signed(self):
        out = os.path.join(tempfile.mkdtemp(prefix="manifest-"), "m.json")
        proc = subprocess.run(
            [sys.executable, os.path.join(S.REPO_ROOT, "tools", "bin", "ggc"),
             "manifest", "--check-reproducible", "2", "-o", out,
             os.path.join(S.EXAMPLES_DIR, "hello.gg")],
            capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("reproducibility: yes", proc.stdout)
        with open(out, encoding="utf-8") as handle:
            data = json.load(handle)
        for key in ("digest", "signature", "public_key", "signed"):
            self.assertIn(key, data, f"the release manifest needs {key}")


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
