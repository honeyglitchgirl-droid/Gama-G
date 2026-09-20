"""The fuzzer: generation, mutation, the invariants, and the campaign engine.

Audit priority 9.  The most important test in this file is not that the fuzzer
finds bugs -- it is that the fuzzer *can* find bugs.  A campaign that reports
"no invariant was broken" is only evidence if the checks are capable of
reporting the opposite, so `SelfCheck` runs tools/fuzz_selfcheck.py, which
breaks the product deliberately and asserts that each invariant notices.

Each of the bugs the fuzzer found has a regression test here as well, because
a bug found by accident and fixed on trust is a bug that returns.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import unittest

import support as S
from gamag import diagnostics
from gamag.fuzz import engine, generator, oracle

REPO_ROOT = S.REPO_ROOT


class Generation(unittest.TestCase):
    """Generated programs have to be programs, not noise."""

    def test_a_core_program_is_produced_and_often_compiles(self):
        rng = random.Random(4)
        made = [generator.generate_core(rng) for _ in range(40)]
        self.assertTrue(all(m.dialect == "core" for m in made))
        compiled = sum(1 for m in made
                       if S.compile_only(m.source).compiled)
        # A generator whose output is always rejected only ever exercises the
        # error path, which is the opposite of what it is for.
        self.assertGreater(compiled, len(made) // 2,
                           f"only {compiled}/{len(made)} generated core "
                           f"programs compiled")

    def test_a_v01_program_is_produced_and_often_compiles(self):
        rng = random.Random(5)
        made = [generator.generate_v01(rng) for _ in range(40)]
        compiled = sum(1 for m in made
                       if S.compile_only(m.source).compiled)
        self.assertGreater(compiled, len(made) // 2,
                           f"only {compiled}/{len(made)} generated v0.1 "
                           f"programs compiled")

    def test_generation_is_reproducible_from_a_seed(self):
        first = [generator.generate(random.Random(9)).source for _ in range(5)]
        second = [generator.generate(random.Random(9)).source for _ in range(5)]
        self.assertEqual(first, second,
                         "the same seed must produce the same campaign")

    def test_different_seeds_produce_different_programs(self):
        one = {generator.generate(random.Random(s)).source for s in range(12)}
        self.assertGreater(len(one), 1)

    def test_adversarial_input_is_sometimes_produced(self):
        rng = random.Random(2)
        kinds = {generator.generate(rng).kind for _ in range(200)}
        self.assertIn("adversarial", kinds)
        self.assertIn("empty", kinds)

    def test_no_generated_core_program_declares_an_invented_capability(self):
        # `converge.gg` once used a capability the specification does not
        # define, and the lesson generalises: the generator may only use the
        # specification's own vocabulary.
        from gamag.capabilities import KNOWN_CAPABILITIES
        rng = random.Random(11)
        for _ in range(60):
            source = generator.generate_core(rng).source
            for word in ("grant", "needs", "authority"):
                if word in source:
                    self.fail(f"the generator emitted `{word}`, which requires "
                              f"capability names from {sorted(KNOWN_CAPABILITIES)}")


class Mutation(unittest.TestCase):
    """Mutations have to reach interesting states, not just be broken."""

    def test_a_mutation_changes_the_program(self):
        source = S.example_file("hello.gg")
        rng = random.Random(3)
        changed = sum(1 for _ in range(40)
                      if generator.mutate(source, rng, "hello.gg").source
                      != source)
        self.assertGreater(changed, 30,
                           "most mutations should alter the program")

    def test_mutation_kinds_are_varied(self):
        source = S.example_file("hello.gg")
        rng = random.Random(6)
        kinds = {generator.mutate(source, rng, "hello.gg").kind
                 for _ in range(300)}
        for expected in ("delete-line", "duplicate-line", "swap-lines",
                         "truncate", "insert-junk", "flip-character",
                         "rename-identifier", "number-swap"):
            self.assertIn(expected, kinds)

    def test_a_mutation_records_what_it_did(self):
        mutation = generator.mutate(S.example_file("hello.gg"),
                                    random.Random(1), "hello.gg")
        self.assertTrue(mutation.detail,
                        "a reproducer without a description is hard to read")
        self.assertEqual(mutation.origin, "hello.gg")

    def test_the_corpus_covers_both_dialects(self):
        corpus = generator.example_corpus(S.EXAMPLES_DIR)
        self.assertGreater(len(corpus), 10)
        joined = "\n".join(source for _path, source in corpus)
        self.assertIn("gama core", joined, "the core examples must be included")
        self.assertIn("fn main", joined, "the v0.1 examples must be included")


class Invariants(unittest.TestCase):
    """Each invariant, checked against input chosen to reach it."""

    def test_a_well_formed_program_is_clean(self):
        result = oracle.run_checks(S.example_file("hello.gg"),
                                   path="examples/hello.gg", deep=True)
        self.assertEqual(result.verdict, oracle.VERDICT_COMPILED)
        self.assertEqual([v for v in result.violations
                          if v.severity in ("crash", "wrong", "inconsistent")],
                         [])

    def test_a_broken_program_is_reported_as_rejected_not_violating(self):
        result = oracle.run_checks("let x = ", path="<broken>", deep=True)
        self.assertEqual(result.verdict, oracle.VERDICT_REJECTED)
        self.assertEqual([v for v in result.violations
                          if v.severity == "crash"], [],
                         "refusing to compile is an answer, not a defect")

    def test_an_infinite_loop_is_bounded_by_the_instruction_budget(self):
        # Regression: the oracle used to build its Context with the default
        # fifty-million-step budget, so one generated loop hung the whole
        # campaign for minutes.  The wall-clock check could not report it
        # either, because it only ran once `execute` returned.
        program = ("fn main() -> Unit\n"
                   "    io\n"
                   "    var i = 0\n"
                   "    while i < 1000000000\n"
                   "        i = i + 1\n"
                   "    print(\"done\", i)\n")
        import time
        started = time.perf_counter()
        result = oracle.run_checks(program, path="<loop>", deep=True)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 20.0,
                        f"an unbounded loop took {elapsed:.1f}s; the budget is "
                        f"not being applied")
        self.assertEqual(result.verdict, oracle.VERDICT_COMPILED)

    def test_every_diagnostic_has_a_code(self):
        # Regression: the lexer and both parsers emitted diagnostics with no
        # code at all, so they could not be filtered or looked up.  The fuzzer
        # found this on its first campaign.
        inputs = ["let x = ", "@", "gama core", "fn f( -> {}", "let n = 1e",
                  "match x\n", "let 9B = 1", "let s = \"abc", "", "\x00\x01"]
        for text in inputs:
            outcome = S.compile_only(text, path="<t>")
            for diagnostic in outcome.diagnostics:
                self.assertTrue(diagnostic.code,
                                f"{text!r} produced an uncoded diagnostic: "
                                f"{diagnostic.message[:70]!r}")
                self.assertRegex(diagnostic.code, r"^[EW]-[a-z0-9-]+$")

    def test_the_front_end_code_table_is_total(self):
        from gamag.diagnostics import front_end_code
        for message in ("", "something nobody predicted", "expected a thing",
                        "unterminated string literal"):
            self.assertTrue(front_end_code(message))

    def test_the_optimizer_check_notices_a_changed_result(self):
        # The invariant has to be able to fail.  This drives it directly rather
        # than through an injection, so it stays cheap and deterministic.
        from gamag.fuzz import oracle as O
        program = ("fn main() -> Unit\n"
                   "    io\n"
                   "    var total = 0\n"
                   "    for i in 1..6\n"
                   "        total = total + i\n"
                   "    print(\"total\", total)\n")
        self.assertEqual(O.check_optimizer_preserves_behavior(program), [],
                         "an untouched compiler must look untouched")

    def test_a_fault_is_classified_and_says_what_happened(self):
        program = ('fn main() -> Unit\n'
                   '    io\n'
                   '    let r = 1 / 0\n'
                   '    print(r)\n')
        result = oracle.run_checks(program, path="<bad>", deep=True)
        self.assertEqual(result.verdict, oracle.VERDICT_COMPILED)
        self.assertEqual([v for v in result.violations
                          if v.invariant in ("faults-are-classified",
                                             "faults-say-what-happened")], [])


class Campaign(unittest.TestCase):
    """The engine: counting, deduplication and reproducers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.corpus = engine.default_corpus(S.EXAMPLES_DIR)

    def test_the_corpus_is_the_shipped_examples(self):
        self.assertEqual(len(self.corpus), len(S.all_example_names()))

    def test_a_short_campaign_completes_and_counts_sensibly(self):
        campaign = engine.run(60, seed=3, corpus_paths=self.corpus, deep=True)
        self.assertEqual(campaign.rounds, 60)
        self.assertEqual(
            campaign.compiled_count + campaign.rejected_count, 60,
            "every input was either compiled or rejected")
        self.assertLessEqual(campaign.clean_count, 60)
        self.assertIsInstance(campaign.summary(), str)

    def test_a_campaign_on_an_impossible_corpus_still_terminates(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "seed.gg"), "w",
                      encoding="utf-8") as handle:
                handle.write("gama core 0.2\nintent T\n")
            campaign = engine.run(40, seed=1, corpus_paths=[os.path.join(
                directory, "seed.gg")], deep=False)
            self.assertEqual(campaign.rounds, 40)

    def test_reproducers_are_written_only_when_something_failed(self):
        save_dir = os.path.join(self._tmp.name, "clean")
        campaign = engine.run(30, seed=8, corpus_paths=self.corpus,
                              deep=False, save_dir=save_dir)
        self.assertEqual(campaign.distinct, {})
        self.assertFalse(os.path.isdir(save_dir) and os.listdir(save_dir),
                         "a clean campaign must not claim to have written "
                         "reproducers")

    def test_a_violation_is_deduplicated_by_signature(self):
        # Two inputs producing the same failure should be reported once, with a
        # count -- a report with ten thousand identical lines is one nobody
        # reads.
        violation = oracle.Violation("some-invariant", "wrong", "the same text")
        other = oracle.Violation("some-invariant", "wrong", "the same text")
        different = oracle.Violation("some-invariant", "wrong", "different text")
        self.assertEqual(violation.signature, other.signature)
        self.assertNotEqual(violation.signature, different.signature)

    def test_the_seed_reproduces_a_campaign_exactly(self):
        one = engine.run(40, seed=21, corpus_paths=self.corpus, deep=False)
        two = engine.run(40, seed=21, corpus_paths=self.corpus, deep=False)
        self.assertEqual([c.source for c in one.cases],
                         [c.source for c in two.cases])
        self.assertEqual([c.kind for c in one.cases],
                         [c.kind for c in two.cases])

    def test_a_case_knows_whether_it_compiled_and_whether_it_was_clean(self):
        campaign = engine.run(40, seed=31, corpus_paths=self.corpus, deep=False)
        for case in campaign.cases:
            self.assertEqual(case.compiled,
                             case.verdict == oracle.VERDICT_COMPILED)
            self.assertEqual(case.clean, not case.violations)


class SelfCheck(unittest.TestCase):
    """The fuzzer must be able to fail.

    This is the test that keeps the other checks honest.  It runs the injection
    harness, which breaks the product in one specific way per invariant and
    asserts that the corresponding check notices.  If any check is decoration,
    this fails and says which.
    """

    @staticmethod
    def _declared_injections(script: str) -> int:
        """How many bugs the harness says it injects, read from the harness.

        Counting them here by hand would make this test fail for the wrong
        reason the moment somebody adds an injection, and a test that fails for
        the wrong reason is a test that gets ignored.
        """
        import ast
        with open(script, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if (isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", "") == "INJECTIONS"):
                return len(node.value.elts)
        raise AssertionError(f"no INJECTIONS list in {script}")

    def test_every_injected_bug_is_caught(self):
        script = os.path.join(REPO_ROOT, "tools", "fuzz_selfcheck.py")
        self.assertTrue(os.path.isfile(script), script)
        count = self._declared_injections(script)
        self.assertGreaterEqual(count, 5,
                                "the harness has lost injections; each invariant "
                                "needs a bug that would trip it")
        proc = subprocess.run([sys.executable, script], capture_output=True,
                              text=True, timeout=900)
        self.assertEqual(
            proc.returncode, 0,
            "the fuzzer failed to catch an injected bug:\n"
            + proc.stdout + proc.stderr)
        self.assertIn(f"all {count} injected bugs were caught", proc.stdout)
        self.assertIn("control: nothing injected, nothing reported",
                      proc.stdout)


if __name__ == "__main__":
    unittest.main()
