"""End-to-end: every shipped example must compile, run and say what it claims.

The examples in `examples/` are the documentation a newcomer actually reads,
so a regression there is a regression in the product.  Each test runs the real
file through the real driver and asserts on output that would be wrong if the
feature it demonstrates were broken -- not merely that the process exited 0.
"""

from __future__ import annotations

import os
import unittest

import support as S


class ExampleInventory(unittest.TestCase):
    def test_the_eight_examples_are_present(self):
        expected = {
            "hello.gg", "medical_dosing.gg", "parallel_pipeline.gg",
            "policy_transaction_agent.gg", "property_tests.gg",
            "security_audit.gg", "self_healing_service.gg",
            "train_linear_model.gg",
        }
        self.assertEqual(set(S.example_names()), expected,
                         "the example set changed; update this test and the "
                         "README's tour of the examples")

    def test_the_six_core_examples_are_present(self):
        expected = {
            "core/classify.gg", "core/converge.gg", "core/dose.gg",
            "core/ledger.gg", "core/selection.gg", "core/traverse.gg",
        }
        self.assertEqual(set(S.core_example_names()), expected,
                         "the core example set changed; update this test, the "
                         "README's tour and docs/DESIGN_v0_2.md")

    def test_every_example_uses_only_supported_comment_syntax(self):
        """Comments are `//` and `/* */`; a `#` comment is a lex error."""
        for name in S.all_example_names():
            with self.subTest(example=name):
                path = S.example(name)
                with open(path, "r", encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        stripped = line.lstrip()
                        self.assertFalse(
                            stripped.startswith("#"),
                            f"{name}:{number} uses a `#` comment, which Gama-G "
                            f"does not have")

    def test_every_example_is_documented_with_how_to_run_it(self):
        for name in S.all_example_names():
            with self.subTest(example=name):
                with open(S.example(name), "r", encoding="utf-8") as handle:
                    head = "".join(
                        handle.readline() for _ in range(12))
                self.assertIn("ggc", head,
                              f"{name} does not tell the reader how to run it")

    def test_every_example_compiles_under_the_strict_profile(self):
        """The examples are the reference for what strict accepts."""
        for name in S.all_example_names():
            with self.subTest(example=name):
                path = S.example(name)
                with open(path, "r", encoding="utf-8") as handle:
                    source = handle.read()
                outcome = S.compile_only(source, profile="strict", path=name)
                if not outcome.compiled:
                    self.fail(f"{name} does not compile under `strict`:\n"
                              + outcome.messages())


class HelloExample(unittest.TestCase):
    """The first thing a newcomer runs."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = S.run_file(S.example("hello.gg"))

    def test_it_runs(self):
        self.outcome.assert_ran(self)

    def test_arithmetic_and_accumulation(self):
        self.outcome.assert_output_contains(self, "total       : 42",
                                            "sum of 1..5 : 15")

    def test_a_division_by_zero_is_a_value_not_a_crash(self):
        """Spec section 6: errors are values, so the example keeps going."""
        self.outcome.assert_output_contains(self, "1 / 0     : division by zero")
        self.assertIn("first even: 14", self.outcome.output,
                      "execution did not continue past the handled error")

    def test_a_guarded_match_covers_every_band(self):
        self.outcome.assert_output_contains(self,
                                            "score 95 -> distinction",
                                            "score 80 -> pass",
                                            "score 45 -> resit",
                                            "score 12 -> fail")


class MedicalDosingExample(unittest.TestCase):
    """Spec sections 16 and 27: contracts on a clinically shaped program."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = S.run_file(S.example("medical_dosing.gg"))

    def test_it_runs(self):
        self.outcome.assert_ran(self)

    def test_dosing_is_computed(self):
        self.assertTrue(self.outcome.output.strip())

    def test_the_audit_trail_is_valid(self):
        ok, problems = self.outcome.context.audit.verify()
        self.assertTrue(ok, f"the example's own audit chain is invalid: {problems}")

    def test_it_does_not_claim_regulatory_compliance(self):
        """Spec section 43: clinical correctness needs domain validation."""
        with open(S.example("medical_dosing.gg"), "r", encoding="utf-8") as handle:
            text = handle.read().lower()
        for phrase in ("fda approved", "hipaa compliant", "certified",
                       "guaranteed safe", "clinically validated"):
            self.assertNotIn(phrase, text,
                             f"the example claims {phrase!r}, which spec "
                             f"section 43 forbids")


class SelfHealingExample(unittest.TestCase):
    """Spec section 10: bounded recovery."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = S.run_file(S.example("self_healing_service.gg"))

    def test_it_recovers(self):
        self.outcome.assert_ran(self)
        self.outcome.assert_output_contains(self, "recovered      : true")

    def test_it_reports_which_level_recovered_it(self):
        self.assertIn("local retry", self.outcome.output)

    def test_every_recovery_action_was_audited(self):
        actions = [r.fields.get("recovery_action")
                   for r in self.outcome.context.audit.records
                   if r.action == "RECOVERY_ACTION"]
        self.assertTrue(actions, "recovery happened without an audit record")
        for action in actions:
            self.assertTrue(action, f"a recovery action was recorded as {action!r}")


class ParallelPipelineExample(unittest.TestCase):
    """Spec section 3: independent operations, and the pipeline form."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = S.run_file(S.example("parallel_pipeline.gg"))

    def test_the_independent_region_completes(self):
        self.outcome.assert_ran(self)
        self.outcome.assert_output_contains(self, "loading vitals")

    def test_the_dependent_region_is_still_correct(self):
        self.outcome.assert_output_contains(self, "a = 2  b = 20  c = 22")

    def test_two_regions_do_not_interfere(self):
        """Regression: task functions used to be named per function."""
        self.assertIn("summary:", self.outcome.output)
        self.assertIn("a = 2", self.outcome.output)


class SecurityAuditExample(unittest.TestCase):
    """Spec sections 8, 12 and 13."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = S.run_file(S.example("security_audit.gg"))

    def test_the_capability_handle_restricts_access(self):
        self.outcome.assert_ran(self)
        self.outcome.assert_output_contains(self,
                                            "grants Read   : true",
                                            "grants Write  : false")

    def test_the_secret_is_never_printed(self):
        self.assertNotIn("hunter2-token", self.outcome.output,
                         "the example leaked the secret it was protecting")
        self.assertIn("<redacted", self.outcome.output)

    def test_a_tampered_signature_does_not_verify(self):
        self.outcome.assert_output_contains(self,
                                            "verifies      : true",
                                            "tamper check  : false")

    def test_the_audit_chain_is_valid_and_chained(self):
        ok, problems = self.outcome.context.audit.verify()
        self.assertTrue(ok, problems)
        records = self.outcome.context.audit.records
        self.assertGreaterEqual(len(records), 3)
        for previous, current in zip(records, records[1:]):
            self.assertEqual(current.prev_hash, previous.hash)

    def test_execution_is_reproducible(self):
        """Spec section 1.3, demonstrated by the example itself."""
        again = S.run_file(S.example("security_audit.gg"))
        again.assert_ran(self)
        self.assertEqual(self.outcome.output, again.output)
        self.assertEqual(self.outcome.context.audit.head_hash,
                         again.context.audit.head_hash)


class PolicyTransactionAgentExample(unittest.TestCase):
    """Spec sections 9B, 17 and 18."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = S.run_file(S.example("policy_transaction_agent.gg"))

    def test_the_policy_explains_itself(self):
        self.outcome.assert_ran(self)
        self.outcome.assert_output_contains(self, "payroll_admin -> ALLOW",
                                            "contractor -> DENY")
        self.assertIn('role("contractor")', self.outcome.output,
                      "the decision did not quote the rule that produced it")

    def test_a_require_gate_denies(self):
        self.outcome.assert_output_contains(self, "tests fail  -> DENY")

    def test_a_transaction_that_fails_is_aborted_then_retried(self):
        actions = [r.action for r in self.outcome.context.audit.records]
        self.assertIn("TRANSACTION_ABORT", actions)
        self.assertIn("TRANSACTION_COMMIT", actions)
        self.assertEqual(actions.count("TRANSACTION_BEGIN"),
                         actions.count("TRANSACTION_COMMIT")
                         + actions.count("TRANSACTION_ABORT"),
                         "a transaction began and never reached an outcome")

    def test_the_agent_receives_its_messages(self):
        self.outcome.assert_output_contains(self,
                                            "Monitor received alert: cpu above 90%",
                                            "Monitor is alive")


class TrainLinearModelExample(unittest.TestCase):
    """Spec sections 14 and 15."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = S.run_file(S.example("train_linear_model.gg"))

    def test_it_converges_on_the_true_parameters(self):
        """The data is y = 2x + 1, so the fit should land on w=2, b=1."""
        self.outcome.assert_ran(self)
        lines = self.outcome.output.splitlines()
        weight = [l for l in lines if "learned w" in l][0].split(":")[1]
        bias = [l for l in lines if "learned b" in l][0].split(":")[1]
        self.assertAlmostEqual(float(weight), 2.0, places=2)
        self.assertAlmostEqual(float(bias), 1.0, places=2)

    def test_the_ai_safety_gates_are_exercised(self):
        self.outcome.assert_output_contains(self, "AI-safety requires")

    def test_broadcasting_is_demonstrated(self):
        self.outcome.assert_output_contains(
            self, "[[101.0], [102.0], [103.0], [104.0]]")


class PropertyTestsExample(unittest.TestCase):
    """Spec section 26, run through `ggc test`."""

    def test_every_declared_test_passes(self):
        import subprocess
        result = subprocess.run(
            [os.path.join(S.REPO_ROOT, "tools", "bin", "ggc"), "test",
             S.example("property_tests.gg")],
            capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0,
                         f"`ggc test` failed:\n{result.stdout}\n{result.stderr}")
        self.assertIn("8/8 test(s) passed", result.stdout)

    def test_a_property_actually_quantifies(self):
        """The domain-less `for all` form must sample, not run zero times."""
        outcome = S.run("""
test property "nonzero division"
    for all x where x != 0
        assert x / x == 1
""", entry="test:nonzero division")
        self.assertTrue(outcome.compiled, outcome.messages())
        outcome.assert_ran(self)

    def test_a_broken_property_fails_the_suite(self):
        import subprocess
        path = os.path.join(S.REPO_ROOT, "tests", "_tmp_broken.gg")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('test property "wrong"\n'
                         '    for all x in range(0, 5)\n'
                         '        assert x < 2\n')
        try:
            result = subprocess.run(
                [os.path.join(S.REPO_ROOT, "tools", "bin", "ggc"), "test", path],
                capture_output=True, text=True, timeout=180)
            self.assertNotEqual(result.returncode, 0,
                                "a failing property exited 0")
            self.assertIn("FAIL", result.stdout)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
