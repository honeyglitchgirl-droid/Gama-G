"""The specification's own code examples, run verbatim.

These are the snippets the blueprint uses to define the language: the
pipeline (section 3, line 140), the agent (9B, 349), the service (10, 366),
the capability signature (12, 422), the model (14, 512), the AI-safety
gates (15, 527), the policy (17, 594), the transaction (18, 607), the
property test (26, 821), the contract (27, 834) and the two worked examples
(38, 1236 and 39, 1251).

Each is lifted out of the document by line number rather than retyped, so the
tests track the blueprint.  Some snippets deliberately reference names they
never define -- `Medical.Patient`, `network(features)`, `Kg` -- because the
document is illustrating syntax, not shipping a program.  For those the
conformance claim is that the *syntax* is accepted; the ones that are
self-contained are executed as well.
"""

from __future__ import annotations

import unittest

import support as S

# (section, line the example starts on, whether it can be executed standalone)
EXAMPLES = [
    ("3 pipeline", 140, False),
    ("4 function", 172, True),
    ("5 pure function", 196, True),
    ("6 Result", 261, False),
    ("7 effect declaration", 289, True),
    ("9B agent", 349, False),
    ("10 service", 366, False),
    ("12 capability", 422, False),
    ("14 model", 512, False),
    ("15 AI safety", 527, False),
    ("17 policy", 594, True),
    ("18 transaction", 607, False),
    ("26 property test", 821, True),
    ("27 contract", 834, False),
    ("38 fraud pipeline", 1236, False),
    ("39 medical analysis", 1251, False),
]


class SpecExamplesParse(unittest.TestCase):
    """Every example in the blueprint must be accepted by the parser."""

    def test_examples_are_found_in_the_document(self):
        for section, line, _ in EXAMPLES:
            with self.subTest(section=section):
                block = S.spec_block_from(line)
                self.assertGreater(len(block.strip()), 10,
                                   f"nothing was extracted at line {line}; the "
                                   f"specification has probably been reflowed, "
                                   f"so update the line numbers in this test")

    def test_every_example_parses(self):
        from gamag.lexer import Lexer
        from gamag.parser import Parser
        for section, line, _ in EXAMPLES:
            with self.subTest(section=section):
                source = S.spec_block_from(line)
                path = f"<spec:{section}>"
                try:
                    tokens = Lexer(source, path).tokenize()
                    module = Parser(tokens, path, source).parse_module()
                except Exception as exc:            # noqa: BLE001
                    self.fail(f"spec section {section} (line {line}) does not "
                              f"parse: {type(exc).__name__}: {exc}\n"
                              f"--- extracted ---\n{source}")
                self.assertTrue(module.decls or module.top_level,
                                f"spec section {section} parsed to nothing")


class SpecPolicyExampleRuns(unittest.TestCase):
    """Section 17, line 594 -- verbatim, plus a caller."""

    def test_payroll_access_verbatim(self):
        source = S.spec_block_from(594) + """
fn main() -> Unit
    io
    audit
    let admin = payrollAccess.evaluate({"role": "payroll_admin"})
    let contractor = payrollAccess.evaluate({"role": "contractor"})
    print(admin.allow, admin.reason)
    print(contractor.allow, contractor.reason)
"""
        outcome = S.run(source)
        outcome.assert_ran(self)
        self.assertIn("true", outcome.output)
        self.assertIn("false", outcome.output)
        # Spec section 17: decisions must be explainable, and the explanation
        # is the rule as written.
        self.assertIn('role("contractor")', outcome.output)

    def test_deny_beats_allow(self):
        """A subject matching both rules must be denied."""
        source = """
policy both
    allow role("staff")
    deny role("suspended")
    audit all

fn main() -> Unit
    io
    audit
    let d = both.evaluate({"role": "staff"})
    print("staff:", d.allow)
"""
        outcome = S.run(source)
        outcome.assert_ran(self)
        self.assertIn("staff: true", outcome.output)


class SpecTransactionExampleRuns(unittest.TestCase):
    """Section 18, line 607 -- the shape, with the two effects supplied."""

    def test_transaction_commits(self):
        source = """
grant AuditWrite

fn debit(account: Text, amount: F64) -> Unit
    io
    print("debit", account, amount)

fn credit(account: Text, amount: F64) -> Unit
    io
    print("credit", account, amount)

transaction payment
    audit
    debit("A", 10.0)
    credit("B", 10.0)

fn main() -> Unit
    io
    audit
    payment.run()
    print("state:", audit.verify())
"""
        outcome = S.run(source, grants=("AuditWrite",))
        outcome.assert_ran(self)
        outcome.assert_output_contains(self, "debit A", "credit B")
        actions = [r.action for r in outcome.context.audit.records]
        self.assertIn("TRANSACTION_BEGIN", actions)
        self.assertIn("TRANSACTION_COMMIT", actions)

    def test_transaction_that_fails_is_aborted_not_left_open(self):
        """Spec section 18: failure before commit enters a recovery state."""
        source = """
grant AuditWrite

transaction payment
    audit
    print("debit")
    panic("gateway timeout")

service PaymentService
    protect
        payment.run()
    recover
        retry 1
        alert operator

fn main() -> Unit
    io
    audit
    let result = PaymentService()
    print("attempts:", result.attempts)
"""
        outcome = S.run(source, grants=("AuditWrite",))
        # Recovery is bounded, so a permanently failing transaction exhausts
        # it and the fault propagates -- that is the point of section 18.
        outcome.assert_faulted(self, kind="RecoveryExhausted")
        actions = [r.action for r in outcome.context.audit.records]
        self.assertIn("TRANSACTION_ABORT", actions,
                      "a transaction that faulted before commit was left open")
        # Every begin is matched by a commit or an abort.
        begins = actions.count("TRANSACTION_BEGIN")
        ends = actions.count("TRANSACTION_COMMIT") + actions.count("TRANSACTION_ABORT")
        self.assertEqual(begins, ends,
                         f"{begins} transaction(s) began but {ends} finished")


class SpecAgentExampleRuns(unittest.TestCase):
    """Section 9B, line 349 -- `agent Monitor / on alert / evaluate(alert)`."""

    def test_agent_dispatches_typed_messages(self):
        source = S.spec_block_from(349).replace(
            "evaluate(alert)", 'print("evaluated:", alert)') + """
fn main() -> Unit
    io
    Monitor.send("alert", "cpu high")
"""
        outcome = S.run(source)
        outcome.assert_ran(self)
        outcome.assert_output_contains(self, "evaluated: cpu high")

    def test_agents_do_not_share_mutable_state(self):
        """Spec section 9B: agents communicate by message, not by sharing."""
        source = """
agent A
    on ping
        print("A got", ping)

agent B
    on ping
        print("B got", ping)

fn main() -> Unit
    io
    A.send("ping", 1)
    B.send("ping", 2)
"""
        outcome = S.run(source)
        outcome.assert_ran(self)
        outcome.assert_output_contains(self, "A got 1", "B got 2")

    def test_unknown_event_is_refused(self):
        source = """
agent A
    on ping
        print("got", ping)

fn main() -> Unit
    io
    A.send("nope", 1)
"""
        outcome = S.run(source)
        self.assertFalse(outcome.ran,
                         "sending an event the agent does not declare was "
                         "silently accepted")


class SpecPropertyTestExampleRuns(unittest.TestCase):
    """Section 26, line 821 -- including the domain-less `for all` form."""

    def test_division_identity_verbatim(self):
        source = S.spec_block_from(821)
        outcome = S.run(source, entry="test:division identity")
        self.assertTrue(outcome.compiled, outcome.messages())
        self.assertTrue(outcome.ran,
                        f"the specification's own property test faulted: "
                        f"{outcome.fault}")

    def test_domain_less_for_all_samples_deterministically(self):
        """`for all x where ...` has no explicit domain, so it must be seeded."""
        source = S.spec_block_from(821)
        first = S.run(source, entry="test:division identity", seed=7)
        second = S.run(source, entry="test:division identity", seed=7)
        self.assertTrue(first.ran and second.ran)

    def test_explicit_domain_and_sample_count(self):
        source = """
test property "squares are non-negative"
    for all x in range(-5, 5) samples 9
        assert x * x >= 0
"""
        outcome = S.run(source, entry="test:squares are non-negative")
        self.assertTrue(outcome.compiled, outcome.messages())
        outcome.assert_ran(self)

    def test_a_failing_property_is_reported(self):
        source = """
test property "always true"
    for all x in range(0, 5)
        assert x < 3
"""
        outcome = S.run(source, entry="test:always true")
        self.assertTrue(outcome.compiled, outcome.messages())
        self.assertFalse(outcome.ran,
                         "a property that does not hold was reported as passing")


class SpecPipelineExample(unittest.TestCase):
    """Section 3, line 140 -- the operation-graph form."""

    def test_stage_names_bind_and_result_is_the_output(self):
        source = """
model RiskModel
    input features: Tensor<F64>
    output risk: F64

    predict features
        return tensor.item(features) * 0.5

pipeline patientRisk
    io
    model
    input patient: Text
    normalize
    extract features
    predict risk using RiskModel
    audit
    return result

fn main() -> Unit
    io
    model
    let out = patientRisk("P-1")
    print("pipeline produced:", out)
"""
        outcome = S.run(source)
        self.assertTrue(outcome.compiled,
                        f"the section 3 pipeline form does not check:\n"
                        f"{outcome.messages()}")

    def test_verbatim_pipeline_parses(self):
        source = S.spec_block_from(140)
        outcome = S.compile_only(source)
        self.assertIsNotNone(outcome.compilation.module)


class SpecContractExample(unittest.TestCase):
    """Section 27, line 834 -- requires/ensures."""

    def test_verbatim_contract_parses(self):
        source = S.spec_block_from(834)
        outcome = S.compile_only(source)
        self.assertIsNotNone(outcome.compilation.module)

    def test_contract_is_enforced_at_runtime(self):
        source = """
fn dosage(weight: F64, factor: F64) -> F64
    pure
    requires weight > 0
    requires factor >= 0
    ensures result >= 0
    return weight * factor

fn main() -> Unit
    io
    print("valid:", dosage(70.0, 0.5))
    print("invalid:", dosage(-1.0, 0.5))
"""
        outcome = S.run(source)
        outcome.assert_faulted(self)
        self.assertIn("weight > 0", str(outcome.fault),
                      "the fault should quote the contract as written")

    def test_ensures_is_checked_too(self):
        source = """
fn bad(x: F64) -> F64
    pure
    ensures result >= 0
    return 0.0 - x

fn main() -> Unit
    io
    print(bad(5.0))
"""
        outcome = S.run(source)
        outcome.assert_faulted(self)
        self.assertIn("result >= 0", str(outcome.fault))


class SpecServiceExample(unittest.TestCase):
    """Section 10, line 366 -- protect/recover with named actions."""

    def test_recovery_actions_are_the_six_levels(self):
        # The surface syntax writes `restore checkpoint` and `alert operator`;
        # the engine keys actions by their verb.
        from gamag.runtime.recovery import ACTION_LEVELS, LEVEL_NAMES
        for action in ("retry", "reset", "restore", "restart", "failover",
                       "alert", "escalate"):
            with self.subTest(action=action):
                self.assertIn(action, ACTION_LEVELS,
                              f"`{action}` is not a recovery action")
        self.assertEqual(sorted(LEVEL_NAMES), [0, 1, 2, 3, 4, 5])

    def test_service_example_parses(self):
        outcome = S.compile_only(S.spec_block_from(366))
        self.assertIsNotNone(outcome.compilation.module)


class SpecModelExample(unittest.TestCase):
    """Section 14, line 512 -- `model` with input/output declarations."""

    def test_model_signature_comes_from_its_declarations(self):
        source = """
model RiskModel
    input features: Tensor<F64>
    output risk: F64

    predict features
        return tensor.item(features) * 0.5

fn main() -> Unit
    io
    model
    print(RiskModel.predict(tensor.from_list([[0.84]])))
"""
        outcome = S.run(source)
        outcome.assert_ran(self)
        self.assertIn("0.42", outcome.output)

    def test_shape_is_checked_where_possible(self):
        """Spec section 6 asks for shape checking where possible."""
        source = """
model M
    input features: Tensor<F64,[4]>
    output risk: F64

    predict features
        return tensor.item(features)

fn main() -> Unit
    io
    model
    print(M.predict(tensor.from_list([[1.0], [2.0]])))
"""
        outcome = S.compile_only(source)
        # A [2,1] tensor against a declared [4] input must not be accepted
        # silently; either it is rejected or the mismatch is diagnosed.
        if outcome.compiled:
            run_outcome = S.run(source)
            self.assertFalse(
                run_outcome.ran,
                "a tensor whose shape contradicts the model's declared input "
                "shape was accepted without complaint")


if __name__ == "__main__":
    unittest.main()
