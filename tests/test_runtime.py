"""The runtime: tensors, autodiff, the audit chain, recovery and secrets.

These test the parts of Gama-G whose correctness cannot be established by
type checking alone -- a hash chain that does not detect tampering, or a
gradient that points the wrong way, both type-check perfectly.
"""

from __future__ import annotations

import unittest

import support as S


class TensorOperations(unittest.TestCase):
    """Spec section 14: a tensor type with shape and element type."""

    def run_tensor(self, body, grants=()):
        return S.run(f"fn main() -> Unit\n    io\n    model\n{body}",
                     grants=grants)

    def test_shape_and_rank(self):
        outcome = self.run_tensor(
            '    let t = tensor.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n'
            '    print(tensor.shape(t), tensor.rank(t))\n')
        outcome.assert_output_contains(self, "[2, 3]", "2")

    def test_matmul_is_correct(self):
        outcome = self.run_tensor(
            '    let a = tensor.from_list([[1.0, 2.0], [3.0, 4.0]])\n'
            '    let b = tensor.from_list([[5.0, 6.0], [7.0, 8.0]])\n'
            '    print(tensor.to_list(tensor.matmul(a, b)))\n')
        outcome.assert_ran(self)
        self.assertIn("[[19.0, 22.0], [43.0, 50.0]]", outcome.output)

    def test_transpose(self):
        outcome = self.run_tensor(
            '    let a = tensor.from_list([[1.0, 2.0], [3.0, 4.0]])\n'
            '    print(tensor.to_list(tensor.transpose(a)))\n')
        outcome.assert_ran(self)
        self.assertIn("[[1.0, 3.0], [2.0, 4.0]]", outcome.output)

    def test_softmax_sums_to_one(self):
        outcome = self.run_tensor(
            '    let a = tensor.from_list([[1.0, 2.0, 3.0]])\n'
            '    let s = tensor.softmax(a)\n'
            '    print(tensor.item(tensor.sum(s)))\n')
        outcome.assert_ran(self)
        total = float(outcome.output.strip().splitlines()[-1])
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_broadcasting(self):
        """A [4,1] column against a [1,1] scalar (spec section 14)."""
        outcome = self.run_tensor(
            '    let x = tensor.from_list([[1.0], [2.0], [3.0], [4.0]])\n'
            '    let y = x + tensor.from_list([[100.0]])\n'
            '    print(tensor.to_list(y))\n')
        outcome.assert_ran(self)
        self.assertIn("[[101.0], [102.0], [103.0], [104.0]]", outcome.output)

    def test_operators_match_the_functions(self):
        """`a + b` and `tensor.add(a, b)` must agree, line by line."""
        outcome = self.run_tensor(
            '    let a = tensor.from_list([[1.0, 2.0]])\n'
            '    let b = tensor.from_list([[3.0, 4.0]])\n'
            '    print(tensor.to_list(a + b))\n'
            '    print(tensor.to_list(tensor.add(a, b)))\n'
            '    print(tensor.to_list(a * 2.0))\n'
            '    print(tensor.to_list(tensor.mul(a, tensor.from_list([[2.0, 2.0]]))))\n')
        outcome.assert_ran(self)
        lines = outcome.output.strip().splitlines()
        self.assertEqual(lines[0], lines[1],
                         "`a + b` and `tensor.add(a, b)` differ")
        self.assertEqual(lines[2], lines[3],
                         "`a * 2.0` and `tensor.mul(a, 2.0)` differ")
        self.assertEqual(lines[0], "[[4.0, 6.0]]")


class AutomaticDifferentiation(unittest.TestCase):
    """Spec section 14: reverse-mode autodiff."""

    def test_gradients_match_the_analytic_answer(self):
        """For mse(x*w + b, y) the gradients have closed forms.

        With x = 2, y = 6, w = 1, b = 0: prediction is 2, loss is (2-6)^2 = 16,
        dL/dw = 2*(pred - y)*x = -16 and dL/db = 2*(pred - y) = -8.  A sign
        error or a missing factor here would still type-check and would still
        "train", just in the wrong direction or at the wrong rate.
        """
        outcome = S.run("""
fn main() -> Unit
    io
    model
    autodiff.begin()
    let x = tensor.from_list([[2.0]])
    let y = tensor.from_list([[6.0]])
    let w = autodiff.parameter(tensor.from_list([[1.0]]))
    let b = autodiff.parameter(tensor.from_list([[0.0]]))
    let loss = autodiff.mse(autodiff.linear(x, w, b), y)
    autodiff.backward(loss)
    print(tensor.item(loss))
    print(tensor.to_list(autodiff.grad(w)))
    print(tensor.to_list(autodiff.grad(b)))
""")
        outcome.assert_ran(self)
        loss, dw, db = outcome.output.strip().splitlines()
        self.assertAlmostEqual(float(loss), 16.0, places=9)
        self.assertEqual(dw, "[[-16.0]]")
        self.assertEqual(db, "[[-8.0]]")

    def test_a_gradient_step_moves_downhill(self):
        """One step of size lr must reduce the loss, not increase it."""
        outcome = S.run("""
fn main() -> Unit
    io
    model
    autodiff.begin()
    let x = tensor.from_list([[2.0]])
    let y = tensor.from_list([[6.0]])
    let w = autodiff.parameter(tensor.from_list([[1.0]]))
    let b = autodiff.parameter(tensor.from_list([[0.0]]))
    print(tensor.item(autodiff.mse(autodiff.linear(x, w, b), y)))
    autodiff.backward(autodiff.mse(autodiff.linear(x, w, b), y))
    autodiff.step([w, b], 0.01)
    print(tensor.item(autodiff.mse(autodiff.linear(x, w, b), y)))
    print(tensor.to_list(w))
""")
        outcome.assert_ran(self)
        before, after, w = outcome.output.strip().splitlines()
        self.assertLess(float(after), float(before),
                        "a gradient step increased the loss")
        self.assertEqual(w, "[[1.16]]",
                         "w should move from 1.0 toward 2.0 by lr * 16")

    def test_training_converges_to_the_true_parameters(self):
        """Fit y = 2x + 1 from w = 0, b = 0 and check it lands on 2 and 1."""
        outcome = S.run("""
fn main() -> Unit
    io
    model
    let x = tensor.from_list([[1.0], [2.0], [3.0], [4.0]])
    let y = tensor.from_list([[3.0], [5.0], [7.0], [9.0]])
    autodiff.begin()
    let w = autodiff.parameter(tensor.from_list([[0.0]]))
    let b = autodiff.parameter(tensor.from_list([[0.0]]))
    var step = 0
    while step < 400
        let prediction = autodiff.linear(x, w, b)
        let loss = autodiff.mse(prediction, y)
        autodiff.backward(loss)
        autodiff.step([w, b], 0.05)
        step = step + 1
    print(tensor.item(w), tensor.item(b))
""")
        outcome.assert_ran(self)
        w, b = (float(part) for part in outcome.output.split())
        self.assertAlmostEqual(w, 2.0, places=2)
        self.assertAlmostEqual(b, 1.0, places=2)

    def test_the_loss_decreases(self):
        outcome = S.run("""
fn main() -> Unit
    io
    model
    let x = tensor.from_list([[1.0], [2.0], [3.0]])
    let y = tensor.from_list([[2.0], [4.0], [6.0]])
    autodiff.begin()
    let w = autodiff.parameter(tensor.from_list([[0.0]]))
    let b = autodiff.parameter(tensor.from_list([[0.0]]))
    let first = autodiff.mse(autodiff.linear(x, w, b), y)
    print("first:", tensor.item(first))
    var step = 0
    while step < 200
        autodiff.backward(autodiff.mse(autodiff.linear(x, w, b), y))
        autodiff.step([w, b], 0.05)
        step = step + 1
    print("last:", tensor.item(autodiff.mse(autodiff.linear(x, w, b), y)))
""")
        outcome.assert_ran(self)
        values = [float(line.split(":")[1]) for line in
                  outcome.output.strip().splitlines()]
        self.assertLess(values[1], values[0],
                        "gradient descent increased the loss")
        self.assertLess(values[1], 0.05, "training did not converge")


class AuditChain(unittest.TestCase):
    """Spec section 13: append-only, hash-chained, tamper-evident."""

    def source(self, records=3):
        body = "".join(
            f'    audit.record {{\n        actor: "a{i}"\n'
            f'        action: "ACT{i}"\n        reason: "r{i}"\n    }}\n'
            for i in range(records))
        return ("grant AuditWrite\n\nfn main() -> Unit\n    io\n    audit\n"
                + body + "    print(audit.count(), audit.verify())\n")

    def test_the_chain_is_valid_when_untouched(self):
        outcome = S.run(self.source(), grants=("AuditWrite",))
        outcome.assert_ran(self)
        self.assertIn("true", outcome.output)

    def test_records_are_sequential(self):
        outcome = S.run(self.source(3), grants=("AuditWrite",))
        outcome.assert_ran(self)
        sequences = [r.seq for r in outcome.context.audit.records]
        self.assertEqual(sequences, list(range(len(sequences))))

    def test_the_genesis_record_has_no_predecessor(self):
        outcome = S.run(self.source(1), grants=("AuditWrite",))
        outcome.assert_ran(self)
        first = outcome.context.audit.records[0]
        self.assertEqual(set(first.prev_hash), {"0"},
                         "the first record should chain from a zero hash")

    def test_each_record_chains_to_the_previous(self):
        outcome = S.run(self.source(4), grants=("AuditWrite",))
        outcome.assert_ran(self)
        records = outcome.context.audit.records
        for previous, current in zip(records, records[1:]):
            self.assertEqual(current.prev_hash, previous.hash,
                             f"record {current.seq} does not chain to "
                             f"{previous.seq}")

    def test_every_record_is_signed(self):
        outcome = S.run(self.source(2), grants=("AuditWrite",))
        outcome.assert_ran(self)
        for record in outcome.context.audit.records:
            self.assertTrue(record.signature,
                            f"record {record.seq} is unsigned")

    def test_tampering_with_a_field_is_detected(self):
        """The point of a hash chain: an edited record must not verify."""
        outcome = S.run(self.source(3), grants=("AuditWrite",))
        outcome.assert_ran(self)
        ok, problems = outcome.context.audit.verify()
        self.assertTrue(ok, f"a freshly built chain did not verify: {problems}")
        target = outcome.context.audit.records[1]
        target.fields["reason"] = "edited after the fact"
        ok, problems = outcome.context.audit.verify()
        self.assertFalse(ok, "editing an audit record went undetected")
        self.assertTrue(problems, "verification failed without saying why")

    def test_tampering_with_an_action_is_detected(self):
        outcome = S.run(self.source(3), grants=("AuditWrite",))
        outcome.assert_ran(self)
        outcome.context.audit.records[2].action = "SOMETHING_ELSE"
        self.assertFalse(outcome.context.audit.verify()[0])

    def test_removing_a_record_is_detected(self):
        outcome = S.run(self.source(3), grants=("AuditWrite",))
        outcome.assert_ran(self)
        del outcome.context.audit.records[1]
        self.assertFalse(outcome.context.audit.verify()[0],
                         "deleting a record from the middle went undetected")

    def test_the_export_round_trips(self):
        outcome = S.run(self.source(2), grants=("AuditWrite",))
        outcome.assert_ran(self)
        import json
        from gamag.runtime.audit import AuditRecord
        lines = outcome.context.audit.to_jsonl().strip().splitlines()
        self.assertEqual(len(lines), len(outcome.context.audit.records))
        for line in lines:
            payload = json.loads(line)
            restored = AuditRecord.from_dict(payload)
            self.assertEqual(restored.hash, payload["hash"])


class Recovery(unittest.TestCase):
    """Spec section 10: bounded recovery through six levels."""

    def test_the_six_levels_are_named_in_order(self):
        from gamag.runtime.recovery import LEVEL_NAMES
        self.assertEqual(
            [LEVEL_NAMES[i] for i in range(6)],
            ["local retry", "resource reset", "state checkpoint restore",
             "component restart", "failover", "operator escalation"])

    def test_recovery_succeeds_when_the_fault_is_transient(self):
        outcome = S.run("""
var attempts = 0

fn flaky() -> I64
    io
    attempts = attempts + 1
    if attempts < 3
        panic("not yet")
    return attempts

service S
    protect
        let v = flaky()
        print("value:", v)
    recover
        retry 5
        alert operator

fn main() -> Unit
    io
    audit
    let outcome = S()
    print("recovered:", outcome.recovered)
    print("attempts:", outcome.attempts)
    print("level:", outcome.level, outcome.level_name)
""")
        outcome.assert_ran(self)
        outcome.assert_output_contains(self, "recovered: true", "value: 3")
        self.assertIn("local retry", outcome.output)

    def test_recovery_is_bounded(self):
        """Spec section 10: recovery must not retry forever."""
        outcome = S.run("""
fn alwaysFails() -> Unit
    io
    panic("permanent")

service S
    protect
        alwaysFails()
    recover
        retry 3
        alert operator

fn main() -> Unit
    io
    audit
    let outcome = S()
    print("recovered:", outcome.recovered)
""")
        outcome.assert_faulted(self, kind="RecoveryExhausted")
        attempts = outcome.context.audit.records
        actions = [r for r in attempts if r.action == "RECOVERY_ACTION"]
        self.assertLessEqual(len(actions), 4,
                             "recovery exceeded its declared bound")

    def test_every_recovery_action_is_audited(self):
        outcome = S.run("""
var attempts = 0

fn flaky() -> Unit
    io
    attempts = attempts + 1
    if attempts < 2
        panic("once")

service S
    protect
        flaky()
    recover
        retry 2

fn main() -> Unit
    io
    audit
    let outcome = S()
    print("actions:", outcome.actions)
""")
        outcome.assert_ran(self)
        actions = [r.action for r in outcome.context.audit.records]
        self.assertIn("RECOVERY_ACTION", actions,
                      "a recovery action was taken without being audited")


class Secrets(unittest.TestCase):
    """Spec section 8: a secret cannot leak by accident."""

    def test_redaction_never_shows_the_value(self):
        outcome = S.run("""
grant SecretExpose

fn main() -> Unit
    io
    crypto
    let s = secrets.wrap("supercalifragilistic", "api_key")
    print(secrets.redact(s))
""", grants=("SecretExpose",))
        outcome.assert_ran(self)
        self.assertNotIn("supercalifragilistic", outcome.output)
        self.assertIn("redacted", outcome.output)

    def test_a_fingerprint_is_stable_but_not_reversing(self):
        outcome = S.run("""
grant SecretExpose

fn main() -> Unit
    io
    crypto
    let a = secrets.wrap("same", "k")
    let b = secrets.wrap("same", "k")
    let c = secrets.wrap("different", "k")
    print(secrets.fingerprint(a) == secrets.fingerprint(b))
    print(secrets.fingerprint(a) == secrets.fingerprint(c))
    print(secrets.fingerprint(a))
""", grants=("SecretExpose",))
        outcome.assert_ran(self)
        lines = outcome.output.strip().splitlines()
        self.assertEqual(lines[0], "true", "equal secrets fingerprinted apart")
        self.assertEqual(lines[1], "false", "different secrets collided")
        self.assertNotIn("same", lines[2])

    def test_exposing_is_audited_with_its_reason(self):
        outcome = S.run("""
grant SecretExpose, AuditWrite

fn main() -> Unit
    io
    crypto
    audit
    let s = secrets.wrap("tok", "api")
    let shown = secrets.expose(s, "incident 4412")
    print(shown)
""", grants=("SecretExpose", "AuditWrite"))
        outcome.assert_ran(self)
        reasons = [r.reason for r in outcome.context.audit.records
                   if r.reason]
        self.assertIn("incident 4412", reasons,
                      "a secret was exposed without the reason being audited")


class PolicyEngine(unittest.TestCase):
    """Spec section 17: explainable decisions."""

    def test_the_reason_quotes_the_rule_as_written(self):
        outcome = S.run("""
policy access
    deny role("contractor")
    allow role("staff")
    audit all

fn main() -> Unit
    io
    audit
    let d = access.evaluate({"role": "contractor"})
    print(d.allow)
    print(d.reason)
""")
        outcome.assert_ran(self)
        self.assertIn("false", outcome.output)
        self.assertIn('role("contractor")', outcome.output,
                      "the explanation should quote the rule that fired")

    def test_the_matched_trace_is_recorded(self):
        outcome = S.run("""
policy access
    allow role("staff")
    audit all

fn main() -> Unit
    io
    audit
    let d = access.evaluate({"role": "staff"})
    print(d.matched)
""")
        outcome.assert_ran(self)
        self.assertIn("allow", outcome.output)

    def test_decisions_are_audited(self):
        outcome = S.run("""
grant AuditWrite

policy access
    deny role("contractor")
    audit all

fn main() -> Unit
    io
    audit
    let d = access.evaluate({"role": "contractor"})
    print(d.allow)
""", grants=("AuditWrite",))
        outcome.assert_ran(self)
        actions = [r.action for r in outcome.context.audit.records]
        self.assertIn("POLICY_DENY", actions)

    def test_an_unknown_subject_attribute_does_not_allow(self):
        """Failing closed: an absent attribute must not satisfy `allow`."""
        outcome = S.run("""
policy access
    allow role("staff")

fn main() -> Unit
    io
    let d = access.evaluate({})
    print(d.allow)
""")
        outcome.assert_ran(self)
        self.assertIn("false", outcome.output)


class Determinism(unittest.TestCase):
    """Spec section 1.3: reproducible execution."""

    SOURCE = """
grant AuditWrite

fn main() -> Unit
    io
    crypto
    audit
    print(crypto.random_int(1, 1000000))
    print(crypto.random_int(1, 1000000))
    print(secrets.fingerprint(secrets.wrap("tok", "k")))
    audit.record {
        actor: "a"
        action: "ACT"
    }
    print(audit.head())
"""

    def test_the_same_seed_reproduces_the_same_output(self):
        first = S.run(self.SOURCE, grants=("AuditWrite",), seed=1234)
        second = S.run(self.SOURCE, grants=("AuditWrite",), seed=1234)
        first.assert_ran(self)
        second.assert_ran(self)
        self.assertEqual(first.output, second.output)

    def test_the_audit_head_is_reproducible(self):
        first = S.run(self.SOURCE, grants=("AuditWrite",), seed=99)
        second = S.run(self.SOURCE, grants=("AuditWrite",), seed=99)
        self.assertEqual(first.context.audit.head_hash,
                         second.context.audit.head_hash)

    def test_a_different_seed_diverges(self):
        """The seed is what makes it reproducible, not a frozen constant."""
        first = S.run(self.SOURCE, grants=("AuditWrite",), seed=1)
        second = S.run(self.SOURCE, grants=("AuditWrite",), seed=2)
        first.assert_ran(self)
        second.assert_ran(self)
        self.assertNotEqual(first.output, second.output)


class Checkpoints(unittest.TestCase):
    """Spec section 10: `restore checkpoint` returns to genuinely recorded state."""

    def test_restore_checkpoint_is_audited_with_its_level(self):
        """`restore checkpoint` reports what it actually restored."""
        outcome = S.run("""
service S
    checkpoint every 5s

    protect
        panic("immediately")

    recover
        restore checkpoint
        alert operator

fn main() -> Unit
    io
    audit
    let outcome = S()
    print("recovered:", outcome.recovered)
""")
        # The fault is permanent, so restoring state cannot make the body
        # succeed; recovery must escalate and then exhaust rather than loop.
        outcome.assert_faulted(self, kind="RecoveryExhausted")
        actions = [r.fields for r in outcome.context.audit.records
                   if r.action == "RECOVERY_ACTION"]
        self.assertTrue(actions, "recovery acted without auditing what it did")
        restores = [a for a in actions
                    if a.get("recovery_action") == "restore checkpoint"]
        self.assertEqual(len(restores), 1, restores)
        restored = restores[0]
        self.assertEqual(restored["recovery_level"], 2,
                         "`restore checkpoint` is spec level 2")
        self.assertEqual(restored["recovery_level_name"],
                         "state checkpoint restore")
        # It must say what it restored, not merely that it tried.
        self.assertIn("restored", str(restored.get("detail", "")))

    def test_a_transient_fault_recovers_at_level_zero(self):
        outcome = S.run("""
var attempts = 0

fn flaky() -> I64
    io
    attempts = attempts + 1
    if attempts < 2
        panic("once")
    return 42

service S
    protect
        print("value:", flaky())
    recover
        retry 3

fn main() -> Unit
    io
    audit
    let outcome = S()
    print("recovered:", outcome.recovered)
    print("level:", outcome.level, outcome.level_name)
""")
        outcome.assert_ran(self)
        outcome.assert_output_contains(self, "recovered: true", "value: 42",
                                       "local retry")

    def test_escalation_reaches_the_operator_when_nothing_else_works(self):
        outcome = S.run("""
fn alwaysFails() -> Unit
    io
    panic("permanent")

service S
    protect
        alwaysFails()
    recover
        retry 2
        restart
        alert operator

fn main() -> Unit
    io
    audit
    let outcome = S()
    print(outcome.recovered)
""")
        outcome.assert_faulted(self, kind="RecoveryExhausted")
        actions = [r.fields.get("recovery_action")
                   for r in outcome.context.audit.records
                   if r.action == "RECOVERY_ACTION"]
        self.assertIn("alert operator", actions)
        exhausted = [r.fields for r in outcome.context.audit.records
                     if r.fields.get("exhausted")]
        self.assertEqual(len(exhausted), 1,
                         "exhaustion should be recorded exactly once")
        self.assertIn("final_error", exhausted[0],
                      "the record should carry the fault that exhausted recovery")


if __name__ == "__main__":
    unittest.main()
