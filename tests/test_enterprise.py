"""The interoperability and enterprise modules.

Audit priorities 15 and 16: FHIR, terminology, provenance and consent on one
side, and database, identity, workflow, messaging and observability on the
other.

Most of these tests check the same two properties in different clothes: that a
module refuses to answer outside its authority, and that it says *why* when it
refuses.  A consent decision that returns a bare Bool, or a workflow that
advances without recording where it went, would pass a test that only checked
the happy path.
"""

from __future__ import annotations

import json
import unittest

import support as S
from gamag.std import enterprise, interop

#: Every module here is capability-gated, so a test that wants the happy path
#: has to grant what the module needs and say so in the program text.
GRANTS = ("PatientRead", "PatientWrite", "AuditWrite", "DatabaseRead",
          "DatabaseWrite", "CryptoSign")


def run(source: str, *, grants=GRANTS):
    enterprise.reset()
    return S.run(source, grants=grants)


FHIR_FIELDS = ('id: "P-1", name: "Ada", age: 34, sex: "female", '
               'mrn: "MRN-9"')


class Fhir(unittest.TestCase):

    def test_a_patient_serializes_with_the_specification_element_names(self):
        outcome = run(f'''grant PatientRead, PatientWrite

fn main() -> Unit
    io
    medical
    let p = Patient {{ {FHIR_FIELDS} }}
    print(fhir.serialize(p))
''')
        outcome.assert_ran(self)
        payload = json.loads(outcome.output.strip())
        self.assertEqual(payload["resourceType"], "Patient")
        self.assertEqual(payload["id"], "P-1")
        self.assertEqual(payload["name"], "Ada")
        self.assertEqual(payload["identifier"], "MRN-9")
        self.assertEqual(payload["gender"], "female")

    def test_validation_reports_which_element_is_missing(self):
        outcome = run('''grant PatientRead

fn main() -> Unit
    io
    medical
    print(fhir.validate("{\\"resourceType\\": \\"Observation\\", \\"code\\": \\"x\\"}"))
''')
        outcome.assert_ran(self)
        self.assertIn("status", outcome.output,
                      "the message should name the element that is missing")

    def test_validation_accepts_what_serialization_produces(self):
        # The two halves have to agree: a validator that rejects its own
        # serializer's output is a validator with the wrong schema.
        outcome = run(f'''grant PatientRead, PatientWrite

fn main() -> Unit
    io
    medical
    let p = Patient {{ {FHIR_FIELDS} }}
    let problem = fhir.validate(fhir.serialize(p))
    if problem == ""
        print("valid")
    else
        print("invalid:", problem)
''')
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "valid")

    def test_a_round_trip_preserves_the_record(self):
        outcome = run(f'''grant PatientRead, PatientWrite

fn main() -> Unit
    io
    medical
    let p = Patient {{ {FHIR_FIELDS} }}
    let back = fhir.parse(fhir.serialize(p))
    print(back.id, back.name, back.mrn, back.age)
''')
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "P-1 Ada MRN-9 34")

    def test_a_secret_cannot_be_serialized(self):
        # Spec section 8: secrets have restrictions on serialization.  A FHIR
        # resource is a serialization, so a secret must not reach one.
        #
        # Checked at the unit level because the type system stops the
        # program-level version first: a record cannot hold a Secret, so the
        # question "what if one did" only arises for a value that reached the
        # serializer another way.
        from gamag.diagnostics import GamaRuntimeFault
        from gamag.runtime.values import GSecret
        with self.assertRaises(GamaRuntimeFault) as raised:
            interop._to_fhir(GSecret("classified", "ssn"))
        self.assertEqual(raised.exception.kind, "SecretLeak")
        self.assertIn("secret", raised.exception.message.lower())

    def test_parse_refuses_an_invalid_resource(self):
        outcome = run('''grant PatientRead

fn main() -> Unit
    io
    medical
    print(fhir.parse("{\\"resourceType\\": \\"Observation\\"}"))
''')
        self.assertIsNotNone(outcome.execution.fault)
        self.assertIn("FHIRValidationError", str(outcome.execution.fault))


class Terminology(unittest.TestCase):

    PROGRAM = '''grant AuditWrite

fn main() -> Unit
    io
    medical
    terminology.load("http://snomed.info/sct", "{\\"codes\\": {\\"73211009\\": \\"Diabetes\\", \\"38341003\\": \\"Hypertension\\"}}")
    print(terminology.validate("http://snomed.info/sct", "73211009"))
    print(terminology.validate("http://snomed.info/sct", "99999"))
    print(terminology.lookup("http://snomed.info/sct", "38341003"))
'''

    def test_codes_validate_against_a_loaded_system(self):
        outcome = run(self.PROGRAM)
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.split(), ["true", "false", "Hypertension"])

    def test_an_unloaded_system_is_refused_rather_than_answered(self):
        # Answering "false" for an unknown system would be a wrong answer that
        # looks like a right one: the code might be perfectly valid and simply
        # not loaded.
        outcome = run('''fn main() -> Unit
    io
    medical
    print(terminology.validate("http://example.org/unknown", "x"))
''')
        self.assertIsNotNone(outcome.execution.fault)
        self.assertIn("UnknownCodeSystem", str(outcome.execution.fault))

    def test_require_names_the_bad_code(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    medical
    terminology.load("s", "{\\"codes\\": {\\"a\\": \\"A\\"}}")
    terminology.require("s", "b")
    print("not reached")
''')
        self.assertIsNotNone(outcome.execution.fault)
        self.assertIn("TerminologyError", str(outcome.execution.fault))
        self.assertIn("`b`", str(outcome.execution.fault))


class Provenance(unittest.TestCase):

    def test_entries_chain_and_verify(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    medical
    audit
    provenance.record("record created", "intake form")
    provenance.record("observation added", "device")
    print("count:", provenance.count(), "verifies:", provenance.verify())
''')
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "count: 2 verifies: true")

    def test_editing_an_entry_breaks_the_chain(self):
        # A chain that survives editing is decoration, so this drives the
        # structure directly rather than through a program that could not edit
        # it in the first place.
        chain = interop.ProvenanceChain()
        chain.append({"what": "one", "source": "a", "actor": "u"})
        chain.append({"what": "two", "source": "b", "actor": "u"})
        self.assertTrue(chain.verify()[0])
        chain.entries[0]["what"] = "tampered"
        ok, problems = chain.verify()
        self.assertFalse(ok)
        self.assertIn("edited", " ".join(problems))

    def test_removing_an_entry_is_detected(self):
        chain = interop.ProvenanceChain()
        chain.append({"what": "one", "source": "a", "actor": "u"})
        chain.append({"what": "two", "source": "b", "actor": "u"})
        del chain.entries[0]
        ok, problems = chain.verify()
        self.assertFalse(ok)


class Consent(unittest.TestCase):

    def test_consent_covers_its_subject_and_purpose(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    medical
    audit
    let c = consent.grant("P-1", "treatment")
    print(consent.check(c, "treatment", "P-1").allow)
    print(consent.check(c, "research", "P-1").allow)
    print(consent.check(c, "treatment", "P-2").allow)
''')
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.split(),
                         ["true", "false", "false"])

    def test_a_decision_states_its_reason(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    medical
    audit
    let c = consent.grant("P-1", "treatment")
    print(consent.check(c, "treatment", "P-2").reason)
''')
        outcome.assert_ran(self)
        self.assertIn("P-1", outcome.output)
        self.assertIn("P-2", outcome.output)

    def test_revoking_removes_the_permission(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    medical
    audit
    let c = consent.grant("P-1", "treatment")
    let revoked = consent.revoke(c)
    print(consent.check(revoked, "treatment", "P-1").allow)
''')
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "false")

    def test_require_fails_without_consent(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    medical
    audit
    let c = consent.grant("P-1", "treatment")
    consent.require(c, "treatment", "P-2")
    print("not reached")
''')
        self.assertIsNotNone(outcome.execution.fault)
        self.assertIn("ConsentError", str(outcome.execution.fault))

    def test_every_decision_is_audited(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    medical
    audit
    let c = consent.grant("P-1", "treatment")
    consent.check(c, "treatment", "P-1")
    consent.check(c, "treatment", "P-2")
    print("records:", audit.count())
''')
        outcome.assert_ran(self)
        self.assertIn("records:", outcome.output)
        count = int(outcome.output.split(":")[1])
        self.assertGreaterEqual(count, 3,
                                "grant plus both checks should be recorded")


#: Programs that are expected to be *refused*, kept here so their nested quotes
#: do not collide with the triple-quoted strings in the test bodies.
PROGRAMS = {
    "read_without_capability": """grant DatabaseWrite

fn main() -> Unit
    io
    storage
    let db = database.open(":memory:")
    print(database.tables(db))
""",
    "bind_a_secret": """grant DatabaseRead, DatabaseWrite, SecretExpose

fn main() -> Unit
    io
    storage
    crypto
    let db = database.open(":memory:")
    database.execute(db, "CREATE TABLE t (s TEXT)")
    let s = secrets.wrap("classified", "value")
    database.execute(db, "INSERT INTO t VALUES (?)", s)
    print("inserted")
""",
    "role_without_capability": """fn main() -> Unit
    io
    audit
    identity.register("ada", "x")
    identity.grant_role("ada", "admin")
""",
}


class Database(unittest.TestCase):

    PROGRAM = '''grant DatabaseRead, DatabaseWrite, AuditWrite

fn main() -> Unit
    io
    storage
    let db = database.open(":memory:")
    database.execute(db, "CREATE TABLE obs (id TEXT, value F64)")
    database.execute(db, "INSERT INTO obs VALUES (?, ?)", "O-1", 7.5)
    database.execute(db, "INSERT INTO obs VALUES (?, ?)", "O-2", 9.25)
    print(database.tables(db))
    print(database.query(db, "SELECT SUM(value) AS total FROM obs"))
'''

    def test_a_real_database_round_trips(self):
        outcome = run(self.PROGRAM)
        outcome.assert_ran(self)
        self.assertIn("obs", outcome.output)
        self.assertIn("16.75", outcome.output)

    def test_reading_requires_the_read_capability(self):
        # The checker refuses this before the program runs, which is the better
        # place to refuse: the user is told while reading the code rather than
        # when a deployment happens to exercise the path.
        outcome = run(PROGRAMS["read_without_capability"],
                      grants=("DatabaseWrite",))
        self.assertFalse(outcome.compiled)
        self.assertIn("E-capability-missing", outcome.codes())
        self.assertIn("DatabaseRead", outcome.messages())

    def test_a_secret_cannot_be_bound_into_a_statement(self):
        # A parameter binding is a serialization, and spec section 8 restricts
        # serializing secrets.  The checker knows which builtins may receive a
        # secret and refuses the rest, so this is caught before it runs.
        outcome = run(PROGRAMS["bind_a_secret"])
        self.assertFalse(outcome.compiled)
        self.assertIn("E-secret-leak", outcome.codes())

    def test_binding_a_secret_is_refused_at_runtime_too(self):
        # Belt and braces: a value that reaches the binding another way must
        # still be refused, because a static check cannot see through every
        # path and the runtime is the last line.
        from gamag.diagnostics import GamaRuntimeFault
        from gamag.runtime.values import GSecret
        with self.assertRaises(GamaRuntimeFault) as raised:
            enterprise._sql_value(GSecret("classified", "value"))
        self.assertEqual(raised.exception.kind, "SecretLeak")

    def test_a_bad_statement_is_reported_with_the_statement(self):
        outcome = run('''grant DatabaseRead, DatabaseWrite, AuditWrite

fn main() -> Unit
    io
    storage
    let db = database.open(":memory:")
    database.execute(db, "NOT VALID SQL")
''')
        self.assertIsNotNone(outcome.execution.fault)
        self.assertIn("DatabaseError", str(outcome.execution.fault))

    def test_a_closed_database_is_refused(self):
        outcome = run('''grant DatabaseRead, DatabaseWrite, AuditWrite

fn main() -> Unit
    io
    storage
    let db = database.open(":memory:")
    database.close(db)
    print(database.tables(db))
''')
        self.assertIsNotNone(outcome.execution.fault)


class Identity(unittest.TestCase):

    PROGRAM = '''grant AuditWrite

fn main() -> Unit
    io
    audit
    identity.register("ada", "s3cret")
    print(identity.authenticate("ada", "s3cret"))
    print(identity.authenticate("ada", "wrong"))
    print(identity.authenticate("nobody", "s3cret"))
    identity.grant_role("ada", "clinician")
    print(identity.has_role("ada", "clinician"))
    print(identity.roles("ada"))
'''

    def test_credentials_and_roles(self):
        outcome = run(self.PROGRAM)
        outcome.assert_ran(self)
        lines = outcome.output.split()
        self.assertEqual(lines[0], "true")
        self.assertEqual(lines[1], "false")
        self.assertEqual(lines[2], "false")
        self.assertIn("clinician", outcome.output)

    def test_granting_a_role_needs_the_audit_capability(self):
        outcome = run(PROGRAMS["role_without_capability"], grants=())
        self.assertFalse(outcome.compiled)
        self.assertIn("E-capability-missing", outcome.codes())
        self.assertIn("AuditWrite", outcome.messages())

    def test_an_unknown_principal_cannot_be_authenticated(self):
        outcome = run('''fn main() -> Unit
    io
    audit
    print(identity.authenticate("ghost", "x"))
''')
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "false")


class Workflow(unittest.TestCase):

    PROGRAM = '''grant AuditWrite

fn main() -> Unit
    io
    audit
    workflow.define("intake", "receive, triage, treat")
    var run = workflow.start("intake")
    print(workflow.state(run))
    run = workflow.advance(run)
    print(workflow.state(run))
    run = workflow.advance(run)
    print(workflow.state(run), run.done)
    print(workflow.history(run))
'''

    def test_a_run_advances_step_by_step(self):
        outcome = run(self.PROGRAM)
        outcome.assert_ran(self)
        self.assertIn("receive", outcome.output)
        self.assertIn("triage", outcome.output)
        self.assertIn("treat true", outcome.output)

    def test_the_history_explains_the_state(self):
        outcome = run(self.PROGRAM)
        outcome.assert_ran(self)
        history = outcome.output.strip().splitlines()[-1]
        self.assertIn("started at receive", history)
        self.assertIn("advanced to triage", history)
        self.assertIn("advanced to treat", history)

    def test_advancing_past_the_end_does_not_overshoot(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    audit
    workflow.define("one", "only")
    var run = workflow.start("one")
    run = workflow.advance(run)
    run = workflow.advance(run)
    print(workflow.state(run), run.done)
''')
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "only true")

    def test_an_undefined_workflow_is_refused(self):
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    audit
    workflow.start("nothing")
''')
        self.assertIsNotNone(outcome.execution.fault)
        self.assertIn("WorkflowError", str(outcome.execution.fault))


class Messaging(unittest.TestCase):

    def test_publish_and_drain_in_order(self):
        outcome = run('''grant AuditWrite, DatabaseRead

fn main() -> Unit
    io
    storage
    messaging.subscribe("vitals")
    messaging.publish("vitals", "hr=72")
    messaging.publish("vitals", "hr=75")
    print(messaging.depth("vitals"))
    print(messaging.drain("vitals"))
    print(messaging.depth("vitals"))
''')
        outcome.assert_ran(self)
        self.assertIn("2", outcome.output.splitlines()[0])
        self.assertIn("hr=72", outcome.output)
        self.assertEqual(outcome.output.strip().splitlines()[-1], "0")

    def test_publishing_with_no_subscriber_is_refused(self):
        # Silently buffering for a subscriber that never comes is how a queue
        # grows without bound in production.
        outcome = run('''grant AuditWrite

fn main() -> Unit
    io
    storage
    messaging.publish("nobody-listening", "x")
''')
        self.assertIsNotNone(outcome.execution.fault)
        self.assertIn("MessagingError", str(outcome.execution.fault))


class Observability(unittest.TestCase):

    def test_counters_gauges_logs_and_spans_accumulate(self):
        outcome = run('''fn main() -> Unit
    io
    audit
    observability.counter("reads", 2)
    observability.counter("reads", 3)
    observability.gauge("temp", 36.6)
    observability.log("info", "done")
    observability.span("intake", 12.5)
    print(observability.counters())
    print(observability.gauges())
    print(observability.logs())
    print(observability.spans())
''')
        outcome.assert_ran(self)
        self.assertIn("reads=5", outcome.output)
        self.assertIn("temp=36.6", outcome.output)
        self.assertIn("[info] done", outcome.output)
        self.assertIn("intake=12.500ms", outcome.output)

    def test_the_snapshot_is_deterministic(self):
        # Spec section 1.3 leads with reproducibility.  Span ids come from a
        # counter, not from the clock, so two runs produce the same report --
        # and a report that changes between runs cannot be compared with itself.
        source = '''fn main() -> Unit
    io
    audit
    observability.counter("c", 1)
    observability.span("s", 1.0)
    print(observability.snapshot())
'''
        first = run(source)
        second = run(source)
        first.assert_ran(self)
        second.assert_ran(self)
        self.assertEqual(first.output, second.output)

    def test_the_snapshot_is_valid_json(self):
        outcome = run('''fn main() -> Unit
    io
    audit
    observability.counter("c", 1)
    print(observability.snapshot())
''')
        outcome.assert_ran(self)
        payload = json.loads(outcome.output.strip())
        self.assertEqual(payload["counters"], {"c": 1})


class ModuleRegistration(unittest.TestCase):
    """A module that is registered but unreachable is a module nobody can use."""

    def test_every_new_module_is_reachable_from_a_program(self):
        from gamag.std import library as L
        for module in ("fhir", "terminology", "provenance", "consent",
                       "database", "identity", "workflow", "messaging",
                       "observability", "ffi"):
            self.assertIn(module, L.MODULE_TYPE_NAMES,
                          f"`{module}` is registered in the builtin table but "
                          f"not in the checker's module names")

    def test_a_removed_roadmap_entry_is_no_longer_declared_unimplemented(self):
        from gamag.std import library as L
        for module in ("fhir", "terminology", "provenance", "consent",
                       "database", "identity", "workflow", "messaging",
                       "observability"):
            self.assertNotIn(module, L.UNIMPLEMENTED_MODULES,
                             f"`{module}` is implemented but still declared "
                             f"roadmap, which is a false claim about the "
                             f"toolchain (spec section 43)")


if __name__ == "__main__":
    unittest.main()
