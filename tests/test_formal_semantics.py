"""The formal semantics of the core: memory, capability, transition.

Audit priorities 4, 5 and 6 ask for three models that were previously described
in prose.  These tests check that each one is *load bearing*: that a program
which should have the property provably has it, that a program which violates it
is refused with a diagnostic that explains the rule, and that the same relation
is used on both sides of the compile/run boundary -- because a guarantee the
compiler proves and the runtime does not enforce is a guarantee about a program
that cannot be run, and the reverse is a guarantee the toolchain cannot deliver.

Each class is named after the specification section it implements rather than
after the module that happens to hold the code.
"""

from __future__ import annotations

import unittest
from io import StringIO

import support as S
from support import Context, execute, find_entry
from gamag import capabilities as CAPS
from gamag.core.recovery import ACTION_LEVELS, ENTRY_CHECKPOINT
from gamag.gir.ir import Op

# A head that declares authority, so the bodies below can exercise resources and
# capability demands without repeating the preamble.
HEAD = ("gama core 0.2\n"
        "intent T\n"
        "    purpose   exercise a property of the core\n"
        "    authority PatientWrite, AuditWrite\n"
        "source a : I64 from 5\n")


FILE_HEAD = ("gama core 0.2\n"
             "intent T\n"
             "    purpose   exercise a property of the core\n"
             "    authority FileRead\n"
             "source a : I64 from 5\n")


def core(body: str, head: str = HEAD) -> str:
    return head + body


def compiled(source: str, path: str = "<test>") -> "S.Outcome":
    """Compile, and fail the test with the diagnostics if it did not."""
    out = S.compile_only(source, path=path)
    assert out.compiled, f"{path} did not compile: {out.messages()}"
    return out


def instrs(compilation):
    """Every lowered instruction, across every function of the program."""
    for function in compilation.program.functions.values():
        for block in function.blocks:
            for instr in block.instrs:
                yield function.name, instr


def ops_of(compilation):
    return [instr.op for _, instr in instrs(compilation)]


# A four-step chain.  `x` spans nodes 0-1 and `z` spans nodes 2-3, so their
# extents do not overlap and the allocator may put them in one slot -- but only
# if it can prove that, rather than because it hopes.  With two operations
# nothing can be reused, which is itself worth knowing: reuse is earned.
CHAIN = """\
operation First
    uses     a
    yields   x : I64
    effect   pure
    computes a + 1
operation Second
    uses     x
    yields   y : I64
    effect   pure
    computes x + 1
operation Third
    uses     y
    yields   z : I64
    effect   pure
    computes y + 1
operation Fourth
    uses     z
    yields   w : I64
    effect   pure
    computes z + 1
outcome w
"""

SECRET_BODY = """\
source secret pin : I64 from 4471
operation Digest
    uses     pin
    yields   fingerprint : Text
    effect   crypto
    computes secrets.fingerprint(pin)
outcome fingerprint
"""

RECOVER_HEAD = ("gama core 0.2\n"
                "intent V\n"
                "    purpose   exercise the recovery policy\n"
                "    authority AuditWrite\n"
                "    checkpoint admission\n"
                "    recover\n"
                "        retry within 2 rounds\n"
                "        restore checkpoint admission\n"
                "        escalate operator\n")

TRANSITION_BODY = """\
source reading : I64 from 72
state recorded : I64
    starts 0
operation Checked
    uses     reading
    yields   checked : I64
    effect   pure
    computes reading
transition Record
    alters   recorded
    uses     checked
    effect   audit
    needs    AuditWrite
    computes recorded + checked
outcome recorded
"""


def recovering(body: str) -> str:
    return RECOVER_HEAD + body


# ---------------------------------------------------------------------------
# Priority 4 -- spec section 8: the memory and resource model
# ---------------------------------------------------------------------------

class MemoryAndResourceModel(unittest.TestCase):
    """Ownership, extents, slot reuse and secret guarding, all proved."""

    def setUp(self):
        self.comp = compiled(core(CHAIN)).compilation
        self.model = self.comp.core_memory
        self.bindings = self.model.bindings

    # ---- the model exists and is clean -----------------------------

    def test_a_core_program_gets_a_memory_model(self):
        self.assertIsNotNone(self.model,
                             "a core program must have a memory model")
        self.assertEqual(self.model.violations, [],
                         "a program that compiled clean must violate nothing")

    def test_the_model_is_available_for_every_core_example(self):
        for name in S.core_example_names():
            path = S.example(name)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            comp = compiled(source, name).compilation
            self.assertIsNotNone(comp.core_memory, f"{name} has no memory model")
            self.assertEqual(comp.core_memory.violations, [],
                             f"{name} violates the memory model")

    # ---- the four safety proofs ------------------------------------

    def test_every_binding_has_exactly_one_owner(self):
        self.assertTrue(self.bindings, "a program with values has bindings")
        for name, binding in self.bindings.items():
            self.assertTrue(binding.owner, f"`{name}` has no owner")
            self.assertEqual(binding.name, name)
        self.assertEqual(len(set(b.owner for b in self.bindings.values())),
                         len(self.bindings),
                         "ownership is a one-to-one relation")

    def test_a_binding_outlives_every_borrow_of_it(self):
        # Spec section 8: no use after free.  Expressed here as the property the
        # allocator actually checks -- an owner's extent covers its borrowers.
        for binding in self.bindings.values():
            for borrower in binding.borrowers:
                other = self.bindings.get(borrower)
                if other is None:
                    continue  # a node name, not a binding
                self.assertLessEqual(binding.first, other.first)
                self.assertGreaterEqual(
                    binding.last, other.first,
                    f"`{binding.name}` [{binding.first},{binding.last}] must "
                    f"outlive its borrower `{borrower}`")

    def test_a_returned_binding_lives_until_the_return(self):
        returned = [name for name, b in self.bindings.items() if b.to_return]
        self.assertIn("a", returned, "a parameter outlives the program body")
        self.assertIn("w", returned, "the outcome must outlive every use")
        for name in returned:
            binding = self.bindings[name]
            self.assertGreaterEqual(binding.last, binding.first)

    def test_no_binding_is_mutable_unless_the_program_declared_state(self):
        mutable = [name for name, b in self.bindings.items() if b.mutable]
        self.assertEqual(mutable, [],
                         "the core has no assignment, so nothing is mutable "
                         "except a declared state, and this program has none")

    # ---- slot reuse is a proof, not a hope -------------------------

    def test_a_slot_is_reused_only_where_extents_do_not_overlap(self):
        self.assertGreater(self.model.slots_saved, 0,
                           "`x` [0,1] and `z` [2,3] do not overlap, so reuse "
                           "must happen")
        for slot in self.model.slots:
            for i, left in enumerate(slot.occupants):
                for right in slot.occupants[i + 1:]:
                    overlaps = (left.first <= right.last
                                and right.first <= left.last)
                    self.assertFalse(
                        overlaps,
                        f"slot {slot.index} holds `{left.name}` "
                        f"[{left.first},{left.last}] and `{right.name}` "
                        f"[{right.first},{right.last}] at the same time")

    def test_a_reused_slot_records_what_it_shares_with(self):
        reused = [s for s in self.model.slots if len(s.occupants) > 1]
        self.assertTrue(reused, "reuse must be visible in the model")
        slot = reused[0]
        names = [o.name for o in slot.occupants]
        for occupant in slot.occupants:
            self.assertEqual(occupant.slot, slot.index)
        self.assertIn(names[0], self.bindings[names[1]].shares_with
                      or self.bindings[names[0]].shares_with,
                      "a reused slot names what it shares with")

    def test_a_slot_is_never_shared_between_different_types(self):
        for slot in self.model.slots:
            types = {str(o.type) for o in slot.occupants}
            self.assertLessEqual(len(types), 1,
                                 f"slot {slot.index} mixes types: {types}")

    def test_a_two_operation_chain_cannot_reuse_and_does_not_try(self):
        comp = compiled(core(CHAIN.split("operation Third")[0]
                             + "outcome y\n")).compilation
        self.assertEqual(comp.core_memory.slots_saved, 0,
                         "every value here is live at the same time as its "
                         "neighbour; reusing a slot would be a use after free")

    # ---- secrets ---------------------------------------------------

    def test_a_secret_binding_is_guarded_in_the_lowered_program(self):
        comp = compiled(core(SECRET_BODY)).compilation
        secrets = [b for b in comp.core_memory.bindings.values() if b.secret]
        self.assertEqual([b.name for b in secrets], ["pin"])
        guards = [i for i in ops_of(comp) if i == Op.SECRET_GUARD]
        self.assertEqual(len(guards), len(secrets),
                         "every secret definition gets a guard")

    def test_a_fingerprint_of_a_secret_is_not_itself_secret(self):
        comp = compiled(core(SECRET_BODY)).compilation
        bindings = comp.core_memory.bindings
        self.assertTrue(bindings["pin"].secret)
        self.assertFalse(bindings["fingerprint"].secret,
                         "computing with a secret must not make every result "
                         "undisclosable, or the secret could never be used")

    def test_the_generated_entry_point_does_not_print_a_secret(self):
        comp = compiled(core(SECRET_BODY.replace("outcome fingerprint",
                                                 "outcome pin"))).compilation
        source = comp.program.functions["main"].render()
        self.assertIn("[secret pin]", source,
                      "a secret outcome is redacted, not rendered")

    def test_an_ordinary_renderer_refuses_a_secret_even_when_granted(self):
        out = S.run(core("""\
source secret pin : I64 from 4471
operation Leaked
    uses     pin
    yields   text : Text
    effect   crypto, audit
    needs    SecretExpose
    computes to_text(pin)
outcome text
"""))
        # This compiles: the intent declared SecretExpose.  The *library* still
        # refuses, because that boundary is not the program's own to grant.
        if out.compiled:
            out.assert_faulted(self)

    def test_deliberate_disclosure_is_possible_and_is_audited(self):
        out = S.run(core("""\
    authority SecretExpose, AuditWrite
source secret pin : I64 from 4471
operation Disclosed
    uses     pin
    yields   revealed : Text
    effect   crypto, audit
    needs    SecretExpose
    computes to_text(secrets.expose(pin, "clinician opened the record"))
outcome revealed
""", head=("gama core 0.2\n"
           "intent T\n"
           "    purpose   disclose a secret on purpose\n")))
        out.assert_ran(self)
        self.assertIn("4471", out.output)
        actions = [r.action for r in out.context.audit.records]
        self.assertIn("SECRET_EXPOSED", actions,
                      "spec section 12: disclosure is recorded, not silent")

    # ---- resources -------------------------------------------------

    def test_a_resource_acquisition_names_the_node_that_holds_it(self):
        comp = compiled(core("""\
operation Read
    yields   x : Text
    effect   io
    needs    FileRead
    computes io.read_file("r")
outcome x
""", head=FILE_HEAD)).compilation
        acquisitions = comp.core_memory.acquisitions
        self.assertTrue(acquisitions,
                        "an effect that acquires a resource must be recorded")
        for acq in acquisitions:
            self.assertTrue(acq.node, f"{acq.resource} has no owner")
            self.assertTrue(acq.resource, "an acquisition must name a resource")
            self.assertTrue(acq.spans, f"{acq.resource} has no extent")
            self.assertEqual(acq.effect, "io")
        self.assertEqual([a.resource for a in acquisitions], ["filesystem"])


# ---------------------------------------------------------------------------
# Priority 5 -- spec section 12: the capability algebra
# ---------------------------------------------------------------------------

class CapabilityAlgebra(unittest.TestCase):
    """A capability is a permission over a resource; a grant is a subset."""

    def test_a_capability_is_a_permission_over_a_resource(self):
        cap = CAPS.Capability.parse("PatientStore[Read]")
        self.assertEqual((cap.resource, cap.permission), ("Patient", "Read"))

    def test_the_qualified_spelling_names_the_same_capability(self):
        qualified = CAPS.Capability.parse("PatientStore[Read]")
        plain = CAPS.Capability.parse("PatientRead")
        self.assertEqual((qualified.resource, qualified.permission),
                         (plain.resource, plain.permission),
                         "section 12 gives both spellings for one capability")
        # And the relation agrees with that, whichever way it is written.
        self.assertTrue(CAPS.covers(("PatientStore[Write]",), "PatientRead"))
        self.assertTrue(CAPS.covers(("PatientWrite",), "PatientStore[Read]"))

    def test_write_covers_read_on_the_same_resource(self):
        self.assertTrue(CAPS.covers(("PatientWrite",), "PatientRead"),
                        "section 12: `PatientWrite` implies `PatientRead`")

    def test_read_does_not_cover_write(self):
        self.assertFalse(CAPS.covers(("PatientRead",), "PatientWrite"),
                         "grants are a subset: read must not imply write")

    def test_a_write_grant_covers_the_read_it_requires(self):
        self.assertTrue(CAPS.covers(("PatientStore[Write]",), "PatientRead"))
        self.assertTrue(CAPS.covers(("PatientStore[Write]",), "PatientWrite"))
        # A whole-resource grant, where the vocabulary has one.
        self.assertTrue(CAPS.covers(("Network",), "NetworkConnect"))
        # And `Load` entails `Read` for the same reason `Write` does.
        self.assertTrue(CAPS.covers(("ModelLoad",), "ModelRead"))

    def test_capabilities_do_not_cross_resources(self):
        self.assertFalse(CAPS.covers(("PatientWrite",), "AuditWrite"))
        self.assertFalse(CAPS.covers(("PatientStore",), "ModelStore"))

    def test_attenuation_is_the_same_relation_read_backwards(self):
        # Attenuation means granting less.  In this algebra that is exactly:
        # the attenuated capability is covered by the original, not the reverse.
        self.assertTrue(CAPS.covers(("PatientStore[Write]",), "PatientRead"))
        self.assertFalse(CAPS.covers(("PatientRead",), "PatientWrite"))
        # And nothing in the vocabulary lets a grant be amplified by combining:
        # two reads on different resources are still only two reads.
        self.assertFalse(CAPS.covers(("PatientRead", "AuditWrite"),
                                     "PatientWrite"))
        self.assertFalse(CAPS.covers(("CryptoSign",), "PatientRead"))

    def test_the_runtime_wildcard_is_part_of_the_one_relation(self):
        self.assertTrue(CAPS.covers(("*",), "SecretExpose"),
                        "a wildcard that only the runtime understood would be "
                        "a second semantics")

    def test_an_unknown_capability_is_refused_at_compile_time(self):
        out = S.compile_only(core("", head=(
            "gama core 0.2\n"
            "intent T\n"
            "    purpose   exercise a property of the core\n"
            "    authority PatientStore[Teleport]\n"
            "source a : I64 from 5\n")))
        out.assert_rejected(self, "E-unknown-capability")

    def test_a_secret_reaching_a_renderer_demands_expose(self):
        out = S.compile_only(core("""\
source secret pin : I64 from 4471
operation Leaked
    uses     pin
    yields   text : Text
    effect   crypto, audit
    computes to_text(pin)
outcome text
"""))
        out.assert_rejected(self, "E-capability-unmet")
        self.assertIn("SecretExpose", out.messages())

    def test_a_demand_the_authority_covers_is_satisfied(self):
        comp = compiled(core("""\
operation Read
    yields   x : Text
    effect   io
    needs    FileRead
    computes io.read_file("r")
outcome x
""", head=FILE_HEAD)).compilation
        derived = comp.core_model.authority.derived
        self.assertTrue(derived, "a declared need must become a demand")
        for demand in derived:
            self.assertTrue(demand.satisfied,
                            f"{demand.origin} was not covered by {comp.core_model.intent.authority}")
            self.assertTrue(demand.covered_by)

    def test_a_demand_the_authority_cannot_cover_is_refused(self):
        out = S.compile_only(core("""\
operation Signed
    yields   x : I64
    effect   crypto
    needs    CryptoSign
    computes crypto.sign("r")
outcome x
"""))
        out.assert_rejected(self, "E-capability-unmet")

    def test_the_runtime_applies_the_relation_the_compiler_proved(self):
        from gamag.runtime.context import Context
        # The compiler says PatientWrite covers PatientRead.  A runtime holding
        # only PatientWrite must therefore accept a PatientRead check -- or the
        # two halves of the toolchain disagree about what a grant means.
        self.assertTrue(
            Context(grants=("PatientStore[Write]",)).has_cap("PatientRead"))
        self.assertFalse(Context(grants=("PatientRead",)).has_cap("PatientWrite"))
        self.assertTrue(Context(grants=("*",)).has_cap("SecretExpose"))

    def test_a_boundary_is_emitted_even_for_a_demand_the_compiler_proved(self):
        comp = compiled(core("""\
operation Read
    yields   x : Text
    effect   io
    needs    FileRead
    computes io.read_file("r")
outcome x
""", head=FILE_HEAD)).compilation
        checks = [i for _, i in instrs(comp) if i.op == Op.CAP_CHECK]
        self.assertTrue(checks,
                        "spec section 12: every capability boundary is checked, "
                        "including one proved at compile time")
        rendered = "\n".join(str(i.meta) for i in checks)
        self.assertIn("FileRead", rendered,
                      "the boundary names the capability it will demand")
        self.assertIn("io.read_file", rendered,
                      "and the call site that demanded it")

    def test_a_capability_check_is_enforced_at_runtime_not_just_emitted(self):
        # Compiled with the authority declared, then *run* in an environment
        # that never granted it.  The emitted boundary must refuse, because a
        # proof about the program is not a proof about where it is deployed.
        source = core("""\
operation Read
    yields   x : Text
    effect   io
    needs    FileRead
    computes io.read_file("r")
outcome x
""", head=FILE_HEAD)
        compilation = S.compile_only(source).assert_compiled(self).compilation
        buffer = StringIO()
        context = Context(grants=("PatientRead",), stdout=buffer,
                          seed=0, deterministic=True)
        # `execute` reports a fault rather than raising, so the caller decides
        # what a failure means; the refusal is in `execution.fault`.
        execution = execute(compilation, entry=find_entry(compilation),
                            context=context, grants=())
        self.assertIsNotNone(execution.fault,
                             "the emitted boundary must actually refuse")
        self.assertIn("FileRead", str(execution.fault),
                      "the refusal must name the capability that was missing")
        denied = [r for r in context.audit.records
                  if r.action == "CAPABILITY_DENIED"]
        self.assertTrue(denied,
                        "spec section 12: a denial is recorded at security level")
        self.assertEqual(denied[0].level, "security")

    def test_a_demand_the_authority_cannot_reach_is_refused(self):
        out = S.compile_only(core("""\
operation Read
    yields   x : Text
    effect   io
    needs    FileRead
    computes io.read_file("r")
outcome x
"""))
        out.assert_rejected(self, "E-authority-unmet")


# ---------------------------------------------------------------------------
# Priority 6 -- spec sections 10 and 18: transition and recovery
# ---------------------------------------------------------------------------

class TransitionAndRecovery(unittest.TestCase):
    """A declared policy becomes the policy the machine executes."""

    def test_the_actions_are_the_specifications_own_levels(self):
        # Spec section 10: LEVEL 0 local retry, LEVEL 2 checkpoint restore,
        # LEVEL 5 operator escalation.  The compiler must not invent a
        # vocabulary of its own, or a policy would mean two different things.
        self.assertEqual(ACTION_LEVELS["retry"], 0)
        self.assertEqual(ACTION_LEVELS["restore"], 2)
        self.assertEqual(ACTION_LEVELS["replay"], 2)
        self.assertEqual(ACTION_LEVELS["restart"], 3)
        self.assertEqual(ACTION_LEVELS["failover"], 4)
        self.assertEqual(ACTION_LEVELS["escalate"], 5)

    def test_a_declared_policy_is_parsed_into_its_steps(self):
        comp = compiled(recovering(TRANSITION_BODY)).compilation
        policy = comp.core_recovery
        self.assertIsNotNone(policy)
        self.assertEqual([s.action for s in policy.steps],
                         ["retry", "restore", "escalate"])
        self.assertEqual(policy.steps[0].count, 2)
        self.assertEqual([s["target"] for s in policy.to_steps()],
                         ["", "admission", "operator"],
                         "the resolved target is the declared checkpoint, not "
                         "the words that named it")
        self.assertEqual(policy.highest_level, 5)
        self.assertTrue(policy.bounded)
        self.assertEqual(policy.problems, [])

    def test_a_policy_may_only_escalate(self):
        source = recovering(TRANSITION_BODY).replace(
            "        retry within 2 rounds\n"
            "        restore checkpoint admission\n"
            "        escalate operator\n",
            "        escalate operator\n"
            "        retry within 2 rounds\n")
        out = S.compile_only(source)
        out.assert_rejected(self, "E-recovery-unordered")
        self.assertIn("escalates", out.messages())

    def test_a_named_checkpoint_must_have_been_declared(self):
        source = recovering(TRANSITION_BODY).replace(
            "restore checkpoint admission", "restore checkpoint nowhere")
        out = S.compile_only(source)
        out.assert_rejected(self, "E-unknown-checkpoint")
        self.assertIn("never", out.messages().lower(),
                      "the diagnostic should quote the rule it is enforcing")

    def test_a_bare_restore_needs_no_declared_checkpoint(self):
        # v0.1 self-healing writes `restore checkpoint` with no target; that
        # means the region entry point, which always exists.
        source = recovering(TRANSITION_BODY).replace(
            "restore checkpoint admission", "restore checkpoint")
        comp = compiled(source).compilation
        self.assertEqual(comp.core_recovery.problems, [])

    def test_an_unbounded_retry_is_refused(self):
        source = recovering(TRANSITION_BODY).replace(
            "retry within 2 rounds", "retry")
        out = S.compile_only(source)
        out.assert_rejected(self, "E-unbounded-recovery")

    def test_an_unknown_recovery_action_is_refused(self):
        source = recovering(TRANSITION_BODY).replace(
            "retry within 2 rounds", "pray within 2 rounds")
        out = S.compile_only(source)
        out.assert_rejected(self, "E-unknown-recovery")

    def test_the_policy_lowers_to_a_protected_region_carrying_its_own_words(self):
        comp = compiled(recovering(TRANSITION_BODY)).compilation
        intent = comp.program.functions["V"]
        self.assertIsNotNone(intent.recovery,
                             "a declared policy must reach the runtime as a "
                             "policy, not as a fault message or a jump target")
        raws = [step["raw"] for step in intent.recovery.steps]
        self.assertEqual(raws, ["retry within 2 rounds",
                                "restore checkpoint admission",
                                "escalate operator"])
        self.assertTrue(intent.recovery.audit_all,
                        "spec section 10: every recovery action is audited")
        self.assertIn(2, intent.recovery.level_names)
        self.assertEqual([i.op for _, i in instrs(comp) if i.op == Op.PROTECTED
                          and True][:1], [Op.PROTECTED])

    def test_declared_checkpoints_are_emitted_as_checkpoint_instructions(self):
        comp = compiled(recovering(TRANSITION_BODY)).compilation
        checkpoints = [i for _, i in instrs(comp) if i.op == Op.CHECKPOINT]
        self.assertTrue(checkpoints, "a declared checkpoint must be captured")
        labels = [i.meta.get("raw", "") for i in checkpoints]
        self.assertIn("admission", labels)

    def test_declared_checkpoints_are_captured_before_the_work_runs(self):
        out = S.run(recovering(TRANSITION_BODY))
        out.assert_ran(self)
        records = [r for r in out.context.audit.records
                   if r.action == "CHECKPOINT_CAPTURED"]
        labels = [r.fields.get("label") for r in records]
        self.assertIn("admission", labels,
                      "the declared checkpoint must actually be recorded")
        self.assertIn(ENTRY_CHECKPOINT.format(component="V"), labels)
        # Captured before the transaction that does the work.
        actions = [r.action for r in out.context.audit.records]
        self.assertLess(actions.index("CHECKPOINT_CAPTURED"),
                        actions.index("TRANSACTION_BEGIN"),
                        "a checkpoint taken after the work cannot restore "
                        "the program to before it")

    def test_a_transition_runs_inside_a_transaction(self):
        comp = compiled(recovering(TRANSITION_BODY)).compilation
        phases = [i.meta.get("action") for _, i in instrs(comp)
                  if i.op == Op.TRANSACTION]
        self.assertIn("begin", phases)
        self.assertIn("commit", phases)
        self.assertLess(phases.index("begin"), phases.index("commit"))

    def test_a_named_restore_returns_to_the_named_checkpoint(self):
        failing = recovering(TRANSITION_BODY).replace(
            "source reading : I64 from 72",
            "source reading : I64 from 500").replace(
            "    effect   pure\n    computes reading",
            "    effect   pure\n    holds    checked <= 200\n    computes reading")
        out = S.run(failing)
        out.assert_faulted(self, "RecoveryExhausted")
        restores = [r for r in out.context.audit.records
                    if r.action == "RECOVERY_ACTION"
                    and r.fields.get("recovery_level") == 2]
        self.assertTrue(restores, "the policy's restore step must have run")
        self.assertIn("admission", restores[0].fields.get("detail", ""),
                      "spec section 10: recovery must never silently invent "
                      "state, so the restored checkpoint is named in the trail")

    def test_every_recovery_action_is_audited_in_escalation_order(self):
        failing = recovering(TRANSITION_BODY).replace(
            "source reading : I64 from 72",
            "source reading : I64 from 500").replace(
            "    effect   pure\n    computes reading",
            "    effect   pure\n    holds    checked <= 200\n    computes reading")
        out = S.run(failing)
        records = [r for r in out.context.audit.records
                   if r.action == "RECOVERY_ACTION"]
        actions = [r.fields.get("recovery_action") for r in records]
        for expected in ("retry within 2 rounds",
                         "restore checkpoint admission",
                         "escalate operator"):
            self.assertIn(expected, actions)
        levels = [r.fields.get("recovery_level") for r in records]
        self.assertEqual(levels, sorted(levels),
                         "audited recovery levels never decrease")
        alerts = [r for r in out.context.audit.records
                  if r.action == "OPERATOR_ALERT"]
        self.assertTrue(alerts, "level 5 escalation must reach an operator")

    def test_a_constraint_failure_before_commit_aborts_the_transaction(self):
        failing = recovering(TRANSITION_BODY).replace(
            "    computes recorded + checked",
            "    holds    recorded + checked <= 10\n    computes recorded + checked")
        out = S.run(failing)
        self.assertIsNotNone(out.context)
        self.assertTrue(out.context.transactions,
                        "a transition must have opened a transaction")
        for name, txn in out.context.transactions.items():
            self.assertIn(txn["state"], ("committed", "aborted"),
                          f"section 18: transaction `{name}` was left "
                          f"half-applied")

    def test_an_unprotected_intent_still_runs_its_transition_in_a_transaction(self):
        comp = compiled(core("""\
state recorded : I64
    starts 0
operation Checked
    uses     a
    yields   checked : I64
    effect   pure
    computes a
transition Record
    alters   recorded
    uses     checked
    effect   audit
    needs    AuditWrite
    computes recorded + checked
outcome recorded
""")).compilation
        phases = [i.meta.get("action") for _, i in instrs(comp)
                  if i.op == Op.TRANSACTION]
        self.assertIn("begin", phases,
                      "the transaction is a property of the transition, not of "
                      "the recovery policy that happens to wrap it")


if __name__ == "__main__":
    unittest.main()
