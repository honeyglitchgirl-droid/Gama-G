"""What a `holds` promise can be established from before anything runs.

`holds` is a promise the program makes about a value it produces.  Until v1.3
the compiler only *quoted* that promise when it broke: every clause became an
``Op.REQUIRE`` and the runtime decided.  That was correct and it was weak -- a
program whose own constants make the promise inevitable was checked exactly like
one that might fail, and a program whose promise is impossible compiled without
a word said.

These tests are about the two arguments the prover is allowed to make, and --
more importantly -- about the four it must refuse.  A prover that overclaims is
worse than one that does nothing, because the reader stops checking.  So half
of what follows asserts that a verdict was *not* reached, and that the reason
recorded names the real obstacle.

The runtime check is asserted to survive every proof.  That is the policy of
`native.capability_checks` extended to promises: a proof is a reason to trust
the program, not a reason to remove the boundary.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest

import support as S

from gamag.cli.main import main
from gamag.core import contractproof
from gamag.core import mir as M
from gamag.gir.ir import Op

HEAD = "gama core 0.2\nintent T\n    purpose    test\n"


def core(body: str, head: str = HEAD) -> str:
    return head + body


def _compile(source: str):
    """Compile without running: a core intent needs no entry point here."""
    out = S.compile_only(source)
    if not out.compilation.ok:
        raise AssertionError("program did not compile:\n" + out.messages())
    return out


def holds(out):
    """The `holds` clauses as the model records them, in source order."""
    model = out.compilation.core_model
    return [c for c in model.constraints.constraints if c.kind == "holds"]


def require_messages(out):
    """Every ``Op.REQUIRE`` the lowering actually emitted, by its message."""
    messages = []
    for fn in out.compilation.program.functions.values():
        for block in fn.blocks:
            for instr in block.instrs:
                if instr.op == Op.REQUIRE:
                    messages.append(instr.meta.get("message", ""))
    return messages


class Constants(unittest.TestCase):
    """A closed expression is evaluated, not estimated."""

    def test_a_promise_the_constants_make_inevitable_is_proven(self):
        out = _compile(core("""
source a : I64 from 5

operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b > 3
    computes a

outcome b
"""))
        clause = holds(out)[0]
        self.assertEqual(clause.discharge, M.DISCHARGE_PROVEN)
        self.assertIn("evaluates to true", clause.proof)
        # the note names the value, so the reader can check the arithmetic
        self.assertIn("`b` = 5", clause.proof)

    def test_a_proven_promise_still_keeps_its_runtime_check(self):
        out = _compile(core("""
source a : I64 from 5

operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b > 3
    computes a

outcome b
"""))
        self.assertEqual(holds(out)[0].discharge, M.DISCHARGE_PROVEN)
        self.assertIn("holds `b > 3`", require_messages(out),
                      "a discharged promise must still be enforced: the check "
                      "is what a hand-edited IR meets")

    def test_a_refuted_promise_is_said_out_loud(self):
        out = _compile(core("""
source a : I64 from 5

operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b < 3
    computes a

outcome b
"""))
        codes = [d.code for d in out.compilation.warnings]
        self.assertIn("W-contract-refuted", codes,
                      f"the compiler knows this promise cannot hold and said "
                      f"nothing: {out.messages()}")
        self.assertIn("`b` = 5", out.messages(),
                      "a warning that says `impossible` without saying why is "
                      "not actionable")

    def test_a_refuted_promise_still_compiles_and_still_faults(self):
        # A refutation is information, not a verdict on the program's right to
        # run: the classified fault the specification asks for stays.
        out = S.run(core("""
source a : I64 from 5

operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b < 3
    computes a

outcome b
"""))
        out.assert_faulted(self, "ContractViolation")
        self.assertIn("b < 3", str(out.fault.message))

    def test_float_arithmetic_of_constants_is_folded(self):
        # The fold is not an interval argument, so NaN is not in play: the
        # values are the ones the runtime would have computed.
        out = _compile(core("""
source weight : F64 from 34.5
source factor : F64 from 15.0

operation RawDose
    uses     weight, factor
    yields   rawDose : F64
    effect   pure
    holds    rawDose >= 0.0
    computes weight * factor

outcome rawDose
"""))
        self.assertEqual(holds(out)[0].discharge, M.DISCHARGE_PROVEN)
        self.assertIn("517.5", holds(out)[0].proof)

    def test_text_promises_are_folded_too(self):
        out = _compile(core("""
operation Label
    yields   band : Text
    effect   pure
    holds    band != ""
    computes "urgent"

outcome band
"""))
        self.assertEqual(holds(out)[0].discharge, M.DISCHARGE_PROVEN)


class Refusals(unittest.TestCase):
    """The fragment ends where a second copy of the runtime would begin."""

    def test_division_is_not_folded(self):
        out = _compile(core("""
source n : F64 from 4.0

operation Half
    uses     n
    yields   h : F64
    effect   pure
    holds    h > 0.0
    computes n / 2.0

outcome h
"""))
        clause = holds(out)[0]
        self.assertEqual(clause.discharge, M.DISCHARGE_RUNTIME)
        self.assertIn("`/` is not folded", clause.proof)

    def test_a_standard_library_call_is_not_modelled(self):
        out = _compile(core("""
source rawDose : F64 from 517.5

operation Dose
    uses     rawDose
    yields   dose : F64
    effect   pure
    holds    dose <= 500.0
    computes math.clamp(rawDose, 0.0, 500.0)

outcome dose
"""))
        clause = holds(out)[0]
        self.assertEqual(clause.discharge, M.DISCHARGE_RUNTIME)
        self.assertIn("math.clamp", clause.proof)
        self.assertIn("call", clause.proof)

    def test_a_refined_binding_is_left_to_the_runtime(self):
        out = _compile(core("""
source n : F64 from 2.0

refine Guess
    uses     n
    yields   g : F64
    effect   pure
    holds    g > 0.0
    starts   n / 2.0
    repeats  (g + n / g) / 2.0
    until    abs(g * g - n) < 0.000001
    within   60 rounds

outcome g
"""))
        clause = holds(out)[0]
        self.assertEqual(clause.discharge, M.DISCHARGE_RUNTIME)
        self.assertIn("refined", clause.proof)
        self.assertIn("every round", clause.proof)

    def test_a_state_binding_is_never_assumed_constant(self):
        out = _compile(core("""
state balance : I64
    starts 0

source amount : I64 from 20

transition Deposit
    alters   balance
    uses     amount
    effect   pure
    holds    amount >= 0
    computes balance + amount

outcome balance
"""))
        by_node = {c.node: c for c in holds(out)}
        self.assertEqual(by_node["Deposit"].discharge, M.DISCHARGE_PROVEN,
                         "`amount` is a source with a literal origin, so the "
                         "promise about it is decidable")

    def test_a_state_promise_is_refused_as_a_state(self):
        out = _compile(core("""
state balance : I64
    starts 0

source amount : I64 from -20

transition Withdraw
    alters   balance
    uses     amount
    effect   pure
    holds    balance >= 0
    computes balance + amount

outcome balance
"""))
        clause = [c for c in holds(out) if "balance" in c.text][0]
        self.assertEqual(clause.discharge, M.DISCHARGE_RUNTIME)
        self.assertIn("is a `state`", clause.proof)

    def test_a_constant_that_does_not_fit_its_type_is_not_a_fact(self):
        # `vm.store` raises IntegerOverflow on the way in, so the operation
        # faults before the promise is ever tested.  A folded constant that does
        # not fit its type therefore proves nothing, and the prover declines it
        # even though the arithmetic in front of it is exact.
        self.assertFalse(contractproof._fits("I8", 300))
        self.assertFalse(contractproof._fits("U8", -1))
        self.assertTrue(contractproof._fits("I8", 127))
        self.assertTrue(contractproof._fits("F64", 1e300),
                        "floats have no declared range to break; the fold "
                        "stands for them")
        self.assertTrue(contractproof._fits("", 300),
                        "no type means no bounds, and no bounds means no "
                        "overflow claim either way")
        out = S.compile_only(core("""
source big : I8 from 300

operation B
    uses     big
    yields   b : I8
    effect   pure
    holds    b > 3
    computes big

outcome b
"""))
        # the checker gets there first, and says so in its own words
        self.assertFalse(out.compilation.ok, out.messages())
        self.assertIn("I8", out.messages())


class Intervals(unittest.TestCase):
    """One sized-integer binding, its guard, and its type's range."""

    def test_the_range_of_a_type_discharges_a_promise_about_it(self):
        out = _compile(core("""
source u : U8

operation Shift
    uses     u
    yields   seen : U8
    effect   pure
    holds    u >= 0
    computes u

outcome seen
"""))
        clause = holds(out)[0]
        self.assertEqual(clause.discharge, M.DISCHARGE_PROVEN)
        self.assertIn("type's own range", clause.proof)

    def test_a_when_guard_discharges_a_promise_it_implies(self):
        out = _compile(core("""
source score : I64

operation High
    uses     score
    yields   band : Text
    effect   pure
    holds    score > 0
    when     score >= 75
    computes "high"

operation Low
    uses     score
    yields   band : Text
    effect   pure
    when     not (score >= 75)
    computes "low"

outcome band
"""))
        clause = [c for c in holds(out) if c.node == "High"][0]
        self.assertEqual(clause.discharge, M.DISCHARGE_PROVEN)
        self.assertIn("`when` guard is what establishes it", clause.proof)

    def test_a_guard_about_another_binding_is_not_assumed(self):
        out = _compile(core("""
source score : I64

operation A
    uses     score
    yields   other : I64
    effect   pure
    holds    other >= 0
    when     score >= 75
    computes score - 75

outcome other
"""))
        clause = holds(out)[0]
        self.assertEqual(clause.discharge, M.DISCHARGE_RUNTIME)
        # whichever explanation the prover reaches, it must name the guard it
        # declined to assume rather than shrug
        self.assertIn("the guard bounds `score`", clause.proof,
                      f"the note should say which binding the guard bounds: "
                      f"{clause.proof}")

    def test_a_float_subject_is_refused_a_bounds_argument(self):
        out = _compile(core("""
source temp : F64

operation Warm
    uses     temp
    yields   ok : Bool
    effect   pure
    holds    temp >= 0.0
    when     temp >= 0.0
    computes true

outcome ok
"""))
        clause = holds(out)[0]
        self.assertEqual(clause.discharge, M.DISCHARGE_RUNTIME)
        self.assertIn("NaN", clause.proof)

    def test_a_promise_both_reachable_and_unreachable_stays_runtime(self):
        out = _compile(core("""
source score : I64

operation A
    uses     score
    yields   band : Text
    effect   pure
    holds    score >= 0
    when     score >= 75
    computes "x"

operation B
    uses     score
    yields   band : Text
    effect   pure
    when     not (score >= 75)
    computes "y"

outcome band
"""))
        clause = [c for c in holds(out) if c.node == "A"][0]
        self.assertEqual(clause.discharge, M.DISCHARGE_PROVEN,
                         "`score >= 0` follows from `score >= 75`, so an "
                         "unprovable verdict here would be the prover hiding "
                         "a proof it has")


class TheExport(unittest.TestCase):
    """The obligations left over, handed to a solver nobody here runs."""

    PROGRAM = """
source score : I64

operation A
    uses     score
    yields   band : Text
    effect   pure
    holds    score > 0
    when     score >= 75
    computes "x"

operation B
    uses     score
    yields   band : Text
    effect   pure
    holds    score < -1000
    when     not (score >= 75)
    computes "y"

outcome band
"""

    def text(self):
        out = _compile(core(self.PROGRAM))
        return contractproof.to_smtlib2(out.compilation.core_model)

    def test_the_export_states_the_question_in_standard_form(self):
        text = self.text()
        self.assertIn("(set-logic QF_LIA)", text)
        self.assertIn("(declare-fun score () Int)", text)
        # the type's own range is an assumption, not a wish
        self.assertIn("(assert (and (>= score -9223372036854775808)", text)
        # B's guard is `not (score >= 75)`: the export asserts what the guard
        # says, not what a reader might infer from it
        self.assertIn("(assert (not (>= score 75)))   ; when `not (score >= 75)`",
                      text)
        self.assertIn("(assert (not (< score (- 1000))))", text)
        self.assertNotIn("holds `score > 0`", text,
                        "A's promise was discharged by the prover, so it is "
                        "not a question to ask a solver")
        self.assertIn("(check-sat)", text)

    def test_the_export_records_what_it_cannot_encode_instead_of_guessing(self):
        out = _compile(core("""
source rawDose : F64 from 517.5

operation Dose
    uses     rawDose
    yields   dose : F64
    effect   pure
    holds    dose <= 500.0
    computes math.clamp(rawDose, 0.0, 500.0)

outcome dose
"""))
        text = contractproof.to_smtlib2(out.compilation.core_model)
        self.assertIn("; not encoded", text)
        self.assertIn("dose <= 500.0", text)
        self.assertNotIn("(declare-fun dose", text,
                        "a float binding must not be declared as an Int to fit "
                        "the encoding into a shape the prover can answer")

    def test_the_export_never_claims_a_solver_ran(self):
        text = self.text()
        self.assertIn("never runs a solver", text)
        self.assertIn("unanswered question, not a result", text)

    def test_a_discharged_clause_is_not_exported_as_a_question(self):
        # A and B both carry a `holds`; only the one the prover could not
        # settle belongs in the file.
        out = _compile(core(self.PROGRAM))
        text = contractproof.to_smtlib2(out.compilation.core_model)
        self.assertEqual(text.count("(check-sat)"), 1,
                         "the export asked about a promise the compiler had "
                         "already established")


class Commands(unittest.TestCase):
    """`ggc check --smt` and the notes every report shows."""

    @staticmethod
    def run_cli(*argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 0
        return code, out.getvalue(), err.getvalue()

    def test_check_writes_the_export_to_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "a.gg")
            target = os.path.join(tmp, "a.smt2")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(core(self.UNSETTLED))
            code, out, err = self.run_cli("check", source, "--smt", target)
            self.assertEqual(code, 0, err)
            with open(target, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("(set-logic QF_LIA)", text)
            self.assertIn("verification condition", out)

    def test_a_v01_program_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "hello.gg")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write('fn main() -> Unit\n    io\n    print(1)\n')
            code, out, err = self.run_cli("check", source,
                                          "--smt", os.path.join(tmp, "o.smt2"))
            self.assertNotEqual(code, 0)
            self.assertIn("core program", err)
            self.assertFalse(os.path.exists(os.path.join(tmp, "o.smt2")))

    UNSETTLED = """
source score : I64

operation A
    uses     score
    yields   band : Text
    effect   pure
    holds    score > 0
    when     score >= 75
    computes "x"

operation B
    uses     score
    yields   band : Text
    effect   pure
    holds    score < -1000
    when     not (score >= 75)
    computes "y"

outcome band
"""

    def test_the_graph_and_the_json_agree_on_the_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "a.gg")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(core(self.UNSETTLED))
            code, plain, _ = self.run_cli("graph", source)
            self.assertEqual(code, 0)
            self.assertIn("holds `score > 0`", plain)
            code, text, _ = self.run_cli("graph", source, "--json")
            self.assertEqual(code, 0)
            import json
            payload = json.loads(text)
            proven = [c for c in payload["constraints"]
                      if c["discharge"] == "proven"]
            self.assertTrue(proven, payload["constraints"])
            self.assertTrue(all(c.get("proof") for c in proven),
                            "a proven clause with no reason beside it is a "
                            "claim, not a proof")

    def test_the_document_carries_the_reason(self):
        out = _compile(core(self.UNSETTLED))
        from gamag.toolchain import docgen
        text = docgen.render_markdown(out.compilation)
        self.assertIn("proven at compile time", text)
        self.assertIn("`when` guard", text)


class TheExamples(unittest.TestCase):
    """The corpus is the test that the prover is not a toy."""

    def test_every_core_example_keeps_a_honest_record(self):
        import glob
        root = os.path.join(S.EXAMPLES_DIR, "core")
        proved = 0
        for path in sorted(glob.glob(os.path.join(root, "*.gg"))):
            with self.subTest(example=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
                out = S.compile_only(text, path=path).assert_compiled(self)
                for clause in [c for c in
                               out.compilation.core_model.constraints.constraints
                               if c.kind == "holds"]:
                    # the culture of this compiler: a status you cannot question
                    # is not worth recording
                    self.assertTrue(
                        clause.proof,
                        f"{os.path.basename(path)}: `{clause.text}` on "
                        f"{clause.node} has a discharge status and no reason")
                    if clause.discharge == M.DISCHARGE_PROVEN:
                        proved += 1
        self.assertGreaterEqual(proved, 3,
                                "no example's promises are provable, which would "
                                "mean this prover only ever says it cannot")


if __name__ == "__main__":
    unittest.main()
