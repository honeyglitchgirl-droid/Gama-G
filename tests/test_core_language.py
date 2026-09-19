"""What the Gama-G v0.2 core language guarantees.

These tests are about the *language*, not the machine that runs it.  Each class
names a property the audit report asked the redesign to establish (sections 16
to 18), and the tests check both directions where that makes sense: that the
property holds for programs that should have it, and that a program violating it
is refused with a diagnostic that explains the rule rather than just naming a
line.

The point of the core is that whole categories of mistake are not expressible.
So several tests here assert on *rejection* -- not because rejection is
interesting in itself, but because an unbounded loop, an assignment, or a
selection with a hole in it is a production incident in most languages, and
here it is a compile error with a fix-it.
"""

from __future__ import annotations

import glob
import os
import unittest

import support as S

HEAD = "gama core 0.2\nintent T\nsource a : I64 from 5\n"


def core(body: str, head: str = HEAD) -> str:
    return head + body


class DerivedOrder(unittest.TestCase):
    """The execution graph is derived from relationships, never written."""

    def test_order_comes_from_uses_and_yields(self):
        out = S.run(core("""
operation First
    uses     a
    yields   x : I64
    effect   pure
    computes a + 1
operation Second
    uses     x
    yields   y : I64
    effect   pure
    computes x * 2
outcome y
"""))
        out.assert_ran(self)
        graph = out.compilation.core_graph
        self.assertEqual(graph.levels, [["First"], ["Second"]])
        self.assertEqual(graph.nodes["Second"].upstream, ["First"])
        self.assertEqual(graph.nodes["First"].downstream, ["Second"])

    def test_writing_the_operations_backwards_does_not_change_the_order(self):
        """Declaration order is not execution order -- that is the whole point.

        Both programs below are the same program: the second merely lists its
        operations in the opposite order, and the derived graph is identical.
        """
        forward = core("""
operation First
    uses     a
    yields   x : I64
    effect   pure
    computes a + 1
operation Second
    uses     x
    yields   y : I64
    effect   pure
    computes x * 2
outcome y
""")
        backward = core("""
operation Second
    uses     x
    yields   y : I64
    effect   pure
    computes x * 2
operation First
    uses     a
    yields   x : I64
    effect   pure
    computes a + 1
outcome y
""")
        a = S.run(forward).assert_ran(self)
        b = S.run(backward).assert_ran(self)
        self.assertEqual(a.compilation.core_graph.levels,
                         b.compilation.core_graph.levels)
        self.assertEqual(a.output.strip(), b.output.strip())

    def test_independent_operations_share_a_level(self):
        out = S.run(core("""
operation Left
    uses     a
    yields   l : I64
    effect   pure
    computes a + 1
operation Right
    uses     a
    yields   r : I64
    effect   pure
    computes a + 2
operation Both
    uses     l, r
    yields   t : I64
    effect   pure
    computes l + r
outcome t
"""))
        out.assert_ran(self)
        self.assertEqual(out.compilation.core_graph.levels,
                         [["Left", "Right"], ["Both"]])
        self.assertEqual(out.output.strip(), "13")

    def test_a_cycle_is_refused_and_the_cycle_is_named(self):
        out = S.run(core("""
operation X
    uses     a, yv
    yields   xv : I64
    effect   pure
    computes a + yv
operation Y
    uses     xv
    yields   yv : I64
    effect   pure
    computes xv
outcome xv
"""))
        out.assert_rejected(self, "E-graph-cycle")
        self.assertIn("X -> Y -> X", out.messages())

    def test_the_dialect_is_detected_from_the_pragma(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
outcome b
"""))
        out.assert_ran(self)
        self.assertEqual(out.compilation.dialect, "core")
        self.assertIsNotNone(out.compilation.core)
        self.assertIsNotNone(out.compilation.core_graph)

    def test_a_v01_program_is_still_the_v01_dialect(self):
        """The redesign adds a language; it does not confiscate the old one."""
        out = S.run("fn main() -> Unit\n    print(1 + 1)\n")
        out.assert_ran(self)
        self.assertEqual(out.compilation.dialect, "v0.1")
        self.assertIsNone(out.compilation.core_graph)


class Relationships(unittest.TestCase):
    """Declared relationships must match the real ones."""

    def test_a_real_dependency_must_be_declared(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
operation C
    yields   c : I64
    effect   pure
    computes b + 1
outcome c
"""))
        out.assert_rejected(self, "E-undeclared-relationship")
        self.assertIn("does not declare it in `uses`", out.messages())
        self.assertIn("add it to `uses`", out.messages())

    def test_a_relationship_to_nothing_is_refused(self):
        out = S.run(core("""
operation B
    uses     ghost
    yields   b : I64
    effect   pure
    computes 1
outcome b
"""))
        out.assert_rejected(self, "E-unresolved-relationship")
        self.assertIn("which nothing produces", out.messages())

    def test_a_declared_but_unused_relationship_warns(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes 9
outcome b
"""))
        out.assert_ran(self)
        self.assertIn("W-unused-relationship",
                      [d.code for d in out.diagnostics])
        self.assertIn("never refers to it", out.messages())

    def test_an_intent_with_no_outcome_produces_nothing(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
"""))
        out.assert_rejected(self, "E-no-outcome")

    def test_an_outcome_nothing_produces_is_refused(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
outcome missing
"""))
        out.assert_rejected(self, "E-unresolved-relationship")

    def test_two_unconditional_producers_of_one_binding_are_refused(self):
        """Without guards the intent does not say which producer applies."""
        out = S.run(core("""
operation P
    uses     a
    yields   s : I64
    effect   pure
    computes 1
operation Q
    uses     a
    yields   s : I64
    effect   pure
    computes 2
outcome s
"""))
        out.assert_rejected(self, "E-ambiguous-selection")
        self.assertIn("no `when` guard", out.messages())

    def test_a_binding_redeclared_by_a_source_is_refused(self):
        out = S.run("""gama core 0.2
intent T
source a : I64 from 5
source a : I64 from 6
outcome a
""")
        out.assert_rejected(self, "E-duplicate-binding")


class Selection(unittest.TestCase):
    """Guards replace `if`: several operations, one binding."""

    def test_complementary_guards_are_proven_exclusive_and_exhaustive(self):
        out = S.run(core("""
operation P
    uses     a
    yields   s : I64
    effect   pure
    when     a >= 3
    computes 1
operation Q
    uses     a
    yields   s : I64
    effect   pure
    when     not (a >= 3)
    computes 2
outcome s
"""))
        out.assert_ran(self)
        alt = out.compilation.core_graph.alternatives["s"]
        self.assertTrue(alt.proven_exhaustive)
        self.assertTrue(alt.proven_exclusive)
        self.assertEqual(out.output.strip(), "1")

    def test_the_active_alternative_is_the_one_that_runs(self):
        body = """
operation P
    uses     a
    yields   s : Text
    effect   pure
    when     a >= 3
    computes "high"
operation Q
    uses     a
    yields   s : Text
    effect   pure
    when     not (a >= 3)
    computes "low"
outcome s
"""
        self.assertEqual(
            S.run(core(body, "gama core 0.2\nintent T\nsource a : I64 from 9\n"))
            .assert_ran(self).output.strip(), "high")
        self.assertEqual(
            S.run(core(body, "gama core 0.2\nintent T\nsource a : I64 from 1\n"))
            .assert_ran(self).output.strip(), "low")

    def test_guards_that_are_not_complementary_are_not_claimed_to_be(self):
        # Compiled, not run: with `a` = 5 neither guard holds, so the honest
        # runtime behaviour is the NoActiveAlternative fault checked below.  The
        # claim being tested here is only about what the compiler *proves*.
        out = S.compile_only(core("""
operation P
    uses     a
    yields   s : I64
    effect   pure
    when     a >= 75
    computes 1
operation Q
    uses     a
    yields   s : I64
    effect   pure
    when     a >= 50
    computes 2
outcome s
"""))
        out.assert_compiled(self)
        alt = out.compilation.core_graph.alternatives["s"]
        self.assertFalse(alt.proven_exhaustive)
        self.assertFalse(alt.proven_exclusive)

    def test_no_active_alternative_is_a_classified_fault(self):
        """The proof being absent is why the runtime check stays."""
        out = S.run(core("""
operation P
    uses     a
    yields   s : I64
    effect   pure
    when     a >= 75
    computes 1
operation Q
    uses     a
    yields   s : I64
    effect   pure
    when     a >= 50
    computes 2
outcome s
"""))
        out.assert_faulted(self, "Panic")
        self.assertIn("NoActiveAlternative", str(out.fault.message))
        self.assertIn("`s`", str(out.fault.message))

    def test_one_unguarded_alternative_among_guarded_ones_is_refused(self):
        out = S.run(core("""
operation P
    uses     a
    yields   s : I64
    effect   pure
    when     a >= 3
    computes 1
operation Q
    uses     a
    yields   s : I64
    effect   pure
    computes 2
outcome s
"""))
        out.assert_rejected(self, "E-ambiguous-selection")
        self.assertIn("does not say which one applies", out.messages())


class BoundedRepetition(unittest.TestCase):
    """There is no unbounded loop, because there is no way to write one."""

    def test_refinement_without_a_bound_cannot_be_written(self):
        out = S.run(core("""
refine G
    uses     a
    yields   g : I64
    effect   pure
    starts   a
    repeats  g + 1
    until    g > 100
outcome g
"""))
        out.assert_rejected(self, "E-unbounded-refinement")
        self.assertIn("within 50 rounds", out.messages())

    def test_a_bound_of_zero_rounds_is_refused(self):
        out = S.run(core("""
refine G
    uses     a
    yields   g : I64
    effect   pure
    starts   a
    repeats  g + 1
    until    g > 100
    within   0 rounds
outcome g
"""))
        out.assert_rejected(self, "E-unbounded-refinement")

    def test_refinement_converges(self):
        out = S.run("""gama core 0.2
intent SquareRoot
source n : F64 from 2.0
refine Guess
    uses     n
    yields   g : F64
    effect   pure
    starts   n / 2.0
    repeats  (g + n / g) / 2.0
    until    abs(g * g - n) < 0.000001
    within   60 rounds
outcome g
""")
        out.assert_ran(self)
        self.assertAlmostEqual(float(out.output.strip()), 2 ** 0.5, places=6)

    def test_a_diverging_refinement_stops_at_its_bound_and_says_so(self):
        out = S.run(core("""
refine G
    uses     a
    yields   g : I64
    effect   pure
    starts   a
    repeats  g + 1
    until    g > 1000000
    within   4 rounds
outcome g
"""))
        out.assert_faulted(self, "Panic")
        message = str(out.fault.message)
        self.assertIn("RefinementDiverged", message)
        self.assertIn("within 4 rounds", message)
        self.assertIn("g > 1000000", message)

    def test_refinement_needs_every_one_of_its_parts(self):
        out = S.run(core("""
refine G
    uses     a
    yields   g : I64
    effect   pure
    starts   a
    until    g > 100
    within   10 rounds
outcome g
"""))
        out.assert_rejected(self, "E-refine-incomplete")
        self.assertIn("repeats", out.messages())


class Constraints(unittest.TestCase):
    """`holds` is a promise the program makes and the runtime keeps."""

    def test_a_violated_constraint_quotes_the_constraint(self):
        out = S.run(core("""
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

    def test_a_held_constraint_does_not_fire(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b > 3
    computes a
outcome b
"""))
        out.assert_ran(self)
        self.assertEqual(out.output.strip(), "5")

    def test_a_constraint_violation_is_recorded_in_the_audit_trail(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b < 3
    computes a
outcome b
"""))
        out.assert_faulted(self, "ContractViolation")
        # the action is the record's own field, not one of its extra fields
        actions = [record.action for record in out.context.audit.records]
        self.assertIn("REQUIRE_FAILED", actions)

    def test_an_unknown_effect_name_is_refused(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   telepathy
    computes a
outcome b
"""))
        out.assert_rejected(self, "E-unknown-effect")
        self.assertIn("medical", out.messages())


class StateAndAuthority(unittest.TestCase):
    """Only `state` changes, and only through a `transition`."""

    def test_a_transition_commits_after_the_compute_phase(self):
        out = S.run("""gama core 0.2
intent Ledger
    authority  AuditWrite
source amount : I64 from 100
state balance : I64
    starts     0
    authority  AuditWrite
operation Seen
    uses     amount
    yields   seen : I64
    effect   pure
    computes amount
transition Deposit
    alters   balance
    uses     seen
    effect   storage
    holds    balance >= 0
    computes balance + seen
    trail    deposit
outcome balance
""")
        out.assert_ran(self)
        self.assertEqual(out.output.strip(), "100")

    def test_a_state_without_an_initial_value_is_refused(self):
        out = S.run("""gama core 0.2
intent L
source amount : I64 from 1
state balance : I64
    authority AuditWrite
operation Seen
    uses     amount
    yields   seen : I64
    effect   pure
    computes amount
outcome seen
""")
        out.assert_rejected(self, "E-state-uninitialised")

    def test_a_transition_may_only_alter_a_state(self):
        """A binding produced by an operation is single-assignment."""
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
transition T1
    alters   b
    computes 1
outcome b
"""))
        out.assert_rejected(self, "E-transition-target")
        self.assertIn("single-assignment", out.messages())

    def test_a_transition_may_not_retype_the_state_it_alters(self):
        out = S.run("""gama core 0.2
intent L
state balance : I64
    starts 0
transition T1
    alters   balance : I64
    computes 1
outcome balance
""")
        out.assert_rejected(self)
        self.assertIn("may not retype it", out.messages())

    def test_compute_may_not_depend_on_a_committed_change(self):
        out = S.run("""gama core 0.2
intent L
state st : I64
    starts 0
operation B
    uses     st
    yields   b : I64
    effect   pure
    computes st
transition T1
    alters   st
    computes st + 1
outcome b
""")
        out.assert_rejected(self, "E-compute-after-commit")
        self.assertIn("never on a committed change", out.messages())

    def test_a_transition_with_no_target_is_refused(self):
        out = S.run("""gama core 0.2
intent L
source a : I64 from 1
transition T1
    computes 1
outcome a
""")
        out.assert_rejected(self, "E-transition-target")


class FanOutAndDispatch(unittest.TestCase):
    """`each` and `resolve`: traversal and exhaustive dispatch."""

    def test_each_produces_a_collection(self):
        out = S.run("""gama core 0.2
intent D
source readings : List<I64> from [3, 5, 8]
each Doubled
    over     readings as reading
    yields   doubled : I64
    effect   pure
    computes reading * 2
outcome doubled
""")
        out.assert_ran(self)
        self.assertEqual(out.output.strip(), "[6, 10, 16]")

    def test_the_over_clause_is_itself_the_relationship(self):
        """`each` must not also be told to `use` what it traverses."""
        out = S.run("""gama core 0.2
intent D
source readings : List<I64> from [1, 2]
each Doubled
    over     readings as reading
    yields   doubled : I64
    effect   pure
    computes reading * 2
outcome doubled
""")
        out.assert_ran(self)
        self.assertEqual(
            [d.code for d in out.diagnostics if d.code == "E-undeclared-relationship"],
            [])

    def test_each_needs_an_item_name(self):
        out = S.run("""gama core 0.2
intent D
source readings : List<I64> from [1, 2]
each Doubled
    over     readings
    yields   doubled : I64
    effect   pure
    computes 1
outcome doubled
""")
        out.assert_rejected(self, "E-each-incomplete")
        self.assertIn("over readings as reading", out.messages())

    def test_resolve_dispatches_exhaustively(self):
        out = S.run("""gama core 0.2
intent C
source code : Text from "green"
resolve Label
    over     code
    yields   label : Text
    effect   pure
    choose
        "red" => "stop"
        "green" => "go"
        _ => "wait"
outcome label
""")
        out.assert_ran(self)
        self.assertEqual(out.output.strip(), "go")

    def test_resolve_with_no_alternatives_is_refused(self):
        out = S.run("""gama core 0.2
intent C
source code : Text from "green"
resolve Label
    over     code
    yields   label : Text
    effect   pure
outcome label
""")
        out.assert_rejected(self, "E-resolve-incomplete")


class SurfaceRules(unittest.TestCase):
    """The declaration grammar teaches its own rules."""

    def test_an_unknown_clause_names_the_ones_that_belong(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    magically  computes a
outcome b
"""))
        out.assert_rejected(self)
        self.assertIn("is not a clause of an operation declaration",
                      out.messages())
        self.assertIn("`computes`", out.messages())

    def test_a_statement_is_not_a_declaration(self):
        """The core has no statements, so the grammar refuses to pretend."""
        out = S.run("""gama core 0.2
intent T
source a : I64 from 5
let x = 1
outcome a
""")
        out.assert_rejected(self)
        self.assertIn("there are no statements", out.messages())

    def test_two_intents_are_refused(self):
        out = S.run("""gama core 0.2
intent One
intent Two
source a : I64 from 1
outcome a
""")
        out.assert_rejected(self)
        self.assertIn("one intent", out.messages())

    def test_purpose_and_trail_are_kept_verbatim(self):
        out = S.run("""gama core 0.2
intent T
    purpose    keep   the   spacing exactly as written
source a : I64 from 5
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
    trail    dose decision
outcome b
""")
        out.assert_ran(self)
        intent = out.compilation.core.intent
        self.assertEqual(intent.purpose,
                         "keep   the   spacing exactly as written")
        self.assertEqual(out.compilation.core.operations[0].trail,
                         "dose decision")


class CoreExamples(unittest.TestCase):
    """Every shipped core example compiles, runs, and is really core."""

    def examples(self):
        root = os.path.join(S.EXAMPLES_DIR, "core")
        return sorted(glob.glob(os.path.join(root, "*.gg")))

    def test_there_are_core_examples(self):
        self.assertTrue(self.examples(),
                        "examples/core should ship working core programs")

    def test_every_core_example_compiles_and_runs(self):
        for path in self.examples():
            with self.subTest(example=os.path.basename(path)):
                out = S.run_file(path)
                out.assert_compiled(self)
                out.assert_ran(self)
                self.assertEqual(out.compilation.dialect, "core")

    def test_every_core_example_declares_a_purpose(self):
        for path in self.examples():
            with self.subTest(example=os.path.basename(path)):
                out = S.run_file(path)
                out.assert_compiled(self)
                self.assertTrue(out.compilation.core.intent.purpose.strip(),
                                "an intent should say what it is for")


class OriginalityGuarantees(unittest.TestCase):
    """The properties that make this a different language, checked as such.

    Section 18 of the audit report asks whether a construct is merely a renamed
    convention.  These tests pin down the ones that are not: in each case the
    conventional spelling is absent from the grammar and the property that
    replaces it is enforced.
    """

    def test_the_core_grammar_has_no_assignment_keyword(self):
        from gamag.core.parser import CLAUSES, DECL_WORDS
        forbidden = {"if", "else", "while", "for", "return", "match", "fn",
                     "let", "var", "assign", "break", "continue", "loop"}
        self.assertEqual(forbidden & DECL_WORDS, set(),
                         "a core declaration word must not be a conventional "
                         "statement keyword")
        for kind, clauses in CLAUSES.items():
            self.assertEqual(forbidden & set(clauses), set(),
                             f"`{kind}` accepts a conventional keyword")

    def test_every_repetition_construct_carries_a_bound(self):
        from gamag.core.parser import CLAUSES
        self.assertIn("within", CLAUSES["refine"],
                      "bounded repetition is the only repetition the core has")

    def test_no_clause_produces_a_binding_without_a_producer(self):
        """A binding comes from a source, a state, or exactly one operation."""
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
operation C
    uses     b
    yields   c : I64
    effect   pure
    computes b
outcome c
"""))
        out.assert_ran(self)
        graph = out.compilation.core_graph
        self.assertEqual(graph.producers_of["b"], ["B"])
        self.assertEqual(graph.producers_of["c"], ["C"])

    def test_the_derived_graph_is_explainable(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
outcome b
"""))
        out.assert_ran(self)
        text = out.compilation.core_graph.describe()
        self.assertIn("level 0: B", text)
        self.assertIn("B yields b", text)


if __name__ == "__main__":
    unittest.main()
