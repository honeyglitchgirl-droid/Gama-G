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
        self.assertIsNotNone(out.compilation.core_model)
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

    def test_two_operations_with_one_name_are_refused(self):
        """A name identifies one node in the operation graph (spec section 9).

        Before this rule the graph -- a dict keyed by name -- silently kept
        whichever operation came last, so `ggc graph` showed one `Calc` where
        the program declared two, while `producers_of` still listed both.  The
        graph disagreed with itself and nothing said so.
        """
        out = S.run(core("""
source weight : F64 from 2.0
operation Calc
    uses     weight
    yields   first : F64
    effect   pure
    computes weight + 1.0
operation Calc
    uses     weight
    yields   second : F64
    effect   pure
    computes weight + 2.0
outcome first
"""))
        out.assert_rejected(self, "E-duplicate-operation")

    def test_a_duplicate_name_is_reported_once_and_does_not_cascade(self):
        """The refused operation keeps its binding, so the author sees one
        mistake rather than a page of "cannot find" errors about a binding
        they did write."""
        out = S.run(core("""
source weight : F64 from 2.0
operation Calc
    uses     weight
    yields   first : F64
    effect   pure
    computes weight + 1.0
operation Calc
    uses     weight
    yields   second : F64
    effect   pure
    computes weight + 2.0
outcome second
"""))
        self.assertNotIn("E-unresolved", out.messages())
        self.assertNotIn("E-ice", out.messages())

    def test_selection_needs_no_duplicate_names(self):
        """Several operations may yield one binding (that is how a choice is
        written); it is the *name* that must be unique, and the shipped
        selection example names its alternatives distinctly."""
        out = S.run(core("""
source temperature : F64 from 39.0
operation Urgent
    uses     temperature
    yields   action : Text
    effect   pure
    when     temperature > 38.0
    computes "treat"
operation Routine
    uses     temperature
    yields   action : Text
    effect   pure
    when     temperature <= 38.0
    computes "wait"
outcome action
"""))
        out.assert_compiled(self)


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
        alt = out.compilation.core_graph.selections["s"]
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
        alt = out.compilation.core_graph.selections["s"]
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
        out.assert_faulted(self, "NoActiveAlternative")
        self.assertEqual(out.fault_kind(), "NoActiveAlternative")
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
        out.assert_faulted(self, "RefinementDiverged")
        # the kind is the classification; the message names the constraint and
        # the bound, in the program's own words
        self.assertEqual(out.fault_kind(), "RefinementDiverged")
        message = str(out.fault.message)
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
        # The message names what *is* allowed, which now includes `fn`
        # helpers:
        # the two declaration families share a file (see tests/test_unified.py),
        # so a message that still said "there are no statements" and stopped
        # would be describing the language this used to be.
        self.assertIn("there are no top-level statements", out.messages())
        self.assertIn("fn", out.messages())

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
        intent = out.compilation.core_model.intent
        self.assertEqual(intent.purpose,
                         "keep   the   spacing exactly as written")
        self.assertEqual(out.compilation.core_model.operations.nodes["B"].trail,
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
                self.assertTrue(out.compilation.core_model.intent.purpose.strip(),
                                "an intent should say what it is for")


class LanguageInvariants(unittest.TestCase):
    """Properties of the language, checked mechanically.

    Section 14 of the second audit is precise about what these tests do and do
    not establish: `assert "if" not in grammar` proves that `if` is not in the
    grammar.  It does not prove the replacement has no prior art anywhere.  That
    is a different category of claim, and it lives in
    :class:`ProvenanceEvidence` below, which is deliberately not a set of
    assertions about history.
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
        text = out.compilation.core_model.render()
        self.assertIn("level 0: B", text)
        self.assertIn("derived execution order", text)
        # the edges are a fact about the model, not a formatting choice
        self.assertEqual(out.compilation.core_graph.edges(), [])


class NativeLowering(unittest.TestCase):
    """v0.3: the core compiles to GIR without passing through the older AST.

    The second audit's central structural finding was that v0.2 elaborated into
    the older language's abstract syntax, so `let`, `if`, `while` and `match`
    reappeared as an intermediate representation.  These tests check that it no
    longer does -- not by inspecting source text, but by checking what the
    compiler actually built.
    """

    def compile(self, body: str, head: str = HEAD):
        return S.compile_only(core(body, head)).assert_compiled(self)

    SIMPLE = """
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a + 1
outcome b
"""

    def test_no_older_abstract_syntax_is_built_for_a_core_program(self):
        out = self.compile(self.SIMPLE)
        self.assertIsNone(out.compilation.module,
                          "a core program must not be elaborated into the older "
                          "language's AST; it lowers to GIR from the semantic "
                          "model directly")

    def test_the_semantic_model_is_what_gets_compiled(self):
        out = self.compile(self.SIMPLE)
        self.assertIsNotNone(out.compilation.core_syntax)
        self.assertIsNotNone(out.compilation.core_model)
        self.assertIsNotNone(out.compilation.program)

    def test_the_generated_function_is_an_intent_not_a_function(self):
        out = self.compile(self.SIMPLE)
        functions = out.compilation.program.functions
        self.assertIn("T", functions)
        self.assertEqual(functions["T"].kind, "intent")

    def test_the_intent_carries_its_authority_as_capabilities(self):
        out = S.compile_only("""gama core 0.2
intent T
    authority PatientRead, AuditWrite
source a : I64 from 5
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
outcome b
""").assert_compiled(self)
        self.assertEqual(sorted(out.compilation.program.functions["T"].caps),
                         ["AuditWrite", "PatientRead"])

    def test_the_derived_graph_is_recorded_in_the_gir(self):
        """The graph survives the descent into a register machine.

        GIR has metadata for an operation graph -- reads, writes, dependencies --
        and the derived levels are exactly that.  A backend can therefore see
        the graph without re-deriving it.
        """
        out = self.compile("""
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
        tasks = out.compilation.program.functions["T"].parallel_tasks
        by_name = {t.name: t for t in tasks}
        self.assertEqual(set(by_name), {"First", "Second"})
        self.assertEqual(by_name["Second"].depends_on, ["First"])
        self.assertEqual(by_name["First"].depends_on, [])
        self.assertEqual(by_name["First"].writes, ["x"])
        self.assertIn("x", by_name["Second"].reads)

    def test_every_basic_block_ends_with_a_terminator(self):
        """A block with no terminator reads as `return ()`, so this is not style.

        The reference interpreter treats the end of a block as a return.  A
        lowering that forgot a jump would therefore silently truncate the
        program rather than fail, which makes this worth asserting on every
        example rather than trusting the six outputs to catch it.
        """
        from gamag.gir.ir import TERMINATORS
        root = os.path.join(S.EXAMPLES_DIR, "core")
        for path in sorted(glob.glob(os.path.join(root, "*.gg"))):
            with self.subTest(example=os.path.basename(path)):
                out = S.compile_only(open(path, encoding="utf-8").read(),
                                     path=path).assert_compiled(self)
                for fn in out.compilation.program.functions.values():
                    for block in fn.blocks:
                        self.assertTrue(
                            block.instrs and
                            block.instrs[-1].op in TERMINATORS,
                            f"{fn.name}/{block.id} ({block.label}) has no "
                            f"terminator")

    def test_slots_are_named_after_bindings(self):
        out = self.compile(self.SIMPLE)
        names = [s.name for s in out.compilation.program.functions["T"].slots]
        self.assertIn("b", names, "a binding should be visible in the IR by the "
                                  "name the program gave it")
        self.assertIn("a", names)

    def test_a_constraint_reaches_the_ir_with_its_own_words(self):
        out = self.compile("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b < 100
    computes a
outcome b
""")
        rendered = out.compilation.program.functions["T"].render()
        self.assertIn("holds `b < 100`", rendered)

    def test_a_bound_reaches_the_ir_as_a_number(self):
        out = self.compile("""
refine G
    uses     a
    yields   g : I64
    effect   pure
    starts   a
    repeats  g + 1
    until    g > 100
    within   7 rounds
outcome g
""")
        rendered = out.compilation.program.functions["T"].render()
        # the bound is a literal in the IR, and the diverged path is a block
        # named after the core's own concept rather than after a loop keyword
        self.assertIn("binop >= %t0, 7", rendered.replace("%t1", "%t0"))
        self.assertIn("refine.G.diverged", rendered)
        self.assertIn("within 7 rounds", rendered)

    def test_the_fault_kinds_are_the_cores_own(self):
        """Not a call to a panic function: GIR terminators with core kinds."""
        out = self.compile("""
refine G
    uses     a
    yields   g : I64
    effect   pure
    starts   a
    repeats  g + 1
    until    g > 100
    within   7 rounds
outcome g
""")
        from gamag.gir.ir import Op
        faults = [i.meta.get("kind")
                  for fn in out.compilation.program.functions.values()
                  for b in fn.blocks for i in b.instrs if i.op is Op.FAULT]
        self.assertIn("RefinementDiverged", faults)


class NativeChecks(unittest.TestCase):
    """What the native checker refuses, without help from the older one."""

    def test_a_yield_type_mismatch_is_refused(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : Text
    effect   pure
    computes a + 1
outcome b
"""))
        out.assert_rejected(self, "E-yield-type")
        self.assertIn("does not convert between types silently",
                      out.messages())

    def test_an_unknown_library_operation_is_refused(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes math.teleport(a)
outcome b
"""))
        out.assert_rejected(self, "E-unknown-call")

    def test_a_wrong_arity_is_refused(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : F64
    effect   pure
    computes math.clamp(float(a), 0.0)
outcome b
"""))
        out.assert_rejected(self, "E-arity")

    def test_an_unknown_type_is_refused(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : Money
    effect   pure
    computes a
outcome b
"""))
        out.assert_rejected(self, "E-unknown-type")

    def test_a_pure_node_may_not_call_effectful_work(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : Unit
    effect   pure
    computes print(a)
outcome b
"""))
        out.assert_rejected(self, "E-effect-undeclared")

    def test_a_constraint_must_be_a_question(self):
        out = S.run(core("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b + 1
    computes a
outcome b
"""))
        out.assert_rejected(self, "E-constraint-type")
        self.assertIn("a yes or no answer", out.messages())

    def test_a_secret_must_keep_propagating(self):
        out = S.run("""gama core 0.2
intent T
source secret pin : I64 from 1234
operation B
    uses     pin
    yields   b : I64
    effect   crypto
    computes pin
outcome b
""")
        out.assert_rejected(self, "E-secret-escape")
        self.assertIn("yields secret b", out.messages())

    def test_a_secret_binding_may_be_declared_as_such(self):
        out = S.compile_only("""gama core 0.2
intent T
source secret pin : I64 from 1234
operation B
    uses     pin
    yields   secret b : I64
    effect   crypto
    computes pin
outcome b
""")
        out.assert_compiled(self)

    def test_an_operation_may_not_exceed_the_intents_authority(self):
        out = S.run("""gama core 0.2
intent T
    authority PatientRead
source a : I64 from 5
operation B
    uses     a
    yields   b : I64
    effect   pure
    needs    CryptoSign
    computes a
outcome b
""")
        out.assert_rejected(self, "E-authority-unmet")
        self.assertIn("but the intent holds", out.messages())

    def test_a_dispatch_without_a_catch_all_is_refused(self):
        out = S.run("""gama core 0.2
intent T
source code : Text from "red"
resolve Label
    over     code
    yields   label : Text
    effect   pure
    choose
        "red" => "stop"
        "green" => "go"
outcome label
""")
        out.assert_rejected(self, "E-dispatch-not-exhaustive")
        self.assertIn("_ =>", out.messages())


class TheFiveGraphs(unittest.TestCase):
    """The model is five inspectable views, not one opaque structure."""

    def model(self, body: str, head: str = HEAD):
        return S.compile_only(core(body, head)).assert_compiled(
            self).compilation.core_model

    def test_every_graph_is_present_and_separate(self):
        model = self.model("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    holds    b > 0
    computes a
outcome b
""")
        self.assertTrue(model.intent.name)
        self.assertTrue(model.operations.nodes)
        self.assertTrue(model.constraints.constraints)
        self.assertIsNotNone(model.authority)
        self.assertIsNotNone(model.recovery)

    def test_constraints_are_grouped_by_how_they_are_discharged(self):
        model = self.model("""
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
""")
        from gamag.core import mir as MIR
        kinds = {c.kind: c.discharge for c in model.constraints.constraints}
        self.assertEqual(kinds["exclusive"], MIR.DISCHARGE_PROVEN)
        self.assertTrue(all(c.discharge == MIR.DISCHARGE_RUNTIME
                            for c in model.constraints.constraints
                            if c.kind == "when"))

    def test_an_unprovable_selection_is_recorded_as_unprovable(self):
        """Honesty about the limits of the proof is part of the model."""
        model = self.model("""
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
""")
        from gamag.core import mir as MIR
        unprovable = model.constraints.unprovable()
        self.assertEqual(len(unprovable), 1)
        self.assertEqual(unprovable[0].discharge, MIR.DISCHARGE_UNPROVABLE)

    def test_the_recovery_graph_enumerates_every_bound(self):
        model = self.model("""
refine G
    uses     a
    yields   g : I64
    effect   pure
    starts   a
    repeats  g + 1
    until    g > 100
    within   12 rounds
outcome g
""")
        refinements = [o for o in model.recovery.obligations
                       if o.kind == "refinement"]
        self.assertEqual(len(refinements), 1)
        self.assertEqual(refinements[0].bound, 12)
        self.assertEqual(refinements[0].fault, "RefinementDiverged")

    def test_the_authority_graph_records_what_is_held(self):
        model = self.model("""
operation B
    uses     a
    yields   b : I64
    effect   pure
    computes a
outcome b
""", "gama core 0.2\nintent T\n    authority PatientRead\nsource a : I64 from 5\n")
        self.assertEqual(model.authority.held, ["PatientRead"])
        self.assertEqual(model.authority.unmet(), [])


class ProvenanceEvidence(unittest.TestCase):
    """What would be needed to support an originality claim -- and what is not.

    The second audit is explicit that a test asserting `"if" not in grammar`
    proves an architectural property and *not* historical originality, and that
    the project should keep the two categories apart.  So this class contains no
    assertion that the language has no prior art, because no test can make one.

    What can be checked mechanically is the weaker, useful part: that the
    repository's own claims stay inside what the evidence supports.
    """

    def read(self, relative: str) -> str:
        with open(os.path.join(S.REPO_ROOT, relative), encoding="utf-8") as fh:
            return fh.read()

    def test_no_document_claims_the_concepts_are_unprecedented(self):
        """The defensible claim is the combination, not the invention."""
        for relative in ("README.md", "docs/IMPLEMENTATION.md",
                         "docs/DESIGN_v0_2.md"):
            text = self.read(relative).lower()
            with self.subTest(document=relative):
                for phrase in ("invented dependency graph",
                               "first language to",
                               "no prior art",
                               "never been done",
                               "completely unprecedented"):
                    self.assertNotIn(phrase, text,
                                     f"{relative} claims more than the evidence "
                                     f"supports")

    def test_the_design_document_separates_proven_from_checked(self):
        """Section 14: the three categories must not be blurred into one list."""
        text = self.read("docs/DESIGN_v0_3.md")
        self.assertIn("Proven at compile time", text)
        self.assertIn("Checked only at runtime", text)
        self.assertIn("Not checked at all", text)

    def test_the_audit_reports_are_kept_as_evidence(self):
        for relative in ("Gama-G_Detailed_Audit_and_Verification_Report.txt",
                         "Gama-G_Complete_Originality_and_Technical_Audit.txt"):
            with self.subTest(report=relative):
                self.assertTrue(os.path.exists(
                    os.path.join(S.REPO_ROOT, relative)),
                    "an audit the project responds to should stay in the "
                    "repository; the response is only checkable against it")


if __name__ == "__main__":
    unittest.main()
