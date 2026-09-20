"""One language: both declaration families in one file, one pipeline.

Spec section 2 describes a single language, and the milestones built its two
surfaces separately -- `fn`/`let`/`var` first, then the `intent`/`operation`
core.  A reader of the specification would not guess that from the toolchain,
because a core file could not contain a function and a core expression could not
call one.

This module tests the unification: a core program may declare `fn` helpers, those
helpers are compiled by the same parser, the same checker and the same lowering
a v0.1 module uses, and the result is one GIR module.  The tests are grouped by
what they would catch -- a helper that is parsed but unreachable, one that is
reachable but untyped, one whose diagnostics point at the wrong place.
"""

from __future__ import annotations

import unittest

import support as S

HEADER = 'gama core 0.2\nintent T\n    purpose   exercise the unified surface\nsource a : I64 from 5\n'


def core(body: str) -> str:
    return HEADER + body


DOUBLER = '''
fn double(x: I64) -> I64
    pure
    return x * 2

operation Doubled
    uses     a
    yields   b : I64
    effect   pure
    computes double(a)
outcome b
'''


class HelpersInACoreFile(unittest.TestCase):
    """A helper is parsed, typed, lowered and reachable."""

    def test_a_core_program_may_declare_a_function(self):
        # Regression: this used to fail with "expected a core declaration but
        # found `fn`", which is the error that made the two surfaces look like
        # two languages.
        outcome = S.run(core(DOUBLER), path="u.gg")
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "10")

    def test_the_helper_lands_in_the_same_gir_module(self):
        outcome = S.run(core(DOUBLER), path="u.gg")
        outcome.assert_ran(self)
        functions = outcome.compilation.program.functions
        self.assertIn("double", functions,
                      "the helper should be a GIR function of this module, not "
                      "a separate compilation")
        self.assertIn("T", functions,
                      "and the core's own functions should still be there")

    def test_the_module_still_reports_the_core_dialect(self):
        outcome = S.compile_only(core(DOUBLER), path="u.gg")
        self.assertTrue(outcome.compiled)
        self.assertEqual(outcome.compilation.dialect, "core")

    def test_a_helper_can_be_called_by_several_operations(self):
        source = core('''
fn double(x: I64) -> I64
    pure
    return x * 2

operation Twice
    uses     a
    yields   b : I64
    effect   pure
    computes double(a)

operation FourTimes
    uses     b
    yields   c : I64
    effect   pure
    computes double(b)
outcome c
''')
        outcome = S.run(source, path="u.gg")
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "20")

    def test_a_helper_can_call_another_helper(self):
        source = core('''
fn double(x: I64) -> I64
    pure
    return x * 2

fn quadruple(x: I64) -> I64
    pure
    return double(double(x))

operation Quad
    uses     a
    yields   b : I64
    effect   pure
    computes quadruple(a)
outcome b
''')
        outcome = S.run(source, path="u.gg")
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "20")

    def test_a_helper_can_use_the_standard_library(self):
        source = core('''
fn rounded(x: F64) -> I64
    pure
    return math.round(x)

operation Rounded
    uses     a
    yields   b : I64
    effect   pure
    computes rounded(5.0)
outcome b
''')
        outcome = S.run(source, path="u.gg")
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "5")

    def test_helpers_are_optional(self):
        # The unification must not make a helper mandatory: every example that
        # existed before has to keep working unchanged.
        source = core('''
operation Doubled
    uses     a
    yields   b : I64
    effect   pure
    computes a * 2
outcome b
''')
        outcome = S.run(source, path="u.gg")
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.strip(), "10")


class DiagnosticsPointAtTheHelper(unittest.TestCase):
    """A diagnostic inside a helper must name the helper's line."""

    def test_a_type_error_inside_a_helper_is_reported(self):
        source = core('''
fn bad(x: I64) -> I64
    pure
    return "not an integer"

operation O
    uses     a
    yields   b : I64
    effect   pure
    computes a
outcome b
''')
        outcome = S.compile_only(source, path="u.gg")
        self.assertFalse(outcome.compiled)
        self.assertTrue(outcome.diagnostics)

    def test_the_position_is_in_the_user_file(self):
        # The capture keeps tokens rather than text precisely so this holds:
        # a helper compiled from a synthesized source would report a line in a
        # file the user has never seen.
        source = core('''
fn bad(x: I64) -> I64
    pure
    return "not an integer"

operation O
    uses     a
    yields   b : I64
    effect   pure
    computes a
outcome b
''')
        outcome = S.compile_only(source, path="u.gg")
        lines = [d.pos.line for d in outcome.diagnostics if d.pos]
        self.assertTrue(lines, "a diagnostic with no position is hard to act on")
        self.assertTrue(any(line >= 6 for line in lines),
                        f"the error is on line 6 of the file; got {lines}")

    def test_an_arity_error_names_the_helper(self):
        source = core('''
fn double(x: I64) -> I64
    pure
    return x * 2

operation O
    uses     a
    yields   b : I64
    effect   pure
    computes double(a, a)
outcome b
''')
        outcome = S.compile_only(source, path="u.gg")
        self.assertFalse(outcome.compiled)
        self.assertIn("E-arity", outcome.codes())
        self.assertIn("double", outcome.messages())

    def test_an_argument_type_error_names_the_parameter(self):
        source = core('''
fn double(x: I64) -> I64
    pure
    return x * 2

operation O
    uses     a
    yields   b : I64
    effect   pure
    computes double("text")
outcome b
''')
        outcome = S.compile_only(source, path="u.gg")
        self.assertFalse(outcome.compiled)
        self.assertIn("E-arg-type", outcome.codes())
        self.assertIn("x", outcome.messages())

    def test_an_unknown_call_still_names_what_is_missing(self):
        # The helper lookup must not swallow the unknown-call diagnostic: a
        # name that is neither a builtin nor a helper is still an error.
        source = core('''
operation O
    uses     a
    yields   b : I64
    effect   pure
    computes nosuchfunction(a)
outcome b
''')
        outcome = S.compile_only(source, path="u.gg")
        self.assertFalse(outcome.compiled)
        self.assertIn("E-unknown-call", outcome.codes())


class TheRestIsUnchanged(unittest.TestCase):
    """Unification must not have altered either surface's own semantics."""

    def test_every_core_example_still_compiles_and_runs(self):
        for name in S.core_example_names():
            with self.subTest(example=name):
                outcome = S.run_file(S.example(name))
                outcome.assert_compiled(self)

    def test_every_v01_example_still_compiles(self):
        for name in S.example_names():
            with self.subTest(example=name):
                outcome = S.compile_only(S.example_file(name), path=name)
                outcome.assert_compiled(self)

    def test_a_helper_does_not_leak_into_another_compilation(self):
        # The GIR module is per compilation; a helper compiled for one program
        # must not appear in the next.  Module-level state here would make the
        # second program in a process different from the first.
        first = S.compile_only(core(DOUBLER), path="u1.gg")
        second = S.compile_only(core('''
operation Only
    uses     a
    yields   b : I64
    effect   pure
    computes a + 1
outcome b
'''), path="u2.gg")
        self.assertTrue(first.compiled and second.compiled)
        self.assertIn("double", first.compilation.program.functions)
        self.assertNotIn("double", second.compilation.program.functions)


if __name__ == "__main__":
    unittest.main()
