"""The three ways input used to crash the toolchain.

A compiler is a total function: every input, however malformed, has to come
back as a diagnostic.  Three defects escaped that rule, and they are grouped
here because they are one class of bug -- input the toolchain did not bound:

* a source file that was not UTF-8 reached the user as a raw
  ``UnicodeDecodeError`` traceback;
* source nested deeply enough exhausted the host stack inside a
  recursive-descent parser, also as a traceback;
* a program that recursed too far was reported as an *internal error*, which
  tells the user the compiler broke when in fact their program asked for more
  call depth than the host has.

Each is tested through the entry point that actually failed -- ``compile_file``
for the file cases, both parsers for nesting, the interpreter for call depth --
because the reason these survived earlier rounds is that every existing test
went through a path that could not reach them.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import support as S

from gamag import driver
from gamag.diagnostics import GamaRuntimeFault
from gamag.fuzz import oracle
from gamag.nesting import (E_NESTING_CODE, MAX_AST_DEPTH, MAX_PARSE_NESTING,
                           ast_depth_limit, parse_nesting_limit,
                           python_frame_depth)
from gamag.runtime.vm import VM


def _core_program(expression: str) -> str:
    """A valid core program whose one computed value is ``expression``."""
    return ("gama core 0.2\n\n"
            "intent Probe\n"
            "    purpose    exercise the parser\n\n"
            "source weight : F64 from 2.0\n\n"
            "operation Calc\n"
            "    uses     weight\n"
            "    yields   v : F64\n"
            "    effect   pure\n"
            f"    computes {expression}\n\n"
            "outcome v\n")


def _v01_program(expression: str, ty: str = "I64") -> str:
    return (f"fn main() -> Unit\n"
            f"    let x: {ty} = {expression}\n"
            f"    print(x)\n")


class SourceFilesThatAreNotText(unittest.TestCase):
    """Defect 2: a file that does not decode is a diagnostic, not a crash."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ggtest-")
        self.addCleanup(self._tmp.cleanup)

    def path(self, name: str, payload: bytes) -> str:
        candidate = os.path.join(self._tmp.name, name)
        with open(candidate, "wb") as handle:
            handle.write(payload)
        return candidate

    def test_a_valid_file_with_one_bad_byte_is_rejected_cleanly(self):
        good = b"fn main() -> Unit\n    print(1)\n"
        path = self.path("mixed.gg", good[:12] + b"\x9e\xfe" + good[12:])
        compilation = driver.compile_file(path)
        self.assertFalse(compilation.ok)
        self.assertEqual(compilation.stopped_at, "lex")
        codes = [d.code for d in compilation.bag.diagnostics]
        self.assertIn("E-source-unreadable", codes)

    def test_the_message_says_where_the_bad_byte_is(self):
        path = self.path("bad.gg", b"\xff\xfe\x00not text")
        compilation = driver.compile_file(path)
        message = " ".join(d.message for d in compilation.bag.diagnostics)
        self.assertIn("UTF-8", message)
        self.assertIn("offset", message)

    def test_arbitrary_binary(self):
        path = self.path("binary.gg", bytes(range(256)))
        self.assertFalse(driver.compile_file(path).ok)

    def test_a_missing_file_is_reported_not_raised(self):
        missing = os.path.join(self._tmp.name, "absent.gg")
        compilation = driver.compile_file(missing)
        self.assertFalse(compilation.ok)
        self.assertEqual(compilation.stopped_at, "lex")

    def test_a_directory_is_reported_not_raised(self):
        compilation = driver.compile_file(self._tmp.name)
        self.assertFalse(compilation.ok)
        self.assertEqual(compilation.stopped_at, "lex")

    def test_a_file_that_is_not_there_still_has_a_diagnostic(self):
        """A rejection with nothing to read is not a usable error message."""
        compilation = driver.compile_file(os.path.join(self._tmp.name, "no.gg"))
        self.assertTrue(compilation.bag.diagnostics)
        self.assertTrue(compilation.bag.errors)

    def test_an_empty_file_is_not_an_error(self):
        """The bound must reject bad input without rejecting valid input."""
        path = self.path("empty.gg", b"")
        self.assertTrue(driver.compile_file(path).ok)

    def test_utf8_text_still_works(self):
        """The decoder must not have been fixed by refusing non-ASCII."""
        self.path("unicode.gg",
                  'fn main() -> Unit\n    print("caf\u00e9 \u2014 \u00fc")\n'
                  .encode("utf-8"))
        outcome = S.run_file(os.path.join(self._tmp.name, "unicode.gg"))
        outcome.assert_output_contains(self, "caf\u00e9")

    def test_byte_order_mark_is_a_diagnostic_not_a_crash(self):
        """A BOM is rejected, and rejected *cleanly* -- documented behaviour."""
        path = self.path("bom.gg", b"\xef\xbb\xbffn main() -> Unit\n    print(1)\n")
        compilation = driver.compile_file(path)
        self.assertFalse(compilation.ok)
        self.assertTrue(compilation.bag.diagnostics)

    def test_crlf_line_endings_are_accepted(self):
        path = self.path("crlf.gg", b'fn main() -> Unit\r\n    print("ok")\r\n')
        self.assertTrue(driver.compile_file(path).ok)

    def test_the_fuzzer_now_exercises_the_file_path(self):
        """The check that closes this gap must be able to fail.

        A check that cannot fail is not a check.  This reintroduces the defect
        in ``compile_file`` and asserts the new fuzz invariant reports it, then
        restores the real function.
        """
        real = oracle.compile_file

        def naive(path, **kwargs):
            with open(path, "r", encoding="utf-8") as handle:
                return driver.compile_source(handle.read(), path, **kwargs)

        oracle.compile_file = naive
        try:
            violations = oracle.check_file_path_is_total(
                "fn main() -> Unit\n    print(1)\n")
        finally:
            oracle.compile_file = real
        self.assertTrue(violations, "the file-path check did not notice the "
                                    "regression it exists to catch")
        self.assertTrue(all(v.invariant == "file-path-is-total"
                            for v in violations))

    def test_the_check_is_silent_when_the_code_is_correct(self):
        violations = oracle.check_file_path_is_total(
            "fn main() -> Unit\n    print(1)\n")
        self.assertEqual([v.detail for v in violations], [])


class DeeplyNestedSource(unittest.TestCase):
    """Defect 3: nesting is bounded in both dialects, as a diagnostic."""

    #: Comfortably past the bound, and past what an unguarded parser survived.
    DEEP = 400

    def assert_too_deep(self, source: str, what: str, dialect: str) -> None:
        outcome = S.compile_only(source)
        outcome.assert_rejected(self, code=E_NESTING_CODE, phase="parse")
        message = outcome.messages()
        self.assertIn("nest", message.lower(),
                      f"{dialect} {what} was rejected for the wrong reason:\n"
                      + message)

    # ---- the v0.1 surface --------------------------------------------
    def test_v01_parentheses(self):
        self.assert_too_deep(_v01_program("(" * self.DEEP + "1" + ")" * self.DEEP),
                             "parentheses", "v0.1")

    def test_v01_lists(self):
        self.assert_too_deep(_v01_program("[" * self.DEEP + "1" + "]" * self.DEEP),
                             "list literals", "v0.1")

    def test_v01_call_arguments(self):
        source = ("fn f(v: I64) -> I64\n    return v\n\n"
                  "fn main() -> Unit\n    print("
                  + "f(" * self.DEEP + "1" + ")" * self.DEEP + ")\n")
        self.assert_too_deep(source, "call arguments", "v0.1")

    def test_v01_unary_run(self):
        self.assert_too_deep(_v01_program("-" * self.DEEP + "1"),
                             "a unary run", "v0.1")

    def test_v01_right_associative_power(self):
        self.assert_too_deep(_v01_program(" ** ".join(["2"] * self.DEEP)),
                             "powers", "v0.1")

    # ---- the core dialect --------------------------------------------
    def test_core_parentheses(self):
        self.assert_too_deep(
            _core_program("(" * self.DEEP + "1.0" + ")" * self.DEEP),
            "parentheses", "core")

    def test_core_unary_run(self):
        self.assert_too_deep(_core_program("-" * self.DEEP + "1.0"),
                             "a unary run", "core")

    def test_core_call_arguments(self):
        self.assert_too_deep(
            _core_program("math.clamp(" * self.DEEP + "1.0"
                          + ", 0.0, 9.0)" * self.DEEP),
            "call arguments", "core")

    # ---- flat source, deep tree --------------------------------------
    def test_a_flat_chain_is_bounded_by_the_tree_it_builds(self):
        """``1+1+...`` is *flat* source and a very deep tree.

        The parser never recurses over it, so no parser guard can catch this:
        it is the recursive consumers of the AST -- checker, lowering,
        interpreter -- that run out of stack.  This is why the bound is also
        measured over the completed tree.
        """
        terms = "+".join(["1"] * 4000)
        self.assert_too_deep(_v01_program(terms), "a flat chain", "v0.1")

    # ---- the bound must not be a nuisance ----------------------------
    def test_ordinary_nesting_still_compiles(self):
        for depth in (1, 5, 20, 40):
            with self.subTest(depth=depth):
                outcome = S.compile_only(
                    _v01_program("(" * depth + "1" + ")" * depth))
                self.assertTrue(outcome.compiled,
                                f"nesting {depth} deep was rejected:\n"
                                + outcome.messages())

    def test_every_shipped_example_still_compiles(self):
        """The bound is worthless if it rejects the project's own corpus."""
        for name in S.all_example_names():
            with self.subTest(example=name):
                outcome = S.compile_only(S.example_file(name))
                self.assertTrue(
                    outcome.compiled,
                    f"{name} no longer compiles:\n" + outcome.messages())

    def test_the_limits_are_finite_and_within_their_ceilings(self):
        self.assertGreaterEqual(parse_nesting_limit(), 8)
        self.assertLessEqual(parse_nesting_limit(), MAX_PARSE_NESTING)
        self.assertGreaterEqual(ast_depth_limit(), 8)
        self.assertLessEqual(ast_depth_limit(), MAX_AST_DEPTH)

    def test_the_diagnostic_names_the_limit(self):
        """The number reported must be a real limit the user can act on.

        It is measured where the parser starts, which is a few frames deeper
        than this test's own stack, so the exact value is not asserted -- what
        matters is that a finite limit is named rather than "too deep".
        """
        outcome = S.compile_only(
            _v01_program("(" * self.DEEP + "1" + ")" * self.DEEP))
        message = outcome.messages()
        self.assertIn("levels", message)
        self.assertTrue(any(ch.isdigit() for ch in message),
                        f"the diagnostic names no limit:\n{message}")


class CallDepth(unittest.TestCase):
    """Defect 1: the depth guard has to fire before the host stack does.

    The interpreter used to be recursive -- ``execute`` called ``run_block``
    called an opcode handler called ``execute`` again, about six host frames
    per Gama-G call -- so the call-depth limit had to be derived from the
    host's configured stack.  On a default CPython that put the *effective*
    ceiling near 129 frames while ``VM.MAX_DEPTH`` said 1500, which meant the
    language's own limit was unreachable and a legitimate deep recursion was
    refused for a reason that had nothing to do with the language.
    """

    DOWN = ("fn down(n: I64) -> I64\n"
            "    if n <= 0\n"
            "        return 0\n"
            "    return down(n - 1) + 1\n\n"
            "fn main() -> Unit\n"
            "    print(down({n}))\n")

    #: Deeper than the host-derived ceiling that used to apply (~129 frames on
    #: this host), and well inside the language's policy ceiling of 1500.
    DEEP = 1200

    def test_recursion_within_the_limit_runs(self):
        outcome = S.run(self.DOWN.format(n=50))
        outcome.assert_output_contains(self, "50")

    def test_deep_recursion_runs(self):
        """The acceptance criterion for an explicit stack.

        This is the case the old interpreter could not run: 1200 frames, which
        the host stack could not carry but the language's ceiling allows.
        """
        outcome = S.run(self.DOWN.format(n=self.DEEP))
        outcome.assert_output_contains(self, str(self.DEEP))

    def test_deep_recursion_costs_no_host_frames(self):
        """The witness that the depth is the interpreter's own, not the host's.

        With the host's recursionlimit lowered to a value that a recursive
        interpreter could not have run this program under at all -- 1200
        frames at six host frames each -- the program still runs, because a
        Gama-G frame no longer occupies a host frame.
        """
        before = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(300)
            outcome = S.run(self.DOWN.format(n=self.DEEP))
        finally:
            sys.setrecursionlimit(before)
        outcome.assert_output_contains(self, str(self.DEEP))

    def test_unbounded_recursion_is_a_fault(self):
        outcome = S.run(self.DOWN.format(n=5000))
        outcome.assert_faulted(self, kind="StackOverflow")

    def test_the_fault_is_a_runtime_fault_not_a_python_error(self):
        """The failure mode this defect had: ``internal error: RecursionError``.

        An internal error means "the compiler is broken".  Running out of
        stack means "your program recursed too far".  Reporting the second as
        the first sends the user to the wrong place entirely, so the fault is
        checked by type and not only by exit status.
        """
        outcome = S.run(self.DOWN.format(n=5000))
        self.assertIsInstance(outcome.fault, GamaRuntimeFault)
        self.assertNotIsInstance(outcome.fault, RecursionError)
        self.assertEqual(outcome.fault_kind(), "StackOverflow")

    def test_the_fault_says_what_to_do_about_it(self):
        outcome = S.run(self.DOWN.format(n=5000))
        self.assertIn("call depth exceeded", str(outcome.fault))
        self.assertIsNotNone(getattr(outcome.fault, "hint", None))

    def test_the_limit_is_the_policy_ceiling(self):
        """Past the ceiling the fault must name the language's number.

        It used to name whatever the host stack happened to carry, which is
        not a number any Gama-G program can be written against.
        """
        outcome = S.run(self.DOWN.format(n=5000))
        self.assertIn(str(VM.MAX_DEPTH), str(outcome.fault))
        self.assertEqual(outcome.fault.context.get("limit"), VM.MAX_DEPTH)

    def test_the_limit_does_not_follow_the_host_stack(self):
        """Inversion of the property that used to hold.

        Raising the host's recursionlimit used to raise the effective limit;
        raising or lowering it must now change nothing, because the limit is
        the interpreter's own.
        """
        before = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(before * 20)
            outcome = S.run(self.DOWN.format(n=5000))
        finally:
            sys.setrecursionlimit(before)
        self.assertIn(str(VM.MAX_DEPTH), str(outcome.fault))

    def test_python_frame_depth_is_measurable(self):
        """The parse and AST budgets are still derived from it."""
        self.assertGreater(python_frame_depth(), 0)



class Totality(unittest.TestCase):
    """The contract all three defects broke, asserted over many inputs."""

    def test_no_input_raises_a_non_toolchain_exception(self):
        cases = {
            "empty": "",
            "whitespace": "   \n\n\t\n",
            "truncated declaration": "fn main() -> Unit\n    let x = ",
            "unbalanced brackets": "fn main() -> Unit\n    print((((((1)\n",
            "deep nesting": _v01_program("(" * 400 + "1" + ")" * 400),
            "deep chain": _v01_program("+".join(["1"] * 4000)),
            "nul bytes": "fn main() -> Unit\n\x00\x00    print(1)\n",
            "lone surrogate text": "fn main() -> Unit\n    print(1)\n\ud800",
            "core without an outcome": "gama core 0.2\n\nintent X\n",
            "very long identifier": "fn main() -> Unit\n    let "
                                    + "a" * 200_000 + " = 1\n",
        }
        for description, source in cases.items():
            with self.subTest(case=description):
                compilation = driver.compile_source(source, "totality.gg")
                # Reaching here without an exception is the assertion; the
                # second is that a rejection is explained rather than silent.
                if not compilation.ok:
                    self.assertTrue(
                        compilation.bag.diagnostics,
                        f"{description} was rejected with no diagnostic")


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
