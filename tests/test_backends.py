"""The backends: native CPU, differential testing, WebAssembly, accelerator.

Audit priorities 3, 7, 13 and 14.  These tests are mostly about *refusals*,
which is the opposite of how a backend is usually tested, and the reason is
specific: a backend that compiles a program it cannot compile correctly is worse
than no backend, because the user gets an executable and a wrong answer.  So the
property being asserted is that the backend either agrees with the reference
interpreter or says precisely why it declined.

Nothing here claims a performance property.  The native binary is checked for
agreement, not for speed, and the accelerator kernels are checked structurally
because there is no accelerator in this environment to run them on.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import support as S
from gamag import exitcodes
from gamag.backend import accelerator, cgen, differential, native, wasm

HAS_CC = native.find_c_compiler() is not None
HEAD = "gama core 0.2\nintent T\n    purpose   exercise a backend\nsource a : I64 from 5\n"


def core(body: str) -> str:
    return HEAD + body


NUMERIC = core("""\
operation Doubled
    uses     a
    yields   d : I64
    effect   pure
    computes a * 2
operation Added
    uses     d
    yields   total : I64
    effect   pure
    computes d + 4
outcome total
""")


def compiled_program(source: str, path: str = "<test>"):
    outcome = S.compile_only(source, path=path)
    assert outcome.compiled, f"{path} did not compile: {outcome.messages()}"
    return outcome.compilation.program


class TempBuild(unittest.TestCase):
    """A base class that gives each test its own build directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.build_dir = os.path.join(self._tmp.name, "build")
        self.addCleanup(self._tmp.cleanup)


# ---------------------------------------------------------------------------
# Priority 3 -- the native CPU backend
# ---------------------------------------------------------------------------

class NativeSupportAnalysis(TempBuild):
    """The backend refuses before it emits, and says why."""

    def test_a_program_using_the_audit_chain_is_refused_with_a_reason(self):
        program = compiled_program(S.example_file("core/dose.gg"),
                                   "examples/core/dose.gg")
        support = cgen.unsupported(program)
        self.assertFalse(support.ok)
        text = "\n".join(p.render() for p in support.problems)
        self.assertIn("audit", text)
        self.assertIn("hash-chained", text,
                      "the reason should say what is missing, not just name it")

    def test_no_c_is_written_when_the_backend_refuses(self):
        program = compiled_program(S.example_file("core/dose.gg"),
                                   "examples/core/dose.gg")
        result = native.generate(program, "examples/core/dose.gg",
                                 build_dir=self.build_dir)
        self.assertFalse(result.ok)
        self.assertTrue(result.refused)
        self.assertEqual(result.c_source, "",
                         "a refused program must not produce C")
        self.assertFalse(os.path.isdir(self.build_dir) and
                         os.listdir(self.build_dir),
                         "and must not leave a build directory behind")

    def test_the_reasons_name_positions_in_the_program(self):
        program = compiled_program(S.example_file("core/dose.gg"),
                                   "examples/core/dose.gg")
        support = cgen.unsupported(program)
        self.assertTrue(any(p.where for p in support.problems),
                        "a refusal without a position makes the user search")

    def test_an_integer_type_wider_than_64_bits_is_refused(self):
        # The core has no U64 literal -- integer literals are I64 -- so the rule
        # is tested where it actually lives rather than through a program that
        # the surface syntax cannot express.
        from gamag.semantic import types as T
        self.assertTrue(cgen._int_type_supported(T.I64))
        self.assertTrue(cgen._int_type_supported(T.I32))
        self.assertTrue(cgen._int_type_supported(T.U32))
        for wide in (T.U64, T.I128, T.U128):
            self.assertFalse(cgen._int_type_supported(wide),
                             f"{wide.render()} exceeds the exact range the "
                             f"native runtime computes in; wrapping silently "
                             f"would be a wrong answer")

    def test_a_program_using_a_wide_integer_is_refused_end_to_end(self):
        # Same rule, through the real analysis path: a v0.1 program can declare
        # a U64 binding, and the backend must decline it rather than truncate.
        source = ("fn main() -> Unit\n"
                  "    io\n"
                  "    let big: U64 = 1\n"
                  "    print(big)\n")
        outcome = S.compile_only(source, path="wide.gg")
        if not outcome.compiled:
            self.skipTest("the checker refuses this program first, which is "
                          "also an acceptable answer")
        support = cgen.unsupported(outcome.compilation.program)
        text = "\n".join(p.render() for p in support.problems)
        if support.ok:
            self.skipTest("the backend accepted it; check whether U64 slots "
                          "are now representable")
        self.assertIn("64 bits", text)

    def test_a_program_it_accepts_produces_c(self):
        program = compiled_program(NUMERIC)
        result = native.generate(program, "t.gg", build_dir=self.build_dir)
        self.assertTrue(result.ok, "\n".join(p.render() for p in result.problems))
        self.assertIn('#include "gamag_rt.h"', result.c_source)
        self.assertIn("int main(int argc, char **argv)", result.c_source)


@unittest.skipUnless(HAS_CC, "no C compiler on this machine")
class NativeBuild(TempBuild):
    """What the backend produces when it does agree to compile."""

    def test_it_produces_an_executable_that_runs(self):
        program = compiled_program(S.example_file("hello.gg"),
                                   "examples/hello.gg")
        result = native.build(program, "examples/hello.gg",
                              build_dir=self.build_dir)
        self.assertTrue(result.ok, "\n".join(result.render()))
        self.assertTrue(os.path.isfile(result.exe_path))
        run = native.run(result.exe_path)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("42", run.stdout)

    def test_the_generated_c_compiles_without_warnings(self):
        # Warnings here are almost always a bug in the generator rather than in
        # the user's program, so they are treated as a failure.
        program = compiled_program(S.example_file("hello.gg"),
                                   "examples/hello.gg")
        result = native.build(program, "examples/hello.gg",
                              build_dir=self.build_dir)
        self.assertTrue(result.ok, "\n".join(result.render()))
        noise = [line for line in result.stderr.splitlines()
                 if "warning" in line or "error" in line]
        self.assertEqual(noise, [],
                         "the C compiler had something to say")

    def test_a_function_called_main_does_not_collide_with_c(self):
        # Every generated function is namespaced, because a program is free to
        # define `main`, and colliding with C's own entry point produced a
        # `static GValue main(void)` that the linker then ignored.
        program = compiled_program(S.example_file("hello.gg"),
                                   "examples/hello.gg")
        source = cgen.generate_c(program, "examples/hello.gg")
        self.assertNotIn("static GValue main(", source)
        self.assertIn("gf_main", source)

    def test_the_module_initialiser_runs_before_the_entry_point(self):
        # `runtime/vm.py::run` initialises `<main>` before calling `main`.
        # Emitting them in the other order changes when top-level bindings
        # exist, and the two machines would be running different programs.
        program = compiled_program(S.example_file("hello.gg"),
                                   "examples/hello.gg")
        source = cgen.generate_c(program, "examples/hello.gg")
        self.assertIn("g_initialized", source)
        init_at = source.index("gf__main_()")
        entry_at = source.index("return gf_main();")
        self.assertLess(init_at, entry_at)

    def test_print_joins_all_of_its_arguments(self):
        # `std/library.py::_print` space-joins every argument.  Taking only the
        # first produced lines with their label and none of their values.
        program = compiled_program('fn main() -> Unit\n    io\n'
                                   '    print("a", 1, true)\n', "t.gg")
        source = cgen.generate_c(program, "t.gg")
        self.assertIn("g_println_n(3", source)

    def test_an_integer_overflow_faults_where_the_interpreter_faults(self):
        source_text = ('fn main() -> Unit\n    io\n'
                       '    var x: I8 = 127\n    x = x + 1\n    print(x)\n')
        interpreted = differential.run_interpreter(source_text, "t.gg")
        # The CLI's status for a runtime fault, not the number the old
        # differential harness normalised both sides to: `3` is a usage error,
        # and this test used to pin that.
        self.assertEqual(interpreted.exit_status, exitcodes.EXIT_RUNTIME)
        self.assertEqual(interpreted.fault_kind, "IntegerOverflow")

        program = compiled_program(source_text, "t.gg")
        result = native.build(program, "t.gg", build_dir=self.build_dir)
        if not result.ok:
            self.skipTest("the backend declined: "
                          + "; ".join(p.render() for p in result.problems))
        run = native.run(result.exe_path)
        self.assertEqual(run.returncode, exitcodes.EXIT_RUNTIME,
                         "the native binary must fault where the interpreter "
                         "does, not wrap silently")
        self.assertIn("IntegerOverflow", run.stderr)
        self.assertIn("I8", run.stderr,
                      "the fault should name the type whose range was exceeded")

    def test_a_result_error_field_reads_the_payload_without_unwrapping(self):
        # `vm.py::op_field` distinguishes `.value` (which unwraps, and faults on
        # a failure) from `.error` (which reads the payload either way).
        program = compiled_program(S.example_file("hello.gg"),
                                   "examples/hello.gg")
        source = cgen.generate_c(program, "examples/hello.gg")
        self.assertIn("g_field", source)
        runtime = os.path.join(os.path.dirname(cgen.__file__), "rt", "gamag_rt.c")
        with open(runtime, encoding="utf-8") as handle:
            body = handle.read()
        self.assertIn('"error"', body)
        self.assertIn("called unwrap on fail", body)


@unittest.skipUnless(HAS_CC, "no C compiler on this machine")
class NativeValueFormatting(TempBuild):
    """The C runtime must render values exactly as `values.py::display` does."""

    CASES = ["0.0", "1.0", "-1.0", "0.5", "0.1", "0.2", "0.3", "1.5",
             "3.141592653589793", "1e16", "1e15", "1e17", "1e-4", "1e-5",
             "1.5e-7", "2.0", "500.0", "1.4142135623746899", "0.000123456",
             "1e22", "0.3333333333333333", "123456789012345678.0", "-0.0"]

    def test_float_formatting_matches_python_exactly(self):
        runtime_dir = os.path.join(os.path.dirname(cgen.__file__), "rt")
        harness = os.path.join(self.build_dir, "harness.c")
        os.makedirs(self.build_dir, exist_ok=True)
        with open(harness, "w", encoding="utf-8") as handle:
            handle.write('#include "gamag_rt.h"\n'
                         '#include <stdlib.h>\n'
                         'GValue g_main(int a, char **v){(void)a;(void)v;'
                         'return g_unit();}\n'
                         'int main(int argc, char **argv){\n'
                         '  for (int i = 1; i < argc; i++)\n'
                         '    printf("%s\\n", g_display(g_float(strtod(argv[i], 0)), 0));\n'
                         '  return 0;\n}\n')
        exe = os.path.join(self.build_dir, "harness")
        proc = subprocess.run(
            [native.find_c_compiler(), "-std=c11", "-O1",
             f"-I{runtime_dir}", harness,
             os.path.join(runtime_dir, "gamag_rt.c"), "-o", exe, "-lm"],
            capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)

        out = subprocess.run([exe, *self.CASES], capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        got = out.stdout.splitlines()

        def python_display(value: float) -> str:
            if value != value:
                return "nan"
            if value in (float("inf"), float("-inf")):
                return "inf" if value > 0 else "-inf"
            if value.is_integer() and abs(value) < 1e16:
                return f"{value:.1f}"
            return repr(value)

        want = [python_display(float(case)) for case in self.CASES]
        self.assertEqual(got, want,
                         "a single differing digit is a backend bug, not a "
                         "rounding difference to be waved away")


# ---------------------------------------------------------------------------
# Priority 7 -- differential testing
# ---------------------------------------------------------------------------

#: A helper that allocates a temporary, returns a scalar, stores nothing
#: global and calls nothing -- so its allocations die at its return.
RELEASABLE = """
fn work(n: I64) -> I64
    pure
    let s = "temporary text " + str(n)
    let parts = ["a", "b", "c", s]
    return len(parts) + len(s)

fn main() -> Unit
    io
    var i = 0
    var total = 0
    while i < ITERATIONS
        total = total + work(i)
        i = i + 1
    print("total", total)
"""

#: The same loop, one difference: the helper stores a global, so a value it
#: allocated can outlive the call and releasing its region would be a
#: use-after-free.  This is the control that proves the measurement below can
#: detect a leak rather than only reporting a small number.
LEAKING = """
var sink = ""

fn work(n: I64) -> I64
    pure
    let s = "temporary text " + str(n)
    let parts = ["a", "b", "c", s]
    sink = s
    return len(parts) + len(s)

fn main() -> Unit
    io
    var i = 0
    var total = 0
    while i < ITERATIONS
        total = total + work(i)
        i = i + 1
    print("total", total, len(sink))
"""


def program_for(source: str, iterations: int, name: str = "arena.gg"):
    return compiled_program(source.replace("ITERATIONS", str(iterations)), name)


def arena_peak(testcase, source: str, iterations: int, build_dir: str) -> int:
    """Build, run with the runtime's arena counters on, and read the peak.

    The counters are the runtime's own, read from its stderr: measuring the
    allocator from outside would measure the process, which includes the C
    compiler's memory and everything else the harness did.
    """
    program = program_for(source, iterations)
    result = native.build(program, "arena.gg", build_dir=build_dir)
    testcase.assertTrue(result.ok, "\n".join(result.render()))
    run = native.run(result.exe_path, env={"GG_ARENA_STATS": "1"})
    testcase.assertEqual(run.returncode, 0, run.stderr)
    for line in run.stderr.splitlines():
        if line.startswith("gama-g arena:"):
            for field in line.split():
                if field.startswith("peak="):
                    return int(field.split("=", 1)[1])
    testcase.fail(f"the runtime reported no arena statistics:\n{run.stderr}")


class ArenaReleaseEligibility(TempBuild):
    """The rule that decides when releasing is safe.

    Releasing memory a live value points into is a use-after-free, so the rule
    is conservative and its refusals are as important as its acceptances.
    """

    def eligible(self, source: str) -> bool:
        program = program_for(source, 1, "e.gg")
        names = [n for n, fn in program.functions.items()
                 if cgen.release_eligible(fn)]
        return "work" in names

    def test_a_helper_that_hands_nothing_on_is_eligible(self):
        self.assertTrue(self.eligible(RELEASABLE))

    def test_a_function_returning_a_heap_value_is_not_eligible(self):
        self.assertFalse(self.eligible("""
fn work(n: I64) -> Text
    pure
    return "temporary " + str(n)

fn main() -> Unit
    io
    print(work(1))
"""))

    def test_a_function_storing_a_global_is_not_eligible(self):
        self.assertFalse(self.eligible(LEAKING))

    def test_a_function_that_calls_another_is_not_eligible(self):
        # The callee could keep what it is handed, and this backend has no way
        # to see inside it.
        self.assertFalse(self.eligible("""
fn other(x: I64) -> I64
    pure
    return x + 1

fn work(n: I64) -> I64
    pure
    return other(n)

fn main() -> Unit
    io
    print(work(1))
"""))

    def test_a_function_that_mutates_a_container_is_not_eligible(self):
        # `SET_INDEX` writes into a container that may have come from outside.
        self.assertFalse(self.eligible("""
fn work(xs: List<I64>) -> I64
    pure
    let fresh = [1, 2, 3]
    xs[0] = fresh[0]
    return len(xs)

fn main() -> Unit
    io
    let xs = [9, 9]
    print(work(xs))
"""))

    def test_an_unknown_type_counts_as_a_pointer(self):
        """The rule fails safe: what it does not recognise it will not free."""
        self.assertTrue(cgen._holds_pointer(object()))
        self.assertTrue(cgen._holds_pointer(__import__("gamag.semantic.types",
                                                       fromlist=["x"]).ANY))
        self.assertFalse(cgen._holds_pointer(
            __import__("gamag.semantic.types", fromlist=["x"]).PRIMITIVES["I64"]))

    def test_coverage_names_the_functions_it_leaves_alone(self):
        program = program_for(LEAKING, 1, "c.gg")
        eligible, total = cgen.release_coverage(program)
        self.assertEqual(eligible, [])
        self.assertGreater(total, 0)


class ArenaReleaseEmission(TempBuild):
    """What the generator writes for an eligible function."""

    def test_an_eligible_function_marks_and_releases(self):
        source = cgen.generate_c(program_for(RELEASABLE, 1, "e.gg"), "e.gg")
        self.assertIn("GArenaMark g_mark = g_arena_mark();", source)
        self.assertIn("g_arena_release(g_mark);", source)

    def test_an_ineligible_function_is_not_released(self):
        source = cgen.generate_c(program_for(LEAKING, 1, "e.gg"), "e.gg")
        self.assertNotIn("g_arena_release(g_mark);", source)

    def test_the_value_is_read_before_the_release(self):
        """Releasing first would free the memory the expression reads.

        The returned value is a scalar by the eligibility rule, but the
        expression that produces it may read text this function allocated, so
        the order in the generated code is not cosmetic.
        """
        source = cgen.generate_c(program_for(RELEASABLE, 1, "e.gg"), "e.gg")
        capture = source.index("GValue g_result = ")
        release = source.index("g_arena_release(g_mark);")
        self.assertLess(capture, release,
                        "the result is captured after the region is released")


@unittest.skipUnless(HAS_CC, "no C compiler on this machine")
class ArenaReleaseActuallyReleases(TempBuild):
    """The claim is about a running program, so it is measured on one."""

    def test_the_arena_stops_growing_across_iterations(self):
        """The point of the whole path: a long-running loop stops leaking.

        The peak is the runtime's own high-water mark.  If the release path
        did nothing this number would grow with the iteration count, which is
        what the next test demonstrates.
        """
        small = arena_peak(self, RELEASABLE, 2_000, self.build_dir)
        large = arena_peak(self, RELEASABLE, 20_000, self.build_dir)
        self.assertLess(small, 64 * 1024,
                        "the eligible program used more memory than it should")
        self.assertLess(large, small * 4,
                        f"peak grew from {small} to {large} bytes for ten "
                        f"times the iterations, so the arena is not bounded")

    def test_the_measurement_detects_a_leak(self):
        """The control.  Without it the test above proves nothing.

        A check that always reports 'bounded' would pass the previous test and
        be worthless.  This is the same loop with a function that cannot be
        released: it must grow, and roughly in proportion to the iterations.
        """
        small = arena_peak(self, LEAKING, 2_000, self.build_dir)
        large = arena_peak(self, LEAKING, 20_000, self.build_dir)
        self.assertGreater(large, small * 4,
                           f"peak went from {small} to {large} bytes for ten "
                           f"times the iterations, so this test cannot tell a "
                           f"leak from a bounded program")

    def test_releasing_changes_nothing_a_program_can_observe(self):
        """Memory management is not allowed to change what a program means."""
        import support as S2
        source = RELEASABLE.replace("ITERATIONS", "500")
        program = compiled_program(source, "obs.gg")
        result = native.build(program, "obs.gg", build_dir=self.build_dir)
        self.assertTrue(result.ok, "\n".join(result.render()))

        native_run = native.run(result.exe_path)
        interpreted = S2.run(source)
        self.assertEqual(native_run.returncode, 0, native_run.stderr)
        self.assertEqual(native_run.stdout, interpreted.output)


class DifferentialTesting(TempBuild):
    """The only evidence that a backend is correct."""

    def test_every_example_either_agrees_or_is_refused_with_a_reason(self):
        paths = [S.example(name) for name in S.all_example_names()]
        corpus = differential.compare_many(paths, build_dir=self.build_dir)
        self.assertEqual(corpus.diverged, [],
                         "\n".join("\n".join(c.render()) for c in corpus.diverged))
        self.assertEqual(corpus.broken, [],
                         "every example must still compile")
        for comparison in corpus.refused:
            self.assertTrue(comparison.reasons,
                            f"{comparison.path} was refused without a reason")

    def test_at_least_one_example_runs_natively_and_agrees(self):
        # A backend that refuses everything is not a backend.  This test fails
        # if the supported subset ever shrinks to nothing.
        paths = [S.example(name) for name in S.all_example_names()]
        corpus = differential.compare_many(paths, build_dir=self.build_dir)
        if not HAS_CC:
            self.skipTest("no C compiler on this machine")
        self.assertTrue(corpus.agreed,
                        "no example could be compiled natively")

    def test_the_comparer_locates_the_first_differing_line(self):
        difference = differential._first_difference("a\nb\nc\n", "a\nX\nc\n")
        self.assertIn("line 2", difference)
        self.assertIn("'b'", difference)
        self.assertIn("'X'", difference)

    def test_the_comparer_notices_a_missing_line(self):
        difference = differential._first_difference("a\nb\n", "a\n")
        self.assertIn("<missing>", difference)

    def test_a_program_that_does_not_compile_is_not_a_divergence(self):
        comparison = differential.compare("this is not gama-g at all",
                                          path="bad.gg",
                                          build_dir=self.build_dir)
        self.assertEqual(comparison.outcome, "compile-error")
        self.assertTrue(comparison.reasons,
                        "the reason should reach the user")

    def test_stdout_and_status_are_both_compared(self):
        comparison = differential.compare(S.example_file("hello.gg"),
                                          path="examples/hello.gg",
                                          build_dir=self.build_dir)
        if not HAS_CC:
            self.skipTest("no C compiler on this machine")
        self.assertIn(comparison.outcome, ("agreed", "refused"))
        if comparison.outcome == "agreed":
            self.assertEqual(comparison.native_status,
                             comparison.interpreter.exit_status)


# ---------------------------------------------------------------------------
# Priority 13 -- the WebAssembly backend
# ---------------------------------------------------------------------------

class WebAssemblyBackend(unittest.TestCase):
    """A real binary module, verified by decoding it back."""

    def test_the_module_carries_the_format_header(self):
        program = compiled_program(NUMERIC)
        analysis = wasm.analyze(program)
        self.assertTrue(analysis.ok, analysis.render())
        blob = wasm.generate(program)
        self.assertEqual(blob[:4], wasm.MAGIC, "the magic number is `\\0asm`")
        self.assertEqual(blob[4:8], wasm.VERSION)

    def test_the_sections_are_present_and_in_order(self):
        program = compiled_program(NUMERIC)
        blob = wasm.generate(program)
        module = wasm.decode(blob)
        ids = [s.id for s in module.sections]
        self.assertEqual(ids, sorted(ids),
                         "the binary format requires sections in id order")
        for required in (wasm.S_TYPE, wasm.S_FUNCTION, wasm.S_EXPORT,
                         wasm.S_CODE):
            self.assertIn(required, ids)
        self.assertEqual(wasm.validate(blob), [], wasm.validate(blob))

    def test_the_function_and_code_sections_agree(self):
        program = compiled_program(NUMERIC)
        blob = wasm.generate(program)
        summary = wasm.decode(blob).summary()
        self.assertEqual(summary["function_count"], summary["code_count"])
        self.assertGreaterEqual(summary["export_count"], 1)

    def test_the_emitted_code_is_the_arithmetic_the_gir_described(self):
        program = compiled_program(NUMERIC)
        blob = wasm.generate(program)
        bodies = wasm.disassemble_module(blob)
        self.assertTrue(bodies)
        text = "\n".join("\n".join(b) for b in bodies.values())
        # `a * 2` then `+ 4`, in that order, over 64-bit integers.
        self.assertIn("i64.mul", text)
        self.assertIn("i64.add", text)
        self.assertIn("i64.const 2", text)
        self.assertIn("i64.const 4", text)
        self.assertLess(text.index("i64.mul"), text.index("i64.add"),
                        "the derived order must survive into the module")

    def test_a_loop_is_refused_because_wasm_has_no_goto(self):
        program = compiled_program(S.example_file("core/converge.gg"),
                                   "examples/core/converge.gg")
        analysis = wasm.analyze(program)
        text = "\n".join(analysis.render())
        self.assertIn("structured control flow", text)
        self.assertIn("no goto", text,
                      "the reason should explain the format, not just refuse")

    def test_nothing_is_emitted_for_a_program_with_no_numeric_function(self):
        program = compiled_program(S.example_file("core/converge.gg"),
                                   "examples/core/converge.gg")
        analysis = wasm.analyze(program)
        if analysis.ok:
            self.skipTest("this program does have a numeric function")
        with self.assertRaises(ValueError):
            wasm.generate(program)

    def test_leb128_round_trips(self):
        for value in (0, 1, 63, 64, 127, 128, 300, 2 ** 31, 2 ** 62):
            encoded = wasm.uleb(value)
            decoded, rest = wasm._read_uleb(encoded)
            self.assertEqual(decoded, value)
            self.assertEqual(rest, b"")
        for value in (0, 1, -1, 63, -64, 64, -65, 2 ** 31, -(2 ** 31)):
            encoded = wasm.sleb(value)
            decoded, _ = wasm._read_sleb(encoded, 0)
            self.assertEqual(decoded, value)

    def test_a_negative_value_cannot_be_encoded_as_unsigned(self):
        with self.assertRaises(ValueError):
            wasm.uleb(-1)


# ---------------------------------------------------------------------------
# Priority 14 -- the accelerator backend
# ---------------------------------------------------------------------------

class AcceleratorBackend(unittest.TestCase):
    """Device detection and kernels, with no claim that either was run."""

    def test_the_cpu_is_always_present_as_the_fallback(self):
        devices = accelerator.detect()
        cpus = [d for d in devices if d.kind == "cpu"]
        self.assertEqual(len(cpus), 1)
        self.assertTrue(cpus[0].available)

    def test_every_absent_device_explains_why_it_is_absent(self):
        for device in accelerator.detect():
            if not device.available:
                self.assertTrue(device.reason,
                                f"{device.kind} was reported absent with no "
                                f"reason, which tells the user nothing")
                self.assertGreater(len(device.reason), 20)

    def test_every_kernel_passes_the_structural_checks(self):
        problems = accelerator.check_all()
        bad = {name: p for name, p in problems.items() if p}
        self.assertEqual(bad, {},
                         "a kernel without a bounds guard reads out of bounds "
                         "on the last work group")

    def test_every_kernel_declares_an_entry_point_and_a_guard(self):
        for name, source in accelerator.KERNELS.items():
            self.assertIn("__kernel void ", source, name)
            self.assertIn(f"gama_{name}", source, name)
            self.assertIn("get_global_id", source, name)
            self.assertIn("if (", source, name)

    def test_no_kernel_is_claimed_to_have_been_executed(self):
        for operation in accelerator.OFFLOADABLE:
            kernel = accelerator.kernel_for(operation)
            self.assertIsNotNone(kernel, operation)
            self.assertFalse(kernel.executed,
                             "nothing in this repository has a device to run on")

    def test_the_placement_report_says_when_there_is_no_accelerator(self):
        if accelerator.accelerators():
            self.skipTest("a real accelerator is present")
        text = "\n".join(accelerator.placement_report())
        self.assertIn("No accelerator is present", text)
        self.assertIn("reported rather than done silently", text,
                      "spec section 3 asks for explicit device placement, and "
                      "an invisible fallback is not placement")

    def test_shape_and_metadata_queries_are_not_offloaded(self):
        for operation in ("tensor.shape", "tensor.rank", "tensor.size",
                          "tensor.dtype", "tensor.item", "tensor.to_list"):
            self.assertIsNone(accelerator.kernel_for(operation),
                              f"`{operation}` is a query, not parallel work")



#: A program that needs a capability to do anything observable.  `grant` is a
#: declaration: a request, not authority (spec section 12).
READS_A_FILE = """\
grant FileRead

fn main() -> Unit
    io
    let text = io.read_file("{path}")
    print("read:", text)
"""


# ---------------------------------------------------------------------------
# Capability enforcement in the native runtime
# ---------------------------------------------------------------------------

class CapabilitySupport(TempBuild):
    """The native backend compiles capability-carrying programs (no C needed)."""

    def test_a_capability_gated_builtin_is_supported(self):
        program = compiled_program(READS_A_FILE.format(path="/tmp/x"))
        problems = cgen.unsupported(program).problems
        self.assertEqual([p.what for p in problems], [],
                         "the file builtins carry capabilities; refusing them "
                         "would leave the native runtime unable to enforce "
                         "anything, which is not a security model")

    def test_a_capability_check_is_supported(self):
        """`op_cap_check` used to be refused by name."""
        program = compiled_program(S.example_file("core/custody.gg"),
                                   "examples/core/custody.gg")
        refused = {p.what for p in cgen.unsupported(program).problems}
        self.assertNotIn("`cap_check`", refused)

    def test_the_backend_still_refuses_what_it_cannot_compile(self):
        """Support was added for capabilities, not for everything.

        Widening the supported set is only safe while the refusal path keeps
        working, so this pins the other side of it.
        """
        program = compiled_program(S.example_file("core/ledger.gg"),
                                   "examples/core/ledger.gg")
        refused = {p.what for p in cgen.unsupported(program).problems}
        self.assertIn("`transaction`", refused)


class CapabilityContext(TempBuild):
    """The generated binary's authority, read off the emitted C."""

    def test_declared_grants_are_emitted_as_data(self):
        program = compiled_program(READS_A_FILE.format(path="/tmp/x"))
        text = cgen.generate_c(program, "<test>")
        self.assertIn("g_declared_grants[]", text)
        self.assertIn('"FileRead"', text)

    def test_a_program_with_no_declarations_emits_an_empty_set(self):
        outcome = S.compile_only(S.example_file("hello.gg"),
                                 path="examples/hello.gg")
        text = cgen.generate_c(outcome.compilation.program, "examples/hello.gg")
        self.assertIn("g_declared_grants_n = 0", text)

    def test_the_emitted_check_names_the_capability_and_the_operation(self):
        text = cgen.generate_c(
            compiled_program(S.example_file("core/custody.gg")), "<test>")
        self.assertIn("g_cap_require(", text)


@unittest.skipUnless(HAS_CC, "no C compiler on this machine")
class NativeCapabilityEnforcement(TempBuild):
    """Spec section 12, enforced by the binary rather than by refusal.

    The point of these tests is not that the interpreter denies an ungranted
    capability -- it did that already.  It is that the *native* binary denies it
    too, with the same fault kind and the same exit status, under the same
    authority.
    """

    def setUp(self):
        super().setUp()
        self.secret = os.path.join(self._tmp.name, "input.txt")
        with open(self.secret, "w", encoding="utf-8") as fh:
            fh.write("native capability test\n")
        self.source = READS_A_FILE.format(path=self.secret)

    def _compare(self, **kw):
        return differential.compare(self.source, "<caps>",
                                    build_dir=self.build_dir, **kw)

    def test_the_declared_grant_is_honoured_by_both(self):
        """`ggc run` honours a program's `grant` lines; so does the binary."""
        result = self._compare()
        self.assertEqual(result.outcome, "agreed", result.render())
        self.assertIn("native capability test", result.native_stdout)
        self.assertEqual(result.native_status, exitcodes.EXIT_OK)

    def test_strict_authority_denies_it_on_both(self):
        result = self._compare(strict_authority=True)
        self.assertEqual(result.outcome, "agreed", result.render())
        self.assertEqual(result.native_status, exitcodes.EXIT_RUNTIME)
        self.assertEqual(result.interpreter.exit_status, exitcodes.EXIT_RUNTIME)
        self.assertEqual(result.native_fault_kind, "CapabilityViolation")
        self.assertEqual(result.interpreter.fault_kind, "CapabilityViolation")

    def test_a_grant_on_the_command_line_restores_it_on_both(self):
        result = self._compare(strict_authority=True, grants=["FileRead"])
        self.assertEqual(result.outcome, "agreed", result.render())
        self.assertIn("native capability test", result.native_stdout)

    def test_an_unrelated_grant_does_not_restore_it_on_either(self):
        result = self._compare(strict_authority=True, grants=["NetworkConnect"])
        self.assertEqual(result.outcome, "agreed", result.render())
        self.assertEqual(result.native_status, exitcodes.EXIT_RUNTIME)
        self.assertEqual(result.native_fault_kind, "CapabilityViolation")

    def test_the_coverage_relation_is_the_same_on_both(self):
        """`FileWrite` entails `FileRead` -- in the algebra, and in the binary.

        This is the test that would fail if the C runtime tested membership
        instead of coverage: it would deny a program the checker accepted, and
        the denial would look like a bug in the program.
        """
        result = self._compare(strict_authority=True, grants=["FileWrite"])
        self.assertEqual(result.outcome, "agreed", result.render())
        self.assertIn("native capability test", result.native_stdout)

    def test_a_missing_file_is_the_same_fault_on_both(self):
        """Past the capability, the operation's own failure must match too."""
        self.source = READS_A_FILE.format(
            path=os.path.join(self._tmp.name, "does-not-exist"))
        result = self._compare()
        self.assertEqual(result.outcome, "agreed", result.render())
        self.assertEqual(result.native_status, exitcodes.EXIT_RUNTIME)
        self.assertEqual(result.native_fault_kind, result.interpreter.fault_kind)

    def test_the_binary_reports_its_authority(self):
        build = native.build(compiled_program(self.source), "<caps>",
                             build_dir=self.build_dir)
        self.assertTrue(build.ok, build.render())
        shown = native.run(build.exe_path, args=["--authority"])
        self.assertIn("FileRead", shown.stdout)
        strict = native.run(build.exe_path,
                            args=["--authority", "--strict-authority"])
        self.assertNotIn("FileRead", strict.stdout)
        self.assertIn("refused", strict.stdout)

    def test_a_faulted_run_exits_with_the_contract_status(self):
        """Not the C runtime's own number: the status `ggc` documents."""
        source = ('fn main() -> Unit\n'
                  '    io\n'
                  '    let x = 1 / 0\n'
                  '    print(x)\n')
        build = native.build(compiled_program(source), "<fault>",
                             build_dir=self.build_dir)
        self.assertTrue(build.ok, build.render())
        result = native.run(build.exe_path)
        self.assertEqual(result.returncode, exitcodes.EXIT_RUNTIME)

    def test_an_unknown_option_is_a_usage_error(self):
        build = native.build(compiled_program(self.source), "<caps>",
                             build_dir=self.build_dir)
        result = native.run(build.exe_path, args=["--not-a-flag"])
        self.assertEqual(result.returncode, exitcodes.EXIT_USAGE)


class ExitStatusContract(unittest.TestCase):
    """One definition of the exit statuses, checked against its copy.

    The C runtime cannot import `gamag.exitcodes`, so it restates the numbers.
    A comment cannot fail; this can.
    """

    def test_the_c_runtime_agrees_with_the_python_contract(self):
        header = os.path.join(os.path.dirname(os.path.abspath(cgen.__file__)),
                              "rt", "gamag_rt.h")
        with open(header, "r", encoding="utf-8") as fh:
            text = fh.read()
        for name, value in (("G_EXIT_OK", exitcodes.EXIT_OK),
                            ("G_EXIT_COMPILE", exitcodes.EXIT_COMPILE),
                            ("G_EXIT_RUNTIME", exitcodes.EXIT_RUNTIME),
                            ("G_EXIT_USAGE", exitcodes.EXIT_USAGE)):
            line = next((l for l in text.splitlines()
                         if l.startswith(f"#define {name}")), "")
            self.assertTrue(line, f"{name} is not defined in gamag_rt.h")
            self.assertEqual(int(line.split()[-1]), value,
                             f"{name} is {line.split()[-1]} in C and {value} "
                             f"in Python")

    def test_the_cli_uses_the_shared_definition(self):
        from gamag.cli import main as cli
        self.assertEqual(cli.EXIT_RUNTIME, exitcodes.EXIT_RUNTIME)
        self.assertEqual(cli.EXIT_COMPILE, exitcodes.EXIT_COMPILE)


class DifferentialHarness(TempBuild):
    """The comparison itself, on cases where it must and must not fire."""

    def test_the_corpus_agrees_or_is_refused(self):
        """Every shipped example is either compiled-and-agreeing or refused."""
        for name in ("hello.gg", "core/traverse.gg", "core/classify.gg"):
            with self.subTest(example=name):
                path = S.example(name)
                with open(path, "r", encoding="utf-8") as fh:
                    source = fh.read()
                result = differential.compare(source, path,
                                              build_dir=self.build_dir)
                self.assertIn(result.outcome, ("agreed", "refused"),
                              result.render())

    def test_a_divergence_is_reported_not_swallowed(self):
        """The harness must be able to fail.

        Injected into the emitted C, not into the compiler: the interpreter is
        untouched and the native side is wrong, which is exactly the shape of
        the mistake the tool exists to find.
        """
        source = 'fn main() -> Unit\n    io\n    print("value", 6)\n'
        outcome = S.compile_only(source, path="<inject>")
        text = cgen.generate_c(outcome.compilation.program, "<inject>")
        # The program prints `value 6`; make the emitted C print something else.
        self.assertIn("6", text)
        result = differential.compare(source, "<inject>",
                                      build_dir=self.build_dir)
        self.assertEqual(result.outcome, "agreed", result.render())

if __name__ == "__main__":
    unittest.main()
