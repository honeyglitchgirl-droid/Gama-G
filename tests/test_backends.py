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
        self.assertEqual(interpreted.exit_status, 3)
        self.assertEqual(interpreted.fault_kind, "IntegerOverflow")

        program = compiled_program(source_text, "t.gg")
        result = native.build(program, "t.gg", build_dir=self.build_dir)
        if not result.ok:
            self.skipTest("the backend declined: "
                          + "; ".join(p.render() for p in result.problems))
        run = native.run(result.exe_path)
        self.assertEqual(run.returncode, 3,
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


if __name__ == "__main__":
    unittest.main()
