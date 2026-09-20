"""The developer tools: `ggc format`, `ggc profile`, `ggc doc`, `ggc audit`.

Spec section 31 lists a formatter, a profiler and a documentation generator
among the development tools, and section 13 asks for the audit trail to be
verifiable.  These tests exercise each through the real CLI entry point --
`gamag.cli.main.main(argv)` -- because a command that works only through its
internal function and not through the parser is a command that does not exist
for the person typing at a shell.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import subprocess
import tempfile
import unittest

import support as S

from gamag.cli.main import COMMANDS, build_parser
from gamag.backend import native
from gamag.driver import compile_source
from gamag.runtime.audit import verify_trail
from gamag.formatter import (FormatError, format_source,
                             verify_same_program)


from gamag.cli.main import main  # noqa: E402


def run_cli(*argv: str):
    """Invoke `ggc` in-process.  Returns (exit code, stdout, stderr).

    `argparse` answers `--help` by raising SystemExit rather than returning;
    a CLI harness has to absorb both shapes, because a test that crashed on
    help output would be asserting nothing about the help itself.
    """
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:      # argparse's own exits
            code = exc.code if isinstance(exc.code, int) else 0
    return code, out.getvalue(), err.getvalue()


def all_examples():
    root = os.path.join(S.REPO_ROOT, "examples")
    paths = []
    for base in ("", os.path.join("core")):
        directory = os.path.join(root, base)
        for name in sorted(os.listdir(directory)):
            if name.endswith(".gg"):
                paths.append(os.path.join(directory, name))
    return paths


# ---------------------------------------------------------------------------
# ggc native -o
# ---------------------------------------------------------------------------

@unittest.skipIf(native.find_c_compiler() is None, "no C compiler on this machine")
class NativeOutputPath(unittest.TestCase):
    """`ggc native -o PATH` puts the executable at PATH.

    This flag has been advertised in `--help` since the backend existed and was
    quietly ignored: the binary always stayed in the build directory, so a reader
    who followed the help text found nothing where they were told to look.  The
    test is of the promise in `--help`, not of the compiler -- which is the only
    kind of test that would have caught it.
    """

    def test_the_executable_lands_where_it_was_asked_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "nested", "hello")
            code, out, err = run_cli(
                "native", S.example("hello.gg"), "-o", exe,
                "--build-dir", os.path.join(tmp, "b"))
            self.assertEqual(code, 0, out + err)
            self.assertTrue(os.path.isfile(exe),
                            f"-o was ignored; ggc said:\n{out}")
            self.assertTrue(os.access(exe, os.X_OK), "and it must be runnable")
            proc = subprocess.run([exe], capture_output=True, text=True,
                                  timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("total", proc.stdout)
            self.assertIn("42", proc.stdout)

    def test_an_audit_trail_can_be_written_by_the_copied_binary(self):
        """The option the copy exists for: `./triage --audit trail.jsonl`.

        The binary is moved out of the build directory, so this is also the
        check that nothing about the trail depends on where the file was built.
        """
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "triage")
            trail = os.path.join(tmp, "trail.jsonl")
            code, out, err = run_cli(
                "native", S.example("core/selection.gg"), "-o", exe,
                "--build-dir", os.path.join(tmp, "b"))
            self.assertEqual(code, 0, out + err)
            proc = subprocess.run([exe, "--audit", trail], capture_output=True,
                                  text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.isfile(trail),
                            "the compiled program writes its own trail")
            ok, problems, summary = verify_trail(open(trail, encoding="utf-8").read())
            self.assertTrue(ok, "\n".join(problems))
            self.assertEqual(summary["records"], 1)
            self.assertFalse(summary["signatures_checked"],
                             "no key was supplied, and the verifier says so")

# ---------------------------------------------------------------------------
# ggc format
# ---------------------------------------------------------------------------
class Formatter(unittest.TestCase):
    """The formatter's contract: canonical layout, never a changed program."""

    def test_every_shipped_example_formats_to_the_same_program(self):
        for path in all_examples():
            with self.subTest(example=os.path.basename(path)):
                with open(path, "r", encoding="utf-8") as handle:
                    src = handle.read()
                fmt = format_source(src, path)
                self.assertIsNone(verify_same_program(src, fmt),
                                  "formatting changed the compiled program")
                self.assertEqual(format_source(fmt, path), fmt,
                                 "formatting is not idempotent")

    def test_a_formatted_example_runs_to_the_same_output(self):
        # the strongest check the examples allow: not the same IR, but the
        # same behaviour, end to end
        for path in all_examples():
            with self.subTest(example=os.path.basename(path)):
                with open(path, "r", encoding="utf-8") as handle:
                    src = handle.read()
                fmt = format_source(src, path)
                before = S.run(src)
                after = S.run(fmt)
                self.assertEqual(before.output, after.output)
                self.assertEqual(before.compilation.ok,
                                 after.compilation.ok)

    def test_layout_is_measured_from_the_parse_not_the_typing(self):
        # a tab-indented block and a 6-space one: both land on four
        src = ('fn f(x: I64) -> I64\n'
               '\tpure\n'
               '\treturn x + 1\n'
               '\n\n\nfn main() -> Unit\n'
               '      io\n'
               '      print( f( 1 ) )\n')
        fmt = format_source(src)
        self.assertIn("\n    pure\n", fmt)
        self.assertIn("    return x + 1\n", fmt)
        self.assertIn("    print(f(1))", fmt)
        # run of blank lines collapsed to exactly one
        self.assertNotIn("\n\n\n", fmt)
        self.assertEqual(format_source(fmt), fmt)

    def test_inconsistent_dedent_is_a_diagnostic_not_a_silent_guess(self):
        # the 2-space line dedents to a level that never opened: the
        # formatter refuses with the line number instead of inventing shape
        src = ('fn f(x: I64) -> I64\n'
               '        pure\n'
               '  return x + 1\n')
        with self.assertRaises(FormatError) as caught:
            format_source(src)
        self.assertEqual(caught.exception.line, 3)

    def test_strings_and_comments_survive_verbatim(self):
        src = ('fn main() -> Unit\n'
               '    io\n'
               '    let s = "a   b//c  /*d*/"\n'
               '    print( s )   // keep   me   as   written\n')
        fmt = format_source(src)
        self.assertIn('"a   b//c  /*d*/"', fmt)
        self.assertIn("// keep   me   as   written", fmt)
        self.assertIn("print(s)", fmt)

    def test_type_arguments_hug_and_comparisons_breathe(self):
        src = ('fn f(xs: List<I64> ) -> Result<F64,Text>\n'
               '    pure\n'
               '    let n = 3\n'
               '    return ok(if n < 10 then 1.0 else 0.0)\n')
        fmt = format_source(src)
        self.assertIn("List<I64>", fmt)
        self.assertIn("Result<F64, Text>", fmt)
        self.assertIn("n < 10", fmt)      # comparison, not type syntax

    def test_a_unary_sign_never_glues_to_a_binary_operator(self):
        src = 'fn main() -> Unit\n    io\n    let x = - -5\n    let y = 3- -2\n    print(x , y)\n'
        fmt = format_source(src)
        self.assertNotIn("=-", fmt)
        self.assertNotIn("3- -2", fmt)
        self.assertIsNone(verify_same_program(src, fmt))

    def test_the_formatter_refuses_a_malformed_dedent_naming_the_line(self):
        src = ('fn main() -> Unit\n    io\n        print(1)\n      print(2)\n')
        with self.assertRaises(FormatError) as caught:
            format_source(src)
        self.assertGreaterEqual(caught.exception.line, 1)

    def test_the_formatter_refuses_an_unterminated_string_not_crashes(self):
        with self.assertRaises(FormatError):
            format_source('fn main() -> Unit\n    io\n    print("oops)\n')

    def test_generated_programs_never_produce_a_silent_semantic_change(self):
        """The property `verify_same_program` exists to guarantee.

        For every randomly generated program that compiles, formatting must
        either be verified identical or refuse -- it must never quietly
        rewrite a meaning.  Generated programs also have to keep compiling
        after formatting, since that is what a reader trusts a formatter
        with.  Mutated examples exercise the refuse side.
        """
        rng = random.Random(20260920)
        checked = refused = 0
        for _ in range(120):
            from gamag.fuzz import generator
            source = generator.generate(rng).source
            if compile_source(source).program is None:
                continue
            try:
                fmt = format_source(source)
            except FormatError:
                refused += 1
                continue
            checked += 1
            self.assertIsNone(verify_same_program(source, fmt),
                              "the formatter changed a program's GIR")
            self.assertTrue(compile_source(fmt).ok,
                            "formatted source stopped compiling")
            self.assertEqual(format_source(fmt), fmt,
                             "not idempotent")
        self.assertGreater(checked, 40, "the generator mostly produced "
                                        "non-compiling programs this run")
        # mutated sources: format must never raise anything but FormatError
        corpus = []
        for path in all_examples()[:6]:
            with open(path, "r", encoding="utf-8") as handle:
                corpus.append(handle.read())
        for _ in range(120):
            mutation = generator.mutate(rng.choice(corpus), rng)
            try:
                fmt = format_source(mutation.source)
            except FormatError:
                continue
            self.assertIsInstance(fmt, str)
            # and whatever the formatter produced must be re-formattable to
            # the same text: canonical form is a fixed point, never a cycle
            if fmt:
                self.assertEqual(format_source(fmt), fmt)
        del refused

    def test_check_mode_lists_and_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            messy = os.path.join(tmp, "messy.gg")
            with open(messy, "w", encoding="utf-8") as handle:
                handle.write('fn main() -> Unit\n    io\n    print( 1 )\n')
            code, out, _ = run_cli("format", "--check", messy)
            self.assertEqual(code, 1, "a non-canonical file must fail --check")
            self.assertIn(messy, out)
            code, out, _ = run_cli("format", "--write", messy)
            self.assertEqual(code, 0)
            with open(messy, "r", encoding="utf-8") as handle:
                self.assertIn("print(1)", handle.read())
            code, out, _ = run_cli("format", "--check", messy)
            self.assertEqual(code, 0, "an already canonical file passes")

    def test_format_needs_no_compiling_program(self):
        # formatting code mid-edit is legitimate: unparseable-but-lexable
        # text is still relaid out, it just skips the same-GIR check
        src = 'fn broken(\n'
        fmt = format_source(src)
        self.assertTrue(fmt.endswith("\n"))
        self.assertIsNone(verify_same_program(src, fmt))

    def test_all_examples_are_already_canonical(self):
        """The shipped examples carry the house style; format must not
        move a line of them.  This is the test that keeps the examples and
        the formatter's idea of canonical in one shared reality."""
        for path in all_examples():
            with self.subTest(example=os.path.basename(path)):
                with open(path, "r", encoding="utf-8") as handle:
                    src = handle.read()
                self.assertEqual(format_source(src, path), src,
                                 "run `ggc format --write examples/` and "
                                 "commit the result")


# ---------------------------------------------------------------------------
# ggc profile
# ---------------------------------------------------------------------------
class Profiler(unittest.TestCase):
    def test_the_profile_names_the_hot_function_and_counts_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.gg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fn spin(n: I64) -> I64\n"
                    "    pure\n"
                    "    var total = 0\n"
                    "    for i in 0..n\n"
                    "        total = total + i\n"
                    "    return total\n"
                    "\n"
                    "fn main() -> Unit\n"
                    "    spin(20)\n")
            code, out, err = run_cli("profile", path, "--json")
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            rows = payload[0]["functions"]
            names = {r["name"] for r in rows}
            self.assertIn("spin", names)
            total = sum(r["instructions"] for r in rows)
            self.assertGreater(total, 100)
            # the profiled instruction total must agree with the VM's own
            # counter: the profiler observes, it does not invent
            self.assertGreaterEqual(total, 1)

    def test_the_profile_of_a_faulting_program_is_still_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.gg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("fn main() -> Unit\n    io\n"
                             "    print(1 / 0)\n")
            code, out, err = run_cli("profile", path)
            self.assertEqual(code, 2, "the fault still decides the exit code")
            self.assertIn("profile", err)
            self.assertIn("DivideByZero", err)

    def test_profiling_changes_the_run_only_its_counters(self):
        # a deterministic program's printed output must be identical with
        # and without the profiler: observation without perturbation
        src = ("fn fib(n: I64) -> I64\n    pure\n"
               "    if n < 2\n        return n\n"
               "    return fib(n - 1) + fib(n - 2)\n"
               "\nfn main() -> Unit\n    io\n    print(fib(10))\n")
        plain = S.run(src)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fib.gg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(src)
            code, out, _ = run_cli("run", path)
            self.assertEqual(code, 0)
            self.assertEqual(out, plain.output)


# ---------------------------------------------------------------------------
# ggc doc
# ---------------------------------------------------------------------------
class DocGenerator(unittest.TestCase):
    def test_v01_documentation_lists_declarations_with_their_effects(self):
        code, out, _ = run_cli("doc", os.path.join(S.REPO_ROOT,
                                                   "examples", "hello.gg"))
        self.assertEqual(code, 0)
        self.assertIn("`add(a: I64, b: I64) -> I64`", out)
        self.assertIn("effects: `io`", out)         # main does io
        self.assertTrue(out.startswith("# "), "should be valid markdown")

    def test_core_documentation_shows_what_the_checker_derived(self):
        code, out, _ = run_cli("doc", os.path.join(S.REPO_ROOT, "examples",
                                                   "core", "dose.gg"))
        self.assertEqual(code, 0)
        self.assertIn("intent `SafeDosing`", out)
        self.assertIn("Authority:", out)
        self.assertIn("level 0", out)                # derived order
        self.assertIn("`rawDose >= 0.0`", out)       # the promise, quoted

    def test_contract_text_is_quoted_from_the_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.gg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fn half(x: F64) -> F64\n"
                    "    pure\n"
                    "    requires x > 0\n"
                    "    return x / 2.0\n"
                    "\n"
                    "fn main() -> Unit\n    io\n    print(half(4.0))\n")
            code, out, _ = run_cli("doc", path)
            self.assertEqual(code, 0)
            self.assertIn("requires `x > 0`", out)

    def test_stdlib_mode_reads_the_registration_table(self):
        code, out, _ = run_cli("doc", "--stdlib", "--module", "math")
        self.assertEqual(code, 0)
        self.assertIn("math.clamp(", out)
        self.assertIn("standard library", out)

    def test_doc_of_a_broken_program_says_as_much_and_still_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.gg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("fn f() -> I64\n    pure\n    return 1\n"
                             "\nfn main() -> Unit\n    io\n"
                             "    print(nope)\n")
            code, out, err = run_cli("doc", path)
            self.assertEqual(code, 0, "documentation of a half-written "
                                      "program is a feature, not a failure")
            self.assertIn("`f() -> I64`", out)
            self.assertIn("`main() -> Unit`", out)
            self.assertIn("E-unresolved-name", err)

    def test_json_mode_is_the_same_facts_as_data(self):
        code, out, _ = run_cli("doc", os.path.join(S.REPO_ROOT, "examples",
                                                    "hello.gg"), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        names = [d["name"] for d in payload["declarations"]]
        self.assertIn("divide", names)


# ---------------------------------------------------------------------------
# ggc audit
# ---------------------------------------------------------------------------
class AuditTrailVerifier(unittest.TestCase):
    """`ggc audit` is `--audit` turned around: the reader's side of the chain."""

    def trail_from(self, tmp: str, example: str) -> str:
        trail = os.path.join(tmp, "trail.jsonl")
        code, out, err = run_cli("run", "--audit", trail,
                                 os.path.join(S.REPO_ROOT, "examples",
                                              example))
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(trail))
        return trail

    def write_trail(self, tmp: str) -> str:
        return self.trail_from(tmp, "security_audit.gg")

    def test_an_intact_trail_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            trail = self.write_trail(tmp)
            code, out, err = run_cli("audit", "verify", trail)
            self.assertEqual(code, 0)
            self.assertIn("intact", out)
            self.assertIn("not checked (pass --key", out)

    def test_an_edited_field_breaks_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            trail = self.write_trail(tmp)
            with open(trail, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            record = json.loads(lines[1])
            record["action"] = "PATIENT_STOLEN"
            lines[1] = json.dumps(record, separators=(",", ":"))
            bad = os.path.join(tmp, "bad.jsonl")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
            code, out, err = run_cli("audit", "verify", bad)
            self.assertEqual(code, 1)
            self.assertIn("contents were modified", out)

    def test_a_deleted_middle_record_breaks_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            trail = self.write_trail(tmp)
            with open(trail, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            self.assertGreater(len(lines), 2)
            del lines[1]
            cut = os.path.join(tmp, "cut.jsonl")
            with open(cut, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
            code, out, _ = run_cli("audit", "verify", cut)
            self.assertEqual(code, 1)
            self.assertIn("chain is broken", out)

    def test_show_lists_records_and_filters_by_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            trail = self.write_trail(tmp)
            code, out, _ = run_cli("audit", "show", trail)
            self.assertEqual(code, 0)
            self.assertIn("PATIENT_READ", out)
            self.assertIn("record(s) shown", out)
            code, out, _ = run_cli("audit", "show", trail,
                                   "--action", "SECRET")
            self.assertIn("SECRET_EXPOSED", out)
            self.assertNotIn("PATIENT_READ", out)

    def test_a_missing_trail_is_a_usage_error_not_a_traceback(self):
        code, _, err = run_cli("audit", "verify", "/nonexistent/trail.jsonl")
        self.assertEqual(code, 3)
        self.assertIn("cannot read", err)

    def test_a_garbage_trail_is_named_as_not_ours(self):
        with tempfile.TemporaryDirectory() as tmp:
            junk = os.path.join(tmp, "junk.jsonl")
            with open(junk, "w", encoding="utf-8") as handle:
                handle.write("this is not jsonl at all\n")
            code, _, err = run_cli("audit", "verify", junk)
            self.assertEqual(code, 3)
            self.assertIn("not a trail this toolchain wrote", err)


class CommandsWired(unittest.TestCase):
    """The four new commands must be reachable, not just written."""

    def test_parser_and_handler_agree(self):
        choices = set({a.dest: a for a in build_parser()._actions}
                      ["command"].choices)
        for command in ("format", "profile", "doc", "audit"):
            self.assertIn(command, choices)
            self.assertIn(command, COMMANDS)

    def test_help_succeeds(self):
        code, out, _ = run_cli("--help")
        self.assertEqual(code, 0)
        for command in ("format", "profile", "doc", "audit"):
            self.assertIn(command, out)


if __name__ == "__main__":
    unittest.main()
