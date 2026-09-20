"""Foreign functions, and who decides what a program is allowed to do.

Audit priority 12 (FFI) and spec section 29, plus the authority model of spec
section 12 that the FFI work put under enough pressure to expose a hole.

Two bugs found while building this, both with regression tests here:

* The v0.1 parser diverted an `unsafe` declaration into a flag nothing read, so
  the effect checker then reported the function as performing `unsafe` without
  declaring it.  The marker section 29 requires for foreign calls could never be
  satisfied.
* A program could confer a capability on itself with a `grant` line, which is
  the ambient authority section 12 forbids by name.
"""

from __future__ import annotations

import ctypes.util
import os
import unittest

import support as S
from gamag import driver
from gamag.driver import compile_source, declared_grants, execute, program_grants
from gamag.runtime.context import Context
from gamag.std import ffi as ffi_module
from gamag.std import library as L

#: A C library that exists on every platform this toolchain is tested on, and a
#: function in it whose answer is unambiguous.
LIBM = ctypes.util.find_library("m")


def run_with(source: str, compile_grants=("ForeignCall",), authority=None,
             entry: str = "main"):
    """Compile and run, keeping the two grants apart.

    Compiling needs the capability, because the checker refuses a program that
    uses authority it was not given.  Running needs it too, and that is the step
    a deployment controls.  Conflating them would make it impossible to ask
    "does this program run without authority?", which is the question the
    ambient-authority test asks.
    """
    compilation = compile_source(source, "ffi.gg",
                                 grants=tuple(compile_grants))
    if not compilation.ok:
        return compilation, None, ""
    import io
    if authority is None:
        authority = set(compile_grants)
    buffer = io.StringIO()
    context = Context(grants=set(authority), stdout=buffer, seed=0,
                      deterministic=True, max_steps=200_000)
    execution = execute(compilation, entry=entry, context=context, grants=())
    return compilation, execution, buffer.getvalue()


FFI_PROGRAM = '''grant ForeignCall

fn main() -> Unit
    io
    unsafe
    let lib = ffi.open("{library}")
    let power = ffi.bind(lib, "pow", "f64,f64->f64")
    let value = ffi.call(power, 2.0, 10.0)
    print("pow:", value)
'''


@unittest.skipUnless(LIBM, "no libm on this machine")
class ForeignCalls(unittest.TestCase):

    def program(self, library: str = "") -> str:
        return FFI_PROGRAM.format(library=library or LIBM)

    def test_the_module_is_visible_to_the_checker(self):
        # Regression: `ffi` was registered in the builtin table but not in the
        # module-name table, which is snapshotted at import time, so every call
        # failed with "cannot find `ffi` in this scope" while the effect checker
        # could see the effect perfectly well.
        self.assertIn("ffi", L.MODULE_TYPE_NAMES)
        self.assertIn("ffi.open", L.BUILTINS)

    def test_a_real_foreign_call_returns_the_right_answer(self):
        compilation, execution, output = run_with(self.program(),
                                                  compile_grants=("ForeignCall",))
        self.assertTrue(compilation.ok, [d.message for d in
                                         compilation.bag.diagnostics])
        self.assertIsNone(execution.fault)
        self.assertIn("pow: 1024.0", output)

    def test_the_capability_is_required_at_run_time(self):
        compilation, execution, _ = run_with(self.program(),
                                             authority=set())
        self.assertTrue(compilation.ok, [d.message for d in
                                         compilation.bag.diagnostics])
        self.assertIsNotNone(execution.fault)
        self.assertIn("ForeignCall", str(execution.fault))

    def test_every_foreign_call_is_audited(self):
        # A foreign call is where the language stops being able to make
        # promises, so it is where the record matters most.  The record is
        # written before the call, so a call that corrupts memory still leaves
        # evidence that it happened.
        compilation = compile_source(self.program(), "ffi.gg",
                                     grants=("ForeignCall",))
        self.assertTrue(compilation.ok)
        import io
        context = Context(grants={"ForeignCall"}, stdout=io.StringIO(), seed=0,
                          deterministic=True, max_steps=200_000)
        execute(compilation, entry="main", context=context, grants=())
        recorded = [record.action for record in context.audit.records]
        self.assertIn("FOREIGN_OPEN", recorded)
        self.assertIn("FOREIGN_BIND", recorded)
        self.assertIn("FOREIGN_CALL", recorded)
        self.assertTrue(context.audit.verify()[0],
                        "the chain the foreign calls were recorded in must "
                        "still verify")

    def test_a_signature_naming_a_pointer_is_refused(self):
        # Spec section 29: mark unsafe where ownership cannot be verified.
        # Allowing a pointer while claiming it is "marked unsafe" would make the
        # marking decorative.
        source = self.program().replace('"f64,f64->f64"', '"ptr,i64->i64"')
        compilation, execution, _ = run_with(source,
                                             compile_grants=("ForeignCall",))
        self.assertIsNotNone(execution.fault)
        self.assertIn("lifetime", str(execution.fault))

    def test_an_unknown_c_type_is_refused(self):
        source = self.program().replace('"f64,f64->f64"', '"double,double->double"')
        compilation, execution, _ = run_with(source,
                                             compile_grants=("ForeignCall",))
        self.assertIsNotNone(execution.fault)
        self.assertIn("not a C type", str(execution.fault))

    def test_a_signature_without_an_arrow_is_refused(self):
        source = self.program().replace('"f64,f64->f64"', '"f64"')
        compilation, execution, _ = run_with(source,
                                             compile_grants=("ForeignCall",))
        self.assertIsNotNone(execution.fault)
        self.assertIn("->", str(execution.fault))

    def test_a_missing_symbol_names_the_library(self):
        source = self.program().replace('"pow"', '"no_such_function_here"')
        compilation, execution, _ = run_with(source,
                                             compile_grants=("ForeignCall",))
        self.assertIsNotNone(execution.fault)
        self.assertIn("no_such_function_here", str(execution.fault))

    def test_a_library_that_does_not_exist_is_reported(self):
        source = self.program("/nonexistent/libnothing.so")
        compilation, execution, _ = run_with(source,
                                             compile_grants=("ForeignCall",))
        self.assertIsNotNone(execution.fault)
        self.assertIn("cannot load", str(execution.fault))

    def test_the_wrong_number_of_arguments_is_refused(self):
        source = self.program().replace("ffi.call(power, 2.0, 10.0)",
                                        "ffi.call(power, 2.0)")
        compilation, execution, _ = run_with(source,
                                             compile_grants=("ForeignCall",))
        self.assertIsNotNone(execution.fault)
        self.assertIn("argument", str(execution.fault))

    def test_an_out_of_range_integer_is_refused_rather_than_truncated(self):
        # Passing 2**40 to an i32 parameter would truncate silently in C, which
        # is exactly the kind of difference that makes a call unverifiable.
        source = self.program().replace(
            'ffi.bind(lib, "pow", "f64,f64->f64")',
            'ffi.bind(lib, "abs", "i32->i32")').replace(
            "ffi.call(power, 2.0, 10.0)", "ffi.call(power, 1099511627776)")
        compilation, execution, _ = run_with(source,
                                             compile_grants=("ForeignCall",))
        self.assertIsNotNone(execution.fault)
        self.assertIn("fit", str(execution.fault))

    def test_the_type_vocabulary_is_reported(self):
        compilation, execution, output = run_with(
            'grant ForeignCall\n\nfn main() -> Unit\n    io\n    unsafe\n'
            '    print(ffi.types())\n', compile_grants=("ForeignCall",))
        self.assertIsNone(execution.fault)
        self.assertIn("supported:", output)
        self.assertIn("refused:", output)



class UnsafeDeclaration(unittest.TestCase):
    """Regression: the parser dropped the `unsafe` declaration."""

    def test_declaring_unsafe_satisfies_the_effect_checker(self):
        # The declaration belongs on the function that performs the call.  A
        # caller inherits the effect and must declare it too, which is the point
        # of tracking effects at all.
        source = ('grant ForeignCall\n\n'
                  'fn touch() -> Unit\n'
                  '    io, unsafe\n'
                  '    let lib = ffi.open("libm.so.6")\n'
                  '    print(lib)\n')
        outcome = S.compile_only(source, grants=("ForeignCall",))
        self.assertNotIn("E-effect-undeclared", outcome.codes(),
                         "a function that reads `unsafe` must be treated as "
                         "declaring it")

    def test_a_caller_inherits_the_effect_and_must_declare_it(self):
        source = ('grant ForeignCall\n\n'
                  'fn touch() -> Unit\n'
                  '    unsafe\n'
                  '    let lib = ffi.open("libm.so.6")\n'
                  '    print(lib)\n'
                  '\nfn caller() -> Unit\n'
                  '    io\n'
                  '    touch()\n')
        outcome = S.compile_only(source, grants=("ForeignCall",))
        self.assertIn("E-effect-undeclared", outcome.codes(),
                      "effects propagate to callers; that is what makes an "
                      "effect signature a contract")

    def test_the_declaration_reaches_the_ast(self):
        from gamag.lexer import tokenize
        from gamag.parser import Parser
        source = "fn f() -> Unit\n    io\n    unsafe\n    print(1)\n"
        module = Parser(tokenize(source, "<t>"), source).parse_module()
        fn = module.decls[0]
        self.assertIn("unsafe", fn.effects)
        self.assertTrue(fn.unsafe)
        self.assertIn("io", fn.effects)

    def test_an_undeclared_unsafe_effect_is_still_reported(self):
        # The fix must not make the check toothless: performing `unsafe` without
        # declaring it is still an error in the strict profile.
        source = ('grant ForeignCall\n\n'
                  'fn touch() -> Unit\n'
                  '    io\n'
                  '    let lib = ffi.open("libm.so.6")\n'
                  '    print(lib)\n'
                  '\nfn main() -> Unit\n    io\n    touch()\n')
        outcome = S.compile_only(source, grants=("ForeignCall",))
        self.assertIn("E-effect-undeclared", outcome.codes())


class Authority(unittest.TestCase):
    """Specification section 12: no ambient filesystem access."""

    SELF_GRANTING = ('grant FileRead\n\nfn main() -> Unit\n'
                     '    io\n'
                     '    let text = io.read_file("/etc/hostname")\n'
                     '    print(len(text))\n')

    def test_a_program_cannot_grant_itself_authority(self):
        # Regression.  `program_grants` used to union the program's own `grant`
        # lines into the authority the runtime received, so a program could read
        # any file it liked by declaring that it might -- ambient authority
        # under another name, and precisely what section 12 forbids.
        compilation = compile_source(self.SELF_GRANTING, "g.gg")
        self.assertTrue(compilation.ok)
        self.assertEqual(declared_grants(compilation), {"FileRead"})
        self.assertEqual(program_grants(compilation, ()), set(),
                         "the library gives a caller only what it supplied")

    def test_the_library_refuses_the_read(self):
        compilation = compile_source(self.SELF_GRANTING, "g.gg")
        import io
        context = Context(grants=program_grants(compilation, ()),
                          stdout=io.StringIO(), seed=0, deterministic=True,
                          max_steps=200_000)
        execution = execute(compilation, entry="main", context=context,
                            grants=())
        self.assertIsNotNone(execution.fault)
        self.assertIn("FileRead", str(execution.fault))

    def test_a_caller_that_supplies_the_capability_gets_it(self):
        compilation = compile_source(self.SELF_GRANTING, "g.gg")
        import io
        buffer = io.StringIO()
        context = Context(grants=program_grants(compilation, ("FileRead",)),
                          stdout=buffer, seed=0, deterministic=True,
                          max_steps=200_000)
        execution = execute(compilation, entry="main", context=context,
                            grants=())
        self.assertIsNone(execution.fault)
        self.assertTrue(buffer.getvalue().strip())

    def test_declared_grants_are_still_visible_as_a_request(self):
        compilation = compile_source(self.SELF_GRANTING, "g.gg")
        self.assertEqual(declared_grants(compilation), {"FileRead"},
                         "the request is still recorded, so a deployment can "
                         "decide whether to honour it")

    def test_the_command_line_honours_declarations_unless_told_not_to(self):
        from gamag.cli.main import build_parser, _authority
        parser = build_parser()
        compilation = compile_source(self.SELF_GRANTING, "g.gg")
        trusted = parser.parse_args(["run", "g.gg"])
        strict = parser.parse_args(["run", "--strict-authority", "g.gg"])
        self.assertEqual(_authority(compilation, trusted), {"FileRead"})
        self.assertEqual(_authority(compilation, strict), set(),
                         "--strict-authority is for code the user has not read")


if __name__ == "__main__":
    unittest.main()
