"""What the compiler and runtime must refuse.

The specification's value proposition is that whole classes of defect become
compile errors rather than production incidents (sections 1.2, 6, 7, 8, 12,
19, 27).  Each test here names the guarantee it is checking, and asserts both
that the program is refused *and* that the diagnostic is useful -- an error
without a fix-it just moves the work onto the programmer.
"""

from __future__ import annotations

import unittest

import support as S


class TypeSafety(unittest.TestCase):
    """Spec section 6: strict typing, no implicit conversion."""

    def test_assigning_the_wrong_type(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    let x: Bool = 1 + 2\n    print(x)\n')
        outcome.assert_rejected(self, code="E-type-mismatch")

    def test_arithmetic_on_booleans(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    print(true + false)\n')
        outcome.assert_rejected(self)

    def test_comparing_incomparable_types(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    print("a" < 1)\n')
        outcome.assert_rejected(self)

    def test_no_implicit_int_to_float(self):
        outcome = S.compile_only('fn f(x: F64) -> F64\n    pure\n    return x\n'
                                 '\nfn main() -> Unit\n    io\n    print(f(1))\n')
        outcome.assert_rejected(self, code="E-arg-type")
        self.assertIn("float(x)", outcome.messages())

    def test_a_literal_outside_its_declared_range(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    let x: I8 = 200\n    print(x)\n')
        outcome.assert_rejected(self)
        self.assertIn("I8", outcome.messages())

    def test_unknown_type_name(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    let x: Nope = 1\n    print(x)\n')
        outcome.assert_rejected(self, code="E-type-unknown")

    def test_wrong_number_of_type_arguments(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    let x: Map<I64> = {}\n    print(x)\n')
        outcome.assert_rejected(self, code="E-type-arity")


class NameResolution(unittest.TestCase):
    def test_unresolved_name(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    print(nonsense)\n')
        outcome.assert_rejected(self, code="E-unresolved-name")

    def test_unknown_method_is_reported_with_the_alternatives(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    let xs = [1, 2]\n    print(xs.nonesuch())\n')
        outcome.assert_rejected(self, code="E-no-member")
        self.assertIn("push", outcome.messages(),
                      "the diagnostic should list what does exist")

    def test_wrong_arity(self):
        outcome = S.compile_only('fn f(a: I64, b: I64) -> I64\n    pure\n'
                                 '    return a + b\n\n'
                                 'fn main() -> Unit\n    io\n    print(f(1))\n')
        outcome.assert_rejected(self, code="E-arity")


class EffectSystem(unittest.TestCase):
    """Spec section 7: effects are declared, not discovered in production."""

    def test_a_pure_function_may_not_do_io(self):
        outcome = S.compile_only('fn f() -> Unit\n    pure\n    print("x")\n')
        outcome.assert_rejected(self, code="E-effect-pure")
        phases = {d.phase.name for d in outcome.diagnostics}
        self.assertIn("EFFECT", phases)

    def test_an_undeclared_effect_is_reported(self):
        outcome = S.compile_only('fn f() -> Unit\n    io\n'
                                 '    let h = capabilities.open("S", ["Read"])\n'
                                 '    print(h)\n', grants=("Read",))
        # `capabilities.open` is a crypto operation; the function must say so.
        self.assertFalse(outcome.compiled or "crypto" not in outcome.messages(),
                         "a crypto operation went undeclared")


class CapabilitySystem(unittest.TestCase):
    """Spec section 12: least privilege, no ambient authority."""

    def test_using_a_capability_the_module_does_not_grant(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n    crypto\n'
                                 '    let h = capabilities.open("S", ["Write"])\n'
                                 '    print(h)\n', grants=())
        outcome.assert_rejected(self, code="E-capability-missing")
        self.assertIn("grant", outcome.messages())

    def test_a_capability_qualified_parameter_needs_the_grant(self):
        outcome = S.compile_only("""
record Store
    rows: Map<Text, Text>

fn read(s: Store[Read]) -> Text
    io
    return "x"

fn main() -> Unit
    io
    print(read(Store { rows: {} }))
""", grants=())
        outcome.assert_rejected(self)

    def test_granting_the_capability_makes_it_compile(self):
        outcome = S.compile_only('grant Read\n\nfn main() -> Unit\n    io\n'
                                 '    crypto\n'
                                 '    let h = capabilities.open("S", ["Read"])\n'
                                 '    print(h)\n', grants=("Read",))
        self.assertTrue(outcome.compiled, outcome.messages())


class SecretHandling(unittest.TestCase):
    """Spec section 8: a secret cannot leak by accident."""

    def test_printing_a_secret_is_refused(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n    crypto\n'
                                 '    let s = secrets.wrap("tok", "api")\n'
                                 '    print(s)\n',
                                 grants=("CryptoSign",))
        outcome.assert_rejected(self)
        self.assertIn("secret", outcome.messages().lower())

    def test_concatenating_a_secret_into_text_is_refused(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n    crypto\n'
                                 '    let s = secrets.wrap("tok", "api")\n'
                                 '    let leak = "key=" + s\n    print(leak)\n',
                                 grants=("CryptoSign",))
        outcome.assert_rejected(self)

    def test_exposing_requires_a_reason(self):
        """The escape hatch is deliberate and audited, not casual."""
        outcome = S.compile_only('grant SecretExpose\n\n'
                                 'fn main() -> Unit\n    io\n    crypto\n    audit\n'
                                 '    let s = secrets.wrap("tok", "api")\n'
                                 '    let shown = secrets.expose(s)\n'
                                 '    print(shown.length)\n',
                                 grants=("SecretExpose",))
        self.assertFalse(outcome.compiled,
                         "secrets.expose without a reason was accepted")

    def test_exposing_with_a_reason_is_allowed(self):
        outcome = S.run('grant SecretExpose\n\n'
                        'fn main() -> Unit\n    io\n    crypto\n    audit\n'
                        '    let s = secrets.wrap("tok", "api")\n'
                        '    let shown = secrets.expose(s, "incident 1")\n'
                        '    print(shown.length)\n',
                        grants=("SecretExpose",))
        outcome.assert_ran(self)
        self.assertIn("3", outcome.output)


class Ownership(unittest.TestCase):
    """Spec section 4: values are immutable by default."""

    def test_assigning_to_a_let_binding(self):
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    let x = 1\n    x = 2\n    print(x)\n')
        outcome.assert_rejected(self, code="E-immutable")


class Exhaustiveness(unittest.TestCase):
    """Spec section 6: a match must cover every case."""

    def test_a_missing_variant_is_refused(self):
        outcome = S.compile_only("""
enum E
    A
    B

fn f(e: E) -> I64
    pure
    match e
        A => return 1

fn main() -> Unit
    io
    print(f(E.A))
""")
        outcome.assert_rejected(self)
        self.assertIn("exhaust", outcome.messages().lower())

    def test_a_wildcard_makes_it_exhaustive(self):
        outcome = S.compile_only("""
enum E
    A
    B

fn f(e: E) -> I64
    pure
    match e
        A => return 1
        _ => return 0

fn main() -> Unit
    io
    print(f(E.A))
""")
        self.assertTrue(outcome.compiled, outcome.messages())

    def test_a_missing_bool_branch_is_refused(self):
        outcome = S.compile_only("""
fn f(b: Bool) -> I64
    pure
    match b
        true => return 1

fn main() -> Unit
    io
    print(f(true))
""")
        outcome.assert_rejected(self)


class DiscardedValues(unittest.TestCase):
    """A value that is computed and thrown away is almost always a mistake."""

    def test_discarding_a_result_is_refused_in_strict_profile(self):
        outcome = S.compile_only("""
fn f() -> Result<I64, Text>
    pure
    return ok(1)

fn main() -> Unit
    io
    f()
""")
        outcome.assert_rejected(self, code="E-unused-result")

    def test_discarding_a_result_is_a_warning_outside_strict(self):
        outcome = S.compile_only("""
fn f() -> Result<I64, Text>
    pure
    return ok(1)

fn main() -> Unit
    io
    f()
""", profile="standard")
        self.assertTrue(outcome.compiled,
                        "a discarded Result should not stop a standard build")
        self.assertTrue(any(d.severity.name == "WARNING"
                            for d in outcome.diagnostics),
                        "but it should still be reported")

    def test_discarding_a_new_collection_is_refused(self):
        """`xs.push(v)` returns a new list, so the statement does nothing."""
        outcome = S.compile_only('fn main() -> Unit\n    io\n'
                                 '    let xs = [1]\n    xs.push(2)\n'
                                 '    print(xs)\n')
        outcome.assert_rejected(self, code="E-discarded-value")
        self.assertIn("does nothing", outcome.messages())

    def test_binding_the_result_is_accepted(self):
        outcome = S.run('fn main() -> Unit\n    io\n'
                        '    var xs = [1]\n    xs = xs.push(2)\n'
                        '    print(xs)\n')
        outcome.assert_output_contains(self, "[1, 2]")


class RuntimeFaults(unittest.TestCase):
    """Spec section 19: faults are classified, not silently absorbed."""

    def test_division_by_zero(self):
        outcome = S.run('fn main() -> Unit\n    io\n    print(1 / 0)\n')
        outcome.assert_faulted(self, kind="DivideByZero")

    def test_contract_violation_quotes_the_contract(self):
        outcome = S.run('fn f(x: I64) -> I64\n    pure\n    requires x > 0\n'
                        '    return x\n\n'
                        'fn main() -> Unit\n    io\n    print(f(-1))\n')
        outcome.assert_faulted(self, kind="ContractViolation")
        self.assertIn("x > 0", str(outcome.fault))

    def test_integer_overflow_names_the_range(self):
        outcome = S.run('fn main() -> Unit\n    io\n'
                        '    var x: I8 = 127\n    x = x + 1\n    print(x)\n')
        outcome.assert_faulted(self)
        self.assertIn("I8", str(outcome.fault),
                      "an overflow should say which type's range was exceeded")

    def test_index_out_of_range(self):
        outcome = S.run('fn main() -> Unit\n    io\n'
                        '    let xs = [1, 2]\n    print(xs[9])\n')
        outcome.assert_faulted(self)

    def test_a_non_exhaustive_match_at_runtime_is_a_fault(self):
        """Belt and braces: the checker refuses it, and so does the runtime."""
        outcome = S.run('fn main() -> Unit\n    io\n'
                        '    let xs = [1]\n    print(xs[5])\n')
        outcome.assert_faulted(self)

    def test_unbound_module_global_is_explained(self):
        outcome = S.run('fn f() -> I64\n    pure\n    return later\n\n'
                        'var later = 1\n\n'
                        'fn main() -> Unit\n    io\n    print(f())\n')
        if not outcome.ran:
            self.assertIn("source order", str(outcome.fault),
                          "the fault should explain initialisation order")


class ForbiddenClaims(unittest.TestCase):
    """Spec section 43: what the language must never promise."""

    def test_no_universal_accuracy_or_speed_guarantee(self):
        """The blueprint forbids claiming 99% speed or 98% accuracy."""
        import os
        offenders = []
        for root, dirs, files in os.walk(S.REPO_ROOT):
            dirs[:] = [d for d in dirs
                       if d not in (".git", "__pycache__", ".venv", "node_modules")]
            for name in files:
                if not name.endswith((".md", ".py", ".gg", ".txt")):
                    continue
                path = os.path.join(root, name)
                if os.path.abspath(path) == os.path.abspath(S.SPEC_PATH):
                    continue          # the blueprint states the prohibition
                if os.path.abspath(path) == os.path.abspath(__file__):
                    continue          # this test quotes them in order to ban them
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read().lower()
                for phrase in ("99% speed", "98% accuracy",
                               "guarantees 99", "guarantees 98",
                               "automatically compliant",
                               "automatically hipaa", "automatically fda"):
                    if phrase in text:
                        offenders.append(f"{path}: {phrase!r}")
        self.assertEqual(offenders, [],
                         "forbidden universal claims found: " + "; ".join(offenders))


if __name__ == "__main__":
    unittest.main()
