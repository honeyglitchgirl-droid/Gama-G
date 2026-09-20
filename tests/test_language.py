"""The language's positive semantics: what a correct program does.

Each test names the specification section it exercises.  The negative
counterparts -- what the compiler must refuse -- are in test_enforcement.py.
"""

from __future__ import annotations

import unittest

import support as S


class Lexing(unittest.TestCase):
    """Spec section 2: lexical structure."""

    def tokenize(self, source):
        from gamag.lexer import Lexer
        return Lexer(source, "<test>").tokenize()

    def kinds(self, source):
        from gamag.tokens import TokenKind
        return [t.kind for t in self.tokenize(source)
                if t.kind not in (TokenKind.NEWLINE, TokenKind.INDENT,
                                  TokenKind.DEDENT, TokenKind.EOF)]

    def test_line_and_block_comments(self):
        from gamag.tokens import TokenKind
        tokens = self.tokenize('// a comment\nlet x = 1 /* inline */\n')
        self.assertEqual(self.kinds("// just a comment\n"), [])
        self.assertIn(TokenKind.LET, self.kinds("// c\nlet x = 1\n"))

    def test_three_char_range_is_not_two_chars(self):
        """`..=` must lex as one token, not `..` followed by `=`."""
        from gamag.tokens import TokenKind
        self.assertEqual(self.kinds("1..=3"),
                         [TokenKind.INT, TokenKind.RANGE_INC, TokenKind.INT])
        self.assertEqual(self.kinds("1..3"),
                         [TokenKind.INT, TokenKind.RANGE, TokenKind.INT])

    def test_arrow_versus_fat_arrow(self):
        from gamag.tokens import TokenKind
        self.assertEqual(self.kinds("-> =>"),
                         [TokenKind.ARROW, TokenKind.FATARROW])

    def test_string_escapes(self):
        tokens = [t for t in self.tokenize(r'"a\nb\t\"c\""')
                  if t.kind.name == "STRING"]
        self.assertEqual(tokens[0].value, 'a\nb\t"c"')


class Parsing(unittest.TestCase):
    """Spec sections 3-5: declarations, blocks and indentation."""

    def parse(self, source):
        from gamag.lexer import Lexer
        from gamag.parser import Parser
        tokens = Lexer(source, "<test>").tokenize()
        return Parser(tokens, "<test>", source).parse_module()

    def test_indented_and_braced_blocks_agree(self):
        """The two block spellings must produce the same tree shape."""
        indented = self.parse("fn f() -> I64\n    pure\n    return 1\n")
        braced = self.parse("fn f() -> I64 {\n    pure\n    return 1\n}\n")
        self.assertEqual(len(indented.decls), len(braced.decls))
        self.assertEqual(indented.decls[0].name, braced.decls[0].name)

    def test_multi_line_match_arm_body(self):
        """A newline after `=>` opens an indented body (regression)."""
        module = self.parse("""
fn f(x: I64) -> I64
    pure
    match x
        1 =>
            let y = x + 1
            return y
        _ => return 0
""")
        self.assertEqual(len(module.decls), 1)

    def test_arrow_allows_the_return_type_on_the_next_line(self):
        module = self.parse("fn f(x: I64)\n    -> I64\n    pure\n    return x\n")
        self.assertEqual(len(module.decls), 1)

    def test_trailing_operator_continues_a_line(self):
        module = self.parse("fn f() -> I64\n    pure\n    return 1 +\n        2\n")
        self.assertEqual(len(module.decls), 1)

    def test_declaration_kinds_are_recognised(self):
        module = self.parse("""
grant Read
record R
    a: I64
enum E
    V
fn f() -> Unit
    pure
    return
model M
    input x: Tensor<F64>
    output y: F64
    predict x
        return tensor.item(x)
policy P
    allow role("a")
transaction T
    audit
agent A
    on ping
        print(ping)
service S
    protect
        print("p")
    recover
        retry 1
pipeline PL
    io
    input a: Text
    normalize
    return result
test "t"
    assert true
""")
        names = {type(d).__name__ for d in module.decls}
        for expected in ("GrantDecl", "RecordDecl", "EnumDecl", "FnDecl",
                         "ModelDecl", "PolicyDecl", "TransactionDecl",
                         "AgentDecl", "ServiceDecl", "PipelineDecl",
                         "TestDecl"):
            self.assertIn(expected, names, f"{expected} was not parsed")


class ArithmeticAndTypes(unittest.TestCase):
    """Spec section 6: a strict, explicitly converted type system."""

    def test_integer_arithmetic(self):
        outcome = S.run("""
fn main() -> Unit
    io
    print(2 + 3 * 4, (2 + 3) * 4, 17 / 5, 17 % 5, 2 - 10)
""")
        outcome.assert_output_contains(self, "14 20 3 2 -8")

    def test_no_implicit_numeric_widening(self):
        outcome = S.compile_only("""
fn f(x: F64) -> F64
    pure
    return x

fn main() -> Unit
    io
    print(f(1))
""")
        outcome.assert_rejected(self, code="E-arg-type")
        self.assertIn("float(x)", outcome.messages(),
                      "the diagnostic should say how to fix it")

    def test_explicit_conversion_is_accepted(self):
        outcome = S.run("""
fn main() -> Unit
    io
    print(float(1) + 0.5, int(2.0))
""")
        outcome.assert_output_contains(self, "1.5 2")

    def test_a_lossy_conversion_is_refused_rather_than_truncating(self):
        """`int(2.9)` would silently lose information, so it faults instead."""
        outcome = S.run("""
fn main() -> Unit
    io
    print(int(2.9))
""")
        outcome.assert_faulted(self)
        self.assertIn("truncation", str(outcome.fault))

    def test_integer_literals_adopt_a_sized_type(self):
        """A literal that does not fit its declared type is rejected."""
        outcome = S.compile_only("""
fn main() -> Unit
    io
    let small: I8 = 200
    print(small)
""")
        outcome.assert_rejected(self)

    def test_boolean_arithmetic_is_refused(self):
        outcome = S.compile_only("""
fn main() -> Unit
    io
    print(true + 1)
""")
        outcome.assert_rejected(self)

    def test_short_circuit_truth_tables(self):
        """`and`/`or` must not evaluate the right operand needlessly."""
        outcome = S.run("""
fn boom() -> Bool
    pure
    panic("evaluated")

fn main() -> Unit
    io
    print(false and boom())
    print(true or boom())
    print(true and true)
    print(false or false)
""")
        outcome.assert_ran(self)
        self.assertEqual(outcome.output.split(),
                         ["false", "true", "true", "false"])

    def test_text_operations(self):
        outcome = S.run("""
fn main() -> Unit
    io
    let s = "Hello, world"
    print(s.upper(), s.length, s.contains("world"))
    print(text.split(s, ", "), s.slice(0, 5))
""")
        outcome.assert_output_contains(self, "HELLO, WORLD", "12", "true",
                                       "[Hello, world]", "Hello")


class ControlFlow(unittest.TestCase):
    """Spec sections 4 and 5."""

    def test_if_requires_a_real_boolean(self):
        outcome = S.compile_only("""
fn main() -> Unit
    io
    if 1
        print("truthy")
""")
        outcome.assert_rejected(self, code="E-cond-type")

    def test_while_and_for_over_a_range(self):
        outcome = S.run("""
fn main() -> Unit
    io
    var total = 0
    for i in 1..=5
        total = total + i
    print("exclusive-then-inclusive:", total)
    var n = 0
    while n < 3
        n = n + 1
    print("while:", n)
""")
        outcome.assert_output_contains(self, "15", "while: 3")

    def test_match_is_exhaustive_and_binds_payloads(self):
        outcome = S.run("""
enum Shape
    Circle(r: F64)
    Square(side: F64)

fn area(s: Shape) -> F64
    pure
    match s
        Circle(r) => return 3.14159 * r * r
        Square(side) => return side * side

fn main() -> Unit
    io
    print(area(Circle(1.0)), area(Square(3.0)))
""")
        outcome.assert_output_contains(self, "3.14159", "9.0")

    def test_guards_select_the_first_matching_arm(self):
        outcome = S.run("""
fn classify(n: I64) -> Text
    pure
    match n
        x if x < 0 => return "negative"
        0 => return "zero"
        _ => return "positive"

fn main() -> Unit
    io
    print(classify(-5), classify(0), classify(5))
""")
        outcome.assert_output_contains(self, "negative zero positive")


class ResultAndOption(unittest.TestCase):
    """Spec section 6: errors are values."""

    def test_result_chaining(self):
        outcome = S.run("""
fn parse(s: Text) -> Result<I64, Text>
    pure
    // `text.parse_int` yields an Option: parsing can legitimately fail.
    match text.parse_int(s)
        some(n) => return ok(n)
        none => return fail("unparseable")

fn double(s: Text) -> Result<I64, Text>
    pure
    match parse(s)
        ok(n) => return ok(n * 2)
        fail(e) => return fail(e)

fn main() -> Unit
    io
    match double("21")
        ok(n) => print("ok:", n)
        fail(e) => print("err:", e)
    match double("bad")
        ok(n) => print("ok:", n)
        fail(e) => print("err:", e)
""")
        outcome.assert_output_contains(self, "ok: 42", "err: unparseable")

    def test_option(self):
        outcome = S.run("""
fn find(xs: List<I64>, needle: I64) -> Option<I64>
    pure
    for x in xs
        if x == needle
            return some(x)
    return none

fn main() -> Unit
    io
    match find([1, 2, 3], 2)
        some(v) => print("found", v)
        none => print("missing")
    match find([1, 2, 3], 9)
        some(v) => print("found", v)
        none => print("missing")
""")
        outcome.assert_output_contains(self, "found 2", "missing")


class RecordsAndOwnership(unittest.TestCase):
    """Spec sections 4 and 6: records, and immutable by default."""

    def test_record_construction_and_access(self):
        outcome = S.run("""
record Point
    x: F64
    y: F64
    label: Text

fn main() -> Unit
    io
    let p = Point { x: 1.0, y: 2.0, label: "origin" }
    print(p.x, p.y, p.label)
""")
        outcome.assert_output_contains(self, "1.0 2.0 origin")

    def test_let_is_immutable_and_var_is_not(self):
        outcome = S.compile_only("""
fn main() -> Unit
    io
    let x = 1
    x = 2
    print(x)
""")
        outcome.assert_rejected(self, code="E-immutable")
        phases = {d.phase.name for d in outcome.diagnostics}
        self.assertIn("OWNERSHIP", phases,
                      "immutability is an ownership concern (spec section 6)")

    def test_var_can_be_reassigned(self):
        outcome = S.run("""
fn main() -> Unit
    io
    var x = 1
    x = 2
    print(x)
""")
        outcome.assert_output_contains(self, "2")

    def test_collections_are_functional_not_mutating(self):
        """`push` returns a new list; the receiver is unchanged."""
        outcome = S.run("""
fn main() -> Unit
    io
    let xs = [1, 2]
    let ys = xs.push(3)
    print("original:", xs)
    print("extended:", ys)
""")
        outcome.assert_output_contains(self, "original: [1, 2]",
                                       "extended: [1, 2, 3]")


class Contracts(unittest.TestCase):
    """Spec section 27."""

    def test_requires_is_checked_on_entry(self):
        outcome = S.run("""
fn f(x: I64) -> I64
    pure
    requires x > 0
    return x

fn main() -> Unit
    io
    print(f(0))
""")
        outcome.assert_faulted(self, kind="ContractViolation")
        self.assertIn("x > 0", str(outcome.fault))

    def test_a_satisfied_contract_is_silent(self):
        outcome = S.run("""
fn f(x: I64) -> I64
    pure
    requires x > 0
    ensures result > x
    return x + 1

fn main() -> Unit
    io
    print(f(1))
""")
        outcome.assert_output_contains(self, "2")


class Recursion(unittest.TestCase):
    def test_recursive_function(self):
        outcome = S.run("""
fn fact(n: I64) -> I64
    pure
    requires n >= 0
    if n <= 1
        return 1
    return n * fact(n - 1)

fn main() -> Unit
    io
    print(fact(10))
""")
        outcome.assert_output_contains(self, "3628800")


class ParallelRegions(unittest.TestCase):
    """Spec section 3: independent operations may run concurrently."""

    def test_independent_tasks_all_complete(self):
        outcome = S.run("""
fn double(x: I64) -> I64
    pure
    return x * 2

fn main() -> Unit
    io
    parallel
        a = double(1)
        b = double(2)
        c = double(3)
    print(a, b, c)
""")
        outcome.assert_output_contains(self, "2 4 6")

    def test_a_task_may_not_perform_an_effect(self):
        """Spec section 9C: a region computes its outputs and nothing else.

        Committing writes in program order orders the state a region leaves
        behind; it cannot order what its tasks say to the outside world, so an
        effect inside a region would make the program's output depend on the
        schedule.  Measured before the rule existed: three printing tasks of
        deliberately unequal cost produced `gamma, beta, alpha` once and
        `beta, gamma, alpha` another time, on separate runs of one unchanged
        program.
        """
        outcome = S.run("""
fn loud(x: I64) -> I64
    io
    print("computing", x)
    return x * 2

fn main() -> Unit
    io
    parallel
        a = loud(1)
        b = loud(2)
    print(a, b)
""")
        outcome.assert_rejected(self, "E-parallel-effect")

    def test_an_effect_is_refused_even_when_the_region_writes_state(self):
        """A second witness for the same rule, through a different statement.

        `audit.record` is an effect too, and it is refused inside a region for
        the same reason `print` is.
        """
        outcome = S.run("""
fn main() -> Unit
    io
    audit
    parallel
        a = 1
        b = 2
        audit.record { actor: "x", action: "y" }
    print(a + b)
""")
        outcome.assert_rejected(self, "E-parallel-effect")

    def test_an_effect_before_the_region_does_not_mask_one_inside_it(self):
        """Regression: the first implementation compared the effect set before
        and after the region, so an `io` call before the region hid an `io`
        call inside it and a racy region was accepted."""
        outcome = S.run("""
fn loud(x: I64) -> I64
    io
    return x * 2

fn main() -> Unit
    io
    print("start")
    parallel
        a = loud(1)
        b = loud(2)
    print(a, b)
""")
        outcome.assert_rejected(self, "E-parallel-effect")

    def test_a_pure_region_that_reads_the_results_before_it_is_accepted(self):
        """The shape the rule pushes programs towards: gather, compute, report."""
        outcome = S.run("""
fn double(x: I64) -> I64
    pure
    return x * 2

fn main() -> Unit
    io
    let raw = 21
    parallel
        a = double(raw)
    print(a)
""")
        outcome.assert_output_contains(self, "42")

    def test_dependent_tasks_are_ordered(self):
        outcome = S.run("""
fn main() -> Unit
    io
    parallel
        a = 1 + 1
        b = a * 10
        c = b + a
    print(a, b, c)
""")
        outcome.assert_output_contains(self, "2 20 22")

    def test_two_regions_in_one_function_do_not_collide(self):
        """Regression: task functions used to be named per function, not per region."""
        outcome = S.run("""
fn main() -> Unit
    io
    parallel
        p = 1
        q = 2
    parallel
        r = 3
        s = 4
    print(p, q, r, s)
""")
        outcome.assert_output_contains(self, "1 2 3 4")


class DeclarationNames(unittest.TestCase):
    """One module, one namespace: a name identifies one declaration.

    Nothing in the specification has ever permitted two declarations to share
    a name, and three different things used to happen depending on the kind.
    A duplicate record or enum was accepted in silence, the second definition
    quietly replacing the first.  A duplicate function surfaced as ``E-ice:
    internal compiler error ... a later definition would silently replace
    it`` -- the compiler reporting a plain user error as its own bug, three
    phases after the mistake, with a message that named no line of source.
    """

    def test_a_duplicate_function_is_a_name_resolution_error(self):
        outcome = S.run("""
fn calc(x: I64) -> I64
    pure
    return x + 1

fn calc(x: I64) -> I64
    pure
    return x + 2

fn main() -> Unit
    io
    print(calc(1))
""")
        outcome.assert_rejected(self, "E-duplicate-declaration")
        self.assertNotIn("E-ice", outcome.messages())

    def test_a_duplicate_record_is_refused(self):
        outcome = S.run("""
record Point
    x: I64

record Point
    y: I64

fn main() -> Unit
    io
    print(1)
""")
        outcome.assert_rejected(self, "E-duplicate-declaration")

    def test_a_duplicate_enum_is_refused(self):
        outcome = S.run("""
enum Colour
    Red

enum Colour
    Blue

fn main() -> Unit
    io
    print(1)
""")
        outcome.assert_rejected(self, "E-duplicate-declaration")

    def test_the_namespace_is_shared_across_kinds(self):
        """A function and a record cannot both be `Widget` either."""
        outcome = S.run("""
record Widget
    x: I64

fn Widget(x: I64) -> I64
    pure
    return x

fn main() -> Unit
    io
    print(1)
""")
        outcome.assert_rejected(self, "E-duplicate-declaration")

    def test_a_function_may_still_shadow_a_builtin(self):
        """The rule is about the module's own namespace.  A user function named
        after a builtin is a deliberate, already-tested feature, and it stays:
        `print` here is the user's, not the prelude's."""
        outcome = S.run("""
fn print(a: I64) -> I64
    pure
    return a + 1

fn main() -> Unit
    io
    let plain = print(1)
    println(plain)
""")
        self.assertTrue(outcome.compiled, outcome.messages())


class GirAndOptimizer(unittest.TestCase):
    """Spec section 23: the GIR and its optimization pipeline."""

    def test_gir_is_produced_for_every_declaration(self):
        outcome = S.compile_only("""
fn f(x: I64) -> I64
    pure
    return x + 1

fn main() -> Unit
    io
    print(f(1))
""")
        self.assertTrue(outcome.compiled, outcome.messages())
        program = outcome.compilation.program
        self.assertIn("f", program.functions)
        self.assertIn("main", program.functions)

    def test_gir_is_serialisable(self):
        """Tooling depends on the GIR being data, not objects."""
        import json
        outcome = S.compile_only("fn f(x: I64) -> I64\n    pure\n    return x\n")
        self.assertTrue(outcome.compiled, outcome.messages())
        payload = json.loads(outcome.compilation.program.to_json())
        self.assertIn("functions", payload)

    def test_optimization_preserves_behaviour(self):
        source = """
fn f(x: I64) -> I64
    pure
    return (x + 1) * 2

fn main() -> Unit
    io
    print(f(3), f(10))
"""
        for level in (0, 1, 2):
            with self.subTest(opt_level=level):
                outcome = S.run(source, opt_level=level)
                outcome.assert_output_contains(self, "8 22")

    def test_dead_code_is_identified(self):
        outcome = S.compile_only("""
fn main() -> Unit
    io
    let unused = 1 + 2
    print("done")
""", opt_level=2)
        self.assertTrue(outcome.compiled, outcome.messages())
        report = outcome.compilation.optimization
        self.assertIsNotNone(report)


if __name__ == "__main__":
    unittest.main()
