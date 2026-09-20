"""Conformance with the enumerations in the production specification.

The blueprint does not only show code; it *lists* the vocabulary the language
must have -- scalar types (spec lines 208-216), container types (220-227),
domain types (231-236), effect names (277-285), recovery levels (380-385) and
capability names (428-436).  Each list is read back out of the document at
run time, so if the specification is edited these tests follow it instead of
silently testing a stale copy.
"""

from __future__ import annotations

import unittest

import support as S


class ScalarTypeVocabulary(unittest.TestCase):
    """Spec lines 208-216: every scalar type name must be usable."""

    # (type name, a value of that type) -- the value proves the type is not
    # merely a reserved word but actually inhabited.
    SCALARS = [
        ("Bool", "true"), ("I8", "1"), ("I16", "1"), ("I32", "1"),
        ("I64", "1"), ("I128", "1"),
        ("U8", "1"), ("U16", "1"), ("U32", "1"), ("U64", "1"), ("U128", "1"),
        ("F16", "1.0"), ("F32", "1.0"), ("F64", "1.0"),
        ("Decimal", "1.5"), ("Char", "'a'"), ("Text", '"a"'),
        ("Bytes", 'bytes.from_text("ab")'),
    ]

    def test_spec_lists_these_names(self):
        """Guard against the test drifting away from the document."""
        listed = " ".join(S.SpecList.load("scalars", 208, 216).items)
        for name, _ in self.SCALARS:
            self.assertIn(name, listed,
                          f"{name} is tested but not in spec lines 208-216")
        self.assertIn("Unit", listed)

    def test_every_scalar_is_inhabited(self):
        for name, value in self.SCALARS:
            with self.subTest(type=name):
                outcome = S.compile_only(
                    f"fn probe(x: {name}) -> {name}\n    pure\n    return x\n"
                    f"\nfn main() -> Unit\n    io\n    let v: {name} = {value}\n"
                    f"    print(v)\n")
                if not outcome.compiled:
                    self.fail(f"{name} is in the specification but does not "
                              f"compile:\n{outcome.messages()}")


class ContainerTypeVocabulary(unittest.TestCase):
    """Spec lines 220-227: the container and sum types."""

    CONTAINERS = [
        ("List<I64>", "[1, 2]"),
        ("Map<Text, I64>", '{"a": 1}'),
        ("Set<I64>", "{1, 2}"),
        ("Option<I64>", "some(1)"),
        ("Result<I64, Text>", "ok(1)"),
    ]

    def test_spec_lists_these_names(self):
        listed = " ".join(S.SpecList.load("containers", 220, 227).items)
        for stem in ("List", "Map", "Set", "Tuple", "Record", "Enum",
                     "Option", "Result"):
            self.assertIn(stem, listed)

    def test_every_container_is_inhabited(self):
        for name, value in self.CONTAINERS:
            with self.subTest(type=name):
                outcome = S.compile_only(
                    f"fn main() -> Unit\n    io\n    let v: {name} = {value}\n"
                    f"    print(v)\n")
                self.assertTrue(outcome.compiled,
                                f"{name} failed:\n{outcome.messages()}")

    def test_tuple_record_and_enum_are_declarable(self):
        """`Tuple(...)`, `Record` and `Enum` are forms, not literal names."""
        outcome = S.run("""
record Point
    x: F64
    y: F64

enum Colour
    Red
    Green

fn main() -> Unit
    io
    let pair: Tuple(I64, Text) = (1, "a")
    let p = Point { x: 1.0, y: 2.0 }
    print(pair, p.x, Colour.Red)
""")
        self.assertTrue(outcome.ran, outcome.messages() or str(outcome.fault))


class DomainTypeVocabulary(unittest.TestCase):
    """Spec lines 231-236: AI/ML and identity types."""

    def test_spec_lists_these_names(self):
        listed = " ".join(S.SpecList.load("domain", 231, 236).items)
        for stem in ("Tensor", "Matrix", "Duration", "Instant", "UUID", "URI"):
            self.assertIn(stem, listed)

    def test_tensor_and_matrix_are_usable(self):
        outcome = S.run("""
fn main() -> Unit
    io
    model
    let t: Tensor<F64> = tensor.from_list([[1.0, 2.0]])
    print(tensor.shape(t), tensor.rank(t))
""")
        self.assertTrue(outcome.ran, outcome.messages() or str(outcome.fault))
        self.assertIn("[1, 2]", outcome.output)

    def test_duration_instant_uuid_uri(self):
        outcome = S.run("""
fn main() -> Unit
    io
    crypto
    let d: Duration = time.duration(5.0)
    let now: Instant = time.now()
    let id: UUID = crypto.uuid()
    let where: URI = uri("https://example.invalid/a")
    let raw: Bytes = bytes.from_text("ab")
    print(d.seconds, id.text, where.text, bytes.length(raw))
    print(now.epoch > 0.0)
""")
        self.assertTrue(outcome.ran, outcome.messages() or str(outcome.fault))


class EffectVocabulary(unittest.TestCase):
    """Spec lines 277-285: the nine effect names."""

    EFFECTS = ["pure", "io", "network", "storage", "crypto", "model",
               "medical", "audit", "unsafe"]

    def test_matches_the_specification_exactly(self):
        listed = S.SpecList.load("effects", 277, 285).items
        self.assertEqual(listed, self.EFFECTS,
                         "the effect list in the specification changed")

    def test_matches_the_token_vocabulary(self):
        from gamag.tokens import EFFECT_NAMES
        self.assertEqual(set(EFFECT_NAMES), set(self.EFFECTS))

    def test_every_effect_is_declared(self):
        for effect in self.EFFECTS:
            with self.subTest(effect=effect):
                outcome = S.compile_only(
                    f"fn probe() -> Unit\n    {effect}\n    return\n")
                self.assertTrue(outcome.compiled,
                                f"effect `{effect}` rejected:\n"
                                f"{outcome.messages()}")


class RecoveryLevelVocabulary(unittest.TestCase):
    """Spec lines 380-385: the six bounded recovery levels."""

    LEVELS = ["local retry", "resource reset", "state checkpoint restore",
              "component restart", "failover", "operator escalation"]

    def test_spec_lists_six_levels(self):
        listed = S.SpecList.load("levels", 380, 385).items
        self.assertEqual(len(listed), 6)
        for index, name in enumerate(self.LEVELS):
            self.assertTrue(listed[index].startswith(f"LEVEL {index}"),
                            f"expected LEVEL {index}, got {listed[index]!r}")
            self.assertEqual(listed[index].split(":", 1)[1].strip(), name)

    def test_runtime_names_them_in_order(self):
        from gamag.runtime.recovery import LEVEL_NAMES
        self.assertEqual(sorted(LEVEL_NAMES), list(range(6)))
        for index, name in enumerate(self.LEVELS):
            with self.subTest(level=index):
                self.assertEqual(LEVEL_NAMES[index].lower(), name)


class CapabilityVocabulary(unittest.TestCase):
    """Spec lines 428-436: the capability names a program can be granted."""

    CAPABILITIES = ["FileRead", "FileWrite", "NetworkConnect", "DatabaseRead",
                    "DatabaseWrite", "PatientRead", "PatientWrite",
                    "CryptoSign", "AuditWrite"]

    def test_matches_the_specification_exactly(self):
        listed = S.SpecList.load("capabilities", 428, 436).items
        self.assertEqual(listed, self.CAPABILITIES,
                         "the capability list in the specification changed")

    def test_every_capability_can_be_granted(self):
        for capability in self.CAPABILITIES:
            with self.subTest(capability=capability):
                outcome = S.compile_only(
                    f"grant {capability}\n\nfn main() -> Unit\n    io\n"
                    f"    print(\"granted\")\n",
                    grants=(capability,))
                self.assertTrue(outcome.compiled,
                                f"`grant {capability}` rejected:\n"
                                f"{outcome.messages()}")

    def test_ungranted_capability_is_refused(self):
        """Spec section 12: no ambient authority."""
        source = ('grant FileRead\n\nfn main() -> Unit\n    io\n'
                  '    crypto\n'
                  '    let h = capabilities.open("Store", ["Write"])\n'
                  '    print(h)\n')
        # The permission list is a literal, so this is caught before running.
        outcome = S.compile_only(source, grants=("FileRead",))
        outcome.assert_rejected(self, code="E-capability-missing")
        # A list computed at run time cannot be judged statically, so the
        # runtime must still refuse it.
        dynamic = S.run(
            'grant FileRead\n\nfn main() -> Unit\n    io\n    crypto\n'
            '    let perms = ["Write"]\n'
            '    let h = capabilities.open("Store", perms)\n    print(h)\n',
            grants=("FileRead",))
        dynamic.assert_faulted(self, kind="CapabilityViolation")


class StandardLibraryVocabulary(unittest.TestCase):
    """Spec lines 855-897: the standard library's module groups."""

    GROUPS = {
        "core": (855, 862),
        "security": (866, 871),
        "ai": (875, 881),
        "medical": (885, 889),
        "enterprise": (893, 898),
    }

    def test_modules_are_listed(self):
        found = set()
        for _, (first, last) in self.GROUPS.items():
            found.update(S.SpecList.load("stdlib", first, last).items)
        for name in ("collections", "text", "math", "time", "crypto",
                     "secrets", "audit", "policy", "tensor", "autodiff",
                     "medical", "database", "http"):
            self.assertIn(name, found, f"module `{name}` missing from spec")

    def test_every_listed_module_is_implemented_or_declared_unimplemented(self):
        """Spec section 43: never claim a capability the toolchain lacks.

        A module the blueprint names must either expose dotted builtins or be
        listed in UNIMPLEMENTED_MODULES *with a reason*, so a programmer who
        reaches for `fhir.serialize(...)` is told it is roadmap rather than
        getting a confusing "cannot find `fhir`".

        `core` is the one exception: its functions are the unqualified prelude
        (`print`, `len`, `str`), so it has no `core.` prefix.
        """
        from gamag.std import library as L
        unexplained = []
        for _, (first, last) in self.GROUPS.items():
            for module in S.SpecList.load("stdlib", first, last).items:
                if module == "core":
                    continue
                prefix = module + "."
                implemented = any(n.startswith(prefix) for n in L.BUILTINS)
                declared = module in L.UNIMPLEMENTED_MODULES
                if not implemented and not declared:
                    unexplained.append(module)
                if implemented and declared:
                    unexplained.append(f"{module} (both)")
        self.assertEqual(
            unexplained, [],
            "these modules are neither implemented nor declared as roadmap: "
            + ", ".join(sorted(set(unexplained))))

    def test_core_prelude_is_unqualified(self):
        from gamag.std import library as L
        for name in ("print", "len", "str", "sorted", "contains"):
            self.assertIn(name, L.PRELUDE,
                          f"`{name}` should be prelude, not module-qualified")

    def test_unimplemented_modules_explain_themselves(self):
        from gamag.std import library as L
        self.assertTrue(L.UNIMPLEMENTED_MODULES,
                        "no roadmap modules are declared at all")
        placeholders = {"todo", "tbd", "n/a", "later", "fixme", ""}
        for module, reason in L.UNIMPLEMENTED_MODULES.items():
            with self.subTest(module=module):
                text = reason.strip()
                self.assertNotIn(text.lower(), placeholders,
                                 f"the reason for `{module}` is a placeholder, "
                                 f"which tells a programmer nothing")
                self.assertGreater(len(text), 8,
                                   f"the reason for `{module}` is too short to "
                                   f"be useful: {reason!r}")
                self.assertNotIn(module, L.MODULE_TYPE_NAMES,
                                 f"`{module}` is declared unimplemented but is "
                                 f"also registered as a module name")


class ToolchainVocabulary(unittest.TestCase):
    """Spec lines 949-957: the commands `ggc` must provide."""

    def test_commands_are_listed(self):
        listed = S.SpecList.load("commands", 949, 957).items
        for command in ("ggc build", "ggc run", "ggc test", "ggc check",
                        "ggc audit"):
            self.assertIn(command, listed)

    def test_implemented_subcommands(self):
        from gamag.cli.main import build_parser
        parser = build_parser()
        actions = {a.dest: a for a in parser._actions}
        choices = set(actions["command"].choices)
        for command in ("check", "build", "run", "test", "explain"):
            self.assertIn(command, choices,
                          f"`ggc {command}` is not implemented")

    #: Commands the specification describes and this toolchain does not have.
    #: Spec section 43 forbids claiming capability the toolchain lacks, so each
    #: must be absent rather than present-and-pretending.
    ROADMAP_COMMANDS = ("profile", "format", "doc")

    def test_roadmap_commands_are_declared_as_unimplemented(self):
        """A command that does not exist must not appear to.

        `bench` was on this list until audit priority 8 implemented it; a
        command leaving this list is the signal that its documentation and
        roadmap entry need updating too.
        """
        from gamag.cli.main import build_parser
        choices = set(
            {a.dest: a for a in build_parser()._actions}["command"].choices)
        for command in self.ROADMAP_COMMANDS:
            self.assertNotIn(
                command, choices,
                f"`ggc {command}` now exists; move it out of the roadmap list "
                f"in docs/IMPLEMENTATION.md")

    def test_the_commands_that_do_exist_are_all_reachable(self):
        # The other half of the check above: a command that has been built must
        # actually be wired in, not left as a parser with no handler.
        from gamag.cli.main import COMMANDS, build_parser
        choices = set(
            {a.dest: a for a in build_parser()._actions}["command"].choices)
        for command in ("check", "build", "gir", "graph", "memory", "run",
                        "test", "explain", "native", "difftest", "wasm",
                        "device", "fuzz", "gpm", "manifest", "bench"):
            self.assertIn(command, choices, f"`ggc {command}` is missing")
            self.assertIn(command, COMMANDS,
                          f"`ggc {command}` has no handler")


if __name__ == "__main__":
    unittest.main()
