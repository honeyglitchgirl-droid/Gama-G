"""Signing, reproducible builds and the package manager.

Audit priorities 10 and 11, and spec sections 12 ("signed packages preferred,
dependency integrity verification, reproducible builds supported") and 30 (gpm).

The Ed25519 tests use the RFC's own test vectors.  That is the point of having
them: they are published, they are independent of this implementation, and they
cover key generation, signing and verification, so a mistake in the field
arithmetic or the encoding shows up as a failing vector rather than as a
signature that happens to verify against itself.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import unittest

import support as S
from gamag.crypto import ed25519
from gamag.gpm import package as gpm
from gamag.gpm.semver import Constraint, Version, VersionError, highest
from gamag.toolchain import buildinfo


# ---------------------------------------------------------------------------
# Ed25519
# ---------------------------------------------------------------------------

class Ed25519Vectors(unittest.TestCase):
    """RFC 8032 section 7.1, verbatim."""

    VECTORS = [
        ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
         "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
         "",
         "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
        ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
         "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
         "72",
         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
        ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
         "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
         "af82",
         "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
    ]

    def test_public_keys_match_the_rfc(self):
        for secret, public, _message, _signature in self.VECTORS:
            self.assertEqual(
                ed25519.to_hex(ed25519.public_key(ed25519.from_hex(secret))),
                public)

    def test_signatures_match_the_rfc(self):
        for secret, _public, message, signature in self.VECTORS:
            self.assertEqual(
                ed25519.to_hex(
                    ed25519.sign(ed25519.from_hex(secret),
                                 ed25519.from_hex(message))),
                signature)

    def test_the_rfc_signatures_verify(self):
        for _secret, public, message, signature in self.VECTORS:
            self.assertTrue(ed25519.verify(ed25519.from_hex(public),
                                           ed25519.from_hex(message),
                                           ed25519.from_hex(signature)))

    def test_a_changed_message_does_not_verify(self):
        for _secret, public, message, signature in self.VECTORS:
            self.assertFalse(ed25519.verify(ed25519.from_hex(public),
                                            ed25519.from_hex(message) + b"!",
                                            ed25519.from_hex(signature)))

    def test_a_changed_signature_does_not_verify(self):
        _secret, public, message, signature = self.VECTORS[0]
        broken = bytearray(ed25519.from_hex(signature))
        broken[0] ^= 0x01
        self.assertFalse(ed25519.verify(ed25519.from_hex(public),
                                        ed25519.from_hex(message),
                                        bytes(broken)))

    def test_the_wrong_key_does_not_verify(self):
        _s, _p, message, signature = self.VECTORS[0]
        other = ed25519.from_hex(self.VECTORS[1][1])
        self.assertFalse(ed25519.verify(other, ed25519.from_hex(message),
                                        ed25519.from_hex(signature)))

    def test_a_malformed_key_or_signature_is_refused_not_raised(self):
        # Verification returns a bool.  A caller that has to handle an exception
        # here is a caller that will eventually forget to.
        for public, signature in ((b"", b""), (b"\x00" * 31, b"\x00" * 64),
                                  (b"\x00" * 32, b"\x00" * 63)):
            self.assertFalse(ed25519.verify(public, b"message", signature))

    def test_a_non_canonical_scalar_is_rejected_under_strict(self):
        # RFC 8032 section 8.4: a malleable signature is one an attacker can
        # change while keeping it valid, so S >= L is refused.
        _secret, public, message, signature = self.VECTORS[0]
        raw = bytearray(ed25519.from_hex(signature))
        scalar = int.from_bytes(raw[32:], "little") + ed25519.L
        raw[32:] = scalar.to_bytes(32, "little")
        key = ed25519.from_hex(public)
        self.assertFalse(ed25519.verify(key, ed25519.from_hex(message),
                                        bytes(raw), strict=True))

    def test_a_wrong_length_key_is_refused(self):
        with self.assertRaises(ValueError):
            ed25519.public_key(b"short")
        with self.assertRaises(ValueError):
            ed25519.sign(b"short", b"message")

    def test_key_generation_is_deterministic_from_a_seed(self):
        seed = bytes(range(32))
        first_secret, first_public = ed25519.keypair_from_seed(seed)
        second_secret, second_public = ed25519.keypair_from_seed(seed)
        self.assertEqual(first_secret, second_secret)
        self.assertEqual(first_public, second_public)
        self.assertEqual(len(first_public), 32)


# ---------------------------------------------------------------------------
# Semantic versioning
# ---------------------------------------------------------------------------

class SemVer(unittest.TestCase):

    #: SemVer 2.0.0 section 11, in precedence order.
    CHAIN = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
             "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0",
             "1.0.1", "1.1.0", "2.0.0"]

    def test_the_precedence_chain_from_the_specification(self):
        versions = [Version.parse(text) for text in self.CHAIN]
        for left, right in zip(versions, versions[1:]):
            self.assertLess(left, right, f"{left} should sort below {right}")

    def test_a_prerelease_sorts_below_its_release(self):
        self.assertLess(Version.parse("1.0.0-rc.1"), Version.parse("1.0.0"))

    def test_a_numeric_prerelease_sorts_below_an_alphanumeric_one(self):
        self.assertLess(Version.parse("1.0.0-1"), Version.parse("1.0.0-a"))

    def test_build_metadata_is_ignored_for_precedence(self):
        self.assertEqual(Version.parse("1.0.0+a"), Version.parse("1.0.0+b"))
        self.assertFalse(Version.parse("1.0.0+a") < Version.parse("1.0.0+b"))

    def test_malformed_versions_are_refused(self):
        for text in ("1.0", "1", "1.0.0.0", "v1.0.0", "1.00.0",
                     "1.0.0-", "1.0.0-01", ""):
            with self.assertRaises(VersionError, msg=text):
                Version.parse(text)

    def test_caret_allows_the_whole_major_range(self):
        constraint = Constraint.parse("^1.2.3")
        self.assertTrue(constraint.accepts(Version.parse("1.2.3")))
        self.assertTrue(constraint.accepts(Version.parse("1.9.9")))
        self.assertFalse(constraint.accepts(Version.parse("2.0.0")))
        self.assertFalse(constraint.accepts(Version.parse("1.2.2")))

    def test_caret_is_stricter_below_one_point_zero(self):
        # `^0.2.3` must not accept 0.3.0: in 0.x every minor release may break,
        # and a resolver that ignores this installs an incompatible version.
        constraint = Constraint.parse("^0.2.3")
        self.assertTrue(constraint.accepts(Version.parse("0.2.9")))
        self.assertFalse(constraint.accepts(Version.parse("0.3.0")))

    def test_tilde_allows_patches_only(self):
        constraint = Constraint.parse("~1.2.3")
        self.assertTrue(constraint.accepts(Version.parse("1.2.9")))
        self.assertFalse(constraint.accepts(Version.parse("1.3.0")))

    def test_prereleases_are_not_installed_by_default(self):
        self.assertFalse(Constraint.parse("^1.0.0").accepts(
            Version.parse("2.0.0-rc1")))
        self.assertFalse(Constraint.parse("^1.0.0").accepts(
            Version.parse("1.5.0-beta")))
        self.assertTrue(Constraint.parse("^1.0.0").accepts(
            Version.parse("1.5.0-beta"), allow_prerelease=True))

    def test_highest_picks_the_top_of_the_accepted_range(self):
        versions = [Version.parse(v) for v in
                    ("1.0.0", "1.4.2", "2.1.0", "1.3.0")]
        self.assertEqual(highest(versions, Constraint.parse("^1.0.0")),
                         Version.parse("1.4.2"))
        self.assertIsNone(highest(versions, Constraint.parse("^3.0.0")))


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------

class PackageTree(unittest.TestCase):
    """A small registry, built once per test and thrown away."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry_root = os.path.join(self._tmp.name, "registry")

    def make(self, name, version, dependencies=None, extra_file=None):
        directory = os.path.join(self.registry_root, name, version)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, gpm.MANIFEST_NAME), "w",
                  encoding="utf-8") as handle:
            json.dump({"name": name, "version": version,
                       "dependencies": dependencies or {}}, handle)
        with open(os.path.join(directory, "main.gg"), "w",
                  encoding="utf-8") as handle:
            handle.write(f"// {name} {version}\n")
        if extra_file:
            with open(os.path.join(directory, extra_file), "w",
                      encoding="utf-8") as handle:
                handle.write("// extra\n")
        return directory

    def registry(self):
        return gpm.Registry(roots=[self.registry_root])

    def root(self, name, version, dependencies):
        return gpm.PackageManifest(
            name=name, version=Version.parse(version),
            dependencies={k: Constraint.parse(v)
                          for k, v in dependencies.items()})


class Resolution(PackageTree):

    def test_the_highest_satisfying_version_wins(self):
        self.make("codec", "1.0.0")
        self.make("codec", "1.1.0")
        self.make("codec", "2.0.0")
        result = gpm.resolve(self.root("app", "0.1.0", {"codec": "^1.0.0"}),
                             self.registry())
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.resolved["codec"].version,
                         Version.parse("1.1.0"))

    def test_a_transitive_dependency_is_resolved(self):
        self.make("codec", "1.1.0")
        self.make("terminology", "1.3.0", {"codec": "^1.0.0"})
        result = gpm.resolve(self.root("app", "0.1.0",
                                       {"terminology": "^1.3.0"}),
                             self.registry())
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(set(result.resolved), {"terminology", "codec"})

    def test_a_later_constraint_narrows_an_earlier_choice(self):
        # Regression.  A one-pass resolver picks the highest version when it
        # first meets a package and never revisits the choice, so when a
        # *later* dependency forbids it the resolver reports success while
        # installing a version a constraint excludes.  `aa` sorts before `b`,
        # which is what made the first implementation look correct on the
        # example that happened to sort the other way.
        self.make("a", "1.0.0", {"aa": ">=1.0.0"})
        self.make("b", "1.0.0", {"aa": "<1.5.0"})
        self.make("aa", "1.0.0")
        self.make("aa", "1.9.0")
        result = gpm.resolve(self.root("app", "0.1.0",
                                       {"a": "^1.0.0", "b": "^1.0.0"}),
                             self.registry())
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.resolved["aa"].version, Version.parse("1.0.0"),
                         "`b` requires aa <1.5.0, so 1.9.0 is forbidden")

    def test_the_same_case_with_favourable_alphabetical_order(self):
        # The same graph with names that sort the other way, so both orders are
        # covered and the answer cannot depend on which.
        self.make("a", "1.0.0", {"c": ">=1.0.0"})
        self.make("b", "1.0.0", {"c": "<1.5.0"})
        self.make("c", "1.0.0")
        self.make("c", "1.9.0")
        result = gpm.resolve(self.root("app", "0.1.0",
                                       {"a": "^1.0.0", "b": "^1.0.0"}),
                             self.registry())
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.resolved["c"].version, Version.parse("1.0.0"))

    def test_an_unsatisfiable_conflict_is_reported_not_guessed(self):
        self.make("a", "1.0.0", {"c": ">=2.0.0"})
        self.make("b", "1.0.0", {"c": "<1.0.0"})
        self.make("c", "1.0.0")
        self.make("c", "2.0.0")
        result = gpm.resolve(self.root("app", "0.1.0",
                                       {"a": "^1.0.0", "b": "^1.0.0"}),
                             self.registry())
        self.assertFalse(result.ok)
        self.assertTrue(any("c" in problem for problem in result.problems))
        joined = " ".join(result.problems)
        self.assertIn(">=2.0.0", joined)
        self.assertIn("<1.0.0", joined)
        self.assertEqual(result.resolved, {},
                         "a failed resolution must not half-apply")

    def test_a_missing_package_is_named(self):
        result = gpm.resolve(self.root("app", "0.1.0", {"absent": "^1.0.0"}),
                             self.registry())
        self.assertFalse(result.ok)
        self.assertIn("absent", " ".join(result.problems))

    def test_resolution_is_deterministic(self):
        self.make("codec", "1.0.0")
        self.make("codec", "1.1.0")
        self.make("terminology", "1.0.0", {"codec": "^1.0.0"})
        self.make("medical.fhir", "5.0.0", {"terminology": "^1.0.0"})
        manifest = self.root("app", "0.1.0", {"medical.fhir": "^5.0.0"})
        first = gpm.resolve(manifest, self.registry())
        second = gpm.resolve(manifest, self.registry())
        self.assertEqual([e.to_dict() for e in first.ordered()],
                         [e.to_dict() for e in second.ordered()])

    def test_the_lock_file_orders_dependencies_first(self):
        self.make("codec", "1.0.0")
        self.make("terminology", "1.0.0", {"codec": "^1.0.0"})
        result = gpm.resolve(self.root("app", "0.1.0",
                                       {"terminology": "^1.0.0"}),
                             self.registry())
        order = [entry.name for entry in result.ordered()]
        self.assertLess(order.index("codec"), order.index("terminology"))

    def test_a_cyclic_dependency_does_not_hang(self):
        self.make("a", "1.0.0", {"b": "^1.0.0"})
        self.make("b", "1.0.0", {"a": "^1.0.0"})
        result = gpm.resolve(self.root("app", "0.1.0", {"a": "^1.0.0"}),
                             self.registry())
        self.assertEqual(set(result.resolved), {"a", "b"})
        self.assertEqual(len(result.ordered()), 2)

    def test_a_directory_whose_manifest_disagrees_is_not_a_candidate(self):
        # A directory named `foo` holding a package called `bar` is a hijack
        # attempt or a mistake; either way it must not be installed as `foo`.
        directory = os.path.join(self.registry_root, "foo", "1.0.0")
        os.makedirs(directory)
        with open(os.path.join(directory, gpm.MANIFEST_NAME), "w",
                  encoding="utf-8") as handle:
            json.dump({"name": "bar", "version": "1.0.0"}, handle)
        result = gpm.resolve(self.root("app", "0.1.0", {"foo": "^1.0.0"}),
                             self.registry())
        self.assertFalse(result.ok)


class Integrity(PackageTree):

    def test_a_lock_file_verifies_against_unchanged_packages(self):
        self.make("codec", "1.0.0")
        result = gpm.resolve(self.root("app", "0.1.0", {"codec": "^1.0.0"}),
                             self.registry())
        lock = gpm.Lock.from_resolution(result)
        verification = gpm.verify_lock(lock, self.registry())
        self.assertTrue(verification.ok, verification.problems)

    def test_a_changed_package_fails_verification(self):
        self.make("codec", "1.0.0")
        result = gpm.resolve(self.root("app", "0.1.0", {"codec": "^1.0.0"}),
                             self.registry())
        lock = gpm.Lock.from_resolution(result)
        with open(os.path.join(self.registry_root, "codec", "1.0.0",
                               "main.gg"), "a", encoding="utf-8") as handle:
            handle.write("// tampered\n")
        verification = gpm.verify_lock(lock, self.registry())
        self.assertFalse(verification.ok)
        self.assertIn("contents", " ".join(verification.problems))

    def test_renaming_a_file_changes_the_packages_identity(self):
        directory = self.make("codec", "1.0.0", extra_file="a.gg")
        before = gpm.content_hash(directory)
        os.rename(os.path.join(directory, "a.gg"),
                  os.path.join(directory, "b.gg"))
        self.assertNotEqual(before, gpm.content_hash(directory),
                            "two packages that differ only in a filename are "
                            "different packages")

    def test_an_unsigned_package_is_reported_as_unsigned(self):
        self.make("codec", "1.0.0")
        result = gpm.resolve(self.root("app", "0.1.0", {"codec": "^1.0.0"}),
                             self.registry())
        verification = gpm.verify_lock(gpm.Lock.from_resolution(result),
                                       self.registry())
        self.assertEqual(verification.unsigned, 1)
        self.assertEqual(verification.verified_signatures, 0)
        self.assertIn("no signature", " ".join(verification.render()))

    def test_a_signed_package_verifies_and_a_tampered_one_does_not(self):
        directory = self.make("codec", "1.0.0")
        secret, public = ed25519.keypair_from_seed(bytes(range(32)))
        manifest = gpm.PackageManifest.read(directory)
        manifest.sign(secret, directory)
        with open(os.path.join(directory, gpm.MANIFEST_NAME), "w",
                  encoding="utf-8") as handle:
            handle.write(manifest.to_json())

        result = gpm.resolve(self.root("app", "0.1.0", {"codec": "^1.0.0"}),
                             self.registry())
        lock = gpm.Lock.from_resolution(result)
        verification = gpm.verify_lock(lock, self.registry())
        self.assertEqual(verification.verified_signatures, 1, verification.problems)

        # Change the contents and the signature no longer covers them.
        with open(os.path.join(directory, "main.gg"), "w",
                  encoding="utf-8") as handle:
            handle.write("// replaced\n")
        again = gpm.verify_lock(lock, self.registry())
        self.assertFalse(again.ok)

    def test_a_lock_file_from_a_newer_version_is_refused(self):
        with self.assertRaises(gpm.PackageError):
            gpm.Lock.parse(json.dumps({"lock_version": 99, "packages": []}))

    def test_a_malformed_lock_entry_is_refused(self):
        with self.assertRaises(gpm.PackageError):
            gpm.Lock.parse(json.dumps({"lock_version": 1,
                                       "packages": [{"name": "x"}]}))


class Advisories(PackageTree):

    def test_an_affected_version_is_reported(self):
        self.make("codec", "1.0.0")
        result = gpm.resolve(self.root("app", "0.1.0", {"codec": "^1.0.0"}),
                             self.registry())
        advisory = gpm.Advisory(identifier="GG-2026-0001", package="codec",
                                affected=Constraint.parse("<1.1.0"),
                                severity="high", summary="decoder overflow")
        report = gpm.audit(gpm.Lock.from_resolution(result), [advisory])
        self.assertEqual(len(report.hits), 1)
        self.assertIn("GG-2026-0001", " ".join(report.render()))

    def test_an_absent_advisory_database_is_not_a_clean_bill_of_health(self):
        self.make("codec", "1.0.0")
        result = gpm.resolve(self.root("app", "0.1.0", {"codec": "^1.0.0"}),
                             self.registry())
        report = gpm.audit(gpm.Lock.from_resolution(result), [])
        rendered = " ".join(report.render())
        self.assertIn("nothing was checked", rendered)
        self.assertNotIn("no vulnerabilities", rendered.lower())


# ---------------------------------------------------------------------------
# Reproducible signed builds
# ---------------------------------------------------------------------------

class BuildManifests(unittest.TestCase):

    def _program(self, path="hello.gg"):
        source = S.example_file(path)
        compilation = S.compile_only(source, path=path).compilation
        return source, compilation

    def test_the_same_source_produces_the_same_manifest(self):
        source, compilation = self._program()

        def build():
            return buildinfo.build_manifest(
                source_path="examples/hello.gg", source_text=source,
                program=compilation.program)

        report = buildinfo.reproducibility_report("examples/hello.gg", source,
                                                  build)
        self.assertTrue(report.reproducible, report.render())
        self.assertEqual(report.first_digest, report.second_digest)

    def test_a_different_program_produces_a_different_manifest(self):
        one_source, one_compilation = self._program("hello.gg")
        two_source, two_compilation = self._program("core/dose.gg")
        one = buildinfo.build_manifest(source_path="a.gg", source_text=one_source,
                                       program=one_compilation.program)
        two = buildinfo.build_manifest(source_path="b.gg", source_text=two_source,
                                       program=two_compilation.program)
        self.assertNotEqual(buildinfo.manifest_digest(one),
                            buildinfo.manifest_digest(two))

    def test_the_manifest_records_the_compiler_as_well_as_the_program(self):
        source, compilation = self._program()
        manifest = buildinfo.build_manifest(
            source_path="examples/hello.gg", source_text=source,
            program=compilation.program)
        self.assertIn("compiler", manifest)
        self.assertTrue(manifest["compiler"]["compiler_source_sha256"])
        self.assertEqual(manifest["compiler"]["gir_version"], "1.0")

    def test_canonical_json_is_order_independent(self):
        self.assertEqual(buildinfo.canonical_json({"b": 1, "a": 2}),
                         buildinfo.canonical_json({"a": 2, "b": 1}))

    def test_signing_and_verifying_a_manifest(self):
        source, compilation = self._program()
        manifest = buildinfo.build_manifest(
            source_path="examples/hello.gg", source_text=source,
            program=compilation.program)
        secret, public = ed25519.keypair_from_seed(bytes(range(32)))
        signed = buildinfo.sign_manifest(manifest, secret)
        self.assertTrue(signed.signed)
        self.assertEqual(buildinfo.verify_manifest(signed), [])

    def test_an_edited_manifest_fails_verification(self):
        source, compilation = self._program()
        manifest = buildinfo.build_manifest(
            source_path="examples/hello.gg", source_text=source,
            program=compilation.program)
        secret, _public = ed25519.keypair_from_seed(bytes(range(32)))
        signed = buildinfo.sign_manifest(manifest, secret)
        signed.manifest["build"]["opt_level"] = 3
        problems = buildinfo.verify_manifest(signed)
        self.assertTrue(problems)
        self.assertIn("edited after signing", " ".join(problems))

    def test_a_manifest_signed_by_another_key_fails_verification(self):
        source, compilation = self._program()
        manifest = buildinfo.build_manifest(
            source_path="examples/hello.gg", source_text=source,
            program=compilation.program)
        secret, _public = ed25519.keypair_from_seed(bytes(range(32)))
        other_secret, other_public = ed25519.keypair_from_seed(bytes(range(1, 33)))
        signed = buildinfo.sign_manifest(manifest, secret)
        problems = buildinfo.verify_manifest(signed, public_key=other_public)
        self.assertTrue(problems)
        self.assertIn("does not verify", " ".join(problems))

    def test_an_unsigned_manifest_says_so(self):
        source, compilation = self._program()
        manifest = buildinfo.build_manifest(
            source_path="examples/hello.gg", source_text=source,
            program=compilation.program)
        signed = buildinfo.sign_manifest(manifest, None)
        self.assertFalse(signed.signed)
        self.assertIn("not signed", " ".join(buildinfo.verify_manifest(signed)))

    def test_a_manifest_round_trips_through_json(self):
        source, compilation = self._program()
        manifest = buildinfo.build_manifest(
            source_path="examples/hello.gg", source_text=source,
            program=compilation.program)
        secret, _public = ed25519.keypair_from_seed(bytes(range(32)))
        signed = buildinfo.sign_manifest(manifest, secret)
        restored = buildinfo.SignedManifest.from_json(signed.to_json())
        self.assertEqual(buildinfo.verify_manifest(restored), [])
        self.assertEqual(restored.digest, signed.digest)

    def test_the_signing_key_is_not_world_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "key")
            buildinfo.generate_key(path)
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode, 0o600,
                             "a world-readable signing key is not a key")
            self.assertEqual(len(buildinfo.load_key(path)), 32)


if __name__ == "__main__":
    unittest.main()
