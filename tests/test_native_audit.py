"""The audit chain in the native runtime (spec section 13).

Two implementations of a tamper-evident log either agree on every byte or they
are two logs.  The digest of a record is taken over canonical JSON text, so key
order, separators, string escaping, the spelling of a float and the shape of a
null are all part of the format, not details of one implementation.  These tests
therefore do not check that the native trail "looks reasonable": they check that
it *is* the interpreter's trail, byte for byte, for the same program, and that
the reference verifier accepts what the compiled binary wrote.

The C primitives are checked against Python's own `hashlib`, which is the only
independent oracle available here.  That establishes the implementation matches
the published algorithms on the FIPS 180-4 vectors and on block boundaries; it
is not a security audit, and nothing here claims one (see
`docs/IMPLEMENTATION.md`, "the native runtime's crypto is in-house and
unaudited").
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import support as S
from gamag import __version__
from gamag.backend import cgen, native
from gamag.driver import execute, find_entry
from gamag.runtime.audit import AuditLog, verify_trail
from gamag.runtime.context import Context

HAS_CC = native.find_c_compiler() is not None
RT_DIR = os.path.join(os.path.dirname(cgen.__file__), "rt")

HEAD = ("gama core 0.2\n\nintent Chained\n"
        "    purpose  show what a trail looks like across two operations\n"
        "    authority  AuditWrite\n"
        "    trail      the chain of this intent\n\n"
        "source a : I64 from 21\n\n")

TWO_TRAILS = HEAD + """\
operation Doubled
    uses     a
    yields   d : I64
    effect   pure
    computes a * 2
    trail    doubled the input

operation Added
    uses     d
    yields   total : I64
    effect   pure
    computes d + 4
    trail    added four

outcome total
"""

ONE_TRAIL = HEAD + """\
operation Doubled
    uses     a
    yields   d : I64
    effect   pure
    computes a * 2
    trail    doubled the input

outcome d
"""

KEY_HEX = "00112233445566778899aabbccddeeff"
KEY = bytes.fromhex(KEY_HEX)


def interpreter_trail(source: str, key: bytes = KEY) -> str:
    """The trail `ggc run --audit` writes, assembled the way the CLI does.

    The point of doing it here rather than shelling out is that the two runs can
    then be compared with the same clock and the same key: what is under test is
    the C implementation, not the wall clock.
    """
    outcome = S.compile_only(source, profile="standard")
    assert outcome.compiled, outcome.messages()
    compilation = outcome.compilation
    context = Context(grants=set(S.declared_grants(compilation)),
                      stdout=io.StringIO(), seed=0, deterministic=True,
                      program_version=__version__, audit_key=key, epoch=0.0)
    execute(compilation, entry=find_entry(compilation), context=context,
            grants=())
    return context.audit.to_jsonl()


def native_trail(source: str, build_dir: str, audit_path: str,
                 key_hex: str = KEY_HEX, deterministic: bool = True,
                 expect_status: int = 0) -> str:
    """Compile `source` and run the binary, writing its trail to `audit_path`."""
    outcome = S.compile_only(source, profile="standard")
    assert outcome.compiled, outcome.messages()
    program = outcome.compilation.program
    support = cgen.unsupported(program)
    assert support.ok, "the backend still refuses the audit chain: " + \
        "\n".join(p.render() for p in support.problems)
    result = native.build(program, "t.gg", build_dir=build_dir)
    assert result.ok, "\n".join(result.render())
    env = {"GAMAG_AUDIT_KEY": key_hex} if key_hex else {}
    args = ["--audit", audit_path]
    if not deterministic:
        # The default is the virtual clock; the wall clock has to be asked for,
        # on both machines, and `ggc run` calls that `--lenient-runtime`.
        env["GAMAG_AUDIT_REALTIME"] = "1"
    run = native.run(result.exe_path, args, env=env)
    assert run.returncode == expect_status, (
        f"expected exit {expect_status}, got {run.returncode}: {run.stderr}")
    with open(audit_path, "r", encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------

HARNESS = r'''
#include "gamag_rt.h"
#include <stdio.h>
#include <string.h>
GValue g_main(int a, char **v) { (void)a; (void)v; return g_unit(); }
/* stdin: "k <hex>" sets the key, "s <text>" hashes it, "m <text>" signs it. */
int main(void)
{
    unsigned char key[256];
    size_t keylen = 0;
    char line[4096], out[65];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = strlen(line);
        if (n && line[n - 1] == '\n') line[--n] = '\0';
        if (n < 2) { printf("?\n"); continue; }
        char mode = line[0];
        char *text = line + 2;
        if (mode == 'k') {
            keylen = strlen(text) / 2;
            if (keylen > sizeof key) keylen = sizeof key;
            for (size_t i = 0; i < keylen; i++) {
                int hi = text[i * 2], lo = text[i * 2 + 1];
                hi = (hi <= '9') ? hi - '0' : (hi | 32) - 'a' + 10;
                lo = (lo <= '9') ? lo - '0' : (lo | 32) - 'a' + 10;
                key[i] = (unsigned char)((hi << 4) | lo);
            }
            printf("k ok %zu\n", keylen);
        } else if (mode == 's') {
            g_sha256_hex(text, out);
            printf("s %s\n", out);
        } else if (mode == 'm') {
            g_hmac_sha256_hex(key, keylen, text, strlen(text), out);
            printf("m %s\n", out);
        }
    }
    return 0;
}
'''


@unittest.skipUnless(HAS_CC, "no C compiler on this machine")
class AuditPrimitives(unittest.TestCase):
    """SHA-256 and HMAC-SHA-256, measured against `hashlib`.

    The lengths are not arbitrary: 55, 56 and 63 straddle the padding rule, 64
    is exactly one block, 119 and 127 straddle the two-block key rule, and the
    quoted vector is FIPS 180-4's own multi-block example.
    """

    CASES = ["", "abc",
             "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
             "a" * 55, "a" * 56, "a" * 63, "a" * 64, "a" * 65, "a" * 119,
             "a" * 127, "a" * 128,
             "naïve ✓ — a trail records text, not what it means",
             'quotes " and backslash \\ and \ttab',
             '{"action":"Doubled","seq":0}']

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        harness = os.path.join(cls._tmp.name, "harness.c")
        with open(harness, "w", encoding="utf-8") as handle:
            handle.write(HARNESS)
        exe = os.path.join(cls._tmp.name, "harness")
        proc = subprocess.run(
            [native.find_c_compiler(), "-std=c11", "-O1", "-Wall", "-Wextra",
             f"-I{RT_DIR}", harness, os.path.join(RT_DIR, "gamag_rt.c"),
             "-o", exe, "-lm"],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise AssertionError("the harness does not build: " + proc.stderr)
        cls.exe = exe

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _drive(self, lines):
        proc = subprocess.run([self.exe], input="\n".join(lines) + "\n",
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip().split("\n")

    def test_sha256_matches_hashlib(self):
        lines, want = [], []
        for case in self.CASES:
            lines.append("s " + case)
            want.append("s " + hashlib.sha256(case.encode("utf-8")).hexdigest())
        got = self._drive(lines)
        self.assertEqual(len(got), len(want))
        bad = [(c, w, g) for c, w, g in zip(self.CASES, want, got) if w != g]
        self.assertEqual(bad, [], "the C SHA-256 disagrees with hashlib: "
                         + "; ".join(f"{c[:20]!r}" for c, _, _ in bad[:3]))

    def test_hmac_sha256_matches_python(self):
        # A short key, a key of exactly the block size, and one longer than it:
        # the last case is the one an implementation gets wrong by forgetting to
        # hash the key first.
        keys = [bytes(range(16)), b"k" * 64, b"k" * 200]
        lines, want = [], []
        for key in keys:
            lines.append("k " + key.hex())
            want.append(f"k ok {len(key)}")
            for case in self.CASES[:5]:
                lines.append("m " + case)
                want.append("m " + hmac.new(
                    key, case.encode("utf-8"), hashlib.sha256).hexdigest())
        got = self._drive(lines)
        bad = [(w, g) for w, g in zip(want, got) if w != g]
        self.assertEqual(bad, [], f"{len(want) - len(bad)}/{len(want)} agree; "
                         f"first mismatch: {bad[:1]}")

    def test_the_fips_vector_is_what_the_standard_says(self):
        # The digest of the 56-character "abcdbcde..." string, from FIPS 180-4's
        # example 2, spelled out so that a change to either implementation has
        # to be a deliberate one.
        want = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        self.assertEqual(self._drive(["s " + self.CASES[2]]), ["s " + want])


# ---------------------------------------------------------------------------
# the chain, end to end
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_CC, "no C compiler on this machine")
class NativeTrail(unittest.TestCase):
    """A compiled program's trail is the interpreter's, and verifies offline."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.build_dir = os.path.join(self._tmp.name, "build")
        self.audit = os.path.join(self._tmp.name, "trail.jsonl")

    def test_the_two_trails_are_the_same_bytes(self):
        """The claim this whole file exists to keep.

        Same program, same clock (deterministic), same key: the file the native
        binary writes and the file the interpreter writes are identical.  Every
        byte of canonical JSON, the key order, the float spelling, the digest
        and the signature all have to agree for this to pass.
        """
        want = interpreter_trail(ONE_TRAIL)
        got = native_trail(ONE_TRAIL, self.build_dir, self.audit)
        self.assertEqual(got, want, "the native trail differs from the "
                         "interpreter's: the canonical JSON or the digest "
                         "has drifted")

    def test_two_records_chain(self):
        """`prev_hash` on the second record is the first record's digest."""
        want = interpreter_trail(TWO_TRAILS)
        got = native_trail(TWO_TRAILS, self.build_dir, self.audit)
        self.assertEqual(got, want)
        records = [json.loads(line) for line in got.splitlines() if line]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["prev_hash"], "0" * 64)
        self.assertEqual(records[1]["prev_hash"], records[0]["hash"])
        self.assertEqual([r["seq"] for r in records], [0, 1])
        self.assertNotEqual(records[0]["hash"], records[1]["hash"],
                            "two records with the same digest means the "
                            "payload or the chain is not being hashed")
        self.assertEqual(records[1]["fields"]["intent"], "Chained",
                         "what the program declared is what the trail holds")

    def test_the_native_trail_verifies_offline(self):
        got = native_trail(TWO_TRAILS, self.build_dir, self.audit)
        ok, problems, summary = verify_trail(got, key=KEY)
        self.assertTrue(ok, "\n".join(problems))
        self.assertEqual(summary["records"], 2)
        self.assertTrue(summary["signatures_checked"])
        self.assertEqual(summary["by_action"], {"Doubled": 1, "Added": 1})

    def test_a_tampered_native_record_is_caught(self):
        """Edit one byte of one field and the chain says so.

        This is the entire purpose of the structure: not to stop anyone, but to
        make a modification visible to someone who was not there.
        """
        got = native_trail(TWO_TRAILS, self.build_dir, self.audit)
        tampered = got.replace('"doubled the input"', '"doubled the output"', 1)
        self.assertNotEqual(tampered, got, "the test did not tamper with anything")
        ok, problems, _ = verify_trail(tampered, key=KEY)
        self.assertFalse(ok)
        self.assertEqual(len(problems), 1, f"{problems}")
        self.assertIn("record 0", problems[0])
        self.assertIn("modified", problems[0])
        self.assertIn("contents were modified", problems[0],
                      "the message should say what kind of break this is")

    def test_a_deleted_record_is_caught(self):
        """Drop a record and both guarantees fail at once.

        The digest of a record proves it was not edited; the `prev_hash` link
        and the sequence number are what prove nothing was *removed*.  A trail
        with one record missing is caught twice, by two different readers.
        """
        got = native_trail(TWO_TRAILS, self.build_dir, self.audit)
        second = [line for line in got.splitlines() if line][1]
        ok, problems, _ = verify_trail(second, key=KEY)
        self.assertFalse(ok)
        kinds = " ".join(problems)
        self.assertIn("seq is 1, expected 0", kinds, f"{problems}")
        self.assertIn("does not match the previous record hash", kinds,
                      f"{problems}")

    def test_an_unsigned_trail_is_not_a_signed_one(self):
        """No key in the environment, no signature -- and the verifier says so.

        The chain still proves what it always proves without a key (a record was
        not edited after it was written, and the order is intact), but it cannot
        prove *who* wrote it.  Keeping those two claims apart is the reason
        `ggc audit verify` prints whether signatures were checked.
        """
        got = native_trail(TWO_TRAILS, self.build_dir, self.audit, key_hex="")
        for line in got.splitlines():
            self.assertEqual(json.loads(line)["signature"], "")
        ok, problems, summary = verify_trail(got)
        self.assertTrue(ok, "\n".join(problems))
        self.assertFalse(summary["signatures_checked"],
                         "an unsigned trail must not be reported as signed")
        ok_signed, problems_signed, summary_signed = verify_trail(got, key=KEY)
        self.assertFalse(ok_signed,
                         "an unsigned trail must fail once a key is supplied")
        self.assertTrue(all("signature" in p for p in problems_signed),
                        f"{problems_signed}")
        self.assertTrue(summary_signed["signatures_checked"])

    def test_a_real_clock_still_produces_a_valid_chain(self):
        """Outside reproducible mode the native runtime reads the clock.

        It stamps whole seconds where the interpreter stamps fractions, which is
        a difference worth knowing about -- but a trail is a trail: the digests
        recompute and the links hold, and nothing here pretends the two
        timestamps are equal.
        """
        got = native_trail(ONE_TRAIL, self.build_dir, self.audit,
                           deterministic=False)
        ok, problems, _ = verify_trail(got, key=KEY)
        self.assertTrue(ok, "\n".join(problems))
        record = json.loads(got)
        self.assertAlmostEqual(record["timestamp"] % 1.0, 0.0, places=9,
                               msg="the native stamp is expected to be a whole "
                                   "second; a fractional one would mean this "
                                   "test has drifted")

    def test_the_audit_chain_is_no_longer_a_reason_to_refuse(self):
        """The gap this closes, stated as a test so it stays closed."""
        self.assertNotIn(cgen.Op.AUDIT, cgen.UNSUPPORTED_REASONS)
        program = S.compile_only(ONE_TRAIL, profile="standard").compilation.program
        self.assertTrue(cgen.unsupported(program).ok)
        emitted = native.generate(program, "t.gg", build_dir=self.build_dir)
        self.assertIn("g_audit_record(", emitted.c_source)
        self.assertIn("g_audit_set_context(", emitted.c_source,
                      "the trail has to name the build that wrote it, and the "
                      "interpreter takes that from the toolchain version")

    def test_the_examples_with_trails_agree_and_the_ones_without_are_both_empty(
            self):
        """Every shipped example the backend accepts, both writers.

        `selection.gg` and `dose.gg` are the two that arrived with `trail`
        clauses and are accepted natively, so they are compared as files.
        `hello.gg` has no trail at all: both sides must write nothing, because a
        native runtime that invented a record -- or a checker that quietly
        dropped one the interpreter wrote -- would otherwise pass unnoticed.
        """
        for name, expect_records in (("core/selection.gg", 1),
                                     ("core/dose.gg", 1),
                                     ("hello.gg", 0)):
            with self.subTest(example=name):
                source = S.example_file(name)
                want = interpreter_trail(source)
                got = native_trail(source, self.build_dir, self.audit)
                self.assertEqual(got, want,
                                 f"{name}: the native trail differs from the "
                                 f"interpreter's")
                self.assertEqual(len([line for line in got.splitlines() if line]),
                                 expect_records)

    def test_the_native_trail_records_only_what_the_program_declared(self):
        """The gap, stated instead of hidden.

        The reference runtime also records its own events -- a failed `require`
        appends a security-level record, as do capability denials, transactions
        and recovery actions.  The C runtime records what the program's `trail`
        clauses declare and nothing more, so a native trail of a *faulting*
        program is shorter than the interpreter's by exactly those events.  Both
        facts are useful to a reader; pretending otherwise would make the
        byte-identity claim above look stronger than it is.
        """
        failing = ONE_TRAIL.replace(
            "    computes a * 2\n", "    computes a * 2\n    holds    d > 1000\n")
        outcome = S.compile_only(failing, profile="standard")
        self.assertTrue(outcome.compiled, outcome.messages())
        interp = [line for line in interpreter_trail(failing).splitlines() if line]
        self.assertEqual(len(interp), 1, f"{interp}")
        self.assertIn("REQUIRE_FAILED", interp[0],
                      "the interpreter records the violation it raised")
        got = [line for line in native_trail(failing, self.build_dir,
                                             self.audit, expect_status=2)
                .splitlines() if line]
        self.assertEqual(got, [],
                         "the native binary faults on the same check, which the "
                         "lowering places *before* the `trail` clause, so it "
                         "records nothing at all -- and it must not invent a "
                         "record the program did not declare")
        self.assertLess(len(got), len(interp),
                        "this is the asymmetry: an interpreter trail describes "
                        "the runtime's own events, a native trail describes "
                        "only what the program said to record")

    def test_the_record_fields_the_c_side_sorts_are_the_python_ones(self):
        """The native writer hardcodes the key order; this is the other half.

        `sort_keys=True` is a rule, and a rule one implementation hardcodes can
        be broken by the other adding a field.  The digest comparison in
        `test_the_two_trails_are_the_same_bytes` catches that too, but it
        catches it as one failure with three possible causes; this names the
        cause.
        """
        log = AuditLog(actor="program", authority="gama-g/runtime",
                       program_version="0.1.0", policy_version="0",
                       deterministic=True)
        record = log.record("X", intent="I", record="r")
        payload = sorted(record.digest_payload().keys())
        self.assertEqual(payload, [
            "action", "actor", "authority", "event_id", "fields", "level",
            "object", "policy_version", "prev_hash", "program_version",
            "reason", "seq", "timestamp"])
        whole = sorted(record.to_dict().keys())
        self.assertEqual(whole, payload[:5] + ["hash"] + payload[5:12]
                         + ["signature"] + payload[12:],
                         "`signature` sorts after `seq`, not before it -- the C "
                         "writer emits it in this order")

    def test_the_native_binary_documents_the_option(self):
        """A user reading `--help` on the binary must find the audit flag."""
        outcome = S.compile_only(ONE_TRAIL, profile="standard")
        result = native.build(outcome.compilation.program, "t.gg",
                              build_dir=self.build_dir)
        run = native.run(result.exe_path, ["--help"])
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--audit", run.stdout)
        self.assertIn("--audit-realtime", run.stdout)
        self.assertIn("GAMAG_AUDIT_KEY", run.stdout)
