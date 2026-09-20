#!/usr/bin/env python3
"""Prove the fuzzer can fail.

A fuzzer that reports "no invariant was broken" is only evidence if it is
capable of reporting the opposite.  So this script deliberately breaks the
product -- one known bug at a time -- and checks that the oracle notices each
one.  An injection that is *not* caught means the corresponding check is
decoration, and that is a worse bug than anything the campaign might find.

Run it with:

    python3 tools/fuzz_selfcheck.py

Each injection is applied by monkeypatching in memory, so nothing on disk is
ever modified and no state survives the script.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "compiler"))

from gamag import diagnostics                     # noqa: E402
from gamag.driver import compile_source            # noqa: E402
from gamag.fuzz import oracle                      # noqa: E402
from gamag.gir import optimizer                    # noqa: E402
from gamag.runtime import context as rt_context    # noqa: E402

#: A tiny program that compiles and runs, so the deep checks have something to
#: work with.  Written in the language's real surface syntax: indentation-based,
#: no colons, `io` qualifier on a function that prints.
PROGRAM = '''fn main() -> Unit
    io
    var total = 0
    for i in 1..6
        total = total + i
    print("total", total)
'''

#: Inputs chosen to reach the front end's error paths rather than its happy one.
BROKEN = [
    "let x = ",
    "@",
    "gama core",
    "fn f( -> {}",
    "let n = 1e",
    "match x\n",
]


class Injection:
    """One deliberate bug, and the invariant that is supposed to catch it."""

    def __init__(self, name, invariant, apply, restore):
        self.name = name
        self.invariant = invariant
        self._apply = apply
        self._restore = restore

    def __enter__(self):
        self._apply()
        return self

    def __exit__(self, *exc):
        self._restore()
        return False


def _inject_missing_code():
    """The original finding: a diagnostic with no code cannot be filtered."""
    real = diagnostics.front_end_code
    diagnostics.front_end_code = lambda message, phase=None: None
    # The parsers captured the symbol at import time, so patch them too.
    from gamag import parser as v01_parser
    from gamag.core import parser as core_parser
    saved = (v01_parser.front_end_code, core_parser.front_end_code)
    v01_parser.front_end_code = lambda message, phase=None: None
    core_parser.front_end_code = lambda message, phase=None: None

    def restore():
        diagnostics.front_end_code = real
        v01_parser.front_end_code, core_parser.front_end_code = saved
    return restore


def _inject_optimizer_bug():
    """-O2 changes what the program computes.

    Integer addition is rewritten as subtraction at level 2.  That is crude,
    but it is exactly what a real constant-folding or strength-reduction bug
    looks like from the outside: the program still compiles, still runs, and
    prints the wrong number.

    Two details matter, and both were found the hard way:

    * `optimize` rewrites the program *in place* and returns a report about what
      it did, so the argument is the thing to corrupt -- corrupting the return
      value touches nothing.
    * Bumping every integer constant sounds equivalent but is not: in the test
      program the constants that exist are dead slots, so the output does not
      change and the injection silently proves nothing.  An injection that
      cannot fail the check is worse than no injection, which is why this one
      reports how much it changed.

    Written against the GIR as it actually is: `program.functions` is a dict
    keyed by name, a block holds `instrs`, and the operator of a BINOP lives in
    the instruction's `meta`.
    """
    from gamag.gir.ir import Op

    real_optimize = optimizer.optimize
    changed = []

    def buggy(program, level=1, *args, **kwargs):
        report = real_optimize(program, level, *args, **kwargs)
        if level < 2:
            return report
        functions = getattr(program, "functions", None) or {}
        iterable = functions.values() if isinstance(functions, dict) \
            else functions
        for fn in iterable:
            for block in getattr(fn, "blocks", []) or []:
                for instr in getattr(block, "instrs", []) or []:
                    if instr.op != Op.BINOP:
                        continue
                    if instr.meta.get("operator") == "+":
                        instr.meta["operator"] = "-"
                        changed.append(fn.name)
        return report

    optimizer.optimize = buggy
    # the driver imported the symbol directly, so it keeps the real one unless
    # this is patched too
    from gamag import driver
    saved_driver = driver.optimize
    driver.optimize = buggy

    def restore():
        optimizer.optimize = real_optimize
        driver.optimize = saved_driver
        if not changed:
            print("          (warning: the injection rewrote no instruction, "
                  "so it proved nothing)")
    return restore


def _inject_nondeterminism():
    """Two runs of the same program print different things.

    Injected at the point the runtime opens its output, which is where a real
    leak of wall-clock time or process state would show up.
    """
    real_init = rt_context.Context.__init__

    def flaky(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        stream = getattr(self, "stdout", None)
        if stream is not None:
            try:
                stream.write("<%d>" % random.randrange(1 << 30))
            except Exception:
                pass

    rt_context.Context.__init__ = flaky

    def restore():
        rt_context.Context.__init__ = real_init
    return restore


def _inject_crash():
    """The driver raises instead of returning a diagnostic."""
    real_compile = oracle.compile_source

    def explode(source, path="<fuzz>", *args, **kwargs):
        if "let" in source or "print" in source:
            raise ValueError("the compiler fell over")
        return real_compile(source, path, *args, **kwargs)

    oracle.compile_source = explode

    def restore():
        oracle.compile_source = real_compile
    return restore


def _inject_silent_rejection():
    """A program is rejected with no diagnostic, so the user has nothing."""
    real_compile = oracle.compile_source

    def silent(source, path="<fuzz>", *args, **kwargs):
        compilation = real_compile(source, path, *args, **kwargs)
        if compilation.ok:
            # force a failure and empty the bag
            compilation.bag.diagnostics.clear()
            compilation.stopped_at = None
            object.__setattr__(compilation, "program", None) \
                if hasattr(compilation, "__dict__") is False else None
            compilation.program = None
        return compilation

    oracle.compile_source = silent

    def restore():
        oracle.compile_source = real_compile
    return restore


INJECTIONS = [
    ("a diagnostic with no code", "diagnostic-has-a-code",
     ["broken"], _inject_missing_code),
    ("the optimizer changes behavior", "optimization-preserves-behavior",
     ["program"], _inject_optimizer_bug),
    ("two runs differ", "runs-are-deterministic",
     ["program"], _inject_nondeterminism),
    ("the compiler raises instead of explaining", "compiles-or-explains",
     ["program"], _inject_crash),
    ("a rejection with no diagnostic", "compiles-or-explains",
     ["program"], _inject_silent_rejection),
]


def _sources(which):
    out = []
    if "program" in which:
        out.append((PROGRAM, "<injected-program>"))
    if "broken" in which:
        out.extend((text, "<injected-broken>") for text in BROKEN)
    return out


def main() -> int:
    print("fuzzer self-check: does each invariant actually fire?\n")
    failures = []
    for name, invariant, which, make in INJECTIONS:
        restore = make()
        caught = []
        try:
            for source, origin in _sources(which):
                result = oracle.run_checks(source, path=origin, origin=origin,
                                           deep=True)
                for violation in result.violations:
                    if violation.invariant == invariant:
                        caught.append(violation)
        finally:
            restore()
        if caught:
            print(f"  caught  {name}")
            print(f"          -> {invariant}: {caught[0].detail[:88]}")
        else:
            print(f"  MISSED  {name}")
            print(f"          -> expected {invariant} and saw none")
            failures.append(name)

    # And the control: with nothing injected, the same inputs must be clean.
    clean = []
    for source, origin in _sources(["program", "broken"]):
        result = oracle.run_checks(source, path=origin, origin=origin, deep=True)
        for violation in result.violations:
            if violation.severity in ("crash", "wrong", "inconsistent"):
                clean.append((origin, violation))
    if clean:
        print(f"\n  MISSED  control: {len(clean)} violation(s) with nothing "
              f"injected")
        for origin, violation in clean[:5]:
            print(f"          [{origin}] {violation.invariant}: "
                  f"{violation.detail[:70]}")
        failures.append("control")
    else:
        print("\n  clean   control: nothing injected, nothing reported")

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) are decoration: "
              f"{', '.join(failures)}")
        return 1
    print(f"OK: all {len(INJECTIONS)} injected bugs were caught, and the "
          f"control was clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
