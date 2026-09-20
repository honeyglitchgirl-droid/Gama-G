# Where this implementation is not production grade

An assessment of the v1.0 toolchain, written by probing it rather than by
reading it.  Spec section 43 forbids claiming capabilities the toolchain does
not have, and that applies to this document too: everything below is either a
reproduction command or a count.

The short version: **the specification is not fully built, and the toolchain is
not production grade.**  Nothing in the repository claims otherwise -- the
README says "working vertical slice", `pyproject.toml` says
`Development Status :: 3 - Alpha`, and the phrase "production grade" appears
nowhere outside this file.  What follows is where it lacks.

---

## 1. What is built

All eleven remaining audit priorities (P3 - P16) plus the version merge: one
language, one pipeline, four ways to run it, 548 tests.  The native CPU backend
emits C and compiles it; the WebAssembly encoder emits a module; the
accelerator layer detects devices and refuses rather than falling back
silently; the fuzzer finds real bugs; there is a package manager, a signing
tool, an FFI, three interoperability modules and five enterprise modules.

That is a vertical slice of a specification whose own scope is Phases 0 - 7.
The gap between the two is the subject of this document.

---

## 2. Defects found by auditing, and their status

Three defects were found by attacking the toolchain's own stated contract --
that no input produces a Python traceback, and that a program which recurses
too far gets a fault rather than a compiler bug.  All three are **fixed**, and
each fix has a permanent regression test in `tests/test_robustness.py`.

### 2.1 Call depth used to be reported as a compiler bug (fixed)

The interpreter had a call-depth guard set to 1500 frames.  The host Python
stack is 1000 frames and each Gama-G call costs about four of them, so the
guard could never fire: users hit the host limit first and were told
`internal error: RecursionError`.  An internal error means "the compiler is
broken"; running out of stack means "your program recursed too far".  The
second was reported as the first, and the language's documented depth guarantee
was unreachable code.

Measured before the fix: recursion 200 worked, 250 faulted as an internal
error.  After it, the limit is derived from the host stack (129 frames on the
default CPython configuration), the guard fires first, and the result is a
`StackOverflow` runtime fault naming its limit, with a hint, exit status 2.

```
ggc run deep_but_legitimate.gg     # 126 levels of recursion: runs
ggc run runaway_recursion.gg       # past the limit: runtime fault [StackOverflow]
```

The honest reading of this fix: a realistic recursive algorithm is limited to
about 129 frames on a default host.  Raising `sys.setrecursionlimit` raises the
limit proportionally, and the remaining real fix -- an interpreter with an
explicit call stack rather than a recursive one -- is future work.  The limit
is documented rather than silent, which is the part that matters.

### 2.2 A file that was not UTF-8 crashed the compiler (fixed)

`ggc check` on any file that is not valid UTF-8 produced a raw
`UnicodeDecodeError` traceback: exit 1, stack trace, no diagnostic.
Reproduced with `/dev/urandom`, with a single invalid byte inside otherwise
valid source, and through `ggc build` as well as `ggc check`.

The file path now goes through one function that turns a decode failure into a
diagnostic naming the byte and its offset, with the `E-source-unreadable` code.
The same function covers a missing file and a directory passed where a file was
expected, which previously raised `FileNotFoundError` and `IsADirectoryError`.

### 2.3 Deeply nested input crashed the compiler (fixed)

Three routes reached a raw `RecursionError` in a recursive-descent parser or a
recursive tree walker:

| input | why |
|---|---|
| `((((...))))` 500 deep | parser recursion, ~16 frames per level |
| `1+1+1+...` 5000 terms | flat source, but a 5000-deep tree; the parser never recurses, the checker does |
| `[[[[...]]]]`, `f(f(f(...)))` | parser recursion through containers and call arguments |

Both bounds are now derived from the host stack rather than hardcoded: the
parsers bound syntactic nesting at every point where they descend into a
nested construct, and the completed tree is measured -- iteratively -- before
the recursive phases walk it.  Exceeding either produces
`E-nesting-too-deep` naming the limit, not a traceback.  `compile_source` also
keeps a `RecursionError` backstop so that an input nobody anticipated still
arrives as a diagnostic.

Measured limits on a default host: about 50 levels of syntactic nesting, and
about 256 for tree depth.  Flat chains up to 240 terms compile and run; 250 and
above are refused with a reason.  Real code is nowhere near either bound -- all
sixteen shipped examples compile -- but a generated program can be, and it now
gets an explanation instead of a stack trace.

### 2.4 The fuzzer could not have found two of the three

Worth recording because it is a defect in the *verification*, not the code.
Every fuzz invariant fed `compile_source` a Python string, which has already
decoded; the file path where 2.2 lived was never exercised.  A new invariant,
`file-path-is-total`, writes bytes to disk and calls `compile_file` -- the same
function the command line uses.  It is tested for the ability to fail: with the
defect reintroduced it reports four violations, and with the fix in place it is
silent.

### 2.5 A packaging defect, found while adding CI

The native backend emits C and compiles it with `gamag/backend/rt/gamag_rt.c`.
That file is not a Python module, so setuptools left it out of the wheel and
`ggc native` would have failed on every installed copy while working perfectly
from a checkout.  Declared as package data, and the CI packaging job now runs
`ggc native` after `pip install .`, so the two cannot drift apart again.

---

## 3. Scope limits that block production use

These are not bugs.  They are the reasons the toolchain is a slice, and each
one is already stated in `docs/DESIGN_v1_0.md` section 5.

- **Native coverage is 4 of 16 examples.**  Programs using the audit chain,
  capabilities, transactions, checkpoints, recovery regions, tensors, autodiff,
  agents, method dispatch or indirect calls are refused by name -- never
  miscompiled -- and run on the interpreter instead.
- **The native runtime does not enforce capabilities.**  There is no capability
  logic in `gamag_rt.c` at all; those programs are refused, so nothing is
  miscompiled, but there is no second line of defence either.  The interpreter
  is where the security model lives.
- **Memory is an arena that never frees.**  `ggc memory` reports the
  deterministic-destruction extents the memory model derives, but the native
  runtime does not act on them.  A long-running service leaks.
- **Ed25519 is not constant-time.**  Documented, and disqualifying for any use
  against an attacker who can measure timing.  Use an audited library.
- **No WebAssembly module has ever been executed** and **no accelerator kernel
  has ever run**.  Both are structurally verified only.  `ggc wasm` says so.
- **The package manager has no network and its resolver does not backtrack**, so
  a satisfiable graph can be reported as a conflict rather than guessed at.
- **`http`, `process`, `accelerator` and `identity_provider` are roadmap**, and
  `ggc format`, `ggc profile` and `ggc doc` do not exist.
- **Interop covers FHIR, terminology, provenance and consent**, not the wider
  specification surface.

---

## 4. Engineering process

- **CI now exists** (`.github/workflows/ci.yml`) and runs on every push and pull
  request across Python 3.9 - 3.13, plus a job that installs the package and
  runs the native backend from the installed copy.  Before this, the suite ran
  only when a person ran it, which is how a defect like 2.2 shipped.
- **There is still one implementation.**  Every test compares the toolchain
  against itself, except the RFC 8032 vectors and the differential backend test.
  An independent conformance suite exercising the specification directly does
  not exist.
- **Fuzzing is hours, not sustained.**  Campaigns found real bugs; a campaign
  that ran for a weekend would find more, and "no violations in 60 rounds" is
  not "no violations".
- **No performance baseline.**  The interpreter is a tree-walking evaluator in
  CPython and no measurement against any other implementation is claimed.
- **Sixteen examples, no releases, no versioned artifacts.**  v1.0.1 is the
  first tagged state, and package metadata still says Alpha.

---

## 5. What would have to change

In rough order of return on effort:

1. An interpreter with an explicit call stack, removing the ~129-frame recursion
   limit for real recursive algorithms.
2. Capability enforcement in the native runtime, so the security model does not
   depend on refusal alone.
3. A free/release path in the arena, so the memory model's extents mean
   something at run time.
4. An independent conformance test suite that reads the specification rather
   than the implementation.
5. Constant-time crypto, or removal of crypto from the shipped surface.
6. A benchmark baseline, published with its conditions.
7. Releases, and the metadata to match.

None of that is a weekend.  Naming it is the point: the toolchain is honest
about its edges, and this file is how that honesty is kept checkable.

---

## 6. How to check any of this yourself

```
python -m unittest discover -s tests -t tests -q      # 548 tests
python tools/fuzz_selfcheck.py                        # the checks can fail
./tools/bin/ggc difftest examples/*.gg examples/core/*.gg

# the three defects, as they were reported
printf '\x9e\xfe' > /tmp/bad.gg && ./tools/bin/ggc check /tmp/bad.gg
python3 -c "open('/tmp/pp.gg','w').write('fn main() -> Unit\n    let x = ' + '('*500 + '1' + ')'*500 + '\n    print(x)\n')" && ./tools/bin/ggc check /tmp/pp.gg
printf 'fn down(n: I64) -> I64\n    if n <= 0\n        return 0\n    return down(n - 1) + 1\n\nfn main() -> Unit\n    print(down(5000))\n' > /tmp/deep.gg && ./tools/bin/ggc run /tmp/deep.gg
```

Each of the three should print a diagnostic and exit without a Python
traceback.
