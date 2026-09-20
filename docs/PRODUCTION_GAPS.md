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
language, one pipeline, four ways to run it, 644 tests and a conformance suite
that checks the toolchain against the specification document rather than
against itself.  The native CPU backend
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

The honest reading of that fix, at the time: it made the guard fire, but it
also made the *effective* limit 129 frames -- a number that moves when
`sys.setrecursionlimit` moves, and that no Gama-G program can be written
against.

**That is now repaired.**  The interpreter keeps its activations on an explicit
stack (`run_from` suspends at a call, the trampoline pushes the activation), so
a Gama-G frame costs no host frame.  Measured: 1200-deep recursion runs;
`VM.MAX_DEPTH` (1500) is the limit that actually applies and the fault names
it; and lowering `sys.setrecursionlimit` to 300 changes none of that.  The cost
is about 13% on a call-bound benchmark (40 x `fib(16)`, 127,722 calls: 1.64 s
before, 1.86 s after, median of five runs on this host).

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
- **Memory is released at function return, not per extent.**  The arena used
  to never free; it now marks and releases around every function that provably
  cannot hand an allocated value on (see `docs/DESIGN_v1_0.md` section 5).
  A loop through such a function is bounded -- 368 bytes peak at 2 000 and at
  200 000 iterations -- while one that stores a global, calls another function,
  mutates a container or returns text still grows without bound.  `ggc native`
  prints the coverage, so how much of a program is covered is visible rather
  than assumed.  The memory model's per-level extents are still not what the
  runtime acts on.
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
- **The conformance suite exists now** (`conformance/`, `ggc conform`), and it
  reads `Gama-G_v1.0_Production_Specification.txt` at run time: section 5's
  type lists, section 22's fourteen stages, section 33's twenty requirements,
  and programs the prose implies.  It is a real improvement on "every test
  compares the toolchain with itself" -- but it is not independent of the
  author.  The same person wrote the compiler and chose which claims to make,
  and no third party has audited either.  What it does buy is that a
  disagreement with the document is now a *finding*: an unrecorded deviation
  fails, an invented stage fails, a requirement that vanishes from the
  checklist fails.  Section 33 item 10 (independent security review) is still
  answered `not-claimed`, and `ggc conform --strict` lists it.
- **Fuzzing is hours, not sustained.**  Campaigns found real bugs; a campaign
  that ran for a weekend would find more, and "no violations in 60 rounds" is
  not "no violations".
- **No published benchmark baseline.**  There are measurements, stated with
  their conditions, in `docs/COMPARISON.md` (the scalar specialiser against the
  same `gcc`, the interpreter against CPython, Node and Perl) and
  `docs/PRODUCTION_GAPS.md` section 2.1.  What does not exist is a *suite*:
  a checked-in benchmark set that runs the same way on another machine, so
  section 33 item 14 is answered `partial` and `ggc conform --strict` lists it.
  A number in a document is not a baseline.
- **There is a release, and it is Alpha.**  `v1.2.0` is tagged and published
  from `78e8a51`, with the wheel, source distribution, a reproducible build
  manifest, checksums, and the conformance report and `--strict` list attached.
  The digest of the manifest was `ecd0e756b5097f80b95d825c946368d039a02814de372c591d7a48b6ed40ab40`
  on the runner and the manifest is `signed: no`, because the repository has no
  `RELEASE_SIGNING_KEY`; the release notes say exactly that rather than
  implying provenance.  The metadata bar for leaving Alpha is stated in
  `docs/RELEASES.md` and is the same command as everything else --
  `ggc conform --strict` -- which currently lists 13 things.  So the release
  exists, and it says it is not production grade, which is the arrangement
  spec section 43 asks for.

---

## 5. What would have to change

In rough order of return on effort:

1. ~~An interpreter with an explicit call stack, removing the ~129-frame
   recursion limit~~ -- **done** in this revision, with its cost measured and
   stated rather than assumed.
2. Capability enforcement in the native runtime, so the security model does not
   depend on refusal alone.
3. ~~A free/release path in the arena~~ -- done for the functions where
   releasing is provably safe; still open is driving it from the memory
   model's per-level extents rather than from the codegen's own conservative
   rule, which would cover the functions that rule currently declines.
4. ~~A conformance suite that reads the specification rather than the
   implementation~~ -- **built** (`conformance/`, `ggc conform`).  What remains
   is what item 10 of section 33 asks for and this cannot supply: a review by
   someone who did not write it.
5. Constant-time crypto, or removal of crypto from the shipped surface.
6. A published benchmark suite, not just published measurements.
7. ~~Releases, and the metadata to match~~ -- **done**: `v1.2.0` is published
   with its conformance report attached, and a release now has to pass the
   suite, the tag check and the conformance suite before it can exist.  What
   remains is signing: the manifest is reproducible but `signed: no` until a
   `RELEASE_SIGNING_KEY` secret is added, and `docs/RELEASES.md` says how.

None of that is a weekend.  Naming it is the point: the toolchain is honest
about its edges, and this file is how that honesty is kept checkable.

---

## 6. How to check any of this yourself

```
python -m unittest discover -s tests -t tests -q      # 644 tests
python tools/fuzz_selfcheck.py                        # the checks can fail
./tools/bin/ggc difftest examples/*.gg examples/core/*.gg

# the toolchain against the specification, and against the production bar
./tools/bin/ggc conform            # 13 claims: exit 0
./tools/bin/ggc conform --strict   # what is not evidenced: exit 4, 13 items

# the three defects, as they were reported
printf '\x9e\xfe' > /tmp/bad.gg && ./tools/bin/ggc check /tmp/bad.gg
python3 -c "open('/tmp/pp.gg','w').write('fn main() -> Unit\n    let x = ' + '('*500 + '1' + ')'*500 + '\n    print(x)\n')" && ./tools/bin/ggc check /tmp/pp.gg
printf 'fn down(n: I64) -> I64\n    if n <= 0\n        return 0\n    return down(n - 1) + 1\n\nfn main() -> Unit\n    print(down(5000))\n' > /tmp/deep.gg && ./tools/bin/ggc run /tmp/deep.gg
```

Each of the three should print a diagnostic and exit without a Python
traceback.

The conformance run is the check to read rather than to trust.  It names the
section each claim comes from and the detail behind each result, so a claim
that passes is a claim with evidence attached, and the `--strict` list is the
distance to the production bar in the toolchain's own words.
