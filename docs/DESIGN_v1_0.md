# Gama-G 1.0

This is the consolidated design of the released toolchain.  It replaces the
per-milestone design notes as the description of what the language *is*; those
notes (`DESIGN_v0_2.md`, `DESIGN_v0_3.md`, `DESIGN_v0_4.md`) remain as the record
of how each part was built and why, and are worth reading for the reasoning
rather than the summary.

The release is one language and one version.  Earlier milestones built the two
surfaces of the language separately -- `fn`/`let`/`var` first, then the
`intent`/`operation` core -- and a reader of the specification would not have
guessed that from the toolchain.  That split is gone.

---

## 1. One language

A program is a sequence of declarations written in one syntax.  There are two
*families* of declaration, and they live in the same file, share a lexer, a
parser, a checker and a pipeline:

```gama
gama core 0.2
intent Rounding
    purpose   round a measurement to a whole number

source reading : F64 from 5.4

fn round_half_up(x: F64) -> I64          // a `fn` helper
    pure
    return math.round(x)

operation Rounded                         // a core operation calling it
    uses     reading
    yields   whole : I64
    effect   pure
    computes round_half_up(reading)

outcome whole
```

The helper is compiled by the same front end that compiles a module of `fn`
declarations, and its GIR is merged into the same module the core produced.  By
the time anything downstream sees the program, there is one `GProgram` with one
optimizer and one set of backends, and nothing can tell which family a function
came from.

The seam is the token stream.  `CoreParser` captures a `fn` declaration as a
token *span*, not as an AST, and the driver hands that span to the older parser.
Positions are preserved, so a diagnostic inside a helper points at the user's
own line.  Re-implementing a function body in the core parser would have meant
two definitions of what a function is, which is the problem the unification
exists to remove.

What is *not* unified: the two families still have different checking rules
appropriate to their shapes.  A core `operation` is a node in a derived graph
with an effect and a declared yield; a `fn` is an ordinary function with a
statement body.  Unifying their *semantics* would lose the graph the core exists
to provide.

---

## 2. The pipeline

```
source
  │
  ├─ lexer ───────────────────────────── one lexer for both families
  │
  ├─ core parser ──┐   ┌── declaration parser
  │                │   │
  │   fn helpers ──┴───┘        (captured spans, compiled by the same checker)
  │
  ├─ semantic graph (the core's derived order, edges, selections)
  ├─ checker (types, effects, capabilities, secret flow, ownership)
  ├─ the three v0.4 models: memory, capability, recovery
  │
  └─ GIR ── optimizer ──┬── interpreter (reference semantics)
                        ├── native CPU backend (GIR → C → machine code)
                        ├── WebAssembly backend (numeric subset)
                        └── accelerator layer (OpenCL C kernels)
```

Everything after GIR is shared.  Nothing after GIR knows which declaration
family a function came from, which is the property that makes "one language" a
fact about the implementation rather than a claim in a document.

---

## 3. What is implemented, and how it is checked

Every row names the evidence, because a feature list without one is a wish.

| Area | Where | How it is checked |
|---|---|---|
| Language surface | `parser.py`, `core/parser.py` | 501 tests; the spec vocabulary tests read the specification's own enumerations at run time |
| Types | `semantic/checker.py` | spec types 208-236, each inhabited by a compiling program |
| Effects | `semantic/checker.py` | declared-versus-inferred comparison; `unsafe` is a declared effect |
| Capabilities | `capabilities.py` | one algebra shared by compiler and runtime; attenuation only |
| Memory model | `core/memory.py` | ownership, extents, slot sharing, secrets; `ggc memory` |
| Recovery | `core/recovery.py` | levels 0-5, non-decreasing; `ggc graph` |
| GIR | `gir/` | 41 ops, JSON round-trip |
| Interpreter | `runtime/` | the reference semantics everything else is compared against |
| Native backend | `backend/cgen.py`, `backend/rt/` | differential testing against the interpreter |
| WebAssembly | `backend/wasm.py` | decoded back and structurally validated; **never executed** |
| Accelerator | `backend/accelerator.py` | device detection and kernel structure; **never executed** |
| Benchmarking | `bench/harness.py` | reports conditions and percentiles; makes no claim |
| Fuzzing | `fuzz/` | `ggc fuzz`, plus `tools/fuzz_selfcheck.py` proving the checks can fail |
| Signed builds | `toolchain/buildinfo.py` | Ed25519 against RFC 8032 vectors |
| Packages | `gpm/` | SemVer 2.0.0 precedence chain; resolver fixed point; content hashes |
| FFI | `std/ffi.py` | live libm calls through the C ABI, capability-gated |
| Medical interop | `std/interop.py` | FHIR round-trips; consent per subject and purpose |
| Enterprise | `std/enterprise.py` | SQLite-backed database, identity, workflow, messaging, observability |

---

## 4. The three backends, and what is honest about each

**Native CPU.**  GIR becomes C, and the system compiler turns that into machine
code.  The binding constraint is not code generation but agreement: the C
runtime reproduces three properties of the interpreter rather than approximating
them -- exact integers with the interpreter's range checks at *store* and
*return*, Python's shortest-round-trip float formatting (including its switch to
exponential notation outside `-4 < decpt <= 16`), and variadic space-joined
output.  A support analysis runs *before* any C is written, so a program using
the audit chain, capabilities, transactions, tensors or parallel regions is
refused with a reason naming the construct and its position.  Four of the
sixteen shipped examples compile natively today; twelve are refused, none
diverges.

**WebAssembly.**  A self-contained binary encoder, with a decoder and a
disassembler so the emitted instructions can be read against the GIR they came
from.  A program with control flow is refused rather than approximated, because
WebAssembly has structured control flow and no `goto`: restructuring an
arbitrary control-flow graph is a relooper, and emitting `unreachable` in its
place would produce a module that validates and then traps.  **There is no
WebAssembly runtime in this environment, so no module produced here has ever
been executed.**  The claim is structural conformance, and `ggc wasm` says so
when it writes a file.

**Accelerator.**  Device detection that reports every class it looked for and,
for each absence, why; and OpenCL C kernels for the operations that are
genuinely parallel, each checked for an entry point, an address space and a
bounds guard.  **No accelerator is present here, so no kernel has been run.**
No speedup is claimed anywhere: spec section 43 forbids the claim and the
absence of hardware makes it unverifiable twice over.

---

## 5. What 1.0 does not do

Written down rather than left to be discovered.  Spec section 43 governs this
section: never claim a capability the toolchain lacks.

- **The native and WebAssembly backends cover a subset.**  The audit chain,
  capabilities, transactions, checkpoints, recovery regions, tensors, autodiff,
  agents, method dispatch and indirect calls are not implemented natively.  They
  are refused by name, never miscompiled.  A program that uses them runs on the
  interpreter.
- **The native runtime allocates from an arena** and does not free until exit.
  The language's deterministic-destruction guarantee is a property the *memory
  model* computes over the derived graph and `ggc memory` reports; the native
  runtime does not yet act on those extents.
- **No GPU has ever run a kernel from this repository.**  Detection and code
  generation exist; execution does not, here.
- **`messaging` is in-process.**  A broker needs `NetworkConnect` plumbing.
- **`http` is not implemented** and is still declared roadmap.
- **`ggc profile`, `ggc format` and `ggc doc` do not exist**, and are declared
  roadmap rather than stubbed.
- **The package manager does not reach the network.**  A registry is a directory
  of packages, which makes the offline cache and a private registry the same
  mechanism.  Resolution is a fixed point without backtracking: a graph that can
  only be satisfied by choosing *below* the highest satisfying version of
  something is reported as a conflict rather than guessed at.
- **Ed25519 here is not constant-time** and must not be used where an attacker
  can measure it.  It exists because the alternative was a symmetric MAC, which
  proves possession of a shared secret rather than identity; in production, use
  an audited library.
- **The fuzzer found eight bugs and no more.**  That is a statement about the
  campaigns that were run, not about the absence of bugs.
- **No performance claim is made anywhere.**  The benchmark harness reports
  measurements with their conditions and refuses to compare runs whose
  conditions differ.

---

## 6. The bugs the work found

Recorded because each one was found by a *tool* rather than by reading, and each
now has a regression test.  They are the evidence that the verification is doing
something.

1. **Thirty-nine front-end diagnostics had no code.**  The fuzzer found them by
   asking every diagnostic it met whether it had one.
2. **The fuzzer's own execution budget was the interpreter's default.**  Fifty
   million VM instructions is minutes in CPython, so one generated loop hung a
   campaign; the wall-clock check could not report it because it only ran once
   `execute` returned.
3. **Two oracle checks swallowed internal errors** and reported "no difference"
   for a compiler that had fallen over -- the one answer those checks must never
   give.
4. **The campaign engine counted clean runs as compiled ones**, so a report
   could not distinguish a bad generator from a permissive compiler.
5. **A program could confer a capability on itself.**  `program_grants` unioned
   the program's own `grant` lines into the authority the runtime received, so
   `grant FileRead` was enough to read any file: ambient authority under another
   name, forbidden by spec section 12.  Authority now comes only from the
   caller, and the one place that decides whether to honour a declaration is
   `ggc run`, visibly, refusable with `--strict-authority`.
6. **The `unsafe` effect declaration was unusable.**  The parser diverted it
   into a flag nothing read, so the effect checker reported the function as
   performing `unsafe` without declaring it -- which made the marker section 29
   requires for foreign calls impossible to satisfy.
7. **The package resolver installed versions a constraint forbade.**  A one-pass
   resolver picks the highest version when it first meets a package and never
   revisits; a constraint from a package that sorted later could not withdraw
   the choice, and the resolver reported success.
8. **`consent.grant(...)` could not be parsed**, because `grant` is a hard
   keyword and the member-access rule required an identifier after `.`.  Every
   builtin whose name collided with a keyword was uncallable.
9. **Every fix-it hint on a runtime fault was dropped.**  Builtins pass `hint=`
   into `**ctx`, which lands in `context`, and nothing read it back out.
10. **A signed package could never verify.**  The signature covered the content
    hash, the signature was stored in the manifest, and writing the manifest
    changed the very hash that was signed.

---

## 7. How to check any of this yourself

```sh
./tools/bin/ggc run examples/hello.gg          # the language
./tools/bin/ggc difftest examples/hello.gg     # interpreter vs native
./tools/bin/ggc fuzz --rounds 500              # try to break it
python3 tools/fuzz_selfcheck.py                # prove the fuzzer can fail
python3 -m unittest discover -s tests -t tests # 501 tests
./tools/bin/ggc manifest examples/hello.gg --check-reproducible 3
./tools/bin/ggc bench examples/hello.gg --repeats 20
./tools/bin/ggc device                         # is there an accelerator? no
./tools/bin/ggc wasm examples/core/classify.gg --analyze
```

The fuzzing self-check is the one worth running first.  A tool that reports "no
invariant was broken" is only evidence if it is capable of reporting the
opposite, and that script breaks the product five ways to prove it is.
