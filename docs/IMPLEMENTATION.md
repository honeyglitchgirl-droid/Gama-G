# Gama-G: implementation status

This document says, as precisely as it can, **what is actually implemented** in
this repository and **what is not**. It is the honest counterpart to
`Gama-G_v1.0_Production_Specification.txt`, which describes a multi-year,
multi-team production language. This repository is a working vertical slice of
it: a complete compiler front end, an intermediate representation, an
optimizer, and a reference interpreter, with the safety systems that define the
language — types, effects, capabilities, secrets, audit, recovery — enforced
end to end.

This repository is **Gama-G 1.0**: one language, one pipeline, and four ways to
run it.

Earlier milestones built the language's two surfaces separately, and the
toolchain showed it -- a core file could not contain a function and a core
expression could not call one. Both declaration families now live in one file and
share a lexer, a parser, a checker and a pipeline; the helper's GIR is merged
into the module the core produced, so nothing after lowering can tell which
family a function came from. See [`DESIGN_v1_0.md`](DESIGN_v1_0.md) §1.

After GIR the same machine serves the reference interpreter (the semantics
everything else is compared against), a **native CPU backend** that emits C and
compiles it to machine code, a **WebAssembly encoder**, and an **accelerator
layer**. The native backend is validated by *differential testing*: the same
program is run on the interpreter and on the compiled binary, and stdout, exit
status and fault kind are compared.

The per-milestone design notes remain as the record of how each part was built:
[`DESIGN_v0_2.md`](DESIGN_v0_2.md) (the language core),
[`DESIGN_v0_3.md`](DESIGN_v0_3.md) (the native semantic IR) and
[`DESIGN_v0_4.md`](DESIGN_v0_4.md) (the three formal models). This document is the
honest status; `DESIGN_v1_0.md` is the consolidated design.

Everything claimed below is exercised by the test suite (`python3 -m unittest
discover -s tests`, **628 tests**) and demonstrated by a runnable example in
`examples/` or `examples/core/`. Where a claim is partial, the missing part is
named.

---

## 1. Summary

| | |
|---|---|
| Implementation language | Python 3.11+ (reference implementation) |
| Compiler source | ~30,700 lines across 64 modules, plus a C runtime |
| Standard library | 266 builtins across 27 modules |
| Language surface | 23 hard keywords, 84 contextual keywords, 9 effects |
| AST node types | 70 |
| GIR operations | 41 |
| Tests | 628 (all passing under `unittest discover`) |
| Examples | 8 v0.1 + 8 core, each runnable with `ggc run` |
| Language core | 8 modules, ~4,300 lines in `compiler/gamag/core/`; see §10–12 |
| Formal models | memory · capability · recovery, one module each, plus a
  shared capability algebra; see §12 |
| Licence | Apache-2.0 (`LICENSE`); version in `VERSION`, packaging in `pyproject.toml` |
| Dialects | selected per file by the `gama core <version>` pragma |
| Native backend | implemented for a subset; the rest is refused by name — §5 |
| Other backends | WebAssembly (encodes; never executed here), accelerator (detects; never run here) — §5 |
| Toolchain | fuzzing, benchmarking, signed builds, package manager, FFI — §5 |
| Interop | FHIR, terminology, provenance, consent, database, identity, workflow, messaging, observability — §5 |

The compiler runs from a checkout with no installation step:

```sh
./tools/bin/ggc run examples/hello.gg
./tools/bin/ggc test examples/property_tests.gg
python3 -m unittest discover -s tests
```

---

## 2. The compilation pipeline (spec §22)

Spec §22 lists fourteen stages. Their status:

| # | Stage | Status | Notes |
|---|---|---|---|
| 1 | Lexing | **done** | `lexer.py`. Indentation-sensitive with layout tokens; `..=` lexes as one token; `//` and `/* */` comments; a bare `_` is the wildcard, not a name. |
| 2 | Parsing | **done** | `parser.py`. Both indented and braced blocks, 70 node types, contract and policy-rule source text captured for explainability. |
| 3 | Name resolution | **done** | `semantic/checker.py`. Scoping, forward references, policy subjects, pipeline stage bindings. |
| 4 | Type checking | **done** | Full lattice in `semantic/types.py`; sized integer ranges; no implicit numeric conversion (§6). |
| 5 | Effect checking | **done** | Nine effects (§7). A `pure` function that performs I/O is an error; an undeclared effect is an error under `strict`, a warning otherwise. |
| 6 | Capability checking | **done** | §12. `grant` headers, capability-qualified parameters, literal `capabilities.open` permission lists checked at compile time. |
| 7 | Ownership analysis | **partial** | `let` immutability and secret propagation are enforced. There is **no borrow checker and no move semantics**; aliasing is permitted and the interpreter's values are reference types. |
| 8 | Operation-graph construction | **partial** | `parallel` regions are analysed into a task DAG with real dependency levels. `pipeline` declarations parse and type-check but their stage graph is not yet fused or scheduled. |
| 9 | GIR generation | **done** | `gir/builder.py`, `gir/ir.py`. 41 operations, serialisable to JSON. |
| 10 | Optimization | **done** | `gir/optimizer.py`, ordered per §23: correctness → security → determinism → latency → memory → size. |
| 11 | Backend lowering | **not done** | |
| 12 | Machine code / WASM generation | **not done** | |
| 13 | Link / package | **not done** | No `gpm`; see §5. |
| 14 | Reproducibility verification | **partial** | Deterministic mode gives a virtual clock, seeded RNG and reproducible event ids, so the audit chain hash is stable across runs (tested). There is no separate reproducibility *attestation* artifact. |

Stage 14 is the reason `driver.execute` defaults to deterministic: a program
whose audit hash changes between runs cannot be reproduced, and reproducibility
is the property spec §1.3 leads with. `--lenient-runtime` opts out.

---

## 3. Language features, by specification section

Implemented and tested:

- **§3 Operation graphs** — `parallel` regions with dependency analysis and
  concurrent execution of independent tasks; `pipeline` declarations with stage
  bindings and a `result` name.
- **§4 Basic syntax** — `fn`, `let`/`var`, `record`, `enum` (with payloads),
  `match` with guards, `for`/`while`, both block spellings.
- **§5–6 Types** — every scalar, container and domain type the spec lists,
  including `Tensor<T,Shape>`, `Matrix`, `Duration`, `Instant`, `UUID`, `URI`,
  `Bytes`, `Tuple(...)` and capability-qualified types. `Result`/`Option` are
  the error model (§19); discarding a `Result` is an error under `strict`.
  Match exhaustiveness is checked for enums, `Result`, `Option` **and** `Bool`.
- **§7 Effects** — all nine, declared and inferred, with a diagnostic that
  names the undeclared effect.
- **§8 Memory / secrets** — `secret T` is a distinct type. Printing,
  concatenating or auditing a secret is a compile error. `secrets.expose`
  requires a reason and that reason is audited. For the **core** there is now an
  explicit memory model — one owner per binding, an extent in the derived order,
  a slot assigned only where extents provably do not overlap, a `SECRET_GUARD`
  per secret definition, and `ggc memory` to print it. See §12.
- **§9 Concurrency** — structured `parallel` regions. Tasks are ordered by
  dependency; independent ones run concurrently on real threads.
- **§9B Agents** — `agent` declarations with typed `on <event>` handlers and
  message dispatch. Agents share no mutable state. An event no handler declares
  is refused.
- **§10–11 Self-healing and checkpointing** — `service` with `protect`/
  `recover`, the six recovery levels in spec order, bounded retry, checkpoint
  capture and restore, operator escalation. Every recovery action is audited
  with its level, whether it succeeded, and what it did. `recover` and
  `checkpoint` are now **core syntax too**, lowered as a policy into a `PROTECTED`
  region that carries the declared steps in the program's own words. See §12.
- **§12 Security** — `grant` headers; capability handles minted at run time and
  permission-checked on every access; no ambient authority. One capability
  algebra is now shared by the checker and the runtime, so `covers()` means the
  same thing on both sides of the boundary, and every demand a core node derives
  from what it calls becomes a `CAP_CHECK` — even one the compiler already
  proved. See §12.
- **§13 Audit** — append-only, hash-chained, signed records. Tampering with a
  field, an action, or deleting a record from the middle is detected. Exports as
  newline-delimited JSON that round-trips.
- **§14 AI/ML** — `Tensor` with broadcasting, `matmul`, `transpose`, `softmax`
  and friends; reverse-mode autodiff verified against closed-form gradients;
  `model` declarations whose signature comes from `input`/`output`.
- **§15 AI safety** — `require` gates checked at run time. Model output is data,
  not authority.
- **§16 Medical** — `medical.Patient`/`Observation`/`Medication`/`Encounter`/
  `DiagnosticReport` types and a dosing example with contracts. **No compliance
  claim is made or implied** — see §6.
- **§17 Enterprise** — `policy` declarations with `allow`/`deny`/`require`/
  `audit` rules producing explainable decisions that quote the rule text.
- **§18 Transactions** — `transaction` blocks that commit or abort; a fault
  before commit records `TRANSACTION_ABORT` with the reason rather than leaving
  the transaction open. Every core **transition** is emitted inside one, whether
  or not the intent declares a recovery policy.
- **§19 Errors** — classified faults (`DivideByZero`, `ContractViolation`,
  `CapabilityViolation`, `RecoveryExhausted`, …) with source positions.
- **§21 GIR** — a serialisable IR with 41 operations, inspectable per function
  with `ggc gir --fn <name>` and as JSON with `ggc build --json`.
- **§23 Optimization** — the six-tier ordering is implemented in
  `gir/optimizer.py`; `-O0/-O1/-O2` are behaviour-preserving (tested).
- **§26 Testing** — `test`, `test property`, `for all x in <domain> where …`,
  the domain-less `for all x where …` form with deterministic sampling, and an
  optional `samples N`. `ggc test` exits non-zero on failure.
- **§27 Contracts** — `requires`/`ensures` with the condition quoted verbatim in
  the violation message.
- **§28 Standard library** — 27 modules, 266 builtins.

---

## 4. Toolchain (spec §31)

`ggc` implements **all nine commands the spec lists**, plus six the audits
asked for (build inspection, backends, fuzzing, packaging, signing,
diff-testing).  The nine of spec section 31:

| Command | Status |
|---|---|
| `ggc check` | **done** — diagnostics only, with `--profile` and `--json` |
| `ggc build` | **done** — compiles to GIR, `--stats`, `-O0/1/2` |
| `ggc run` | **done** — `--grant`, `--audit PATH`, `--entry`, `--lenient-runtime` |
| `ggc test` | **done** — `--json`; exit code 4 on failure |
| `ggc audit` | **done** — `audit verify` recomputes every digest and every link of a written trail, offline, and `audit show` prints it; signature checking runs when `--key` supplies the deployment key, and the output says which mode ran |
| `ggc bench` | **done** — §6; a published baseline is in `docs/BENCHMARK_BASELINE.md` |
| `ggc profile` | **done** — per-function calls, instructions and inclusive time, counted by the interpreter itself; instruction totals agree with the VM's own counters, and the table goes to stderr so the program's own output stays on stdout |
| `ggc format` | **done** — canonical layout with a **same-GIR safety proof**: every rewrite is compiled before and after and compared apart from positions, so a formatting that would change a program is refused, never applied; clause bodies (quoted verbatim by failures) are never re-spaced; idempotent, and all sixteen examples are already canonical |
| `ggc doc` | **done** — markdown for a program (including what the checker derived: levels, authority, selection proofs) and for the standard library, rendered from the same registration table the checker reads |

Exit codes: `0` success, `1` compile error, `2` runtime fault, `3` usage,
`4` test failure.

The vocabulary test keeps a `ROADMAP_COMMANDS` guard -- empty now, but the
mechanism stands so a command can never be *claimed* before it is *built*.

---

## 5. What is not implemented

Named deliberately, because spec §43 forbids claiming capability the toolchain
does not have.

**The native backend covers a subset**, and refuses by name whatever it cannot
compile, before writing any C. `ggc native` emits C and compiles it to machine
code; `ggc difftest` runs the same program on both machines and compares stdout,
exit status and fault kind. The audit chain, capabilities, transactions,
checkpoints, recovery regions, tensors, autodiff, agents, method dispatch and
indirect calls are not implemented natively. Four of the sixteen shipped examples
compile natively today; twelve are refused; none diverges. A refused program
still runs on the reference interpreter, which remains the definition of the
language.

The native runtime allocates heap values from an arena and does not free until
exit, so it does not yet act on the extents the v0.4 memory model computes. It
computes integers in 64 bits, which is why an integer type wider than that is
refused at analysis time rather than truncated.

**The WebAssembly backend has never been executed here.** There is no WASM
runtime in this environment, so what is claimed is structural conformance, and
`ggc wasm` says so when it writes a file. The **accelerator layer has never run a
kernel here** either, for the same reason: no device is present, and the
placement report says the CPU ran it rather than falling back silently.

**No performance claim is made anywhere.** `ggc bench` measures and reports, with
the conditions of the measurement (machine, Python, profile, tier) attached to
every number, percentiles rather than means, and instruction counts alongside
the timings because they do not depend on machine load. It refuses to compare
two runs whose conditions differ, because a ratio between incomparable runs is a
made-up number.

`Tensor` operations use NumPy when it is importable and fall back to a
pure-Python implementation when it is not, so the language has no hard
dependency on it. The test suite passes either way.

**The package manager does not reach the network.** `gpm` resolves, locks,
verifies and audits, but a registry is a directory of packages rather than a
server, which makes the offline cache and a private registry the same mechanism.
Resolution is a fixed point with a bounded backtracking search around it: a
graph satisfiable only below the highest satisfying version of something now
resolves, deterministically, and a graph beyond the search budget is still
reported as a named conflict rather than guessed at (see `gpm/package.py` for
why *bounded* is the honest word for an NP-complete subproblem). Signing is Ed25519, implemented
from RFC 8032 and validated against the RFC's vectors; it is **not
constant-time**, so it must not be used where an attacker can measure signing
time.

**No borrow checker in the Rust sense.** The **core** has an explicit memory
model since v0.4: one owner per binding, extents computed from the derived
execution order, slots shared only where extents provably do not overlap, and
use-after-free prevented by that analysis rather than by the host's collector.
What it does not have is alias analysis over arbitrary mutation — there is no
arbitrary mutation in the core to analyse, no lifetimes to infer across function
boundaries, and no move semantics, because the core is single-assignment by
construction. The **v0.1 surface** still relies on immutability by default plus
secret-type propagation, and for it the older description holds: the interpreter
is garbage collected, so that class of defect cannot arise, but neither is it
prevented by analysis.

**No pipeline stage fusion.** §3's operation-graph analysis exists for
`parallel` regions. `pipeline` declarations parse, type-check and run, but the
compiler does not yet fuse compatible stages or schedule them across devices,
which is what §38's fraud-detection example is meant to demonstrate.

**Some standard-library modules are declared but not implemented**, and say so
rather than being stubbed: `http`, plus the roadmap commands `profile`, `format`
and `doc`. Nine modules that were on that list have since been implemented --
`fhir`, `terminology`, `provenance`, `consent`, `database`, `identity`,
`workflow`, `messaging` and `observability` -- and a test asserts that a module
is not declared roadmap once it is registered, because that would be a false
claim about the toolchain. `messaging` is in-process: publishing to a broker
needs `NetworkConnect` plumbing, and none is claimed.

**Interop (§29) is partial.** The C ABI is available through `ffi`, gated by the
`ForeignCall` capability and the `unsafe` effect, with a closed type vocabulary:
a pointer is not expressible, because a pointer Gama-G cannot verify is what
§29's "marked unsafe" rule is about. JSON is available. CBOR, Protocol Buffers,
database protocols beyond SQLite and a Python embedding API are not
implemented.

**Formal verification (§27) is contracts only.** `requires`/`ensures` are
checked at run time and quoted in violations. There is no proof assistant, no
SMT backend, no static verification of the contracts.

---

## 6. Claims this implementation does not make

Spec §43 lists what Gama-G must never claim. These prohibitions are enforced
here in two ways: by not making the claims, and by a test
(`test_enforcement.ForbiddenClaims`) that scans every `.md`, `.py`, `.gg` and
`.txt` file in the repository for the forbidden phrases and fails if any appear.

- **No universal performance guarantee.** A native backend exists and `ggc bench`
  measures it, but a measurement of one program on one machine is not a property
  of the language. Measurements are reported with their conditions and are not
  comparisons. There is **no performance claim** anywhere in this repository.
- **No universal accuracy guarantee.** The autodiff and training example
  converges on `w=2, b=1` for `y = 2x + 1` because that is what gradient
  descent does on a linear model — it is a correctness test, not an accuracy
  claim about any real dataset.
- **No automatic medical, legal or regulatory compliance.** `examples/medical_dosing.gg`
  demonstrates contracts and domain types. It does not make anything compliant
  with anything. Clinical correctness requires domain validation and the
  appropriate regulatory process, exactly as spec §25 states.
- **No claim that the audit chain is legally admissible, or that signing it
  establishes non-repudiation.** Records are HMAC-signed with a key the process
  holds; key management, witnessed countersigning and tamper-evident storage
  are outside this implementation.
- **No claim that recovery makes a system self-healing in production.** The
  recovery engine is bounded and audited, and a permanent fault exhausts it and
  propagates. That is the design, not a success story.

---

## 7. Architecture

```
source → lexer → parser → checker ─────────────┐
                              │                 │
              name resolution │  type / effect / │  diagnostics
              capability      │  ownership       │  (recovering)
                              ▼                 │
                        gir.builder → gir.ir ───┘
                              │
                        gir.optimizer  (§23 ordering)
                              │
                              ▼
                        runtime.vm (GAEM interpreter)
                              │
        ┌──────────┬──────────┼──────────┬────────────┐
        ▼          ▼          ▼          ▼            ▼
    context     values     tensor    recovery      audit
   (grants,   (records,  (NumPy,   (six levels,  (hash chain,
    clock,     variants,  tape)     checkpoints)  signatures)
    RNG)       secrets)
```

Notable design decisions:

- **Diagnostics recover.** The checker accumulates errors so an editor gets
  every problem in a file, not just the first. Phases that cannot continue
  raise instead.
- **One table for methods.** `methods.py` is read by both the checker and the
  VM, so a method cannot be callable at run time while rejected at compile
  time. This was a real defect before it was unified.
- **Contracts keep their source text.** `requires weight > 0` produces a
  violation reading `requires contract violated: weight > 0`, because the
  parser records the condition as written. The same mechanism makes policy
  decisions explainable.
- **Synthetic names are namespaced.** Parallel tasks, protect bodies and policy
  contexts generate functions; each producer namespaces its own, and
  `GProgram.add` raises on a duplicate rather than silently replacing one.

---

## 8. Mapping to spec §44, "recommended first implementation"

§44 lists seventeen items to build first. This is where each stands:

| # | Item | Status |
|---|---|---|
| 1 | Formal language specification | the blueprint document; no separate formal grammar artifact |
| 2 | Lexer / parser | **done** |
| 3 | Static type checker | **done** |
| 4 | Result / Option types | **done** |
| 5 | Ownership / memory safety model | **partial** — immutability and secret propagation, no borrow checker |
| 6 | Gama IR | **done** |
| 7 | Reference interpreter | **done** |
| 8 | Native AOT backend | **not started** |
| 9 | Operation-graph optimizer | **partial** — GIR optimizer done, pipeline fusion not |
| 10 | Capability system | **done** |
| 11 | Audit system | **done** |
| 12 | Structured concurrency | **partial** — `parallel` regions only |
| 13 | Recovery engine | **done** |
| 14 | Tensor subsystem | **done** |
| 15 | Medical / enterprise libraries | **partial** — types and in-language policy/transaction, no FHIR or database |
| 16 | Package manager | **not started** |
| 17 | Security hardening | **partial** — capabilities, secrets and audit done; no sandboxing, fuzzing or supply-chain controls |

Nine of the seventeen are done outright. Five are partial — ownership, the
operation-graph optimizer, structured concurrency, the medical/enterprise
libraries and security hardening. Two are not started: the native AOT backend
and the package manager. Item 1 is the blueprint document itself. The gaps are
the work spec §34 schedules across Phases 2 to 7, and the reason this repository
is described as a vertical slice rather than an implementation of the
specification.

---

## 9. Roadmap

The audit report re-sequenced this roadmap: *"Only after that redesign should
native backends and production v1.0 be built."* Item 1 is therefore done at the
language level, and the remaining items are ordered behind extending the core
rather than behind the v0.1 surface.

1. ~~**Pipeline stage fusion and scheduling** (§3, §38)~~ — **done as the v0.2
   core**: the operation-graph story is now the language's actual program model
   rather than a feature of one syntactic form. See §10. What remains is
   *acting* on the derived levels (parallel execution of a level), not deriving
   them.
1a. **Extend the core** — modules and composition between intents, more than two
   guarded alternatives with a real exhaustiveness proof, and a container choice
   for `each` other than `List`. (Recovery policy in core syntax is done — see
   §12.)
2. **A real backend** (§22 stages 11–12) — WASM first, because it gives
   portability and a sandbox boundary at once; native afterwards.
3. **Borrow checking** (§8) — narrowed by v0.4's memory model for the core, and
   still the largest remaining gap for the v0.1 surface, where there are no
   lifetimes and no alias analysis.
4. **`gpm`** (§30) — needed before any library ecosystem is possible.
5. **Fuzzing and property generation** (§26) — `test fuzz` is parsed as a
   category but generates nothing.
6. **FHIR and database modules** (§16, §17) — the two declared-unimplemented
   modules with the clearest demand.
7. ~~**Benchmarking** (§24)~~ — **done**: `ggc bench` measures with its
   conditions, and `docs/BENCHMARK_BASELINE.md` publishes a baseline.  A
   compute-bound corpus with a tail is the open improvement (§5, baseline doc).

---

## 10. The v0.2 original language core

The audit's verdict on v0.1 was that the implementation is real but the language
surface is not original enough: `fn`, `let`, `var`, `if`, `else`, `for`,
`while`, `return`, `match`, `enum`, `record` and conventional `Result`/`Option`
syntax belong to other languages. §16 says to redesign around Intent, Operation,
Relationship, Constraint, Capability, Effect, State, Recovery, Audit and the
Execution Graph, and explicitly *not* to rename.

That redesign is implemented, tested and runnable.

### 10.1 The inversion

A v0.1 program is a sequence of statements. A v0.2 core program is a **set of
operations**, each declaring the bindings it consumes (`uses`) and the one
binding it produces (`yields`). The compiler matches producers to consumers,
derives the graph, validates it, and only then produces an order. **Order is an
output of compilation, never an input.**

`ggc graph` prints what was derived, and `tests/test_core_language.py` compiles
the same program with its operations listed in the opposite order and asserts
the derived levels are identical.

### 10.2 What is implemented

| Piece | Path | Status |
|---|---|---|
| Native semantic IR: expressions, patterns, constraints, node kinds, five graphs | `core/mir.py` | complete |
| Declaration grammar, clause table, verbatim text capture | `core/parser.py` | complete |
| Relationship validation, cycle detection, level derivation, guard proofs | `core/graph.py` | complete |
| Native checking and lowering to GIR | `core/native.py` | complete |
| Dialect detection, two-phase front end | `driver.py::is_core_dialect` | complete |
| `ggc graph [--edges] [--json]` | `cli/main.py` | complete |
| Six examples, one per construct family | `examples/core/` | complete |
| 77 tests | `tests/test_core_language.py` | complete |

Constructs: `intent` (with `purpose`, `authority`, `trail`), `source` (with
`secret` and `from`), `state` (with `starts`, `authority`), `operation`,
`refine` (bounded repetition), `each` (fan-out), `resolve` (exhaustive
dispatch), `transition` (state change under authority), `outcome`. Clauses:
`uses`, `yields`, `effect`, `needs`, `holds`, `when`, `computes`, `trail`,
`starts`, `repeats`, `until`, `within`, `over … as …`, `choose`, `alters`.

### 10.3 What the core refuses

`E-graph-cycle` (naming the cycle), `E-undeclared-relationship`,
`E-unresolved-relationship`, `W-unused-relationship`, `E-ambiguous-selection`,
`E-duplicate-binding`, `E-unbounded-refinement`, `E-refine-incomplete`,
`E-each-incomplete`, `E-resolve-incomplete`, `E-compute-after-commit`,
`E-transition-target`, `E-state-uninitialised`, `E-no-outcome`,
`E-no-yields`, `E-no-yields-type`, `E-no-computes`, `E-unknown-effect`,
`E-authority-unmet`, `E-dispatch-not-exhaustive`, `E-dispatch-unsupported`,
`E-unknown-type`, `E-type-arity`, `E-yield-type`, `E-binop-type`, `E-unop-type`,
`E-arg-type`, `E-arity`, `E-unknown-call`, `E-constraint-type`,
`E-effect-undeclared`, `E-secret-escape`, `E-ice` — 33 codes in total. At run
time the core adds three classified faults: `NoActiveAlternative`,
`RefinementDiverged` and `UnresolvedDispatch`, plus `ContractViolation` for a
`holds` constraint that data breaks.

### 10.4 What is proven versus only checked

Proven at compile time: acyclicity; that declared relationships match real ones
in both directions; one producer per binding unless every alternative is
guarded; that every repetition has a bound of at least one round; mutual
exclusivity and exhaustiveness for a two-guard selection whose guards are
syntactically complementary; that compute never depends on a committed change;
plus types, arity, library signatures, effect coverage and secret propagation,
all checked natively by `core/native.py`.

Checked only at run time: data-dependent constraints, selections whose guards
fall outside the prover's fragment, and refinements that reach their bound.

**The selection decision procedure.**  A `when`-guarded selection used to be
provable only for two guards that were complements in *spelling*.  Since v1.2
`core/guardproof.py` decides exhaustiveness and exclusivity **exactly** for
the decidable fragment: boolean combinations (`and`, `or`, `not`) of
comparisons of one sized-integer binding against integer constants, computed
as finite unions of integer intervals.  A three-way partition
(`a >= 90` / `a >= 75 and a < 90` / `a < 75`) is now `proven`, recorded with
the proof note the `ggc graph` and `ggc doc` output show, and the recovery
graph carries no obligation for it.  Constant guards (`true`, `false`) are
decided too.  What stays `unprovable` is stated rather than discovered:
float subjects, where NaN makes interval coverage a lie; guards reading more
than one binding; function calls inside guards.  For those, the
`NoActiveAlternative` fault still runs, which is precisely what recording
`unprovable` promises.

**Not checked at all:** re-deriving an operation's declared `effect` from its
own clauses rather than trusting the declaration.

### 10.5 What the core does not do

No native backend (it runs on the reference VM). No performance claim of any
kind — nothing is benchmarked. No parallel execution: same-level operations are
*known* to be independent — and since v0.3 that knowledge is recorded on the GIR
function as per-task reads, writes and dependencies — but the emission is still
sequential. `each` yields a `List`; there is no container choice. No
modules, imports, generics or user-defined types in the core — a core program is
one intent, and composition between intents is not designed. Recovery is bounded
repetition plus classified faults; the specification's richer
retry/compensate/escalate policy is implemented in the reference runtime but not
yet expressible in core syntax. A `source` with no `from` makes the program a
library: it compiles and checks, but no `main` is generated.

### 10.6 Why v0.1 was kept

Three reasons. It is the machine that runs v0.2. It is the tested evidence that
the assets the audit lists in its §4 — effect system, capabilities, secret
propagation, hash-chained audit, determinism, bounded recovery, tensors and
autodiff, GIR, reference VM — actually work. And deleting a working
implementation to make a report look tidier would be a worse engineering
decision than the one the report criticises.

---

## 11. The v0.3 native semantic IR

`Gama-G_Complete_Originality_and_Technical_Audit.txt` §13 made one structural
finding: v0.2 had a new source language but compiled it by elaborating into the
older implementation's abstract syntax, so `let`, `var`, `if`, `while` and
`match` reappeared as the intermediate representation of a language whose purpose
is not to have them. Priorities 1 and 2 of its §15 ask for a native semantic IR
and a compiler that reaches GIR without the older AST.

Both are implemented. Full detail is in [`DESIGN_v0_3.md`](DESIGN_v0_3.md).

### 11.1 What was built

| Piece | Path | Status |
|---|---|---|
| Native semantic IR — expressions, patterns, constraints with a `discharge`, seven node kinds | `core/mir.py` | complete |
| Five graphs — intent, operations, constraints, authority, recovery | `core/mir.py`, built by `core/graph.py` | complete |
| Native checker — types, arity, effects, authority, secrecy | `core/native.py::NativeChecker` | complete |
| Native lowerer — semantic model → GIR, no older AST | `core/native.py::Lowerer` | complete |
| Derived graph recorded on the GIR function | `gir/ir.py::GFunction.parallel_tasks` | complete |
| `core/ast.py`, `core/elaborate.py` | — | **deleted** |

### 11.2 What is checkable about it

The property that matters is not "the source has no `if`" but "the compiler built
no older AST". That is asserted directly:
`test_no_older_abstract_syntax_is_built_for_a_core_program` compiles a core
program and requires `compilation.module` to be `None`.

Also asserted: the generated function's kind is `intent`; every basic block in
every example ends with a terminator (the reference interpreter treats an
unterminated block as `return ()`, so a missing jump would silently truncate
rather than fail); the derived levels appear as GIR operation-graph metadata;
constraint discharges are distinguished as proven / runtime / unprovable; the
recovery graph enumerates every bound with its fault kind; and `ggc graph` prints
all five graphs.

### 11.3 What v0.3 does not do

At the time of v0.3 the audit's priorities 3–16 were untouched, in the order it
sequences them. Priorities 4–6 have since been built as v0.4 (§12). Still
untouched: the native CPU backend, WASM and GPU targets, `gpm`, FFI, the FHIR
profile, benchmarks and fuzzing. **No performance number appears anywhere in this
repository**, and §6's list of claims not made is unchanged.

### 11.4 Test split (audit §14)

The audit is right that `assert "if" not in grammar` proves an architectural
property and not historical originality, and that presenting the first as
evidence for the second is the failure mode to avoid. `tests/test_core_language.py`
is now split accordingly:

| Class | What it claims |
|---|---|
| `LanguageInvariants` | properties of the grammar and compiler, checked mechanically |
| `NativeLowering`, `NativeChecks`, `TheFiveGraphs` | properties of the v0.3 pipeline |
| `ProvenanceEvidence` | **no** assertion of originality — only that the repository's own documents do not claim more than the evidence supports, and that the audits they answer are kept |

The class formerly named `OriginalityGuarantees` is `LanguageInvariants`. The old
name asserted more than the tests delivered.

---

## 12. The v0.4 formal semantics of the core

The audit's priorities 4–6, built as three models the toolchain computes rather
than three properties a document repeats. [`DESIGN_v0_4.md`](DESIGN_v0_4.md) has
the detail; this is the status.

### 12.1 What was built

| Priority | Model | Module | Diagnostics it can raise |
|---|---|---|---|
| 4 | Explicit memory/resource | `core/memory.py` | `E-memory-model` (driver), `E-secret-escape` (native) |
| 5 | Formal capability semantics | `core/capability.py`, `capabilities.py` | `E-unknown-capability`, `E-capability-unmet`, `E-authority-unmet`, `E-unknown-effect`, `E-effect-undeclared` |
| 6 | Direct transition/recovery | `core/recovery.py` | `E-unknown-recovery`, `E-recovery-unordered`, `E-unknown-checkpoint`, `E-unbounded-recovery` |

What reaches the machine:

* `SECRET_GUARD` per secret definition, `CAP_CHECK` per derived demand,
  `CHECKPOINT` per declared label, `PROTECTED` per recovery policy,
  `TRANSACTION` begin/commit per transition.
* A `GFunction.recovery` plan carrying the declared steps in the program's own
  words, with the specification's level names — so the policy that runs is the
  policy that was written.
* One capability algebra (`capabilities.py::covers`) used by the checker and by
  `Context.has_cap`, including the runtime's `"*"` wildcard.

`ggc memory <file>` prints the model; `--slots` lists slot lifetimes and reuse;
`--json` dumps bindings, slots, acquisitions, `slots_saved` and violations.
`ggc graph` already showed the capability and recovery sections, because they are
part of `model.render()` — the models are inspectable through the command that
existed for the graphs.

### 12.2 What is checkable about it

47 tests in `tests/test_formal_semantics.py`, named after the specification
sections they implement rather than after the modules that hold the code:

* **Memory** — ownership is one-to-one; every borrower falls inside its owner's
  extent; parameters and outcomes live to the return; nothing is mutable without
  a declared `state`; a reused slot's occupants provably do not overlap and are
  of one type; **a two-operation chain reuses nothing**, which is the negative
  case that makes the positive one mean something; secrets are guarded, a
  fingerprint of a secret is not itself secret, and the generated entry point
  prints `[secret <name>]`; acquisitions name their owner, resource and extent.
* **Capability** — the two spellings of one capability parse alike; `Write ⇒ Read`
  on a resource and nothing across resources; `Connect`/`Sign`/`Spawn` entail
  nothing; attenuation is coverage read backwards and no combination amplifies;
  an unknown capability is refused; a secret reaching a renderer demands
  `SecretExpose`; a covered demand is satisfied and an uncovered one is refused;
  **the runtime applies the same relation the compiler proved**; a boundary is
  emitted even for a proven demand, and refusing it at runtime records
  `CAPABILITY_DENIED` at security level.
* **Recovery** — the levels are the specification's; a policy parses into its
  steps with resolved targets; levels may not decrease; a named checkpoint must
  have been declared while a bare `restore checkpoint` needs none; an unbounded
  retry and an unknown action are refused; the policy lowers to a `PROTECTED`
  region carrying its own words with `audit_all`; declared checkpoints are
  captured **before** the transaction; a named restore names the checkpoint it
  used in the audit trail; audited levels never decrease; a constraint failing in
  the commit phase leaves the transaction `aborted`, never half-applied; and a
  transition with no recovery policy still runs inside a transaction.

Two examples were added: `examples/core/custody.gg` and
`examples/core/recover.gg`. The core set is now eight programs.

### 12.3 Two defects the models found

Both were invisible to the 269 tests that passed before them, and both were found
by running a program the model said should work:

* **A core program's declared `authority` never reached the runtime.** Four call
  sites read `compilation.checker.grants`, which is `None` for a core program, so
  the compiler proved every demand covered and the runtime denied every one.
  Fixed by `driver.program_grants(compilation, extra)`, which unions the
  program's own grants, the checker's and the caller's; every run path uses it.
* **`effect` was singular, so multi-effect builtins were uncallable.** The
  standard library's own `secrets.expose` (crypto + audit),
  `medical.fhir_serialize` (medical + io) and `model.load` (model + storage)
  could not be called from the core at all — a formally checked capability that
  could not be exercised. `OpNode.effects` is now a list, with `effect` kept as a
  read-only property returning the first.

`tests/test_enforcement.py::SourceTreeHygiene` now scans every module for a
top-level name defined twice and for two functions with identical bodies, because
the first of those was how a duplicated `known()` survived a refactor unnoticed.

### 12.4 What v0.4 does not do

* **Priority 3, the native CPU backend, is not started.** It is a multi-session
  code-generation effort rather than a model, and starting it badly would be
  worse than naming it absent. Priorities 7–16 are untouched.
* **No alias analysis over arbitrary mutation, no lifetime inference across
  function boundaries, no move semantics** — see §5. The core has no arbitrary
  mutation to analyse, which is a stronger starting position and not the same
  thing as a borrow checker.
* **No data-race detector.** The derived graph makes the order explicit and
  `parallel` regions carry dependency analysis, but nothing here proves the
  absence of races in a program that has been given one.
* **No claim that a recovery policy will succeed.** Bounded, ordered, named and
  audited is the guarantee. Whether retrying twice is *enough* is a property of
  the failure.
* **The memory model describes the core only.** The v0.1 surface keeps its own
  ownership checking in `semantic/checker.py`, unchanged.
* `slots_saved` is a count from the model, not a measurement. **No performance
  number appears anywhere in this repository.**
