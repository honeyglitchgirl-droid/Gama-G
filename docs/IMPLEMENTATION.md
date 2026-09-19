# Gama-G: implementation status

This document says, as precisely as it can, **what is actually implemented** in
this repository and **what is not**. It is the honest counterpart to
`Gama-G_v1.0_Production_Specification.txt`, which describes a multi-year,
multi-team production language. This repository is a working vertical slice of
it: a complete compiler front end, an intermediate representation, an
optimizer, and a reference interpreter, with the safety systems that define the
language — types, effects, capabilities, secrets, audit, recovery — enforced
end to end.

Everything claimed below is exercised by the test suite (`python3 -m unittest
discover -s tests`, 185 tests) and demonstrated by a runnable example in
`examples/`. Where a claim is partial, the missing part is named.

---

## 1. Summary

| | |
|---|---|
| Implementation language | Python 3.11+ (reference implementation) |
| Compiler source | ~14,500 lines across 28 modules |
| Standard library | 215 builtins across 18 modules |
| Language surface | 23 hard keywords, 84 contextual keywords, 9 effects |
| AST node types | 70 |
| GIR operations | 41 |
| Tests | 185 (all passing) |
| Examples | 8, each runnable with `ggc run` |
| Native backend | **not implemented** — see §5 |

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
  requires a reason and that reason is audited.
- **§9 Concurrency** — structured `parallel` regions. Tasks are ordered by
  dependency; independent ones run concurrently on real threads.
- **§9B Agents** — `agent` declarations with typed `on <event>` handlers and
  message dispatch. Agents share no mutable state. An event no handler declares
  is refused.
- **§10–11 Self-healing and checkpointing** — `service` with `protect`/
  `recover`, the six recovery levels in spec order, bounded retry, checkpoint
  capture and restore, operator escalation. Every recovery action is audited
  with its level, whether it succeeded, and what it did.
- **§12 Security** — `grant` headers; capability handles minted at run time and
  permission-checked on every access; no ambient authority.
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
  the transaction open.
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
- **§28 Standard library** — 18 modules, 215 builtins.

---

## 4. Toolchain (spec §31)

`ggc` implements five of the nine commands the spec lists:

| Command | Status |
|---|---|
| `ggc check` | **done** — diagnostics only, with `--profile` and `--json` |
| `ggc build` | **done** — compiles to GIR, `--stats`, `-O0/1/2` |
| `ggc run` | **done** — `--grant`, `--audit PATH`, `--entry`, `--lenient-runtime` |
| `ggc test` | **done** — `--json`; exit code 4 on failure |
| `ggc explain` | **done** — a diagnostic, a decision, or the standard library |
| `ggc audit` | **not done** as a subcommand; `--audit PATH` on `run` writes and verifies the trail |
| `ggc bench` | **not done** — see §6 |
| `ggc profile` | **not done** |
| `ggc format` / `ggc doc` | **not done** |

Exit codes: `0` success, `1` compile error, `2` runtime fault, `3` usage,
`4` test failure.

A test asserts that the unimplemented commands are *absent* rather than present
and broken, so they cannot silently start pretending to work.

---

## 5. What is not implemented

Named deliberately, because spec §43 forbids claiming capability the toolchain
does not have.

**No native backend.** There is no AOT compiler, no LLVM or WASM target, no
machine code. Programs execute on the GAEM reference interpreter
(`runtime/vm.py`), a tree-walking evaluator over GIR. Consequences:

- Performance is interpreter-grade. **No benchmark against native code has been
  run, and none is claimed.** Spec §24's "≤ 1.05× equivalent optimized native
  implementation median" is a target for a backend that does not exist here.
- `Tensor` operations use NumPy when it is importable and fall back to a
  pure-Python implementation when it is not, so the language has no hard
  dependency on it. The test suite passes either way; only the speed differs.
  Tensor-heavy code is therefore not interpreter-bound when NumPy is present,
  but everything else is.

**No package manager.** `gpm` does not exist. There is no dependency
resolution, registry, lockfile or vendoring. A program is a file or a directory
of files.

**No borrow checker.** §8's ownership model is implemented as immutability by
default plus secret-type propagation. There are no lifetimes, no move
semantics, no aliasing analysis and no use-after-free prevention (the
interpreter is garbage collected, so that class of defect cannot arise — but
neither is it *prevented by analysis*, which is what the spec describes).

**No pipeline stage fusion.** §3's operation-graph analysis exists for
`parallel` regions. `pipeline` declarations parse, type-check and run, but the
compiler does not yet fuse compatible stages or schedule them across devices,
which is what §38's fraud-detection example is meant to demonstrate.

**Sixteen standard-library modules are declared but not implemented**, and say
so rather than failing obscurely: `fhir`, `terminology`, `provenance`, `consent`,
`database`, `http`, `messaging`, `workflow`, `transaction`, `observability`,
`accelerator`, `process`, `concurrency`, `train`, `infer`, `identity_provider`.
Each carries a reason in `std/library.py::UNIMPLEMENTED_MODULES`, and a test
asserts every module the spec names is either implemented or on that list with a
non-placeholder reason.

**Interop (§29) is not implemented.** No C ABI, no Python embedding API beyond
importing the compiler as a library, no FFI.

**Formal verification (§27) is contracts only.** `requires`/`ensures` are
checked at run time and quoted in violations. There is no proof assistant, no
SMT backend, no static verification of the contracts.

---

## 6. Claims this implementation does not make

Spec §43 lists what Gama-G must never claim. These prohibitions are enforced
here in two ways: by not making the claims, and by a test
(`test_enforcement.ForbiddenClaims`) that scans every `.md`, `.py`, `.gg` and
`.txt` file in the repository for the forbidden phrases and fails if any appear.

- **No universal performance guarantee.** Nothing here is benchmarked against a
  native implementation. The interpreter is a reference, not a product backend.
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

Roughly in the order the spec's own phases imply, and ordered by how much of the
language's promise each unlocks:

1. **Pipeline stage fusion and scheduling** (§3, §38) — completes the
   operation-graph story, which is the language's central idea.
2. **A real backend** (§22 stages 11–12) — WASM first, because it gives
   portability and a sandbox boundary at once; native afterwards.
3. **Borrow checking** (§8) — the largest remaining gap between what the spec
   describes and what is enforced.
4. **`gpm`** (§30) — needed before any library ecosystem is possible.
5. **Fuzzing and property generation** (§26) — `test fuzz` is parsed as a
   category but generates nothing.
6. **FHIR and database modules** (§16, §17) — the two declared-unimplemented
   modules with the clearest demand.
7. **Benchmarking** (§24) — only meaningful once (2) exists.
