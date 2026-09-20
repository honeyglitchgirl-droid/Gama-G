# Gama-G: analysis, comparison, and what competition would require

This document answers the two questions the repository's own audits kept
deferring: *what is this thing, actually* -- and *how does it stand next to
the languages a team would otherwise pick*.  It follows the house rule
(spec section 43, enforced by `tests/test_enforcement.py`): nothing here is
a universal claim.  Where a comparison would become one, the comparison is
described structurally instead of numerically.

---

## 1. What Gama-G is

Stripped to its architecture, Gama-G is four ideas stacked:

1. **A declaration language about obligations.**  Effects, capabilities,
   contracts, secret flow and audit trails are declarations the compiler
   checks, not conventions the team keeps.
2. **A derived execution graph (the core).**  In the `gama core` dialect a
   program is a set of operations declaring what they consume and produce;
   order is computed by the compiler and *output*, never written by hand.
   Unbounded control flow does not exist: repetition carries a mandatory
   bound, so a core program that cannot be written as a hang cannot hang.
3. **Three formal models below the checker.**  Ownership/extents/secret
   custody, a capability algebra shared by checker *and* runtime, and a
   recovery policy lowered as a policy -- computed by the toolchain and
   printed by `ggc graph` / `ggc memory`.
4. **Verifiable artefacts.**  A hash-chained signed audit trail, reproducible
   signed builds, differential testing of every backend against the
   reference interpreter, and a fuzzer whose invariants are permanent tests
   (`compiles-or-explains`, `file-path-is-total`, `formatter-is-total`,
   `prover-is-sound`, …).

The whole is a **vertical slice of a specification** (`Gama-G_v1.0_Production_
Specification.txt`, Phases 0-7) -- a complete front end, one native backend
for a subset, four ways to run programs, and the toolchain around them.
`docs/IMPLEMENTATION.md` is the line-by-line status; this file is the
positioning.

### 1.1 The genuinely distinctive parts

Things that are *not* borrowed, measured against what mainstream languages
offer:

| Feature | Nearest mainstream analogue | What Gama-G does differently |
|---|---|---|
| Order as an output (`uses`/`yields` → derived graph) | build systems (Make, Bazel), dataflow languages | a *program* is the dependency graph; the unit is an operation with a type, a capability demand and a proof obligation, not a target with a command line |
| Mandatory loop bounds | Sparkle/Why3 termination checks; Rust has none | the bound is syntactically required, and the recovery graph enumerates every bound in the program as data |
| Guard selection proofs | Rust/OCaml match exhaustiveness is over *data shapes*, never over guard values | `core/guardproof.py` decides exhaustiveness and exclusivity exactly for integer-comparison guard sets (interval arithmetic over the type's bounds); floats stay honestly `unprovable` because of NaN |
| One capability algebra on both sides of the compile/run boundary | seL4/capsicum capabilities are OS-level; language capability systems (Joe-E, E) are research | the *same function* (`capabilities.py::covers`) answers "may I prove this call" and "may this runtime check pass"; attenuation can never amplify |
| Secrets as a type with custody states | no mainstream equivalent; `const`/private fields are visibility, not flow | `secret T` taints through computation; rendering needs `secrets.expose(value, reason)` under a granted `SecretExpose`, and the exposure is written to the trail |
| Audited recovery as syntax | Erlang/OTP supervision: process-level, policy in `restart: temporary` atoms | six escalation levels, named checkpoints, policy lowered *as* a policy, every action in the trail with level and outcome |
| Contract text quoted verbatim in violations | Dafny/SPARK prove instead of quoting; Python `assert` disappears with `-O` | the failure message is the program's own words at the position the compiler recorded -- and since v1.3 the promise is *discharged first* where the graph already entails it, with the reason it could not be discharged kept beside the clause |

No mainstream language ships all seven.  Several ship none.

### 1.2 The equally honest weaknesses

An analysis that stops at the features list is marketing.  The structural
weaknesses, each one checkable in this repository:

* **It is one implementation.**  Every guarantee is delivered by a single
  Python tree-walking interpreter (a native C backend for a subset aside).
  There is no second implementation, no committee, no independent
  conformance suite -- and the spec-vocabulary tests read the repository's
  own specification, which is self-consistency, not cross-validation.
* **No ecosystem.**  No packages exist beyond the in-tree ones; `gpm` works
  offline against directory registries.  Library gravity is the single
  biggest reason language comparisons are lopsided, and Gama-G is at zero
  here.
* **Performance is unproven and the reference is CPython.**  The native
  backend refuses the very constructs (capabilities, audit, tensors,
  transactions) that define the language -- so on the flagship programs the
  *only* execution model is the interpreter.  `docs/BENCHMARK_BASELINE.md`
  is honest that startup dominates every measurement in it.
* **The safety story has a runtime dependency.**  The native runtime does
  not enforce capabilities; it refuses programs that use them.  A
  compile-time proof plus a runtime that cannot check is the guarantee
  halving in strength exactly where speed would matter.
* **Two dialects, one grammar budget.**  v0.1 and core share a pipeline
  (v1.0) but not a soul: the core has no modules, no generics, no
  user-defined types; v0.1 is the conventional language the audits judged
  "not original".  Maintaining both is real cost.
* **Contract verification is partial by construction.**  `core/contractproof.py`
  discharges a `holds` over constants or over one sized-integer binding under
  its guard; it refuses division, library calls, `state`, refined bindings and
  any covering argument over a float, and it never deletes the runtime check.
  There is no interprocedural analysis, so v0.1 `requires`/`ensures` remain
  runtime-only, and the SMT-LIB2 export is a question handed to a solver this
  toolchain does not run.
* **No IDE story.**  There is no language server, no incremental parse,
  and the parser's nesting limit (≈52 levels) is a compiler-robustness
  virtue that is an editor latency nightmare if anyone ever wraps it in a
  server naively.

---

## 2. Against the languages a team would actually pick

The question is never "is Gama-G better than Rust" in the abstract; it is
"for a system where a silent mistake is expensive, what do I give up to
gain what".  Structured, not scored.

### 2.1 vs Rust

| axis | Rust | Gama-G |
|---|---|---|
| memory safety | borrow checking over all of `unsafe`-free code; lifetime inference across function boundaries | ownership *model* for the core (extents, slots, one owner per binding, proven slot reuse) + immutability by default elsewhere; no borrow checker over arbitrary mutation |
| what is statically excluded | data races, use-after-free, null, unhandled error (in style) | unbounded loops, silent `Result` drops, undeclared effects, ambient capability use, secret flow to renderers, *guard-infeasible selection cases* |
| proofs vs tests | the compiler proves memory properties; everything else is a test | the compiler proves a different, narrower property set; the runtime re-checks capabilities; violations quote themselves |
| errors | `Result`, `?`, huge ecosystem convention | `Result`/`Option` with a *checked* discard rule, plus contracts with quoted text |
| effect system | none (async trait still settling) | nine effects, declared, checked, `pure` verified |
| performance | LLVM, decades of optimization | reference interpreter; a subset-compiling C backend |
| ecosystem | crates.io, tens of thousands of maintainers | one repository |

Rust wins execution speed, memory-safety coverage, and (overwhelmingly)
ecosystem.  Gama-G's advantage is *domain-adjacent* static truth: a Rust
program with an audited medical workflow writes the audit, the reason
strings, the retry policy and the loop bounds as *comments plus discipline*;
the Gama-G program writes them as syntax the checker refuses to skip.  If
the property you need to hold is "no unbounded loop exists anywhere in this
binary", Rust cannot answer; the core can.

### 2.2 vs Go

Go's bet is that a small language with one obvious way to do things, a
garbage collector and channels scales better than a prover.  Gama-G agrees
about one formatter (v1.2: `ggc format`, same-GIR-checked) and disagrees
about the rest.  Go gives: production backends everywhere, the goroutine
model, and `error` values nobody forces you to handle.  Gama-G gives:
mandatory handling of `Result`, declared effects, capability authority,
structured *recovery with escalation levels and checkpoints* (Go has
`defer`/`panic` as a convention, not an audited state machine), and
termination as a type-level property.  On deployment, Go's static binaries
with a GC beat a CPython toolchain flatly; a native Gama-G binary exists
only for the construct-free subset.

### 2.3 vs Python

The reference implementation *is* Python, which makes the comparison
instruction-by-instruction fair on one axis: expressiveness per runtime
guarantee.  A Django service trusts: every function can do any I/O (no
effect tracking), every import grants ambient authority to the filesystem,
`except Exception: pass` can swallow an audit failure, loops are unbounded
by construction, and a NaN-poisoned comparison silently reorders a triage
branch.  Gama-G's v0.1 surface runs in the same 30k-line interpreter
Python-style, but *checks* those five things; the core dialect rewrites
them away syntactically.  Python's counterweight is everything else in the
universe: NumPy to CUDA, a decade of ML, a million packages.  Gama-G is
not competing with Python for general workloads; it is competing for the
"the service that must be explainable to an auditor" slice of them.

### 2.4 vs TypeScript

TypeScript proved that a gradual checker over an existing ecosystem is a
winning adoption strategy, and then shipped: `any` (an escape hatch Gama-G
refuses to need, but has via v0.1 `Any` in capability handles -- declared
in `docs/PRODUCTION_GAPS.md`), structural typing (Gama-G is nominal), and
no runtime guarantees whatsoever ("types vanish at runtime" is the motto).
Gama-G's semantics survive to runtime -- the capability check runs, the
secret guard runs, the contract quotes itself when violated.  Where TS has
`tsconfig.json` strictness flags as a team negotiation, Gama-G bakes
strict/standard/lenient in and makes the *strict* profile the default for
diagnostics but not for correctness properties: a discarded `Result` is an
error in strict, a warning elsewhere -- the same negotiation, smaller
surface.

### 2.5 vs Zig, and "systems without the borrow checker"

Zig's position -- small language, comptime as the universal macro,
explicit allocators, `no-Runtime` -- is the nearest philosophy to Gama-G's
core in current systems programming: both shrink what a program *may be* to
keep the semantics legible.  Zig answers memory with explicit allocators
and AddressSanitizer at test time; Gama-G answers it with ownership extents
computed at compile time.  Zig has no effect system, no capabilities, no
audit; its comptime is more powerful than anything here (it is a full
second language), and its generated code is competitive because LLVM is.

### 2.6 vs Ada/SPARK, the closest historical relative

Gama-G's real lineage is Ada.  SPARK's `Global`/`Depends` contracts, ghost
state, flow analysis, and runtime "checking modes" (prove / check / ignore)
are the same three-way honesty this repo formalizes as
`proven`/`runtime`/`unprovable` -- and SPARK still has twenty years of
DO-178B artifacts, a certifiable toolchain, and GNAT-backed code generation.
What Gama-G adds over SPARK is the program *shape* (derived order, bounded
repetition as the only loop), capabilities as first-class typed handles,
and the audit trail as a language artifact rather than a logging practice.
What SPARK adds is: it exists in the world, provably, with proof
obligations discharged by SMT backends.  Gama-G's proofs were purely
structural until v1.3 (guard intervals, graph properties, slot overlaps) and
its `holds` clauses were quoted at failure rather than checked at build.  The
gap has narrowed and remains: `core/contractproof.py` decides a `holds` over
constants or over one sized-integer binding under its guard, and exports the
rest as SMT-LIB2 for a solver Gama-G does not run; SPARK's `depends` clauses,
pointer freedom and interprocedural proofs are still theirs alone, and v0.1
`requires` are still runtime-checked.  No document can argue the rest away: it
needs a solver, and then a call-site analysis behind it.

### 2.7 vs Koka / effect systems, and Lean / Coq

Koka demonstrates that algebraic effects can be a *mainstream-shaped*
language feature with real compilers; Gama-G's effects are coarser (sets
of names, checked for declaration, not inferred to kinds in v0.1; the core
declares them per operation) -- Koka's are finer and its handler story is
richer, while its handling of capabilities, audit and secrets is: none.
Against the proof assistants, the relationship is the opposite: Lean
discharges goals no one is typing at that level of ceremony for
application code; Gama-G's bet is that the *useful* 20% (guard
partitions, capability coverage, loop bounds, dependency honesty, secret
flow) can ride in the syntax of an ordinary-looking language.  A future
the export from `holds` obligations to SMT or Lean (spec §27's "formal
verification is contracts only") is the bridge neither side has built -- v1.3
laid the first plank, an SMT-LIB2 `QF_LIA` export of every obligation the
built-in prover could not settle, and stopped short of invoking a solver or
pretending a `; not encoded` comment is a proof.

### 2.8 vs Erlang/Elixir

For *self-healing services specifically*, OTP is the incumbent:
supervision trees, let-it-crash, hot upgrade.  Gama-G's recovery levels map
directly (restart = level 3, escalate to operator = level 5) but add what
OTP does not: the recovery is audited as a first-class trail, checkpoints
are *named and language-scoped* rather than process-state by convention,
and recovery is bounded by syntax (a `retry` without a bound is a compile
error; `recover.gg`'s policy levels may not decrease, checked at compile
time).  Erlang counters with twenty-five years of battle-tested scheduler
engineering, distribution, and a whole platform.  Gama-G's story here is
"OTP's ideas, made statically checkable and forensically legible", not "a
BEAM replacement".

### 2.9 Summary matrix

| | Rust | Go | Python | TypeScript | Zig | Ada/SPARK | Koka | Erlang | **Gama-G** |
|---|---|---|---|---|---|---|---|---|---|
| memory errors stopped statically | ● | ○ (GC) | ○ (GC) | ○ (GC) | ◐ (explicit) | ● | ○ | ○ | ◐ core model |
| effects declared + checked | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ (Global) | ● | ✗ | ● |
| capability-based authority | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ◐ | ○ (sandboxes) | ● |
| loops provably terminating | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ (variant) | ◐ (totality exp.) | ✗ | ● core |
| data-race freedom | ● | ○ (CSP by convention) | ◐ (GIL) | ◐ | ◐ | ● | ○ | ● | ◐ (graph, GIL) |
| `Result`-discard an error | ◐ (must_use) | ✗ | ✗ | ✗ | ◐ | ● | ● | ✗ | ● |
| secrets as a flow-checked type | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ (flow in SPARK) | ✗ | ✗ | ● |
| audited trail in-language | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ✗ | ◐ (lager) | ● |
| recovery as syntax with levels | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ● |
| contracts statically discharged | ✗ | ✗ | ✗ | ✗ | ✗ | ● | ✗ | ✗ | ◐ core `holds`: constants + one-interval guards; v0.1 runtime-checked |
| one canonical formatter | ● rustfmt | ● | ◐ (black, not universal) | ◐ prettier | ● | ◐ | ◐ | ✗ | ● (same-GIR proof) |
| native performance backends | ● LLVM | ● | ◐ (depends) | ◐ (JS) | ● LLVM | ● GNAT | ◐ | ● BEAM | ◐ subset C |
| package ecosystem | ● crates | ● modules | ● PyPI | ● npm | ◐ | ◐ | ○ | ● hex | ○ none |

● full, ◐ partial, ○ by platform/convention, ✗ absent.  The table says what
a careful reader already suspects: Gama-G has the most *checked properties
per syntax token* of any of them, and the least *world* of any of them.

---

## 3. What competition would require, in order of return on effort

The gap is not ideas -- several columns above are things competitors do not
have.  The gap is *distribution of the guarantees to real workloads*.  In
order:

1. **A native backend that covers the security constructs.**  Capabilities,
   audit, contracts and transactions refused-by-name are the difference
   between "a language for safety-critical systems" and "a reference
   implementation of one".  It starts where `cgen.py` is: implement the
   audit chain and `CAP_CHECK` in the C runtime (SHA-256 + HMAC are already
   specced in the Python side), keep the refuse-by-name discipline for the
   rest.  *Open; the largest single item.*
2. ~~**Static discharge for `requires`/`ensures`.~~  **The core's `holds` is
   done in v1.3** (`core/contractproof.py`): closed-form evaluation over the
   constants the graph fixes, interval implication for one sized-integer binding
   under its own `when` or its type's range, `W-contract-refuted` when the
   constants contradict the promise, and an SMT-LIB2 export of what is left.  No
   proof removes a runtime check.  What remains is the wider half: interprocedural
   discharge of v0.1 `requires`/`ensures` (they are verbatim clause text on a
   function, so this needs a call-site pass) and division/calls/state in the
   prover's fragment.  *Partly open.*
3. **A language server.**  The checker already accumulates every diagnostic
   in a file rather than stopping at the first -- that is a half-written LSP
   backend.  Nobody adopts a language their editor cannot talk to.
   *Open.*
4. **Core generics and composition between intents.**  Without them the
   core cannot express shared libraries, and the language's best ideas
   stay confined to single-intent programs. *Open (roadmap 1a).*
5. **Modules and imports across the whole language** (v0.1 has `import`
   syntax parsed; the core has none) -- the precondition for any ecosystem.
   *Open.*
6. **A second implementation, or a conformance suite the world can run.**
   The spec-vocabulary tests reading `Gama-G_v1.0_Production_Specification.txt`
   is the seed: an executable conformance format over that same document
   would let anyone write implementation two.  *Open.*
7. ~~**The toolchain's front-of-house: format, doc, profile, bench,
   offline audit verification.**~~ **Done in v1.2** -- `ggc format` with the
   same-GIR safety proof, `ggc doc`, `ggc profile`, and `ggc audit verify`.
8. ~~**A package manager that resolves like a solver, not a greedy pass.**~~
   **Done in v1.2** -- bounded deterministic backtracking, fail-closed past
   the budget.  The network and a public registry remain out of scope by
   design until signing is production crypto.
9. **Constant-time crypto** (or an audited library behind a feature flag);
   Ed25519 in pure Python disqualifies the audit chain from adversarial
   timing settings today, and `docs/PRODUCTION_GAPS.md` says so. *Open.*
10. **Releases, tags, and the boring metadata.** A language is installed,
    versioned and pinned by CI before it is loved. *Open.*

## 4. The honest bottom line

Compared as it stands, Gama-G loses to Rust, Go, TypeScript, Zig, Erlang
and Ada on the two axes that decide adoption -- execution performance and
ecosystem -- and beats all of them on the axes its specification chose:
declared effects, capability authority, provable termination, secret flow,
audited recovery, quoted contracts, derived order, guard-completeness
*decision procedures*, and tooling whose output is machine-verified against
the compiler's own semantics.  Its current form is the best possible
*reference implementation* of that idea set: 628 tests, differential
backends, permanent fuzz invariants, and documentation that says what it
does not do.

Whether it becomes competitive depends on items 1 and 3 -- **native enforcement
and a language server**.  Item 2, real proofs for the contracts, started in
v1.3: the core's `holds` is now discharged where it is decidable, which is
exactly the half of the argument a reference compiler can carry alone; the rest
of it is a solver, an ecosystem to run one over, and a second implementation.  Those three move the
same guarantees from "checked in a reference interpreter" to "enforced in
production and felt while typing" -- and production enforcement plus editor
feel is exactly where the incumbent languages are weak or absent, which is
the only ground on which a new language wins.
