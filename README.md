# Gama-G

A programming language for systems where a silent mistake is expensive: AI/ML
pipelines, medical software, enterprise systems, cybersecurity controls and
self-healing services.

The design goal, from the specification, is **less code, more operations**. A
Gama-G program says what it needs and what it promises, and the compiler holds
it to both:

```gamag
grant PatientRead, Read, CryptoSign, SecretExpose, AuditWrite

fn readPatient(store: PatientStore[Read], key: Text) -> Result<Text, Text>
    io
    audit
    let rows = store.rows
    if not contains(rows, key)
        return fail("no such patient")
    audit.record {
        actor: "clinical-app"
        action: "PATIENT_READ"
        object: key
        reason: "direct clinical care"
    }
    return ok(rows[key])
```

Four guarantees are visible in those lines, and none of them are conventions:

- **`PatientStore[Read]`** — authority is a parameter. A caller without a `Read`
  handle cannot call this function at all.
- **`Result<Text, Text>`** — the failure case is part of the type. Discarding
  this `Result` is a compile error, not a dropped exception.
- **`io` and `audit`** — effects are declared. A `pure` function that prints is
  rejected.
- **`audit.record { … }`** — the access is written to a hash-chained, signed,
  append-only trail, with a reason.

---

## v0.2: the original language core

The example above is the v0.1 surface. An
[audit of this repository](Gama-G_Detailed_Audit_and_Verification_Report.txt)
found the implementation real and working but the *language* insufficiently
original: `fn`, `let`, `var`, `if`, `while`, `match` are other languages'
constructs, and its recommendation was to redesign the core rather than rename
them.

That redesign is implemented. A **Gama-G v0.2 core** program is not a sequence
of statements — it is a set of operations that declare what they consume and
what they produce, and the compiler **derives** the execution order:

```gamag
gama core 0.2

intent SafeDosing
    purpose    compute one paediatric dose and be able to show why
    authority  PatientRead, AuditWrite
    trail      dose decision

source weight : F64 from 34.5
source factor : F64 from 15.0

operation RawDose
    uses     weight, factor
    yields   rawDose : F64
    effect   pure
    holds    rawDose >= 0.0
    computes weight * factor

operation Dose
    uses     rawDose
    yields   dose : F64
    effect   medical
    holds    dose <= 500.0
    computes math.clamp(rawDose, 0.0, 500.0)
    trail    dose decision

outcome dose
```

Nothing there says "first RawDose, then Dose". Ask the compiler what it derived:

```sh
$ ggc graph examples/core/dose.gg
intent SafeDosing
  purpose    compute one paediatric dose and be able to show why
  authority  PatientRead, AuditWrite
  outcome    dose

derived execution order (never written in the source):
  level 0: RawDose
  level 1: Dose
```

What the core removes, and what replaces it:

| Gone | Replaced by | The consequence |
|---|---|---|
| assignment | single-assignment bindings | a binding has one definition and one lifetime |
| `if`/`else` | sibling operations yielding one binding under complementary `when` guards | the compiler **proves** the alternatives exclusive and exhaustive, and says so in `ggc graph` |
| `while`, unbounded `for` | `refine … until … within N rounds` | the bound is mandatory, so a program that cannot be written as a hang cannot hang |
| `return` | the `yields` binding, and the intent's `outcome` | an operation's value is a relationship, not a jump |
| written control flow | `uses`/`yields` matched into a graph | a dependency that is not real cannot be written down, and one that is real cannot be forgotten |

The relationships are checked in both directions: referring to a binding without
declaring it in `uses` is `E-undeclared-relationship`, and declaring one you
never use is a warning. Without that check the derived order would mean nothing.

[`docs/DESIGN_v0_2.md`](docs/DESIGN_v0_2.md) is the design document. It maps the
audit's proposed execution model onto the actual modules, answers its
ten-question originality test feature by feature — including the three places
where the honest answer is "partly yes", with the technical reason — and
separates what is proven at compile time from what is only checked at runtime.

v0.1 is not deleted. Its tested type lattice, standard library, GIR, optimizer
and reference interpreter are what actually run the core's promises.

---

## v0.3: a native semantic IR

A [second audit](Gama-G_Complete_Originality_and_Technical_Audit.txt) accepted
the language and found a problem underneath it. v0.2 compiled by *elaborating*
the core into the older implementation's abstract syntax — so `let`, `var`, `if`,
`while` and `match`, the constructs the core exists to avoid, reappeared one
layer down as the intermediate representation. Its words:

> the compiler still translates the new surface into the old model before doing
> anything with it

v0.3 removes that layer. The core now compiles through a **native semantic IR**
(`compiler/gamag/core/mir.py`) that represents operations, dependencies,
constraints, authority, transitions, refinement and outcomes directly:

```
core source → CoreParser → SemanticModel → NativeChecker → Lowerer → GIR → VM
                              │
                    five inspectable graphs:
                    intent · operations · constraints · authority · recovery
```

`core/ast.py` and `core/elaborate.py` are deleted from the tree, so nothing can
fall back to elaborating. The property is asserted, not just claimed:
for a core program `compilation.module` — the older AST — is `None`.

What else changed:

* **Blocks are named after core concepts.** The GIR for a `refine` reads
  `refine.G.until`, `refine.G.round`, `refine.G.diverged`, `refine.G.next`,
  `refine.G.done`. A fan-out reads `each.X.item` and `each.X.skip`.
* **Faults are terminators, not calls.** A diverging refinement emits a GIR
  `FAULT` with `kind = "RefinementDiverged"`, so the classification is data
  rather than a string prefix to be parsed.
* **The derived graph survives into the GIR.** Each intent function carries
  per-task reads, writes and dependencies, which is what a parallel backend
  would need — and which the emission does not yet use.
* **Constraints record how they are discharged**: `proven`, `runtime`, or
  `unprovable`. The third value exists so that guard exhaustiveness beyond two
  alternatives is reported as something the compiler cannot prove, rather than
  quietly assumed.

The language surface is unchanged — `gama core 0.2` still parses — because v0.3
changed how the core is *compiled*, not what it says. The pragma names the
dialect; the toolchain has its own version (`VERSION`).

[`docs/DESIGN_v0_3.md`](docs/DESIGN_v0_3.md) is the design document: the IR, the
five graphs, the 33 diagnostics, the lowering patterns, and the three categories
of claim kept strictly apart — proven at compile time, checked only at runtime,
and not checked at all.

---

## v0.4: the formal semantics of the core

The same audit sequences the rest of the work, and its priorities 4, 5 and 6 are
the three models the specification writes down formally but that v0.3 could only
describe in prose — *explicit memory/resource model*, *formal capability
semantics*, *direct transition/recovery representation*. v0.4 builds all three,
so the properties are computed by the toolchain rather than asserted in a
document.

```
core source → CoreParser → SemanticModel → NativeChecker → Lowerer → GIR → VM
                              │                 │
                     five inspectable     three formal models:
                     graphs (v0.3):       memory · capability · recovery
                     intent · operations
                     constraints · authority · recovery
```

**Memory (priority 4, spec §8).** Every binding gets exactly one owner, an extent
`[first, last]` in the derived order, and a slot. A slot is shared only where two
extents provably do not overlap, and `slots_saved` counts the reuse the allocator
*earned* rather than assumed — a two-operation chain reuses nothing, because
there every value is live at the same time as its neighbour. Secrets are marked
where they enter, guarded by a `SECRET_GUARD` instruction, and an ordinary
renderer refuses them. The only way a value leaves is
`secrets.expose(value, reason)` under `SecretExpose`, which the audit log records
as `SECRET_EXPOSED` at security level. `ggc memory <file>` prints the model, its
slots and any violation.

**Capability (priority 5, spec §12).** One algebra — `compiler/gamag/capabilities.py`
— is used by *both* the checker and the runtime, because a guarantee the compiler
proves and the runtime does not enforce is a guarantee about a program that cannot
be run. A capability is a permission over a resource, so `PatientStore[Write]`
covers `PatientRead` and nothing crosses resources. Grants attenuate and never
amplify: two reads on different resources are still only two reads. What a node
needs is *derived* from what it calls, so a `CAP_CHECK` boundary is emitted even
for a demand the compiler already proved — the proof is about the program, not
about the environment it is deployed into.

**Transition and recovery (priority 6, spec §10 and §18).** `recover` and
`checkpoint` are core syntax, and a declared policy is lowered *as a policy*: a
`PROTECTED` region carrying those very steps in the specification's own words, at
its own levels (0 local retry … 5 operator escalation), which may not decrease.
A named checkpoint is captured before the work runs, so `restore checkpoint
admission` returns to `admission` and never to whatever was recorded most
recently; restoring to a checkpoint the intent never declared is a compile error,
because §10 forbids silently inventing state. Every transition runs inside a
`TRANSACTION`, so a constraint that fails in the commit phase is recorded as an
abort instead of being left half-applied.

Two examples were added for it: [`custody.gg`](examples/core/custody.gg) follows
one secret through its whole lifecycle, and
[`recover.gg`](examples/core/recover.gg) declares a policy and runs it. Change
`recover.gg`'s `reading` to `500` and the audit trail shows the policy escalate
for real — retry, restore to the named `admission` checkpoint, operator alert,
`RecoveryExhausted` — with every action recorded in the program's own words.

[`docs/DESIGN_v0_4.md`](docs/DESIGN_v0_4.md) is the design document: the three
models, what each one proves, where the same relation is used on both sides of
the compile/run boundary, and what remains description rather than code.

Priority 3 (a native CPU backend) and priorities 7–16 are still untouched.

---

## v1.0: one language, one version

Earlier milestones built the language's two surfaces separately, and the
toolchain showed it: a core file could not contain a function, and a core
expression could not call one.  That split is gone.  Both declaration families
now live in one file and share a lexer, a parser, a checker and a pipeline:

```gamag
gama core 0.2
intent Rounding
    purpose   round a measurement to a whole number

source reading : F64 from 5.4

fn round_half_up(x: F64) -> I64          // a helper
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
declarations, and its GIR is merged into the module the core produced.  By the
time anything downstream sees the program there is one module with one optimizer
and one set of backends.

This release also adds the rest of the second audit's priorities: a **native CPU
backend** with **differential testing** against the interpreter, a
**WebAssembly** encoder, an **accelerator** device layer, **benchmarking**,
**fuzzing** as a permanent tool, **reproducible signed builds**, a **package
manager**, an **FFI** to the C ABI, and the **medical** and **enterprise**
library modules.

---

## Status

This repository is a **working vertical slice** of
[`Gama-G_v1.0_Production_Specification.txt`](Gama-G_v1.0_Production_Specification.txt),
not a finished implementation of it.  The specification describes a multi-year,
multi-team production language across Phases 0-7.

What is here: one language, one pipeline, and four ways to run it.  The compiler
lexes, parses, derives the semantic graph, checks types, effects, capabilities
and secret flow, builds the three formal models of v0.4, and lowers to GIR; after
GIR the same machine serves the reference interpreter, a native CPU backend that
emits C and compiles it to machine code, a WebAssembly encoder, and an
accelerator layer.

Those tests compare the toolchain with itself.  `ggc conform` compares it with
the blueprint: it reads `Gama-G_v1.0_Production_Specification.txt` at run time
and checks section 5's type names, section 22's fourteen pipeline stages,
section 33's twenty v1.0 requirements, and programs the prose implies.  It
passes 13 claims and reports 3 places where the toolchain knowingly differs
from the document (`conformance/deviations.json`); `ggc conform --strict`
reports those three plus the ten requirements that are `partial` or
`not-claimed`, and is the bar `docs/RELEASES.md` sets for leaving Alpha.  A
deviation that is not recorded is a failure, so the gap can only get smaller by
being closed or larger by being written down -- never by being forgotten.

**642 tests pass**, and CI runs them on every push and pull request across
Python 3.9 - 3.13 (`.github/workflows/ci.yml`).  The native backend is
validated by *differential testing*:
the same program is run on the interpreter and on the compiled binary, and their
stdout, exit status and fault kind are compared.  Where the two could differ --
integer range checks, float formatting, variadic output -- the C runtime
reproduces the interpreter rather than approximating it.

**The first release is out: [`v1.2.0`](https://github.com/honeyglitchgirl-droid/Gama-G/releases/tag/v1.2.0).**
Pushing the tag ran `.github/workflows/release.yml`, which checks that the tag
matches `VERSION`, that the suite passes and that every conformance claim
passes, installs the built wheel into a clean environment and makes it compile a
native program, and publishes the wheel, the source distribution, a reproducible
build manifest and checksums -- with the conformance report and the `--strict`
list attached, so what the release does not evidence travels with it instead of
expiring with a CI log.  The release notes are generated from that run by
`tools/release_notes.py` rather than written beside it.  `docs/RELEASES.md` says
what a version number here does and does not promise, including that the
manifest is `unsigned` until the repository holds a signing key -- it says so
rather than generating a throwaway one.

What is not: **the native and WebAssembly backends cover a subset.**  Programs
using the audit chain, capabilities, transactions, checkpoints, tensors,
autodiff or agents are *refused with a reason naming the construct* and run on
the interpreter instead -- never miscompiled.  Four of the sixteen shipped
examples compile natively; twelve are refused; none diverges.  There is no
WebAssembly runtime here, so no module has been executed, and no accelerator
here, so no kernel has been run.

There is **no wasm runtime** in this environment, so no module this toolchain
emits has ever been executed; `ggc wasm` says so when it writes one.  There is no
accelerator either, so no kernel has been run: `ggc device` reports the CPU as
the device that ran the work rather than falling back silently.

**Input is bounded, not crashed on.**  Syntactic nesting and tree depth are
limited by what the host stack can carry -- about 52 levels of nesting and 256
levels of tree depth on a default CPython host -- and exceeding either is a
diagnostic naming the limit, never a Python traceback.  Call depth is *not*
host-derived: the interpreter keeps its activations on an explicit stack, so a
Gama-G frame costs no host frame and the limit is the language's own ceiling of
1500 frames.  A file that is not UTF-8 is a diagnostic too.

**The native arena releases, where it can prove that is safe.**  A function
that cannot hand an allocated value on -- no heap return, no global store, no
call, no container mutation -- marks the arena on entry and releases before it
returns, so a loop through such a function does not grow.  Measured with the
runtime's own counters: 368 bytes peak at both 2 000 and 200 000 iterations,
against 7.3 MB at 20 000 through a function that stores a global and is
therefore not released.  `ggc native` prints how many functions are covered.

**The native backend emits machine types for the functions it can.**  A
function whose values are all scalars, that names no container and calls no
method, is emitted with `int64_t`/`double` slots and the arithmetic inline,
instead of as a 16-byte tagged union whose operator is selected by comparing the
operator's name with `strcmp` at run time.  Both emitters stay in the binary --
`ggc native` prints how many functions get the unboxed one and names the rest --
and a test builds every program in a corpus both ways and requires the two
binaries to print the same thing, which is how `not true` returning `true` was
caught.  Measured on one x86-64 machine, minimum of 5 runs, in one session, and
recorded with its conditions in [`docs/COMPARISON.md`](docs/COMPARISON.md): a
30-million-iteration integer loop went from 1046.97 ms to 20.75 ms (C with
`gcc -O2`: 1.63 ms), a call-heavy workload from 2462.08 ms to 78.65 ms (C:
61.12 ms), and an allocation-heavy one did not change, because the function
holding the loop uses `Text` and is not eligible.

**No performance claim** is made anywhere.  `ggc bench` measures and reports the
conditions of the measurement, and refuses to compare two runs whose conditions
differ, and every number above is a measurement of three programs on one machine
rather than a property of the compiler.

For the honest list of what this toolchain is *not* -- the defects found by
auditing it, the specification surface that is still missing, and the process
gaps -- see [`docs/PRODUCTION_GAPS.md`](docs/PRODUCTION_GAPS.md).

---

## Quickstart

No installation. Python 3.11+ is the only requirement; NumPy is used when
present and there is a pure-Python fallback when it is not.

```sh
git clone https://github.com/honeyglitchgirl-droid/Gama-G
cd Gama-G

./tools/bin/ggc run examples/core/dose.gg      # run a v0.2 core program
./tools/bin/ggc graph examples/core/dose.gg --edges  # the derived order
./tools/bin/ggc run examples/hello.gg          # run a v0.1 program
./tools/bin/ggc check examples/hello.gg        # diagnostics only
./tools/bin/ggc test examples/property_tests.gg # run its tests
./tools/bin/ggc gir --fn main examples/hello.gg # inspect the IR

python3 -m unittest discover -s tests           # the compiler's own suite
./tools/bin/ggc conform                         # the specification's own suite
./tools/bin/ggc conform --strict                # what is not evidenced yet
.venv/bin/python -m pytest tests/ -q            # the same suite under pytest
```

A file is core when its first tokens are `gama core <version>`; otherwise it is
v0.1. Detection is on tokens, so a comment or a string cannot change a
program's dialect.

`ggc` exit codes: `0` success, `1` compile error, `2` runtime fault, `3` usage
error, `4` test failure.

---

## The examples

Sixteen programs, each runnable and each covered by tests that assert on their
output rather than merely on their exit code.

### The v0.2 core — `examples/core/`

Eight programs: one per construct family, and two for the v0.4 models.

| Example | What it demonstrates |
|---|---|
| [`dose.gg`](examples/core/dose.gg) | Two operations, a derived order, constraints, an audited trail |
| [`selection.gg`](examples/core/selection.gg) | Guarded sibling operations replacing `if`; the guards are proven exclusive and exhaustive |
| [`converge.gg`](examples/core/converge.gg) | `refine … until … within 60 rounds`: bounded repetition, converging on √2 |
| [`traverse.gg`](examples/core/traverse.gg) | `each … over X as x`: fan-out as a graph node |
| [`classify.gg`](examples/core/classify.gg) | `resolve … choose`: exhaustive dispatch |
| [`ledger.gg`](examples/core/ledger.gg) | `state` + `transition`: the only mutable resource, under authority, in the commit phase |
| [`custody.gg`](examples/core/custody.gg) | A secret's whole lifecycle: guarded, unrenderable, and disclosable only under `SecretExpose` with a reason and an audit record |
| [`recover.gg`](examples/core/recover.gg) | `checkpoint` + `recover` as a lowered policy: named restore, escalating levels, a transition in a transaction |

### The v0.1 reference surface — `examples/`

| Example | What it demonstrates | Spec |
|---|---|---|
| [`hello.gg`](examples/hello.gg) | `let`/`var`, loops, `Result`, guarded `match`, errors as values | §4–6, §19 |
| [`medical_dosing.gg`](examples/medical_dosing.gg) | Contracts, records, enums with payloads, an audited clinical calculation | §16, §27 |
| [`parallel_pipeline.gg`](examples/parallel_pipeline.gg) | `parallel` regions with real dependency analysis; the `pipeline` form | §3 |
| [`security_audit.gg`](examples/security_audit.gg) | Capability handles, the secret lifecycle, signing, the hash chain | §8, §12, §13 |
| [`self_healing_service.gg`](examples/self_healing_service.gg) | `service` / `protect` / `recover`, bounded retry, checkpoints | §10, §11 |
| [`policy_transaction_agent.gg`](examples/policy_transaction_agent.gg) | Explainable policy decisions, all-or-nothing transactions, agents | §9B, §17, §18 |
| [`train_linear_model.gg`](examples/train_linear_model.gg) | Tensors, reverse-mode autodiff, AI-safety `require` gates | §14, §15 |
| [`property_tests.gg`](examples/property_tests.gg) | `test`, `test property`, `for all … where …` | §26 |

The training example fits `y = 2x + 1` from `w = 0, b = 0` and lands on
`w ≈ 2.000`, `b ≈ 0.999` — the language's own autodiff, not a library call.

---

## What the compiler refuses

The point of the language is that classes of defect become compile errors. Each
of these is a test:

```gamag
let x = 1
x = 2                          // E-immutable: values are immutable by default

print(secretValue)             // a secret cannot be printed, concatenated or audited

fn f() -> Unit
    pure
    print("x")                 // E-effect-pure: a pure function cannot do I/O

capabilities.open("S", ["Write"])   // E-capability-missing: not granted

f()                            // E-unused-result: a Result must be handled

xs.push(2)                     // E-discarded-value: returns a new list; this does nothing

match maybeBool
    true => return 1           // E-match-exhaustive: missing false

let small: I8 = 200            // outside I8's range

f(1)                           // E-arg-type: no implicit Int -> Float
                               //   help: write `float(x)` explicitly
```

The v0.2 core refuses a different set, because a different set is expressible:

```gamag
refine G
    starts  a
    repeats g + 1
    until   g > 100            // E-unbounded-refinement: `within` is mandatory
                               //   help: write `within 50 rounds`

operation C
    yields   c : I64
    computes b + 1             // E-undeclared-relationship: refers to `b`
                               //   without declaring it in `uses`

operation X
    uses     a, yv             // E-graph-cycle: X -> Y -> X, named in full
    yields   xv : I64
    computes a + yv

operation Q
    yields   s : I64           // E-ambiguous-selection: two producers of `s`,
    computes 2                 //   and this one has no `when` guard

operation B
    uses     st                // E-compute-after-commit: `st` is altered by a
    yields   b : I64           //   transition; compute may depend only on data
    computes st

transition T
    alters   b                 // E-transition-target: only `state` changes;
    computes 1                 //   bindings are single-assignment

operation Z
    effect   telepathy         // E-unknown-effect: the algebra is the spec's

let x = 1                      // not a declaration: "there are no statements"
```

At run time, faults are classified with source positions rather than surfacing
as host-language tracebacks: `DivideByZero`, `ContractViolation`,
`CapabilityViolation`, `RecoveryExhausted`, `AssertionFailed`, and from the core
`NoActiveAlternative` and `RefinementDiverged`.

Contracts quote themselves when they fail, in both dialects — the core keeps
the verbatim text of every `holds`, `when` and `until` clause precisely so that
it can:

```
runtime fault [ContractViolation] at dosing.gg:12:5:
  requires contract violated: weight > 0

Panic: holds `dose <= 500.0`
Panic: RefinementDiverged: `Guess` did not satisfy `g > 1000000` within 4 rounds
Panic: NoActiveAlternative: nothing yields `s` — 2 guarded alternatives, none active
```

---

## The audit trail

Every capability issued, policy decision, transaction and recovery action is
written to an append-only, hash-chained, signed log.

```sh
./tools/bin/ggc run --audit /tmp/trail.jsonl examples/security_audit.gg
# audit log written to /tmp/trail.jsonl (3 records, chain valid)
```

Each record carries `seq`, `actor`, `action`, `object`, `reason`, `timestamp`,
`program_version`, `prev_hash`, `hash` and `signature`, chained from a zero
genesis hash. Editing a field, editing an action, or deleting a record from the
middle is detected — all three are tested.

Execution is reproducible: the same program under the same seed produces the
same RNG stream and the same chain head on every run. `--lenient-runtime` opts
out.

---

## Recovery is bounded

A `service` declares what it protects and how it recovers. The six levels are
the specification's, in order:

```gamag
service PatientMonitor
    checkpoint every 5s

    protect
        let raw = readSensors()
        publish(analyze(raw))

    recover
        retry 3
        restore checkpoint
        restart
        alert operator
```

`retry` is level 0, `restore checkpoint` level 2, `restart` level 3,
`alert operator` level 5. Every action is audited with its level, whether it
succeeded, and what it did. **A permanent fault exhausts the policy and
propagates** — the engine does not retry forever and does not pretend to have
recovered. `restore checkpoint` returns to genuinely recorded state or fails;
it never invents state.

---

## What this does not claim

Spec §43 lists what Gama-G must never claim, and a test scans the repository for
the forbidden phrases so they cannot creep in:

- **No universal performance guarantee.** A native backend now exists and
  `ggc bench` measures it, but a measurement of one program on one machine is
  not a property of the language.  The harness prints its conditions and refuses
  to divide two runs whose conditions differ.
- **No universal accuracy guarantee.** The training example converging on the
  parameters of a linear model is a correctness test, not an accuracy claim
  about any real dataset.
- **No automatic medical, legal or regulatory compliance.** `medical_dosing.gg`
  demonstrates contracts and domain types. It makes nothing compliant with
  anything. Clinical correctness requires domain validation and the appropriate
  regulatory process.
- **No claim that the audit chain is legally admissible.** Records are signed
  with a key the process holds; key management and tamper-evident storage are
  outside this implementation.

---

## Repository layout

```
Gama-G_v1.0_Production_Specification.txt   the blueprint
Gama-G_Detailed_Audit_and_Verification_Report.txt   the first audit
Gama-G_Complete_Originality_and_Technical_Audit.txt  the audit v0.3 answers
LICENSE                                      Apache-2.0
VERSION pyproject.toml                       the version, stated once
conformance/                                 claims against the specification,
  claims.json deviations.json                each citing its section, with the
  stage-mapping.json requirements.json       stages and requirements that are
  programs/*.gg README.md                    not met written down as findings
.github/workflows/{ci,release}.yml           tests on every push; a tag builds
                                             a wheel, checks it, and publishes
                                             with the conformance report
compiler/gamag/
  core/{mir,parser,graph,native}.py        the language core and its own IR
  core/{memory,capability,recovery}.py     the three v0.4 models
  capabilities.py                          one capability algebra, used by the
                                           checker and the runtime alike
  lexer.py parser.py ast_nodes.py          front end
  semantic/{types,checker}.py              type lattice and checking
  gir/{ir,builder,optimizer}.py            the IR, lowering, optimization
  runtime/{vm,context,values,tensor,       the reference interpreter and
           recovery,audit,checkpoint,ops}.py  its subsystems
  std/library.py                           266 builtins
  methods.py                               one method table, shared by checker and VM
  cli/main.py driver.py                    ggc
tools/bin/{ggc,ggtest}                     entry points
tools/release_notes.py                     release notes, generated from the
                                           conformance run they describe
examples/core/                             eight core programs
examples/                                  eight v0.1 programs
tests/                                     642 tests
docs/DESIGN_v0_4.md                        the memory, capability and recovery
                                           models, and what each one proves
docs/DESIGN_v0_3.md                        the native IR, the five graphs, and
                                           what is proven vs. only checked
docs/DESIGN_v0_2.md                        the language design, and the first
                                           audit's ten originality questions
docs/IMPLEMENTATION.md                     honest status, section by section
docs/SPEC_DECISIONS.md                     what the specification left
                                           open, and what was decided
docs/COMPARISON.md                         what compiling to native code
                                           bought, measured, before and after
docs/RELEASES.md                           what a version number promises, the
                                           bar for leaving Alpha, how to tag
docs/PRODUCTION_GAPS.md                    where this is not production grade
```

## Layout of the tests

```
tests/support.py             harness; reads examples and word lists out of the
                             specification so they cannot drift from it
tests/test_spec_vocabulary.py  the enumerations the spec lists (types, effects,
                             recovery levels, capabilities, modules, commands)
tests/test_spec_examples.py  the spec's own code examples, lifted by line number
tests/test_core_language.py  the core: derived order, relationship checking,
                             selection proofs, bounded repetition, constraints,
                             state and authority, the native lowering, the five
                             graphs, and what the tests do not prove
tests/test_language.py       positive semantics
tests/test_enforcement.py    what must be refused, the claims the docs may not
                             make, and the packaging that must agree
tests/test_runtime.py        autodiff, the audit chain, recovery, secrets,
                             determinism
tests/test_formal_semantics.py
                             the v0.4 models: ownership and extents, slot reuse
                             as a proof, secret guarding, the capability algebra
                             on both sides of the compile/run boundary, recovery
                             levels, named checkpoints, transactions
tests/test_examples.py       all sixteen examples, end to end
tests/test_conformance.py    that the conformance suite can *fail*: it mutates
                             copies of the suite and requires the runner to
                             refuse an unrecorded deviation, a stale one, an
                             alternative that does not compile, an invented
                             stage, a dropped requirement, a wrong expectation
tests/test_release.py        that the release policy and the measurement agree:
                             the classifier may not claim more than
                             `ggc conform --strict` supports, and the workflow
                             may not pass the release script a flag it does not
                             accept
```

The vocabulary and example suites read the blueprint out of the repository at
run time, so editing the specification updates what is tested rather than
leaving a stale copy behind.
