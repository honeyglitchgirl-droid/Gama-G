# Gama-G v0.4 — the formal semantics of the core

> The second audit's priorities 4, 5 and 6: an **explicit memory/resource
> model**, **formal capability semantics**, and a **direct transition/recovery
> representation**. v0.3 gave the core its own IR so that nothing had to be
> elaborated into the older language's model. v0.4 gives that IR the three
> models the specification writes down formally in §8, §12, §10 and §18 — so
> those properties are computed by the toolchain instead of being repeated as
> prose in a document that no test reads.
>
> 316 tests pass. Nothing here claims a performance property, and priority 3 —
> a native CPU backend — is not started.

---

## 1. What changed

v0.3 compiled a core program through five inspectable graphs. v0.4 adds three
models alongside them, each with its own module, its own diagnostics, and its
own tests:

| Priority | Model | Module | Reaches the machine as |
|---|---|---|---|
| 4 | memory / resource | `core/memory.py` | `SECRET_GUARD`, slot assignment, `ggc memory` |
| 5 | capability | `core/capability.py` + `capabilities.py` | `CAP_CHECK` |
| 6 | transition / recovery | `core/recovery.py` | `PROTECTED`, `CHECKPOINT`, `TRANSACTION` |

The distinction that mattered while building them: a **graph** describes what a
program says, and a **model** computes a property of it that can be *wrong*. The
five v0.3 graphs could be inspected; they could not be violated. Each of these
three can be violated, and each violation is a diagnostic that names the rule.

Two supporting changes were forced by the work rather than chosen:

* **`compiler/gamag/capabilities.py`** — one capability algebra, imported by
  both the core checker and the runtime. Before this the two halves of the
  toolchain had separate notions of what a grant meant (§4.2).
* **`OpNode.effect` became `OpNode.effects`** — a list, with `effect` kept as a
  read-only property returning the first. The standard library's own builtins
  declare several effects (`secrets.expose` is crypto *and* audit,
  `medical.fhir_serialize` is medical *and* io, `model.load` is model *and*
  storage), so a singular field made a formally-checked capability impossible to
  exercise (§8).

The language surface is unchanged: `gama core 0.2` still parses, and no v0.3
example needed editing except `converge.gg`, which had been written with an
invented capability name.

---

## 2. The pipeline

```
core source → CoreParser → SemanticModel → NativeChecker → Lowerer → GIR → VM
                              │                 │
                     five graphs (v0.3)   three models (v0.4)
                     intent · operations   memory · capability · recovery
                     constraints ·              │
                     authority · recovery       └─ all three are consulted by
                                                   the Lowerer, so what they
                                                   prove is what gets emitted
```

`driver._compile_core` builds them in dependency order:

```python
c.core_model      = core_graph.build(c.core_syntax, model_bag)
checker           = core_native.check(c.core_model, model_bag)
core_capability.all_demands(c.core_model, checker.secret_bindings(),
                            c.core_model.intent.authority)
c.core_recovery   = core_recovery.policy_of(c.core_model.intent)
c.core_memory     = core_memory.build(c.core_model, checker=checker)
c.program         = core_native.lower(c.core_model, checker, lower_bag, ...)
```

A violation in the memory model is reported as `E-memory-model`. It is an
internal-consistency failure — the model contradicting itself about a program
that compiled — so it is a compiler bug and exits with the compile status, not
the runtime one.

---

## 3. The memory and resource model (priority 4, spec §8)

§8 asks for deterministic resource management with explicit ownership and
borrowing, deterministic destruction, and no use-after-free, double-free, data
race or garbage collector. It gives sensitive data "stronger lifecycle controls,
including restrictions on logging, serialization, and accidental conversion to
ordinary Text."

### 3.1 Ownership and extents

`MemoryModel.bindings` maps a name to a `Binding`:

```python
Binding(name, kind, owner, borrowers, secret, type, mutable,
        first, last, to_return, slot, shares_with)
```

* **`owner`** is exactly one node per binding, and the relation is one-to-one:
  two bindings never share an owner.
* **`kind`** is `param`, `value`, `state` or `outcome` — the core's own words.
* **`first`/`last`** are indices in the derived execution order, so an extent is
  a claim about *when a value is live*, computed from the graph rather than from
  where a name happens to appear in the source.
* **`borrowers`** are the nodes that read the binding. Every borrower's first
  index falls inside the owner's extent; that is the no-use-after-free property,
  stated as something the allocator can check.
* **`to_return`** marks what must outlive the body: the parameters and the
  outcome.
* **`mutable`** is false for everything except a declared `state`, because the
  core has no assignment. A model that reported a mutable binding in a program
  with no state would be describing a different language.

### 3.2 Slot allocation is a proof, not a hope

Slots are assigned by extent. Two bindings share a slot only if their extents do
not overlap, and `slots_saved` counts what that earned:

```
slot 0: x [0,1] · z [2,3]      ← reused, provably disjoint
slot 1: a [0,4]                ← a parameter, live to the return
slot 2: y [1,2]
slot 3: w [3,4]                ← the outcome
slots_saved: 1
```

The negative case is the informative one, and it is tested: a **two-operation
chain reuses nothing**, because there every value is live at the same time as its
neighbour. Reuse that appeared in that program would be a use-after-free wearing
the name of an optimization.

A slot is also never shared between different types, and `shares_with` records
the pairing so the reuse is visible rather than merely counted.

### 3.3 Resources

An effect that acquires a resource produces an `Acquisition(node, resource,
effect, level, spans)`. The resource is named in the domain's own terms, from a
fixed table of what an effect implies — `io` acquires `filesystem`, `crypto`
acquires `key material`, `medical` acquires `patient store`, `model` acquires
`model store`, `audit` acquires `audit log` — and it is owned by exactly one
node, with an extent. This is what makes deterministic destruction describable: a
resource has a place in the order where it stops being held.

### 3.4 Secrets

A `source secret` binding is marked at the point it enters, and three things
follow from that mark, at three different layers:

1. **Compile time.** Anything that would let the value reach a renderer demands
   `SecretExpose`; without it the program does not compile (`E-capability-unmet`).
   `E-secret-escape` covers the case where the value escapes through a path the
   demand analysis can name directly.
2. **Lowering.** Every secret definition emits a `SECRET_GUARD` instruction —
   one per secret binding, asserted by test.
3. **Runtime and library.** `to_text` refuses a secret *even when the program
   was granted `SecretExpose`*, because that boundary belongs to the library and
   not to the program. The only way out is `secrets.expose(value, reason)`,
   which requires the capability, requires a reason, and records
   `SECRET_EXPOSED` at security level in the audit chain.

The generated entry point prints `[secret <name>]` for a secret outcome rather
than its value. A program that wants a secret on stdout should say so in its own
body — and when it does, the audit log says so too.

Computing *with* a secret does not make every result secret: in
`examples/core/custody.gg`, `secrets.fingerprint(pin)` yields an ordinary `Text`.
If it did not, a secret could never be used for anything.

### 3.5 `ggc memory`

```sh
ggc memory examples/core/dose.gg              # the model, rendered
ggc memory examples/core/traverse.gg --slots  # slot lifetimes and reuse
ggc memory examples/core/traverse.gg --json   # bindings, slots, acquisitions,
                                              # slots_saved, violations
```

A program that is not core exits with the usage status and says so; a model that
contradicts itself exits with the compile status, because that is a compiler bug
and not something a user's program did.

---

## 4. Formal capability semantics (priority 5, spec §12)

§12 models a capability as access to a store — `PatientStore[Read]` versus
`PatientStore[Write]` — and requires that grants be attenuable but never
amplifiable.

### 4.1 A capability is a permission over a resource

`Capability.parse` splits a written name into `(resource, permission)`, so the
two spellings §12 gives for one capability really are one capability:

```python
Capability.parse("PatientStore[Read]")  # resource=Patient, permission=Read
Capability.parse("PatientRead")         # resource=Patient, permission=Read
```

The entailment between permissions is written down with its reason rather than
buried in a comparison operator:

* `Write ⇒ Read`, because writing a record requires reading it back.
* `Expose ⇒ Read` and `Load ⇒ Read`, for the same reason: the second act cannot
  happen without the first.
* `Connect`, `Sign` and `Spawn` entail **nothing**. They are actions, not
  accesses. Granting the ability to sign does not grant the ability to read, and
  pretending otherwise would silently widen every capability that names one.

Nothing crosses resources: `PatientWrite` does not cover `AuditWrite`, and
`PatientStore` does not cover `ModelStore`.

The vocabulary is `KNOWN_CAPABILITIES` — fourteen names, the specification's own
— and writing one it does not contain is `E-unknown-capability`. Inventing a
capability is how a program ends up depending on a guarantee nobody agreed to.

### 4.2 One relation, on both sides of the boundary

`covers(held, wanted)` is defined once, in `capabilities.py`, and used by both
the checker and `Context.has_cap`. This is not tidiness. Before it, the checker
used coverage and the runtime used exact set membership, so a program the
compiler *proved* would be granted `PatientRead` because it declared
`PatientWrite` was refused at runtime. A guarantee the compiler proves and the
runtime does not enforce is a guarantee about a program that cannot be run; the
reverse is a guarantee the toolchain cannot deliver.

The runtime's `"*"` wildcard is handled inside `covers` for the same reason: a
wildcard that only the runtime understood would be a second semantics.

### 4.3 Derived demands: effect ⇒ implied capability

What a node needs is *derived* from what it calls, not only from what it
declares. `core/capability.py` computes a `Demand` per node:

```python
Demand(node, origin, alternatives, pos, secret)
```

`origin` says why the demand exists — ``Read: calls `io.read_file` `` — so the
diagnostic can name the call site rather than the intent. `alternatives` lists
what would satisfy it, which is where coverage shows up: a demand for
`PatientRead` is satisfied by `PatientWrite`.

Every derived demand becomes a `CAP_CHECK` instruction at the boundary, **even
when the compiler already proved it will pass**. The proof is about the program;
the check is about the environment it is deployed into. `CAP_CHECK` carries the
capability and its origin in `meta`, and at runtime lowers to
`Context.require_capability`, which records `CAPABILITY_DENIED` at security level
and refuses. A denial is data in the audit chain, not an exception message to be
parsed.

Unsatisfied demands are `E-capability-unmet`; an intent that cannot satisfy a
need its own operations declare is `E-authority-unmet`. Effects are checked the
same way: `E-unknown-effect` for an effect the specification does not list,
`E-effect-undeclared` for one a node uses without declaring, and the diagnostic
suggests the list form because that is usually what was meant.

### 4.4 Attenuation, and the absence of amplification

Attenuation is not a separate operation in this algebra — it is coverage read
backwards. Granting less is exactly: the attenuated capability is covered by the
original, and not the reverse.

```python
covers(("PatientStore[Write]",), "PatientRead")   # True  — attenuation
covers(("PatientRead",), "PatientWrite")          # False — never the reverse
```

Amplification is what the algebra refuses, and the interesting case is
combination: two reads on different resources are still only two reads.

```python
covers(("PatientRead", "AuditWrite"), "PatientWrite")   # False
covers(("CryptoSign",), "PatientRead")                  # False
```

There is no rule anywhere that unions, promotes or infers a capability from the
presence of others. This is the property that makes the whole model worth having:
if a grant could be amplified, every proof about attenuation would be about a
system that does not enforce it.

---

## 5. Direct transition/recovery representation (priority 6, spec §10, §18)

§10 makes recovery a language capability with six levels, and asks two things
that are easy to write down and hard to honour: that it **never silently invents
state**, and that **every action is observable and auditable**. §18 asks that a
failure before commit leave a recovery state rather than a half-applied change.

"Direct representation" is the point of the priority. Before v0.4 a recovery
policy could only reach the machine as a fault message or a jump target —
something to be re-parsed at the moment it mattered least.

### 5.1 The policy is data

`recover` and `checkpoint` are core syntax on the intent:

```
intent Vitals
    checkpoint admission
    recover
        retry within 2 rounds
        restore checkpoint admission
        escalate operator
```

`RecoveryPolicy(steps, checkpoints, component, problems)` holds it, where each
step is `RecoveryStep(action, count, target, raw, pos)`. `raw` keeps the
program's own words, and `to_steps()` resolves each target:

```python
[{'action': 'retry',    'count': 2,    'target': '',          'raw': 'retry within 2 rounds'},
 {'action': 'restore',  'count': None, 'target': 'admission', 'raw': 'restore checkpoint admission'},
 {'action': 'escalate', 'count': None, 'target': 'operator',  'raw': 'escalate operator'}]
```

The lowered intent function carries exactly that as its `GFunction.recovery` — a
`RecoveryPlan(steps, audit_all=True, level_names)` — inside a `PROTECTED` region.
The policy the program declared is the policy the machine executes.

### 5.2 Levels, and why they cannot decrease

```python
ACTION_LEVELS = {'retry': 0, 'reset': 1, 'reconnect': 1, 'restore': 2,
                 'replay': 2, 'restart': 3, 'failover': 4,
                 'alert': 5, 'escalate': 5}
```

These are the specification's levels, not an implementation's convenience — the
vocabulary test reads them out of the blueprint at run time. A policy may not
step back down (`E-recovery-unordered`), because reordering would let a program
retry something it had already escalated, which is how a bounded policy becomes
an unbounded loop. An action outside the vocabulary is `E-unknown-recovery`.

`retry` with no bound is `E-unbounded-recovery`. Unbounded recovery is the same
defect as unbounded refinement wearing different clothes, and the core refuses
both for the same reason.

### 5.3 Named checkpoints, and "never invent state"

A declared `checkpoint admission` emits a `CHECKPOINT` instruction
(`meta = {'mode': 'at', 'raw': 'admission'}`) *before* the work runs, and is
recorded in the audit chain as `CHECKPOINT_CAPTURED` with `label: 'admission'`.

`restore checkpoint admission` then restores **that** checkpoint. Restoring to a
label the intent never declared is `E-unknown-checkpoint`, and the diagnostic
quotes the rule:

> spec section 10: the runtime must never silently invent state during recovery.

A bare `restore checkpoint` — what v0.1's `self_healing_service.gg` writes —
means the region entry point, which always exists. The two forms are kept apart
deliberately: `_checkpoint_label()` returns `""` for the bare form so it resolves
to the most recent, and collapses nothing.

### 5.4 Transactions

A transition is the only way a `state` changes, it runs after every derivation,
and it is emitted inside a `TRANSACTION` with `meta['action']` of `begin` and
`commit`. The transaction is a property of the *transition*, not of the recovery
policy that happens to wrap it — a program with no `recover` clause still gets
one. A constraint that fails in the commit phase therefore produces
`TRANSACTION_BEGIN → REQUIRE_FAILED → TRANSACTION_ABORT`, and
`Context.transactions` ends in `committed` or `aborted`, never in between.

### 5.5 What the audit trail shows

Running `examples/core/recover.gg` with `reading` changed to `500` — an
implausible vital — produces the policy executing itself:

```
CHECKPOINT_CAPTURED  label=Vitals-entry
CHECKPOINT_CAPTURED  label=admission
RECOVERY_ACTION      level 0 (local retry)              retry within 2 rounds
CHECKPOINT_CAPTURED  label=admission
RECOVERY_ACTION      level 2 (state checkpoint restore) restore checkpoint admission
                     detail: restored 5 state binding(s) from checkpoint cp-000003 (`admission`)
OPERATOR_ALERT       escalate operator
RECOVERY_ACTION      level 5 (operator escalation)      escalate operator
→ RecoveryExhausted
```

The detail line is the property: the restore names the checkpoint it used, so
"never silently invent state" is something a reader of the log can verify rather
than something the implementation promises. Audited levels never decrease, which
is asserted over the records rather than over the source.

---

## 6. Proven at compile time, checked only at runtime, and not checked at all

Keeping these three apart is what makes the rest of this document trustworthy.

### Proven at compile time

* Ownership is one-to-one, and every borrower falls inside its owner's extent.
* Slot reuse happens only between provably disjoint extents.
* A secret is marked, guarded, and cannot reach a renderer without `SecretExpose`.
* Every derived capability demand is covered by the declared authority, or the
  program does not compile.
* A recovery policy's levels do not decrease, its retries are bounded, and every
  named checkpoint was declared.
* Every transition is emitted inside a transaction.

### Checked only at runtime

* `CAP_CHECK` at each boundary, including demands the compiler proved — the
  environment is not the program.
* `SECRET_GUARD` at each secret definition, and the library's own refusal to
  render a secret.
* `REQUIRE` for constraints recorded as `runtime` rather than `proven`.
* The recovery policy executing, and `RecoveryExhausted` when it does not suffice.
* Transaction abort when the commit phase fails.

### Not checked at all

* **No borrow checker in the Rust sense.** Extents are computed from the derived
  order, which the core's own rules keep acyclic and single-assignment. That is a
  much stronger starting position than a general-purpose language has, and it is
  not the same thing as alias analysis over arbitrary mutation. There is no
  arbitrary mutation to analyse.
* **No native backend.** Slots are assigned; no machine code is generated, and no
  allocation is elided at runtime. `slots_saved` describes the model, not a
  measured saving.
* **No data-race detector.** The derived graph makes the order explicit and
  `parallel` regions carry real dependency analysis, but nothing here proves the
  absence of races in a program that has been given one.
* **No proof that a recovery policy will succeed.** Bounded, ordered and audited
  is what is guaranteed. Whether retrying twice is *enough* is a property of the
  failure, not of the language.

---

## 7. What v0.4 does not do

* **Priority 3, the native CPU backend, is not started.** It is a
  multi-session code-generation effort rather than a model, and starting it
  badly would be worse than naming it as absent.
* **Priorities 7–16 are untouched**: differential interpreter/native testing,
  and everything the audit lists after it.
* The memory model describes the *core*. The v0.1 surface keeps its own
  ownership checking in `semantic/checker.py`, unchanged.
* `slots_saved` is a count from the model. It is not a measurement, and nothing
  here should be read as a performance claim.

---

## 8. Bugs found by building the models

Both were found by trying to run a program that the model said should work,
which is the argument for building the models.

**A core program's declared `authority` never reached the runtime.** Four call
sites read `compilation.checker.grants`, which is `None` for a core program —
the checker's grants belong to the v0.1 path. So the compiler proved every demand
was covered and the runtime denied every one of them. The fix is
`program_grants(compilation, extra)` in `driver.py`, which unions the program's
own grants (a core intent's `authority`, a v0.1 `grant` header), the checker's,
and the caller's; every run path uses it instead of re-reading the checker.

**`effect` was singular, so multi-effect builtins were uncallable.**
`secrets.expose` is crypto *and* audit. With one effect field per node, a program
that declared `effect crypto, audit` could not be written, and a program that
declared only one of them was refused with `E-effect-undeclared` — a formally
correct capability that could not be exercised. `OpNode.effects` is now a list,
`effect` is a read-only property returning the first, `_check_effect` compares
against a set, and the memory model records one acquisition per effect.

Two smaller ones, in the same category:

* `EXIT_ERROR` does not exist in the CLI. A memory-model violation is a compiler
  bug, so `ggc memory` exits with the compile status — using a name that was
  never defined would have turned a diagnostic into a `NameError`.
* `known()` was defined twice in `capabilities.py` after the algebra moved. The
  bodies were identical, so nothing misbehaved; it is removed, and the test suite
  now has a scan for duplicate top-level definitions to catch the next one.

---

## 9. Where to look

| Concern | File |
|---|---|
| The capability algebra, shared by checker and runtime | `compiler/gamag/capabilities.py` |
| Memory model: bindings, extents, slots, acquisitions | `compiler/gamag/core/memory.py` |
| Derived capability demands | `compiler/gamag/core/capability.py` |
| Recovery policy: levels, ordering, named checkpoints | `compiler/gamag/core/recovery.py` |
| Where the models are consumed and diagnostics raised | `compiler/gamag/core/graph.py`, `core/native.py` |
| Model construction order, `program_grants` | `compiler/gamag/driver.py` |
| `PROTECTED` / `CHECKPOINT` / `TRANSACTION` / `CAP_CHECK` / `SECRET_GUARD` | `compiler/gamag/gir/ir.py`, `runtime/vm.py` |
| Runtime capability checks, checkpoint labels, transactions | `compiler/gamag/runtime/context.py`, `runtime/checkpoint.py`, `runtime/recovery.py` |
| `ggc memory` | `compiler/gamag/cli/main.py` |
| The three models under test (47 tests) | `tests/test_formal_semantics.py` |
| A secret's whole lifecycle | `examples/core/custody.gg` |
| A declared recovery policy, run | `examples/core/recover.gg` |

Specification sections: §8 memory and resource management, §10 self-healing and
recovery, §12 security and capability model, §18 transaction semantics.

Related documents: [`DESIGN_v0_3.md`](DESIGN_v0_3.md) (the native IR and the five
graphs), [`DESIGN_v0_2.md`](DESIGN_v0_2.md) (the language design), and
[`IMPLEMENTATION.md`](IMPLEMENTATION.md) (honest status, section by section).
