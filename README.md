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

## Status

This repository is a **working vertical slice** of
[`Gama-G_v1.0_Production_Specification.txt`](Gama-G_v1.0_Production_Specification.txt),
not a finished implementation of it. The specification describes a multi-year,
multi-team production language across Phases 0–7.

What is here: a complete compiler front end (lexer, parser, name resolution,
type/effect/capability/ownership checking), the Gama IR, an optimizer, and a
reference interpreter — with the safety systems enforced end to end and
185 passing tests.

What is not: **there is no native backend, no package manager and no borrow
checker.** Programs run on an interpreter. Performance is interpreter-grade and
**no benchmark against native code has been run or is claimed.**

[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) goes through the
specification section by section and says what is done, what is partial and what
is missing, including the seventeen-item first-implementation checklist in §44.
Read it before relying on anything here.

---

## Quickstart

No installation. Python 3.11+ is the only requirement; NumPy is used when
present and there is a pure-Python fallback when it is not.

```sh
git clone https://github.com/honeyglitchgirl-droid/Gama-G
cd Gama-G

./tools/bin/ggc run examples/hello.gg          # run a program
./tools/bin/ggc check examples/hello.gg        # diagnostics only
./tools/bin/ggc test examples/property_tests.gg # run its tests
./tools/bin/ggc gir --fn main examples/hello.gg # inspect the IR

python3 -m unittest discover -s tests           # the compiler's own suite
```

`ggc` exit codes: `0` success, `1` compile error, `2` runtime fault, `3` usage
error, `4` test failure.

---

## The examples

Eight programs, each runnable and each covered by tests that assert on their
output rather than merely on their exit code.

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

At run time, faults are classified with source positions rather than surfacing
as host-language tracebacks: `DivideByZero`, `ContractViolation`,
`CapabilityViolation`, `RecoveryExhausted`, `AssertionFailed`.

Contracts quote themselves when they fail:

```
runtime fault [ContractViolation] at dosing.gg:12:5:
  requires contract violated: weight > 0
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

- **No universal performance guarantee.** Nothing is benchmarked against native
  code, because there is no native code.
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
compiler/gamag/
  lexer.py parser.py ast_nodes.py          front end
  semantic/{types,checker}.py              type lattice and checking
  gir/{ir,builder,optimizer}.py            the IR, lowering, optimization
  runtime/{vm,context,values,tensor,       the reference interpreter and
           recovery,audit,checkpoint,ops}.py  its subsystems
  std/library.py                           215 builtins
  methods.py                               one method table, shared by checker and VM
  cli/main.py driver.py                    ggc
tools/bin/{ggc,ggtest}                     entry points
examples/                                  eight runnable programs
tests/                                     185 tests
docs/IMPLEMENTATION.md                     honest status, section by section
```

## Layout of the tests

```
tests/support.py             harness; reads examples and word lists out of the
                             specification so they cannot drift from it
tests/test_spec_vocabulary.py  the enumerations the spec lists (types, effects,
                             recovery levels, capabilities, modules, commands)
tests/test_spec_examples.py  the spec's own code examples, lifted by line number
tests/test_language.py       positive semantics
tests/test_enforcement.py    what must be refused
tests/test_runtime.py        autodiff, the audit chain, recovery, secrets,
                             determinism
tests/test_examples.py       all eight examples, end to end
```

The vocabulary and example suites read the blueprint out of the repository at
run time, so editing the specification updates what is tested rather than
leaving a stale copy behind.
