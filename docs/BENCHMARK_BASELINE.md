# Benchmark baseline

`docs/PRODUCTION_GAPS.md` section 5 item 6 asks for "a benchmark baseline,
published with its conditions".  This is that baseline.  Read the framing
before the numbers:

**These are measurements, not claims.**  Spec section 43 forbids making
performance a property of the language, and nothing below should be read as
one.  The `ggc bench` harness prints its conditions and refuses to divide
two runs whose conditions differ; this document does the same -- it reports
per target and computes no ratio between them.

## How to reproduce

```sh
./tools/bin/ggc bench examples/*.gg examples/core/*.gg --repeats 5
./tools/bin/ggc bench examples/hello.gg examples/core/classify.gg \
    examples/core/converge.gg examples/core/traverse.gg \
    --native --repeats 5
```

## Conditions of this measurement

| | |
|---|---|
| date | 2026-09-20 |
| toolchain | Gama-G 1.4.0 (GIR 1.0) |
| host | x86_64, Linux sandbox, no other load reported |
| interpreter | CPython 3.11.2 |
| profile / optimization | strict, -O1 |
| repeats | 9 measured after 1 warmup |
| native backend | present for six of the sixteen programs; `cc -O2` |

## Reference interpreter

Wall clock is milliseconds for a whole `compile + run`; every program here
finishes inside a millisecond or two, which is dominated by Python import
and per-run bookkeeping, not by the program.  **The instruction counts are
the machine-independent column**: they are what the GAEM executed, exact
and identical across runs.

| program | wall p50 (ms) | instructions |
|---|---:|---:|
| hello.gg | 0.470 | 214 |
| medical_dosing.gg | 0.413 | 166 |
| parallel_pipeline.gg | 0.993 | 140 |
| policy_transaction_agent.gg | 1.886 | 223 |
| property_tests.gg | 0.068 | 3 |
| security_audit.gg | 0.524 | 96 |
| self_healing_service.gg | 0.489 | 68 |
| train_linear_model.gg | 63.958 | 6 097 |
| core/classify.gg | 0.041 | 10 |
| core/converge.gg | 0.166 | 80 |
| core/custody.gg | 0.158 | 15 |
| core/dose.gg | 0.091 | 15 |
| core/ledger.gg | 0.186 | 14 |
| core/recover.gg | 0.339 | 20 |
| core/selection.gg | 0.077 | 12 |
| core/traverse.gg | 0.104 | 43 |

**Every instruction count above is identical to the 1.2.0 and 1.3.0
baselines**, to the program, and that is the point of printing them: v1.3 added
a build-time promise prover that discharges `holds` clauses and v1.4 gave the C
runtime an audit chain, and the language both added instructions to is the
compiler, not the program.  The wall column moved because it always moves; the
instruction column is the one that would have caught a change that started
optimising the examples by accident.

(`property_tests.gg` reports the *entry* run only; `ggc test` executes its
test functions separately, which is the `ggc test` path, not `main`.)

## Native target

The six programs the native backend accepts (the rest are refused by name
and run on the interpreter).  The native figure is *process* wall time of
the compiled binary; at this program size it is startup-dominated in the
same way, and reading a native millisecond against an interpreter
millisecond compares two different things, which is why the harness itself
never divides them.

| program | wall p50 (ms) | note |
|---|---:|---|
| hello.gg | 1.529 | compiled, ran, difftest-agreed |
| core/classify.gg | 1.179 | compiled, ran, difftest-agreed |
| core/converge.gg | 1.164 | compiled, ran, difftest-agreed |
| core/dose.gg | 1.437 | compiled, ran, difftest-agreed (new in 1.4) |
| core/selection.gg | 1.406 | compiled, ran, difftest-agreed (new in 1.4) |
| core/traverse.gg | 1.239 | compiled, ran, difftest-agreed |

`core/dose.gg` and `core/selection.gg` are the two that arrive with `trail`
clauses, and they sit at the top of that column, so the cost of the chain was
measured directly rather than inferred from a table: the *same* binary,
`examples/core/selection.gg`, 300 runs each way, p50 **1.171 ms** without
`--audit` and **1.334 ms** with it (+0.154 ms at the minimum, so the gap is not
noise), writing one 452-byte record.  That figure is `fopen`, one SHA-256, one
HMAC, the canonical-JSON rendering and an `atexit`, taken together and not
separated; the interesting part is that it is a *fixed* cost per run, not a
multiplication of the program's own work -- a program that ran a thousand
operations paid the same 0.16 ms for its trail that this one did, which is what
you would want from a control you are told to turn on.

## What a future baseline should measure instead

1. **Compute-bound programs**, not examples that print a line.  The suite's
   own call-depth benchmark (40 × fib(16), 127 722 calls) is the shape:
   seconds of work per run, so startup disappears.
2. **The same program on the same machine across tiers** (`-O0/-O1/-O2`,
   interpreter and native) -- ratios *within* one set of conditions are
   honest; ratios across machines are not.
3. **Instruction counts as the portable column.**  `ggc bench` already
   prints them next to the timings because they do not depend on load; a
   baseline that leads with them is harder to age.
4. **A corpus with a tail**: the fuzzer's generator makes programs no human
   wrote; timing those catches blow-ups examples would not.

None of this is a performance guarantee in either direction: the numbers
above will be wrong for your machine in an hour, which is exactly why the
conditions travel with them.
