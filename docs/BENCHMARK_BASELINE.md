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
| toolchain | Gama-G 1.2.0 (GIR 1.0) |
| host | x86_64, Linux sandbox, no other load reported |
| interpreter | CPython 3.11.2 |
| profile / optimization | strict, -O1 |
| repeats | 5 measured after 1 warmup |
| native backend | present for four of the sixteen programs; `cc -O2` |

## Reference interpreter

Wall clock is milliseconds for a whole `compile + run`; every program here
finishes inside a millisecond or two, which is dominated by Python import
and per-run bookkeeping, not by the program.  **The instruction counts are
the machine-independent column**: they are what the GAEM executed, exact
and identical across runs.

| program | wall p50 (ms) | instructions |
|---|---:|---:|
| hello.gg | 0.54 | 214 |
| medical_dosing.gg | 0.48 | 166 |
| parallel_pipeline.gg | 0.81 | 140 |
| policy_transaction_agent.gg | 1.88 | 223 |
| property_tests.gg | 0.04 | 3 |
| security_audit.gg | 0.52 | 96 |
| self_healing_service.gg | 0.47 | 68 |
| train_linear_model.gg | 60.37 | 6 097 |
| core/classify.gg | 0.05 | 10 |
| core/converge.gg | 0.17 | 80 |
| core/custody.gg | 0.17 | 15 |
| core/dose.gg | 0.09 | 15 |
| core/ledger.gg | 0.21 | 14 |
| core/recover.gg | 0.33 | 20 |
| core/selection.gg | 0.08 | 12 |
| core/traverse.gg | 0.11 | 43 |

(`property_tests.gg` reports the *entry* run only; `ggc test` executes its
test functions separately, which is the `ggc test` path, not `main`.)

## Native target

The four programs the native backend accepts (the rest are refused by name
and run on the interpreter).  The native figure is *process* wall time of
the compiled binary; at this program size it is startup-dominated in the
same way, and reading a native millisecond against an interpreter
millisecond compares two different things, which is why the harness itself
never divides them.

| program | wall p50 (ms) | note |
|---|---:|---|
| hello.gg | 1.20 | compiled, ran, difftest-agreed |
| core/classify.gg | 1.22 | compiled, ran, difftest-agreed |
| core/converge.gg | 1.21 | compiled, ran, difftest-agreed |
| core/traverse.gg | 1.19 | compiled, ran, difftest-agreed |

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
