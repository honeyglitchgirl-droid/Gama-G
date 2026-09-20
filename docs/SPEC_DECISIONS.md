# Specification decisions

The specification is the source of truth for this project, and it does not
answer every question a compiler has to answer.  This file records the places
where it is silent and the implementation had to decide, so that a reader can
tell a *rule* from an accident.  Every entry gives the question, what the
specification actually says, the decision, the evidence behind it, and where it
lives in the code and the tests.

A decision is not a preference.  Where one was made because a program was
proved to behave differently from how the specification's prose suggests it
should, the proof is the reason and the rule follows from it.  Both entries
below are of that kind, and both were settled by their owner before any code
changed.

---

## D-1: a name identifies one node

**Question.**  May two operations, functions, records or enums in one module
share a name, when each is distinct in every other respect?

**What the specification says.**  Section 9 describes a program as a graph of
typed operations, and every part of the toolchain refers to an operation by its
name: diagnostics, `ggc graph`, the audit trail, the provenance record.  The
specification never states that names are unique, and it was searched for
`duplicate`, `unique` and `unique` variants -- the requirement does not appear
for any kind of declaration.

**Decision.**  Names are unique.  A second `operation`, `fn`, `record`, `enum`,
`pipeline`, `model`, `service`, `policy`, `agent`, `fault` or `transaction`
declaring a name already taken in that module is a name-resolution error
(`E-duplicate-declaration`, or `E-duplicate-operation` for core operation
graphs).  Tests are exempt: a test is never referred to by name from code.

**Why the specification's silence is not permission.**  `source` and `state`
already had this rule (`E-duplicate-binding`, keyed on the *binding* they
produce), so three different things happened by kind:

| declarations | before |
|---|---|
| two `source` / `state` | refused, correctly, as `E-duplicate-binding` |
| two `record` / `enum` | accepted in silence; the second definition quietly replaced the first |
| two `operation`, distinct bindings | accepted; the graph kept one node and `ggc graph` printed one `Calc` where the source declared two |
| two `fn` | `error[gir-generation] E-ice: internal compiler error while lowering to GIR: GIR already contains a function named 'Calc'; a later definition would silently replace it` |

The last row is the one that settles it.  A plain duplication in the source
reached the GIR builder, and the toolchain reported it as its own bug -- exit
code included -- with a message that names no line of source and arrives three
phases after the mistake.  A compiler that calls a user error an internal
compiler error has no defensible position on either.  The third row settles the
other half: the graph was a dictionary keyed by name, so the second operation
was dropped from `operations` while `producers_of` still listed both, and the
graph disagreed with itself with nothing to say so.

Uniqueness does not restrict anything the language expresses.  Spec section 9's
*selection* -- several operations producing one binding under mutually
exclusive `when` guards -- is written with distinctly named alternatives
(`examples/core/selection.gg`: `Urgent`, `Routine`), so it is unaffected.  The
namespace is the module's own: a user function may still shadow a builtin, and
a core helper may share a name with a core operation, because those live in
different namespaces.

**Implemented in.**  `compiler/gamag/core/graph.py::GraphBuilder._collect` (core
operation graphs; a refused node keeps its binding and is reported once, so the
author sees one mistake rather than a cascade of `cannot find` errors about a
binding they did write); `compiler/gamag/semantic/checker.py::Checker._claim_declaration_names`
(the whole-module namespace, run before any body is checked).

**Tested by.**  `tests/test_core_language.py::Relationships` (three tests) and
`tests/test_language.py::DeclarationNames` (five tests, including the two
behaviours that must survive: builtin shadowing and selection).

---

## D-2: a `parallel` task performs no effect

**Question.**  May the tasks of a `parallel` region read files, write files,
talk to the network, print, or record audit entries?

**What the specification says.**  Section 9C: the compiler *may* execute
independent operations concurrently.  Section 3 makes a program a graph of
operations with dependencies, and the implementation commits a region's writes
in program order.  Section 7 lists `pure`, `io`, `network`, `storage`,
`crypto`, `model`, `medical`, `audit` and `unsafe` as the effect vocabulary.
The specification does not say what happens when one of those appears inside a
region; "deterministic ordering" appears once, only as a capability of the
analyzer.

**Decision.**  A parallel task's call graph may perform no effect other than
computation.  `io`, `network`, `storage`, `crypto`, `model`, `medical`, `audit`
and `unsafe` inside a region are refused at check time with
`E-parallel-effect`, reported at the first call that introduces the effect.

**Evidence.**  Commit-in-program-order makes the *state* a region leaves behind
deterministic, and that was measured to be true.  It does not order what the
tasks say to the outside world, and no amount of scheduling can, because the
region's entire purpose is to choose the schedule at run time.  Three tasks
printing `alpha`, `beta`, `gamma` and given deliberately unequal work printed

```
run 1:  gamma, beta, alpha
run 2:  beta, gamma, alpha
run 3:  gamma, beta, alpha
```

on three consecutive runs of one unchanged program.  A program whose observable
output depends on the schedule is a program with a race in it, and the
toolchain's own differential oracle could not see it: `oracle.check_deterministic`
reported zero violations for that program, because it compares values and the
values never varied.

The first measurement of this was wrong, and the mistake is worth keeping.
Twenty runs of three near-instant printing tasks produced program order every
time -- the tasks finish before the scheduler can interleave them, so the
fixture was measuring the absence of contention, not the presence of ordering.
Only after making the first task roughly a thousand times slower did the order
vary.  Equal-cost tasks cannot evidence determinism.

**Why forbidding is the right repair.**  The alternatives were to serialize
effects inside a region (which discards the concurrency the feature exists for,
and quietly changes the meaning of a `parallel` over an effectful call from
"may be concurrent" to "is not"), or to leave the order unspecified and merely
document it.  Forbidding keeps concurrency and makes the outcome total:
every task is pure, so every interleaving computes the same values, and
determinism holds by construction rather than by observation.  The rule is also
enforceable at check time, before any program runs.

**Consequence for the shipped example.**  `examples/parallel_pipeline.gg` had
its `io` calls inside its region, and passed only because of a defect in the
first implementation of this check (see below).  It now loads its inputs before
the region and reports after it, which is the shape the rule pushes programs
towards and which the example now teaches.

**A first implementation that was wrong.**  The check began as a comparison of
the enclosing function's effect set before and after the region body.  That is
blind in a common case: a function that already printed *before* the region
shows no change when a task prints *inside* it, so the racy region was accepted.
It was demonstrated on a program of exactly that shape.  The implementation now
records each effect where it is discovered
(`Checker._note_effect`), counting depth so that nested regions work, and
draining the entries when the region ends.

**Implemented in.**  `compiler/gamag/semantic/checker.py`:
`Checker.PARALLEL_FORBIDDEN`, `Checker._note_effect`,
`Checker.stmt_Parallel`, `Checker._refuse_parallel_effects`, with every
effect-discovery site routed through `_note_effect`.

**Tested by.**  `tests/test_language.py::ParallelRegions` -- four tests, one of
them the regression for the effect-before-the-region case.

---

## How to check these yourself

```sh
# D-1: a duplicate operation is refused, once, without cascading
printf 'gama core 0.2\nintent T\nsource weight : F64 from 2.0\noperation Calc\n    uses weight\n    yields a : F64\n    effect pure\n    computes weight\noperation Calc\n    uses weight\n    yields b : F64\n    effect pure\n    computes weight\noutcome a\n' > /tmp/dup.gg
ggc check /tmp/dup.gg          # E-duplicate-operation

# D-1: a duplicate function is a name error, not E-ice
# D-2: an effect inside a region is refused
ggc check examples/parallel_pipeline.gg   # ok: the region is pure
```

`tests/test_core_language.py`, `tests/test_language.py` and
`docs/DESIGN_v1_0.md` section 6 record the defects these rules came from.
