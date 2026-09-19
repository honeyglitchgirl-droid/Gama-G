# Gama-G v0.2 — the original language core

This document is the answer to sections 16, 17 and 18 of
`Gama-G_Detailed_Audit_and_Verification_Report.txt`.

> **Partly superseded.** The *language* designed here is unchanged and still
> current: the grammar (§5), the construct-by-construct originality answers (§6),
> the inversion of order-from-declaration (§2) and the relationship rules (§2.1).
> What changed is the *compilation path*. §4 and the elaboration table below
> described a compiler that translated the core into the older implementation's
> AST; the second audit found that unacceptable, and v0.3 replaced it with a
> native semantic IR. Read [`DESIGN_v0_3.md`](DESIGN_v0_3.md) for how a core
> program is compiled today. The superseded text is kept below, marked, because
> an audit response is only checkable against what it actually said.

The audit's verdict on v0.1 was that the implementation is real and working, but
that the *language surface* fails the originality requirement: `fn`, `let`,
`var`, `if`, `else`, `for`, `while`, `return`, `match`, `enum`, `record` and
conventional `Result`/`Option` syntax are other languages' constructs, and the
report says plainly: *"Do NOT merely rename conventional constructs."*

v0.2 is the redesign it asked for. It is a new language core, written from the
concepts in section 16 — Intent, Operation, Relationship, Constraint,
Capability, Effect, State, Recovery, Audit, Execution Graph — and it is
implemented, tested and runnable in this repository.

---

## 1. What v0.2 is, and what it is not

**It is** a new source language with a new grammar, new semantics and a new
program model, reachable today:

```
ggc run examples/core/selection.gg
ggc graph examples/core/selection.gg --edges
```

**It is not** a rename of v0.1. None of the words the audit lists appear in the
core grammar as declarations or control flow. `tests/test_core_language.py::
LanguageInvariants::test_the_core_grammar_has_no_assignment_keyword` asserts
this mechanically: it walks the core's declaration words and clause keywords and
fails if `if`, `else`, `while`, `for`, `return`, `match`, `fn`, `let`, `var`,
`assign`, `break`, `continue` or `loop` appears among them.

**It is also not** a replacement for v0.1 in this milestone. v0.1 remains in the
tree as what the audit calls the *research / vertical-slice reference
implementation*. The two dialects are selected per file by a pragma, and both
test suites run green in the same checkout. *(v0.3 note: the core no longer
elaborates into v0.1's AST — see [`DESIGN_v0_3.md`](DESIGN_v0_3.md). v0.1 is
still the machine that runs it, and still supplies the shared type lattice and
standard library.)*

---

## 2. The inversion: order is an output, not an input

This is the single decision the rest of the language follows from.

A conventional program is a **sequence of statements**. The compiler reads the
sequence and may then analyse it. Control flow is written; data flow is inferred.

A Gama-G core program is a **set of operations**. Each operation declares the
bindings it consumes (`uses`) and the one binding it produces (`yields`). The
compiler matches producers to consumers, derives the dependency graph, checks it,
and only then produces an order. Control flow is derived; data flow is declared.

```
operation RawDose
    uses     weight, factor
    yields   rawDose : F64
    effect   pure
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

Nothing above says "first RawDose, then Dose". `tests/test_core_language.py::
DerivedOrder::test_writing_the_operations_backwards_does_not_change_the_order`
compiles the same program with its operations listed in the opposite order and
asserts the derived levels are identical. Declaration order carries no meaning,
because the language gives it none.

`ggc graph` prints what was derived:

```
derived execution order (never written in the source):
  level 0: RawDose
  level 1: Dose
```

Operations in the same level have no dependency between them. That is a fact
the compiler established, not a hint the programmer supplied — which is what
makes it safe for a future backend to parallelise a level.

### 2.1 Relationships must be real

Deriving order from declarations is only trustworthy if the declarations cannot
drift from the code. So the graph builder checks both directions
(`core/graph.py::_validate_relationships`):

* A binding an operation **actually refers to** but does not list in `uses` is
  `E-undeclared-relationship` — *"the execution graph is derived from declared
  relationships, so a real dependency must be written down."*
* A binding listed in `uses` but **never referred to** is
  `W-unused-relationship` — an edge that is not real costs a dependency and can
  create a cycle.
* A name in `uses` that **nothing produces** is `E-unresolved-relationship`.

This is the check that makes the whole model honest. Without it `uses` would be
a comment.

---

## 3. Section 16's concepts, and where each one lives

| Concept | Core construct | Enforced by |
|---|---|---|
| **Intent** | `intent Name` with `purpose`, `authority`, `trail` | parser; `purpose` kept verbatim and required by the example test |
| **Operation** | `operation`, `refine`, `each`, `resolve`, `transition` | `core/graph.py::_validate_shapes` |
| **Relationship** | `uses` / `yields` / `over`, matched into edges | `_validate_relationships`, `_order` |
| **Constraint** | `holds <predicate>` | elaborated to a runtime gate that quotes the constraint back |
| **Capability** | `authority` on the intent, `needs` on an operation | grants; `E-transition-target` for state authority |
| **Effect** | `effect <name>` from the specification's algebra | `E-unknown-effect` |
| **State** | `state Name : T` with `starts` and `authority` | `E-state-uninitialised` |
| **Recovery** | bounded `refine` + classified faults | `E-unbounded-refinement`, `RefinementDiverged` |
| **Audit** | `trail <prose>` per operation and per intent | hash-chained audit records; secret-safe by construction |
| **Execution Graph** | the derived two-phase graph | `_order`, printed by `ggc graph` |

The effect vocabulary is *the specification's own* (`pure`, `io`, `network`,
`storage`, `crypto`, `audit`, `medical`, `model`, `unsafe`) — deliberately not a
new one, because the effect algebra is a v0.1 asset the audit lists in section 4
as worth building on, not a construct to redesign.

---

## 4. Section 17's execution model, mapped to real modules

> **Superseded by v0.3.** The mapping below is what the code did when this
> document was written. It is retained for the audit trail. The current mapping
> is in [`DESIGN_v0_3.md`](DESIGN_v0_3.md) §2 and §9.

The report proposes: Intent → Meaning → Constraints → Operations →
Relationships → Operation Graph → GIR → Optimization → Backend → Runtime.
Here is where each arrow **was** in the code:

| Stage | Implementation (v0.2) | Implementation (v0.3) |
|---|---|---|
| Intent | `core/parser.py` | `core/parser.py` → `mir.IntentGraph` |
| Meaning | `core/ast.py` | `core/mir.py` |
| Constraints | `core/ast.py::Clause` | `mir.Constraint`, carrying verbatim text **and** a `discharge` |
| Operations | `core/parser.py::_parse_operation` | `core/parser.py` → `mir.OpNode` subclasses |
| Relationships | `core/graph.py` | `core/graph.py::build` → `mir.OperationGraph` |
| Operation Graph | `ExecutionGraph` | `mir.OperationGraph` (levels, edges, selections, proofs) |
| GIR | `core/elaborate.py` → **v0.1 AST** → `gir/builder.py` | `core/native.py::lower` → GIR, directly |
| Optimization | `gir/optimizer.py` | `gir/optimizer.py` (unchanged) |
| Backend | the reference VM | the reference VM — **no native backend yet** |
| Runtime | `runtime/*` | `runtime/*` (unchanged) |

The v0.2 boundary was described here as honest, and it was: the core decided
what a program meant and the reference slice decided how that meaning ran. What
the second audit pointed out — correctly — is that "how that meaning runs" was
being decided in the older language's *vocabulary*. `let`, `var`, `if`, `while`
and `match` were the intermediate representation of a language whose entire
purpose is not to have them. The table above existed to show the boundary;
`core/elaborate.py` and `core/ast.py` have since been deleted.

The elaborated forms are recorded here only as history:

| Core construct | v0.2 elaborated form | v0.3 native form |
|---|---|---|
| `source x : T from e` | `let x : T = e`, passed as an argument | a parameter of the intent function |
| `state s : T starts e` | `var s : T = e` | a slot, copied in at entry |
| `operation` | `let` + `require` + audit call | compute in level order; `REQUIRE` with verbatim text; `AUDIT` |
| guarded alternatives | `if`/`elif` chain ending in `panic(…)` | `select.<binding>` blocks ending in a `FAULT` terminator |
| `refine … within n` | `while` with a counter that panics at `n` | `refine.X.until` / `.round` / `.diverged` / `.next` / `.done` |
| `each … over xs as x` | accumulator + `for` | `each.X.more` / `.item` / `.keep` / `.skip` / `.done` |
| `resolve … choose` | `match` | one block per arm plus `resolve.R.join` |
| `transition` | an assignment, in the commit phase | a copy into the state slot, in the commit phase |

---

## 5. Grammar reference

```
program    := pragma? decl*
pragma     := "gama" "core" VERSION

decl       := intent | source | state | operation | refine | each
            | resolve | transition | outcome

intent     := "intent" NAME clause*          # purpose, authority, trail
source     := "source" "secret"? NAME ":" TYPE ("from" expr)?
state      := "state"  "secret"? NAME ":" TYPE clause*   # starts, authority
operation  := "operation" NAME clause*
              # uses, yields, effect, needs, holds, when, computes, trail
refine     := "refine" NAME clause*
              # uses, yields, effect, needs, holds, starts, repeats,
              # until, within (mandatory), trail
each       := "each" NAME clause*
              # over X as ITEM, yields, effect, needs, holds, when, computes
resolve    := "resolve" NAME clause*
              # over, yields, effect, needs, holds, choose, trail
transition := "transition" NAME clause*
              # alters, uses, effect, needs, holds, computes, trail
outcome    := "outcome" NAME
```

Clause blocks are indented, one `keyword  value` per line. An unrecognised
clause is a syntax error that lists the clauses that *do* belong to that
declaration, because in a declaration-driven language a misspelled clause would
otherwise silently drop a constraint or a capability requirement.

`purpose` and `trail` are prose, captured verbatim from the source rather than
rebuilt from tokens — the same reason contracts and policy rules keep their
text. `holds`, `when` and `until` also keep their verbatim text, which is what
lets a fault quote the promise that broke:

```
ContractViolation: holds `b < 3`
Panic: RefinementDiverged: `G` did not satisfy `g > 1000000` within 4 rounds
Panic: NoActiveAlternative: nothing yields `s` — 2 guarded alternatives, none active
```

### 5.1 How the conventional constructs are replaced

| Conventional | Core replacement | Why it is not a rename |
|---|---|---|
| `if`/`else` | several operations yielding one binding, each with `when` | alternatives are *declarations the compiler can compare*, not nested blocks. It proves mutual exclusivity and exhaustiveness where the guards are complementary |
| `while`/`for` (unbounded) | `refine … until … within N` | the bound is **mandatory**; an unbounded repetition is not expressible, so a program that cannot be written as a hang cannot hang |
| iteration over a collection | `each … over X as x` | fan-out is a producer of one collection binding, so it is a node in the graph like any other |
| `match` | `resolve … choose` | the subject comes from a declared relationship (`over`), and exhaustiveness is a graph property |
| assignment | none | bindings are single-assignment. Only `state` changes, only via `transition`, only under `authority` |
| `return` | none | an operation's value *is* the binding it yields; the intent's value is its `outcome` |
| function call as program structure | none | operations are wired by data, not called. Expression-level calls remain, as mathematical notation |

---

## 6. Section 18: the ten originality questions, answered per feature

The report's rule is that a *yes* to questions 6–9 without a strong technical
reason means "redesign it". Answers below are per feature, and the two places
where the honest answer is "partly yes" are marked and justified.

### 6.1 `intent`

1. **Purpose defined independently?** Yes — an intent is the program's contract
   with the world: what it is for, what authority it holds, what it must record.
   No mainstream language has a required top-level declaration of purpose.
2. **Syntax independently designed?** Yes — clause-based, prose-capturing.
3. **Semantics independently defined?** Yes — `authority` becomes the grant set;
   `purpose` and `trail` are retained verbatim in audit records.
4. **Fits the execution model?** Yes — it is stage 1 of section 17.
5. **Expressible with Gama-G concepts?** It *is* one.
6. **Merely renaming?** No. A `module` or `namespace` declares a name; an intent
   declares an obligation.
7–9. **Inheriting control-flow / memory / object model?** No.
10. **Why needed?** Because capability and audit enforcement need a scope to
    attach to that is not a function.

### 6.2 `operation` / `uses` / `yields`

1–3. Yes: a node that declares its inputs and its single output, checked against
what its body actually references.
4–5. Yes: it is the unit of the graph.
6. **No.** A function declares parameters and returns a value to a *caller*; an
   operation declares relationships and produces a binding for *consumers it
   never names*. There is no call, no caller, no call stack in the core's model.
7. **No** — there is no control flow to inherit; order is derived.
8. **No** — single-assignment bindings; nothing is aliased or reassigned.
9. **No** — operations are not objects and not first-class values.
10. Because a derived execution graph needs nodes whose edges are declared
    rather than implied by call order.

### 6.3 `holds` (Constraint)

1–5. Yes.
6. **Partly** — a precondition resembles a design-by-contract `requires`. The
   strong technical reason: the constraint is *data the compiler keeps*, not just
   a check. Its verbatim text is carried into the graph, into the fault message
   and into the audit record, so a violation is explainable after the fact. That
   is the audit requirement in section 16, which a conventional assertion does
   not satisfy.
7–9. No.
10. Because a language that promises explainable failures must keep the promise
   it broke in the words the program used.

### 6.4 `when` (guarded selection)

1–5. Yes.
6. **No.** Guards on *sibling declarations* are not `if`. The compiler compares
   alternatives to each other — something it cannot do with nested blocks — and
   records a proof (`proven_exhaustive`, `proven_exclusive`) that `ggc graph`
   prints. Where it cannot prove it, the runtime keeps a classified
   `NoActiveAlternative` fault rather than producing nothing silently.
7. **No** — this is precisely the construct that removes inherited control flow.
8–9. No.
10. Because exhaustiveness of a decision is a property worth checking, and it is
   only checkable when the alternatives are declarations.

### 6.5 `refine … until … within N`

1–5. Yes.
6. **No.** A loop is a jump with a condition; a refinement is a convergence
   obligation with a *mandatory* budget. The bound is not a style choice, it is
   part of the grammar — `E-unbounded-refinement` rejects the program, and the
   fix-it shows the syntax.
7. **No** — there is no unbounded repetition in the model to inherit.
8–9. No.
10. Because "this computation terminates" should be a stated, checked property,
   and because bounded recovery is one of the assets section 4 says to build on.

### 6.6 `each … over X as item`

1–5. Yes.
6. **Partly** — the shape resembles a comprehension. The strong technical
   reason: `each` produces exactly one binding and is therefore a *graph node*,
   ordered by the same rules as everything else, and the `over` clause is itself
   the declaration of the relationship (an `each` must not also list the
   collection in `uses`). A comprehension is an expression inside a statement;
   this is a producer inside a graph.
7–9. No.
10. Because fan-out must participate in derived ordering to be parallelisable.

### 6.7 `resolve … choose`

1–5. Yes.
6. **Partly** — pattern dispatch is pattern dispatch. The strong technical
   reason: the subject is reached through a declared relationship, and
   exhaustiveness is discharged by reusing the reference slice's existing,
   already-tested exhaustiveness rule rather than by adding a second one. Adding
   a competing implementation of the same guarantee would be worse engineering.
7–9. No.
10. Because exhaustive dispatch over a value is a real need, and the graph must
   be able to see it as one node.

### 6.8 `state` / `transition`

1–5. Yes.
6. **No.** There is no assignment in the core. A `state` is a resource with a
   declared initial value and declared authority; a `transition` is the only way
   it changes, and it may not retype the state it alters. Compute operations are
   forbidden from depending on a transition (`E-compute-after-commit`), which
   puts every effect after every derivation of data.
7. **No** — the two-phase split (compute, then commit) is the core's own model.
8. **Yes, deliberately, and narrowly:** mutation exists, because a language that
   cannot change anything cannot do anything. The technical reason it is not
   *inherited* is that mutation is confined to one declared kind of resource,
   reachable only through one declared kind of operation, under declared
   authority, in a phase of its own.
9. No.
10. Because state change is the one thing that genuinely needs authority and
   audit, and confining it is what makes those enforceable.

### 6.9 Expression notation (arithmetic, comparison, library calls)

6. **Yes — and this is intentional.** `a + b`, `x < 3`, `math.clamp(...)` are
   mathematical and library notation, not another language's control flow or
   program structure. Section 16 lists what must be reconsidered *at the
   language-core level*; it does not ask for a new spelling of addition, and
   inventing one would make the language harder to read without making it more
   original. The audit's own section 4 lists the tensor/autodiff work as an asset
   to build on, which assumes ordinary mathematical notation.

Everything else in this section answers 7–9 with "no" for the same reason: the
core has no control-flow model, no memory model and no object model to inherit,
because it has no statements, no assignment and no first-class operations.

---

## 7. What is proven at compile time, and what is only checked at runtime

Being precise about this is the difference between a language and a marketing
claim.

**Proven at compile time**

* the graph is acyclic (`E-graph-cycle`, naming the cycle)
* every relationship is declared, and every declared relationship is real
* every binding has exactly one producer, unless the alternatives are guarded
* every repetition has a bound, and the bound is at least one round
* a two-guard selection whose guards are syntactically complementary is mutually
  exclusive and exhaustive
* effects are drawn from the specification's algebra
* a compute operation does not depend on a committed change
* a transition alters a declared state and does not retype it
* the intent has an outcome, and something produces it
* types, arity, library signatures, effect declarations, secret propagation and
  dispatch exhaustiveness — checked natively, by `core/native.py`

**Checked at runtime, with a classified fault**

* a constraint that depends on data (`ContractViolation`, quoting the constraint)
* a selection whose guards were not provably complementary
  (`NoActiveAlternative`)
* a refinement that reaches its bound (`RefinementDiverged`, naming the
  constraint and the bound)

**Not checked at all yet**

* Guard exhaustiveness for more than two alternatives. The compiler records
  `proven_exhaustive = False` and relies on the runtime fault. Proving
  exhaustiveness over arbitrary predicate sets is a decision-problem, and
  pretending otherwise would be the kind of claim this project forbids.
* Whether an operation's declared `effect` matches what it actually does. The
  checker verifies that the library calls used are covered by the declared
  effect, but it does not re-derive the effect from the operation's own clauses.

---

## 8. What v0.2 does not do

Stated plainly, because the audit's central complaint about the original
material was over-claiming.

* **No native backend.** v0.2 runs on the reference VM. Section 14 of the audit
  sequences native backends *after* the redesign, and that is the order being
  followed.
* **No performance claim of any kind.** Nothing here is benchmarked against
  anything. See §9 for the methodology that would have to be satisfied first.
* **No parallel execution.** Same-level operations are *known* to be independent,
  which is the precondition for parallelising them, and since v0.3 that
  knowledge is recorded on the GIR function — but the emission is still
  sequential.
* **`each` yields a `List`.** Its declared type is the element type; there is no
  choice of container yet.
* **No modules, imports, generics or user-defined types in the core.** A core
  program is one intent. Composition between intents is not designed yet.
* **Recovery is bounded repetition plus classified faults.** The richer recovery
  model in the specification (retry/compensate/escalate policy) is not yet
  expressible in core syntax, though the reference runtime implements it.
* **Sources with no `from` make the program a library.** It compiles and can be
  checked, but there is no generated `main`, so `ggc run` has no entry point.

---

## 9. Benchmark methodology (audit section 9)

The audit is right that a speed target cannot currently be verified, and right
that it should stay a measurable engineering target rather than a claim. No
benchmark result is stated anywhere in this repository. When one is, this is the
methodology it must satisfy — every field reported, or the number is not
reported at all:

| Field | What must be stated |
|---|---|
| **Workload** | the program, its inputs and their size, and why that workload is representative |
| **Hardware** | CPU model and count, core/thread configuration, clock behaviour (governor, turbo), RAM, storage class, OS and kernel |
| **Compiler version** | the git commit, not a release label |
| **Optimization level** | `ggc build -O<n>`, and whether the deterministic profile was on |
| **Reference implementation** | what is being compared against, its version, and how the two were made to compute the same thing |
| **Warmup** | iterations discarded and why |
| **Variance** | run count, the dispersion measure, and the outliers policy |
| **Throughput / latency** | which one, over what window, at what percentile |
| **Memory** | peak RSS and allocator behaviour, measured the same way for both sides |

Until a native backend exists, any Gama-G number is a *reference-VM* number, and
comparing it to a compiled language would measure the interpreter rather than
the language. That comparison will not be published.

---

## 10. Relationship to v0.1

| | v0.1 | v0.2 core |
|---|---|---|
| Status | research / vertical-slice reference implementation | the language core |
| Program model | statements in a function | operations in a derived graph |
| Order | written by the programmer | derived by the compiler |
| Assignment | `var`, `=` | none; single-assignment bindings, plus `state`/`transition` |
| Repetition | `while`, `for` | `refine` with a mandatory bound |
| Selection | `if`, `match` | guarded sibling operations, `resolve` |
| Selected by | absence of the `gama core` pragma | `gama core <version>` as the first tokens |

Detection is on tokens, not text, so a comment or a string containing the words
cannot change a program's dialect (`driver.py::is_core_dialect`).

v0.1 is kept for three reasons. It is the machine that runs v0.2. It is the
tested evidence that the assets the audit lists in section 4 — effect system,
capabilities, secret propagation, hash-chained audit, determinism, bounded
recovery, tensors, GIR, reference VM — actually work. And deleting a working
implementation to make a report look tidier would be a worse engineering
decision than the one the report is criticising.

---

## 11. Where to look

| Path | Contents |
|---|---|
| `compiler/gamag/core/mir.py` | the native semantic IR *(v0.3; was `core/ast.py`)* |
| `compiler/gamag/core/parser.py` | the declaration grammar, clause table, verbatim text capture |
| `compiler/gamag/core/graph.py` | relationship validation, cycle detection, level derivation, guard proofs |
| `compiler/gamag/core/native.py` | native checking and lowering to GIR *(v0.3; was `core/elaborate.py`)* |
| `compiler/gamag/driver.py` | `is_core_dialect`, the two-phase front end, `Compilation.core_graph` |
| `compiler/gamag/cli/main.py` | `ggc graph [--edges] [--json]` |
| `examples/core/*.gg` | six runnable programs, one per construct family |
| `tests/test_core_language.py` | 49 tests: derived order, relationships, selection proofs, bounded repetition, constraints, state and authority, fan-out and dispatch, surface rules, examples, originality guarantees |
