# Gama-G v0.3 — a native semantic IR

This document answers section 13 of
`Gama-G_Complete_Originality_and_Technical_Audit.txt`, and priorities 1 and 2 of
its section 15.

Section 13's finding was structural, and it was correct. v0.2 had a new *source
language*, but it compiled by elaborating that language into the older
implementation's abstract syntax. So `let`, `var`, `if`, `while`, `for` and
`match` — the very constructs v0.2 exists to avoid — reappeared one layer down,
as the intermediate representation. The audit's summary of that:

> the compiler still translates the new surface into the old model before doing
> anything with it

A language whose meaning is expressed in another language's control flow is not
finished being designed, whatever its surface looks like. v0.3 removes that
layer.

---

## 1. What changed

| | v0.2 | v0.3 |
|---|---|---|
| Intermediate representation | the older language's AST (`let`/`if`/`while`/`match`) | a native semantic model of operations, dependencies, constraints, authority, transitions, refinement and outcomes |
| Path to GIR | core source → older AST → `gir/builder.py` | core source → semantic model → `core/native.py::lower` → GIR |
| `core/ast.py`, `core/elaborate.py` | present | **deleted** |
| Where faults come from | calls to a `panic` library function | GIR `FAULT` terminators carrying a core fault kind |
| Where the derived graph ends up | printed by `ggc graph`, then lost | printed by `ggc graph`, **and** recorded on the GIR function |

There is no compatibility path between the two. `core/ast.py` and
`core/elaborate.py` are gone from the tree, so nothing can silently fall back to
elaborating.

`tests/test_core_language.py::NativeLowering::test_no_older_abstract_syntax_is_built_for_a_core_program`
asserts the property directly: for a core program, `compilation.module` — the
older AST — is `None`.

---

## 2. The pipeline

```
source text
   │  lexer                      (shared: tokenisation is not a language model)
   ▼
tokens
   │  core/parser.py             CoreParser
   ▼
CoreSyntax                       the program as written
   │  core/graph.py              build()
   ▼
SemanticModel                    the program as meant  ← five graphs
   │  core/native.py             check()
   ▼
SemanticModel + NativeChecker    types resolved, authority settled
   │  core/native.py             lower()
   ▼
GProgram                         GIR, with the derived graph attached
   │  gir/optimizer.py
   ▼
GProgram
   │  runtime/vm.py
   ▼
result / classified fault
```

Two things in that list are deliberately shared with the older front end, and
both are named in the driver's docstring:

* **the lexer**, because tokenisation is not a programming model;
* **the type lattice and standard-library signatures** (`semantic/types.py`,
  `stdlib/`). Two competing definitions of what `F64` means would be a defect,
  not independence.

Everything between them is the core's own.

---

## 3. The native semantic IR (`core/mir.py`)

Priority 1 asked for an IR that represents the language's concepts directly
rather than translating them into statements. `mir.py` is that IR.

### 3.1 Expressions

`MExpr` and its subclasses: `MLit`, `MRef`, `MBin`, `MUn`, `MCall`, `MField`,
`MIndex`, `MItems`. There is no statement in this family, no assignment and no
control-flow node — an expression computes a value and nothing else.

`MPattern` and its subclasses (`PWild`, `PLit`, `PBind`, `PTag`) exist so that
`resolve … choose` is a first-class part of the model instead of being
re-expressed as a chain of comparisons.

### 3.2 Constraints

```python
@dataclass
class Constraint:
    kind: str          # holds | when | until | guard
    text: str          # the source, verbatim
    expr: Optional[MExpr]
    discharge: str     # proven | runtime | unprovable
    node: str
```

`text` is kept verbatim so that a fault can quote the program back to the
programmer in the words they wrote, rather than in a rendering of a
post-analysis tree.

`discharge` is the part that matters. It records **how the constraint is
satisfied**, and it has exactly three values:

| Value | Meaning |
|---|---|
| `DISCHARGE_PROVEN` | established at compile time; no runtime check is emitted |
| `DISCHARGE_RUNTIME` | cannot be established statically; a check is emitted and its failure is a classified fault |
| `DISCHARGE_UNPROVABLE` | the compiler knows it cannot prove this here, and says so |

The third value exists because the alternative is worse. Guard exhaustiveness
over an arbitrary set of predicates is a decision problem; the compiler does not
solve it, and rather than pretending, it records `unprovable`, emits the runtime
check, and `ggc graph` prints the constraint under that heading.
`ConstraintGraph.unprovable()` returns them.

`test_the_five_graphs` checks all three values occur and are distinguished.

### 3.3 Operations

`OpNode` is the base; `SourceNode`, `StateNode`, `ComputeNode`,
`RefinementNode`, `FanOutNode`, `DispatchNode` and `TransitionNode` are the
kinds. A node carries its `produces` binding, its declared `type`, its `effect`,
the authority it `needs`, its `trail` prose, its `holds` constraints, its `when`
guard, and — filled in by the graph builder — its `level`, `phase`, `upstream`
and `downstream`.

Note what is absent: a body made of statements. `computes` is a single
expression. Repetition is a property of a node (`starts` / `repeats` / `until` /
`within`), not a loop wrapped around a block. Selection is a property of a group
of nodes (`Selection`), not a branch inside one.

### 3.4 The five graphs

`SemanticModel` is five views over one program, each answering a different
question:

| Graph | Question it answers |
|---|---|
| `IntentGraph` | what is this program for, what outcome does it produce, what authority does it hold, what must it record |
| `OperationGraph` | what are the operations, what does each consume and produce, in what derived order can they run, which bindings have alternatives |
| `ConstraintGraph` | what must be true, and is that proven, checked at runtime, or not provable here |
| `AuthorityGraph` | what capability does each operation demand, and does the intent hold it |
| `RecoveryGraph` | what can fail, what is the bound on each retry, and what fault kind results |

They are separate objects because they are separate concerns with separate
failure modes, and because a tool that wants the security story should not have
to walk the data-flow story to find it. `ggc graph` prints all five.

This matches the architecture the audit recommended in section 15 — operation
graph, dependency graph, constraint graph, security/capability graph and
recovery graph feeding one unified IR — with the dependency graph expressed as
`OperationGraph.levels` plus each node's `upstream`/`downstream`, which is the
same information.

---

## 4. Native checking (`core/native.py::NativeChecker`)

The checker works on the semantic model. It resolves every type, infers every
expression, and settles authority — then it hands its own resolved environment to
the lowerer.

That last part is a real constraint rather than an optimisation. Lowering needs
the type of every expression as it was seen *in the environment the checker
built*. Constructing a second checker for lowering produces an empty environment
and every reference fails to resolve.
`driver.py::_compile_core` therefore keeps the checker returned by `check()` and
passes it to `lower()`.

### 4.1 Diagnostics

The core emits 33 distinct codes. They fall into groups:

* **graph shape** — `E-no-yields`, `E-no-yields-type`, `E-no-computes`,
  `E-no-outcome`, `E-duplicate-binding`, `E-refine-incomplete`,
  `E-each-incomplete`, `E-resolve-incomplete`, `E-transition-target`,
  `E-state-uninitialised`
* **relationships** — `E-unresolved-relationship`, `E-undeclared-relationship`,
  `W-unused-relationship`, `E-graph-cycle`, `E-compute-after-commit`
* **selection and dispatch** — `E-ambiguous-selection`,
  `E-dispatch-not-exhaustive`, `E-dispatch-unsupported`
* **types** — `E-unknown-type`, `E-type-arity`, `E-yield-type`, `E-binop-type`,
  `E-unop-type`, `E-arg-type`, `E-arity`, `E-unknown-call`, `E-constraint-type`
* **authority, effects and secrecy** — `E-authority-unmet`, `E-unknown-effect`,
  `E-effect-undeclared`, `E-secret-escape`
* **recovery** — `E-unbounded-refinement`
* **compiler health** — `E-ice`

Each is emitted at a source position, in a phase, with a `help:` line that says
what to write instead. `E-ice` is the one that should never fire: an unexpected
exception while lowering is reported as a compiler bug rather than allowed to
reach the user as a traceback.

### 4.2 Secret handling

Two rules, which point in opposite directions on purpose:

* A secret value reaching a non-secret binding is `E-secret-escape`. The marking
  must propagate; it cannot be laundered by an operation.
* A non-secret value reaching a binding *declared* secret is allowed. That is a
  strengthening, not an escape, and refusing it would punish the safer
  declaration.

A secret source's `from` clause is checked against the unwrapped type, because
that clause is where the secret enters the program — the declaration is what
makes it secret, not the origin.

Lowering also enforces one thing statically that cannot be checked by type:
**an `AUDIT` record never contains the value of a secret binding.** The trail
text is recorded; the secret is not.

---

## 5. Native lowering (`core/native.py::Lowerer`)

Priority 2: emit GIR from the semantic model directly.

### 5.1 One function per intent

An intent becomes a GIR function whose `kind` is `"intent"` — not `"fn"`. Its
parameters are the intent's sources, its capabilities are the intent's
`authority`, and its return type is the outcome's type.

Inside it, lowering proceeds in two phases that mirror the graph:

1. **compute** — every node with `phase == "compute"`, in derived level order;
2. **commit** — every `transition`, after all derivation.

Resources (`state`) are copied into slots at entry, so a transition writes to a
slot the rest of the program reads. The final instruction returns the outcome.

`E-compute-after-commit` is what keeps that split honest: a compute node may not
depend on a transition, so no derivation can observe a mid-program state change.

### 5.2 Blocks are named after the core's concepts

A lowering that emits `if` and `while` has not escaped the older model; one that
emits named regions has. The block labels come from the construct, not from a
control-flow keyword:

| Construct | Blocks emitted |
|---|---|
| `refine G … until c within n rounds` | `refine.G.until`, `refine.G.round`, `refine.G.diverged`, `refine.G.next`, `refine.G.done` |
| guarded alternatives for `s` | `select.s`, `select.<node>`, `select.<node>.next`, `select.s.join` |
| `each … over xs as item` | `each.X.more`, `each.X.item`, `each.X.keep`, `each.X.skip`, `each.X.done` |
| `resolve … choose` | one block per arm plus `resolve.R.join` |

`refine.G.diverged` is the block that faults. `each.X.skip` is the block a
guarded fan-out item jumps to when its guard is false. Reading the GIR tells you
which core construct you are inside.

Every block ends with a terminator. That is asserted over all six examples by
`test_every_basic_block_ends_with_a_terminator`, and it is not a style rule: the
reference interpreter treats the end of a block as `return ()`, so a missing jump
would silently truncate the program instead of failing.

### 5.3 Faults are terminators, not calls

v0.2 emitted `panic("RefinementDiverged: …")` — a library call, with the
classification encoded in a string prefix. v0.3 emits a GIR `FAULT` terminator
with `kind` in its metadata:

```
b3:    ; refine.G.diverged
      fault message=`G` did not satisfy `g > 100` within 7 rounds
```

The runtime's fault object carries `kind` and `message` separately, so a caller
can match on the classification without string-parsing. The kinds are the core's
own: `RefinementDiverged`, `NoActiveAlternative`, `UnresolvedDispatch`,
`ContractViolation`. The message does not repeat the kind — that is what `kind`
is for.

### 5.4 The derived graph survives into the GIR

GIR functions carry operation-graph metadata: per-task reads, writes and
dependencies. The lowerer populates it from the derived levels, so the graph is
not lost at the boundary:

```
function T [intent]
  operation graph: 2 task(s)
    - First: reads=['a'] writes=['x'] depends_on=-
    - Second: reads=['x'] writes=['y'] depends_on=First
```

A future backend that wants to parallelise a level reads this instead of
re-deriving it. `test_the_derived_graph_is_recorded_in_the_gir` asserts it.

This is also where the parallelism claim stops: the metadata says the tasks are
independent. The emitted code still runs them in sequence, because the reference
VM is single-threaded and nothing here measures anything.

---

## 6. Proven at compile time, checked only at runtime, and not checked at all

Section 14 of the audit asks that these three be kept apart, and that tests not
be presented as evidence for claims they cannot support. So:

### Proven at compile time

* the graph is acyclic (`E-graph-cycle`, naming the cycle)
* every relationship is declared, and every declared relationship is real
* every binding has exactly one producer, unless the alternatives are guarded
* a two-guard selection whose guards are syntactically complementary is mutually
  exclusive and exhaustive — recorded `DISCHARGE_PROVEN`
* every repetition has a bound, and the bound is at least one round
* effects are drawn from the specification's algebra
* an operation may not demand authority its intent does not hold
  (`E-authority-unmet`)
* a compute operation does not depend on a committed change
  (`E-compute-after-commit`)
* a transition alters a declared state and does not retype it
* the intent has an outcome and something produces it
* types, arity, library signatures, secret propagation, dispatch exhaustiveness

### Checked only at runtime

These are emitted as checks because they depend on data, and each failure is a
classified fault:

* a `holds` constraint — `ContractViolation`, quoting the constraint verbatim
* a selection whose guards were not provably complementary —
  `NoActiveAlternative`, naming the binding and the number of alternatives
* a refinement that reaches its bound — `RefinementDiverged`, naming the
  constraint and the bound
* a dispatch that matches no arm and has no `_ =>` — `UnresolvedDispatch`

### Not checked at all

Stated because the alternative is a claim the evidence does not support:

* **Guard exhaustiveness beyond two alternatives.** Recorded `unprovable`; the
  runtime check stays. Proving exhaustiveness over arbitrary predicates is a
  decision problem and the compiler does not solve it.
* **Whether a declared `effect` matches what the operation actually does.** The
  checker verifies that library calls used are covered by the declared effect,
  but it does not re-derive the effect from first principles.
* **Termination of `computes` expressions.** Bounded by construction only for
  `refine`.
* **Anything about performance.** No benchmark exists in this repository, and
  §7 of `DESIGN_v0_2.md` records the methodology any future number must satisfy.

---

## 7. What v0.3 does not do

Carried forward from v0.2, and still true:

* **No native backend.** The GIR is executed by the reference VM. Priority 3 of
  the audit (a native CPU backend) is sequenced after this and has not started.
* **No performance claim of any kind.** Nothing is benchmarked against anything.
* **No parallel execution.** Levels are known independent; the emission is still
  sequential.
* **No WASM, GPU, `gpm`, FFI, FHIR or memory model.** Priorities 4–16 of the
  audit are untouched.
* **No modules, imports, generics or user-defined types in the core.** A core
  program is one intent.
* **Sources with no `from` make the program a library.** It compiles and checks,
  but there is no generated entry point, so `ggc run` has nothing to run.
* **No fuzzing, no differential testing against another implementation.** There
  is no other implementation of this language to differ from.

---

## 8. Relationship to v0.1

v0.1 is still in the tree, and its role changed in v0.3.

| | v0.2 | v0.3 |
|---|---|---|
| v0.1 as an IR | yes — the core elaborated into it | **no** |
| v0.1 as a runtime | yes | yes |
| v0.1 as a language | yes, for `.gg` files without the pragma | yes, unchanged |
| v0.1's type lattice and stdlib | shared | shared |

The audit is explicit that deleting the working implementation to make a report
look tidier would be a worse decision than the one it criticises. v0.1 remains as
the machine that runs the core and as the tested evidence that the assets it
lists in section 4 — effect system, capabilities, secret propagation, hash-chained
audit, determinism, bounded recovery, tensors, GIR, reference VM — actually work.

What v0.3 removes is v0.1's role as the *meaning* of a core program. That role
now belongs to `core/mir.py`.

---

## 9. Where to look

| Path | Contents |
|---|---|
| `compiler/gamag/core/mir.py` | the native semantic IR: expressions, patterns, constraints with discharge, node kinds, the five graphs |
| `compiler/gamag/core/parser.py` | `CoreParser`, the clause table, verbatim constraint capture |
| `compiler/gamag/core/graph.py` | `build()` — relationship validation, cycle detection, level derivation, guard proofs, authority and recovery collection |
| `compiler/gamag/core/native.py` | `NativeChecker` (types, authority, effects, secrecy) and `Lowerer` (semantic model → GIR) |
| `compiler/gamag/driver.py` | `is_core_dialect`, `_compile_core`, `Compilation.core_syntax` / `core_model` / `core_graph` |
| `compiler/gamag/gir/ir.py` | `Op`, `TERMINATORS`, `GFunction.parallel_tasks`, `kind == "intent"` |
| `compiler/gamag/cli/main.py` | `ggc graph [--edges] [--json]` over the five graphs |
| `examples/core/*.gg` | six runnable programs, one per construct family |
| `tests/test_core_language.py` | `LanguageInvariants`, `NativeLowering`, `NativeChecks`, `TheFiveGraphs`, `ProvenanceEvidence` |
