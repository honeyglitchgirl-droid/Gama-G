# What compiling to native code actually bought, and what it did not

This document is the record of one experiment: measure Gama-G against the
languages it is meant to be an alternative to, find out why it was slow, and fix
the reason.  The measurement is the point of it, so the conditions are stated
with every number.  Spec section 43 forbids claiming capabilities the toolchain
does not have; that applies here.  Nothing below says Gama-G "is fast" or "is
competitive" -- it says what three programs did on one machine on one day, and
what the compiler emitted at the time.

The short version: on a 30-million-iteration integer loop the native backend was
**641x slower than the equivalent C**, and the cause was not the C compiler's
optimisation level -- it was already `-O2`.  The cause was that `ggc native`
emitted a faithful *interpreter of GIR, written in C*: every value a 16-byte
tagged union, every operator selected by comparing the operator's **name** with
`strcmp` at run time, every assignment passing the type's name, the variable's
name and the source position.  Specialising the emitter for functions whose
values are all scalars closed most of that gap.  The measurement below is the
before and after, taken in one session, back to back.

One caution about cross-session ratios, from the data: the same sum workload
measured against the same C binary gave 893x in an earlier session, where the C
baseline ran in 1.1 ms rather than 1.63 ms.  The emitted code did not change
between those two runs.  Ratios move when the machine does, which is why every
number below comes from one run of one harness.

---

## 1. What was measured, and how

The harness is not in the repository -- it needs four other language runtimes to
be installed, and the repository should not depend on that.  It is reproducible
from what is here: the three programs are quoted in full in section 2, and
`ggc native` builds the Gama-G ones.

Conditions, as the harness printed them:

```
machine : x86_64  Linux
python  : 3.11.2
gcc     : gcc (Debian 12.2.0-14+deb12u1) 12.2.0
node    : v22.22.3
comment : nothing else was running on this machine during the run
repeats : minimum of 5 (interpreter 2), one warmup
```

* **Estimator.** The minimum of 5 runs, after one warmup, for the compiled
  implementations: for a compute-bound benchmark the fastest run is the one
  least disturbed by the scheduler.  The interpreter is the expensive column, so
  it gets 2 runs and no warmup.  Medians are in `results2.json` next to the
  minima and are not flattering to anyone in particular.
* **Before/after in one session.** The "boxed" column is *this compiler* with the
  unboxed emitter disabled -- `scalar_plan` returning nothing is the previous
  emitter exactly.  Both binaries are built and timed in the same run, on the
  same machine, minutes apart.  Comparing against a remembered number from an
  earlier session would have measured the machine's mood as much as the change.
* **Correctness before speed.** Every implementation prints a value, and each
  one is checked against the expected value *before* it is timed.  The two
  Gama-G binaries are additionally checked against each other: they are the same
  program through two different emitters, and they must agree.
* **The interpreter's ceiling.** The interpreter stops after 5 000 000
  basic-block transitions **per activation** with a `RunawayLoop` fault.  That
  is deliberate -- a runaway program must not hang the toolchain -- and it means
  the interpreter does not *answer* the first two workloads at all.  Those two
  cells say "did not answer", not a time: reporting the time-to-fault next to
  other languages' time-to-answer would be a fabricated comparison.

`-O2` was already the default `--cc-opt`, so none of this is a missing compiler
flag.  `-O3` was not tried; the interesting variable turned out to be what the
emitter wrote, not what gcc did with it.

---

## 2. The three workloads

They are small on purpose, and each one stresses one thing: a tight integer
loop, a call-heavy recursive-ish workload, and allocation.

**sum: a loop of integer arithmetic, 30 million iterations.**

```
fn main() -> Unit
    io
    var i = 0
    var total = 0
    while i < 30000000
        total = total + i
        i = i + 1
    print(total)
```

**collatz: a function called 300 000 times, with branches.** `total_steps(n)`
returns the number of steps to reach 1, and the program sums it over
`1..300000`; it is call-heavy in the way a real program is, and it is the
workload where the unboxed emitter pays most.

**strbuild: 4000 allocations of a 10-byte string, then `len`.** This is the
workload the optimization does *not* help, and it is in the table because a
comparison that only contains the cases an optimization helps is an
advertisement.

All three are written in each language to do the same work with the same
integer types; the C versions are the ones `ggc native` emits for the Gama-G
programs, which keeps the comparison between languages honest about what is
being compared (same C compiler, same flags, same runtime for the Gama-G side).

---

## 3. The numbers

Minimum of 5 runs after one warmup, `min` in milliseconds; the interpreter gets
2 runs.  "before" and "after" are the same compiler in the same session, with
the unboxed emitter disabled and enabled.

| workload | C `-O2` | **Gama-G native, before** | **Gama-G native, after** | Node 22 | Python 3.11 | Perl 5.36 | Gama-G interpreter |
|---|---|---|---|---|---|---|---|
| sum, 30M iterations | 1.63 | 1046.97 | **20.75** | 72.00 | 3015.27 | — | did not answer |
| collatz, 300K calls | 61.12 | 2462.08 | **78.65** | 330.89 | 2044.78 | — | did not answer |
| string build, 4000 allocations | 1.12 | 36.34 | **43.11** | 30.87 | 12.09 | 1.96 | 210.95 |

The same table, as the factor against the fastest implementation in the row:

| workload | C `-O2` | Gama-G before | **Gama-G after** | Node 22 | Python 3.11 | Perl 5.36 | Gama-G interpreter |
|---|---|---|---|---|---|---|---|
| sum | 1.00x | 641x | **12.7x** | 44.1x | 1847x | — | did not answer |
| collatz | 1.00x | 40.3x | **1.29x** | 5.41x | 33.5x | — | did not answer |
| string build | 1.00x | 32.6x | **38.6x** | 27.6x | 10.8x | 1.75x | 189x |

### 3.1 The one row that did not improve, and why it is not a regression

`strbuild` gets **slower** by the table -- 36.34 ms before against 43.11 ms
after -- and that number should not be believed.  The `strbuild` loop concatenates `Text`, so the function holding it cannot be specialized at all:
`main` has a `Text` slot, which is not a scalar.  The only function in the whole
program that the plan touches is the module initializer, which is empty.  The
two generated C files differ in exactly twenty lines, all of them the
initializer:

```
$ diff .ggbuild/on/strbuild.c .ggbuild/off/strbuild.c
14d13
< static void gf__main__s(void);
37,42d35
<     gf__main__s();
<     return g_unit();
...
```

Two binaries with the same hot loop are the same speed, and the difference is
the estimator: the medians are 46.35 ms and 47.11 ms, three per cent apart, and
the "before" minimum is the outlier.  The honest reading is **the unboxed
emitter does nothing for allocation-heavy code** -- it is not a faster build of
the same program, it is the same program.  A min-of-5 in a table with a single
significant figure in the last place will occasionally look like a 20% effect;
the generated code is what settles it.

---

## 4. What the emitted code looked like

This is the whole finding.  The loop body from `sum.gg`, before -- generated by
the boxed emitter:

```c
  b2: ;
    s3 = g_store_int(g_binop("+", s1, s0, ":6:23"), -9223372036854775807LL - 1, 9223372036854775807LL, "I64", "t1", ":6:23");
    s1 = g_store_int(s3, -9223372036854775807LL - 1, 9223372036854775807LL, "I64", "total", ":6:9");
    s4 = g_store_int(g_binop("+", s0, g_int(1LL), ":7:15"), -9223372036854775807LL - 1, 9223372036854775807LL, "I64", "t2", ":7:15");
    s0 = g_store_int(s4, -9223372036854775807LL - 1, 9223372036854775807LL, "I64", "i", ":7:9");
    goto b1;
```

and after -- the same loop from the same compiler, with the unboxed emitter:

```c
  b2: ;
    s3 = (int64_t)(g_iadd(s1, s0, ":6:23"));
    s1 = (int64_t)(s3);
    s4 = (int64_t)(g_iadd(s0, (1LL), ":7:15"));
    s0 = (int64_t)(s4);
    goto b1;
```

Four operations, four boxed values, four 16-byte structs copied by value.  And
each addition compares the operator's name against operator names **ten times**
-- nine in `g_binop`, which walks its whole dispatch chain before reaching `"+"`,
and one more in `num_binop`, which searches again -- so twenty comparisons per
iteration of a loop whose body adds two numbers.

That count is measured, not counted by eye: `g_binop("+", ...)` was called once
in a program whose `strcmp` counts its own invocations, compiling the runtime
with `-fno-builtin-strcmp` (with gcc's builtins on, most of the ten are inlined
into byte comparisons, so a counted call is not the same as a comparison).

After the change: four intrinsics and four register moves, and no comparison of
anything.

### Why the boxed emitter was like that, and why it was not simply wrong

Every one of those arguments is doing something.  `g_binop("+", ...)` is how one
emitter serves every numeric type, including the dynamic cases (`int ** int`
with a negative exponent returns a *Float*, which no static signature expresses).
`g_store_int(v, lo, hi, "I64", "total", ":6:9")` is the integer range check that
makes an `I8` overflow a fault naming the type, the variable and the position --
the same fault the interpreter raises, which is what the differential tester
requires.  The mistake was not the design of the runtime API; it was that
`ggc native` used it for the hot loop of a statically typed program, where all
four of those answers are known at compile time.

---

## 5. What changed

`cgen.py` gained a second emitter.  `scalar_plan()` decides, before anything is
emitted, which functions can be written with machine types: every slot and the
return are `I64`, `F64`, `Bool` or `Unit`; every operation is one the unboxed
emitter knows; no global appears outside a call argument (a global handed to
`print` stays a boxed `GValue` and never enters the arithmetic -- what is refused
is a global in `STORE_GLOBAL` or as an operand of an unboxed operation).  A function that calls another function qualifies
only if that callee does too, computed as a fixpoint, because the call is
emitted as a direct C call with native arguments.

The semantics are not approximations of the boxed path, they are the same
functions: the overflow, division and range checks call the same runtime code,
so the same fault with the same message comes out.  Where the two paths could
have differed on a *detail nobody looks at until it is wrong* -- an integer of
`-2^63` divided by `-1`; a remainder that takes the dividend's sign; a float
comparison against a NaN; the way a failed division names its operands,
`"1.5 / 0"` against `"1.0 / 0.0"` -- the detail is reproduced deliberately and
tested.

Which programs compile is unchanged.  `unsupported()` -- the analysis that
refuses a program the backend cannot compile correctly -- was not touched, and
the differential tester's corpus still agrees in the same 4 places and is
refused in the same 12.

### The one structural decision

A specialized function is emitted **twice**: the body under `gf_name_s`, and a
wrapper under the generic name `gf_name` that unboxes its arguments, calls the
body, and boxes the result.  A specialized function can therefore still be called
from a function that is not specialized -- `main` calling a helper is the common
case, and it is the case that made the first version of this fail to compile at
all, with a `GValue` passed to an `int64_t` parameter.

The alternative -- specialize only functions that nothing plain calls -- would
have given up the optimization for every program with a helper in it.  The
wrapper's cost is one unboxing per call, at the boundary, and its own risk is
that the two ABIs get confused for each other; the names are different, and a
test asserts both prototypes are emitted.

---

## 6. What was found by breaking it

The optimization changed a program's *representation*, and the interesting
failures were all in the details that no test looked at directly.  Recording
them because the way each was caught is more useful than the fix:

| what was wrong | how it was caught | why the obvious test missed it |
|---|---|---|
| a call's result was dropped, because the *caller's* return type was tested instead of the callee's | running the three benchmark programs and comparing the output | the benchmark's expected value, not a test: `collatz` printed `0` |
| `not true` returned `true` -- the emission returned the constant `1` for any Bool operand | a harness that builds every program twice, once per emitter, and compares | the test suite had no `not` on a `Bool`, only `!` and `not` on integers |
| a failed division reported `"division by zero"` where the boxed path reports `"division by zero: 1 / 0"` | the same harness: the fault text is part of the output | the exit status and the fault *kind* were both correct |
| the prototype block emitted two declarations on one line, so the second was parsed as parameters | the existing test suite, immediately | -- |
| the plan filed a function's eligibility under the *builtin's* name (a shadowed loop variable) | printing the plan for the three benchmark programs | it made the function silently not specialized, never wrong |
| the plan refused every function ending in a bare `return`, because a Unit return carries a placeholder operand | the benchmark got *slower* than the baseline | it was a performance bug with no wrong answer to find |
| a NaN compared as equal by the **boxed** runtime, so `<=` and `>=` returned true (this one was already wrong before the optimization; the two emitters merely disagreed) | a new corpus entry that compares `math.nan`, from the same harness -- and the interpreter settled which emitter was right | no example produces a NaN: `0.0 / 0.0` faults here, and `math.nan` has to be spelled out |

The third row is the one worth keeping: no exit code, no fault kind and no test
assertion distinguishes `"division by zero"` from `"division by zero: 1 / 0"`.
Only comparing the two emitters' output does.  That harness is now
`ScalarSpecialisation` in `tests/test_backends.py`, over a corpus of one program
per property, and it is the reason the optimization can be trusted at all.

The last row is the one that is not about this optimization at all: the boxed
runtime had been wrong about NaN since before it, and nothing looked.  A corpus
that compares two emitters finds a bug in whichever one is wrong; it does not
find a bug they *share*.  That is why `NaNIsOrderedWithNothing` asserts the
IEEE answer directly -- `false false false false false true` -- rather than
asserting that the two emitters agree, and why the fix is recorded in
`docs/PRODUCTION_GAPS.md` section 2.6 rather than here.

---

## 7. What this does not claim

* **Not that Gama-G is fast.** Three small programs are not a benchmark suite,
  and one machine is not a result.  See the numbers for what they are.
* **Not that the native backend is competitive with C.** The remaining gap is
  stated in section 3 and it is real.
* **Not that the interpreter is slow.** The interpreter is *bounded*: it refuses
  the first two workloads by policy rather than running them slowly, and the
  bound is the same for every program.  That is a design decision with a cost,
  and the cost is that compute-bound programs cannot be run on the reference
  path at all.
* **Not that other languages were optimised for.** Every implementation here is
  the obvious one, written to do the same work; none was tuned, and the Python
  and Node versions are not written to play to their runtimes' strengths any
  more than the Gama-G version is.

## 8. What would close the rest of the gap

In the order the numbers suggest they are worth:

1. **The string workload.** Every text value is still a heap `GValue`, and every
   concatenation goes through the boxed path -- `strbuild` shows what that costs.
   A specialized `Text` representation is the same kind of change as this one
   and about as large.
2. **The arena release is disabled inside a specialized function**, because the
   release path works on the boxed arena that a specialized body does not have.
   Functions that allocate are rarely the ones this optimization applies to, so
   the loss is small today, but it is a limitation rather than a design.
3. **`CALL_INDIRECT`, method calls and field writes** disqualify a function.
   Modules and models -- the parts of the language that make it Gama-G rather
   than a small imperative language -- are exactly the programs that stay on the
   slow path.
