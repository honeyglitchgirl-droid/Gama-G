# The conformance suite

`ggc conform` checks this toolchain against
`Gama-G_v1.0_Production_Specification.txt` — not against its own past
behaviour, and not against its own tests.

That distinction is the whole point.  A regression suite that compares the
implementation with yesterday's implementation can tell you nothing changed
that you did not mean to change.  It cannot tell you the implementation never
matched the document, because the document is not in the loop.  The suite here
puts it in the loop: it reads the specification at run time and checks the
claims below against what it reads.

```sh
./tools/bin/ggc conform              # 0 = passed, 1 = claimed something false
./tools/bin/ggc conform -k types     # a substring filter over claim ids
./tools/bin/ggc conform --json       # the same report, for tooling
./tools/bin/ggc conform --strict     # 4 = nothing failed, but see what is listed
```

## Exit codes

| code | meaning |
|---|---|
| 0 | every claim passed, and no deviation was discovered that is not recorded |
| 1 | a claim failed, or the specification/suite could not be read (a suite that cannot find its document has checked nothing, and saying so as an error is the only honest answer) |
| 3 | `--strict` was combined with `-k`; a filtered run cannot see the whole picture, so it is refused rather than answered misleadingly |
| 4 | every claim passed, but `--strict` found something not fully evidenced |

The report prints one line per claim with its section citation, and the detail
underneath where there is one, so a passing run is also the evidence for the
pass rather than a tick.

`--strict` never changes whether a claim passed.  `ok` and `strict_failures`
answer two different questions — "is what we claim true?" and "is there
anything we are not claiming?" — and collapsing them would make the first
question unanswerable, because a run in which everything is either satisfied or
excused cannot fail at all.

## The files

| file | holds |
|---|---|
| `claims.json` | the claims.  Each has an `id`, the `spec` section it cites, a `kind`, and the expectation for that kind |
| `deviations.json` | places the toolchain knowingly differs from the specification, each with what the document says, what happens instead, why, and a spelling that *does* work |
| `stage-mapping.json` | all 14 compilation stages of section 22, each marked `implemented`, `partial`, or `not-implemented`, with a reason |
| `requirements.json` | all 20 v1.0 requirements of section 33, each `evidenced`, `partial`, or `not-claimed`, with a reason |
| `programs/*.gg` | the programs the claims run |

The four JSON files are the suite; `ggc conform` is only the runner.

## Claim kinds

| kind | what it does |
|---|---|
| `type-surface` | extracts the type lists from section 5 and tries to declare each name as written.  A name that does not work must be in `deviations.json`, with a working alternative that is *also* compiled — an alternative nobody runs is a claim, not evidence |
| `pipeline` | extracts the numbered stages from section 22 and requires each to be classified |
| `requirements` | extracts the `[ ]` items from section 33 and requires each to be answered |
| `program` | runs a program under `programs/` and compares stdout (and optionally the exit status) with the recorded expectation |
| `fault` | runs a program and requires it to die of a named fault |
| `refused` | compiles a program and requires that exact diagnostic code: the toolchain must refuse it, and refuse it for the stated reason |

`program` claims may set `"repeat": n` — the program is run *n* times and all
runs must agree.  That is how determinism is checked: not by inspecting the
implementation for a seeded generator, but by running twice and comparing.

## The rule that makes the suite worth having

**A claim that fails is a finding, never an edit.**  When the toolchain and the
document disagree there are exactly two honest outcomes:

1. fix the toolchain, or
2. record the disagreement in `deviations.json` (or answer the requirement
   `partial` / `not-claimed`) with a stated reason, where `--strict` will
   report it forever after.

Weakening a claim so that it passes is the third option, and it destroys the
artifact: a suite whose expectations were adjusted to match observed behaviour
is a very elaborate way of writing a mirror.  `tests/test_conformance.py` tries
to make that mistake hard — it mutates copies of the suite and requires the
runner to fail on an unrecorded deviation, on an alternative that does not
compile, on a *stale* deviation (recorded, but no longer true), on an invented
stage, on a dropped requirement, and on a wrong expectation.

## Running it against something else

`--spec PATH` checks another document, `--suite DIR` another suite directory,
and `$GAMA_SPEC` / `$GAMA_CONFORMANCE` do the same without arguments.  Nothing
subtler than searching for `Section 5` headings and `[ ]` lists is required,
and a document that does not have them is refused rather than passed: if no
section 5 types, section 22 stages, or section 33 requirements are found, that
is a `SpecError`.

## What the suite currently reports

13 claims, all passing, 3 recorded deviations (`Record`, `Enum`, `Matrix` —
see `deviations.json` for why and for the spellings that do work).  Under
`--strict`, 13 items are listed: those 3 deviations and 10 of the 20 section 33
requirements answered `partial` or `not-claimed`.  That list is why the package
classifier still says Alpha, and `docs/RELEASES.md` states the bar for leaving
it.
