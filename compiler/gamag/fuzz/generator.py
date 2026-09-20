"""Program generation and mutation for the fuzzer (audit priority 9).

Two generators, because the two failure modes are different and a fuzzer that
only does one finds only one kind of bug:

* **Generation** builds programs from the grammar, mostly well typed.  These
  reach deep into the compiler -- lowering, optimization, the VM -- and find bugs
  in the paths that valid programs take.  A generator that only produced garbage
  would be stopped by the parser every time and would never reach the backend.
* **Mutation** takes the shipped examples and corrupts them at token level.
  These are *almost* valid, which is where the interesting failures live: a
  parser accepts them, a checker has to reason about them, and the diagnostic
  has to be true.  Almost-valid input is also what a real user typo produces.

Neither is random for randomness' sake.  The point of both is to reach a state
where the toolchain must either compile the program or explain why it cannot --
and never do anything else.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

CORE_TYPES = ("I64", "F64", "Bool", "Text")

#: Expressions by result type.  `{a}` and `{b}` are placeholders for operands of
#: the same type, chosen by the generator so that the result type-checks.
CORE_EXPRESSIONS: Dict[str, Tuple[str, ...]] = {
    "I64": ("{a}", "{a} + {b}", "{a} - {b}", "{a} * {b}", "{a} / {b}",
            "{a} % {b}", "math.abs({a})", "({a} + {b}) * 2"),
    "F64": ("{a}", "{a} + {b}", "{a} - {b}", "{a} * {b}", "{a} / {b}",
            "math.sqrt(math.abs({a}))"),
    "Bool": ("{a}", "{a} and {b}", "{a} or {b}", "not {a}"),
    "Text": ('{a}', '{a} + "x"', 'to_text(1)'),
}

#: The effects a generated operation may declare.  Only pure ones: an effect
#: that acquires a resource would need authority the generator does not model,
#: and the resulting refusals would drown out real failures.
CORE_EFFECTS = ("pure",)


@dataclass
class Generated:
    """One generated program, with what the generator intended."""

    source: str
    dialect: str
    kind: str
    notes: List[str] = field(default_factory=list)


def _int_literal(rng: random.Random) -> str:
    choice = rng.random()
    if choice < 0.15:
        return str(rng.choice((0, 1, -1, 2 ** 31, -(2 ** 31), 2 ** 62)))
    if choice < 0.25:
        return str(rng.randint(-1000, 1000))
    return str(rng.randint(1, 50))


def _float_literal(rng: random.Random) -> str:
    if rng.random() < 0.2:
        return rng.choice(("0.0", "1.0", "-0.5", "1e10", "3.141592653589793"))
    return f"{rng.uniform(-100, 100):.6f}"


def _text_literal(rng: random.Random) -> str:
    words = ("alpha", "beta", "", "x", "hello world", "0", "true")
    return '"{}"'.format(rng.choice(words))


def _bool_literal(rng: random.Random) -> str:
    return rng.choice(("true", "false"))


def _literal(rng: random.Random, ty: str) -> str:
    if ty == "I64":
        return _int_literal(rng)
    if ty == "F64":
        return _float_literal(rng)
    if ty == "Bool":
        return _bool_literal(rng)
    return _text_literal(rng)


def _operand(rng: random.Random, ty: str, names: Sequence[str]) -> str:
    """A value of `ty`: an existing binding if one has that type, else a literal."""
    candidates = [n for n, t in names if t == ty]
    if candidates and rng.random() < 0.75:
        return rng.choice(candidates)
    return _literal(rng, ty)


def _expression(rng: random.Random, ty: str,
                names: Sequence[Tuple[str, str]]) -> str:
    template = rng.choice(CORE_EXPRESSIONS[ty])
    if ty in ("I64", "F64") and "/" in template or "%" in template:
        # Division by zero is a runtime fault, not a compile error, and the
        # generator would otherwise produce mostly faults and mostly hide the
        # failures that matter.  A non-zero divisor keeps the program runnable.
        b = _operand(rng, ty, names)
        if b.lstrip("-").replace(".", "", 1).isdigit() and float(b) == 0:
            b = "1" if ty == "I64" else "1.0"
        return template.format(a=_operand(rng, ty, names), b=b)
    return template.format(a=_operand(rng, ty, names),
                           b=_operand(rng, ty, names))


def generate_core(rng: random.Random, *, operations: Optional[int] = None,
                  with_state: bool = False,
                  with_selection: bool = False) -> Generated:
    """A core program: sources, operations in a derived order, an outcome."""
    notes: List[str] = []
    count = operations if operations is not None else rng.randint(1, 4)
    lines = ["gama core 0.2",
             "intent Fuzz",
             "    purpose   a generated program"]

    bindings: List[Tuple[str, str]] = []
    for index in range(rng.randint(1, 2)):
        ty = rng.choice(("I64", "F64"))
        name = f"s{index}"
        lines.append(f"source {name} : {ty} from {_literal(rng, ty)}")
        bindings.append((name, ty))

    yields: List[str] = []
    for index in range(count):
        ty = rng.choice(CORE_TYPES)
        name = f"v{index}"
        uses = [n for n, _ in bindings]
        if not uses:
            uses = [bindings[0][0]] if bindings else []
        lines.append(f"operation Op{index}")
        if uses and rng.random() < 0.8:
            lines.append(f"    uses     {', '.join(uses)}")
        lines.append(f"    yields   {name} : {ty}")
        lines.append(f"    effect   {rng.choice(CORE_EFFECTS)}")
        lines.append(f"    computes {_expression(rng, ty, bindings)}")
        bindings.append((name, ty))
        yields.append(name)

    if with_state and yields:
        lines.append("state total : I64")
        lines.append("    starts 0")
        notes.append("declares a state, so a transition is required")

    if not yields:
        notes.append("no operation produced a value")
        lines.append("outcome s0")
    else:
        outcome = next((y for y, t in zip(reversed(yields),
                                         reversed([b for b in bindings
                                                   if b[0] in yields]))
                        if True), yields[-1])
        lines.append(f"outcome {outcome}")
    return Generated(source="\n".join(lines) + "\n", dialect="core",
                     kind="generated-core", notes=notes)


def generate_v01(rng: random.Random) -> Generated:
    """A v0.1 program: a main with bindings, a conditional and output."""
    lines = ["fn main() -> Unit", "    io"]
    bindings: List[Tuple[str, str]] = []
    for index in range(rng.randint(1, 4)):
        ty = rng.choice(("I64", "F64", "Bool", "Text"))
        name = f"x{index}"
        keyword = "let" if rng.random() < 0.8 else "var"
        lines.append(f"    {keyword} {name}: {ty} = {_literal(rng, ty)}")
        bindings.append((name, ty))

    numeric = [n for n, t in bindings if t in ("I64", "F64")]
    if numeric:
        left = rng.choice(numeric)
        ty = next(t for n, t in bindings if n == left)
        lines.append(f"    let y = {_expression(rng, ty, bindings)}")
        bindings.append(("y", ty))
        lines.append("    print(y)")

    if bindings and rng.random() < 0.5:
        cond = rng.choice([n for n, t in bindings if t == "Bool"] or ["true"])
        lines.append(f"    if {cond} {{")
        lines.append('        print("yes")')
        lines.append("    } else {")
        lines.append('        print("no")')
        lines.append("    }")

    lines.append("    print(" + (rng.choice([n for n, _ in bindings])
                                 if bindings else '"done"') + ")")
    return Generated(source="\n".join(lines) + "\n", dialect="v0.1",
                     kind="generated-v01")


def generate(rng: random.Random) -> Generated:
    """One program of either dialect, plus the deliberately odd cases."""
    roll = rng.random()
    if roll < 0.42:
        return generate_core(rng)
    if roll < 0.72:
        return generate_v01(rng)
    if roll < 0.80:
        return generate_core(rng, with_state=True)
    if roll < 0.86:
        return generate_core(rng, operations=0)
    if roll < 0.92:
        return Generated(source="", dialect="core", kind="empty")
    return Generated(source=_adversarial(rng), dialect="mixed",
                     kind="adversarial")


def _adversarial(rng: random.Random) -> str:
    """Input that is not a program, to check the front end refuses politely."""
    fragments = [
        "gama core 0.2\nintent\n",
        "fn main() ->\n",
        "(((((((((((",
        '"' + "a" * 5000,
        "\x00\x01\x02 binary",
        "outcome " * 200,
        "intent T\n    purpose " + "x" * 10000 + "\n",
        "let x: I64 = " + "9" * 400,
        "gama core 99.99\nintent T\n",
        "operation\n" * 300,
        "// " + "comment\n" * 500,
        "source a : I64 from " + "-" * 50,
    ]
    return rng.choice(fragments)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+\.?\d*|\"[^\"]*\"|\S")

KEYWORDS = ("intent", "purpose", "source", "operation", "uses", "yields",
            "effect", "computes", "outcome", "state", "starts", "transition",
            "alters", "holds", "needs", "trail", "authority", "checkpoint",
            "recover", "retry", "restore", "escalate", "refine", "until",
            "within", "each", "over", "resolve", "choose", "secret",
            "fn", "let", "var", "if", "else", "match", "print", "true",
            "false", "pure", "io", "crypto", "audit")


@dataclass
class Mutation:
    source: str
    kind: str
    detail: str
    origin: str = ""


def mutate(source: str, rng: random.Random,
           origin: str = "") -> Mutation:
    """One token-level corruption of a program that used to be valid."""
    lines = source.split("\n")
    roll = rng.random()

    if roll < 0.12 and len(lines) > 2:
        index = rng.randrange(len(lines))
        removed = lines.pop(index)
        return Mutation("\n".join(lines), "delete-line",
                        f"removed line {index + 1}: {removed.strip()[:40]}",
                        origin)

    if roll < 0.22 and lines:
        index = rng.randrange(len(lines))
        lines.insert(index, lines[index])
        return Mutation("\n".join(lines), "duplicate-line",
                        f"duplicated line {index + 1}", origin)

    if roll < 0.32 and len(lines) > 3:
        i, j = rng.sample(range(len(lines)), 2)
        lines[i], lines[j] = lines[j], lines[i]
        return Mutation("\n".join(lines), "swap-lines",
                        f"swapped lines {i + 1} and {j + 1}", origin)

    if roll < 0.44:
        tokens = TOKEN_RE.findall(source)
        if tokens:
            position = rng.randrange(len(tokens))
            tokens[position] = rng.choice(KEYWORDS)
            return Mutation(_retokenize(source, position, tokens[position]),
                            "keyword-swap",
                            f"token {position} became `{tokens[position]}`",
                            origin)

    if roll < 0.56:
        numbers = re.findall(r"\b\d+\b", source)
        if numbers:
            target = rng.choice(numbers)
            replacement = rng.choice(("0", "-1", "99999999999999999999",
                                      str(2 ** 63), "1"))
            return Mutation(source.replace(target, replacement, 1),
                            "number-swap",
                            f"{target} became {replacement}", origin)

    if roll < 0.66:
        identifiers = sorted(set(re.findall(r"\b[a-z][a-zA-Z0-9_]*\b", source)))
        if len(identifiers) > 1:
            old, new = rng.sample(identifiers, 2)
            return Mutation(re.sub(rf"\b{re.escape(old)}\b", new, source),
                            "rename-identifier", f"`{old}` became `{new}`",
                            origin)

    if roll < 0.76:
        cut = rng.randrange(max(1, len(source)))
        return Mutation(source[:cut], "truncate",
                        f"cut at byte {cut} of {len(source)}", origin)

    if roll < 0.86 and source:
        position = rng.randrange(len(source))
        junk = rng.choice(("@", "#", "\\", ")", "{", ":", ",", '"', "\n\n",
                           "~~~", "\t"))
        return Mutation(source[:position] + junk + source[position:],
                        "insert-junk", f"inserted {junk!r} at {position}",
                        origin)

    if source:
        position = rng.randrange(len(source))
        character = source[position]
        replacement = rng.choice("abcXYZ019 \n(){}[]<>+-*/%=.,:;\"'\\")
        return Mutation(source[:position] + replacement + source[position + 1:],
                        "flip-character",
                        f"byte {position}: {character!r} -> {replacement!r}",
                        origin)
    return Mutation(source, "no-op", "nothing to mutate", origin)


def _retokenize(source: str, index: int, replacement: str) -> str:
    """Replace the nth token in place, keeping the surrounding text."""
    count = -1
    for match in TOKEN_RE.finditer(source):
        count += 1
        if count == index:
            return source[:match.start()] + replacement + source[match.end():]
    return source


def corpus(paths: Sequence[str]) -> List[Tuple[str, str]]:
    """(path, source) for every file the fuzzer should mutate."""
    out: List[Tuple[str, str]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                out.append((path, handle.read()))
        except (OSError, UnicodeDecodeError):
            continue
    return out


def example_corpus(root: str) -> List[Tuple[str, str]]:
    """Every shipped example, which is the seed corpus that matters most.

    These are the programs the documentation tells users to run, so a mutation
    of one that crashes the compiler is a bug a user can reach by mistyping.
    """
    paths: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(".gg"):
                paths.append(os.path.join(dirpath, name))
    return corpus(sorted(paths))
