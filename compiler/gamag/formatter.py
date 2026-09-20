"""``ggc format`` -- the canonical formatter (spec section 31, toolchain).

Every language a team can depend on ships one formatter: ``gofmt`` for Go,
``rustfmt`` for Rust, ``clang-format`` for C.  A formatter is not cosmetics;
it removes formatting from the set of things a review can argue about, and it
makes the layout of a file a property of the language rather than of its
author.

Design
------
Gama-G is indentation-sensitive, so the formatter's one safety property is:

    **the formatted source parses to the same GIR as the original.**

The formatter never invents structure.  It measures the block structure the
lexer itself derives (the same INDENT/DEDENT and continuation rules), and
re-emits every level as exactly four spaces.  Continuation lines -- inside
brackets, or joined by the lexer's trailing/leading continuation rules -- do
not carry layout meaning, so they are placed against their opener without
changing what parses.

What it normalizes:

* indentation: four spaces per level, tabs expanded, inconsistent dedents
  impossible by construction;
* spacing inside a line: operators spaced, ``f(x)``, ``a.b``, ``1..6``,
  ``x: T``, ``a, b``, ``Result<F64, Text>`` all canonical;
* comments: ``//`` text kept exactly as written, full-line comments
  re-indented to the level of the code that follows them (the same skip rule
  the lexer uses, so a comment at a dedent boundary belongs to the next
  declaration, not the block that closed);
* block comments that own their lines: re-indented as a unit, interior
  relative spacing preserved;
* blank lines: runs collapse to at most one; trailing whitespace removed;
  the file ends with exactly one newline.

What it refuses rather than guesses:

* a line mixing code and an inline ``/* ... */`` comment is left as written
  (apart from trailing whitespace) -- moving code around a mid-line block
  comment would be a layout decision the formatter has no authority to make;
* source that does not scan is reported as a diagnostic, never reformatted;
* if both the original and the formatted text compile, their GIR must be
  identical apart from source positions; a disagreement is reported as a
  formatter bug and the original is kept.  ``format`` would rather decline
  than ship a rewrite that changed a program.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .tokens import HARD_KEYWORD_SET

#: Spaces per indentation level.  Four, because every shipped example and the
#: specification's own code use four; the formatter's job is to make that
#: uniform, not to offer a choice.
INDENT_WIDTH = 4

#: Two spaces between code and a trailing comment, the gofmt convention.
COMMENT_GAP = "  "

# Chunk kinds produced by the line scanner.
K_WORD = "word"          # identifier or keyword
K_NUMBER = "number"      # integer, float, duration literal
K_STRING = "string"      # "..." verbatim
K_CHAR = "char"          # '...' verbatim
K_PUNCT = "punct"        # operator or bracket

# Brackets, for depth tracking and spacing.
_OPEN = {"(", "[", "{"}
_CLOSE = {")", "]", "}"}

# Operators that sit tight against both operands: ranges and member access.
_TIGHT = {"..", "..=", "."}

# The lexer's trailing-continuation set, expressed as chunk text: a line that
# *ends* with one of these does not finish its logical line, so the next
# physical line is a continuation and carries no layout meaning.
_TRAILING_CONT = {
    "+", "-", "*", "/", "%", "**", "=", "&&", "||", ",", ".", ":", ";",
    "(", "[", "{", "|", "@", "->", "+=", "-=", "*=", "/=",
}
_TRAILING_CONT_WORDS = {"in", "as"}

# The lexer's leading-continuation prefixes: a line that *starts* with one of
# these joins the previous logical line and also carries no layout meaning.
_LEADING_CONT = (
    "->", "=>", ".", "+", "*", "/", "%", "&&", "||", "==", "!=",
    "<=", ">=", "<", ">", ",",
)

_TWO_CHAR = {
    "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=",
    "/=", "**", "..",
}


class FormatError(Exception):
    """Source the formatter refuses to touch, with a reason."""

    def __init__(self, message: str, line: int):
        super().__init__(message)
        self.line = line


# ----------------------------------------------------------------------
# scanning one physical line
# ----------------------------------------------------------------------
def _scan_chunks(code: str, line_no: int) -> List[Tuple[str, str]]:
    """Split one comment-free code line into atomic chunks.

    Strings and character literals are kept byte-for-byte: a formatter that
    normalized the *contents* of a string would be rewriting data.
    """
    chunks: List[Tuple[str, str]] = []
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        if ch in " \t\r":
            i += 1
            continue
        if ch == '"':
            j = i + 1
            closed = False
            while j < n:
                if code[j] == "\\":
                    if j + 1 >= n:
                        raise FormatError(
                            "unterminated escape sequence in string", line_no)
                    j += 2
                    continue
                if code[j] == '"':
                    j += 1
                    closed = True
                    break
                j += 1
            if not closed:
                raise FormatError("unterminated string literal", line_no)
            chunks.append((K_STRING, code[i:j]))
            i = j
            continue
        if ch == "'":
            j = i + 1
            if j < n and code[j] == "\\":
                if j + 1 < n and code[j + 1] == "x":
                    j += 4     # \xNN
                elif j + 1 < n and code[j + 1] == "u":
                    j += 6     # \uNNNN
                else:
                    j += 2
            else:
                j += 1
            if j >= n or code[j] != "'":
                raise FormatError("unterminated character literal", line_no)
            chunks.append((K_CHAR, code[i:j + 1]))
            i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (code[j].isalnum() or code[j] == "_"):
                j += 1
            chunks.append((K_WORD, code[i:j]))
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and code[i + 1].isdigit()):
            j = i
            # integer part
            while j < n and (code[j].isdigit() or code[j] == "_"):
                j += 1
            # fraction -- but `1..2` is a range, not the float `1.`: a dot
            # only joins the literal when a digit follows it.
            if j < n and code[j] == "." and j + 1 < n and \
                    code[j + 1].isdigit():
                j += 1
                while j < n and (code[j].isdigit() or code[j] == "_"):
                    j += 1
            # exponent
            if j < n and code[j] in "eE":
                k = j + 1
                if k < n and code[k] in "+-":
                    k += 1
                if k < n and code[k].isdigit():
                    j = k
                    while j < n and code[j].isdigit():
                        j += 1
            # duration unit: `5s`, `250ms` -- the unit belongs to the literal
            # and must not be separated from it by spacing.  The number scan
            # only ends here when no `.` or exponent consumed the tail, and a
            # unit is letters immediately after digits with no space.
            if j < n and code[j].isalpha() and code[i:j].replace(
                    ".", "").replace("_", "").isdigit():
                while j < n and code[j].isalpha():
                    j += 1
            chunks.append((K_NUMBER, code[i:j]))
            i = j
            continue
        # punctuation: three-char before two-char before one-char
        if code[i:i + 3] == "..=":
            chunks.append((K_PUNCT, "..="))
            i += 3
            continue
        two = code[i:i + 2]
        if two in _TWO_CHAR:
            chunks.append((K_PUNCT, two))
            i += 2
            continue
        chunks.append((K_PUNCT, ch))
        i += 1
    return chunks


def _split_comment(line: str) -> Tuple[str, Optional[str]]:
    """Split a physical line into (code, trailing-comment-or-None).

    A ``//`` inside a string is data, not a comment; the scan tracks string
    and character literals so it cannot be fooled by either.
    """
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == '"':
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "'":
            i += 1
            if i < n and line[i] == "\\":
                i += 2
            else:
                i += 1
            if i < n and line[i] == "'":
                i += 1
            continue
        if line.startswith("//", i):
            return line[:i], line[i:]
        i += 1
    return line, None


# ----------------------------------------------------------------------
# spacing
# ----------------------------------------------------------------------
#: Words that behave as operators in spacing: a unary sign after one of them
#: is still a unary sign (`and -x`, not `and - x`).
_WORD_OPERATORS = {"and", "or", "not", "in", "is", "as", "if", "else",
                   "then", "do", "of", "to", "where", "when", "over"}


def _mark_type_brackets(chunks: List[Tuple[str, str]]) -> List[Tuple[str, str, bool]]:
    """Tag each `<`/`>` with whether it opens or closes a type application.

    Type names in Gama-G are capitalised (`Result`, `List`, `Map`, `F64`),
    while values that get compared conventionally are not, so a `<` right
    after a capitalised word whose first type argument is another *word*
    opens a type application, and a `>` closes the innermost open one.  A
    `<` before a number or string is a comparison (`Score < 10`), which is
    spaced.  Where the heuristic guesses, only spacing is affected -- never
    parsing, because `<` is one token either way -- so the worst case is
    cosmetic.
    """
    marked: List[Tuple[str, str, bool]] = []
    angle = 0
    for idx, (kind, text) in enumerate(chunks):
        is_type = False
        if kind == K_PUNCT and text == "<":
            prev = chunks[idx - 1] if idx else None
            nxt = chunks[idx + 1] if idx + 1 < len(chunks) else None
            if (prev is not None and prev[0] == K_WORD and prev[1][:1].isupper()
                    and nxt is not None and nxt[0] == K_WORD):
                angle += 1
                is_type = True
        elif kind == K_PUNCT and text == ">" and angle > 0:
            angle -= 1
            is_type = True
        marked.append((kind, text, is_type))
    return marked


def _is_unary_position(prev: Optional[Tuple[str, str]]) -> bool:
    """A `-` or `+` is unary when nothing value-producing precedes it."""
    if prev is None:
        return True
    kind, text = prev
    if kind in (K_NUMBER, K_STRING, K_CHAR):
        return False
    if kind == K_WORD:
        return text in _WORD_OPERATORS
    return text not in _CLOSE and text not in ("..", "..=")


#: The lexemes that would be *created* by gluing two chunks together.  Any
#: rule that removes the gap between two chunks must not let the seam read as
#: one longer token: `Result<F64, Text> = x` may collapse spaces inside the
#: angle brackets, but gluing `>` to `=` would silently turn it into `>=` and
#: change -- here destroy -- the program.  A formatter that does that is worse
#: than no formatter.
_SEAM_TOKENS = {
    "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=",
    "/=", "**", "..",
}


def _bad_glue(prev_text: str, cur_text: str) -> bool:
    """True when `prev_text + cur_text` would start a longer token than the
    two chunks at their seam (or, for `..` + `=`, form the three-character
    `..=`)."""
    if not prev_text or not cur_text:
        return False
    seam = prev_text[-1] + cur_text[0]
    if seam in _SEAM_TOKENS:
        return True
    if seam == ".." and cur_text.startswith("="):
        return True          # `..` + `=` reads as `..=`
    return False


def _tight(prev_text: str, cur_text: str) -> str:
    """The empty gap, unless gluing would form a longer token than the parts."""
    return " " if _bad_glue(prev_text, cur_text) else ""


def _space_between(prev: Tuple[str, str], cur: Tuple[str, str],
                   prev_unary_sign: bool, prev_type_angle: bool,
                   cur_type_angle: bool) -> str:
    """The canonical gap between two adjacent chunks.

    `prev_unary_sign` says whether the previous chunk was a unary ``-``/``+``;
    the two angle flags say whether either chunk is a `<`/`>` that delimits a
    type application.  These carry one chunk of look-behind and look-ahead
    that the pairwise view lacks, so `_join_chunks` supplies them.  Every
    rule that glues is guarded by `_tight`, so no spacing choice can change
    how the line lexes.
    """
    pk, pt = prev
    ck, ct = cur

    # member access and ranges are tight on both sides
    if pt in _TIGHT or ct in _TIGHT:
        return _tight(pt, ct)
    # a type application hugs its brackets: `Result<F64, Text>`
    if prev_type_angle or cur_type_angle:
        return _tight(pt, ct)
    if prev_unary_sign:
        # a unary sign binds directly to its operand...
        if ck in (K_WORD, K_NUMBER, K_STRING, K_CHAR) or \
                (ck == K_PUNCT and ct in _OPEN):
            return _tight(pt, ct)
        # ...but two stacked signs keep a gap, or `- -x` reads like `--x`;
        # anything else (a comma, a closer) takes the ordinary rules below.
        if ck == K_PUNCT and ct in ("-", "+"):
            return " "
    if ck == K_PUNCT and ct in _CLOSE:
        return _tight(pt, ct)
    if pk == K_PUNCT and pt in _OPEN:
        return _tight(pt, ct)
    if ck == K_PUNCT and ct in (",", ";", ":"):
        return _tight(pt, ct)
    # unary minus/plus binds to its operand, but the *sign itself* only takes
    # the tight gap after a bracket or after another sign: after a binary
    # operator the operator keeps its space (`x = -1`, `a > -1`), because
    # `=-1` and `>-1` read like glued operators.
    if ck == K_PUNCT and ct in ("-", "+") and _is_unary_position(prev):
        if pk == K_PUNCT and pt in _OPEN:
            return ""
        return " "
    # a call or index: name/bracket directly against the opening bracket.
    # `{` is deliberately absent: `Patient {` takes a space, because a brace
    # opens a *literal*, not a call, and reading it glued hides that.
    if ck == K_PUNCT and ct in ("(", "["):
        if pk in (K_WORD, K_NUMBER, K_STRING, K_CHAR) and \
                not (pk == K_WORD and pt in HARD_KEYWORD_SET):
            return _tight(pt, ct)
        if pk == K_PUNCT and pt in _CLOSE:
            return _tight(pt, ct)
    # `!` (logical not) binds to what follows
    if pk == K_PUNCT and pt in ("!",):
        return _tight(pt, ct)
    return " "


def _join_chunks(chunks: List[Tuple[str, str]]) -> str:
    """Assemble one line's chunks with canonical spacing.

    A `{` that opens a record literal or a block -- one that follows a name
    or a closer, `Patient { id: ... }` -- is spaced on its inside, the way
    this repository's own examples write it; a `{` in expression position is
    a map literal and stays tight, `{"P-1": "Ada"}`.
    """
    if not chunks:
        return ""
    marked = _mark_type_brackets(chunks)
    parts = [marked[0][1]]
    # a sign opening a line is unary by construction (`-x` as an expression)
    unary_sign = (marked[0][0] == K_PUNCT and marked[0][1] in ("-", "+"))
    brace_stack: List[bool] = []      # True when the open brace is spaced
    for idx in range(1, len(marked)):
        prev = marked[idx - 1]
        cur = marked[idx]
        gap = _space_between(prev[:2], cur[:2], unary_sign,
                             prev[2], cur[2])
        kind, text = cur[0], cur[1]
        if kind == K_PUNCT and text == "{":
            pk, pt = prev[0], prev[1]
            record = ((pk in (K_WORD, K_NUMBER, K_STRING, K_CHAR)
                       and not (pk == K_WORD and pt in HARD_KEYWORD_SET))
                      or (pk == K_PUNCT and pt in _CLOSE))
            brace_stack.append(record)
        elif kind == K_PUNCT and text == "}":
            was_record = brace_stack.pop() if brace_stack else False
            if was_record and gap == "" and not parts[-1].endswith("{"):
                gap = " "
        elif gap == "" and brace_stack and brace_stack[-1] and \
                prev[0] == K_PUNCT and prev[1] == "{":
            gap = " "
        parts.append(gap + text)
        unary_sign = (kind == K_PUNCT and text in ("-", "+")
                      and _is_unary_position(prev[:2]))
    return "".join(parts)


# ----------------------------------------------------------------------
# the formatter proper
# ----------------------------------------------------------------------
def _indent_width(line: str, tab_width: int = 8) -> int:
    width = 0
    for ch in line:
        if ch == " ":
            width += 1
        elif ch == "\t":
            width += tab_width - (width % tab_width)
        else:
            break
    return width


def _starts_continuation(chunks: List[Tuple[str, str]]) -> bool:
    if not chunks:
        return False
    text = chunks[0][1]
    for prefix in _LEADING_CONT:
        if text.startswith(prefix):
            # `->` must win over a bare `-`; longer prefixes are tried first
            return True
    return False


def _ends_continuation(chunks: List[Tuple[str, str]]) -> bool:
    if not chunks:
        return False
    kind, text = chunks[-1]
    if kind == K_PUNCT and text in _TRAILING_CONT:
        return True
    return kind == K_WORD and text in _TRAILING_CONT_WORDS


def _bracket_delta_chunks(chunks: List[Tuple[str, str]]) -> int:
    delta = 0
    for kind, text in chunks:
        if kind != K_PUNCT:
            continue
        if text in _OPEN:
            delta += 1
        elif text in _CLOSE:
            delta -= 1
    return delta


def _bracket_events(code: str):
    """Yield (column, "open"|"close") for every bracket in `code`.

    String and character literals are skipped, because a `{` inside text is
    data, not layout.
    """
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == '"':
            i += 1
            while i < n:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == '"':
                    break
                i += 1
            i += 1
            continue
        if ch == "'":
            i += 1
            if i < n and code[i] == "\\":
                i += 2
            else:
                i += 1
            if i < n and code[i] == "'":
                i += 1
            continue
        if ch in _OPEN:
            yield i, "open"
        elif ch in _CLOSE:
            yield i, "close"
        i += 1


def _push_frames(frames: List[Tuple[int, int]], code: str, indent: int
                 ) -> None:
    """Apply one emitted line's brackets to the frame stack.

    A bracket that ends its line opens a continuation block one step in, at
    `indent + INDENT_WIDTH`.  A bracket with content after it on the same
    line continues at the column right after it -- the aligned-argument
    convention the language's own examples already use.
    """
    tail = code.rstrip()
    for col, kind in _bracket_events(code):
        if kind == "open":
            at_line_end = col == len(tail) - 1
            frames.append((indent,
                           indent + INDENT_WIDTH if at_line_end else col + 1))
        elif frames:
            frames.pop()


#: Words that open a core clause line (`uses   a`, `holds   b > 0`).  The
#: core's house style aligns the clause bodies inside a block, and an
#: intentional alignment is content in the visual sense the formatter should
#: keep, the same way `gofmt` keeps struct-field alignment.
CLAUSE_WORDS = frozenset({
    "purpose", "authority", "outcome", "trail", "uses", "yields", "effect",
    "needs", "holds", "when", "computes", "over", "starts", "repeats",
    "until", "within", "alters", "choose",
})

#: v0.1 statements whose trailing text the parser captures verbatim (see
#: `parser.py`: contract and policy-rule phrases are source slices, so a
#: failure can quote them as written).  Their bodies are never re-spaced.
V01_VERBATIM_WORDS = frozenset({
    "requires", "ensures", "ensure", "require", "assert", "allow", "deny",
})


def format_source(source: str, filename: str = "<input>") -> str:
    """Return the canonical formatting of `source`.

    Raises :class:`FormatError` when the source cannot be scanned safely.
    The result is idempotent: ``format_source(format_source(s)) ==
    format_source(s)``.
    """
    # Dialect detection must agree with the compiler's: run it on tokens so
    # a header comment cannot hide the pragma.  If the source does not lex,
    # it does not compile either, and the safe fallback is the text test.
    try:
        from .driver import is_core_dialect
        from .lexer import tokenize
        is_core = is_core_dialect(tokenize(source, filename))
    except Exception:                              # noqa: BLE001
        is_core = source.lstrip().startswith("gama core")
    lines = source.split("\n")
    out: List[str] = []
    # parallel to `out`: per emitted code line, the parts the reflow pass
    # needs to preserve alignment -- (indent, code, comment, kw, body) -- or
    # None for blanks, comment-only lines, and anything else a run rule must
    # not touch
    infos: List[Optional[dict]] = []
    stack: List[int] = [0]           # original widths of open indent levels
    #: One frame per currently-open bracket: ``(indent of the bracket's own
    #: line, the column content continues at)``.  A bracket that ends its
    #: line opens a block continued one step in; a bracket mid-line, like
    #: `f(` at the end of nothing, continues at the column right after it --
    #: which is what the aligned calls in this repository's own examples
    #: look like, so the formatter reaches the same shape from a messier one.
    frames: List[Tuple[int, int]] = []
    in_block_comment = False
    block_comment_lead: Optional[int] = None
    prev_chunks: List[Tuple[str, str]] = []
    blank_before = False             # a blank line separates the previous line
    # buffered comment/blank lines: None is a blank, (text, extra) is a
    # comment whose extra indent (beyond the flush level) may be 0 or more
    pending: List[Optional[Tuple[str, Optional[int]]]] = []

    def emit(text: str, info: Optional[dict] = None) -> None:
        out.append(text)
        infos.append(info)

    def flush(indent: int) -> None:
        """Emit buffered comment and blank lines at `indent`.

        The lexer skips comment and blank lines when it measures layout, so
        a comment between the end of a block and the next dedented
        declaration belongs to the declaration's level, not to the block that
        just closed.  Buffering until the next layout decision is made is
        what makes the formatter agree with the compiler.

        A pending entry is None for a blank line, or ``(text, extra)`` where
        ``extra`` is None for an ordinary ``//`` comment (flushed at exactly
        the block's indent) and an int for a block-comment interior line
        (flushed at the block's indent plus its distance from the block's
        first line, preserving the art inside the comment).
        """
        prev_blank = bool(out) and out[-1] == ""
        pending_local = list(pending)
        pending.clear()
        for entry in pending_local:
            if entry is None:                     # a blank line
                if out and not prev_blank:
                    emit("")
                prev_blank = True
                continue
            text, extra = entry
            if text is None:                      # blank inside a block
                if out and not prev_blank:
                    emit("")
                prev_blank = True
                continue
            if extra is None:
                emit(" " * indent + text if text else "")
            else:
                emit(" " * (indent + max(0, extra)) + text if text else "")
            prev_blank = False

    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip()

        # ----- inside a multi-line block comment --------------------
        if in_block_comment:
            if not line.strip():
                pending.append((None, None))
            else:
                lead = _indent_width(line)
                relative = (lead - block_comment_lead
                           if block_comment_lead is not None else 0)
                pending.append((line.lstrip(" \t"), relative))
            if "*/" in line:
                in_block_comment = False
                block_comment_lead = None
            continue

        stripped = line.strip(" \t")

        # ----- blank lines: buffered, collapsed to one at flush -----
        if not stripped:
            pending.append(None)
            prev_chunks = []
            blank_before = True
            continue

        # ----- a block comment that opens here ----------------------
        if stripped.startswith("/*") and "*/" not in stripped:
            in_block_comment = True
            block_comment_lead = _indent_width(line)
            pending.append((stripped, 0))
            prev_chunks = []
            blank_before = False
            continue

        code, trailing = _split_comment(line)
        code_stripped = code.strip(" \t")

        # ----- comment-only lines: buffered until layout is known ----
        if not code_stripped:
            if trailing is not None:
                pending.append((trailing.strip(), None))
            prev_chunks = []
            continue

        # ----- inline block comment: leave the line as written -------
        if "/*" in code:
            if frames:
                indent = frames[-1][1]
            else:
                indent = INDENT_WIDTH * (len(stack) - 1)
            flush(indent)
            text = " " * indent + code_stripped
            if trailing:
                text += COMMENT_GAP + trailing.strip()
            emit(text)
            _push_frames(frames, text, indent)
            prev_chunks = []
            blank_before = False
            continue

        # ----- ordinary code line ------------------------------------
        chunks = _scan_chunks(code_stripped, line_no)

        is_continuation = False
        if frames:
            is_continuation = True
        elif not blank_before and _starts_continuation(chunks):
            is_continuation = True
        elif prev_chunks and _ends_continuation(prev_chunks):
            # prev_chunks is cleared by blank lines, which is also what the
            # lexer does: a blank line terminates the logical line
            is_continuation = True

        if is_continuation:
            if frames:
                closer_line = chunks and all(
                    k == K_PUNCT and t in _CLOSE for k, t in chunks)
                indent = frames[-1][0] if closer_line else frames[-1][1]
            else:
                # a continuation of a plain statement, not of an open
                # bracket: one step inside the enclosing block
                indent = INDENT_WIDTH * len(stack)
        else:
            width = _indent_width(line)
            top = stack[-1]
            if width > top:
                stack.append(width)
            elif width < top:
                while len(stack) > 1 and width < stack[-1]:
                    stack.pop()
                if width != stack[-1]:
                    # inconsistent dedent: the compiler will say so with a
                    # position; the formatter declines to guess a structure.
                    raise FormatError(
                        "inconsistent dedent: this line does not match any "
                        "outer indentation level", line_no)
            level = len(stack) - 1
            indent = INDENT_WIDTH * level

        flush(indent)
        # A clause body is program text: the core parser captures every
        # clause and prose line verbatim so that a failure can quote it in
        # the program's own words, and v0.1 does the same for contracts and
        # policy rules.  Normalizing the spacing *inside* such a body would
        # change the quoted text -- so the formatter never rewrites a clause
        # body.  It only re-indents these lines and, below, widens padding to
        # an alignment the author began.
        info: Optional[dict] = None
        code_part = " " * indent + _join_chunks(chunks)
        src_indent = _indent_width(line)
        kw = chunks[0][1] if chunks and chunks[0][0] == K_WORD else ""
        verbatim = False
        body = ""
        pad_src = 1
        body_rel: Optional[int] = None
        if kw and (is_core or kw in V01_VERBATIM_WORDS):
            rest = code_stripped[len(kw):]
            if rest[:1] in (" ", "\t") and rest.strip():
                verbatim = True
                pad_src = len(rest) - len(rest.lstrip(" \t"))
                body = rest.lstrip(" \t")
                code_part = (" " * indent + kw
                             + " " * max(1, pad_src) + body)
        comment_part = trailing.strip() if trailing else None
        text = code_part
        if comment_part is not None:
            text += COMMENT_GAP + comment_part
        if is_core and kw in CLAUSE_WORDS and verbatim:
            # the body's start column relative to the line's indent; runs
            # align to the widest one present, never narrower
            body_rel = pad_src + len(kw)
        comment_rel = None
        if comment_part is not None:
            comment_rel = line.index("//") - src_indent
        info = {"indent": indent, "code": code_part, "comment": comment_part,
                "kw": kw if body_rel is not None else "", "body": body,
                "body_rel": body_rel, "comment_rel": comment_rel}
        emit(text, info)

        # record the brackets this line opened or closed, so the next line
        # (or its closers) can align against them
        _push_frames(frames, code_part, indent)
        prev_chunks = chunks
        blank_before = False

    # comments at the end of the file take the outermost level
    flush(0)
    # trailing blank lines do not survive the canonical form
    while out and out[-1] == "":
        out.pop()
        infos.pop()
    if not out:
        return "\n"
    _reflow(out, infos)
    return "\n".join(out) + "\n"


def _reflow(out: List[str], infos: List[Optional[dict]]) -> None:
    """Re-apply the two alignments the source intended, in place.

    * **Clause bodies (core only):** a run of consecutive lines at one indent
      whose first word is a clause keyword aligns its bodies at the widest
      column the run already reaches.  Alignment is only ever *grown* toward
      the run's own maximum, never cut, so an intentional column survives and
      a messy block converges to one; the rule is idempotent because the
      second pass sees the column it set.
    * **Trailing comments:** runs of adjacent code lines that each end in a
      `//` comment align the comments to the widest source column in the run,
      and at least two spaces after the longest code -- gofmt's rule, and
      the one this repository's own examples follow.

    Bodies themselves are never rewritten: see `format_source` on why a
    clause body is program text.
    """
    n = len(out)

    # ---- pass 1: clause-body alignment (core runs) ------------------
    i = 0
    while i < n:
        info = infos[i]
        if info is None or not info.get("kw"):
            i += 1
            continue
        j = i
        target = 0
        while j < n:
            nxt = infos[j]
            # one run = consecutive clause lines at the same indent,
            # whatever their keywords are; the house style aligns a whole
            # block (`uses`, `yields`, `holds`...) to one body column
            if nxt is None or not nxt.get("kw") \
                    or nxt["indent"] != info["indent"]:
                break
            target = max(target, nxt["indent"] + nxt["body_rel"])
            j += 1
        for k in range(i, j):
            entry = infos[k]
            pad = target - entry["indent"] - len(entry["kw"])
            entry["code"] = (" " * entry["indent"] + entry["kw"]
                             + " " * max(1, pad) + entry["body"])
        i = max(j, i + 1)

    # ---- pass 2: trailing-comment columns ----------------------------
    i = 0
    while i < n:
        info = infos[i]
        if info is None or info.get("comment") is None:
            i += 1
            continue
        j = i
        target = 0
        while j < n:
            nxt = infos[j]
            if (nxt is None or nxt.get("comment") is None
                    or nxt["indent"] != info["indent"]):
                break
            if nxt.get("comment_rel") is not None:
                target = max(target, nxt["indent"] + nxt["comment_rel"])
            target = max(target, 0)
            j += 1
        for k in range(i, j):
            infos[k]["comment_at"] = target
        i = max(j, i + 1)

    # ---- rebuild every tracked line ----------------------------------
    for k in range(n):
        entry = infos[k]
        if entry is None:
            continue
        line = entry["code"]
        if entry.get("comment") is not None:
            at = max(entry.get("comment_at") or 0,
                     len(line) + len(COMMENT_GAP))
            line = line + " " * (at - len(line)) + entry["comment"]
        out[k] = line




# ----------------------------------------------------------------------
# the safety check: same program in, same GIR out
# ----------------------------------------------------------------------
def _strip_positions(payload) -> object:
    """Remove source positions from a GIR JSON payload.

    Positions are exactly what a formatter changes: a column number differs
    whenever a line is re-spaced, so comparing them would reject every
    formatting that did anything.  Everything else -- operations, operands,
    types, verbatim constraint text, metadata -- is meaning, and must agree.
    """
    if isinstance(payload, dict):
        return {k: _strip_positions(v) for k, v in payload.items()
                if k not in ("pos", "end")}
    if isinstance(payload, list):
        return [_strip_positions(v) for v in payload]
    return payload


def verify_same_program(original: str, formatted: str) -> Optional[str]:
    """Compile both texts; if both compile, their GIR must agree.

    The comparison is over everything except source positions, which a
    formatter changes by definition.  Returns None when the check passes (or
    cannot be run because one side does not compile -- formatting code
    mid-edit is legitimate), or a message describing the divergence when the
    formatter changed meaning.
    """
    import json

    from .driver import compile_source

    try:
        before = compile_source(original)
    except Exception:                              # noqa: BLE001
        return None
    try:
        after = compile_source(formatted)
    except Exception:                              # noqa: BLE001
        return None
    if before.program is None and after.program is None:
        return None          # the file does not compile either way; formatting
                             # broken code mid-edit is legitimate
    if before.program is None or after.program is None:
        # One side compiles and the other does not.  If the *original*
        # compiles, formatted code must too: a formatter that breaks a
        # compiling program is the defect this check exists to catch.
        return ("the formatted source does not compile while the original "
                "does; the formatter declined to rewrite the file")
    if before.program.to_json() == after.program.to_json():
        return None
    if _strip_positions(json.loads(before.program.to_json())) != \
            _strip_positions(json.loads(after.program.to_json())):
        return ("the formatted source compiled to different GIR than the "
                "original; the formatter declined to rewrite the file")
    return None
