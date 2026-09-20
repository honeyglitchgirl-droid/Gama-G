"""The Gama-G lexer (spec section 22, step 1).

Layout rules
------------
Gama-G uses indentation for blocks (spec section 4: "The language may use
indentation for blocks in the initial grammar").  The lexer therefore emits
``INDENT``/``DEDENT`` tokens exactly as a Python-style tokenizer does.

Two continuation rules are needed so that the specification's own examples
lex correctly:

1. *Trailing* continuation -- a physical line ending in an operator, comma,
   colon, arrow or open bracket does not terminate a logical line.
2. *Leading* continuation -- a physical line beginning with ``->``, ``=>``,
   ``.``, ``+``, ``*`` or a comparison operator joins the previous line.
   Spec section 39 relies on this::

       fn evaluate(obs: medical.Observation)
           -> Result<Risk, MedicalError>

Inside brackets, newlines become soft ``SEPARATOR`` tokens so that the
brace-record syntax of spec section 13 -- whose fields are newline-separated
rather than comma-separated -- parses naturally.
"""

from __future__ import annotations

from typing import List, Optional

from .diagnostics import (LexError, Diagnostic, Phase, Severity, SourcePos,
                            front_end_code)
from .tokens import DURATION_UNITS, HARD_KEYWORDS, HARD_KEYWORD_SET, Token, TokenKind

# A newline directly after one of these never ends a logical line.
# Note that comparison operators are deliberately absent: `a > b` ends with
# `b`, and `Map<K,V>` ends a type annotation, so a trailing `<`/`>`/`==` must
# still terminate the logical line.
#
# `=>` is absent too.  A match arm may put its body on the following lines,
# indented, and that body is a layout block: suppressing the newline after
# `=>` would also suppress the INDENT that opens it, desynchronising the
# indentation stack for the rest of the match.
_TRAILING_CONTINUATION = {
    TokenKind.PLUS, TokenKind.MINUS, TokenKind.STAR, TokenKind.SLASH,
    TokenKind.PERCENT, TokenKind.POWER, TokenKind.ASSIGN,
    TokenKind.AND, TokenKind.OR, TokenKind.NOT, TokenKind.ARROW,
    TokenKind.COMMA, TokenKind.DOT, TokenKind.COLON,
    TokenKind.LPAREN, TokenKind.LBRACKET, TokenKind.LBRACE,
    TokenKind.AMPAMP, TokenKind.PIPEPIPE, TokenKind.IN, TokenKind.AS,
    TokenKind.PLUSASSIGN, TokenKind.MINUSASSIGN, TokenKind.STARASSIGN,
    TokenKind.SLASHASSIGN, TokenKind.PIPE, TokenKind.AT, TokenKind.SEMI,
    TokenKind.INDENT,
}

# A newline directly *before* one of these never ends a logical line.
_LEADING_CONTINUATION = (
    "->", "=>", ".", "+", "*", "/", "%", "&&", "||", "==", "!=",
    "<=", ">=", "<", ">", ",",
)

_SIMPLE_TOKENS = {
    "+": TokenKind.PLUS, "-": TokenKind.MINUS, "*": TokenKind.STAR,
    "/": TokenKind.SLASH, "%": TokenKind.PERCENT, "=": TokenKind.ASSIGN,
    "<": TokenKind.LT, ">": TokenKind.GT, "!": TokenKind.BANG,
    "(": TokenKind.LPAREN, ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET, "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE, "}": TokenKind.RBRACE,
    ",": TokenKind.COMMA, ":": TokenKind.COLON, ";": TokenKind.SEMI,
    "|": TokenKind.PIPE, "@": TokenKind.AT, ".": TokenKind.DOT,
}

_TWO_CHAR_TOKENS = {
    "->": TokenKind.ARROW, "=>": TokenKind.FATARROW, "==": TokenKind.EQ,
    "!=": TokenKind.NE, "<=": TokenKind.LE, ">=": TokenKind.GE,
    "&&": TokenKind.AMPAMP, "||": TokenKind.PIPEPIPE,
    "+=": TokenKind.PLUSASSIGN, "-=": TokenKind.MINUSASSIGN,
    "*=": TokenKind.STARASSIGN, "/=": TokenKind.SLASHASSIGN,
    "**": TokenKind.POWER,
    "..": TokenKind.RANGE,
}

_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'",
    "0": "\0", "a": "\a", "b": "\b", "f": "\f", "v": "\v",
}


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


class Lexer:
    """Tokenizes Gama-G source into a stream including layout tokens."""

    def __init__(self, source: str, filename: str = "<input>", tab_width: int = 8):
        self.src = source
        self.file = filename
        self.tab_width = tab_width
        self.i = 0
        self.line = 1
        self.line_start = 0
        self.tokens: List[Token] = []
        self.indents: List[int] = [0]
        self.depth = 0            # bracket nesting depth
        self.paren_depth = 0
        self.at_line_start = True

    # ------------------------------------------------------------------
    # position helpers
    # ------------------------------------------------------------------
    def _pos(self, offset: Optional[int] = None) -> SourcePos:
        off = self.i if offset is None else offset
        return SourcePos(self.file, self.line, off - self.line_start + 1, off)

    def _fail(self, message: str, pos: Optional[SourcePos] = None,
              help_text: Optional[str] = None, end: Optional[SourcePos] = None,
              code: Optional[str] = None) -> "LexError":
        # Every diagnostic gets a code so it can be filtered, counted and looked
        # up.  The front end had been emitting none; see front_end_code.
        return LexError(Diagnostic(
            Severity.ERROR, Phase.LEX, message, pos=pos or self._pos(),
            end=end, help_text=help_text,
            code=code or front_end_code(message, Phase.LEX),
        ))

    def _emit(self, kind: TokenKind, text: str, pos: SourcePos,
              value: object = None, end: Optional[SourcePos] = None) -> None:
        self.tokens.append(Token(kind, text, pos, value, end or self._pos()))

    @property
    def _last(self) -> Optional[Token]:
        return self.tokens[-1] if self.tokens else None

    # ------------------------------------------------------------------
    # whitespace / comments
    # ------------------------------------------------------------------
    def _measure_indent(self):
        """Return (expanded width, index just past the leading whitespace)."""
        width = 0
        j = self.i
        while j < len(self.src):
            ch = self.src[j]
            if ch == " ":
                width += 1
            elif ch == "\t":
                width += self.tab_width - (width % self.tab_width)
            else:
                break
            j += 1
        return width, j

    def _skip_inline_space(self) -> None:
        while self.i < len(self.src) and self.src[self.i] in " \t\r":
            self.i += 1

    def _skip_comment(self) -> bool:
        """Skip a comment if one starts here.  Returns True if it did."""
        if self.src.startswith("//", self.i):
            while self.i < len(self.src) and self.src[self.i] != "\n":
                self.i += 1
            return True
        if self.src.startswith("/*", self.i):
            start = self._pos()
            self.i += 2
            while self.i < len(self.src) and not self.src.startswith("*/", self.i):
                if self.src[self.i] == "\n":
                    self.line += 1
                    self.i += 1
                    self.line_start = self.i
                else:
                    self.i += 1
            if self.i >= len(self.src):
                raise self._fail("unterminated block comment", start,
                                 help_text="add a closing `*/`")
            self.i += 2
            return True
        return False

    def _line_is_blank_or_comment(self, j: int) -> bool:
        while j < len(self.src) and self.src[j] in " \t\r":
            j += 1
        if j >= len(self.src):
            return True
        if self.src[j] == "\n":
            return True
        return self.src.startswith("//", j) or self.src.startswith("/*", j)

    def _advance_newline(self) -> None:
        self.line += 1
        self.i += 1
        self.line_start = self.i

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def tokenize(self) -> List[Token]:
        first_indent, first_end = self._measure_indent()
        if first_indent > 0:
            self.indents.append(first_indent)
            self.i = first_end
            self._emit(TokenKind.INDENT, "", self._pos())

        while self.i < len(self.src):
            ch = self.src[self.i]

            if ch == "\n":
                self._handle_newline()
                continue

            if ch in " \t\r":
                self._skip_inline_space()
                continue

            if self._skip_comment():
                continue

            start = self._pos()

            if _is_ident_start(ch):
                self._lex_ident(start)
                continue
            if ch.isdigit():
                self._lex_number(start)
                continue
            if ch == '"':
                self._lex_string(start)
                continue
            if ch == "'":
                self._lex_char(start)
                continue

            # Three-character operators must be tested before two-character
            # ones, or `..=` would lex as RANGE followed by ASSIGN.
            three = self.src[self.i:self.i + 3]
            if three == "..=":
                self.i += 3
                self._emit(TokenKind.RANGE_INC, three, start)
                continue

            two = self.src[self.i:self.i + 2]
            if two in _TWO_CHAR_TOKENS:
                self.i += 2
                self._emit(_TWO_CHAR_TOKENS[two], two, start)
                continue

            if ch in _SIMPLE_TOKENS:
                kind = _SIMPLE_TOKENS[ch]
                self.i += 1
                if kind is TokenKind.LPAREN or kind is TokenKind.LBRACKET \
                        or kind is TokenKind.LBRACE:
                    self.depth += 1
                elif kind is TokenKind.RPAREN or kind is TokenKind.RBRACKET \
                        or kind is TokenKind.RBRACE:
                    self.depth -= 1
                    if self.depth < 0:
                        raise self._fail("unmatched closing bracket", start)
                self._emit(kind, ch, start)
                continue

            raise self._fail(
                f"unexpected character {ch!r}", start,
                help_text="Gama-G source must be valid UTF-8 text; this "
                          "character cannot begin any token.",
            )

        self._finish()
        return self.tokens

    # ------------------------------------------------------------------
    # newline / layout handling
    # ------------------------------------------------------------------
    def _continues_after_newline(self) -> bool:
        """True when a newline here must be suppressed (logical-line join)."""
        last = self._last
        if last is not None and last.kind in _TRAILING_CONTINUATION:
            return True
        j = self.i + 1
        while j < len(self.src) and self.src[j] in " \t\r":
            j += 1
        rest = self.src[j:j + 2]
        if not rest or rest[0] == "\n":
            return False
        if self.src.startswith("//", j):
            return False
        for prefix in _LEADING_CONTINUATION:
            if rest.startswith(prefix):
                # `->` must win over a bare `-`; longer prefixes are listed
                # first in _LEADING_CONTINUATION so this is safe.
                return True
        return False

    def _handle_newline(self) -> None:
        if self.depth > 0:
            start = self._pos()
            self._advance_newline()
            last = self._last
            if last is not None and last.kind is TokenKind.SEPARATOR:
                return  # collapse runs of blank lines inside brackets
            self._emit(TokenKind.SEPARATOR, "\n", start)
            return

        if self._continues_after_newline():
            self._advance_newline()
            self._skip_inline_space()
            return

        start = self._pos()
        self._advance_newline()
        # A trailing newline after the last token is only meaningful if we
        # have emitted anything on the current logical line.
        last = self._last
        if last is not None and last.kind not in (
            TokenKind.NEWLINE, TokenKind.INDENT, TokenKind.DEDENT
        ):
            self._emit(TokenKind.NEWLINE, "\n", start)

        self._emit_layout_for_next_line()

    def _emit_layout_for_next_line(self) -> None:
        """Skip blank/comment lines, then emit INDENT/DEDENT for the next one.

        The indentation must be measured *before* any whitespace is consumed,
        so this loop advances ``self.i`` explicitly rather than calling
        ``_skip_inline_space``.
        """
        while self.i < len(self.src):
            width, content = self._measure_indent()
            if content >= len(self.src):
                self.i = content
                return
            ch = self.src[content]
            if ch == "\n":
                self.i = content
                self._advance_newline()
                continue
            if self.src.startswith("//", content) or self.src.startswith("/*", content):
                self.i = content
                self._skip_comment()
                continue
            # Real content begins here, at this indentation level.
            self.i = content
            if self.depth > 0:
                return
            pos = self._pos()
            top = self.indents[-1]
            if width > top:
                self.indents.append(width)
                self._emit(TokenKind.INDENT, "", pos)
            elif width < top:
                while len(self.indents) > 1 and width < self.indents[-1]:
                    self.indents.pop()
                    self._emit(TokenKind.DEDENT, "", pos)
                if width != self.indents[-1]:
                    raise self._fail(
                        "inconsistent dedent: this line does not match any "
                        "outer indentation level",
                        pos,
                        help_text=f"expected {self.indents[-1]} spaces, "
                                  f"found {width}",
                    )
            return

        return

    def _finish(self) -> None:
        last = self._last
        if last is not None and last.kind not in (
            TokenKind.NEWLINE, TokenKind.INDENT, TokenKind.DEDENT
        ):
            self._emit(TokenKind.NEWLINE, "\n", self._pos())
        eof_pos = self._pos()
        while len(self.indents) > 1:
            self.indents.pop()
            self._emit(TokenKind.DEDENT, "", eof_pos)
        self._emit(TokenKind.EOF, "", eof_pos)

    # ------------------------------------------------------------------
    # individual token classes
    # ------------------------------------------------------------------
    def _lex_ident(self, start: SourcePos) -> None:
        j = self.i
        while j < len(self.src) and _is_ident_char(self.src[j]):
            j += 1
        text = self.src[self.i:j]
        self.i = j
        if text == "_":
            # A bare `_` is the wildcard, not a name.  It has to be decided
            # here: `_` is a legal identifier start, so this scanner always
            # reaches it before the dedicated UNDERSCORE branch below could,
            # and `_ => ...` was being parsed as a *binding* named `_`.  That
            # silently broke match exhaustiveness, because the checker was
            # told the arm covered nothing.
            self._emit(TokenKind.UNDERSCORE, text, start, text)
            return
        kind = HARD_KEYWORDS.get(text) if text in HARD_KEYWORD_SET else TokenKind.IDENT
        self._emit(kind, text, start, text)

    def _lex_number(self, start: SourcePos) -> None:
        j = self.i
        if self.src[j] == "0" and j + 1 < len(self.src) and self.src[j + 1] in "xXoObB":
            base_char = self.src[j + 1].lower()
            base = {"x": 16, "o": 8, "b": 2}[base_char]
            j += 2
            digits_start = j
            while j < len(self.src) and (_is_ident_char(self.src[j])):
                j += 1
            body = self.src[digits_start:j].replace("_", "")
            if not body:
                raise self._fail("numeric literal has no digits", start)
            try:
                value = int(body, base)
            except ValueError:
                raise self._fail(
                    f"invalid base-{base} literal `{self.src[self.i:j]}`", start,
                    help_text=f"only digits valid in base {base} may appear here",
                ) from None
            text = self.src[self.i:j]
            self.i = j
            self._emit(TokenKind.INT, text, start, value)
            return

        while j < len(self.src) and (self.src[j].isdigit() or self.src[j] == "_"):
            j += 1
        is_float = False
        if j < len(self.src) and self.src[j] == "." and \
                (j + 1 >= len(self.src) or self.src[j + 1].isdigit()):
            is_float = True
            j += 1
            while j < len(self.src) and (self.src[j].isdigit() or self.src[j] == "_"):
                j += 1
        if j < len(self.src) and self.src[j] in "eE":
            k = j + 1
            if k < len(self.src) and self.src[k] in "+-":
                k += 1
            if k < len(self.src) and self.src[k].isdigit():
                is_float = True
                j = k
                while j < len(self.src) and self.src[j].isdigit():
                    j += 1

        text = self.src[self.i:j]
        body = text.replace("_", "")

        # Duration literal: `<number><unit>` (spec section 11: `checkpoint every 5s`).
        if not is_float and j < len(self.src) and self.src[j].isalpha():
            unit_end = j
            while unit_end < len(self.src) and self.src[unit_end].isalpha():
                unit_end += 1
            unit = self.src[j:unit_end]
            if unit in DURATION_UNITS:
                if unit_end < len(self.src) and _is_ident_char(self.src[unit_end]):
                    raise self._fail(
                        f"invalid literal `{text}{self.src[j:unit_end + 1]}`",
                        start,
                        help_text="a duration literal must end after its unit "
                                  f"({', '.join(sorted(DURATION_UNITS))})",
                    )
                self.i = unit_end
                seconds = float(int(body)) * DURATION_UNITS[unit]
                self._emit(TokenKind.DURATION, text + unit, start,
                           {"seconds": seconds, "unit": unit,
                            "amount": int(body)})
                return
            raise self._fail(
                f"invalid literal `{text}{unit}`", start,
                help_text="numbers cannot be immediately followed by letters; "
                          f"did you mean a duration such as `5s` or `250ms`?",
            )

        self.i = j
        if is_float:
            self._emit(TokenKind.FLOAT, text, start, float(body))
        else:
            self._emit(TokenKind.INT, text, start, int(body))

    def _lex_string(self, start: SourcePos) -> None:
        self.i += 1
        out: List[str] = []
        while True:
            if self.i >= len(self.src):
                raise self._fail("unterminated string literal", start,
                                 help_text='add a closing `"`')
            ch = self.src[self.i]
            if ch == "\n":
                raise self._fail(
                    "unterminated string literal: newline inside a string",
                    start,
                    help_text="Gama-G v0.1 strings are single-line; use "
                              "`text.join` or an escape sequence instead",
                )
            if ch == '"':
                self.i += 1
                break
            if ch == "\\":
                self.i += 1
                if self.i >= len(self.src):
                    raise self._fail("unterminated escape sequence", start)
                esc = self.src[self.i]
                if esc in _ESCAPES:
                    out.append(_ESCAPES[esc])
                    self.i += 1
                elif esc == "x":
                    out.append(self._read_hex_escape(2, start))
                elif esc == "u":
                    out.append(self._read_hex_escape(4, start))
                elif esc == "\n":
                    self._advance_newline()
                    self._skip_inline_space()
                else:
                    raise self._fail(
                        f"unknown escape sequence `\\{esc}`", self._pos(self.i - 1),
                        help_text="valid escapes: \\n \\t \\r \\\\ \\\" \\' \\0 "
                                  "\\xNN \\uNNNN",
                    )
                continue
            out.append(ch)
            self.i += 1
        text = "".join(out)
        self._emit(TokenKind.STRING, self.src[start.offset:self.i], start, text)

    def _read_hex_escape(self, count: int, start: SourcePos) -> str:
        esc_pos = self._pos(self.i)
        self.i += 1
        digits = self.src[self.i:self.i + count]
        if len(digits) != count or not all(c in "0123456789abcdefABCDEF" for c in digits):
            raise self._fail(
                f"invalid escape: expected {count} hexadecimal digits", esc_pos,
            )
        self.i += count
        return chr(int(digits, 16))

    def _lex_char(self, start: SourcePos) -> None:
        self.i += 1
        if self.i >= len(self.src):
            raise self._fail("unterminated character literal", start)
        if self.src[self.i] == "\\":
            self.i += 1
            esc = self.src[self.i]
            if esc in _ESCAPES:
                value = _ESCAPES[esc]
                self.i += 1
            elif esc == "x":
                value = self._read_hex_escape(2, start)
            elif esc == "u":
                value = self._read_hex_escape(4, start)
            else:
                raise self._fail(f"unknown escape `\\{esc}`", self._pos(self.i - 1))
        else:
            value = self.src[self.i]
            self.i += 1
        if self.i >= len(self.src) or self.src[self.i] != "'":
            raise self._fail(
                "unterminated character literal", start,
                help_text="a Char literal holds exactly one character, "
                          "e.g. `'a'` or `'\\n'`",
            )
        self.i += 1
        self._emit(TokenKind.CHAR, self.src[start.offset:self.i], start, value)


def tokenize(source: str, filename: str = "<input>") -> List[Token]:
    """Convenience wrapper used by the CLI, tests and the formatter."""
    return Lexer(source, filename).tokenize()
