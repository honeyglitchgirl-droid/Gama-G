"""Token definitions for Gama-G (spec section 22, step 1: Lexing).

Keyword policy
--------------
Gama-G keeps a deliberately small set of *hard* keywords -- words that can
never be identifiers.  Everything else the specification writes in keyword
position (``pipeline``, ``service``, ``model``, ``audit``, ``recover``, ...)
is a *contextual* keyword: the lexer emits it as ``IDENT`` and the parser
recognises it by text at declaration/statement position.

This is required by the specification itself.  Section 14 declares models
with ``model RiskModel``, while section 38 binds a variable named ``model``
(``model = load Model("fraud-v3")``) and calls ``model.predict(features)``.
A hard ``model`` keyword would make that program illegal.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

from .diagnostics import SourcePos


class TokenKind(Enum):
    # structural
    EOF = auto()
    NEWLINE = auto()      # logical end of line at bracket depth 0
    SEPARATOR = auto()    # newline inside brackets; a soft item separator
    INDENT = auto()
    DEDENT = auto()

    # literals
    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    CHAR = auto()
    DURATION = auto()     # 5s, 250ms, 2h  (spec section 11)

    # hard keywords
    FN = auto()
    LET = auto()
    VAR = auto()
    SECRET = auto()
    IF = auto()
    ELSE = auto()
    MATCH = auto()
    FOR = auto()
    IN = auto()
    WHILE = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    PARALLEL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPORT = auto()
    GRANT = auto()
    AS = auto()

    # operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    POWER = auto()
    ASSIGN = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    BANG = auto()
    AMPAMP = auto()
    PIPEPIPE = auto()
    ARROW = auto()        # ->
    FATARROW = auto()     # =>
    PLUSASSIGN = auto()
    MINUSASSIGN = auto()
    STARASSIGN = auto()
    SLASHASSIGN = auto()

    # punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    DOT = auto()
    RANGE = auto()        # ..   (exclusive upper bound)
    RANGE_INC = auto()    # ..=  (inclusive upper bound)
    COLON = auto()
    SEMI = auto()
    PIPE = auto()
    AT = auto()
    UNDERSCORE = auto()


HARD_KEYWORDS = {
    "fn": TokenKind.FN,
    "let": TokenKind.LET,
    "var": TokenKind.VAR,
    "secret": TokenKind.SECRET,
    "if": TokenKind.IF,
    "else": TokenKind.ELSE,
    "match": TokenKind.MATCH,
    "for": TokenKind.FOR,
    "in": TokenKind.IN,
    "while": TokenKind.WHILE,
    "return": TokenKind.RETURN,
    "break": TokenKind.BREAK,
    "continue": TokenKind.CONTINUE,
    "parallel": TokenKind.PARALLEL,
    "and": TokenKind.AND,
    "or": TokenKind.OR,
    "not": TokenKind.NOT,
    "import": TokenKind.IMPORT,
    "grant": TokenKind.GRANT,
    "as": TokenKind.AS,
    "true": TokenKind.IDENT,   # handled as literal-ish ident by the parser
    "false": TokenKind.IDENT,
    "none": TokenKind.IDENT,
}
# true/false/none are lexed as IDENT so they stay usable in patterns and as
# qualified names (Option.none); the parser folds them into literals.
HARD_KEYWORDS["true"] = TokenKind.IDENT
HARD_KEYWORDS["false"] = TokenKind.IDENT
HARD_KEYWORDS["none"] = TokenKind.IDENT
HARD_KEYWORD_SET = {"fn", "let", "var", "secret", "if", "else", "match", "for",
                    "in", "while", "return", "break", "continue", "parallel",
                    "and", "or", "not", "import", "grant", "as"}

# Words the specification uses in keyword position but which remain valid
# identifiers elsewhere.  The parser matches these by text.
CONTEXTUAL_KEYWORDS = frozenset({
    "pipeline", "service", "agent", "policy", "transaction", "model",
    "fault", "record", "enum", "struct", "test", "checkpoint",
    "protect", "recover", "restart", "restore", "replay", "alert",
    "retry", "reconnect", "failover", "escalate", "operator",
    "audit", "commit", "input", "output", "predict", "normalize",
    "extract", "features", "using", "every", "at", "boundary", "all",
    "where", "assert", "human_review", "pure", "io", "network",
    "storage", "crypto", "unsafe", "deterministic", "allow", "deny",
    "require", "requires", "ensures", "ensure", "on", "unit",
    "property", "fuzz", "concurrency", "security", "medical",
    "recovery", "ok", "fail", "some", "action", "actor", "object",
    "reason", "safe", "events", "clean", "derive_features", "summarize",
    "load", "network_fn", "dataset", "metric", "baseline", "split",
    "confidence", "threshold", "approved_version", "role",
    "separationOfDuties", "module",
})

# Effects the runtime tracks (spec section 7).
EFFECT_NAMES = frozenset({
    "pure", "io", "network", "storage", "crypto", "model", "medical",
    "audit", "unsafe",
})

DURATION_UNITS = {
    "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0,
    "m": 60.0, "h": 3600.0, "d": 86400.0,
}


class Token:
    __slots__ = ("kind", "text", "value", "pos", "end")

    def __init__(
        self,
        kind: TokenKind,
        text: str,
        pos: SourcePos,
        value: object = None,
        end: Optional[SourcePos] = None,
    ):
        self.kind = kind
        self.text = text
        self.pos = pos
        self.value = value
        self.end = end or pos

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.value is not None and self.value != self.text:
            return f"Token({self.kind.name}, {self.text!r}, {self.value!r}, {self.pos})"
        return f"Token({self.kind.name}, {self.text!r}, {self.pos})"

    def is_kw(self, *words: str) -> bool:
        """True for a contextual keyword with one of the given texts."""
        return self.kind is TokenKind.IDENT and self.text in words

    @property
    def descr(self) -> str:
        if self.kind is TokenKind.IDENT:
            return f"identifier `{self.text}`"
        if self.kind is TokenKind.EOF:
            return "end of file"
        if self.kind is TokenKind.NEWLINE:
            return "end of line"
        if self.kind is TokenKind.SEPARATOR:
            return "line separator"
        if self.kind is TokenKind.INDENT:
            return "indent"
        if self.kind is TokenKind.DEDENT:
            return "dedent"
        return f"`{self.text}`"
