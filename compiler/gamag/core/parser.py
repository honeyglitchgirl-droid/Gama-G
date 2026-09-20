"""The Gama-G v0.3 core parser.

This parser is native.  It does not subclass or call the older language's
parser, and it does not produce the older language's abstract syntax: it reads
tokens and builds the semantic model in :mod:`gamag.core.mir` directly.

What it still shares with the rest of the compiler is the *lexer* and the type
lattice.  Neither is a language model -- one turns text into tokens, the other
defines what `F64` means -- and having two definitions of either would be a
defect, not independence.

Expressions are parsed here rather than borrowed, which matters for a reason
that is easy to miss: the older grammar's `as` is a cast operator, so a borrowed
expression parser reads `over readings as reading` as a cast and swallows the
item name.  The core has no cast, so the core's parser has no such ambiguity --
and `over` can be parsed by the general expression rule instead of needing a
special restricted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ..diagnostics import (Diagnostic, ParseError, Phase, Severity,
                           SourcePos, front_end_code)
from ..tokens import Token, TokenKind
from . import mir as M

# The clause keywords, by the declaration that accepts them.  An unrecognised
# clause is an error naming what *is* allowed there: in a declaration-driven
# language a misspelled clause would otherwise silently drop a constraint or a
# capability demand.
CLAUSES = {
    "intent": {"purpose", "authority", "trail", "recover", "checkpoint"},
    "source": set(),
    "state": {"starts", "authority"},
    "operation": {"uses", "yields", "effect", "needs", "holds", "when",
                  "computes", "trail"},
    "refine": {"uses", "yields", "effect", "needs", "holds", "starts",
               "repeats", "until", "within", "trail"},
    "each": {"over", "yields", "effect", "needs", "holds", "when", "computes",
             "trail"},
    "resolve": {"over", "yields", "effect", "needs", "holds", "choose",
                "trail"},
    "transition": {"alters", "uses", "effect", "needs", "holds", "computes",
                   "trail"},
}

DECL_WORDS = frozenset(CLAUSES) | {"outcome"}

# Binary operator precedence, lowest first.  These are mathematical operators
# rather than language structure, and the audit's section 12 explicitly accepts
# arithmetic notation: "Mathematical notation predates modern programming
# languages and is a reasonable foundation for arithmetic."
BINARY_LEVELS: Tuple[Tuple[TokenKind, ...], ...] = (
    (TokenKind.OR,),
    (TokenKind.AND,),
    (TokenKind.EQ, TokenKind.NE, TokenKind.LT, TokenKind.GT,
     TokenKind.LE, TokenKind.GE),
    (TokenKind.PLUS, TokenKind.MINUS),
    (TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT),
)

BINARY_TEXT = {
    TokenKind.OR: "or", TokenKind.AND: "and",
    TokenKind.EQ: "==", TokenKind.NE: "!=", TokenKind.LT: "<",
    TokenKind.GT: ">", TokenKind.LE: "<=", TokenKind.GE: ">=",
    TokenKind.PLUS: "+", TokenKind.MINUS: "-", TokenKind.STAR: "*",
    TokenKind.SLASH: "/", TokenKind.PERCENT: "%", TokenKind.POWER: "**",
}


@dataclass
class CoreSyntax:
    """The parsed program, before the graph is derived or anything is checked."""

    version: str = ""
    intent: M.IntentGraph = field(default_factory=M.IntentGraph)
    nodes: List[M.OpNode] = field(default_factory=list)
    outcome: str = ""
    outcome_pos: Optional[SourcePos] = None
    filename: str = "<core>"


class CoreParser:
    def __init__(self, tokens: List[Token], filename: str = "<core>",
                 source: str = ""):
        self.toks = tokens
        self.i = 0
        self.file = filename
        self.source = source

    # ------------------------------------------------------------------
    # token stream (SEPARATOR tokens are transparent, as in the lexer's design)
    # ------------------------------------------------------------------
    def raw(self) -> Token:
        return self.toks[self.i] if self.i < len(self.toks) else self.toks[-1]

    def peek(self, k: int = 0) -> Token:
        index, seen = self.i, 0
        while index < len(self.toks):
            tok = self.toks[index]
            if tok.kind is TokenKind.SEPARATOR:
                index += 1
                continue
            if seen == k:
                return tok
            seen += 1
            index += 1
        return self.toks[-1]

    def adv(self) -> Token:
        while self.i < len(self.toks) and \
                self.toks[self.i].kind is TokenKind.SEPARATOR:
            self.i += 1
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def at(self, *kinds: TokenKind) -> bool:
        return self.peek().kind in kinds

    def at_kw(self, *words: str) -> bool:
        tok = self.peek()
        return tok.kind is TokenKind.IDENT and tok.text in words

    def accept(self, *kinds: TokenKind) -> Optional[Token]:
        return self.adv() if self.at(*kinds) else None

    def expect(self, kind: TokenKind, what: str) -> Token:
        if self.at(kind):
            return self.adv()
        raise self.error(f"expected {what} but found {self.peek().descr}",
                         self.peek())

    def error(self, message: str, tok: Optional[Token] = None,
              help_text: Optional[str] = None,
              code: Optional[str] = None) -> ParseError:
        t = tok or self.peek()
        return ParseError(Diagnostic(
            Severity.ERROR, Phase.PARSE, message, pos=t.pos, end=t.end,
            help_text=help_text,
            code=code or front_end_code(message, Phase.PARSE)))

    def end_of_line(self) -> None:
        while self.raw().kind in (TokenKind.NEWLINE, TokenKind.SEPARATOR,
                                  TokenKind.SEMI):
            self.i += 1

    def at_line_end(self) -> bool:
        return self.raw().kind in (TokenKind.NEWLINE, TokenKind.SEPARATOR,
                                   TokenKind.SEMI, TokenKind.EOF)

    # ------------------------------------------------------------------
    # verbatim source text
    # ------------------------------------------------------------------
    def _text_since(self, offset: int) -> str:
        """The source between `offset` and the last token consumed."""
        if not self.source:
            return ""
        index = min(self.i, len(self.toks)) - 1
        while index > 0 and self.toks[index].kind is TokenKind.SEPARATOR:
            index -= 1
        tok = self.toks[index]
        return self.source[offset:tok.pos.offset + len(tok.text)].strip()

    def _rest_of_line(self) -> str:
        """Prose to the end of the line, captured verbatim.

        `purpose` and `trail` are obligations the program is required to keep,
        so they are taken from the source rather than rebuilt from tokens.
        """
        start = self.peek().pos.offset
        end = start
        while not self.at_line_end():
            tok = self.adv()
            end = tok.pos.offset + len(tok.text)
        return self.source[start:end].strip() if self.source else ""

    # ------------------------------------------------------------------
    # program
    # ------------------------------------------------------------------
    def parse(self) -> CoreSyntax:
        syntax = CoreSyntax(filename=self.file)
        syntax.version = self._pragma()
        while True:
            self.end_of_line()
            if self.at(TokenKind.EOF):
                break
            tok = self.peek()
            word = tok.text if tok.kind is TokenKind.IDENT else ""
            if word not in DECL_WORDS:
                raise self.error(
                    f"expected a core declaration but found {tok.descr}", tok,
                    help_text="a core program is a set of `intent`, `source`, "
                              "`state`, `operation`, `refine`, `each`, "
                              "`resolve`, `transition` and `outcome` "
                              "declarations; there are no statements")
            if word == "intent":
                if syntax.intent.name:
                    raise self.error("a core program declares one intent", tok)
                syntax.intent = self._intent()
            elif word == "outcome":
                if syntax.outcome:
                    raise self.error("an intent has one outcome", tok)
                self.adv()
                name = self.expect(TokenKind.IDENT, "the outcome binding")
                syntax.outcome = name.text
                syntax.outcome_pos = name.pos
                self.end_of_line()
            elif word == "source":
                syntax.nodes.append(self._source())
            elif word == "state":
                syntax.nodes.append(self._state())
            else:
                syntax.nodes.append(self._operation(word))
        return syntax

    def _pragma(self) -> str:
        """`gama core <version>` -- how the driver detects the dialect."""
        if not self.at_kw("gama"):
            return ""
        start = self.peek().pos.offset
        self.adv()
        if not self.at_kw("core"):
            raise self.error("expected `gama core <version>`", self.peek())
        self.adv()
        tok = self.peek()
        if tok.kind not in (TokenKind.FLOAT, TokenKind.INT, TokenKind.IDENT):
            raise self.error("expected a core version such as `0.3`", tok)
        self.adv()
        text = self._text_since(start)
        self.end_of_line()
        return text or "gama core"

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------
    def _intent(self) -> M.IntentGraph:
        tok = self.adv()
        name = self.expect(TokenKind.IDENT, "an intent name")
        intent = M.IntentGraph(name=name.text, pos=name.pos)
        for keyword, clause in self._clauses("intent", tok):
            if keyword == "purpose":
                intent.purpose = clause.get("text", "")
            elif keyword == "authority":
                intent.authority = list(clause.get("names", []))
            elif keyword == "trail":
                intent.trail = clause.get("text", "")
            elif keyword == "recover":
                intent.recovery = list(clause.get("steps", []))
            elif keyword == "checkpoint":
                intent.checkpoints = list(clause.get("names", []))
        return intent

    def _source(self) -> M.SourceNode:
        tok = self.adv()
        secret = bool(self.accept(TokenKind.SECRET))
        name = self.expect(TokenKind.IDENT, "a binding name")
        self.expect(TokenKind.COLON, "`:` before the type")
        type_ref = self._type()
        node = M.SourceNode(pos=tok.pos, name=name.text, kind="source",
                            produces=name.text, type=type_ref, secret=secret,
                            phase="compute")
        if self.at_kw("from"):
            self.adv()
            node.origin = self._expr()
        self.end_of_line()
        return node

    def _state(self) -> M.StateNode:
        tok = self.adv()
        secret = bool(self.accept(TokenKind.SECRET))
        name = self.expect(TokenKind.IDENT, "a state name")
        self.expect(TokenKind.COLON, "`:` before the type")
        type_ref = self._type()
        node = M.StateNode(pos=tok.pos, name=name.text, kind="state",
                           produces=name.text, type=type_ref, secret=secret,
                           phase="compute")
        for keyword, clause in self._clauses("state", tok):
            if keyword == "starts":
                node.initial = clause.get("expr")
            elif keyword == "authority":
                node.authority = list(clause.get("names", []))
        return node

    def _operation(self, kind: str) -> M.OpNode:
        tok = self.adv()
        name = self.expect(TokenKind.IDENT, f"a {kind} name")
        node = self._new_node(kind, name.text, tok.pos)
        for keyword, clause in self._clauses(kind, tok):
            self._apply(node, keyword, clause)
        return node

    def _new_node(self, kind: str, name: str, pos) -> M.OpNode:
        if kind == "refine":
            return M.RefinementNode(pos=pos, name=name, kind="refine")
        if kind == "each":
            return M.FanOutNode(pos=pos, name=name, kind="fanout")
        if kind == "resolve":
            return M.DispatchNode(pos=pos, name=name, kind="dispatch")
        if kind == "transition":
            return M.TransitionNode(pos=pos, name=name, kind="transition",
                                    phase="commit")
        return M.ComputeNode(pos=pos, name=name, kind="compute")

    def _apply(self, node: M.OpNode, keyword: str, clause: dict) -> None:
        if keyword == "uses":
            node.consumes = list(clause.get("names", []))
        elif keyword == "needs":
            node.needs = list(clause.get("names", []))
        elif keyword == "effect":
            # `effect crypto, audit` -- the algebra is a set, and the standard
            # library's own operations declare more than one
            node.effects = [part.strip() for part in
                            clause.get("text", "").split(",")
                            if part.strip()]
        elif keyword == "trail":
            node.trail = clause.get("text", "")
        elif keyword == "yields":
            node.produces = clause.get("binding", "")
            node.type = clause.get("type")
            node.secret = clause.get("secret", False)
        elif keyword == "alters":
            node.state = clause.get("binding", "")      # type: ignore[attr-defined]
            node.produces = clause.get("binding", "")
        elif keyword == "holds":
            node.holds.append(M.Constraint(
                kind="holds", text=clause.get("text", ""),
                expr=clause.get("expr"), node=node.name,
                binding=node.produces, pos=clause.get("pos")))
        elif keyword == "when":
            node.when = M.Constraint(
                kind="when", text=clause.get("text", ""),
                expr=clause.get("expr"), node=node.name,
                binding=node.produces, pos=clause.get("pos"))
        elif keyword == "computes":
            node.value = clause.get("expr")             # type: ignore[attr-defined]
        elif keyword == "starts":
            node.start = clause.get("expr")             # type: ignore[attr-defined]
        elif keyword == "repeats":
            node.step = clause.get("expr")              # type: ignore[attr-defined]
        elif keyword == "until":
            node.stop = M.Constraint(                   # type: ignore[attr-defined]
                kind="until", text=clause.get("text", ""),
                expr=clause.get("expr"), node=node.name,
                binding=node.produces, pos=clause.get("pos"))
        elif keyword == "within":
            node.bound = clause.get("bound", 1)         # type: ignore[attr-defined]
        elif keyword == "over":
            if isinstance(node, M.FanOutNode):
                node.collection = clause.get("expr")
                node.item = clause.get("item", "")
            else:
                node.subject = clause.get("expr")       # type: ignore[attr-defined]
        elif keyword == "choose":
            node.cases = clause.get("cases", [])        # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # clause blocks
    # ------------------------------------------------------------------
    def _clauses(self, kind: str, head: Token) -> List[Tuple[str, dict]]:
        allowed = CLAUSES.get(kind, set())
        if not self.at(TokenKind.NEWLINE):
            return []
        self.adv()
        if not self.at(TokenKind.INDENT):
            return []
        self.adv()
        out: List[Tuple[str, dict]] = []
        self.end_of_line()
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            tok = self.peek()
            word = tok.text if tok.kind is TokenKind.IDENT else ""
            if word not in allowed:
                shown = tok.text if tok.kind is TokenKind.IDENT else tok.descr
                wanted = ", ".join(f"`{w}`" for w in sorted(allowed))
                raise self.error(
                    f"`{shown}` is not a clause of "
                    f"{'an' if kind[0] in 'aeiou' else 'a'} {kind} declaration",
                    tok, help_text=f"a {kind} accepts: {wanted}")
            out.append((word, self._clause(word)))
            self.end_of_line()
        if not self.at(TokenKind.DEDENT):
            raise self.error(f"unexpected end of file inside {kind}",
                             self.peek())
        self.adv()
        return out

    def _clause(self, keyword: str) -> dict:
        tok = self.adv()
        clause: dict = {"pos": tok.pos}
        if keyword in ("purpose", "trail", "effect"):
            clause["text"] = self._rest_of_line()
            return clause
        if keyword in ("uses", "needs", "authority", "checkpoint"):
            clause["names"] = self._name_list()
            return clause
        if keyword == "recover":
            clause["steps"] = self._recovery_block()
            return clause
        if keyword == "within":
            number = self.expect(TokenKind.INT, "a repetition bound")
            clause["bound"] = int(number.value)
            if self.at(TokenKind.IDENT):        # `rounds`/`passes` are optional
                self.adv()
            return clause
        if keyword in ("yields", "alters"):
            clause["secret"] = bool(self.accept(TokenKind.SECRET))
            name = self.expect(TokenKind.IDENT, "a binding name")
            clause["binding"] = name.text
            if self.accept(TokenKind.COLON):
                if keyword == "alters":
                    raise self.error(
                        "`alters` names a state that is already declared and "
                        "typed; a transition may not retype it", name,
                        help_text="write `alters balance`, not "
                                  "`alters balance : I64`")
                clause["type"] = self._type()
            return clause
        if keyword == "over":
            clause["expr"] = self._expr()
            if self.at(TokenKind.AS):
                self.adv()
                item = self.expect(TokenKind.IDENT, "an item name")
                clause["item"] = item.text
            return clause
        if keyword == "choose":
            clause["cases"] = self._cases()
            return clause
        # holds / when / computes / starts / repeats / until
        start = self.peek().pos.offset
        clause["expr"] = self._expr()
        clause["text"] = self._text_since(start)
        return clause

    def _cases(self) -> List[Tuple[M.MPattern, M.MExpr]]:
        if not self.at(TokenKind.NEWLINE):
            raise self.error("expected an indented block after `choose`",
                             self.peek())
        self.adv()
        if not self.at(TokenKind.INDENT):
            raise self.error("expected indented alternatives after `choose`",
                             self.peek())
        self.adv()
        out: List[Tuple[M.MPattern, M.MExpr]] = []
        self.end_of_line()
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            start = self.peek().pos.offset
            pattern = self._pattern()
            pattern.text = self._text_since(start)
            self.expect(TokenKind.FATARROW, "`=>` before the value")
            out.append((pattern, self._expr()))
            self.end_of_line()
        if not self.at(TokenKind.DEDENT):
            raise self.error("unexpected end of file inside `choose`",
                             self.peek())
        self.adv()
        return out

    def _recovery_block(self) -> List[M.RecoveryStep]:
        """An indented escalation policy, in the shape spec section 10 shows.

        Each line is an action from the runtime's own vocabulary, optionally
        bounded (`retry within 3 rounds`) and optionally aimed at something
        (`restore checkpoint baseline`, `alert operator`). The action word is
        recognised here so that the rest of the line can be kept as the target
        verbatim -- the target is prose the operator reads, not a name the
        compiler resolves.
        """
        if not self.at(TokenKind.NEWLINE):
            raise self.error(
                "expected an indented policy after `recover`", self.peek(),
                help_text="write the actions on their own lines, one per step, "
                          "in escalation order")
        self.adv()
        if not self.at(TokenKind.INDENT):
            raise self.error("expected indented recovery steps after `recover`",
                             self.peek())
        self.adv()
        steps: List[M.RecoveryStep] = []
        self.end_of_line()
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            start = self.peek().pos.offset
            pos = self.peek().pos
            action = self.expect(TokenKind.IDENT, "a recovery action")
            count: Optional[int] = None
            if self.at_kw("within"):
                self.adv()
                number = self.expect(TokenKind.INT, "a number of rounds")
                count = int(number.value)
                if self.at(TokenKind.IDENT):     # `rounds`/`times` are optional
                    self.adv()
            target_start = self.peek().pos.offset
            while not self.at(TokenKind.NEWLINE, TokenKind.DEDENT,
                              TokenKind.EOF):
                self.adv()
            target = self._text_since(target_start).strip()
            steps.append(M.RecoveryStep(
                action=action.text, count=count, target=target,
                raw=self._text_since(start).strip(), pos=pos))
            self.end_of_line()
        if not self.at(TokenKind.DEDENT):
            raise self.error("unexpected end of file inside `recover`",
                             self.peek())
        self.adv()
        return steps

    def _name_list(self) -> List[str]:
        names: List[str] = []
        while True:
            tok = self.expect(TokenKind.IDENT, "a name")
            names.append(tok.text)
            # `PatientStore[Read]` -- a capability-qualified name is still one
            # entry, so the qualifier travels with it.
            if self.at(TokenKind.LBRACKET):
                self.adv()
                while not self.at(TokenKind.RBRACKET, TokenKind.EOF):
                    qualifier = self.adv()
                    names[-1] += f"[{qualifier.text}]"
                    if not self.accept(TokenKind.COMMA):
                        break
                self.expect(TokenKind.RBRACKET, "`]`")
            if not self.accept(TokenKind.COMMA):
                break
        return names

    # ------------------------------------------------------------------
    # types
    # ------------------------------------------------------------------
    def _type(self) -> M.MType:
        tok = self.peek()
        secret = bool(self.accept(TokenKind.SECRET))
        name_tok = self.expect(TokenKind.IDENT, "a type name")
        args: List[M.MType] = []
        if self.at(TokenKind.LT):
            self.adv()
            while not self.at(TokenKind.GT, TokenKind.EOF):
                args.append(self._type())
                if not self.accept(TokenKind.COMMA):
                    break
            self.expect(TokenKind.GT, "`>` to close the type arguments")
        return M.MType(name=name_tok.text, args=args, secret=secret,
                       pos=tok.pos)

    # ------------------------------------------------------------------
    # patterns
    # ------------------------------------------------------------------
    def _pattern(self) -> M.MPattern:
        tok = self.peek()
        if self.accept(TokenKind.UNDERSCORE):
            return M.PWild(pos=tok.pos)
        if tok.kind is TokenKind.INT:
            self.adv()
            return M.PLit(pos=tok.pos, value=tok.value, kind="int")
        if tok.kind is TokenKind.FLOAT:
            self.adv()
            return M.PLit(pos=tok.pos, value=tok.value, kind="float")
        if tok.kind is TokenKind.STRING:
            self.adv()
            return M.PLit(pos=tok.pos, value=tok.value, kind="string")
        if tok.kind is TokenKind.CHAR:
            self.adv()
            return M.PLit(pos=tok.pos, value=tok.value, kind="char")
        if tok.kind is TokenKind.IDENT and tok.text in ("true", "false"):
            self.adv()
            return M.PLit(pos=tok.pos, value=(tok.text == "true"), kind="bool")
        if tok.kind is TokenKind.IDENT:
            self.adv()
            if self.at(TokenKind.LPAREN):
                self.adv()
                args: List[M.MPattern] = []
                while not self.at(TokenKind.RPAREN, TokenKind.EOF):
                    args.append(self._pattern())
                    if not self.accept(TokenKind.COMMA):
                        break
                self.expect(TokenKind.RPAREN, "`)`")
                return M.PTag(pos=tok.pos, tag=tok.text, args=args)
            # A bare name in a dispatch binds the value rather than testing it.
            return M.PBind(pos=tok.pos, binding=tok.text)
        raise self.error(f"expected a dispatch alternative but found "
                         f"{tok.descr}", tok)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def _expr(self) -> M.MExpr:
        return self._binary(0)

    def _binary(self, level: int) -> M.MExpr:
        if level >= len(BINARY_LEVELS):
            return self._power()
        left = self._binary(level + 1)
        while self.at(*BINARY_LEVELS[level]):
            tok = self.adv()
            start = left.pos.offset if left.pos else tok.pos.offset
            right = self._binary(level + 1)
            left = M.MBin(pos=left.pos, op=BINARY_TEXT[tok.kind], left=left,
                          right=right)
            left.text = self._text_since(start)
        return left

    def _power(self) -> M.MExpr:
        left = self._unary()
        if self.at(TokenKind.POWER):
            self.adv()
            right = self._power()               # right-associative
            return M.MBin(pos=left.pos, op="**", left=left, right=right)
        return left

    def _unary(self) -> M.MExpr:
        tok = self.peek()
        if self.accept(TokenKind.NOT):
            return M.MUn(pos=tok.pos, op="not", operand=self._unary())
        if self.accept(TokenKind.BANG):
            return M.MUn(pos=tok.pos, op="not", operand=self._unary())
        if self.accept(TokenKind.MINUS):
            return M.MUn(pos=tok.pos, op="-", operand=self._unary())
        if self.accept(TokenKind.PLUS):
            return self._unary()
        return self._postfix()

    def _postfix(self) -> M.MExpr:
        expr = self._primary()
        while True:
            if self.accept(TokenKind.DOT):
                attr = self.expect(TokenKind.IDENT, "a field or operation name")
                if self.at(TokenKind.LPAREN):
                    # `math.clamp(...)`: a library operation, not a method on a
                    # value.  The core has no user-defined methods.
                    base = expr.binding if isinstance(expr, M.MRef) else ""
                    args = self._call_args()
                    expr = M.MCall(pos=expr.pos, module=base or expr.text,
                                   name=attr.text, args=args)
                else:
                    expr = M.MField(pos=expr.pos, obj=expr, attr=attr.text)
            elif self.at(TokenKind.LBRACKET):
                self.adv()
                index = self._expr()
                self.expect(TokenKind.RBRACKET, "`]`")
                expr = M.MIndex(pos=expr.pos, obj=expr, index=index)
            elif self.at(TokenKind.LPAREN) and isinstance(expr, M.MRef):
                args = self._call_args()
                expr = M.MCall(pos=expr.pos, module="", name=expr.binding,
                               args=args)
            else:
                return expr

    def _call_args(self) -> List[M.MExpr]:
        self.expect(TokenKind.LPAREN, "`(`")
        args: List[M.MExpr] = []
        while not self.at(TokenKind.RPAREN, TokenKind.EOF):
            args.append(self._expr())
            if not self.accept(TokenKind.COMMA):
                break
        self.expect(TokenKind.RPAREN, "`)`")
        return args

    def _primary(self) -> M.MExpr:
        tok = self.peek()
        if self.accept(TokenKind.LPAREN):
            inner = self._expr()
            self.expect(TokenKind.RPAREN, "`)`")
            return inner
        if self.at(TokenKind.LBRACKET):
            self.adv()
            items: List[M.MExpr] = []
            while not self.at(TokenKind.RBRACKET, TokenKind.EOF):
                items.append(self._expr())
                if not self.accept(TokenKind.COMMA):
                    break
            self.expect(TokenKind.RBRACKET, "`]`")
            return M.MItems(pos=tok.pos, items=items)
        if tok.kind is TokenKind.INT:
            self.adv()
            return M.MLit(pos=tok.pos, value=tok.value, kind="int")
        if tok.kind is TokenKind.FLOAT:
            self.adv()
            return M.MLit(pos=tok.pos, value=tok.value, kind="float")
        if tok.kind is TokenKind.STRING:
            self.adv()
            return M.MLit(pos=tok.pos, value=tok.value, kind="string")
        if tok.kind is TokenKind.CHAR:
            self.adv()
            return M.MLit(pos=tok.pos, value=tok.value, kind="char")
        if tok.kind is TokenKind.IDENT and tok.text in ("true", "false"):
            self.adv()
            return M.MLit(pos=tok.pos, value=(tok.text == "true"), kind="bool")
        if tok.kind is TokenKind.IDENT:
            self.adv()
            return M.MRef(pos=tok.pos, binding=tok.text)
        raise self.error(f"expected an expression but found {tok.descr}", tok)
