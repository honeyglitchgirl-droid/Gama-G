"""The Gama-G v0.2 core parser.

It subclasses the v0.1 parser rather than starting over: the token stream, the
layout protocol, expression parsing, type parsing and pattern parsing are all
already written and tested, and none of them are what the audit report objects
to.  What is new is the declaration grammar -- the part that decides what a
program *is*.

A core program is a flat set of declarations.  There is no nesting of
statements, because there are no statements: an operation's body is a single
`computes` expression plus constraints, guards and authority.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .. import ast_nodes as A
from ..parser import Parser
from ..tokens import TokenKind
from . import ast as C

# The clause keywords, by the kind of declaration that accepts them.  An
# unrecognised clause is an error naming what *is* allowed there, because in a
# declaration-driven language a misspelled clause would otherwise silently
# drop a constraint or a capability requirement.
CLAUSES = {
    "intent": {"purpose", "authority", "trail"},
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
    # a transition reads bindings in order to compute the state's next value,
    # so it declares relationships exactly like an operation does
    "transition": {"alters", "uses", "effect", "needs", "holds", "computes",
                   "trail"},
}

DECL_WORDS = {"intent", "source", "state", "operation", "refine", "each",
              "resolve", "transition", "outcome"}


class CoreParser(Parser):
    """Parses the core surface into :mod:`gamag.core.ast`."""

    # ------------------------------------------------------------------
    # program
    # ------------------------------------------------------------------
    def parse_core(self) -> C.CoreModule:
        version = self._parse_pragma()
        module = C.CoreModule(version=version, filename=self.file)
        while True:
            self.end_of_line()
            if self.at(TokenKind.EOF):
                break
            tok = self.peek()
            if tok.kind is not TokenKind.IDENT or tok.text not in DECL_WORDS:
                raise self.error(
                    f"expected a core declaration but found {tok.descr}", tok,
                    help_text="a core program is a set of `intent`, `source`, "
                              "`state`, `operation`, `refine`, `each`, "
                              "`resolve`, `transition` and `outcome` "
                              "declarations; there are no statements")
            word = tok.text
            if word == "intent":
                if module.intent is not None:
                    raise self.error("a core program declares one intent", tok)
                module.intent = self._parse_intent()
            elif word == "source":
                module.sources.append(self._parse_source())
            elif word == "state":
                module.states.append(self._parse_state())
            elif word == "outcome":
                if module.outcome is not None:
                    raise self.error("an intent has one outcome", tok)
                module.outcome = self._parse_outcome()
            else:
                module.operations.append(self._parse_operation(word))
        return module

    def _parse_pragma(self) -> str:
        """`gama core 0.2` -- optional, but it is how the driver detects the
        dialect, so programs are expected to carry it."""
        if not (self.at(TokenKind.IDENT) and self.peek().text == "gama"):
            return ""
        start = self.peek().pos.offset
        self.adv()
        if not (self.at(TokenKind.IDENT) and self.peek().text == "core"):
            raise self.error("expected `gama core <version>`", self.peek())
        self.adv()
        tok = self.peek()
        if tok.kind not in (TokenKind.FLOAT, TokenKind.INT, TokenKind.IDENT):
            raise self.error("expected a core version such as `0.2`", tok)
        self.adv()
        text = self.source[start:tok.pos.offset + len(tok.text)].strip() \
            if self.source else "gama core"
        self.end_of_line()
        return text

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------
    def _parse_intent(self) -> C.IntentDecl:
        tok = self.adv()                       # `intent`
        name_tok = self.expect(TokenKind.IDENT, "an intent name")
        decl = C.IntentDecl(pos=tok.pos, name=name_tok.text,
                            pos_name=name_tok.pos)
        for clause in self._clauses("intent", tok):
            if clause.keyword == "purpose":
                decl.purpose = clause.text
            elif clause.keyword == "authority":
                decl.authority = list(clause.names)
            elif clause.keyword == "trail":
                decl.trail = clause.text
        return decl

    def _parse_source(self) -> C.SourceDecl:
        tok = self.adv()                       # `source`
        secret = self._accept_secret()
        name_tok = self.expect(TokenKind.IDENT, "a binding name")
        self.expect(TokenKind.COLON, "`:` before the type")
        type_ref = self.parse_type()
        decl = C.SourceDecl(pos=tok.pos, name=name_tok.text, type=type_ref,
                            secret=secret)
        if self.at_kw("from"):
            self.adv()
            decl.from_expr = self.parse_expr()
        self.end_of_line()
        return decl

    def _parse_state(self) -> C.StateDecl:
        tok = self.adv()                       # `state`
        secret = self._accept_secret()
        name_tok = self.expect(TokenKind.IDENT, "a state name")
        self.expect(TokenKind.COLON, "`:` before the type")
        type_ref = self.parse_type()
        decl = C.StateDecl(pos=tok.pos, name=name_tok.text, type=type_ref,
                           secret=secret)
        for clause in self._clauses("state", tok):
            if clause.keyword == "starts":
                decl.starts = clause
            elif clause.keyword == "authority":
                decl.authority = list(clause.names)
        return decl

    def _parse_outcome(self) -> C.OutcomeDecl:
        tok = self.adv()                       # `outcome`
        name_tok = self.expect(TokenKind.IDENT, "the outcome binding")
        self.end_of_line()
        return C.OutcomeDecl(pos=tok.pos, binding=name_tok.text)

    def _accept_secret(self) -> bool:
        if self.at(TokenKind.SECRET):
            self.adv()
            return True
        return False

    def _parse_operation(self, kind: str) -> C.OperationDecl:
        tok = self.adv()                       # the declaration word
        name_tok = self.expect(TokenKind.IDENT, f"a {kind} name")
        decl = C.OperationDecl(pos=tok.pos, name=name_tok.text, kind=kind)
        for clause in self._clauses(kind, tok):
            self._apply_clause(decl, clause)
        return decl

    def _apply_clause(self, decl: C.OperationDecl, clause: C.Clause) -> None:
        key = clause.keyword
        if key == "uses":
            decl.uses = list(clause.names)
        elif key == "yields":
            decl.yields = clause.binding
            decl.yields_type = clause.type
            decl.yields_secret = clause.secret
        elif key == "alters":
            decl.alters = clause.binding
            decl.yields = clause.binding
            decl.yields_type = clause.type
        elif key == "effect":
            decl.effect = clause.text.strip() or (
                clause.names[0] if clause.names else "")
        elif key == "needs":
            decl.needs = list(clause.names)
        elif key == "holds":
            decl.holds.append(clause)
        elif key == "when":
            decl.when = clause
        elif key == "computes":
            decl.computes = clause
        elif key == "starts":
            decl.starts = clause
        elif key == "repeats":
            decl.repeats = clause
        elif key == "until":
            decl.until = clause
        elif key == "within":
            decl.within = clause.bound
        elif key == "over":
            decl.over = clause
        elif key == "trail":
            decl.trail = clause.text
        elif key == "choose":
            decl.choices = clause.choices if hasattr(clause, "choices") else []

    # ------------------------------------------------------------------
    # clause blocks
    # ------------------------------------------------------------------
    def _clauses(self, kind: str, head) -> List[C.Clause]:
        """The indented `keyword  value` lines belonging to a declaration."""
        allowed = CLAUSES.get(kind, set())
        if not self.at(TokenKind.NEWLINE):
            return []
        self.adv()                              # NEWLINE
        if not self.at(TokenKind.INDENT):
            return []
        self.adv()                              # INDENT
        out: List[C.Clause] = []
        self.end_of_line()
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            tok = self.peek()
            if tok.kind is not TokenKind.IDENT or tok.text not in allowed:
                wanted = ", ".join(f"`{w}`" for w in sorted(allowed))
                raise self.error(
                    f"`{tok.descr if tok.kind is not TokenKind.IDENT else tok.text}`"
                    f" is not a clause of "
                    f"{'an' if kind[0] in 'aeiou' else 'a'} {kind} declaration",
                    tok,
                    help_text=f"a {kind} accepts: {wanted}")
            out.append(self._clause(tok.text))
            self.end_of_line()
        if not self.at(TokenKind.DEDENT):
            raise self.error(f"unexpected end of file inside {kind}", self.peek())
        self.adv()                              # DEDENT
        return out

    def _clause(self, keyword: str) -> C.Clause:
        tok = self.adv()                        # the clause keyword
        clause = C.Clause(pos=tok.pos, keyword=keyword)
        if keyword in ("purpose", "trail", "effect"):
            clause.text = self._rest_of_line()
            return clause
        if keyword in ("uses", "needs", "authority"):
            clause.names = self._name_list()
            return clause
        if keyword == "within":
            num = self.expect(TokenKind.INT, "a repetition bound")
            clause.bound = int(num.value)
            if self.at(TokenKind.IDENT):        # `rounds` / `passes` -- optional
                self.adv()
            return clause
        if keyword in ("yields", "alters"):
            secret = self._accept_secret()
            clause.secret = secret
            name_tok = self.expect(TokenKind.IDENT, "a binding name")
            clause.binding = name_tok.text
            if self.accept(TokenKind.COLON):
                if keyword == "alters":
                    raise self.error(
                        "`alters` names a state that is already declared and "
                        "typed; a transition may not retype it", name_tok,
                        help_text="write `alters balance`, not "
                                  "`alters balance : I64`")
                clause.type = self.parse_type()
            return clause
        if keyword == "over":
            clause.expr = self._collection_expr()
            if self.at(TokenKind.AS):
                self.adv()
                item = self.expect(TokenKind.IDENT, "an item name")
                clause.item = item.text
            return clause
        if keyword == "choose":
            clause.choices = self._choices()
            return clause
        # holds / when / computes / starts / repeats / until.  The verbatim
        # source text is kept next to the parsed expression: constraints are
        # quoted back in ContractViolation faults, and guard text is what the
        # exhaustiveness proof compares.
        clause.expr = self._span(self.parse_expr)
        clause.text = self.last_span
        return clause

    def _span(self, parse):
        # Run `parse` and also remember the source text it consumed.
        start = self.peek().pos.offset
        value = parse()
        prev = self.toks[self.i - 1] if self.i > 0 else self.peek()
        end = prev.pos.offset + len(prev.text)
        self._last_span_text = (self.source[start:end].strip()
                                if self.source else "")
        return value

    @property
    def last_span(self) -> str:
        return getattr(self, "_last_span_text", "")

    def _choices(self) -> List[Tuple[A.Pattern, A.Expr]]:
        """The `choose` block of a `resolve`: pattern => expression."""
        if not self.at(TokenKind.NEWLINE):
            raise self.error("expected an indented block after `choose`",
                             self.peek())
        self.adv()
        if not self.at(TokenKind.INDENT):
            raise self.error("expected indented alternatives after `choose`",
                             self.peek())
        self.adv()
        out: List[Tuple[A.Pattern, A.Expr]] = []
        self.end_of_line()
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            pattern = self.parse_pattern()
            self.expect(TokenKind.FATARROW, "`=>` before the value")
            value = self.parse_expr()
            out.append((pattern, value))
            self.end_of_line()
        if not self.at(TokenKind.DEDENT):
            raise self.error("unexpected end of file inside `choose`",
                             self.peek())
        self.adv()
        return out

    def _collection_expr(self) -> A.Expr:
        """The collection an `over` clause traverses.

        Parsed without the general expression grammar on purpose: `as` is the
        reference slice's cast operator, so a full expression parse would read
        `over readings as reading` as a cast and swallow the item name.  A
        collection is a binding, optionally reached through fields or indexes,
        and that is all this accepts.
        """
        expr = self.parse_primary()
        while True:
            if self.accept(TokenKind.DOT):
                attr = self.expect(TokenKind.IDENT, "a field name")
                expr = A.Member(pos=expr.pos, obj=expr, attr=attr.text)
            elif self.at(TokenKind.LBRACKET):
                self.adv()
                index = self.parse_expr()
                self.expect(TokenKind.RBRACKET, "`]`")
                expr = A.Index(pos=expr.pos, obj=expr, index=index)
            else:
                return expr

    def _name_list(self) -> List[str]:
        names: List[str] = []
        while True:
            tok = self.expect(TokenKind.IDENT, "a name")
            names.append(tok.text)
            # `PatientStore[Read]` -- a capability-qualified name is still one
            # entry in a list, so the qualifier travels with it.
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

    def _rest_of_line(self) -> str:
        """The remainder of the line, verbatim.

        `purpose` and `trail` are prose the program is required to keep, so
        they are captured from the source rather than rebuilt from tokens --
        the same reason contracts and policy rules keep their source text.
        """
        start = self.peek().pos.offset
        end = start
        while not self.at_line_end() and not self.at(TokenKind.EOF):
            tok = self.adv()
            end = tok.pos.offset + len(tok.text)
        text = self.source[start:end].strip() if self.source else ""
        return text
