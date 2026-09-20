"""The Gama-G parser (spec section 22, step 2).

Produces the AST defined in :mod:`gamag.ast_nodes`.  Two layout styles are
accepted, as spec section 4 allows: indentation-delimited blocks and an
explicit braced form "reserved for generated or embedded source".

Keyword handling
----------------
Most of the specification's keywords are *contextual*.  The parser therefore
dispatches on identifier text at statement/declaration position, and guards
each contextual form so that the same word still works as an ordinary
identifier or call.  ``recover(err)`` (spec section 4) is a call, while
``recover`` inside a ``service`` body opens a recovery section (spec
section 40); ``extract(obs)`` (spec section 39) is a call while
``extract features`` (spec section 3) is a pipeline stage.
"""

from __future__ import annotations

from typing import List, Optional

from . import ast_nodes as A
from .diagnostics import (Diagnostic, ParseError, Phase, Severity, SourcePos,
                          front_end_code)
from .lexer import Lexer
from .nesting import BoundedRecursion
from .tokens import (EFFECT_NAMES, KEYWORD_TOKEN_KINDS, Token,
                     TokenKind)

DECL_KEYWORDS = {
    "pipeline", "service", "agent", "policy", "transaction", "model",
    "fault", "record", "struct", "enum", "test", "checkpoint", "module",
}

TEST_CATEGORIES = {
    "unit", "property", "fuzz", "concurrency", "security", "deterministic",
    "model", "medical", "recovery", "integration",
}

RECOVERY_ACTIONS = {
    "retry", "restore", "restart", "replay", "alert", "reconnect",
    "failover", "escalate",
}

STAGE_WORDS = {
    "normalize", "clean", "extract", "features", "derive", "summarize",
    "train", "infer", "validate", "publish", "load",
}

# Verbs that accept command-style (juxtaposed) arguments inside a pipeline
# or model body.  Only in that context, and only when not followed by `(`,
# so ordinary calls such as `summarize(predictions)` are unaffected.
COMMAND_VERBS = frozenset(STAGE_WORDS) | {"derive_features", "predict"}

POLICY_WORDS = {"allow", "deny"}

MODIFIER_WORDS = {"deterministic", "unsafe"}

_ASSIGN_OPS = {
    TokenKind.ASSIGN: "=", TokenKind.PLUSASSIGN: "+=",
    TokenKind.MINUSASSIGN: "-=", TokenKind.STARASSIGN: "*=",
    TokenKind.SLASHASSIGN: "/=",
}

_UNARY_OPS = {TokenKind.MINUS: "-", TokenKind.BANG: "!", TokenKind.NOT: "not"}

_CMP_OPS = {
    TokenKind.EQ: "==", TokenKind.NE: "!=", TokenKind.LT: "<",
    TokenKind.GT: ">", TokenKind.LE: "<=", TokenKind.GE: ">=",
}


# Type names whose arguments the specification writes in parentheses rather
# than angle brackets: `Tuple(I64, Text)` (spec section 6).
PAREN_TYPE_ARGS = {"Tuple", "Fn"}


class Parser(BoundedRecursion):
    def __init__(self, tokens: List[Token], filename: str = "<input>",
                 source: str = ""):
        self.toks = tokens
        self.i = 0
        self.file = filename
        self.source = source
        # Nesting depth of the expression grammar, bounded so that deeply
        # nested input is a diagnostic rather than a RecursionError.
        self._nesting = 0
        # Depth of "command-style" bodies (pipeline / model).  Inside these,
        # a known verb applied to a following expression on the same line
        # means a call, so `load Model("fraud-v3")` (spec section 38) reads
        # as `load(Model("fraud-v3"))`.
        self.command_ctx = 0

    # ------------------------------------------------------------------
    # token stream helpers (SEPARATOR tokens are transparent)
    # ------------------------------------------------------------------
    def peek(self, k: int = 0) -> Token:
        j = self.i
        seen = 0
        n = len(self.toks)
        while j < n:
            t = self.toks[j]
            if t.kind is TokenKind.SEPARATOR:
                j += 1
                continue
            if seen == k:
                return t
            seen += 1
            j += 1
        return self.toks[-1]

    def raw(self, k: int = 0) -> Token:
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def adv(self) -> Token:
        t = self.peek()
        j = self.i
        while self.toks[j].kind is TokenKind.SEPARATOR:
            j += 1
        self.i = j + 1
        return t

    def at(self, *kinds: TokenKind) -> bool:
        return self.peek().kind in kinds

    def at_kw(self, *words: str) -> bool:
        t = self.peek()
        return t.kind is TokenKind.IDENT and t.text in words

    def kw_ahead(self, offset: int, *words: str) -> bool:
        t = self.peek(offset)
        return t.kind is TokenKind.IDENT and t.text in words

    def accept(self, *kinds: TokenKind) -> Optional[Token]:
        if self.at(*kinds):
            return self.adv()
        return None

    def expect(self, kind: TokenKind, what: Optional[str] = None) -> Token:
        if self.at(kind):
            return self.adv()
        t = self.peek()
        raise self.error(
            f"expected {what or self._kind_name(kind)} but found {t.descr}", t
        )

    @staticmethod
    def _kind_name(kind: TokenKind) -> str:
        return {
            TokenKind.IDENT: "an identifier", TokenKind.NEWLINE: "end of line",
            TokenKind.LPAREN: "`(`", TokenKind.RPAREN: "`)`",
            TokenKind.LBRACE: "`{`", TokenKind.RBRACE: "`}`",
            TokenKind.LBRACKET: "`[`", TokenKind.RBRACKET: "`]`",
            TokenKind.COLON: "`:`", TokenKind.COMMA: "`,`",
            TokenKind.ARROW: "`->`", TokenKind.FATARROW: "`=>`",
            TokenKind.ASSIGN: "`=`", TokenKind.STRING: "a string literal",
            TokenKind.INT: "an integer literal", TokenKind.INDENT: "an indented block",
        }.get(kind, kind.name.lower())

    def error(self, message: str, tok: Optional[Token] = None,
              help_text: Optional[str] = None,
              code: Optional[str] = None) -> ParseError:
        t = tok or self.peek()
        return ParseError(Diagnostic(
            Severity.ERROR, Phase.PARSE, message, pos=t.pos, end=t.end,
            help_text=help_text,
            code=code or front_end_code(message, Phase.PARSE),
        ))

    def end_of_line(self) -> None:
        """Consume statement terminators."""
        while self.raw().kind in (TokenKind.NEWLINE, TokenKind.SEPARATOR,
                                  TokenKind.SEMI):
            self.i += 1

    def at_line_end(self) -> bool:
        """True when the current logical line ends here.

        Unlike :meth:`at`, this does *not* look through ``SEPARATOR`` tokens.
        It is what keeps greedy argument lists from running across lines
        inside a braced block, where physical newlines are separators rather
        than statement terminators -- e.g. the ``normalize`` / ``extract
        features`` / ``audit`` stages of spec section 3.
        """
        return self.raw().kind in (TokenKind.NEWLINE, TokenKind.SEPARATOR,
                                   TokenKind.DEDENT, TokenKind.RBRACE,
                                   TokenKind.EOF, TokenKind.SEMI)

    def _block_follows(self) -> bool:
        """True if an INDENT block starts on the next logical line.

        Used to tell a ``predict`` *stage* (spec section 3) from a ``predict``
        *method* with a body (spec section 14).
        """
        j = self.i
        n = len(self.toks)
        while j < n:
            k = self.toks[j].kind
            if k is TokenKind.NEWLINE:
                j += 1
                while j < n and self.toks[j].kind in (
                    TokenKind.NEWLINE, TokenKind.SEPARATOR
                ):
                    j += 1
                return j < n and self.toks[j].kind is TokenKind.INDENT
            if k in (TokenKind.EOF, TokenKind.DEDENT, TokenKind.RBRACE):
                return False
            j += 1
        return False

    # ------------------------------------------------------------------
    # module
    # ------------------------------------------------------------------
    def parse_module(self) -> A.Module:
        mod = A.Module(pos=SourcePos(self.file, 1, 1, 0), filename=self.file)
        self.end_of_line()
        while not self.at(TokenKind.EOF):
            if self.at(TokenKind.DEDENT):
                self.adv()
                continue
            if self.at(TokenKind.INDENT):
                raise self.error(
                    "unexpected indentation at top level",
                    help_text="top-level declarations and statements must start "
                              "at column 1",
                )
            decl = self.try_parse_decl()
            if decl is not None:
                mod.decls.append(decl)
                if isinstance(decl, A.GrantDecl):
                    mod.grants.extend(decl.caps)
                elif isinstance(decl, A.ImportDecl):
                    mod.imports.append(decl)
                elif isinstance(decl, A.ModuleDecl):
                    mod.module_name = decl.name
            else:
                mod.top_level.append(self.parse_stmt())
            self.end_of_line()
        return mod

    def _decl_ahead(self, kw: str) -> bool:
        """Decide whether a contextual keyword starts a declaration.

        Only the token *immediately* after the keyword matters.  If it is
        `=`, `.`, `(` or `[` the word is being used as an ordinary value
        (`model = ...`, `model.predict(x)`, `model(x)`); if it is an
        identifier we have the declaration form `model RiskModel`.  Looking
        further ahead would wrongly reject `pipeline fraudDetection(...)`,
        where the parenthesis belongs to the declaration's own parameter
        list (spec section 38).
        """
        nxt = self.peek(1)
        if kw == "test":
            if nxt.kind is TokenKind.STRING:
                return True
            return nxt.kind is TokenKind.IDENT and nxt.text in TEST_CATEGORIES
        if kw == "checkpoint":
            return self.kw_ahead(1, "every", "at")
        if kw == "module":
            return nxt.kind is TokenKind.IDENT
        return nxt.kind is TokenKind.IDENT

    def try_parse_decl(self) -> Optional[A.Decl]:
        if self.at(TokenKind.FN):
            return self.parse_fn()
        if self.at(TokenKind.IMPORT):
            return self.parse_import()
        if self.at(TokenKind.GRANT):
            return self.parse_grant()
        t = self.peek()
        if t.kind is TokenKind.IDENT and t.text in DECL_KEYWORDS:
            if not self._decl_ahead(t.text):
                return None
            handler = {
                "pipeline": self.parse_pipeline,
                "service": self.parse_service,
                "agent": self.parse_agent,
                "fault": self.parse_fault,
                "policy": self.parse_policy,
                "transaction": self.parse_transaction,
                "model": self.parse_model,
                "record": self.parse_record,
                "struct": self.parse_record,
                "enum": self.parse_enum,
                "test": self.parse_test,
                "checkpoint": self.parse_checkpoint_decl,
                "module": self.parse_module_decl,
            }[t.text]
            return handler()
        return None

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------
    def parse_fn(self) -> A.FnDecl:
        kw = self.expect(TokenKind.FN)
        name_tok = self.expect(TokenKind.IDENT, "a function name")
        generics: List[str] = []
        if self.at(TokenKind.LT):
            self.adv()
            while not self.at(TokenKind.GT, TokenKind.EOF):
                g = self.expect(TokenKind.IDENT, "a generic parameter name")
                generics.append(g.text)
                if not self.accept(TokenKind.COMMA):
                    break
            self.expect(TokenKind.GT, "`>`")
        params = self.parse_params()
        ret = None
        if self.accept(TokenKind.ARROW):
            ret = self.parse_type()
        body = self.parse_block(f"body of function `{name_tok.text}`")
        fn = A.FnDecl(pos=kw.pos, end=self.peek().pos, name=name_tok.text,
                      params=params, ret=ret, generics=generics, body=body)
        self._peel_modifiers(fn, body)
        return fn

    def _peel_modifiers(self, fn: A.FnDecl, body: A.Block) -> None:
        """Move leading effect/contract lines out of the body (spec 7, 27)."""
        while body.stmts:
            first = body.stmts[0]
            if isinstance(first, A.EffectDecl):
                body.stmts.pop(0)
                for eff in first.effects:
                    if eff == "deterministic":
                        fn.deterministic = True
                    elif eff == "unsafe":
                        # `unsafe` is both a modifier and an effect: the flag
                        # records that the function asked for it, and the effect
                        # list is what the effect checker compares the inferred
                        # set against.  Recording only the flag made the
                        # declaration unusable -- a function that read a line
                        # saying `unsafe` was then told it performed `unsafe`
                        # without declaring it, and the marker spec section 29
                        # requires for foreign calls could never be satisfied.
                        fn.unsafe = True
                        if eff not in fn.effects:
                            fn.effects.append(eff)
                    elif eff not in fn.effects:
                        fn.effects.append(eff)
            elif isinstance(first, A.Contract):
                body.stmts.pop(0)
                fn.contracts.append(first)
            else:
                break

    def parse_params(self) -> List[A.Param]:
        self.expect(TokenKind.LPAREN, "`(`")
        params: List[A.Param] = []
        while not self.at(TokenKind.RPAREN, TokenKind.EOF):
            name_tok = self.expect(TokenKind.IDENT, "a parameter name")
            ty = None
            if self.accept(TokenKind.COLON):
                ty = self.parse_type()
            params.append(A.Param(pos=name_tok.pos, name=name_tok.text, type=ty))
            if not self.accept(TokenKind.COMMA):
                break
        self.expect(TokenKind.RPAREN, "`)`")
        return params

    def parse_import(self) -> A.ImportDecl:
        kw = self.expect(TokenKind.IMPORT)
        parts = [self.expect(TokenKind.IDENT, "a module name").text]
        while self.accept(TokenKind.DOT):
            parts.append(self.expect(TokenKind.IDENT, "a module name").text)
        alias = None
        if self.accept(TokenKind.AS) or self.at_kw("as"):
            self.adv()
            alias = self.expect(TokenKind.IDENT, "an alias").text
        return A.ImportDecl(pos=kw.pos, name=parts[-1], path=".".join(parts),
                            alias=alias)

    def parse_grant(self) -> A.GrantDecl:
        kw = self.expect(TokenKind.GRANT)
        caps: List[str] = []
        while self.at(TokenKind.IDENT):
            caps.append(self.adv().text)
            if not self.accept(TokenKind.COMMA):
                break
        if not caps:
            raise self.error("expected at least one capability after `grant`",
                             help_text="e.g. `grant FileRead, NetworkConnect`")
        return A.GrantDecl(pos=kw.pos, name="grant", caps=caps)

    def parse_module_decl(self) -> A.ModuleDecl:
        kw = self.adv()
        parts = [self.expect(TokenKind.IDENT, "a module name").text]
        while self.accept(TokenKind.DOT):
            parts.append(self.expect(TokenKind.IDENT, "a module name").text)
        return A.ModuleDecl(pos=kw.pos, name=".".join(parts))

    def parse_pipeline(self) -> A.PipelineDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "a pipeline name").text
        params = self.parse_params() if self.at(TokenKind.LPAREN) else []
        ret = self.parse_type() if self.accept(TokenKind.ARROW) else None
        self.command_ctx += 1
        try:
            body = self.parse_block(f"body of pipeline `{name}`")
        finally:
            self.command_ctx -= 1
        decl = A.PipelineDecl(pos=kw.pos, name=name, params=params, ret=ret,
                              body=body)
        for st in body.stmts:
            if isinstance(st, A.IODirective):
                (decl.inputs if st.direction == "input" else decl.outputs).append(st)
        self._peel_pipeline_modifiers(decl, body)
        return decl

    def _peel_pipeline_modifiers(self, decl, body) -> None:
        while body.stmts and isinstance(body.stmts[0], A.EffectDecl):
            for eff in body.stmts.pop(0).effects:
                if eff not in decl.effects:
                    decl.effects.append(eff)

    def parse_service(self) -> A.ServiceDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "a service name").text
        body = self.parse_block(f"body of service `{name}`")
        decl = A.ServiceDecl(pos=kw.pos, name=name, body=body)
        for st in body.stmts:
            if isinstance(st, A.Section) and st.name == "protect":
                decl.protect = st.body
            elif isinstance(st, A.Section) and st.name == "recover":
                decl.recover.extend(st.steps)
            elif isinstance(st, A.CheckpointStmt):
                decl.checkpoints.append(st)
            elif isinstance(st, A.AuditDirective) and st.phrase.startswith("all"):
                decl.audit_all = True
        return decl

    def parse_agent(self) -> A.AgentDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "an agent name").text
        body = self.parse_block(f"body of agent `{name}`")
        decl = A.AgentDecl(pos=kw.pos, name=name, body=body)
        decl.handlers = [s for s in body.stmts if isinstance(s, A.HandlerDecl)]
        return decl

    def parse_fault(self) -> A.FaultDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "a fault-domain name").text
        body = self.parse_block(f"body of fault domain `{name}`")
        decl = A.FaultDecl(pos=kw.pos, name=name, body=body)
        decl.handlers = [s for s in body.stmts if isinstance(s, A.HandlerDecl)]
        return decl

    def parse_policy(self) -> A.PolicyDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "a policy name").text
        body = self.parse_block(f"body of policy `{name}`")
        decl = A.PolicyDecl(pos=kw.pos, name=name, body=body)
        for st in body.stmts:
            if isinstance(st, A.Require):
                # A policy's rules all share one representation, so `require`
                # becomes a rule rather than staying a statement.
                decl.rules.append(A.PolicyRule(
                    pos=st.pos, kind="require", expr=st.expr,
                    phrase=st.phrase or (st.message or "require")))
            elif isinstance(st, (A.PolicyRule, A.AuditDirective)):
                decl.rules.append(st)
        return decl

    def parse_transaction(self) -> A.TransactionDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "a transaction name").text
        body = self.parse_block(f"body of transaction `{name}`")
        return A.TransactionDecl(pos=kw.pos, name=name, body=body)

    def parse_model(self) -> A.ModelDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "a model name").text
        params = self.parse_params() if self.at(TokenKind.LPAREN) else []
        self.command_ctx += 1
        try:
            body = self.parse_block(f"body of model `{name}`")
        finally:
            self.command_ctx -= 1
        decl = A.ModelDecl(pos=kw.pos, name=name, body=body)
        for st in body.stmts:
            if isinstance(st, A.IODirective):
                (decl.inputs if st.direction == "input" else decl.outputs).append(st)
            elif isinstance(st, A.FnDecl):
                st.name = st.name if st.name else "predict"
                decl.methods.append(st)
        if params:
            decl.inputs.extend(
                A.IODirective(pos=p.pos, direction="input", name=p.name, type=p.type)
                for p in params
            )
        return decl

    def parse_test(self) -> A.TestDecl:
        kw = self.adv()
        category = "unit"
        if self.at(TokenKind.IDENT) and self.peek().text in TEST_CATEGORIES:
            category = self.adv().text
        name = ""
        if self.at(TokenKind.STRING):
            name = self.adv().value
        elif self.at(TokenKind.IDENT):
            name = self.adv().text
        body = self.parse_block(f"body of test `{name or category}`")
        return A.TestDecl(pos=kw.pos, name=name or category, category=category,
                          body=body)

    def parse_checkpoint_decl(self) -> A.Decl:
        """Top-level ``checkpoint every 5s`` becomes a service-less policy."""
        stmt = self.parse_checkpoint_stmt()
        svc = A.ServiceDecl(pos=stmt.pos, name="<module>", body=A.Block(
            pos=stmt.pos, stmts=[stmt]))
        svc.checkpoints.append(stmt)
        return svc

    def parse_record(self) -> A.RecordDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "a record name").text
        fields = self._parse_field_list(f"body of record `{name}`")
        return A.RecordDecl(pos=kw.pos, name=name, fields=fields)

    def parse_enum(self) -> A.EnumDecl:
        kw = self.adv()
        name = self.expect(TokenKind.IDENT, "an enum name").text
        variants: List[A.EnumVariant] = []
        if self.at(TokenKind.NEWLINE):
            self.adv()
            if self.accept(TokenKind.INDENT):
                while not self.at(TokenKind.DEDENT, TokenKind.EOF):
                    vname = self.expect(TokenKind.IDENT, "a variant name")
                    params = self.parse_params() if self.at(TokenKind.LPAREN) else []
                    variants.append(A.EnumVariant(pos=vname.pos, name=vname.text,
                                                  params=params))
                    self.end_of_line()
                self.accept(TokenKind.DEDENT)
        return A.EnumDecl(pos=kw.pos, name=name, variants=variants)

    def _parse_field_list(self, what: str) -> List[A.Param]:
        fields: List[A.Param] = []
        if not self.at(TokenKind.NEWLINE):
            return fields
        self.adv()
        if not self.accept(TokenKind.INDENT):
            return fields
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            fname = self.expect(TokenKind.IDENT, "a field name")
            self.expect(TokenKind.COLON, "`:`")
            ftype = self.parse_type()
            fields.append(A.Param(pos=fname.pos, name=fname.text, type=ftype))
            self.end_of_line()
        self.accept(TokenKind.DEDENT)
        return fields

    # ------------------------------------------------------------------
    # types
    # ------------------------------------------------------------------
    def parse_type(self) -> A.TypeRef:
        tok = self.peek()
        if tok.kind is not TokenKind.IDENT:
            raise self.error(f"expected a type but found {tok.descr}", tok,
                             help_text="types are written like `I64`, `Text`, "
                                       "`Result<F64, MathError>` or "
                                       "`PatientStore[Read]`")
        parts = [self.adv().text]
        while self.at(TokenKind.DOT) and self.peek(1).kind is TokenKind.IDENT:
            self.adv()
            parts.append(self.adv().text)
        args: List[A.TypeArg] = []
        if self.at(TokenKind.LT):
            args = self._parse_type_args()
        elif self.at(TokenKind.LPAREN) and parts[-1] in PAREN_TYPE_ARGS:
            # The specification writes this family with parentheses --
            # `Tuple(...)` in section 6 -- rather than angle brackets.  It is
            # limited to the names that use that spelling so a type followed
            # by an unrelated `(` cannot be misread as a type argument list.
            args = self._parse_type_args()
        caps: List[str] = []
        if self.at(TokenKind.LBRACKET):
            self.adv()
            while not self.at(TokenKind.RBRACKET, TokenKind.EOF):
                cap = self.peek()
                if cap.kind is TokenKind.IDENT:
                    caps.append(self.adv().text)
                elif cap.kind is TokenKind.INT:
                    caps.append(str(self.adv().value))
                else:
                    raise self.error(
                        f"expected a capability name but found {cap.descr}", cap,
                        help_text="capability qualifiers look like "
                                  "`Database[Read]` or `Audit[Write]`",
                    )
                if not self.accept(TokenKind.COMMA):
                    break
            self.expect(TokenKind.RBRACKET, "`]`")
        return A.TypeRef(pos=tok.pos, end=self.peek().pos, name=".".join(parts),
                         args=args, caps=caps)

    def _parse_type_args(self) -> List[A.TypeArg]:
        parenthesised = self.at(TokenKind.LPAREN)
        opener = TokenKind.LPAREN if parenthesised else TokenKind.LT
        closer = TokenKind.RPAREN if parenthesised else TokenKind.GT
        self.expect(opener, "`<` or `(` to open a type argument list")
        args: List[A.TypeArg] = []
        while not self.at(closer, TokenKind.EOF):
            if self.at(TokenKind.LBRACKET):
                args.append(self._parse_shape())
            else:
                args.append(self.descend(self.parse_type))
            if not self.accept(TokenKind.COMMA):
                break
        self.expect(closer,
                    ("`)`" if parenthesised else "`>`")
                    + " to close a type argument list")
        return args

    def _parse_shape(self) -> A.ShapeLit:
        tok = self.expect(TokenKind.LBRACKET, "`[`")
        dims: List[object] = []
        while not self.at(TokenKind.RBRACKET, TokenKind.EOF):
            d = self.peek()
            if d.kind is TokenKind.INT:
                dims.append(self.adv().value)
            elif d.kind is TokenKind.IDENT:
                dims.append(self.adv().text)
            else:
                raise self.error(
                    f"expected a tensor dimension but found {d.descr}", d,
                    help_text="shapes are written like `[1,224,224,3]` or `[N]`",
                )
            if not self.accept(TokenKind.COMMA):
                break
        self.expect(TokenKind.RBRACKET, "`]`")
        return A.ShapeLit(pos=tok.pos, dims=dims)

    # ------------------------------------------------------------------
    # blocks
    # ------------------------------------------------------------------
    def parse_block(self, what: str) -> A.Block:
        start = self.peek()
        if self.at(TokenKind.LBRACE):
            self.adv()
            stmts: List[A.Stmt] = []
            self.end_of_line()
            while not self.at(TokenKind.RBRACE, TokenKind.EOF):
                stmts.append(self.parse_stmt())
                self.end_of_line()
            if not self.at(TokenKind.RBRACE):
                raise self.error(f"expected `}}` to close {what}", self.peek())
            self.adv()
            return A.Block(pos=start.pos, stmts=stmts, braced=True)

        if not self.at(TokenKind.NEWLINE):
            raise self.error(
                f"expected an indented block for {what} but found "
                f"{self.peek().descr}", self.peek(),
                help_text="Gama-G uses indentation for blocks; start the body "
                          "on the next line, indented, or use `{ ... }`",
            )
        self.adv()  # NEWLINE
        if not self.at(TokenKind.INDENT):
            return A.Block(pos=start.pos, stmts=[], braced=False)
        self.adv()  # INDENT
        stmts = []
        self.end_of_line()
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            stmts.append(self.parse_stmt())
            self.end_of_line()
        if not self.at(TokenKind.DEDENT):
            raise self.error(f"unexpected end of file inside {what}", self.peek())
        self.adv()
        return A.Block(pos=start.pos, stmts=stmts, braced=False)

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------
    def parse_stmt(self) -> A.Stmt:
        tok = self.peek()
        kind = tok.kind

        if kind is TokenKind.LET:
            return self.parse_let(mutable=False, secret=False)
        if kind is TokenKind.VAR:
            return self.parse_let(mutable=True, secret=False)
        if kind is TokenKind.SECRET:
            return self.parse_let(mutable=False, secret=True)
        if kind is TokenKind.IF:
            return self.parse_if()
        if kind is TokenKind.WHILE:
            return self.parse_while()
        if kind is TokenKind.FOR:
            return self.parse_for()
        if kind is TokenKind.MATCH:
            return self.parse_match()
        if kind is TokenKind.RETURN:
            self.adv()
            if self.at(TokenKind.NEWLINE, TokenKind.DEDENT, TokenKind.RBRACE,
                       TokenKind.EOF):
                return A.Return(pos=tok.pos, value=None)
            return A.Return(pos=tok.pos, value=self.parse_expr())
        if kind is TokenKind.BREAK:
            self.adv()
            return A.Break(pos=tok.pos)
        if kind is TokenKind.CONTINUE:
            self.adv()
            return A.Continue(pos=tok.pos)
        if kind is TokenKind.PARALLEL:
            self.adv()
            return A.Parallel(pos=tok.pos, body=self.parse_block("parallel region"))
        if kind is TokenKind.LBRACE:
            return self.parse_block("block")
        if kind is TokenKind.INDENT:
            self.adv()
            blk = A.Block(pos=tok.pos, stmts=[])
            self.end_of_line()
            while not self.at(TokenKind.DEDENT, TokenKind.EOF):
                blk.stmts.append(self.parse_stmt())
                self.end_of_line()
            self.accept(TokenKind.DEDENT)
            return blk

        if kind is TokenKind.IDENT:
            special = self.try_parse_contextual_stmt()
            if special is not None:
                return special

        expr = self.parse_expr()
        if self.at(*_ASSIGN_OPS.keys()):
            op_tok = self.adv()
            value = self.parse_expr()
            return A.Assign(pos=expr.pos, target=expr,
                            op=_ASSIGN_OPS[op_tok.kind], value=value)
        return A.ExprStmt(pos=expr.pos, expr=expr)

    def try_parse_contextual_stmt(self) -> Optional[A.Stmt]:
        tok = self.peek()
        text = tok.text
        nxt = self.peek(1)
        called = nxt.kind is TokenKind.LPAREN
        assigned = nxt.kind in _ASSIGN_OPS
        member = nxt.kind is TokenKind.DOT

        # audit.record { ... }   (spec section 13)
        if text == "audit" and member and self.kw_ahead(2, "record") \
                and self.peek(3).kind is TokenKind.LBRACE:
            return self.parse_audit_record()
        if text == "audit" and not member and not called and not assigned \
                and nxt.kind is TokenKind.IDENT:
            # `audit all` inside a policy is a directive (spec section 17).
            # A bare `audit` line at the top of a function body is instead the
            # effect declaration of spec section 7, so it falls through to the
            # EFFECT_NAMES branch below.
            self.adv()
            words: List[str] = []
            while not self.at_line_end() and self.at(TokenKind.IDENT):
                words.append(self.adv().text)
            return A.AuditDirective(pos=tok.pos, phrase=" ".join(words))

        if text in ("require", "assert") and not called and not assigned \
                and not member:
            self.adv()
            start_offset = self.peek().pos.offset
            expr = self.parse_expr()
            phrase = ""
            if self.source:
                phrase = self.source[start_offset:self.peek().pos.offset].strip()
            # Both accept a message, in either spelling, so the two gates read
            # the same way.
            msg = None
            if self.at_kw("else"):
                self.adv()
                msg = self.expect(TokenKind.STRING, "a message").value
            elif self.accept(TokenKind.COMMA):
                msg = self.expect(TokenKind.STRING, "a message").value
            return (A.Require(pos=tok.pos, expr=expr, message=msg,
                              phrase=phrase)
                    if text == "require"
                    else A.Assert(pos=tok.pos, expr=expr, message=msg))

        if text in ("requires", "ensures", "ensure") and not called \
                and not assigned and not member:
            self.adv()
            start_offset = self.peek().pos.offset
            expr = self.parse_expr()
            ck = "requires" if text == "requires" else "ensures"
            text_written = ""
            if self.source:
                text_written = self.source[
                    start_offset:self.peek().pos.offset].strip()
            return A.Contract(pos=tok.pos, kind=ck, expr=expr,
                              text=text_written)

        if text == "checkpoint" and self.kw_ahead(1, "every", "at") and not member:
            return self.parse_checkpoint_stmt()

        if text in EFFECT_NAMES or text in MODIFIER_WORDS:
            plain = not called and not assigned and not member
            # `pure` alone on a line, or a comma-separated effect list.
            # Inside a braced block newlines arrive as SEPARATOR rather than
            # NEWLINE, so `audit` on its own line is an effect declaration in
            # both block forms.
            terminator = nxt.kind in (TokenKind.NEWLINE, TokenKind.COMMA,
                                      TokenKind.DEDENT, TokenKind.RBRACE,
                                      TokenKind.SEPARATOR, TokenKind.EOF)
            chained = nxt.kind is TokenKind.IDENT and nxt.text in EFFECT_NAMES
            if plain and (terminator or chained):
                self.adv()
                effects = [text]
                while self.accept(TokenKind.COMMA):
                    e = self.expect(TokenKind.IDENT, "an effect name")
                    effects.append(e.text)
                return A.EffectDecl(pos=tok.pos, effects=effects)

        if text in RECOVERY_ACTIONS and not called and not assigned and not member:
            return self.parse_recovery_step()

        if text in ("protect", "recover") and not called and not assigned \
                and not member:
            self.adv()
            body = self.parse_block(f"`{text}` section")
            steps = [s for s in body.stmts if isinstance(s, A.RecoveryStep)]
            return A.Section(pos=tok.pos, name=text, body=body, steps=steps)

        if text == "on" and not called and not assigned and not member:
            self.adv()
            ev = self.expect(TokenKind.IDENT, "an event name")
            body = self.parse_block(f"handler for `{ev.text}`")
            steps = [s for s in body.stmts if isinstance(s, A.RecoveryStep)]
            return A.HandlerDecl(pos=tok.pos, event=ev.text, body=body,
                                 steps=steps)

        if text in POLICY_WORDS and not called and not assigned and not member:
            self.adv()
            if self.at(TokenKind.IDENT) and self.peek().text in ("all", "required"):
                return A.AuditDirective(pos=tok.pos, phrase=text + " " + self.adv().text)
            start_offset = self.peek().pos.offset
            expr = self.parse_expr()
            # Spec section 17 asks for explainable decisions, so keep the
            # rule exactly as written to use as the explanation.
            phrase = ""
            if self.source:
                end_offset = self.peek().pos.offset
                phrase = self.source[start_offset:end_offset].strip()
            return A.PolicyRule(pos=tok.pos, kind=text, expr=expr,
                                phrase=phrase)

        if text in ("input", "output") and not called and not assigned and not member:
            self.adv()
            name = self.expect(TokenKind.IDENT, "a name")
            ty = self.parse_type() if self.accept(TokenKind.COLON) else None
            return A.IODirective(pos=tok.pos, direction=text, name=name.text,
                                 type=ty)

        if text == "predict" and not called and not assigned and not member:
            return self.parse_predict()

        if text in STAGE_WORDS and not called and not assigned and not member \
                and nxt.kind in (TokenKind.IDENT, TokenKind.NEWLINE,
                                 TokenKind.DEDENT, TokenKind.RBRACE):
            self.adv()
            args: List[A.Expr] = []
            using = None
            # Command-style arguments: `clean data`, `load Model("fraud-v3")`.
            # Bounded by the end of the logical line so that consecutive
            # stages inside a braced pipeline stay separate.
            while not self.at_line_end() and not self.at_kw("using"):
                if self.accept(TokenKind.COMMA):
                    continue
                if not self._starts_expr():
                    break
                args.append(self.parse_expr())
            if self.at_kw("using"):
                self.adv()
                using = self.parse_expr()
            return A.StageDirective(pos=tok.pos, stage=text, args=args,
                                    using=using)

        if text == "for":  # never reached; `for` is a hard keyword
            return None
        return None

    def parse_predict(self) -> A.Stmt:
        """`predict risk using model` (stage) or `predict features` + body (method)."""
        tok = self.adv()
        names: List[str] = []
        while self.at(TokenKind.IDENT) and not self.at_kw("using"):
            names.append(self.adv().text)
            if self.at(TokenKind.NEWLINE, TokenKind.DEDENT, TokenKind.RBRACE,
                       TokenKind.EOF):
                break
        if self.at_kw("using"):
            self.adv()
            using = self.parse_expr()
            return A.StageDirective(
                pos=tok.pos, stage="predict",
                args=[A.Name(pos=tok.pos, id=n) for n in names], using=using)
        if self._block_follows():
            params = [A.Param(pos=tok.pos, name=n) for n in names]
            body = self.parse_block("body of `predict`")
            return A.FnDecl(pos=tok.pos, name="predict", params=params, body=body)
        return A.StageDirective(
            pos=tok.pos, stage="predict",
            args=[A.Name(pos=tok.pos, id=n) for n in names])

    def parse_recovery_step(self) -> A.RecoveryStep:
        tok = self.adv()
        action = tok.text
        count = None
        if self.at(TokenKind.INT):
            count = self.adv().value
        words: List[str] = []
        while not self.at_line_end() and self.at(TokenKind.IDENT):
            words.append(self.adv().text)
        raw = " ".join([action] + ([str(count)] if count is not None else []) + words)
        return A.RecoveryStep(pos=tok.pos, action=action, count=count,
                              target=" ".join(words), raw=raw)

    def parse_checkpoint_stmt(self) -> A.CheckpointStmt:
        tok = self.adv()   # `checkpoint`
        mode_tok = self.expect(TokenKind.IDENT, "`every` or `at`")
        mode = mode_tok.text
        interval = None
        boundary = None
        raw = "checkpoint " + mode
        if mode == "every":
            d = self.peek()
            if d.kind is TokenKind.DURATION:
                self.adv()
                interval = d.value["seconds"]
                raw += f" {d.text}"
            elif d.kind is TokenKind.INT:
                self.adv()
                interval = float(d.value)
                raw += f" {d.text}s"
            else:
                raise self.error(
                    f"expected a duration after `checkpoint every` but found "
                    f"{d.descr}", d, help_text="e.g. `checkpoint every 5s`")
        else:
            words = []
            while self.at(TokenKind.IDENT):
                words.append(self.adv().text)
            boundary = " ".join(words)
            raw += " " + boundary
        return A.CheckpointStmt(pos=tok.pos, mode=mode,
                                interval_seconds=interval, boundary=boundary,
                                raw=raw)

    def parse_audit_record(self) -> A.AuditRecord:
        tok = self.adv()          # audit
        self.adv()                # .
        self.adv()                # record
        self.expect(TokenKind.LBRACE, "`{`")
        fields: List[tuple] = []
        self.end_of_line()
        while not self.at(TokenKind.RBRACE, TokenKind.EOF):
            key_tok = self.peek()
            if key_tok.kind is TokenKind.IDENT:
                key = self.adv().text
            elif key_tok.kind is TokenKind.STRING:
                key = self.adv().value
            else:
                raise self.error(
                    f"expected an audit field name but found {key_tok.descr}",
                    key_tok,
                    help_text="audit records look like "
                              "`audit.record { actor: user.id }`")
            self.expect(TokenKind.COLON, "`:`")
            value = self.parse_expr()
            fields.append((key, value))
            self.accept(TokenKind.COMMA)
            self.end_of_line()
        self.expect(TokenKind.RBRACE, "`}`")
        return A.AuditRecord(pos=tok.pos, fields=fields)

    def parse_let(self, mutable: bool, secret: bool) -> A.LetDecl:
        tok = self.adv()
        name_tok = self.expect(TokenKind.IDENT, "a binding name")
        ty = self.parse_type() if self.accept(TokenKind.COLON) else None
        value = None
        if self.at(*_ASSIGN_OPS.keys()):
            op = self.adv()
            if op.kind is not TokenKind.ASSIGN:
                raise self.error(
                    f"`{op.text}` is not allowed in a declaration", op,
                    help_text="use `=` when introducing a new binding")
            value = self.parse_expr()
        if value is None and ty is None:
            raise self.error(
                f"declaration of `{name_tok.text}` needs a type or an "
                f"initial value", name_tok,
                help_text="write `let x = 1` or `let x: I64`")
        return A.LetDecl(pos=tok.pos, name=name_tok.text, type=ty, value=value,
                         mutable=mutable, secret=secret)

    def parse_if(self) -> A.If:
        tok = self.expect(TokenKind.IF)
        cond = self.parse_expr()
        then_body = self.parse_block("`if` body")
        else_body = None
        if self.accept(TokenKind.ELSE):
            if self.at(TokenKind.IF):
                nested = self.parse_if()
                else_body = A.Block(pos=nested.pos, stmts=[nested])
            else:
                else_body = self.parse_block("`else` body")
        return A.If(pos=tok.pos, cond=cond, then_body=then_body,
                    else_body=else_body)

    def parse_while(self) -> A.While:
        tok = self.expect(TokenKind.WHILE)
        cond = self.parse_expr()
        return A.While(pos=tok.pos, cond=cond, body=self.parse_block("`while` body"))

    def parse_for(self) -> A.Stmt:
        tok = self.expect(TokenKind.FOR)
        if self.at_kw("all"):
            self.adv()
            var = self.expect(TokenKind.IDENT, "a variable name").text
            domain = None
            if self.accept(TokenKind.IN):
                domain = self.parse_expr()
            where = None
            if self.at_kw("where"):
                self.adv()
                where = self.parse_expr()
            samples = 100
            if self.at_kw("samples"):
                self.adv()
                samples = self.expect(TokenKind.INT, "a sample count").value
            body = self.parse_block("`for all` body")
            return A.ForAll(pos=tok.pos, var=var, domain=domain, where=where,
                            body=body, samples=samples)
        var = self.expect(TokenKind.IDENT, "a loop variable").text
        self.expect(TokenKind.IN, "`in`")
        it = self.parse_expr()
        body = self.parse_block("`for` body")
        return A.For(pos=tok.pos, var=var, iter=it, body=body)

    def parse_match(self) -> A.Match:
        tok = self.expect(TokenKind.MATCH)
        subject = self.parse_expr()
        arms: List[A.MatchArm] = []
        if not self.at(TokenKind.NEWLINE):
            raise self.error("expected match arms on the following lines",
                             self.peek())
        self.adv()
        if not self.accept(TokenKind.INDENT):
            raise self.error("expected an indented block of match arms",
                             self.peek())
        while not self.at(TokenKind.DEDENT, TokenKind.EOF):
            arms.append(self.parse_match_arm())
            self.end_of_line()
        if not self.at(TokenKind.DEDENT):
            raise self.error("unexpected end of file inside `match`", self.peek())
        self.adv()
        if not arms:
            raise self.error("`match` requires at least one arm", tok)
        return A.Match(pos=tok.pos, subject=subject, arms=arms)

    def parse_match_arm(self) -> A.MatchArm:
        start = self.peek()
        pattern = self.parse_pattern()
        guard = None
        # `if` is a hard keyword, so it is not caught by at_kw.
        if self.at(TokenKind.IF) or self.at_kw("if"):
            self.adv()
            guard = self.parse_expr()
        body: List[A.Stmt] = []
        if self.accept(TokenKind.FATARROW):
            if self.at(TokenKind.NEWLINE):
                body = self.parse_block("match arm body").stmts
            else:
                body = [self.parse_stmt()]
        elif self.at(TokenKind.NEWLINE):
            body = self.parse_block("match arm body").stmts
        else:
            raise self.error(
                f"expected `=>` after a pattern but found {self.peek().descr}",
                self.peek())
        return A.MatchArm(pos=start.pos, pattern=pattern, body=body, guard=guard)

    def parse_pattern(self) -> A.Pattern:
        tok = self.peek()
        if tok.kind is TokenKind.UNDERSCORE:
            self.adv()
            return A.WildcardPat(pos=tok.pos)
        if tok.kind in (TokenKind.INT, TokenKind.FLOAT, TokenKind.STRING,
                        TokenKind.CHAR):
            self.adv()
            kindmap = {TokenKind.INT: "int", TokenKind.FLOAT: "float",
                       TokenKind.STRING: "string", TokenKind.CHAR: "char"}
            return A.LiteralPat(pos=tok.pos, value=tok.value,
                                lit_kind=kindmap[tok.kind])
        if tok.kind is TokenKind.IDENT:
            if tok.text == "true" or tok.text == "false":
                self.adv()
                return A.LiteralPat(pos=tok.pos, value=tok.text == "true",
                                    lit_kind="bool")
            if tok.text == "none" and self.peek(1).kind is not TokenKind.LPAREN:
                self.adv()
                return A.CtorPat(pos=tok.pos, tag="none", args=[])
            if tok.text in ("ok", "fail", "some", "err", "value") \
                    and self.peek(1).kind is TokenKind.LPAREN:
                self.adv()
                self.adv()
                subs: List[A.Pattern] = []
                while not self.at(TokenKind.RPAREN, TokenKind.EOF):
                    subs.append(self.parse_pattern())
                    if not self.accept(TokenKind.COMMA):
                        break
                self.expect(TokenKind.RPAREN, "`)`")
                return A.CtorPat(pos=tok.pos, tag=tok.text, args=subs)
            if self.peek(1).kind is TokenKind.LPAREN:
                self.adv()
                self.adv()
                subs = []
                while not self.at(TokenKind.RPAREN, TokenKind.EOF):
                    subs.append(self.parse_pattern())
                    if not self.accept(TokenKind.COMMA):
                        break
                self.expect(TokenKind.RPAREN, "`)`")
                return A.CtorPat(pos=tok.pos, tag=tok.text, args=subs)
            self.adv()
            return A.NamePat(pos=tok.pos, name=tok.text)
        raise self.error(f"expected a pattern but found {tok.descr}", tok)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def parse_expr(self) -> A.Expr:
        return self.parse_or()

    def parse_or(self) -> A.Expr:
        left = self.parse_range()
        while self.at(TokenKind.OR, TokenKind.PIPEPIPE):
            op = self.adv()
            right = self.parse_range()
            left = A.Binary(pos=op.pos, op="or", left=left, right=right)
        return left

    def parse_range(self) -> A.Expr:
        """Ranges bind more loosely than arithmetic, so `1..n - 1` means
        `1..(n - 1)`.  They are non-associative: `a..b..c` is an error."""
        left = self.parse_and()
        if self.at(TokenKind.RANGE, TokenKind.RANGE_INC):
            op = self.adv()
            inclusive = op.kind is TokenKind.RANGE_INC
            right = self.parse_and()
            return A.RangeExpr(pos=op.pos, start=left, end=right,
                               inclusive=inclusive)
        return left

    def parse_and(self) -> A.Expr:
        left = self.parse_not()
        while self.at(TokenKind.AND, TokenKind.AMPAMP):
            op = self.adv()
            right = self.parse_not()
            left = A.Binary(pos=op.pos, op="and", left=left, right=right)
        return left

    def parse_not(self) -> A.Expr:
        if self.at(TokenKind.NOT):
            op = self.adv()
            return A.Unary(pos=op.pos, op="not",
                           operand=self.descend(self.parse_not))
        return self.parse_cmp()

    def parse_cmp(self) -> A.Expr:
        left = self.parse_add()
        while self.peek().kind in _CMP_OPS:
            op = self.adv()
            right = self.parse_add()
            left = A.Binary(pos=op.pos, op=_CMP_OPS[op.kind], left=left,
                            right=right)
        return left

    def parse_add(self) -> A.Expr:
        left = self.parse_mul()
        while self.at(TokenKind.PLUS, TokenKind.MINUS):
            op = self.adv()
            right = self.parse_mul()
            left = A.Binary(pos=op.pos, op=op.text, left=left, right=right)
        return left

    def parse_mul(self) -> A.Expr:
        left = self.parse_pow()
        while self.at(TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT):
            op = self.adv()
            right = self.parse_pow()
            left = A.Binary(pos=op.pos, op=op.text, left=left, right=right)
        return left

    def parse_pow(self) -> A.Expr:
        left = self.parse_unary()
        if self.at(TokenKind.POWER):
            op = self.adv()
            right = self.descend(self.parse_pow)   # right-associative
            return A.Binary(pos=op.pos, op="**", left=left, right=right)
        return left

    def parse_unary(self) -> A.Expr:
        if self.peek().kind in _UNARY_OPS:
            op = self.adv()
            return A.Unary(pos=op.pos, op=_UNARY_OPS[op.kind],
                           operand=self.descend(self.parse_unary))
        return self.parse_postfix()

    def parse_postfix(self) -> A.Expr:
        expr = self.parse_primary()
        while True:
            if self.at(TokenKind.DOT):
                self.adv()
                attr = self.peek()
                # A member name may be any word, including one reserved
                # elsewhere: `consent.grant(...)` is a call to a member named
                # `grant`, and requiring an identifier there made every builtin
                # whose name collided with a keyword impossible to call.
                if attr.kind is TokenKind.IDENT \
                        or attr.kind in KEYWORD_TOKEN_KINDS:
                    self.adv()
                    expr = A.Member(pos=expr.pos, obj=expr, attr=attr.text)
                elif attr.kind is TokenKind.INT:
                    self.adv()
                    expr = A.Index(pos=expr.pos, obj=expr,
                                   index=A.Literal(pos=attr.pos,
                                                   value=attr.value,
                                                   lit_kind="int"))
                else:
                    raise self.error(
                        f"expected a field name after `.` but found {attr.descr}",
                        attr)
            elif self.at(TokenKind.LBRACKET):
                self.adv()
                idx = self.parse_expr()
                self.expect(TokenKind.RBRACKET, "`]`")
                expr = A.Index(pos=expr.pos, obj=expr, index=idx)
            elif self.at(TokenKind.LPAREN):
                expr = A.Call(pos=expr.pos, callee=expr,
                              args=self.descend(self.parse_args))
            elif self.at(TokenKind.AS):
                self.adv()
                target = self.parse_type()
                expr = A.Cast(pos=expr.pos, expr=expr, target=target)
            elif (self.command_ctx > 0 and isinstance(expr, A.Name)
                  and expr.id in COMMAND_VERBS and not self.at_line_end()
                  and self._starts_expr()):
                # Command-style application, only inside pipeline/model bodies.
                args: List[A.Expr] = []
                while not self.at_line_end() and not self.at_kw("using"):
                    if self.accept(TokenKind.COMMA):
                        continue
                    if not self._starts_expr():
                        break
                    args.append(self.parse_expr())
                expr = A.Call(pos=expr.pos, callee=expr, args=args)
            else:
                return expr

    def parse_args(self) -> List[A.Expr]:
        self.expect(TokenKind.LPAREN, "`(`")
        args: List[A.Expr] = []
        self.end_of_line()
        while not self.at(TokenKind.RPAREN, TokenKind.EOF):
            args.append(self.parse_expr())
            self.end_of_line()
            if not self.accept(TokenKind.COMMA):
                break
            self.end_of_line()
        self.expect(TokenKind.RPAREN, "`)`")
        return args

    def parse_primary(self) -> A.Expr:
        tok = self.peek()
        k = tok.kind

        if k is TokenKind.INT:
            self.adv()
            return A.Literal(pos=tok.pos, end=tok.end, value=tok.value,
                             lit_kind="int")
        if k is TokenKind.FLOAT:
            self.adv()
            return A.Literal(pos=tok.pos, end=tok.end, value=tok.value,
                             lit_kind="float")
        if k is TokenKind.STRING:
            self.adv()
            return A.Literal(pos=tok.pos, end=tok.end, value=tok.value,
                             lit_kind="string")
        if k is TokenKind.CHAR:
            self.adv()
            return A.Literal(pos=tok.pos, end=tok.end, value=tok.value,
                             lit_kind="char")
        if k is TokenKind.DURATION:
            self.adv()
            return A.Literal(pos=tok.pos, end=tok.end, value=tok.value,
                             lit_kind="duration")

        if k is TokenKind.LPAREN:
            self.adv()
            self.end_of_line()
            if self.at(TokenKind.RPAREN):
                self.adv()
                return A.Literal(pos=tok.pos, value=None, lit_kind="unit")
            first = self.descend(self.parse_expr)
            if self.at(TokenKind.COMMA):
                items = [first]
                while self.accept(TokenKind.COMMA):
                    self.end_of_line()
                    if self.at(TokenKind.RPAREN):
                        break
                    items.append(self.descend(self.parse_expr))
                self.expect(TokenKind.RPAREN, "`)`")
                return A.TupleLit(pos=tok.pos, items=items)
            self.expect(TokenKind.RPAREN, "`)`")
            return first

        if k is TokenKind.LBRACKET:
            self.adv()
            items: List[A.Expr] = []
            self.end_of_line()
            while not self.at(TokenKind.RBRACKET, TokenKind.EOF):
                items.append(self.descend(self.parse_expr))
                self.end_of_line()
                if not self.accept(TokenKind.COMMA):
                    if self.at(TokenKind.RBRACKET, TokenKind.EOF):
                        break
                    if not self._starts_expr():
                        break
                    continue
                self.end_of_line()
            self.expect(TokenKind.RBRACKET, "`]`")
            return A.ListLit(pos=tok.pos, items=items)

        if k is TokenKind.LBRACE:
            return self.descend(self.parse_brace_literal)

        if k is TokenKind.IDENT:
            text = tok.text
            if text == "true" or text == "false":
                self.adv()
                return A.Literal(pos=tok.pos, end=tok.end,
                                 value=(text == "true"), lit_kind="bool")
            if text == "none" and self.peek(1).kind is not TokenKind.LPAREN:
                self.adv()
                return A.Construct(pos=tok.pos, tag="none", args=[])
            if text in ("ok", "fail", "some", "err") \
                    and self.peek(1).kind is TokenKind.LPAREN:
                self.adv()
                args = self.parse_args()
                return A.Construct(pos=tok.pos, tag=text, args=args)
            if text[0].isupper() and self.peek(1).kind is TokenKind.LBRACE:
                self.adv()
                return self.parse_record_literal(text, tok.pos)
            self.adv()
            return A.Name(pos=tok.pos, end=tok.end, id=text)

        raise self.error(
            f"expected an expression but found {tok.descr}", tok,
            help_text="expressions include literals, names, calls, "
                      "`[ ... ]` lists and `{ ... }` maps",
        )

    def _starts_expr(self) -> bool:
        return self.peek().kind in (
            TokenKind.IDENT, TokenKind.INT, TokenKind.FLOAT, TokenKind.STRING,
            TokenKind.CHAR, TokenKind.DURATION, TokenKind.LPAREN,
            TokenKind.LBRACKET, TokenKind.LBRACE, TokenKind.MINUS,
            TokenKind.BANG, TokenKind.NOT,
        )

    def parse_brace_literal(self) -> A.Expr:
        tok = self.expect(TokenKind.LBRACE, "`{`")
        entries: List[tuple] = []
        items: List[A.Expr] = []
        is_map = False
        self.end_of_line()
        while not self.at(TokenKind.RBRACE, TokenKind.EOF):
            key = self.parse_expr()
            if self.accept(TokenKind.COLON):
                is_map = True
                value = self.parse_expr()
                entries.append((key, value))
            else:
                items.append(key)
            self.end_of_line()
            if not self.accept(TokenKind.COMMA):
                self.end_of_line()
                if self.at(TokenKind.RBRACE, TokenKind.EOF):
                    break
                if not self._starts_expr():
                    break
        self.expect(TokenKind.RBRACE, "`}`")
        if is_map:
            return A.MapLit(pos=tok.pos, entries=entries)
        return A.SetLit(pos=tok.pos, items=items)

    def parse_record_literal(self, name: str, pos: SourcePos) -> A.Expr:
        self.expect(TokenKind.LBRACE, "`{`")
        fields: List[tuple] = []
        self.end_of_line()
        while not self.at(TokenKind.RBRACE, TokenKind.EOF):
            key_tok = self.peek()
            if key_tok.kind is not TokenKind.IDENT \
                    and key_tok.kind not in KEYWORD_TOKEN_KINDS:
                raise self.error(
                    f"expected a field name but found {key_tok.descr}", key_tok)
            key = self.adv().text
            self.expect(TokenKind.COLON, "`:`")
            value = self.parse_expr()
            fields.append((key, value))
            self.end_of_line()
            if not self.accept(TokenKind.COMMA):
                self.end_of_line()
                if self.at(TokenKind.RBRACE, TokenKind.EOF):
                    break
        self.expect(TokenKind.RBRACE, "`}`")
        return A.RecordLit(pos=pos, name=name, fields=fields)


def parse_source(source: str, filename: str = "<input>") -> A.Module:
    """Lex and parse a whole compilation unit."""
    tokens = Lexer(source, filename).tokenize()
    return Parser(tokens, filename, source).parse_module()
