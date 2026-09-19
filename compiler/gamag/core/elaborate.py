"""Elaboration: the derived execution graph becomes runnable code.

The core language has no statements, so it cannot be executed directly.  This
module translates a validated :class:`~gamag.core.ast.ExecutionGraph` into the
abstract syntax of the reference slice (v0.1), which already has a tested type
checker, effect checker, secret-propagation analysis, GIR builder, optimiser and
virtual machine.

That division is deliberate and it is the honest one.  The *language* -- what a
program means, what may be written, what the compiler must prove -- lives in
:mod:`gamag.core`.  The *machine* that runs it is the existing vertical slice,
which the audit report already classifies as the research reference
implementation.  Nothing here adds a feature to the core; it only realises the
core's decisions in a form the reference machine understands:

=========================  ==================================================
core construct             elaborated form
=========================  ==================================================
``source x : T from e``    ``let x : T = e`` in ``main``, passed as an argument
``state s : T starts e``   ``var s : T = e``
``operation``              ``let <yields> : T = <computes>`` then ``require``
                           for each ``holds``, then an audit record for ``trail``
guarded alternatives       ``var`` plus an ``if``/``elif`` chain whose ``else``
                           is ``panic("NoActiveAlternative: ...")``
``refine ... within n``    ``var`` plus a ``while`` carrying an explicit round
                           counter that panics ``RefinementDiverged`` at ``n``
``each ... over xs as x``  ``var`` accumulator plus a ``for`` over ``xs``
``resolve ... choose``     ``var`` plus a ``match``, so the existing
                           exhaustiveness check applies
``transition``             an assignment to the state, in the commit phase
=========================  ==================================================

Note what the elaborated code contains that the core program does not: mutable
bindings, a loop, an ``if``.  Those are the reference machine's implementation
of single-assignment bindings, bounded refinement and guarded selection.  A core
program still cannot express an unbounded loop, an assignment, or a selection
whose alternatives are not exhaustive -- the elaboration adds the machinery, but
only in the shapes the core's rules allow.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .. import ast_nodes as A
from . import ast as C


def _text(value: str, pos) -> A.Literal:
    return A.Literal(pos=pos, value=value, lit_kind="string")


def _int(value: int, pos) -> A.Literal:
    return A.Literal(pos=pos, value=value, lit_kind="int")


def _call(name: str, args: List[A.Expr], pos) -> A.Call:
    return A.Call(pos=pos, callee=A.Name(pos=pos, id=name), args=args)


class Elaborator:
    def __init__(self, module: C.CoreModule, graph: C.ExecutionGraph):
        self.m = module
        self.g = graph
        self.stmts: List[A.Stmt] = []
        self.emitted_alternatives: set = set()

    # ------------------------------------------------------------------
    def build(self) -> A.Module:
        intent = self.m.intent
        name = intent.name if intent else "Intent"
        pos = (intent.pos if intent else self.m.pos) or None

        body: List[A.Stmt] = []
        body += self._states()
        body += self._compute_phase()
        body += self._commit_phase()
        body.append(self._outcome(pos))

        effects = self._effects()
        fn = A.FnDecl(
            pos=pos, name=name,
            params=[A.Param(pos=s.pos, name=s.name, type=s.type)
                    for s in self.m.sources],
            ret=self._outcome_type(),
            effects=effects,
            body=A.Block(pos=pos, stmts=body),
            deterministic=("io" not in effects and "network" not in effects
                           and "unsafe" not in effects),
        )

        decls: List[A.Decl] = [fn]
        grants: List[str] = list(intent.authority) if intent else []
        top_level: List[A.Stmt] = []
        main = self._main(name, pos)
        if main is not None:
            decls.append(main)

        return A.Module(pos=pos, decls=decls, top_level=top_level,
                        grants=grants,
                        filename=self.m.filename or "<core>")

    # ------------------------------------------------------------------
    def _effects(self) -> List[str]:
        found = []
        for op in self.m.operations:
            if op.effect and op.effect not in found:
                found.append(op.effect)
            if op.trail and "audit" not in found:
                found.append("audit")
        if not found:
            return ["pure"]
        if "pure" in found and len(found) > 1:
            found.remove("pure")
        return found

    def _outcome_type(self) -> Optional[A.TypeRef]:
        binding = self.g.outcome
        for src in self.m.sources:
            if src.name == binding:
                return src.type
        for st in self.m.states:
            if st.name == binding:
                return st.type
        for op in self.m.operations:
            if op.yields == binding and op.yields_type is not None:
                if op.kind == "each":
                    return A.TypeRef(pos=op.pos, name="List",
                                     args=[op.yields_type])
                return op.yields_type
        return None

    # ------------------------------------------------------------------
    def _states(self) -> List[A.Stmt]:
        out: List[A.Stmt] = []
        for st in self.m.states:
            value = st.starts.expr if st.starts else None
            out.append(A.LetDecl(pos=st.pos, name=st.name, type=st.type,
                                 value=value, mutable=True, secret=st.secret))
        return out

    def _compute_phase(self) -> List[A.Stmt]:
        out: List[A.Stmt] = []
        for name in self.g.order():
            node = self.g.nodes.get(name)
            if node is None:
                continue
            op = node.decl
            if op.kind == "transition":
                continue
            binding = node.produces
            alt = self.g.alternatives.get(binding)
            if alt is not None:
                # emit the whole selection once, at its last member, so every
                # guard's dependencies are already bound
                if name != alt.members[-1].name:
                    continue
                if binding in self.emitted_alternatives:
                    continue
                self.emitted_alternatives.add(binding)
                out += self._selection(alt)
                continue
            out += self._operation(op)
        return out

    def _commit_phase(self) -> List[A.Stmt]:
        out: List[A.Stmt] = []
        for op in self.m.operations:
            if op.kind != "transition":
                continue
            if op.computes is not None:
                out.append(A.Assign(pos=op.pos,
                                    target=A.Name(pos=op.pos, id=op.alters),
                                    value=op.computes.expr))
            out += self._constraints(op)
            out += self._trail(op)
        return out

    # ------------------------------------------------------------------
    def _operation(self, op: C.OperationDecl) -> List[A.Stmt]:
        pos = op.pos
        if op.kind == "operation":
            value = op.computes.expr if op.computes else None
            stmts: List[A.Stmt] = [
                A.LetDecl(pos=pos, name=op.yields, type=op.yields_type,
                          value=value, secret=op.yields_secret)]
        elif op.kind == "refine":
            stmts = self._refine(op)
        elif op.kind == "each":
            stmts = self._each(op)
        elif op.kind == "resolve":
            stmts = self._resolve(op)
        else:
            stmts = []
        stmts += self._constraints(op)
        stmts += self._trail(op)
        return stmts

    def _constraints(self, op: C.OperationDecl) -> List[A.Stmt]:
        """`holds` becomes a runtime gate that quotes the constraint back."""
        out: List[A.Stmt] = []
        for hold in op.holds:
            if hold.expr is None:
                continue
            # The verbatim constraint becomes the failure message: a
            # ContractViolation should say which promise broke, in the words the
            # program used, not merely that one did.
            text = hold.text or None
            out.append(A.Require(pos=hold.pos, expr=hold.expr,
                                 message=(f"holds `{text}`" if text else None),
                                 phrase=text))
        return out

    def _trail(self, op: C.OperationDecl) -> List[A.Stmt]:
        if not op.trail:
            return []
        intent = self.m.intent.name if self.m.intent else ""
        fields = [("action", _text(op.name, op.pos)),
                  ("intent", _text(intent, op.pos)),
                  ("record", _text(op.trail, op.pos))]
        # The produced value is recorded only when it is not secret.  Secret
        # propagation is the reference checker's job and it refuses to let a
        # secret reach an audit field, so the core must not synthesise a program
        # that trips its own safety rule: a secret operation's trail records
        # that it ran, never what it computed.
        if op.yields and op.kind != "transition" and not op.yields_secret:
            fields.append(("value", A.Name(pos=op.pos, id=op.yields)))
        return [A.AuditRecord(pos=op.pos, fields=fields)]

    # ------------------------------------------------------------------
    def _selection(self, alt: C.Alternative) -> List[A.Stmt]:
        """Guarded alternatives -> a chain the machine can run.

        Built from the fallback upwards, so the `else` that faults with
        NoActiveAlternative is always the innermost branch.  It stays even when
        the compiler proved the guards complementary: a syntactic proof is a
        reason to trust the program, not a reason to remove the check.
        """
        first = alt.members[0]
        binding = alt.binding
        pos = first.pos
        message = (f"NoActiveAlternative: nothing yields `{binding}` -- "
                   f"{len(alt.members)} guarded alternatives, none active")
        if alt.proven_exhaustive:
            message += " (the guards were proven complementary)"
        node: A.Stmt = A.ExprStmt(
            pos=pos, expr=_call("panic", [_text(message, pos)], pos))
        node = A.Block(pos=pos, stmts=[node])

        for op in reversed(alt.members):
            body: List[A.Stmt] = [
                A.Assign(pos=op.pos, target=A.Name(pos=op.pos, id=binding),
                         value=op.computes.expr if op.computes else None)]
            body += self._constraints(op)
            body += self._trail(op)
            # `else` always takes a block in the reference AST, so a nested
            # alternative has to be wrapped rather than chained directly.
            otherwise = node if isinstance(node, A.Block) else A.Block(
                pos=op.pos, stmts=[node])
            node = A.If(pos=op.pos,
                        cond=op.when.expr if op.when else None,
                        then_body=A.Block(pos=op.pos, stmts=body),
                        else_body=otherwise)

        decl = A.LetDecl(pos=pos, name=binding, type=first.yields_type,
                         value=None, mutable=True, secret=first.yields_secret)
        return [decl, node]

    def _refine(self, op: C.OperationDecl) -> List[A.Stmt]:
        pos = op.pos
        counter = f"__rounds_{op.yields}"
        bound = op.within if op.within is not None else 1
        until_text = op.until.text if op.until else "the stopping constraint"
        message = (f"RefinementDiverged: `{op.name}` did not satisfy "
                   f"`{until_text}` within {bound} rounds")
        loop_body: List[A.Stmt] = [
            A.If(pos=pos,
                 cond=A.Binary(pos=pos, op=">=",
                               left=A.Name(pos=pos, id=counter),
                               right=_int(bound, pos)),
                 then_body=A.Block(pos=pos, stmts=[
                     A.ExprStmt(pos=pos, expr=_call("panic",
                                                    [_text(message, pos)],
                                                    pos))]),
                 else_body=None),
            A.Assign(pos=pos, target=A.Name(pos=pos, id=op.yields),
                     value=op.repeats.expr if op.repeats else None),
            A.Assign(pos=pos, target=A.Name(pos=pos, id=counter),
                     value=A.Binary(pos=pos, op="+",
                                    left=A.Name(pos=pos, id=counter),
                                    right=_int(1, pos))),
        ]
        if op.holds:
            loop_body += self._constraints(op)
        return [
            A.LetDecl(pos=pos, name=op.yields, type=op.yields_type,
                      value=op.starts.expr if op.starts else None,
                      mutable=True, secret=op.yields_secret),
            A.LetDecl(pos=pos, name=counter,
                      type=A.TypeRef(pos=pos, name="I64"), value=_int(0, pos),
                      mutable=True),
            A.While(pos=pos,
                    cond=A.Unary(pos=pos, op="not",
                                 operand=op.until.expr if op.until else None),
                    body=A.Block(pos=pos, stmts=loop_body)),
        ]

    def _each(self, op: C.OperationDecl) -> List[A.Stmt]:
        pos = op.pos
        item = op.over.item if op.over else "item"
        collection = op.over.expr if op.over else None
        element = A.ListLit(pos=pos,
                            items=[op.computes.expr if op.computes else None])
        append = A.Assign(
            pos=pos, target=A.Name(pos=pos, id=op.yields),
            value=A.Binary(pos=pos, op="+",
                           left=A.Name(pos=pos, id=op.yields), right=element))
        body: List[A.Stmt] = []
        if op.when is not None:
            body.append(A.If(pos=pos, cond=op.when.expr,
                             then_body=A.Block(pos=pos, stmts=[append]),
                             else_body=None))
        else:
            body.append(append)
        list_type = A.TypeRef(pos=pos, name="List",
                              args=[op.yields_type] if op.yields_type else [])
        return [
            A.LetDecl(pos=pos, name=op.yields, type=list_type,
                      value=A.ListLit(pos=pos, items=[]), mutable=True),
            A.For(pos=pos, var=item, iter=collection,
                  body=A.Block(pos=pos, stmts=body)),
        ]

    def _resolve(self, op: C.OperationDecl) -> List[A.Stmt]:
        pos = op.pos
        arms = [A.MatchArm(pos=pos, pattern=pattern,
                           body=[A.Assign(
                               pos=pos,
                               target=A.Name(pos=pos, id=op.yields),
                               value=value)])
                for pattern, value in op.choices]
        return [
            A.LetDecl(pos=pos, name=op.yields, type=op.yields_type, value=None,
                      mutable=True, secret=op.yields_secret),
            A.Match(pos=pos, subject=op.over.expr if op.over else None,
                    arms=arms),
        ]

    # ------------------------------------------------------------------
    def _outcome(self, pos) -> A.Stmt:
        return A.Return(pos=pos,
                        value=A.Name(pos=pos, id=self.g.outcome)
                        if self.g.outcome else None)

    def _main(self, intent_name: str, pos) -> Optional[A.FnDecl]:
        """A runnable entry point, when every source says where it comes from."""
        unbound = [s.name for s in self.m.sources if not getattr(s, "from_expr",
                                                                 None)]
        if unbound:
            return None
        stmts: List[A.Stmt] = []
        args: List[A.Expr] = []
        for src in self.m.sources:
            stmts.append(A.LetDecl(pos=src.pos, name=src.name, type=src.type,
                                   value=src.from_expr, secret=src.secret))
            args.append(A.Name(pos=src.pos, id=src.name))
        result = "__outcome"
        stmts.append(A.LetDecl(pos=pos, name=result, type=self._outcome_type(),
                               value=_call(intent_name, args, pos)))
        stmts.append(A.ExprStmt(pos=pos,
                                expr=_call("print",
                                           [A.Name(pos=pos, id=result)], pos)))
        effects = [e for e in self._effects() if e != "pure"]
        if "io" not in effects:
            effects.append("io")
        return A.FnDecl(pos=pos, name="main", params=[],
                        ret=A.TypeRef(pos=pos, name="Unit"), effects=effects,
                        body=A.Block(pos=pos, stmts=stmts))


def elaborate(module: C.CoreModule, graph: C.ExecutionGraph) -> A.Module:
    """Public entry point."""
    return Elaborator(module, graph).build()
