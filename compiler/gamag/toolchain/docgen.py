"""Documentation generation for `ggc doc` (spec section 31, toolchain).

Two audiences, one source of truth.  The *programmer* mode reads a compiled
source file and documents what it declares, including the parts of a Gama-G
program that only the checker computes -- derived execution order, capability
demands, which `holds` the compiler proved and which it only checks at run
time.  The *library* mode renders the standard library from the same
registration table the checker consults, so documentation and callable
surface cannot drift apart.

Doc comments are the ``//`` block immediately above a declaration -- the only
comment convention this adds, chosen because the lexer already treats comments
as skippable: documentation that the compiler cannot corrupt is documentation
that survives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..std import library as L


# ---------------------------------------------------------------------------
# doc comments
# ---------------------------------------------------------------------------
def collect_doc_comments(source: str) -> Dict[int, str]:
    """Map a declaration line number to the comment block above it.

    A comment block attaches to line L when the lines L-k..L-1 (for some
    k >= 1) are comment lines and L-k-1 is not -- a blank line or code breaks
    the block.  The text of each comment is kept verbatim inside the block;
    only the leading `//` and indentation are removed.
    """
    lines = source.split("\n")
    out: Dict[int, str] = {}
    block: List[str] = []
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("//") and not stripped.startswith("///"):
            block.append(stripped[2:].strip())
            continue
        if stripped.startswith("///"):
            # tolerate the `///` style as well; it is the one doc-comment
            # spelling a reader arrives with from other languages
            block.append(stripped[3:].strip())
            continue
        if block:
            if stripped:                     # a declaration follows directly
                out[idx + 1] = "\n".join(block)
            block = []
    return out


def doc_for(pos: Optional[Any], docs: Dict[int, str]) -> str:
    if pos is None:
        return ""
    return docs.get(getattr(pos, "line", 0), "")


# ---------------------------------------------------------------------------
# v0.1 surface
# ---------------------------------------------------------------------------
def _param_text(p: Any) -> str:
    type_text = ""
    if getattr(p, "type", None) is not None:
        type_text = p.type.render()
    return f"{p.name}: {type_text}" if type_text else p.name


def _contract_lines(decl: Any) -> List[str]:
    out = []
    for c in getattr(decl, "contracts", []) or []:
        out.append(f"{c.kind} `{c.text}`")
    return out


def render_markdown(compilation: Any, title: Optional[str] = None) -> str:
    """The markdown documentation of one compiled file."""
    lines: List[str] = []
    path = compilation.path
    docs = collect_doc_comments(compilation.source)
    lines.append(f"# {title or path}")
    lines.append("")
    lines.append(f"*Gama-G {compilation.dialect} — compiled with the "
                 f"`{compilation.profile}` profile.*")
    lines.append("")

    if compilation.dialect == "core":
        _render_core(compilation, docs, lines)
        return "\n".join(lines) + "\n"
    _render_v01(compilation, docs, lines)
    return "\n".join(lines) + "\n"


def _render_v01(compilation: Any, docs: Dict[int, str],
                lines: List[str]) -> None:
    module = compilation.module
    if module is None:
        lines.append("_No parsed module is available; nothing to document._")
        return
    checker = compilation.checker
    grants = sorted(set(getattr(module, "grants", []) or []))
    if grants:
        lines.append("**Granted capabilities:** " + ", ".join(f"`{g}`"
                                                               for g in grants))
        lines.append("")
    for decl in module.decls:
        kind = type(decl).__name__
        doc = doc_for(getattr(decl, "pos", None), docs)
        if kind == "GrantDecl":
            continue
        if kind == "FnDecl" or kind == "PipelineDecl":
            params = ", ".join(_param_text(p) for p in decl.params)
            ret = decl.ret.render() if getattr(decl, "ret", None) else "Unit"
            effects = list(getattr(decl, "effects", []) or [])
            head = (f"### `{decl.name}({params}) -> {ret}`"
                    if kind == "FnDecl" else
                    f"### pipeline `{decl.name}({params}) -> {ret}`")
            lines.append(head)
            lines.append("")
            if doc:
                lines.append(doc)
                lines.append("")
            info = (checker.functions.get(decl.name)
                    if checker is not None else None)
            facts = []
            if effects:
                facts.append("effects: " + ", ".join(f"`{e}`" for e in effects))
            if info and info.caps:
                facts.append("capabilities: " + ", ".join(
                    f"`{c}`" for c in sorted(info.caps)))
            if facts:
                lines.append("  " + " — ".join(facts))
                lines.append("")
            contract_lines = _contract_lines(decl)
            if contract_lines:
                lines.append("  Contracts (checked at run time, quoted when "
                             "they fail):")
                for c in contract_lines:
                    lines.append(f"  - {c}")
                lines.append("")
            continue
        if kind == "RecordDecl":
            lines.append(f"### record `{decl.name}`")
            lines.append("")
            if doc:
                lines.append(doc)
                lines.append("")
            for f in decl.fields:
                lines.append(f"- `{_param_text(f)}`")
            lines.append("")
            continue
        if kind == "EnumDecl":
            lines.append(f"### enum `{decl.name}`")
            lines.append("")
            if doc:
                lines.append(doc)
                lines.append("")
            for v in decl.variants:
                ps = ", ".join(_param_text(p) for p in v.params)
                lines.append(f"- `{v.name}({ps})`" if ps else f"- `{v.name}`")
            lines.append("")
            continue
        if kind == "ServiceDecl":
            lines.append(f"### service `{decl.name}`")
            lines.append("")
            if doc:
                lines.append(doc)
                lines.append("")
            steps = getattr(decl, "recover", []) or []
            if steps:
                lines.append("Recovery policy, in escalation order:")
                for s in steps:
                    lines.append(f"- `{getattr(s, 'action', s)}`")
                lines.append("")
            continue
        if kind == "TestDecl":
            lines.append(f"### test `{decl.name}`  _[{decl.category}]_")
            if doc:
                lines.append("")
                lines.append(doc)
            lines.append("")
            continue
        if kind == "PolicyDecl":
            lines.append(f"### policy `{decl.name}`")
            lines.append("")
            if doc:
                lines.append(doc)
                lines.append("")
            for r in getattr(decl, "rules", []) or []:
                text = getattr(r, "text", "") or getattr(r, "name", "")
                lines.append(f"- {getattr(r, 'decision', 'rule')}: `{text}`")
            lines.append("")
            continue
        if kind in ("AgentDecl", "TransactionDecl", "ModelDecl"):
            noun = kind.replace("Decl", "").lower()
            lines.append(f"### {noun} `{decl.name}`")
            if doc:
                lines.append("")
                lines.append(doc)
            lines.append("")
            continue


# ---------------------------------------------------------------------------
# core surface -- what the checker derived, not just what was written
# ---------------------------------------------------------------------------
def _render_core(compilation: Any, docs: Dict[int, str],
                 lines: List[str]) -> None:
    model = compilation.core_model
    syntax = compilation.core_syntax
    if model is None:
        lines.append("_No semantic model is available; nothing to document._")
        return
    intent = model.intent
    lines.append(f"## intent `{intent.name}`")
    lines.append("")
    if intent.purpose:
        lines.append(intent.purpose)
        lines.append("")
    if intent.authority:
        lines.append("**Authority:** " + ", ".join(f"`{c}`"
                                                   for c in intent.authority))
        lines.append("")
    if intent.trail:
        lines.append(f"**Trails:** `{intent.trail}`")
        lines.append("")
    graph = model.operations
    if graph.outcome:
        lines.append(f"**Outcome:** `{graph.outcome}`")
        lines.append("")

    lines.append("### Inputs")
    lines.append("")
    # `graph.inputs` and `graph.states` hold the binding-keyed nodes;
    # `graph.nodes` holds only the operations the intent declares.
    for name in sorted(graph.inputs):
        node = graph.inputs[name]
        kind = "secret source" if node.secret else "source"
        t = node.type.render() if node.type else "?"
        fixed = ""
        origin = getattr(node, "origin", None)
        if origin is not None:
            shown = getattr(origin, "value", None)
            shown = shown if shown is not None else getattr(
                origin, "text", "")
            fixed = f", fixed at `{shown}`"
        lines.append(f"- `{node.produces}: {t}` — {kind}{fixed}")
    for name in sorted(graph.resources):
        node = graph.resources[name]
        t = node.type.render() if getattr(node, "type", None) else "?"
        authority = (", authority `" + ", ".join(node.authority) + "`"
                     if getattr(node, "authority", None) else "")
        lines.append(f"- `{name}: {t}` — state, the only mutable resource"
                     f"{authority}")
    if not graph.inputs and not graph.resources:
        lines.append("_The intent takes no inputs: it is a library._")
    lines.append("")

    lines.append("### Operations")
    lines.append("")
    order = graph.order()
    for name in order:
        node = graph.nodes[name]
        if node.kind in ("source", "state"):
            continue
        doc = doc_for(node.pos, docs)
        title = {"compute": "operation", "refine": "refine",
                 "fanout": "each", "dispatch": "resolve",
                 "transition": "transition"}.get(node.kind, node.kind)
        yields = (f"`{node.produces}`" if node.produces else "(no binding)")
        lines.append(f"#### {title} `{node.name}` — yields {yields}"
                     + (f", level {node.level}" if node.level is not None
                        else ""))
        lines.append("")
        if doc:
            lines.append(doc)
            lines.append("")
        if node.consumes:
            lines.append(f"- reads: " + ", ".join(f"`{u}`"
                                                   for u in node.consumes))
        if node.effects:
            lines.append("- effects: " + ", ".join(f"`{e}`"
                                                    for e in node.effects))
        if node.needs:
            lines.append("- needs: " + ", ".join(f"`{n}`"
                                                  for n in node.needs))
        for h in node.holds:
            mark = ("proven at compile time"
                    if getattr(h, "discharge", "") == "proven"
                    else "checked at run time")
            lines.append(f"- holds `{h.text}` ({mark})")
        if node.when is not None:
            lines.append(f"- when `{node.when.text}`")
        lines.append("")

    if graph.selections:
        lines.append("### Selections the compiler proved")
        lines.append("")
        for binding, sel in sorted(graph.selections.items()):
            state = []
            if sel.proven_exclusive:
                state.append("mutually exclusive")
            if sel.proven_exhaustive:
                state.append("exhaustive")
            verdict = (" and ".join(state) if state
                       else "not proven — a run-time fault covers the gap")
            members = ", ".join(f"`{m}`" for m in sel.members)
            lines.append(f"- `{binding}` over {members}: {verdict}")
        lines.append("")


# ---------------------------------------------------------------------------
# standard-library mode
# ---------------------------------------------------------------------------
def render_stdlib_markdown(module: Optional[str] = None) -> str:
    lines = ["# Gama-G standard library", ""]
    lines.append("_Generated from the same table the checker reads "
                 "(`gamag.std.library`); the documentation cannot claim a "
                 "function the compiler does not register._")
    lines.append("")
    names = ([module] if module else sorted(L.MODULES))
    for name in names:
        members = L.MODULES.get(name)
        if members is None:
            lines.append(f"## `{name}`")
            lines.append("")
            reason = L.UNIMPLEMENTED_MODULES.get(
                name, "not a standard library module")
            lines.append(f"_Not implemented: {reason}._")
            lines.append("")
            continue
        lines.append(f"## `{name}` — {len(members)} members")
        lines.append("")
        for member in sorted(members):
            b = L.BUILTINS.get(f"{name}.{member}") or L.BUILTINS.get(member)
            if b is None:
                lines.append(f"- `{name}.{member}`")
                continue
            sig = ", ".join(b.params)
            lines.append(f"- **`{name}.{member}({sig}) -> {b.ret}`**")
            if b.doc:
                lines.append(f"  {b.doc}")
            notes = []
            if b.effects:
                notes.append("effects: " + ", ".join(f"`{e}`"
                                                      for e in b.effects))
            if b.caps:
                notes.append("capabilities: " + ", ".join(f"`{c}`"
                                                           for c in b.caps))
            if notes:
                lines.append("  (" + "; ".join(notes) + ")")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# structured output for tooling
# ---------------------------------------------------------------------------
def doc_dict(compilation: Any) -> Dict[str, Any]:
    """The same facts as `render_markdown`, as data."""
    payload: Dict[str, Any] = {
        "path": compilation.path,
        "dialect": compilation.dialect,
        "profile": compilation.profile,
    }
    if compilation.dialect == "core" and compilation.core_model is not None:
        model = compilation.core_model
        graph = model.operations
        payload["intent"] = {
            "name": model.intent.name,
            "purpose": model.intent.purpose,
            "authority": list(model.intent.authority),
            "outcome": graph.outcome,
        }
        payload["nodes"] = [
            {"name": graph.nodes[k].name, "kind": graph.nodes[k].kind,
             "yields": graph.nodes[k].produces,
             "level": graph.nodes[k].level,
             "consumes": list(graph.nodes[k].consumes),
             "effects": list(graph.nodes[k].effects)}
            for k in graph.order()
        ]
        payload["selections"] = {
            b: {"members": list(s.members),
                "proven_exclusive": s.proven_exclusive,
                "proven_exhaustive": s.proven_exhaustive}
            for b, s in graph.selections.items()}
        return payload
    module = compilation.module
    if module is not None:
        docs = collect_doc_comments(compilation.source)
        decls = []
        for decl in module.decls:
            entry: Dict[str, Any] = {
                "kind": type(decl).__name__.replace("Decl", "").lower(),
                "name": getattr(decl, "name", ""),
            }
            pos = getattr(decl, "pos", None)
            if pos is not None:
                entry["line"] = pos.line
                entry["doc"] = docs.get(pos.line, "")
            if entry["kind"] == "fn":
                entry["params"] = [_param_text(p) for p in decl.params]
                entry["returns"] = (decl.ret.render()
                                    if getattr(decl, "ret", None) else "Unit")
                entry["effects"] = list(getattr(decl, "effects", []) or [])
                entry["contracts"] = _contract_lines(decl)
            if entry["kind"] == "record":
                entry["fields"] = [_param_text(f) for f in decl.fields]
            if entry["kind"] == "enum":
                entry["variants"] = [v.name for v in decl.variants]
            decls.append(entry)
        payload["declarations"] = decls
        payload["grants"] = sorted(set(getattr(module, "grants", []) or []))
    return payload
