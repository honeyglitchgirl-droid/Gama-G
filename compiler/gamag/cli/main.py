"""``ggc`` -- the Gama-G compiler driver (spec section 2).

One entry point for the whole pipeline.  Subcommands stop at whichever stage
the user asked for, which matters for tooling: an editor wants diagnostics
without codegen, a build wants GIR without execution.

Exit codes are stable so scripts can branch on them:
    0  success
    1  compile-time diagnostics (errors) were reported
    2  the program compiled but faulted at runtime
    3  the command line was wrong
    4  a test failed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import GIR_VERSION, SPEC_VERSION, __version__
from ..diagnostics import GamaRuntimeFault
from ..driver import Compilation, compile_file, execute, find_entry, program_grants
from ..std import library as L

EXIT_OK = 0
EXIT_COMPILE = 1
EXIT_RUNTIME = 2
EXIT_USAGE = 3
EXIT_TEST = 4

PROFILES = ("strict", "standard", "lenient")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggc",
        description="The Gama-G compiler driver.",
        epilog="Gama-G " + __version__ + " -- GIR " + GIR_VERSION
               + ", specification " + SPEC_VERSION,
    )
    parser.add_argument("--version", action="version",
                        version=f"ggc {__version__} (GIR {GIR_VERSION}, "
                                f"spec {SPEC_VERSION})")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser, *, files: bool = True) -> None:
        if files:
            p.add_argument("files", nargs="+", metavar="FILE.gg",
                           help="Gama-G source files")
        p.add_argument("--profile", choices=PROFILES, default="strict",
                       help="how aggressively to diagnose (default: strict)")
        p.add_argument("-O", dest="opt", type=int, choices=(0, 1, 2),
                       default=1, metavar="LEVEL",
                       help="optimization level (default: 1)")
        p.add_argument("--grant", action="append", default=[], metavar="CAP",
                       help="grant a capability the module did not ask for "
                            "(repeatable)")
        p.add_argument("--no-color", action="store_true",
                       help="plain diagnostics, no ANSI escapes")
        p.add_argument("--json", action="store_true",
                       help="machine-readable output")

    p = sub.add_parser("check", help="type-check and report diagnostics")
    add_common(p)

    p = sub.add_parser("build", help="compile to GIR")
    add_common(p)
    p.add_argument("-o", "--output", metavar="PATH",
                   help="write GIR JSON here (default: alongside the source)")
    p.add_argument("--stats", action="store_true",
                   help="print instruction counts and pass results")

    p = sub.add_parser("gir", help="print the GIR in readable form")
    add_common(p)
    p.add_argument("--fn", metavar="NAME", help="print only this function")

    p = sub.add_parser(
        "graph",
        help="print the execution graph the core derived from relationships")
    add_common(p)
    p.add_argument("--edges", action="store_true",
                   help="list every derived dependency edge")

    p = sub.add_parser(
        "memory",
        help="print the memory and resource model the core derived")
    add_common(p)
    p.add_argument("--slots", action="store_true",
                   help="list every slot and the bindings that occupied it")

    p = sub.add_parser(
        "native",
        help="compile to machine code through the native backend")
    add_common(p)
    p.add_argument("-o", "--output", metavar="PATH",
                   help="write the executable here (default: the build dir)")
    p.add_argument("--build-dir", metavar="DIR", default=None,
                   help="where to put generated C and the executable "
                        "(default: .ggbuild)")
    p.add_argument("--emit-c", metavar="PATH",
                   help="write the generated C here and do not compile it")
    p.add_argument("--keep-c", action="store_true",
                   help="keep the generated C next to the executable")
    p.add_argument("--cc-opt", dest="cc_opt", type=int, default=2,
                   choices=(0, 1, 2, 3),
                   help="optimisation level passed to the C compiler "
                        "(default: 2). This is not `-O`, which optimises the GIR")
    p.add_argument("--cc", metavar="PROG",
                   help="the C compiler to use (default: $CC, cc, gcc, clang)")

    p = sub.add_parser(
        "difftest",
        help="run a program on the interpreter and the native backend and "
             "compare them")
    add_common(p)
    p.add_argument("--build-dir", metavar="DIR", default=None)
    p.add_argument("--show-refused", action="store_true",
                   help="also list what the backend declined to compile")

    p = sub.add_parser(
        "wasm",
        help="emit a WebAssembly module for the numeric subset")
    add_common(p)
    p.add_argument("-o", "--output", metavar="PATH",
                   help="write the .wasm module here")
    p.add_argument("--disassemble", action="store_true",
                   help="print the emitted instructions instead of writing")
    p.add_argument("--analyze", action="store_true",
                   help="report what can and cannot be expressed, and stop")

    p = sub.add_parser(
        "device",
        help="report the accelerator devices present and the kernels available")
    p.add_argument("--kernels", nargs="*", metavar="NAME",
                   help="print these kernels (default: all)")
    p.add_argument("--list", action="store_true",
                   help="list the offloadable operations and stop")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("run", help="compile and execute")
    add_common(p)
    p.add_argument("--entry", metavar="NAME", default="main",
                   help="function to call (default: main)")
    p.add_argument("--stats", action="store_true",
                   help="print runtime counters afterwards")
    p.add_argument("--audit", metavar="PATH",
                   help="write the hash-chained audit log here")
    p.add_argument("--lenient-runtime", action="store_true",
                   help="run without the deterministic clock and RNG")

    p = sub.add_parser("test", help="run the program's `test` declarations")
    add_common(p)

    p = sub.add_parser("explain", help="describe the language surface")
    p.add_argument("topic", nargs="?", default="modules",
                   choices=("modules", "builtins", "capabilities", "effects",
                            "gaps"),
                   help="what to describe")
    p.add_argument("--module", metavar="NAME",
                   help="with `builtins`, restrict to one module")
    p.add_argument("--json", action="store_true")

    return parser


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _report(compilation: Compilation, color: bool, as_json: bool) -> None:
    if as_json:
        payload = [{
            "severity": d.severity.name.lower(),
            "phase": d.phase.value,
            "code": d.code,
            "message": d.message,
            "file": d.pos.file if d.pos else None,
            "line": d.pos.line if d.pos else None,
            "column": d.pos.col if d.pos else None,
            "help": d.help_text,
        } for d in compilation.bag.diagnostics]
        print(json.dumps({"ok": compilation.ok, "diagnostics": payload},
                         indent=2))
        return
    if compilation.bag.diagnostics:
        print(compilation.diagnostics_text(color), file=sys.stderr)


def _compile(paths: Sequence[str], args: argparse.Namespace
             ) -> Tuple[List[Compilation], int]:
    out: List[Compilation] = []
    status = EXIT_OK
    for path in paths:
        if not os.path.exists(path):
            print(f"ggc: no such file: {path}", file=sys.stderr)
            return [], EXIT_USAGE
        compilation = compile_file(path, profile=args.profile,
                                   opt_level=args.opt, grants=args.grant)
        out.append(compilation)
        if not compilation.ok:
            status = EXIT_COMPILE
    return out, status


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    compilations, status = _compile(args.files, args)
    for compilation in compilations:
        errors = len(compilation.errors)
        warnings = len(compilation.warnings)
        if args.json:
            _report(compilation, not args.no_color, True)
            continue
        _report(compilation, not args.no_color, False)
        if compilation.ok:
            caps = compilation.required_capabilities()
            effects = compilation.declared_effects()
            print(f"{compilation.path}: ok "
                  f"({warnings} warning{'s' if warnings != 1 else ''}"
                  f"{', capabilities: ' + ', '.join(caps) if caps else ''}"
                  f"{', effects: ' + ', '.join(effects) if effects else ''})")
        else:
            print(f"{compilation.path}: {errors} error"
                  f"{'s' if errors != 1 else ''}", file=sys.stderr)
    return status


def cmd_build(args: argparse.Namespace) -> int:
    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status
    for compilation in compilations:
        if compilation.program is None:
            continue
        out_path = args.output
        if out_path is None:
            out_path = os.path.splitext(compilation.path)[0] + ".gir.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(compilation.program.to_json())
        if args.json:
            payload: Dict[str, Any] = {
                "path": compilation.path, "output": out_path,
                "gir_version": GIR_VERSION,
                "stats": compilation.program.stats(),
                "optimization": (compilation.optimization.to_dict()
                                 if compilation.optimization else None),
            }
            print(json.dumps(payload, indent=2))
            continue
        stats = compilation.program.stats()
        print(f"{compilation.path} -> {out_path}")
        print(f"  GIR {GIR_VERSION}: {stats['functions']} functions, "
              f"{stats['blocks']} blocks, "
              f"{stats['live_instructions']} instructions")
        print(f"  audit points: {stats['audit_points']}, "
              f"recovery points: {stats['recovery_points']}, "
              f"parallel tasks: {stats['parallel_tasks']}")
        if args.stats and compilation.optimization:
            print("  " + compilation.optimization.summary().replace("\n", "\n  "))
    return EXIT_OK


def cmd_gir(args: argparse.Namespace) -> int:
    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status
    for compilation in compilations:
        if compilation.program is None:
            continue
        print(compilation.program.render(only=args.fn))
    return EXIT_OK


def cmd_graph(args: argparse.Namespace) -> int:
    """Print the semantic model the core derived from relationships.

    In a language where the programmer never writes an order, being able to see
    what the compiler derived is not a nicety -- it is how the programmer checks
    that the relationships they declared mean what they intended.  It prints all
    five graphs, because a promise that is only checked at runtime and a
    capability demand the intent does not meet are exactly the things a reader
    needs to see next to the order.
    """
    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status
    for compilation in compilations:
        model = compilation.core_model
        if model is None:
            print(f"{compilation.path}: not a core program.  Only a source that "
                  f"opens with `gama core <version>` has a derived model; the "
                  f"v0.1 surface writes its own order.", file=sys.stderr)
            return EXIT_USAGE
        graph = model.operations
        if getattr(args, "json", False):
            print(json.dumps({
                "path": compilation.path,
                "dialect": compilation.dialect,
                "backend": "native semantic IR -> GIR",
                "intent": {"name": model.intent.name,
                           "purpose": model.intent.purpose,
                           "authority": model.intent.authority,
                           "trail": model.intent.trail,
                           "outcome": model.intent.outcome},
                "sources": graph.sources,
                "states": graph.states,
                "levels": graph.levels,
                "edges": [{"from": up, "to": name} for up, name
                          in graph.edges()],
                "selections": {
                    binding: {"alternatives": sel.members,
                              "proven_exhaustive": sel.proven_exhaustive,
                              "proven_exclusive": sel.proven_exclusive}
                    for binding, sel in sorted(graph.selections.items())},
                "constraints": [
                    {"kind": c.kind, "text": c.text, "node": c.node,
                     "discharge": c.discharge, "fault": c.fault}
                    for c in model.constraints.constraints],
                "authority": {
                    "held": model.authority.held,
                    "unmet": [{"node": d.node, "missing": d.missing}
                              for d in model.authority.unmet()]},
                "recovery": [{"node": o.node, "kind": o.kind,
                              "bound": o.bound, "fault": o.fault}
                             for o in model.recovery.obligations],
            }, indent=2))
            continue
        print(model.render())
        if args.edges:
            print()
            print("derived dependency edges:")
            for name in graph.order():
                node = graph.nodes[name]
                upstream = ", ".join(node.upstream) or "(inputs only)"
                writes = node.produces
                print(f"  {writes} <- {upstream}   [{node.kind} {name}]")
        recorded = [f for f in (compilation.program.functions.values()
                                if compilation.program else [])
                    if f.parallel_tasks]
        if recorded and args.edges:
            print()
            print("recorded in GIR as operation-graph metadata:")
            for fn in recorded:
                for task in fn.parallel_tasks:
                    print(f"  {task.name}: reads={','.join(task.reads) or '-'} "
                          f"writes={','.join(task.writes) or '-'} "
                          f"depends_on={','.join(task.depends_on) or '-'}")
    return EXIT_OK


def cmd_memory(args: argparse.Namespace) -> int:
    """Print who owns what, for how long, and what that proves.

    Spec section 8 asks for ownership, borrowing, controlled mutation and
    deterministic destruction. This is those four as facts about one program
    rather than as properties of an allocator: every release point below is a
    level the compiler derived from the graph, and no collector decided any of
    them at some later moment it chose.

    It also prints the violations, which are expected to be none. A model that
    could not report its own inconsistency would only be a report.
    """
    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status
    for compilation in compilations:
        memory = compilation.core_memory
        if memory is None:
            print(f"{compilation.path}: not a core program, so there is no "
                  f"derived memory model to show.", file=sys.stderr)
            return EXIT_USAGE
        if getattr(args, "json", False):
            print(json.dumps({
                "path": compilation.path,
                "bindings": [
                    {"name": b.name, "kind": b.kind, "owner": b.owner,
                     "borrowers": b.borrowers, "type": b.type.render(),
                     "secret": b.secret, "mutable": b.mutable,
                     "first_level": b.first, "last_level": b.last,
                     "released_after": b.released_after,
                     "slot": b.slot, "shares_with": b.shares_with}
                    for b in sorted(memory.bindings.values(),
                                    key=lambda x: (x.first, x.name))],
                "slots": [{"index": life.index,
                           "occupants": [b.name for b in life.occupants],
                           "reused": life.reused} for life in memory.slots],
                "acquisitions": [
                    {"node": a.node, "resource": a.resource,
                     "effect": a.effect, "spans": a.spans}
                    for a in memory.acquisitions],
                "slots_saved": memory.slots_saved,
                "violations": memory.violations,
            }, indent=2))
            continue
        print(memory.render())
        if args.slots:
            print()
            print("slots:")
            for life in memory.slots:
                occupants = ", ".join(f"{b.name}[{b.first}..{b.last}]"
                                      for b in life.occupants)
                note = "  (reused)" if life.reused else ""
                print(f"  %{life.index}: {occupants}{note}")
        if memory.violations:
            # A violation is the compiler disagreeing with itself, not the
            # program being wrong, so it is reported as a compilation failure
            # rather than a runtime one.
            return EXIT_COMPILE
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status
    exit_code = EXIT_OK
    results: List[Dict[str, Any]] = []
    for compilation in compilations:
        entry = find_entry(compilation, args.entry)
        if entry is None:
            print(f"{compilation.path}: no `{args.entry}` function to run",
                  file=sys.stderr)
            exit_code = EXIT_USAGE
            continue
        from ..runtime.context import Context
        ctx = Context(
            deterministic=not args.lenient_runtime,
            grants=program_grants(compilation, args.grant),
            program_version=__version__)
        result = execute(compilation, entry=entry, context=ctx)
        if not result.ok:
            fault = result.fault
            if isinstance(fault, GamaRuntimeFault):
                kind = getattr(fault, "kind", "Fault")
                message = getattr(fault, "message", str(fault))
                position = getattr(fault, "pos", None)
                where = ""
                if position is not None:
                    where = f" at {position.file}:{position.line}:" \
                            f"{position.col}"
                print(f"runtime fault [{kind}]{where}: {message}",
                      file=sys.stderr)
                hint = getattr(fault, "hint", None)
                if hint:
                    print(f"  = help: {hint}", file=sys.stderr)
            else:
                print(f"internal error: {type(fault).__name__}: {fault}",
                      file=sys.stderr)
            exit_code = EXIT_RUNTIME
        if args.audit:
            with open(args.audit, "w", encoding="utf-8") as handle:
                handle.write(ctx.audit.to_jsonl())
            print(f"audit log written to {args.audit} "
                  f"({len(ctx.audit.records)} records, chain "
                  f"{'valid' if ctx.audit.verify()[0] else 'INVALID'})")
        if args.stats:
            counters = ctx.stats.to_dict()
            print("runtime: " + ", ".join(
                f"{k}={v}" for k, v in counters.items()
                if k not in ("started_at", "elapsed_seconds")),
                file=sys.stderr)
            print(f"runtime: elapsed={counters['elapsed_seconds'] * 1000:.2f}ms",
                  file=sys.stderr)
        if args.json:
            results.append({"path": compilation.path, "entry": entry,
                            "ok": result.ok,
                            "fault": (None if result.ok else
                                      str(result.fault))})
    if args.json:
        print(json.dumps(results, indent=2))
    return exit_code


def cmd_test(args: argparse.Namespace) -> int:
    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status
    from ..runtime.context import Context
    total = passed = 0
    failures: List[Dict[str, Any]] = []
    for compilation in compilations:
        if compilation.program is None:
            continue
        categories = compilation.program.metadata.get("test_categories", {})
        names = [n for n in compilation.program.functions
                 if n.startswith("test:")]
        for name in sorted(names):
            label = name.split(":", 1)[1]
            category = categories.get(name, "")
            total += 1
            ctx = Context(deterministic=True,
                          grants=program_grants(compilation, args.grant))
            from ..runtime.vm import VM
            vm = VM(compilation.program, ctx, compilation.checker)
            try:
                vm.run("<main>") if "<main>" in compilation.program.functions \
                    else None
                vm.call_function(name, [])
                passed += 1
                if not args.json:
                    print(f"  pass  {label}"
                          + (f"  [{category}]" if category else ""))
            except BaseException as exc:             # noqa: BLE001
                kind = getattr(exc, "kind", type(exc).__name__)
                message = getattr(exc, "message", str(exc))
                failures.append({"test": label, "category": category,
                                 "kind": kind, "message": str(message)})
                if not args.json:
                    print(f"  FAIL  {label}"
                          + (f"  [{category}]" if category else ""))
                    print(f"        {kind}: {str(message)[:200]}")
    if args.json:
        print(json.dumps({"total": total, "passed": passed,
                          "failed": total - passed, "failures": failures},
                         indent=2))
    else:
        print(f"{passed}/{total} test(s) passed")
    return EXIT_OK if passed == total else EXIT_TEST


def cmd_explain(args: argparse.Namespace) -> int:
    topic = args.topic
    if topic == "modules":
        rows = []
        for name in sorted(L.MODULES):
            members = L.MODULES[name]
            rows.append({"module": name, "members": len(members),
                         "names": sorted(members)})
        gaps = [{"module": name, "reason": reason}
                for name, reason in sorted(L.UNIMPLEMENTED_MODULES.items())]
        if args.json:
            print(json.dumps({"modules": rows, "not_implemented": gaps},
                             indent=2))
            return EXIT_OK
        print(f"{len(rows)} standard library modules "
              f"({sum(r['members'] for r in rows)} members)\n")
        for row in rows:
            print(f"  {row['module']:<14} {row['members']:>4} members")
        print(f"\nNot implemented in v0.1 ({len(gaps)} modules):")
        for gap in gaps:
            print(f"  {gap['module']:<18} {gap['reason']}")
        return EXIT_OK

    if topic == "builtins":
        wanted = args.module
        names = sorted(L.MODULES.get(wanted, [])) if wanted else sorted(
            n for n in L.BUILTINS if not L.BUILTINS[n].hidden)
        if args.json:
            print(json.dumps([L.describe_module(wanted)] if wanted else
                             [{"name": n, "params": list(L.BUILTINS[n].params),
                               "effects": list(L.BUILTINS[n].effects),
                               "caps": list(L.BUILTINS[n].caps),
                               "doc": L.BUILTINS[n].doc} for n in names],
                             indent=2))
            return EXIT_OK
        for name in names:
            b = L.BUILTINS.get(name)
            if b is None:
                continue
            signature = ", ".join(b.params)
            notes = []
            if b.effects:
                notes.append("effects: " + ", ".join(b.effects))
            if b.caps:
                notes.append("capabilities: " + ", ".join(b.caps))
            print(f"  {name}({signature})")
            if b.doc:
                print(f"      {b.doc}")
            if notes:
                print(f"      [{' | '.join(notes)}]")
        return EXIT_OK

    if topic == "capabilities":
        from ..runtime.context import KNOWN_CAPABILITIES
        if args.json:
            print(json.dumps(sorted(KNOWN_CAPABILITIES), indent=2))
            return EXIT_OK
        print("Capabilities named by specification section 12.")
        print("A program has none of them unless its module grants them or a")
        print("capability-qualified parameter supplies them.\n")
        for cap in sorted(KNOWN_CAPABILITIES):
            users = sorted(n for n, b in L.BUILTINS.items()
                           if cap in b.caps and not b.hidden)
            print(f"  {cap:<16} " + (", ".join(users) if users else "(no "
                                     "standard library entry point yet)"))
        return EXIT_OK

    if topic == "effects":
        from ..tokens import EFFECT_NAMES
        if args.json:
            print(json.dumps(sorted(EFFECT_NAMES), indent=2))
            return EXIT_OK
        print("Effects tracked by specification section 7, declared as bare")
        print("lines at the start of a function body:\n")
        for effect in sorted(EFFECT_NAMES):
            print(f"  {effect}")
        print("\n`pure` is the absence of effects and never propagates to a")
        print("caller; a function marked pure that performs io is rejected.")
        return EXIT_OK

    # topic == "gaps"
    gaps = {
        "not_implemented_modules": sorted(L.UNIMPLEMENTED_MODULES),
        "notes": [
            "Parallel regions are scheduled correctly but run under CPython's "
            "GIL, so they do not speed up CPU-bound work.",
            "`checkpoint every <duration>` arms interval polling, which is "
            "only meaningful in non-deterministic mode.",
            "Capability handles returned by `capabilities.open` are typed "
            "Any, so the checker cannot verify their permissions statically.",
            "There is no native codegen backend; GIR runs on the reference "
            "interpreter.",
        ],
    }
    print(json.dumps(gaps, indent=2) if args.json else
          "Known gaps in v0.1 (see docs/IMPLEMENTATION.md):\n\n"
          + "\n".join(f"  - {n}" for n in gaps["notes"])
          + "\n\nStandard library modules not yet implemented:\n"
          + "".join(f"  {m:<18} {L.UNIMPLEMENTED_MODULES[m]}\n"
                    for m in gaps["not_implemented_modules"]))
    return EXIT_OK


def cmd_native(args: argparse.Namespace) -> int:
    """Compile to machine code, or say precisely why that is not possible.

    The support analysis runs before any C is written: a backend that discovers
    a gap halfway through emission has already promised the user an executable.
    """
    from ..backend import native as native_backend

    build_dir = args.build_dir or native_backend.DEFAULT_BUILD_DIR
    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status

    failed = False
    for compilation in compilations:
        if compilation.program is None:
            print(f"{compilation.path}: no GIR to compile", file=sys.stderr)
            failed = True
            continue
        if args.emit_c:
            from ..backend import cgen
            support = cgen.unsupported(compilation.program)
            if support.problems:
                print(f"{compilation.path}:")
                for problem in support.problems:
                    print(f"  - {problem.render()}")
                print("  it runs on the reference interpreter: "
                      "`ggc run <file>`")
                failed = True
                continue
            source = cgen.generate_c(compilation.program, compilation.path)
            with open(args.emit_c, "w", encoding="utf-8") as handle:
                handle.write(source)
            print(f"wrote {source.count(chr(10)) + 1} lines of C to {args.emit_c}")
            continue

        result = native_backend.build(compilation.program, compilation.path,
                                      build_dir=build_dir,
                                      opt_level=args.cc_opt,
                                      compiler=args.cc,
                                      keep_c=args.keep_c)
        if args.json:
            print(json.dumps({
                "path": compilation.path, "ok": result.ok,
                "stage": result.stage, "c_lines": result.c_lines,
                "exe": result.exe_path, "compiler": result.compiler,
                "elapsed_ms": round(result.elapsed_ms, 2),
                "problems": [p.render() for p in result.problems],
                "stderr": result.stderr,
            }, indent=2))
            continue
        for line in result.render():
            print(line)
        if not result.ok:
            failed = True
    return EXIT_COMPILE if failed else EXIT_OK


def cmd_difftest(args: argparse.Namespace) -> int:
    """Run both machines on the same program and report where they differ.

    This is the only evidence that a backend is correct.  "It compiled" is not
    evidence, and a fast backend that is occasionally wrong is worse than none.
    """
    from ..backend import differential, native as native_backend

    build_dir = args.build_dir or native_backend.DEFAULT_BUILD_DIR
    corpus = differential.Corpus()
    for path in args.files:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        corpus.comparisons.append(
            differential.compare(source, path=path, build_dir=build_dir))

    if args.json:
        print(json.dumps([{
            "path": c.path, "outcome": c.outcome, "reasons": c.reasons,
            "first_difference": c.first_difference,
            "interpreter_stdout": (c.interpreter.stdout if c.interpreter else ""),
            "native_stdout": c.native_stdout,
            "interpreter_status": (c.interpreter.exit_status
                                   if c.interpreter else None),
            "native_status": c.native_status,
            "interpreter_fault": (c.interpreter.fault_kind
                                  if c.interpreter else ""),
            "native_fault": c.native_fault_kind,
        } for c in corpus.comparisons], indent=2))
        return EXIT_RUNTIME if corpus.diverged else EXIT_OK

    for comparison in corpus.comparisons:
        if comparison.outcome == "agreed":
            print(f"  agreed    {comparison.path}")
        elif comparison.outcome == "refused":
            mark = "refused  " if args.show_refused else "refused  "
            print(f"  {mark} {comparison.path}")
            if args.show_refused:
                for reason in comparison.reasons[:4]:
                    print(f"              {reason}")
        else:
            print(f"  DIVERGED  {comparison.path}")
            for line in comparison.render()[1:]:
                print(f"          {line.strip()}")
    print()
    print(corpus.summary())
    if not corpus.diverged:
        print("no program the backend accepted disagreed with the interpreter")
    return EXIT_RUNTIME if corpus.diverged else EXIT_OK


def cmd_wasm(args: argparse.Namespace) -> int:
    """Emit a WebAssembly module, or report what cannot be expressed.

    The module has never been executed here -- there is no WASM runtime in this
    repository's environment -- so the claim is structural conformance and
    nothing more.  `--disassemble` prints the instructions so they can be read
    against the GIR they came from.
    """
    from ..backend import wasm

    compilations, status = _compile(args.files, args)
    if status != EXIT_OK:
        for compilation in compilations:
            _report(compilation, not args.no_color, args.json)
        return status

    failed = False
    for compilation in compilations:
        if compilation.program is None:
            print(f"{compilation.path}: no GIR to compile", file=sys.stderr)
            failed = True
            continue
        analysis = wasm.analyze(compilation.program)
        if args.analyze or not analysis.ok:
            for line in analysis.render():
                print(line)
            if not analysis.ok:
                print("nothing in this program can be expressed in the WASM "
                      "subset, so no module was written")
                failed = True
            continue
        blob = wasm.generate(compilation.program)
        problems = wasm.validate(blob)
        if args.disassemble:
            print(f"{compilation.path}: {len(blob)} bytes, "
                  f"{len(analysis.compilable)} function(s)")
            for name, body in wasm.disassemble_module(blob).items():
                print(f"  {name}:")
                for line in body:
                    print(f"    {line}")
            if problems:
                print("  structural problems: " + "; ".join(problems))
                failed = True
            continue
        if problems:
            print(f"{compilation.path}: the module failed structural checks:")
            for problem in problems:
                print(f"  - {problem}")
            failed = True
            continue
        out_path = args.output or (compilation.path.rsplit(".", 1)[0] + ".wasm")
        with open(out_path, "wb") as handle:
            handle.write(blob)
        print(f"wrote {len(blob)} bytes to {out_path}")
        print(f"  exports: {', '.join(analysis.compilable)}")
        print("  not executed: no WebAssembly runtime is present here, so this")
        print("  is structural conformance, not verified behaviour")
    return EXIT_COMPILE if failed else EXIT_OK


def cmd_device(args: argparse.Namespace) -> int:
    """Report the accelerator situation honestly.

    There is no GPU in this repository's test environment.  Saying so, with the
    reason, is the useful output; a command that printed nothing when no device
    was present would leave the program unable to tell whether it ran on an
    accelerator or not.
    """
    from ..backend import accelerator

    if args.list:
        for operation, kernel in sorted(accelerator.OFFLOADABLE.items()):
            print(f"  {operation:22} -> kernel `gama_{kernel}`")
        return EXIT_OK

    if args.json:
        print(json.dumps({
            "devices": [{"kind": d.kind, "name": d.name,
                         "available": d.available, "reason": d.reason}
                        for d in accelerator.detect()],
            "kernels": sorted(accelerator.KERNELS),
            "offloadable": accelerator.OFFLOADABLE,
            "kernel_problems": accelerator.check_all(),
            "executed_anywhere": False,
        }, indent=2))
        return EXIT_OK

    for line in accelerator.placement_report():
        print(line)
    print()
    print("kernels available (OpenCL C, structurally checked, never run here):")
    for name in sorted(accelerator.KERNELS):
        problems = accelerator.check_kernel(
            accelerator.Kernel(operation=name,
                               source=accelerator.KERNELS[name],
                               entry=f"gama_{name}"))
        mark = "ok" if not problems else "PROBLEM"
        print(f"  {mark:8} gama_{name}")
        for problem in problems:
            print(f"           {problem}")
    if args.kernels:
        print()
        for line in accelerator.render_kernels(args.kernels):
            print(line)
    return EXIT_OK


COMMANDS = {
    "check": cmd_check,
    "native": cmd_native,
    "difftest": cmd_difftest,
    "wasm": cmd_wasm,
    "device": cmd_device,
    "build": cmd_build,
    "gir": cmd_gir,
    "graph": cmd_graph,
    "memory": cmd_memory,
    "run": cmd_run,
    "test": cmd_test,
    "explain": cmd_explain,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("ggc: interrupted", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    sys.exit(main())
