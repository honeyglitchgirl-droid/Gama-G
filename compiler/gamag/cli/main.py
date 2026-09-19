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
from ..driver import Compilation, compile_file, execute, find_entry
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
            grants=set(args.grant) | set(
                compilation.checker.grants if compilation.checker else ()),
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
                handle.write(ctx.audit.export())
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
            ctx = Context(deterministic=True, grants=set(args.grant) | set(
                compilation.checker.grants if compilation.checker else ()))
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


COMMANDS = {
    "check": cmd_check,
    "build": cmd_build,
    "gir": cmd_gir,
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
