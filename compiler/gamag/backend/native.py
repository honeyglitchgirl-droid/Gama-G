"""Building and running native code (audit priority 3).

The backend generates C and hands it to the system compiler.  That is a real
native backend -- the artifact is machine code with no interpreter in it -- and
it is the honest one to build first, because a C toolchain is present on every
platform this repository can be tested on and the result can be *executed* and
compared against the reference interpreter (priority 7).

Two rules shape this module:

1. **Refuse before emitting.**  `cgen.unsupported()` runs first.  If a program
   contains anything this backend cannot compile, no C is written and no
   compiler is invoked; the caller gets the list of reasons.  A backend that
   discovers a gap halfway through has already promised the user an executable.
2. **Report what the compiler said.**  Warnings are not swallowed.  The C
   compiler's own diagnostics are part of the build result, because a warning
   here is usually a bug in the generator rather than in the user's program.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..gir.ir import GProgram
from . import cgen

#: Where build artifacts go when the caller does not choose.  Git-ignored.
DEFAULT_BUILD_DIR = ".ggbuild"


def find_c_compiler(preferred: Optional[str] = None) -> Optional[str]:
    """The C compiler to use, or None if there is not one.

    `$CC` wins, because that is the convention every build system follows and
    overriding it would be surprising.  Then the compilers that are actually
    likely to be installed.
    """
    env = os.environ.get("CC")
    candidates = [preferred] if preferred else []
    if env:
        candidates.append(env)
    candidates += ["cc", "gcc", "clang"]
    for name in candidates:
        if not name:
            continue
        path = shutil.which(name)
        if path:
            return path
    return None


@dataclass
class NativeBuild:
    """Everything a caller needs to know about an attempt to build natively."""

    ok: bool = False
    #: Why not, in the program's own terms.  Empty when `ok`.
    problems: List[cgen.Problem] = field(default_factory=list)
    #: Set when the backend agreed to compile but the C toolchain failed.
    stage: str = ""
    c_source: str = ""
    c_path: str = ""
    exe_path: str = ""
    compiler: str = ""
    command: List[str] = field(default_factory=list)
    stderr: str = ""
    returncode: int = 0
    elapsed_ms: float = 0.0
    #: The generated program's size, because a code generator that emits ten
    #: times the source it was given is worth knowing about.
    c_lines: int = 0

    @property
    def refused(self) -> bool:
        """True when the backend declined, as opposed to failing."""
        return bool(self.problems)

    def render(self) -> List[str]:
        out: List[str] = []
        if self.problems:
            out.append("the native backend cannot compile this program:")
            for problem in self.problems:
                out.append(f"  - {problem.render()}")
            out.append("it runs on the reference interpreter: `ggc run <file>`")
            return out
        if not self.ok:
            out.append(f"native build failed during {self.stage or 'the build'}")
            if self.command:
                out.append("  command: " + " ".join(self.command))
            for line in self.stderr.strip().splitlines()[:40]:
                out.append("  " + line)
            return out
        out.append(f"native build ok: {self.exe_path}")
        out.append(f"  compiler: {self.compiler}")
        out.append(f"  generated {self.c_lines} lines of C in "
                   f"{self.elapsed_ms:.1f} ms")
        if self.stderr.strip():
            out.append("  compiler warnings:")
            for line in self.stderr.strip().splitlines()[:20]:
                out.append("    " + line)
        return out


@dataclass
class NativeRun:
    """The result of executing a native binary."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    elapsed_ms: float = 0.0
    #: Peak resident set size in KiB, when the platform reports it.
    max_rss_kib: int = 0


def _safe_name(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".gg"):
        base = base[:-3]
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base) \
        or "program"


def generate(program: GProgram, source_path: str = "<program>",
             build_dir: str = DEFAULT_BUILD_DIR,
             entry: Optional[str] = None) -> NativeBuild:
    """Run the support analysis and emit C.  Does not invoke a compiler."""
    support = cgen.unsupported(program)
    if support.problems:
        return NativeBuild(ok=False, problems=support.problems, stage="analysis")

    os.makedirs(build_dir, exist_ok=True)
    c_path = os.path.join(build_dir, _safe_name(source_path) + ".c")
    source = cgen.generate_c(program, source_path, entry)
    with open(c_path, "w", encoding="utf-8") as handle:
        handle.write(source)
    return NativeBuild(ok=True, stage="codegen", c_source=source, c_path=c_path,
                       c_lines=source.count("\n") + 1)


def build(program: GProgram, source_path: str = "<program>",
          build_dir: str = DEFAULT_BUILD_DIR, opt_level: int = 2,
          compiler: Optional[str] = None, keep_c: bool = True,
          entry: Optional[str] = None) -> NativeBuild:
    """Generate, compile and link.  Refuses before touching the toolchain."""
    result = generate(program, source_path, build_dir, entry)
    if not result.ok:
        return result

    cc = find_c_compiler(compiler)
    if cc is None:
        result.ok = False
        result.stage = "toolchain"
        result.stderr = ("no C compiler was found. Set $CC, or install gcc or "
                         "clang. The native backend generates C and needs one; "
                         "the program still runs on the reference interpreter "
                         "with `ggc run`.")
        return result

    exe_path = os.path.join(build_dir, _safe_name(source_path))
    runtime_dir = cgen.RUNTIME_DIR
    sources = [os.path.join(runtime_dir, name) for name in cgen.RUNTIME_SOURCES]

    command = [cc, f"-O{max(0, min(3, opt_level))}", "-std=c11",
               "-Wall", "-Wextra", "-Wno-unused-parameter",
               f"-I{runtime_dir}", result.c_path, *sources,
               "-o", exe_path, "-lm"]
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=300)
    except subprocess.TimeoutExpired:
        result.ok = False
        result.stage = "compile"
        result.stderr = "the C compiler did not finish within 300 seconds"
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result
    except OSError as exc:
        result.ok = False
        result.stage = "compile"
        result.stderr = f"could not run {cc}: {exc}"
        return result

    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    result.compiler = cc
    result.command = command
    result.stderr = proc.stderr
    result.returncode = proc.returncode
    result.exe_path = exe_path if proc.returncode == 0 else ""
    result.ok = proc.returncode == 0
    result.stage = "compile" if not result.ok else ""
    if not keep_c and os.path.isfile(result.c_path):
        os.remove(result.c_path)
    return result


def run(exe_path: str, args: Sequence[str] = (),
        timeout: float = 30.0, stdin_text: str = "") -> NativeRun:
    """Execute a native binary and capture what it did."""
    started = time.perf_counter()
    try:
        proc = subprocess.run([exe_path, *args], capture_output=True,
                              text=True, timeout=timeout, input=stdin_text)
    except subprocess.TimeoutExpired as exc:
        return NativeRun(returncode=-1,
                         stdout=(exc.stdout or b"").decode("utf-8", "replace")
                         if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                         stderr=f"timed out after {timeout}s",
                         timed_out=True,
                         elapsed_ms=(time.perf_counter() - started) * 1000.0)
    except OSError as exc:
        return NativeRun(returncode=-1, stderr=str(exc),
                         elapsed_ms=(time.perf_counter() - started) * 1000.0)
    elapsed = (time.perf_counter() - started) * 1000.0
    rss = 0
    usage = getattr(proc, "rusage", None)
    if usage is not None:
        rss = int(getattr(usage, "ru_maxrss", 0) or 0)
    return NativeRun(returncode=proc.returncode, stdout=proc.stdout,
                     stderr=proc.stderr, elapsed_ms=elapsed, max_rss_kib=rss)


def runtime_sources() -> List[str]:
    """The C runtime files a build needs, as absolute paths."""
    return [os.path.join(cgen.RUNTIME_DIR, n) for n in cgen.RUNTIME_SOURCES]
