"""Source positions and diagnostics for the Gama-G toolchain.

Implements the error-reporting substrate used by every compiler phase
(spec section 22, steps 1-7).  Diagnostics are structured rather than
bare strings, per spec section 19: "Errors contain structured context
rather than only strings."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


class Phase(Enum):
    """Compilation phase that produced a diagnostic (spec section 22)."""

    LEX = "lexing"
    PARSE = "parsing"
    RESOLVE = "name-resolution"
    TYPE = "type-checking"
    EFFECT = "effect-checking"
    CAPABILITY = "capability-checking"
    OWNERSHIP = "ownership-analysis"
    GIR = "gir-generation"
    OPTIMIZE = "optimization"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class SourcePos:
    """A position in a source file.

    ``offset`` is the absolute character offset; ``line``/``col`` are
    1-based, which is what editors and LSP clients expect.
    """

    file: str = "<unknown>"
    line: int = 1
    col: int = 1
    offset: int = 0

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class Diagnostic:
    severity: Severity
    phase: Phase
    message: str
    pos: Optional[SourcePos] = None
    end: Optional[SourcePos] = None
    code: Optional[str] = None
    help_text: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def render(self, source: Optional[str] = None, color: bool = False) -> str:
        """Render a human-readable, caret-annotated diagnostic."""
        head = f"{self.severity.value}[{self.phase.value}]"
        if self.code:
            head += f" {self.code}"
        lines = [f"{head}: {self.message}"]
        if self.pos is not None:
            lines.append(f"  --> {self.pos}")
            if source is not None:
                src_lines = source.splitlines()
                if 1 <= self.pos.line <= len(src_lines):
                    text = src_lines[self.pos.line - 1]
                    gutter = str(self.pos.line)
                    pad = " " * len(gutter)
                    lines.append(f"  {pad} |")
                    lines.append(f"  {gutter} | {text}")
                    width = 1
                    if self.end is not None and self.end.line == self.pos.line:
                        width = max(1, self.end.col - self.pos.col)
                    caret_pad = " " * (self.pos.col - 1)
                    lines.append(f"  {pad} | {caret_pad}{'^' * width}")
        for note in self.notes:
            lines.append(f"  = note: {note}")
        if self.help_text:
            lines.append(f"  = help: {self.help_text}")
        out = "\n".join(lines)
        if color:
            paint = {
                Severity.ERROR: "\033[1;31m",
                Severity.WARNING: "\033[1;33m",
                Severity.NOTE: "\033[1;36m",
            }[self.severity]
            out = f"{paint}{out}\033[0m"
        return out


class GamaError(Exception):
    """Base class for all Gama-G toolchain errors."""

    def __init__(self, diagnostic: Diagnostic):
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


class LexError(GamaError):
    pass


class ParseError(GamaError):
    pass


class SemanticError(GamaError):
    pass


class DiagnosticBag:
    """Accumulates diagnostics so a phase can report many problems at once.

    Recovery matters for tooling: an editor wants every error in a file,
    not just the first.  Phases that cannot continue raise instead.
    """

    def __init__(self) -> None:
        self.diagnostics: List[Diagnostic] = []

    def add(self, diag: Diagnostic) -> None:
        self.diagnostics.append(diag)

    def error(
        self,
        message: str,
        pos: Optional[SourcePos] = None,
        *,
        phase: Phase = Phase.TYPE,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
        end: Optional[SourcePos] = None,
    ) -> None:
        self.add(
            Diagnostic(
                Severity.ERROR, phase, message, pos=pos, end=end,
                code=code, help_text=help_text,
            )
        )

    def warning(
        self,
        message: str,
        pos: Optional[SourcePos] = None,
        *,
        phase: Phase = Phase.TYPE,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
    ) -> None:
        self.add(
            Diagnostic(
                Severity.WARNING, phase, message, pos=pos,
                code=code, help_text=help_text,
            )
        )

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render_all(self, source: Optional[str] = None, color: bool = False) -> str:
        return "\n\n".join(d.render(source, color) for d in self.diagnostics)

    def raise_if_errors(self) -> None:
        if self.errors:
            raise SemanticError(self.errors[0])


# Runtime-level faults (spec section 19: "Unrecoverable defects enter a
# fault domain") are kept distinct from compile-time diagnostics.
class GamaRuntimeFault(Exception):
    """An unrecoverable defect that enters the fault domain."""

    def __init__(
        self,
        kind: str,
        message: str,
        pos: Optional[SourcePos] = None,
        context: Optional[dict] = None,
    ):
        self.kind = kind
        self.message = message
        self.pos = pos
        self.context = context or {}
        super().__init__(f"{kind}: {message}")

    def structured(self) -> dict:
        """Structured context, per spec section 19."""
        return {
            "kind": self.kind,
            "message": self.message,
            "position": str(self.pos) if self.pos else None,
            "context": dict(self.context),
        }


class ContractViolation(GamaRuntimeFault):
    """A `requires`/`ensures`/`require` contract failed at runtime (spec 27)."""

    def __init__(self, message: str, pos: Optional[SourcePos] = None, **ctx):
        super().__init__("ContractViolation", message, pos, ctx)


class CapabilityViolation(GamaRuntimeFault):
    """An operation ran without a valid capability (spec section 12)."""

    def __init__(self, message: str, pos: Optional[SourcePos] = None, **ctx):
        super().__init__("CapabilityViolation", message, pos, ctx)


class TypeFault(GamaRuntimeFault):
    def __init__(self, message: str, pos: Optional[SourcePos] = None, **ctx):
        super().__init__("TypeFault", message, pos, ctx)


class SecretLeak(GamaRuntimeFault):
    """A `secret` value reached a sink it may not (spec section 8)."""

    def __init__(self, message: str, pos: Optional[SourcePos] = None, **ctx):
        super().__init__("SecretLeak", message, pos, ctx)
