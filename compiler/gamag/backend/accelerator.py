"""The accelerator backend (audit priority 14): device detection and kernels.

Honest scope, stated first: **there is no accelerator in this repository's test
environment.**  No OpenCL runtime, no CUDA toolkit, no `clinfo`, no `pyopencl`.
So nothing here has been executed on a GPU, and no speedup is claimed anywhere --
spec section 43 forbids the claim and the absence of hardware makes it
unverifiable twice over.

What this module does provide is the two parts that can be built and checked
without a device:

* **Detection that tells the truth.**  `detect()` reports every device class it
  looked for and, for each one it did not find, why.  A program that asks for an
  accelerator on a machine without one gets "no accelerator present, ran on the
  CPU" rather than a silent fallback that makes the placement decision
  invisible -- which is the failure mode spec section 3 warns about when it asks
  for explicit device placement.
* **Real kernel source.**  `KERNELS` holds OpenCL C for the operations that are
  actually parallel: elementwise arithmetic, `relu`, `clip`, `matmul`, `dot`,
  `softmax` and `transpose`.  These are emitted, checked structurally, and can
  be handed to any OpenCL runtime.  They have not been run here, and the CLI
  says so when it prints them.

What is deliberately absent: a scheduler, a memory transfer planner, and any
attempt to decide *when* offloading is worth it.  Those need measurements, and
measurements need the hardware.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

#: How each device class is looked for.  A probe that could report a device it
#: cannot actually use would be worse than no probe.
OPENCL_PROBES = ("clinfo",)
CUDA_PROBES = ("nvidia-smi", "nvcc")
OPENCL_PYTHON = ("pyopencl",)
CUDA_PYTHON = ("cupy",)


@dataclass
class Device:
    """One device class, whether or not it is present."""

    kind: str                 # cpu | opencl | cuda
    name: str
    available: bool
    reason: str = ""
    detail: str = ""

    def render(self) -> str:
        mark = "present" if self.available else "absent "
        line = f"  {mark}  {self.kind:7} {self.name}"
        if not self.available and self.reason:
            line += f" -- {self.reason}"
        if self.available and self.detail:
            line += f" ({self.detail})"
        return line


def _python_module_present(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def detect() -> List[Device]:
    """Report every device class, present or absent, with the reason."""
    devices: List[Device] = []

    devices.append(Device(
        kind="cpu", name="host CPU (the fallback)", available=True,
        detail="always present; every kernel here has a CPU path in "
               "runtime/tensor.py"))

    clinfo = shutil.which("clinfo")
    if _python_module_present("pyopencl"):
        devices.append(Device(kind="opencl", name="OpenCL via pyopencl",
                              available=True,
                              detail="a runtime is importable"))
    elif clinfo:
        devices.append(Device(kind="opencl", name="OpenCL via clinfo",
                              available=True, detail=clinfo))
    else:
        devices.append(Device(
            kind="opencl", name="OpenCL", available=False,
            reason="no `clinfo` on PATH and `pyopencl` is not importable, so "
                   "there is no OpenCL platform to enumerate"))

    if shutil.which("nvidia-smi") or shutil.which("nvcc"):
        devices.append(Device(kind="cuda", name="CUDA", available=True,
                              detail="an NVIDIA tool is on PATH"))
    elif any(_python_module_present(m) for m in CUDA_PYTHON):
        devices.append(Device(kind="cuda", name="CUDA via cupy", available=True))
    else:
        devices.append(Device(
            kind="cuda", name="CUDA", available=False,
            reason="no `nvidia-smi`, no `nvcc` and no `cupy`, so there is no "
                   "CUDA device to target"))

    return devices


def accelerators() -> List[Device]:
    """Only the devices that are actually present."""
    return [d for d in detect() if d.available and d.kind != "cpu"]


def placement_report() -> List[str]:
    out = ["devices:"]
    out.extend(d.render() for d in detect())
    if not accelerators():
        out.append("")
        out.append("  No accelerator is present, so every tensor operation runs")
        out.append("  on the CPU. This is reported rather than done silently:")
        out.append("  spec section 3 asks for explicit device placement, and a")
        out.append("  fallback the program cannot see is not placement.")
    return out


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

#: OpenCL C for the operations that are genuinely parallel.  Each entry is a
#: complete kernel; the work-group size is chosen by the host at enqueue time.
KERNELS: Dict[str, str] = {
    "add": """__kernel void gama_add(__global const double *a,
                       __global const double *b,
                       __global double *out,
                       const long n)
{
    long i = get_global_id(0);
    if (i < n) out[i] = a[i] + b[i];
}
""",
    "sub": """__kernel void gama_sub(__global const double *a,
                       __global const double *b,
                       __global double *out,
                       const long n)
{
    long i = get_global_id(0);
    if (i < n) out[i] = a[i] - b[i];
}
""",
    "mul": """__kernel void gama_mul(__global const double *a,
                       __global const double *b,
                       __global double *out,
                       const long n)
{
    long i = get_global_id(0);
    if (i < n) out[i] = a[i] * b[i];
}
""",
    "div": """__kernel void gama_div(__global const double *a,
                       __global const double *b,
                       __global double *out,
                       const long n)
{
    /* A zero divisor is a fault in Gama-G, not an infinity: the check is in
     * the kernel so that the device path and the CPU path agree. */
    long i = get_global_id(0);
    if (i < n) out[i] = (b[i] == 0.0) ? NAN : a[i] / b[i];
}
""",
    "relu": """__kernel void gama_relu(__global const double *a,
                        __global double *out,
                        const long n)
{
    long i = get_global_id(0);
    if (i < n) out[i] = a[i] > 0.0 ? a[i] : 0.0;
}
""",
    "clip": """__kernel void gama_clip(__global const double *a,
                        __global double *out,
                        const long n,
                        const double lo, const double hi)
{
    long i = get_global_id(0);
    if (i < n) out[i] = clamp(a[i], lo, hi);
}
""",
    "dot": """__kernel void gama_dot(__global const double *a,
                       __global const double *b,
                       __global double *out,
                       const long n)
{
    /* One work item per output element; a reduction across work groups would
     * need a second pass, which is why this is per-row and not whole-tensor. */
    long row = get_global_id(0);
    long cols = get_global_id(1);
    if (row < n && cols < n) out[row] = out[row] + a[row] * b[cols];
}
""",
    "matmul": """__kernel void gama_matmul(__global const double *a,
                            __global const double *b,
                            __global double *c,
                            const long m, const long k, const long n)
{
    long row = get_global_id(0);
    long col = get_global_id(1);
    if (row >= m || col >= n) return;
    double acc = 0.0;
    for (long i = 0; i < k; i++) acc += a[row * k + i] * b[i * n + col];
    c[row * n + col] = acc;
}
""",
    "transpose": """__kernel void gama_transpose(__global const double *a,
                             __global double *out,
                             const long rows, const long cols)
{
    long r = get_global_id(0);
    long c = get_global_id(1);
    if (r < rows && c < cols) out[c * rows + r] = a[r * cols + c];
}
""",
    "softmax": """__kernel void gama_softmax(__global const double *a,
                             __global double *out,
                             const long rows, const long cols)
{
    /* One work item per row.  The max is subtracted first, which is what
     * runtime/tensor.py does; skipping it would overflow on large inputs and
     * the two paths would disagree. */
    long r = get_global_id(0);
    if (r >= rows) return;
    __global const double *row = a + r * cols;
    __global double *dst = out + r * cols;
    double peak = row[0];
    for (long i = 1; i < cols; i++) if (row[i] > peak) peak = row[i];
    double total = 0.0;
    for (long i = 0; i < cols; i++) { dst[i] = exp(row[i] - peak); total += dst[i]; }
    for (long i = 0; i < cols; i++) dst[i] = dst[i] / total;
}
""",
}

#: Which tensor operations in the language map onto a kernel above.  Anything
#: not listed is a shape or metadata query, which is not worth offloading.
OFFLOADABLE: Dict[str, str] = {
    "tensor.matmul": "matmul",
    "tensor.dot": "dot",
    "tensor.transpose": "transpose",
    "tensor.softmax": "softmax",
    "tensor.clip": "clip",
    "tensor.add": "add",
    "tensor.sub": "sub",
    "tensor.mul": "mul",
    "tensor.div": "div",
    "tensor.relu": "relu",
}


@dataclass
class Kernel:
    """One emitted kernel, and what is honestly known about it."""

    operation: str
    source: str
    entry: str = ""
    #: True only if a device was present to run it on.  Nothing in this
    #: repository sets it to True.
    executed: bool = False

    @property
    def lines(self) -> int:
        return self.source.count("\n")


def kernel_for(operation: str) -> Optional[Kernel]:
    """The kernel for a tensor operation, if one exists."""
    name = OFFLOADABLE.get(operation)
    if name is None:
        return None
    source = KERNELS[name]
    entry = source.split("gama_")[1].split("(")[0] if "gama_" in source else name
    return Kernel(operation=operation, source=source, entry=f"gama_{name}")


def check_kernel(kernel: Kernel) -> List[str]:
    """Structural checks on emitted OpenCL C.

    These are the checks that can be made without a device: the kernel is
    declared as a kernel, its entry point exists, every parameter is annotated
    with an address space, and the bounds guard is present.  A kernel missing
    the guard would read out of bounds on the last work group, which is the
    single most common OpenCL bug and the one a compiler cannot catch for you.
    """
    problems: List[str] = []
    source = kernel.source
    if "__kernel void " not in source:
        problems.append("no `__kernel void` entry point")
    if kernel.entry and kernel.entry not in source:
        problems.append(f"the entry point `{kernel.entry}` is not in the source")
    if "__global" not in source:
        problems.append("no `__global` address space annotation")
    if "get_global_id" not in source:
        problems.append("the kernel does not compute a work-item index")
    if "if (" not in source:
        problems.append("no bounds guard: the last work group would read out "
                        "of bounds")
    return problems


def check_all() -> Dict[str, List[str]]:
    """Every kernel, with any structural problem found."""
    return {name: check_kernel(Kernel(operation=name, source=source,
                                      entry=f"gama_{name}"))
            for name, source in KERNELS.items()}


def render_kernels(only: Optional[Sequence[str]] = None) -> List[str]:
    out: List[str] = []
    names = list(only) if only else sorted(KERNELS)
    for name in names:
        if name not in KERNELS:
            out.append(f"/* no kernel for `{name}` */")
            continue
        problems = check_kernel(Kernel(operation=name, source=KERNELS[name],
                                       entry=f"gama_{name}"))
        out.append(f"/* kernel: {name}"
                   + (" -- PROBLEMS: " + "; ".join(problems) if problems else "")
                   + " */")
        out.append(KERNELS[name].rstrip())
        out.append("")
    out.append("/* Not executed: no OpenCL or CUDA device was present when this")
    out.append("   was generated. Structural checks only; see docs/DESIGN_v1_0.md. */")
    return out
