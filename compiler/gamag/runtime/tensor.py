"""Tensor subsystem for Gama-G (spec section 14).

A dependency-free n-dimensional array with static shape metadata, plus a
compact reverse-mode automatic-differentiation tape so that the training
graphs required by spec section 14 ("automatic differentiation", "training
graphs", "deterministic inference") are real rather than notional.

Performance note, in the spirit of spec sections 24 and 43: this is the
*reference* implementation.  It is correct and portable, not fast.  A
production Gama-G would lower tensor operations through GIR to a vectorised
native or accelerator backend; no universal speed claim is made here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

from ..diagnostics import GamaRuntimeFault, TypeFault
from .values import GamaValue, display

DTYPES = ("F16", "F32", "F64", "I64", "U8", "Bool")
_FLOAT_DTYPES = ("F16", "F32", "F64")


class TensorError(GamaRuntimeFault):
    def __init__(self, message: str, **ctx):
        super().__init__("TensorError", message, None, ctx)


def _prod(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _coerce(value: Any, dtype: str) -> Any:
    if dtype in _FLOAT_DTYPES:
        return float(value)
    if dtype == "Bool":
        return bool(value)
    return int(value)


def _infer_shape(nested: Any) -> Tuple[int, ...]:
    """Infer a tensor shape from a nested list, rejecting ragged input."""
    if not isinstance(nested, list):
        return ()
    if not nested:
        return (0,)
    subs = [_infer_shape(x) for x in nested]
    first = subs[0]
    if any(s != first for s in subs):
        raise TensorError(
            "ragged nested list cannot form a tensor",
            shapes=[list(s) for s in subs],
            hint="every row of a tensor must have the same length",
        )
    return (len(nested),) + first


def _flatten_data(nested: Any, out: List[Any]) -> None:
    if isinstance(nested, list):
        for item in nested:
            _flatten_data(item, out)
    else:
        out.append(nested)


class GTensor(GamaValue):
    """A dense, row-major n-dimensional array."""

    __slots__ = ("dtype", "shape", "data", "_grad", "_tape")

    def __init__(self, dtype: str, shape: Sequence[int],
                 data: Optional[Sequence[Any]] = None):
        if dtype not in DTYPES:
            raise TensorError(f"unsupported tensor dtype `{dtype}`",
                              supported=list(DTYPES))
        shape = tuple(int(d) for d in shape)
        if any(d < 0 for d in shape):
            raise TensorError("tensor dimensions must be non-negative",
                              shape=list(shape))
        n = _prod(shape)
        self.dtype = dtype
        self.shape = shape
        if data is None:
            self.data: List[Any] = [_coerce(0, dtype)] * n
        else:
            self.data = [_coerce(v, dtype) for v in data]
            if len(self.data) != n:
                raise TensorError(
                    f"cannot build Tensor<{dtype},{list(shape)}>: "
                    f"got {len(self.data)} elements, shape requires {n}",
                    shape=list(shape), got=len(self.data),
                )
        self._grad: Optional["GTensor"] = None
        self._tape: Optional[Tape] = None

    # -- construction --------------------------------------------------
    @staticmethod
    def from_nested(nested: Any, dtype: str = "F64") -> "GTensor":
        if not isinstance(nested, list):
            return GTensor(dtype, (), [nested])
        shape = _infer_shape(nested)
        flat: List[Any] = []
        _flatten_data(nested, flat)
        return GTensor(dtype, shape, flat)

    @staticmethod
    def zeros(shape: Sequence[int], dtype: str = "F64") -> "GTensor":
        return GTensor(dtype, shape)

    @staticmethod
    def ones(shape: Sequence[int], dtype: str = "F64") -> "GTensor":
        return GTensor(dtype, shape, [1] * _prod(shape))

    @staticmethod
    def full(shape: Sequence[int], value: Any, dtype: str = "F64") -> "GTensor":
        return GTensor(dtype, shape, [value] * _prod(shape))

    @staticmethod
    def arange(n: int, dtype: str = "F64") -> "GTensor":
        return GTensor(dtype, (n,), list(range(n)))

    @staticmethod
    def scalar(value: Any, dtype: str = "F64") -> "GTensor":
        return GTensor(dtype, (), [value])

    # -- basic properties ---------------------------------------------
    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return len(self.data)

    def flat(self) -> List[Any]:
        return list(self.data)

    def copy(self) -> "GTensor":
        t = GTensor(self.dtype, self.shape, list(self.data))
        return t

    def type_name(self) -> str:
        return f"Tensor<{self.dtype},[{','.join(str(d) for d in self.shape)}]>"

    def __repr__(self) -> str:
        return f"{self.type_name()}({display(self.to_nested())})"

    # -- indexing ------------------------------------------------------
    def _strides(self) -> Tuple[int, ...]:
        strides = [1] * self.rank
        for i in range(self.rank - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        return tuple(strides)

    def at(self, index: Sequence[int]) -> Any:
        index = tuple(int(i) for i in index)
        if len(index) != self.rank:
            raise TensorError(
                f"expected {self.rank} indices for {self.type_name()}, "
                f"got {len(index)}")
        strides = self._strides()
        off = 0
        for i, (idx, dim, st) in enumerate(zip(index, self.shape, strides)):
            if idx < 0:
                idx += dim
            if not 0 <= idx < dim:
                raise TensorError(
                    f"index {index[i]} out of range for dimension {i} "
                    f"(size {dim})", index=list(index), shape=list(self.shape))
            off += idx * st
        return self.data[off]

    def set_at(self, index: Sequence[int], value: Any) -> None:
        index = tuple(int(i) for i in index)
        strides = self._strides()
        off = 0
        for i, (idx, dim, st) in enumerate(zip(index, self.shape, strides)):
            if idx < 0:
                idx += dim
            if not 0 <= idx < dim:
                raise TensorError(f"index {index[i]} out of range for dim {i}")
            off += idx * st
        self.data[off] = _coerce(value, self.dtype)

    def index1(self, i: int) -> Any:
        """Single-axis indexing: returns a sub-tensor or a scalar."""
        if self.rank == 1:
            if i < 0:
                i += self.shape[0]
            if not 0 <= i < self.shape[0]:
                raise TensorError(f"index {i} out of range for {self.type_name()}")
            return self.data[i]
        if i < 0:
            i += self.shape[0]
        if not 0 <= i < self.shape[0]:
            raise TensorError(f"index {i} out of range for {self.type_name()}")
        stride = self.shape[1:] and _prod(self.shape[1:]) or 1
        start = i * stride
        return GTensor(self.dtype, self.shape[1:],
                       self.data[start:start + stride])

    def to_nested(self) -> Any:
        if self.rank == 0:
            return self.data[0] if self.data else 0
        return self._nest(0, self.shape, self._strides())[0]

    def _nest(self, offset: int, shape: Tuple[int, ...],
              strides: Tuple[int, ...]) -> Tuple[Any, int]:
        if len(shape) == 1:
            chunk = self.data[offset:offset + shape[0]]
            return list(chunk), offset + shape[0]
        out = []
        pos = offset
        for _ in range(shape[0]):
            sub, pos = self._nest(pos, shape[1:], strides[1:])
            out.append(sub)
        return out, pos

    # -- shape ops -----------------------------------------------------
    def reshape(self, shape: Sequence[int]) -> "GTensor":
        shape = tuple(int(d) for d in shape)
        if -1 in shape:
            known = _prod([d for d in shape if d != -1]) or 1
            if self.size % known != 0:
                raise TensorError(
                    f"cannot reshape {self.type_name()} to {list(shape)}: "
                    f"{self.size} elements are not divisible by {known}")
            shape = tuple(self.size // known if d == -1 else d for d in shape)
        if _prod(shape) != self.size:
            raise TensorError(
                f"cannot reshape {self.type_name()} ({self.size} elements) "
                f"to shape {list(shape)} ({_prod(shape)} elements)",
                from_shape=list(self.shape), to_shape=list(shape))
        return GTensor(self.dtype, shape, list(self.data))

    def transpose(self, perm: Optional[Sequence[int]] = None) -> "GTensor":
        if self.rank == 2 and perm is None:
            perm = (1, 0)
        if perm is None:
            perm = tuple(reversed(range(self.rank)))
        perm = tuple(int(p) for p in perm)
        if sorted(perm) != list(range(self.rank)):
            raise TensorError(f"invalid transpose permutation {list(perm)} "
                              f"for rank {self.rank}")
        new_shape = tuple(self.shape[p] for p in perm)
        out = GTensor(self.dtype, new_shape)
        idx = [0] * self.rank
        for off in range(self.size):
            # decode offset in the source layout
            rem = off
            src_strides = self._strides()
            for d in range(self.rank):
                idx[d] = rem // src_strides[d]
                rem %= src_strides[d]
            new_idx = [0] * self.rank
            for dst, srcdim in enumerate(perm):
                new_idx[dst] = idx[srcdim]
            out.set_at(new_idx, self.data[off])
        return out

    # -- elementwise ---------------------------------------------------
    def _binary(self, other: Any, op: Callable[[Any, Any], Any],
                name: str, dtype: Optional[str] = None) -> "GTensor":
        out_dtype = dtype or self.dtype
        if isinstance(other, GTensor):
            if other.shape == ():
                scalar = other.data[0] if other.data else 0
                return GTensor(out_dtype, self.shape,
                               [op(v, scalar) for v in self.data])
            if self.shape == ():
                s = self.data[0] if self.data else 0
                return GTensor(out_dtype, other.shape,
                               [op(s, v) for v in other.data])
            if other.shape != self.shape:
                raise TensorError(
                    f"shape mismatch in `{name}`: {self.type_name()} vs "
                    f"{other.type_name()}",
                    left=list(self.shape), right=list(other.shape),
                    hint="Gama-G v0.1 broadcasts scalars only; align shapes "
                         "explicitly with `tensor.reshape` or `tensor.broadcast`")
            return GTensor(out_dtype, self.shape,
                           [op(a, b) for a, b in zip(self.data, other.data)])
        if isinstance(other, (int, float, bool)):
            return GTensor(out_dtype, self.shape,
                           [op(v, other) for v in self.data])
        raise TensorError(f"`{name}` expects a Tensor or scalar operand, "
                          f"got {type(other).__name__}")

    def add(self, other): return self._binary(other, lambda a, b: a + b, "add")
    def sub(self, other): return self._binary(other, lambda a, b: a - b, "sub")
    def mul(self, other): return self._binary(other, lambda a, b: a * b, "mul")

    def div(self, other):
        def safe(a, b):
            if b == 0:
                raise TensorError("division by zero in tensor operation")
            return a / b
        return self._binary(other, safe, "div", dtype="F64")

    def neg(self) -> "GTensor":
        return GTensor(self.dtype, self.shape, [-v for v in self.data])

    def relu(self) -> "GTensor":
        return GTensor(self.dtype, self.shape,
                       [v if v > 0 else _coerce(0, self.dtype) for v in self.data])

    def sigmoid(self) -> "GTensor":
        return GTensor("F64", self.shape,
                       [1.0 / (1.0 + math.exp(-v)) for v in self.data])

    def tanh(self) -> "GTensor":
        return GTensor("F64", self.shape, [math.tanh(v) for v in self.data])

    def exp(self) -> "GTensor":
        return GTensor("F64", self.shape, [math.exp(v) for v in self.data])

    def softmax(self, axis: int = -1) -> "GTensor":
        if self.rank == 0:
            raise TensorError("softmax is undefined for a rank-0 tensor")
        ax = axis if axis >= 0 else axis + self.rank
        if not 0 <= ax < self.rank:
            raise TensorError(f"axis {axis} out of range for {self.type_name()}")
        if ax == self.rank - 1:
            return self._softmax_last_axis()
        perm = tuple(i for i in range(self.rank) if i != ax) + (ax,)
        inverse = [0] * self.rank
        for dst, src in enumerate(perm):
            inverse[src] = dst
        return self.transpose(perm)._softmax_last_axis().transpose(inverse)

    def _softmax_last_axis(self) -> "GTensor":
        if self.rank <= 1:
            m = max(self.data) if self.data else 0.0
            exps = [math.exp(v - m) for v in self.data]
            s = sum(exps) or 1.0
            return GTensor("F64", self.shape, [e / s for e in exps])
        stride = _prod(self.shape[1:])
        out: List[float] = []
        for start in range(0, self.size, stride):
            row = self.data[start:start + stride]
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps) or 1.0
            out.extend(e / s for e in exps)
        return GTensor("F64", self.shape, out)

    # -- reductions ----------------------------------------------------
    def sum(self, axis: Optional[int] = None) -> Any:
        if axis is None:
            return _scalar_or_tensor(sum(self.data), self.dtype)
        return self._reduce_axis(axis, sum)

    def mean(self, axis: Optional[int] = None) -> Any:
        if axis is None:
            if not self.data:
                raise TensorError("mean of an empty tensor is undefined")
            return GTensor("F64", (), [sum(self.data) / len(self.data)])
        return self._reduce_axis(axis, lambda xs: sum(xs) / len(xs), "F64")

    def max(self, axis: Optional[int] = None) -> Any:
        if axis is None:
            if not self.data:
                raise TensorError("max of an empty tensor is undefined")
            return _scalar_or_tensor(max(self.data), self.dtype)
        return self._reduce_axis(axis, max)

    def min(self, axis: Optional[int] = None) -> Any:
        if axis is None:
            if not self.data:
                raise TensorError("min of an empty tensor is undefined")
            return _scalar_or_tensor(min(self.data), self.dtype)
        return self._reduce_axis(axis, min)

    def argmax(self, axis: Optional[int] = None) -> Any:
        if axis is None:
            if not self.data:
                raise TensorError("argmax of an empty tensor is undefined")
            return GTensor("I64", (), [max(range(len(self.data)),
                                           key=lambda i: self.data[i])])
        def amax(xs):
            return max(range(len(xs)), key=lambda i: xs[i])
        return self._reduce_axis(axis, amax, "I64")

    def _reduce_axis(self, axis: int, fn: Callable[[List[Any]], Any],
                     dtype: Optional[str] = None) -> "GTensor":
        if self.rank == 0:
            raise TensorError("cannot reduce a rank-0 tensor along an axis")
        ax = axis if axis >= 0 else axis + self.rank
        if not 0 <= ax < self.rank:
            raise TensorError(f"axis {axis} out of range for {self.type_name()}")
        new_shape = tuple(d for i, d in enumerate(self.shape) if i != ax)
        strides = self._strides()
        out_size = _prod(new_shape) or 1
        results: List[Any] = [None] * out_size
        new_strides = _strides_for(new_shape)
        for off in range(self.size):
            idx = []
            rem = off
            for d in range(self.rank):
                idx.append(rem // strides[d])
                rem %= strides[d]
            groups: List[int] = [idx[i] for i in range(self.rank) if i != ax]
            gout = 0
            for g, st in zip(groups, new_strides):
                gout += g * st
            if results[gout] is None:
                results[gout] = []
            results[gout].append(self.data[off])
        out_dtype = dtype or self.dtype
        return GTensor(out_dtype, new_shape, [fn(r) for r in results])

    # -- linear algebra -------------------------------------------------
    def matmul(self, other: "GTensor") -> "GTensor":
        if not isinstance(other, GTensor):
            raise TensorError("matmul expects a Tensor operand")
        if self.rank == 2 and other.rank == 2:
            m, k = self.shape
            k2, n = other.shape
            if k != k2:
                raise TensorError(
                    f"matmul shape mismatch: {self.type_name()} @ "
                    f"{other.type_name()} (inner dimensions {k} and {k2} differ)")
            out = [0.0] * (m * n)
            for i in range(m):
                base = i * k
                obase = i * n
                for j in range(n):
                    acc = 0.0
                    for p in range(k):
                        acc += self.data[base + p] * other.data[p * n + j]
                    out[obase + j] = acc
            return GTensor("F64", (m, n), out)
        if self.rank == 1 and other.rank == 2:
            k, n = other.shape
            if self.shape[0] != k:
                raise TensorError(
                    f"matmul shape mismatch: {self.type_name()} @ "
                    f"{other.type_name()}")
            out = [sum(self.data[p] * other.data[p * n + j] for p in range(k))
                   for j in range(n)]
            return GTensor("F64", (n,), out)
        if self.rank == 2 and other.rank == 1:
            m, k = self.shape
            if other.shape[0] != k:
                raise TensorError(
                    f"matmul shape mismatch: {self.type_name()} @ "
                    f"{other.type_name()}")
            out = [sum(self.data[i * k + p] * other.data[p] for p in range(k))
                   for i in range(m)]
            return GTensor("F64", (m,), out)
        if self.rank == 1 and other.rank == 1:
            if self.shape[0] != other.shape[0]:
                raise TensorError("dot product requires equal lengths")
            return GTensor("F64", (),
                           [sum(a * b for a, b in zip(self.data, other.data))])
        raise TensorError(
            f"matmul is implemented for ranks 1 and 2, got "
            f"{self.type_name()} @ {other.type_name()}",
            hint="batched matmul is on the roadmap (spec section 14)")

    def dot(self, other):
        return self.matmul(other)

    # -- misc ----------------------------------------------------------
    def clip(self, lo: float, hi: float) -> "GTensor":
        return GTensor(self.dtype, self.shape,
                       [min(max(v, lo), hi) for v in self.data])

    def broadcast(self, shape: Sequence[int]) -> "GTensor":
        shape = tuple(int(d) for d in shape)
        if self.shape == shape:
            return self.copy()
        if self.shape == ():
            return GTensor(self.dtype, shape,
                           [self.data[0]] * _prod(shape))
        raise TensorError(
            f"cannot broadcast {self.type_name()} to {list(shape)}",
            hint="Gama-G v0.1 broadcasts rank-0 scalars only")

    def allclose(self, other: "GTensor", tol: float = 1e-6) -> bool:
        if self.shape != other.shape:
            return False
        return all(abs(a - b) <= tol for a, b in zip(self.data, other.data))

    def equals(self, other: Any) -> bool:
        if not isinstance(other, GTensor):
            return False
        return self.shape == other.shape and self.data == other.data


def _strides_for(shape: Sequence[int]) -> Tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _scalar_or_tensor(value: Any, dtype: str) -> GTensor:
    return GTensor(dtype if dtype in _FLOAT_DTYPES else "F64", (), [value])


# ----------------------------------------------------------------------
# reverse-mode automatic differentiation (spec section 14)
# ----------------------------------------------------------------------
@dataclass
class TapeNode:
    tensor: GTensor
    parents: Tuple["TapeNode", ...] = ()
    backward: Optional[Callable[[GTensor], None]] = None
    grad: Optional[GTensor] = None
    name: str = ""


class Tape:
    """A minimal reverse-mode autodiff tape.

    Records elementwise and matmul operations; :meth:`backward` propagates
    gradients from a scalar loss back to every leaf marked ``requires_grad``.
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, TapeNode] = {}
        self.leaves: List[TapeNode] = []
        self.ops: int = 0

    def leaf(self, tensor: GTensor, requires_grad: bool = False) -> TapeNode:
        node = TapeNode(tensor=tensor, name="leaf")
        if requires_grad:
            node.grad = GTensor(tensor.dtype, tensor.shape)
            self.leaves.append(node)
        self.nodes[id(tensor)] = node
        return node

    def record(self, out: GTensor, parents: Sequence[TapeNode],
               backward: Callable[[GTensor], None], name: str = "") -> TapeNode:
        node = TapeNode(tensor=out, parents=tuple(parents), backward=backward,
                        name=name)
        self.nodes[id(out)] = node
        self.ops += 1
        return node

    def backward(self, loss_node: TapeNode) -> None:
        if loss_node.tensor.size != 1:
            raise TensorError(
                "backward must start from a scalar loss; reduce the tensor "
                "first (e.g. `loss = mse.sum()`)",
                shape=list(loss_node.tensor.shape))
        loss_node.grad = GTensor(loss_node.tensor.dtype,
                                 loss_node.tensor.shape, [1.0])
        order = self._topo(loss_node)
        for node in reversed(order):
            if node.backward is None or node.grad is None:
                continue
            node.backward(node.grad)
        for leaf in self.leaves:
            if leaf.grad is not None:
                leaf.tensor._grad = leaf.grad

    def _topo(self, root: TapeNode) -> List[TapeNode]:
        order: List[TapeNode] = []
        seen = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            order.append(node)
            stack.extend(node.parents)
        return order

    def accumulate(self, node: TapeNode, grad: GTensor) -> None:
        if node.grad is None:
            node.grad = grad.copy() if isinstance(grad, GTensor) else grad
        else:
            node.grad = node.grad.add(grad)


def parameter(shape: Sequence[int], dtype: str = "F64",
              init: Optional[Sequence[Any]] = None,
              seed: int = 0) -> GTensor:
    """Create a tensor parameter with deterministic initialisation.

    Determinism matters here: spec section 1.3 requires reproducible results
    for critical computations, so initialisation is seeded rather than
    drawing on ambient entropy.
    """
    n = _prod(shape)
    if init is not None:
        return GTensor(dtype, shape, list(init))
    state = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    values = []
    scale = 1.0 / math.sqrt(max(1, n))
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        values.append(((state / 0x7FFFFFFF) * 2.0 - 1.0) * scale)
    return GTensor(dtype, shape, values)
