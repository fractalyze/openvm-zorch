"""SWIRL's stacked layout — greedy column stacking with short-column striding.

Reference: openvm-stark-backend ``StackedLayout`` / ``stacked_matrix``
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/stacked_pcs.rs#L136

Traces (pre-sorted by descending height) are flattened column-by-column into
one matrix of height ``2^(l_skip + n_stack)``. A trace column shorter than
``2^l_skip`` is lifted to length ``2^l_skip`` by striding: value ``i`` lands at
offset ``i * 2^(l_skip - log_height)``, zeros between. Columns are laid
head-to-tail down each stacked column; because every (lifted) height is a
power of two dividing the stacked height, a column never straddles two stacked
columns — the layout arithmetic below relies on that.

The layout is host-side Python over static shapes (heights are trace-time
constants); only the matrix assembly is traced jnp. The stacked width is
``ceil(total_lifted_cells / stacked_height)`` — note the Rust side sizes the
buffer from this cell count, NOT from the last occupied column, so a trailing
all-zero column (when the last lifted column ends exactly on a stacked-column
boundary but padding cells remain) is preserved here byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class StackedSlice:
    """Where one (lifted) trace column lives inside the stacked matrix."""

    col_idx: int
    row_idx: int
    log_height: int

    def lifted_len(self, l_skip: int) -> int:
        return 1 << max(self.log_height, l_skip)

    def stride(self, l_skip: int) -> int:
        return 1 << max(l_skip - self.log_height, 0)


@dataclass(frozen=True)
class StackedLayout:
    """Greedy stacking of ``sorted`` = [(width, log_height)] (descending
    log_height) into a matrix of height ``2^(l_skip + n_stack)``.

    ``sorted_cols[k] = (mat_idx, col_idx_in_mat, slice)`` in stacking order;
    ``mat_starts[m]`` indexes the first entry of matrix ``m``.
    """

    l_skip: int
    height: int
    width: int
    sorted_cols: list[tuple[int, int, StackedSlice]]
    mat_starts: list[int]

    @staticmethod
    def new(
        l_skip: int, log_stacked_height: int, sorted: Sequence[tuple[int, int]]
    ) -> "StackedLayout":
        if l_skip > log_stacked_height:
            raise ValueError(f"l_skip ({l_skip}) > log height ({log_stacked_height})")
        if any(a[1] < b[1] for a, b in zip(sorted, sorted[1:])):
            raise ValueError("traces must be sorted by descending log_height")
        height = 1 << log_stacked_height
        sorted_cols: list[tuple[int, int, StackedSlice]] = []
        mat_starts: list[int] = []
        col_idx = 0
        row_idx = 0
        for mat_idx, (width, log_ht) in enumerate(sorted):
            mat_starts.append(len(sorted_cols))
            if width == 0:
                continue
            if log_ht > log_stacked_height:
                raise ValueError(
                    f"trace log_height {log_ht} exceeds stacked {log_stacked_height}"
                )
            slice_len = 1 << max(log_ht, l_skip)
            for j in range(width):
                if row_idx + slice_len > height:
                    # Power-of-two heights mean overflow only happens at an
                    # exact column boundary; anything else is a logic error.
                    if row_idx != height:
                        raise ValueError(f"column overflow at stacked col {col_idx}")
                    col_idx += 1
                    row_idx = 0
                sorted_cols.append(
                    (mat_idx, j, StackedSlice(col_idx, row_idx, log_ht))
                )
                row_idx += slice_len
        width = col_idx + (1 if row_idx != 0 else 0)
        return StackedLayout(l_skip, height, width, sorted_cols, mat_starts)


def stacked_matrix(
    l_skip: int, n_stack: int, traces: Sequence[Array]
) -> tuple[Array, StackedLayout]:
    """Stack ``traces`` (each ``(height, width)``, pre-sorted by descending
    height) into a ``(2^(l_skip + n_stack), stacked_width)`` matrix.

    Returns ``(matrix, layout)``. Mirrors the Rust ``stacked_matrix`` including
    its buffer sizing: width = ceil(total lifted cells / stacked height), which
    can exceed the last occupied column (trailing zero column).
    """
    dtype = traces[0].dtype
    sorted_meta = [(t.shape[1], log2_strict_usize(t.shape[0])) for t in traces]
    layout = StackedLayout.new(l_skip, l_skip + n_stack, sorted_meta)
    height = layout.height
    total_cells = sum(max(t.shape[0], 1 << l_skip) * t.shape[1] for t in traces)
    width = -(-total_cells // height)

    # Assemble with one on-device GATHER. Each (lifted) source cell's row-major
    # offset in the (height, width) matrix is ``(row_idx + i*stride)*width +
    # col_idx`` — a function of the static layout, computed host-side. We invert
    # that into a per-output-cell gather index and ``take`` the source once.
    #
    # Gather, not scatter: XLA's GPU scatter serializes a large index set into a
    # pathologically slow kernel (~100x the host-bound eager loop here), whereas
    # gather is well-parallelized. Both replace an O(total columns) loop of
    # per-column slice/lift/concatenate dispatches that storms the GPU launch
    # queue (eager dispatch is host-bound there; #46).
    dest = [
        (s.row_idx + np.arange(traces[mat_idx].shape[0]) * s.stride(l_skip)) * width
        + s.col_idx
        for mat_idx, _j, s in layout.sorted_cols
    ]
    # Source cells in the layout's walk order (per matrix, column-major) behind a
    # zero sentinel at index 0; ``t.T.reshape(-1)`` lists column 0's rows, then
    # column 1's, ..., matching the per-column ``sorted_cols`` iteration. Output
    # cells with no source (lifting gaps, zero tail, trailing column) gather the
    # sentinel.
    src = jnp.concatenate([jnp.zeros((1,), dtype)] + [t.T.reshape(-1) for t in traces])
    gather = np.zeros((height * width,), np.int64)
    if dest:
        dest_flat = np.concatenate(dest)
        gather[dest_flat] = np.arange(dest_flat.size) + 1
    return jnp.take(src, jnp.asarray(gather)).reshape(height, width), layout
