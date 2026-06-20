"""SWIRL Stage 1 — stacked PCS trace commitment.

Reference: openvm-stark-backend ``stacked_commit``
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/stacked_pcs.rs#L116

stack (``stacking``) → RS-encode columns (``rs_message``) → query-strided
Merkle (``stacked_merkle``); the commitment is the tree root, observed into
the transcript by the shard prover (no domain separator of its own — SWIRL
binds shape via the verifying key, not the commitment).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Sequence

import jax
from jax import Array

from openvm_zorch.commit.rs_message import rs_code_matrix
from openvm_zorch.commit.stacked_merkle import StackedMerkleTree, stacked_merkle_commit
from openvm_zorch.commit.stacking import StackedLayout, stacked_matrix
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge


@dataclass(frozen=True)
class StackedPcsData:
    """Prover-side committed data: the layout, the stacked evaluation matrix,
    and the Merkle tree over the RS codeword matrix."""

    layout: StackedLayout
    matrix: Array
    tree: StackedMerkleTree

    @property
    def commit(self) -> Array:
        return self.tree.root


_COMMIT_PROFILE = os.environ.get("OPENVM_COMMIT_PROFILE") == "1"


class _StackedProfiler:
    """Coarse, env-guarded sub-op timer for ``stacked_commit`` localization
    (#46, GPU commit gap). No-op unless ``OPENVM_COMMIT_PROFILE=1``. Each
    ``mark`` blocks on the region's output arrays and prints the wall-clock
    since the previous mark, splitting the commit-warm number into stacking /
    RS-NTT / strided Merkle. Mirrors ``prove._CommitProfiler``."""

    def __init__(self) -> None:
        self._t = time.monotonic()

    def mark(self, label: str, *outputs: object) -> None:
        if not _COMMIT_PROFILE:
            return
        jax.block_until_ready(outputs)
        now = time.monotonic()
        print(f"    [stacked {label}] {now - self._t:.3f}s", flush=True)
        self._t = now


def stacked_commit(
    sponge: Sponge,
    compressor: Compression,
    l_skip: int,
    n_stack: int,
    log_blowup: int,
    k_whir: int,
    traces: Sequence[Array],
) -> tuple[Array, StackedPcsData]:
    """Commit ``traces`` (each ``(height, width)``, pre-sorted by descending
    height). Returns ``(root, data)``."""
    _p = _StackedProfiler()
    matrix, layout = stacked_matrix(l_skip, n_stack, traces)
    _p.mark("stacking", matrix)
    codeword = rs_code_matrix(l_skip, log_blowup, matrix)
    _p.mark("rs_ntt", codeword)
    tree = stacked_merkle_commit(
        sponge, compressor, codeword, 1 << k_whir, jit=True
    )
    _p.mark("merkle", tree.root)
    return tree.root, StackedPcsData(layout, matrix, tree)
