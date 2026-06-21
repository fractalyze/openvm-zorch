"""SWIRL Stage 1 — stacked PCS trace commitment.

Reference: openvm-stark-backend ``stacked_commit``
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/stacked_pcs.rs#L116

stack (``stacking``) → RS-encode columns (``rs_message``) → query-strided
Merkle (``stacked_merkle``); the commitment is the tree root, observed into
the transcript by the shard prover (no domain separator of its own — SWIRL
binds shape via the verifying key, not the commitment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
from jax import Array

from openvm_zorch.commit.rs_message import rs_code_matrix
from openvm_zorch.commit.stacked_merkle import StackedMerkleTree, stacked_merkle_commit
from openvm_zorch.commit.stacking import StackedLayout, stacked_layout, stacked_take
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


# Commit as ONE fused kernel: the device gather (stacking), the eval->coeff->
# zero-pad->forward-NTT RS encode, and the query-strided Poseidon2 Merkle all
# lower into a single jit. Previously these were three separately-dispatched
# kernels (eager stacking + a jit per rs_encode + per merkle); fusing removes
# the inter-op launch/sync boundaries and lets the whole commit pipeline. The
# host-only pieces -- the layout and the memoized gather *index* (a pure function
# of trace shapes, native bakes it into tracegen) -- stay outside via
# ``stacked_layout``; only the data-dependent ``src`` gather + encode + hash
# trace in. ``sponge`` / ``compressor`` / shape params are static (mirrors
# stacked_merkle ``_jitted_commit``, which makes its tree static). ``jit=False``
# on the inner merkle so its fusion marker lowers under THIS jit, not a nested one.
def _stacked_commit_device(
    sponge: Sponge,
    compressor: Compression,
    l_skip: int,
    log_blowup: int,
    k_whir: int,
    height: int,
    width: int,
    gather_dev: Array,
    traces: Sequence[Array],
) -> tuple[Array, Array, list[Array]]:
    matrix = stacked_take(traces, gather_dev, height, width)
    codeword = rs_code_matrix(l_skip, log_blowup, matrix)
    tree = stacked_merkle_commit(sponge, compressor, codeword, 1 << k_whir, jit=False)
    return matrix, tree.backing_matrix, tree.digest_layers


_stacked_commit_jit = jax.jit(_stacked_commit_device, static_argnums=(0, 1, 2, 3, 4, 5, 6))


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
    layout, gather_dev, height, width = stacked_layout(l_skip, n_stack, traces)
    matrix, codeword, digest_layers = _stacked_commit_jit(
        sponge, compressor, l_skip, log_blowup, k_whir, height, width, gather_dev, traces
    )
    tree = StackedMerkleTree(codeword, digest_layers, 1 << k_whir)
    return tree.root, StackedPcsData(layout, matrix, tree)
