"""SWIRL Stage 1 — stacked PCS trace commitment.

Reference: openvm-stark-backend ``stacked_commit``
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/stacked_pcs.rs#L116

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


# ``rs_code_matrix`` is pure-eager by definition; jit it here so the whole
# eval->coeff->zero-pad->forward-NTT pipeline fuses into ONE kernel instead of
# dispatching each primitive separately. Eager decomposes the composite (the
# 7 primitives round-trip HBM with no fusion) -- ~63ms vs ~0.5ms jitted at the
# real common-main dims (2^21, ~2); the forward NTT itself is only ~0.2ms, so
# the eager dispatch was the whole pole, not NTT FLOP (#46). ``l_skip`` /
# ``log_blowup`` are static shape params. Mirrors stacked_merkle ``_jitted_commit``.
_rs_encode = jax.jit(rs_code_matrix, static_argnums=(0, 1))


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
    matrix, layout = stacked_matrix(l_skip, n_stack, traces)
    codeword = _rs_encode(l_skip, log_blowup, matrix)
    tree = stacked_merkle_commit(sponge, compressor, codeword, 1 << k_whir)
    return tree.root, StackedPcsData(layout, matrix, tree)
