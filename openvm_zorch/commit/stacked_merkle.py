"""SWIRL's query-strided Merkle tree over zorch's Sponge + Compression.

Reference: openvm-stark-backend ``MerkleTree::new``
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/stacked_pcs.rs#L412

The first ``log2(rows_per_query)`` levels pair leaves at *query stride*
``s = num_leaves / rows_per_query`` — level ``l`` compresses
``(prev[2x·s + y], prev[(2x+1)·s + y])`` into ``next[x·s + y]`` — so the
``rows_per_query = 2^k_whir`` rows a WHIR query opens all sit under one digest
of the first stored layer. Levels above fold plain adjacent pairs, exactly
zorch's ``MerkleTree``. The strided pairing is SWIRL-specific, which is why
this consumer carries its own tree instead of reusing zorch's.

Only the layers from the query layer up are kept (``digest_layers[0]`` has
``num_leaves / rows_per_query`` digests): the strided levels below are
recomputed by the verifier from the opened rows, so storing them would buy
nothing — mirroring the Rust prover, whose proofs index ``digest_layers[0]``
as the leaf layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array

from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.utils.bits import is_power_of_two, log2_strict_usize


@dataclass(frozen=True)
class StackedMerkleTree:
    """Commit-side tree state: the hashed row matrix, the stored digest layers
    (query layer first, root last) and the query structure.

    ``backing_matrix`` is whatever the caller hashed — base-field codeword
    rows for the Stage-1 commitment, the 4-limb view of an extension codeword
    for WHIR's per-round trees (the reference flattens extension elements to
    basis coefficients before hashing, ``MerkleTree::new``'s ``hash_input``).
    """

    backing_matrix: Array
    digest_layers: list[Array]
    rows_per_query: int

    @property
    def root(self) -> Array:
        return self.digest_layers[-1][0]

    @property
    def query_stride(self) -> int:
        return self.digest_layers[0].shape[0]

    def opened_rows(self, index: int) -> Array:
        """The ``rows_per_query`` rows a query at leaf ``index`` opens —
        ``{index + t·query_stride}`` for ``t`` in ``0..rows_per_query``
        (Rust ``get_opened_rows``)."""
        if not 0 <= index < self.query_stride:
            raise ValueError(f"index {index} out of range [0, {self.query_stride})")
        return self.backing_matrix[index :: self.query_stride]

    def query_merkle_proof(self, index: int) -> Array:
        """Sibling digests from the query layer to just below the root,
        ``(proof_depth, digest)`` — the strided levels below the query layer
        are recomputed by the verifier from the opened rows, so the proof
        starts at ``digest_layers[0]`` (Rust ``query_merkle_proof``)."""
        if not 0 <= index < self.query_stride:
            raise ValueError(f"index {index} out of range [0, {self.query_stride})")
        siblings = []
        for layer in self.digest_layers[:-1]:
            siblings.append(layer[index ^ 1])
            index >>= 1
        return jax.numpy.stack(siblings)


def stacked_merkle_commit(
    sponge: Sponge, compressor: Compression, matrix: Array, rows_per_query: int
) -> StackedMerkleTree:
    """Hash each row of ``(height, width)`` ``matrix`` to a leaf, fold
    ``log2(rows_per_query)`` query-strided levels, then plain pairs to the root.
    """
    height = matrix.shape[0]
    if not is_power_of_two(height):
        raise ValueError(f"matrix height ({height}) must be a power of two")
    if not is_power_of_two(rows_per_query):
        raise ValueError(f"rows_per_query ({rows_per_query}) must be a power of two")
    if rows_per_query > height:
        raise ValueError(f"rows_per_query ({rows_per_query}) > leaves ({height})")

    layer = jax.vmap(sponge.hash)(matrix)
    digest = layer.shape[-1]
    query_stride = height // rows_per_query

    # Query-strided levels: reshape (m, ...) as (m/(2s), 2, s, digest) so lanes
    # (2x·s + y, (2x+1)·s + y) land in one compress; the result re-flattens to
    # (m/2, digest) preserving next[x·s + y] order.
    for _ in range(log2_strict_usize(rows_per_query)):
        pairs = layer.reshape(-1, 2, query_stride, digest)
        left = pairs[:, 0].reshape(-1, digest)
        right = pairs[:, 1].reshape(-1, digest)
        layer = jax.vmap(lambda l, r: compressor.compress(jax.numpy.stack([l, r])))(
            left, right
        )

    digest_layers = [layer]
    while layer.shape[0] > 1:
        pairs = layer.reshape(-1, 2, digest)
        layer = jax.vmap(compressor.compress)(pairs)
        digest_layers.append(layer)
    return StackedMerkleTree(matrix, digest_layers, rows_per_query)
