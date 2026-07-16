"""SWIRL's query-strided Merkle tree over zorch's Sponge + Compression.

Reference: openvm-stark-backend ``MerkleTree::new``
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/stacked_pcs.rs#L412

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

import functools
from dataclasses import dataclass

import frx
import frx.numpy as fnp
from frx import Array

from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge


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
        return fnp.stack(siblings)


@functools.partial(frx.jit, static_argnums=0)
def _jitted_commit(tree: StridedMerkleTree, matrix: Array):
    """``commit`` under ``frx.jit`` with ``tree`` static. ``StridedMerkleTree``
    hashes/compares by value (built for static jit-zone keys, zorch #214), so
    FRX's compile cache reuses one lowering per sponge/compressor/stride config."""
    return tree.commit(matrix)


def stacked_merkle_commit(
    sponge: Sponge,
    compressor: Compression,
    matrix: Array,
    rows_per_query: int,
) -> StackedMerkleTree:
    """Hash each row of ``(height, width)`` ``matrix`` to a leaf, fold
    ``log2(rows_per_query)`` query-strided levels, then plain pairs to the root.

    Delegates the strided fold to zorch's scheme-agnostic ``StridedMerkleTree``
    (the query-strided layout is not SWIRL-specific, so it belongs upstream) and
    always commits under ``_jitted_commit``'s cached ``frx.jit``: the tree's
    ``zorch.merkle_commit`` marker only lowers through XLA's
    ``ExpandMerkleCommit`` cross-leaf Poseidon2 fusion inside a jit trace, while
    an eager commit decomposes into one composite dispatch per pair — that
    dispatch storm was Stage-1's whole warm cost (~3.2s eager vs ~1ms jitted,
    issue #1), so an eager mode has no use worth a knob.
    """
    tree = StridedMerkleTree(sponge, compressor, rows_per_query)
    _, digest_layers = _jitted_commit(tree, matrix)
    return StackedMerkleTree(matrix, digest_layers, rows_per_query)
