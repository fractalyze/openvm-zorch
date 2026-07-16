"""Stage 5 — WHIR opening proof (``prove_whir_opening``).

A thin consumer adapter over the generic ``zorch.pcs.whir`` PCS. The
scheme-agnostic round driver — sumcheck folds, per-round RS re-encode +
out-of-domain sample, strided query consistency, final constraint — lives in
``zorch.pcs.whir.WhirProver``; the SWIRL-specific maps (prismalinear initial
message, Möbius weight, no-op transcript bind) live in ``SwirlWhirScheme``
(``scheme.py``). This module builds the prover from the reference's
``WhirConfig`` plus the Stage-1 commitment, drives one open, and repackages the
generic ``WhirProof`` into the reference field layout — so ``prover_test``
byte-matches the fixture unchanged.

The migration replaces the hand-rolled round loop (and its device-compute
islands, which now live generically in zorch): the reference's per-round
geometry is the rate-increasing RS schedule (``log_rs -= 1`` per round), the
query phase opens ``2^k_whir``-row strided cosets, and the only consumer-specific
behaviour rides ``SwirlWhirScheme``.

Reference: openvm-stark-backend ``prove_whir_opening``
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/whir.rs#L78
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array
from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import Opening
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.pcs.whir.config import WhirParams
from zorch.pcs.whir.prover import WhirProver, WhirProverData
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize

from openvm_zorch.commit.stacked_merkle import StackedMerkleTree
from openvm_zorch.fields import F
from openvm_zorch.transcript import ef_from_limbs, grind, sample_ext
from openvm_zorch.whir.scheme import SwirlWhirScheme


@dataclass(frozen=True)
class WhirConfig:
    """The reference ``WhirConfig`` fields the prover consumes."""

    k: int
    num_queries: list[int]  # per WHIR round
    mu_pow_bits: int
    folding_pow_bits: int
    query_phase_pow_bits: int


@dataclass(frozen=True)
class WhirProof:
    """The reference ``WhirProof`` plus the sampled challenges.

    The generic ``zorch.pcs.whir.WhirProof`` carries the same content in a
    PCS-neutral shape (vmapped ``Opening`` pytrees, a single matrix commitment);
    ``prove_whir_opening`` repackages it into this reference layout — per-query
    lists of opened rows and Merkle paths, the codeword roots, the final poly —
    so the byte-match test compares field-for-field against the fixture.
    """

    mu_pow_witness: Array
    whir_sumcheck_polys: list[Array]  # num_whir_rounds·k × (2,) EF, evals at {1, 2}
    codeword_commits: list[Array]  # (num_whir_rounds − 1) × (8,) F
    ood_values: list[Array]  # (num_whir_rounds − 1) × () EF
    folding_pow_witnesses: list[Array]
    query_phase_pow_witnesses: list[Array]
    initial_round_opened_rows: list[list[Array]]  # per commit, per query: (2^k, W) F
    initial_round_merkle_proofs: list[
        list[Array]
    ]  # per commit, per query: (depth, 8) F
    codeword_opened_values: list[list[Array]]  # per later round, per query: (2^k,) EF
    codeword_merkle_proofs: list[list[Array]]
    final_poly: Array  # (2^(m − num_whir_rounds·k),) EF
    mu: Array


def _per_query_rows(opening: Opening) -> list[Array]:
    """The generic ``Opening`` is one pytree vmapped over the ``Q`` queries — its
    ``row`` is ``(Q, rows_per_query, width)``. The reference proof carries a
    per-query list, so unstack the leading query axis."""
    return list(opening.row)


def _per_query_paths(opening: Opening) -> list[Array]:
    """Per-query Merkle authentication path. The generic ``path`` is a list over
    levels (query layer up, leaf-first) each ``(Q, digest_elems)``; the reference
    stores ``(depth, digest_elems)`` per query, so stack the levels into a
    ``(Q, depth, digest_elems)`` array once and unstack the leading query axis
    (one batched ``stack`` rather than one per query)."""
    return list(fnp.stack(opening.path, axis=1))


def prove_whir_opening(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    l_skip: int,
    log_blowup: int,
    config: WhirConfig,
    committed: Sequence[tuple[Array, StackedMerkleTree]],
    u_cube: Sequence[Array],
    jit: bool = False,
) -> tuple[DuplexTranscript, WhirProof]:
    """Drive Stage 5 over the generic ``WhirProver``.

    ``committed`` holds, per commitment (common main first, then each cached/
    preprocessed main), the stacked evaluation matrix (base field, ``(2^m, Wᵢ)``)
    and its Stage-1 tree (whose backing matrix is the RS codeword the round-0
    queries open). The generic driver μ-combines the columns across all
    commitments and opens each commitment's tree at round 0; a single-commitment
    fixture is the length-1 case.

    ``jit`` is accepted for call-site compatibility but no longer changes
    behaviour: the generic driver always lowers its device compute to jitted
    islands and is byte-identical to an eager run (the islands are pure functions
    of their arrays), so both ``prover_test`` paths exercise the same code.
    """
    del jit  # generic driver is always island-jitted; both paths are identical.
    if not committed:
        raise ValueError("WHIR opens at least one commitment, got none")
    # Every committed matrix shares the stacked height (2^m); the μ-combine spans
    # all their columns (common main first) and round 0 opens each commit's tree.
    m = log2_strict_usize(committed[0][0].shape[0])
    k = config.k

    # The codeword domain follows the rate-increasing schedule (``log_rs -= 1``
    # per round); the strided tree opens ``2^k_whir``-row query cosets, matching
    # Stage-1's ``stacked_merkle_commit``. SWIRL's prismalinear/Möbius maps ride
    # the scheme, so the driver stays the scheme-agnostic generic one.
    code = ReedSolomon(message_len=1 << m, blowup=1 << log_blowup, dtype=F)
    strided = StridedMerkleTree(sponge, compressor, 1 << k)
    params = WhirParams(
        k_whir=k,
        num_queries=tuple(config.num_queries),
        mu_pow_bits=config.mu_pow_bits,
        folding_pow_bits=config.folding_pow_bits,
        query_pow_bits=config.query_phase_pow_bits,
        rate_increase=True,
    )
    prover = WhirProver(code, strided, params, SwirlWhirScheme(l_skip))
    # Reuse the Stage-1 commitments directly — each tree already delegates to the
    # generic StridedMerkleTree, so its codeword rows and digest layers feed the
    # round-0 query openings natively (no re-commit, no adapter). One
    # WhirProverData per commitment, common main first.
    prover_datas = [
        WhirProverData(
            mle=matrix, codeword=tree.backing_matrix, digest_layers=tree.digest_layers
        )
        for matrix, tree in committed
    ]

    z = fnp.stack(list(u_cube))  # the opening point (m,) — == u_cube on the cube
    # The reference proof also carries μ (its verifier re-derives it). The scheme
    # binds nothing at WHIR entry, so μ depends only on the entry transcript and
    # the grind — replay grind(mu_pow_bits) → sample on a copy of the transcript
    # (functional, so the discarded copy does not perturb the real threading).
    _, mu = sample_ext(grind(transcript, config.mu_pow_bits)[0])

    _, gproof, transcript = prover.open_batch(prover_datas, [z], transcript)

    return transcript, WhirProof(
        mu_pow_witness=gproof.mu_pow_witness,
        whir_sumcheck_polys=gproof.sumcheck_polys,
        codeword_commits=gproof.codeword_roots,
        ood_values=gproof.ood_values,
        folding_pow_witnesses=gproof.folding_pow_witnesses,
        query_phase_pow_witnesses=gproof.query_pow_witnesses,
        initial_round_opened_rows=[
            _per_query_rows(op) for op in gproof.initial_openings
        ],
        initial_round_merkle_proofs=[
            _per_query_paths(op) for op in gproof.initial_openings
        ],
        codeword_opened_values=[
            list(ef_from_limbs(op.row)) for op in gproof.codeword_openings
        ],
        codeword_merkle_proofs=[
            _per_query_paths(op) for op in gproof.codeword_openings
        ],
        final_poly=gproof.final_poly,
        mu=mu,
    )
