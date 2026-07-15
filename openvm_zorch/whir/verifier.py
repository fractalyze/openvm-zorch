"""Stage-5 verifier: the dual of ``WhirRound`` over the generic ``WhirVerifier``.

``verify_whir`` is the inverse of ``openvm_zorch.whir.prover.prove_whir_opening``:
repackage the reference ``WhirProof`` (per-query lists of opened rows and Merkle
paths) into the generic ``WhirProof`` (``Opening`` pytrees vmapped over the
queries), rebuild the same ``WhirVerifier`` the prover drove, and replay one
``verify`` — the stage math only. The chain Round that drives it
(``WhirVerifierRound``) lives with the other stage duals in
``openvm_zorch/verify.py``.
"""

from __future__ import annotations

from typing import Sequence

import frx.numpy as jnp
from frx import Array, lax

from openvm_zorch.fields import F
from openvm_zorch.poly_common import VerificationError
from openvm_zorch.whir.prover import WhirConfig, WhirProof
from openvm_zorch.whir.scheme import SwirlWhirScheme
from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import Opening
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.pcs.whir.config import WhirParams
from zorch.pcs.whir.config import WhirProof as GenericWhirProof
from zorch.pcs.whir.verifier import WhirVerifier
from zorch.transcript import DuplexTranscript


def _stack_paths(paths: Sequence[Array]) -> list[Array]:
    """Per-query reference Merkle paths (``Q`` × ``(depth, digest_elems)``) → the
    generic ``Opening.path``: a list over levels, each ``(Q, digest_elems)``. The
    inverse of the prover adapter's ``_per_query_paths``."""
    stacked = jnp.stack(list(paths))  # (Q, depth, digest_elems)
    return list(jnp.moveaxis(stacked, 1, 0))  # depth × (Q, digest_elems)


def _opening_from_rows(rows: Sequence[Array], paths: Sequence[Array]) -> Opening:
    """Round-0 strided opening from per-query base-field rows (``Q`` × ``(2^k, W)``)
    and Merkle paths — the inverse of the prover's ``_per_query_rows`` /
    ``_per_query_paths``."""
    return Opening(row=jnp.stack(list(rows)), path=_stack_paths(paths))


def _opening_from_ef_values(values: Sequence[Array], paths: Sequence[Array]) -> Opening:
    """Later-round strided opening. The reference stores the opened coset as ``Q``
    EF values (``(2^k,)`` each); the generic ``Opening.row`` is their base-field
    limbs (``(Q, 2^k, limbs)``), so bitcast back — the inverse of the prover's
    ``ef_from_limbs``."""
    row = lax.bitcast_convert_type(jnp.stack(list(values)), F)  # (Q, 2^k, limbs)
    return Opening(row=row, path=_stack_paths(paths))


def verify_whir(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    l_skip: int,
    n_stack: int,
    log_blowup: int,
    whir: WhirConfig,
    proof: WhirProof,
    stacking_openings: Sequence[Sequence[Array]],
    commitments: Sequence[Array],
    u: Sequence[Array],
) -> DuplexTranscript:
    """Check Stage 5 over the generic ``zorch.pcs.whir`` ``WhirVerifier``.

    The scheme-agnostic round driver — sumcheck replay, query-position sampling,
    strided root reconstruction, k-fold consistency, final residual constraint —
    lives in ``WhirVerifier``; the SWIRL-specific maps (prismalinear initial
    message, Möbius weight, no-op bind) ride ``SwirlWhirScheme``.

    A single common-main commitment is the supported shape, matching the prover
    adapter (SWIRL multi-commitment μ-batching is out of scope).
    """
    if len(commitments) != 1:
        raise VerificationError(
            f"the generic WHIR consumer opens a single commitment, got "
            f"{len(commitments)} (SWIRL multi-commitment μ-batch is out of scope)"
        )
    k = whir.k
    m = l_skip + n_stack
    num_rounds = len(whir.num_queries)

    code = ReedSolomon(message_len=1 << m, blowup=1 << log_blowup, dtype=F)
    strided = StridedMerkleTree(sponge, compressor, 1 << k)
    wparams = WhirParams(
        k_whir=k,
        num_queries=tuple(whir.num_queries),
        mu_pow_bits=whir.mu_pow_bits,
        folding_pow_bits=whir.folding_pow_bits,
        query_pow_bits=whir.query_phase_pow_bits,
        rate_increase=True,
    )
    verifier = WhirVerifier(code, strided, wparams, SwirlWhirScheme(l_skip))

    z = jnp.stack(list(u))  # the opening point (m,) — == u_cube on the cube
    # The running claim is the μ-power combine of the per-column opening claims
    # (the generic verifier's ``eval_coeffs(values, μ)``); pass them as that vector.
    values = jnp.stack(list(stacking_openings[0]))  # (W,) EF

    gproof = GenericWhirProof(
        mu_pow_witness=proof.mu_pow_witness,
        sumcheck_polys=proof.whir_sumcheck_polys,
        codeword_roots=proof.codeword_commits,
        ood_values=proof.ood_values,
        folding_pow_witnesses=proof.folding_pow_witnesses,
        query_pow_witnesses=proof.query_phase_pow_witnesses,
        initial_openings=[
            _opening_from_rows(rows, paths)
            for rows, paths in zip(
                proof.initial_round_opened_rows, proof.initial_round_merkle_proofs
            )
        ],
        codeword_openings=[
            _opening_from_ef_values(
                proof.codeword_opened_values[r], proof.codeword_merkle_proofs[r]
            )
            for r in range(num_rounds - 1)
        ],
        final_poly=proof.final_poly,
    )

    ok, transcript = verifier.verify(commitments[0], [z], values, gproof, transcript)
    if not bool(ok):
        raise VerificationError("WHIR verification failed")
    return transcript
