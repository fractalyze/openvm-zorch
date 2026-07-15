"""Stage-2 verifier: the dual of ``GkrRound`` (verifier/gkr.rs).

``verify_gkr_stage`` checks the LogUp PoW witness, re-derives α/β, verifies
the GKR fractional sumcheck (``verify_gkr``), and pads ξ — the stage math
only. The chain Round that drives it (``GkrVerifierRound``) lives with the
other stage duals in ``openvm_zorch/verify.py``, mirroring sp1-zorch's
``verify_shard`` / ``logup_gkr.verifier`` split.
"""

from __future__ import annotations

from frx import Array

from openvm_zorch.logup_gkr.prover import FracSumcheckProof
from openvm_zorch.poly_common import (
    ONE,
    ZERO,
    VerificationError,
    eq_u32,
    interp_cubic_0123,
    interp_linear_01,
)
from openvm_zorch.transcript import check_witness, sample_ext
from zorch.transcript import DuplexTranscript


def verify_gkr(
    transcript: DuplexTranscript, proof: FracSumcheckProof, total_rounds: int
) -> tuple[DuplexTranscript, Array, Array, list[Array]]:
    if total_rounds == 0:
        if not eq_u32(proof.q0_claim, ONE):
            raise VerificationError("GKR zero-round q0 must be 1")
        return transcript, ZERO, ONE, []

    transcript = transcript.observe(proof.q0_claim)
    claims = proof.claims_per_layer[0]  # (p0, q0, p1, q1)
    transcript = transcript.observe(claims)
    p_cross = claims[0] * claims[3] + claims[2] * claims[1]
    q_cross = claims[1] * claims[3]
    if not eq_u32(p_cross, ZERO):
        raise VerificationError("GKR root zero-check failed")
    if not eq_u32(q_cross, proof.q0_claim):
        raise VerificationError("GKR root denominator consistency failed")
    transcript, mu = sample_ext(transcript)
    numer = interp_linear_01([claims[0], claims[2]], mu)
    denom = interp_linear_01([claims[1], claims[3]], mu)
    gkr_r = [mu]

    for round_ in range(1, total_rounds):
        transcript, lam = sample_ext(transcript)
        claim = numer + lam * denom
        polys = proof.sumcheck_polys[round_ - 1]  # (round_, 3) evals on {1,2,3}
        eq = ONE
        round_r: list[Array] = []
        for sr in range(round_):
            poly = polys[sr]
            transcript = transcript.observe(poly)
            transcript, ri = sample_ext(transcript)
            round_r.append(ri)
            ev0 = claim - poly[0]
            claim = interp_cubic_0123([ev0, poly[0], poly[1], poly[2]], ri)
            xi = gkr_r[sr]
            eq = eq * (xi * ri + (ONE - xi) * (ONE - ri))
        claims = proof.claims_per_layer[round_]
        transcript = transcript.observe(claims)
        p_cross = claims[0] * claims[3] + claims[2] * claims[1]
        q_cross = claims[1] * claims[3]
        expected = (p_cross + lam * q_cross) * eq
        if not eq_u32(expected, claim):
            raise VerificationError(f"GKR layer consistency failed at round {round_}")
        transcript, mu = sample_ext(transcript)
        numer = interp_linear_01([claims[0], claims[2]], mu)
        denom = interp_linear_01([claims[1], claims[3]], mu)
        gkr_r = [mu] + round_r

    return transcript, numer, denom, gkr_r


def verify_gkr_stage(
    transcript: DuplexTranscript,
    l_skip: int,
    logup_pow_bits: int,
    total_interactions: int,
    n_logup: int,
    n_global: int,
    gkr_proof: FracSumcheckProof,
    logup_pow_witness: Array,
) -> tuple[DuplexTranscript, Array, Array, list[Array], Array, Array]:
    """Stage 2 verifier — the dual of ``GkrRound``: check the LogUp PoW witness,
    re-derive α/β, verify the GKR fractional sumcheck, and pad ξ to
    ``l_skip + n_global``. Returns α, β, the padded point ξ, and the reduced
    GKR numerator/denominator claims (``p_xi`` / ``q_xi``) the ZeroCheck stage
    reduces the per-air sum claims against. ``n_logup`` / ``n_global`` are the
    protocol-derived sizes ``verify_chain`` binds, and ``total_interactions``
    the derived count it gates the GKR verify on (the same values the prover's
    ``prove_chain`` derives)."""
    transcript, ok = check_witness(transcript, logup_pow_bits, logup_pow_witness)
    if not bool(ok):
        raise VerificationError("invalid LogUp PoW witness")
    transcript, alpha = sample_ext(transcript)
    transcript, beta = sample_ext(transcript)

    xi: list[Array] = []
    p_xi = ZERO
    q_xi = alpha
    if total_interactions > 0:
        transcript, p_xi, q_xi, xi = verify_gkr(
            transcript, gkr_proof, l_skip + n_logup
        )

    while len(xi) != l_skip + n_global:
        transcript, extra = sample_ext(transcript)
        xi.append(extra)

    return transcript, alpha, beta, xi, p_xi, q_xi
