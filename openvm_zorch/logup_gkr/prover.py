"""SWIRL's fractional sumcheck over the LogUp-GKR circuit (dense variant).

Reuses zorch's dense circuit (``GkrLayer`` / ``build_pyramid`` — the stride-2
pair fold is byte-identical to the reference's segment tree) and its sumcheck
primitives (``lift_to_domain`` / ``fold_pair``), but drives the per-layer
sumcheck itself: the reference's transcript differs from zorch's own GKR
protocol in form, not structure —

- round polynomials are sent as evaluations on ``{1, 2, 3}`` (the verifier
  infers ``s(0)`` from the previous claim); zorch observes all of ``{0..3}``;
- the layer claims go on the wire as ``(p(0,ρ), q(0,ρ), p(1,ρ), q(1,ρ))``;
  zorch orders ``(n0, n1, d0, d1)``;
- λ batches the *denominator* term: ``eq·((p0·q1 + p1·q0) + λ·q0·q1)``;
  zorch's ``logup_combine`` batches the numerator;
- the sumcheck binds the LSB of the layer index first (the reference's MLE
  fold pairs adjacent entries), and the claim-merge challenge μ_j becomes the
  new FIRST coordinate: ``ξ^{(j)} = (μ_j, ρ)``; the eq table is therefore
  little-endian in ξ — ``expand_eq_to_hypercube`` (MSB-first) gets ξ reversed.

Reference: ``fractional_sumcheck`` (logup_zerocheck/fractional_sumcheck_gkr.rs)
plus the ξ padding loop of ``prove_zerocheck_and_logup`` (mod.rs).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from openvm_zorch.transcript import sample_ext
from zorch.logup_gkr.circuit import GkrLayer, build_pyramid
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.prover import fold_pair, lift_to_domain
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize

# eq (deg 1) * projective fraction addition (deg 2).
_DEGREE = 3


@dataclass(frozen=True)
class FracSumcheckProof:
    """The GKR half of the reference ``GkrProof`` (PoW witness excluded)."""

    q0_claim: Array  # () EF — the root denominator (numerator is zero)
    claims_per_layer: list[Array]  # per layer, (4,) EF in wire order p0,q0,p1,q1
    sumcheck_polys: list[Array]  # per layer j>=1, (j, 3) EF evals on {1,2,3}
    # Sampled challenges, for transcript-trajectory comparison in tests.
    lambdas: list[Array]
    mus: list[Array]
    rhos: list[list[Array]]


def _eq_table(xi: list[Array]) -> Array:
    """eq(ξ, y) for y on the hypercube, little-endian in ξ (ξ[0] ↔ bit 0)."""
    point = jnp.stack(xi[::-1])
    return expand_eq_to_hypercube(point, jnp.ones((), point.dtype))


def fractional_sumcheck(
    transcript: DuplexTranscript, num: Array, den: Array
) -> tuple[DuplexTranscript, FracSumcheckProof, list[Array]]:
    """Prove ``Σ num/den == 0`` over the hypercube (``assert_zero`` semantics).

    ``num``/``den`` are the input-layer evaluations (BabyBear⁴, length a power
    of two ≥ 2). Returns the advanced transcript, the proof, and ξ — the final
    evaluation point, little-endian (ξ[0] is the innermost/first variable).
    """
    total_rounds = log2_strict_usize(num.shape[0])
    if total_rounds == 0:
        raise ValueError("input layer must have at least 2 evaluations")

    # zorch's GkrLayer holds the two children of each tree node as separate
    # MLEs: node i of the first layer carries leaves (2i, 2i+1) — exactly the
    # reference's segment-tree pairing.
    layers = build_pyramid(
        GkrLayer(
            numerator_0=num[0::2],
            numerator_1=num[1::2],
            denominator_0=den[0::2],
            denominator_1=den[1::2],
            num_interaction_variables=0,
        )
    )

    # Root fraction: p must vanish (LogUp balance); only q goes on the wire.
    floor = layers[-1]
    p_root = floor.numerator_0 * floor.denominator_1 + floor.numerator_1 * floor.denominator_0
    q_root = floor.denominator_0 * floor.denominator_1
    if int(jnp.sum(p_root != 0)) != 0:
        raise ValueError("non-zero root sum: interactions do not balance")
    transcript = transcript.observe(q_root)

    def layer_claims(layer: GkrLayer) -> Array:
        # Wire order (p_xi_0, q_xi_0, p_xi_1, q_xi_1); each MLE is length 1.
        return jnp.stack(
            [
                layer.numerator_0[0],
                layer.denominator_0[0],
                layer.numerator_1[0],
                layer.denominator_1[0],
            ]
        )

    # Layer 1 is checked by the verifier directly: claims, then μ_1.
    claims = layer_claims(floor)
    transcript = transcript.observe(claims)
    transcript, mu_1 = sample_ext(transcript)
    xi = [mu_1]

    claims_per_layer = [claims]
    sumcheck_polys: list[Array] = []
    lambdas: list[Array] = []
    mus = [mu_1]
    rhos: list[list[Array]] = []
    for round_ in range(1, total_rounds):
        layer = layers[total_rounds - 1 - round_]  # MLEs of length 2^round_
        transcript, lam = sample_ext(transcript)
        lambdas.append(lam)

        state = [
            _eq_table(xi),
            layer.numerator_0,
            layer.denominator_0,
            layer.numerator_1,
            layer.denominator_1,
        ]
        rho: list[Array] = []
        round_polys = []
        for _ in range(round_):
            # Bind the LSB: pair adjacent entries (the reference's MLE fold).
            pairs = [(a[0::2], a[1::2]) for a in state]
            eq, p0, q0, p1, q1 = (
                lift_to_domain(lo, hi, _DEGREE) for lo, hi in pairs
            )
            # Batched summand: eq·((p0·q1 + p1·q0) + λ·q0·q1). Note λ weights
            # the denominator term — the opposite of zorch's logup_combine.
            integrand = eq * ((p0 * q1 + p1 * q0) + lam * (q0 * q1))
            s_evals = jnp.sum(integrand, axis=-1)  # (degree+1,) at {0..3}
            transcript = transcript.observe(s_evals[1:])
            transcript, r_round = sample_ext(transcript)
            state = [fold_pair(lo, hi, r_round) for lo, hi in pairs]
            rho.append(r_round)
            round_polys.append(s_evals[1:])

        folded = GkrLayer(
            numerator_0=state[1],
            numerator_1=state[3],
            denominator_0=state[2],
            denominator_1=state[4],
            num_interaction_variables=0,
        )
        claims = layer_claims(folded)
        transcript = transcript.observe(claims)
        transcript, mu = sample_ext(transcript)
        # ξ^{(j)} = (μ_j, ρ): the merge challenge is the new first coordinate.
        xi = [mu] + rho

        claims_per_layer.append(claims)
        sumcheck_polys.append(jnp.stack(round_polys))
        mus.append(mu)
        rhos.append(rho)

    proof = FracSumcheckProof(
        q0_claim=q_root[0],
        claims_per_layer=claims_per_layer,
        sumcheck_polys=sumcheck_polys,
        lambdas=lambdas,
        mus=mus,
        rhos=rhos,
    )
    return transcript, proof, xi


def pad_xi(
    transcript: DuplexTranscript, xi: list[Array], target_len: int
) -> tuple[DuplexTranscript, list[Array]]:
    """Sample extra ξ coordinates up to ``l_skip + n_global``.

    The GKR point has ``l_skip + n_logup`` coordinates; when some AIR is
    taller than the interactions hypercube (``n_max > n_logup``) the batch
    sumcheck needs more, sampled directly (mod.rs).
    """
    xi = list(xi)
    while len(xi) < target_len:
        transcript, extra = sample_ext(transcript)
        xi.append(extra)
    return transcript, xi
