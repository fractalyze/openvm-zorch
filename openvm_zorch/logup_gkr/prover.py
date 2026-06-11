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

import jax
import jax.numpy as jnp
from jax import Array

from openvm_zorch.transcript import sample_ext
from zorch.logup_gkr.circuit import GkrLayer, build_pyramid
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.prover import fold_pair
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize

# Stage-2 ran fully eager, and the per-round Fiat-Shamir dominated it: a single
# eager `observe + sample_ext` dispatches the width-16 Poseidon2 permutation as
# thousands of tiny ops (~4.4 s/round measured), so the ~O(rounds²) binding
# steps cost ~174 s = 35% of prove — while the round *arithmetic* is only a few
# seconds. jit collapses each Poseidon2 permutation to one kernel (~70000×
# measured on observe+sample). Every transcript touch below is therefore jitted.
# GKR's round loop carries no per-round PoW grind (unlike WHIR, whose grind
# host-reads and breaks the trace), so the transcript threads through @jit
# cleanly. Byte-identical: jit fuses without reassociating field/Poseidon2 ops.
# The transcript runs through the shape-stable `_observe`/`_sample` islands rather
# than bundled into the variable-width round arithmetic — a compile lever explained
# at `_round_poly` (Poseidon2 lowered once, not once per layer width).
#
# Round poly degree: eq (deg 1) * projective fraction addition (deg 2) = 3, so
# four evals {0,1,2,3} determine it — but the prover sends only {1,2,3} (the
# verifier infers s(0) from the running claim s(0)+s(1) = prev). Lifting to the
# SENT domain skips the discarded u=0 across all five MLEs.
_SENT_US = (1, 2, 3)


def _lift_sent(lo: Array, hi: Array) -> Array:
    """Lift a split pair to the SENT eval domain ``{1,2,3}`` (skips u=0).

    ``f[u] = lo + u*(hi - lo)``, shape ``(3, *lo.shape)``. ``us`` uses
    ``jnp.stack`` (not ``jnp.arange``, whose iota is unsupported for extension
    dtypes)."""
    us = jnp.stack([jnp.array(u, lo.dtype) for u in _SENT_US])
    return lo + us.reshape((-1,) + (1,) * lo.ndim) * (hi - lo)


@jax.jit
def _observe(transcript: DuplexTranscript, values: Array) -> DuplexTranscript:
    """Absorb ``values`` inside one fused Poseidon2 kernel."""
    return transcript.observe(values)


@jax.jit
def _sample(transcript: DuplexTranscript) -> tuple[DuplexTranscript, Array]:
    """Squeeze one BabyBear⁴ challenge inside one fused Poseidon2 kernel."""
    return sample_ext(transcript)


# The round splits into two variable-width arithmetic islands (`_round_poly`,
# `_round_fold`) with the Fiat-Shamir transcript run between them via the
# shape-stable `_observe`/`_sample` islands above. Keeping the width-16 Poseidon2
# OUT of the per-round arithmetic is a COMPILE lever, not a warm one: the layer
# loop feeds widths 2^1..2^(rounds-1), so anything jitted with the state re-lowers
# once per width — and the Poseidon2 composite (sponge state / (3,) poly / (4,)
# challenge, all width-invariant) is ~3.6 s to lower vs ~0.1 s for the bare
# arithmetic (measured). Bundling it into the round step re-paid that ~3.6 s every
# width (~90% of GKR compile); routing the transcript through the stable islands
# lowers Poseidon2 ONCE. Warm runtime is unchanged — both keep one fused permutation
# kernel per round — and the split is byte-identical (same ops, same order).
@jax.jit
def _round_poly(
    state: list[Array], lam: Array
) -> tuple[Array, list[tuple[Array, Array]]]:
    """The sent round poly s(1,2,3) and the split pairs for the fold. λ weights the
    denominator term — opposite of logup_combine. No transcript: only this cheap
    arithmetic re-lowers per layer width."""
    pairs = [(a[0::2], a[1::2]) for a in state]
    eq, p0, q0, p1, q1 = (_lift_sent(lo, hi) for lo, hi in pairs)
    s_evals = jnp.sum(eq * ((p0 * q1 + p1 * q0) + lam * (q0 * q1)), axis=-1)
    return s_evals, pairs


@jax.jit
def _round_fold(pairs: list[tuple[Array, Array]], r: Array) -> list[Array]:
    """The LSB fold at challenge r (pair adjacent entries — the reference's MLE
    fold). Variable width, no transcript."""
    return [fold_pair(lo, hi, r) for lo, hi in pairs]


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


@jax.jit
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
    transcript = _observe(transcript, q_root)

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
    transcript = _observe(transcript, claims)
    transcript, mu_1 = _sample(transcript)
    xi = [mu_1]

    claims_per_layer = [claims]
    sumcheck_polys: list[Array] = []
    lambdas: list[Array] = []
    mus = [mu_1]
    rhos: list[list[Array]] = []
    for round_ in range(1, total_rounds):
        layer = layers[total_rounds - 1 - round_]  # MLEs of length 2^round_
        transcript, lam = _sample(transcript)
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
            s_evals, pairs = _round_poly(state, lam)
            transcript = _observe(transcript, s_evals)
            transcript, r_round = _sample(transcript)
            state = _round_fold(pairs, r_round)
            rho.append(r_round)
            round_polys.append(s_evals)

        folded = GkrLayer(
            numerator_0=state[1],
            numerator_1=state[3],
            denominator_0=state[2],
            denominator_1=state[4],
            num_interaction_variables=0,
        )
        claims = layer_claims(folded)
        transcript = _observe(transcript, claims)
        transcript, mu = _sample(transcript)
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
        transcript, extra = _sample(transcript)
        xi.append(extra)
    return transcript, xi
