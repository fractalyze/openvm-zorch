"""SWIRL's fractional sumcheck over the LogUp-GKR circuit (dense variant).

Reuses zorch's dense circuit (``GkrLayer`` / ``build_pyramid`` — the stride-2
pair fold is byte-identical to the reference's segment tree) and drives each
layer's per-variable sumcheck through the generic ``zorch.sumcheck.prove`` (a
register-resident marker when the transcript supports it). The reference's
transcript differs from zorch's own GKR protocol in form, not structure —

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
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array, lax

from openvm_zorch.fields import EF
from openvm_zorch.transcript import EF_LIMBS, sample_ext
from zorch.logup_gkr.circuit import GkrLayer, build_pyramid
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.prover import prove as sumcheck_prove
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize

# Each layer's per-variable sumcheck runs on the generic `zorch.sumcheck.prove`
# driver, not an eager Python round loop. An eager loop dispatches
# `_round_poly`/observe/sample/fold per round — an ~O(rounds²) chain of tiny
# sequential kernels over data that halves each fold, launch-bound on GPU (not
# FLOP-bound). When the transcript carries a dedicated-fusion Poseidon2
# (`has_dedicated_fusion`), `prove` wraps the whole binding scan — round poly,
# fold, and the Fiat-Shamir absorb/squeeze — in one `zorch.sumcheck` composite a
# vendor codegens register-resident, collapsing that storm into a single kernel.
# The SWIRL form rides the marker as `eval_start=1` (the `{1,2,3}` sent domain),
# a custom `_GkrSummand` (λ on the denominator, via the combine-agnostic
# `zorch.sumcheck.combine` seam), and an EF fold challenge (`challenge_dtype=EF`).
# The marker decomposes to the same scan when unrecognized, so the path is
# byte-identical. Only the SWIRL-specific transcript wiring stays here as
# shape-stable `_observe`/`_sample`/`_observe_sample` islands: the head (q_root,
# layer-1 claims, μ_1) and each layer's claims-observe + μ-sample.
#
# Round poly degree: eq (deg 1) * projective fraction addition (deg 2) = 3, so
# four evals {0,1,2,3} determine it — but the prover sends only {1,2,3} (the
# verifier infers s(0) from the running claim s(0)+s(1) = prev). `eval_start=1`
# gives the scan that truncated SENT domain.


@jax.jit
def _observe(transcript: DuplexTranscript, values: Array) -> DuplexTranscript:
    """Absorb ``values`` inside one fused Poseidon2 kernel."""
    return transcript.observe(values)


@jax.jit
def _sample(transcript: DuplexTranscript) -> tuple[DuplexTranscript, Array]:
    """Squeeze one BabyBear⁴ challenge inside one fused Poseidon2 kernel."""
    return sample_ext(transcript)


@jax.jit
def _observe_sample(
    transcript: DuplexTranscript, values: Array
) -> tuple[DuplexTranscript, Array]:
    """Absorb ``values`` then squeeze one challenge in a SINGLE fused Poseidon2
    region — one dispatch in place of a separate ``_observe`` then ``_sample``.

    Used for the claims → μ merge between layers (the binding-loop transcript now
    rides the ``zorch.sumcheck`` marker). Each layer's claims absorb is
    immediately followed by its μ squeeze, so fusing the pair saves a launch per
    layer. Byte-identical to ``_sample(_observe(...))`` — the same Poseidon2
    absorb/squeeze ops in the same order, just one jit boundary. ``values`` is
    shape-stable ((4,) claims), so Poseidon2 lowers once."""
    return sample_ext(transcript.observe(values))


@dataclass(frozen=True)
class _GkrSummand:
    """The SWIRL LogUp round summand the ``zorch.sumcheck`` scan folds, over the
    five MLE factors ``[eq, p0, q0, p1, q1]``: ``eq·((p0·q1 + p1·q0) + λ·q0·q1)``.

    Degree 3 (eq deg 1 × the projective fraction-add deg 2). λ weights the
    *denominator* term — opposite of zorch's ``logup_combine`` (numerator), which
    is why this rides the generic scan with a local summand rather than zorch's
    ``LogupSumcheckRound``. λ is a loop-invariant scalar (fixed across the layer's
    rounds), threaded via ``combine_scalars`` — the marker carries it as the
    ``[combine scalars]`` operand segment of the nested ``zorch.sumcheck.combine``
    region, so the custom combine rides the recognizer with no zkx change."""

    lam: Array

    @property
    def degree(self) -> int:
        return 3

    def combine_scalars(self) -> tuple[Array, ...]:
        return (self.lam,)

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        (lam,) = scalars
        eq, p0, q0, p1, q1 = factors
        return eq * ((p0 * q1 + p1 * q0) + lam * (q0 * q1))

    def _combine(self, *factors: Array) -> Array:
        # Required by the SumcheckSummand protocol; the scan path calls `combine`.
        return self.combine(self.combine_scalars(), *factors)


@jax.jit
def _marked_sumcheck(lam, state, transcript):
    """One layer's per-variable sumcheck through the marker, jitted with a STABLE
    identity so the `zorch.sumcheck` composite lowers once per layer width and
    caches across layers and proofs — the amortization the eager islands get from
    their module-level `@jax.jit`. Calling `zorch.sumcheck.prove` directly in the
    eager layer loop instead re-traces the fresh composite body on every call (no
    cache), which dominates warm runtime."""
    return sumcheck_prove(
        _GkrSummand(lam),
        state,
        transcript,
        eval_start=1,
        challenge_dtype=EF,
        challenge_limbs=EF_LIMBS,
    )


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
    transcript, mu_1 = _observe_sample(transcript, claims)
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

        # Bit-reverse each MLE so the scan's MSB-first block fold reproduces the
        # reference's LSB-first stride fold (mirrors Stage-4 stacking).
        state = [
            lax.bit_reverse(a, dimensions=(0,))
            for a in (
                _eq_table(xi),
                layer.numerator_0,
                layer.denominator_0,
                layer.numerator_1,
                layer.denominator_1,
            )
        ]
        final_state, transcript, msgs = _marked_sumcheck(lam, state, transcript)
        rho = list(msgs.challenge)
        round_polys = msgs.round_poly  # (round_, 3): the sent evals s(1,2,3)

        # Folded claims (num0, den0, num1, den1) at the bound point; the eq factor
        # (final_state[0]) is not on the wire.
        _, num0, den0, num1, den1 = (f[0] for f in final_state)
        claims = jnp.stack([num0, den0, num1, den1])
        transcript, mu = _observe_sample(transcript, claims)
        # ξ^{(j)} = (μ_j, ρ): the merge challenge is the new first coordinate.
        xi = [mu] + rho

        claims_per_layer.append(claims)
        sumcheck_polys.append(round_polys)
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
