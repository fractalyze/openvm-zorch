"""SWIRL's fractional sumcheck over the LogUp-GKR circuit (dense variant).

Reuses zorch's dense circuit (``GkrLayer`` / ``build_pyramid`` — the stride-2
pair fold is byte-identical to the reference's segment tree) and its sumcheck
primitives (``fold``), but drives the per-layer sumcheck itself: the
reference's transcript differs from zorch's own GKR protocol in form, not
structure —

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

import frx
import frx.numpy as fnp
from frx import Array
from zorch.logup_gkr.circuit import GkrLayer, build_pyramid
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.stage import ProveResult, ProverStage
from zorch.sumcheck.domain import fold
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize

from openvm_zorch.logup_gkr.input_layer import gkr_input_evals
from openvm_zorch.transcript import grind, sample_ext
from openvm_zorch.types import (
    LogupClaim,
    LogupGkrProof,
    SystemClaim,
    SystemWitness,
)

# Stage-2 ran fully eager, and the per-round Fiat-Shamir dominated it: a single
# eager `observe + sample_ext` dispatches the width-16 Poseidon2 permutation as
# thousands of tiny ops (~4.4 s/round measured), so the ~O(rounds²) binding
# steps cost ~174 s = 35% of prove — while the round *arithmetic* is only a few
# seconds. jit collapses each Poseidon2 permutation to one kernel (~70000×
# measured on observe+sample). Every transcript touch below is therefore jitted.
# GKR's round loop carries no per-round PoW grind (unlike WHIR, whose grind
# host-reads and breaks the trace), so the transcript threads through @jit
# cleanly. Byte-identical: jit fuses without reassociating field/Poseidon2 ops.
# Each layer's whole sumcheck — λ sample, every round's poly/observe/sample/fold,
# claims, μ — is ONE fused `_prove_layer` zone (transcript threaded inside), so
# the host dispatches once per layer, not ~3× per round: the stage is
# host-launch-bound on GPU, and the O(rounds²) island launches dominated it.
# The price is Poseidon2 re-lowering once per layer width at compile time —
# amortized by FRX_COMPILATION_CACHE_DIR (#120).
#
# Round poly degree: eq (deg 1) * projective fraction addition (deg 2) = 3, so
# four evals {0,1,2,3} determine it — but the prover sends only {1,2,3} (the
# verifier infers s(0) from the running claim s(0)+s(1) = prev). Lifting to the
# SENT domain skips the discarded u=0 across all five MLEs.
_SENT_US = (1, 2, 3)


def _lift_sent(state: Array) -> Array:
    """Lift the paired state to the SENT eval domain ``{1,2,3}`` (skips u=0).

    ``state`` is the stacked ``(5, W)`` round state; the LSB pairing splits it
    to ``lo``/``hi`` ``(5, W/2)`` and ``f[u] = lo + u*(hi - lo)`` lifts all
    five MLEs in ONE broadcast FMA, shape ``(3, 5, W/2)`` — a kernel-count
    lever: the per-MLE lift was 5 slices + 5 FMAs per round. ``us`` uses
    ``fnp.stack`` (not ``fnp.arange``, whose iota is unsupported for extension
    dtypes)."""
    lo, hi = state[:, 0::2], state[:, 1::2]
    us = fnp.stack([fnp.array(u, dtype=lo.dtype) for u in _SENT_US])
    return lo + us.reshape((-1, 1, 1)) * (hi - lo)


@frx.jit
def _observe(transcript: DuplexTranscript, values: Array) -> DuplexTranscript:
    """Absorb ``values`` inside one fused Poseidon2 kernel."""
    return transcript.observe(values)


@frx.jit
def _sample(transcript: DuplexTranscript) -> tuple[DuplexTranscript, Array]:
    """Squeeze one BabyBear⁴ challenge inside one fused Poseidon2 kernel."""
    return sample_ext(transcript)


@frx.jit
def _observe_sample(
    transcript: DuplexTranscript, values: Array
) -> tuple[DuplexTranscript, Array]:
    """Absorb ``values`` then squeeze one challenge in a SINGLE fused Poseidon2
    region — one dispatch in place of a separate ``_observe`` then ``_sample``.
    Used for the floor claims → μ₁ (the per-layer rounds run inside
    ``_prove_layer``). Byte-identical to ``_sample(_observe(...))`` — the same
    Poseidon2 absorb/squeeze ops in the same order, just one jit boundary."""
    return sample_ext(transcript.observe(values))


def _round_poly(state: Array, lam: Array) -> Array:
    """The sent round poly s(1,2,3). λ weights the denominator term — opposite of
    logup_combine. Binds the LSB: pairs adjacent entries (the reference's MLE
    fold). Field ops are exact, so the batched lift/products are byte-identical
    to the former per-MLE form."""
    f = _lift_sent(state)  # (3, 5, W/2)
    eq, p0, q0, p1, q1 = (f[:, i] for i in range(5))
    return fnp.sum(eq * ((p0 * q1 + p1 * q0) + lam * (q0 * q1)), axis=-1)


@frx.jit
def _prove_layer(
    transcript: DuplexTranscript,
    xi: list[Array],
    n0: Array,
    d0: Array,
    n1: Array,
    d1: Array,
) -> tuple[DuplexTranscript, Array, Array, list[Array], Array, Array]:
    """One layer's whole sumcheck as a single fused zone.

    ``len(xi)`` — the layer's round count — is pytree STRUCTURE, so the round
    loop unrolls statically at trace time and frx retraces once per layer
    (widths and xi length differ layer to layer). No ``lax.scan``: a runtime
    While is a fusion barrier that leaves one launch per round; unrolled, XLA
    fuses the inter-round Fiat-Shamir glue and the host dispatches once per
    layer. Same field algebra in the same transcript order as the former
    per-round islands — byte-identical (field ops are exact)."""
    transcript, lam = sample_ext(transcript)
    # One (5, W) stack so each round's lift and fold are ONE broadcast op over
    # all five MLEs, not five — `fold` is ndim-agnostic over the last axis.
    state = fnp.stack([_eq_table(xi), n0, d0, n1, d1])
    rho: list[Array] = []
    round_polys = []
    for _ in range(len(xi)):
        s_evals = _round_poly(state, lam)
        transcript, r_round = sample_ext(transcript.observe(s_evals))
        state = fold(state, r_round, msb=False)
        rho.append(r_round)
        round_polys.append(s_evals)
    # Wire order (p_xi_0, q_xi_0, p_xi_1, q_xi_1); each folded MLE is length 1.
    claims = state[1:, 0]
    transcript, mu = sample_ext(transcript.observe(claims))
    return transcript, lam, fnp.stack(round_polys), rho, claims, mu


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
    point = fnp.stack(xi[::-1])
    return expand_eq_to_hypercube(point, fnp.ones((), point.dtype))


def empty_frac_sumcheck_proof(ef_dtype) -> FracSumcheckProof:
    """The GKR proof for an instance with no interactions.

    When the input layer is empty the reference's ``fractional_sumcheck``
    early-returns ``fractional_sum = (0, 1)`` with empty layer claims and
    sumcheck polys, and an empty ξ — touching neither the transcript nor ξ
    (fractional_sumcheck_gkr.rs:65). ``q0_claim`` carries the denominator 1; the
    numerator is the implicit zero the root balance already satisfies."""
    return FracSumcheckProof(
        q0_claim=fnp.ones((), ef_dtype),
        claims_per_layer=[],
        sumcheck_polys=[],
        lambdas=[],
        mus=[],
        rhos=[],
    )


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
            num_batch_variables=0,
        )
    )

    # Root fraction: p must vanish (LogUp balance); only q goes on the wire.
    floor = layers[-1]
    p_root = (
        floor.numerator_0 * floor.denominator_1
        + floor.numerator_1 * floor.denominator_0
    )
    q_root = floor.denominator_0 * floor.denominator_1
    if int(fnp.sum(p_root != 0)) != 0:
        raise ValueError("non-zero root sum: interactions do not balance")
    transcript = _observe(transcript, q_root)

    def layer_claims(layer: GkrLayer) -> Array:
        # Wire order (p_xi_0, q_xi_0, p_xi_1, q_xi_1); each MLE is length 1.
        return fnp.stack(
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
        transcript, lam, round_polys, rho, claims, mu = _prove_layer(
            transcript,
            xi,
            layer.numerator_0,
            layer.denominator_0,
            layer.numerator_1,
            layer.denominator_1,
        )
        # ξ^{(j)} = (μ_j, ρ): the merge challenge is the new first coordinate.
        xi = [mu] + rho

        claims_per_layer.append(claims)
        sumcheck_polys.append(round_polys)
        lambdas.append(lam)
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


class LogupGkrProver(
    ProverStage[SystemClaim, SystemWitness, LogupClaim, LogupGkrProof, DuplexTranscript]
):
    """Reduce the system's interaction claim to one fraction-sum claim at ξ.

    Grinds the LogUp PoW, samples α/β, builds the GKR input layer from the
    traces, and runs the fractional sumcheck.
    """

    def __init__(self, *, l_skip: int, logup_pow_bits: int) -> None:
        self._l_skip = l_skip
        self._logup_pow_bits = logup_pow_bits

    def prove(
        self,
        claim: SystemClaim,
        witness: SystemWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[LogupClaim, LogupGkrProof, DuplexTranscript]:
        shape = claim.shape
        airs = witness.sorted_airs
        transcript, logup_pow_witness = grind(transcript, self._logup_pow_bits)
        transcript, alpha = sample_ext(transcript)
        transcript, beta = sample_ext(transcript)
        if shape.total_interactions > 0:
            num, den = gkr_input_evals(
                self._l_skip,
                shape.n_logup,
                [a.trace for a in airs],
                [a.dag for a in airs],
                [a.public_values for a in airs],
                [a.needs_next for a in airs],
                [a.cached_mains for a in airs],
                alpha,
                beta,
            )
            transcript, gkr_proof, xi = fractional_sumcheck(transcript, num, den)
        else:
            # No interactions: the reference builds an empty GKR input layer, so
            # ``fractional_sumcheck`` is a no-op — it neither absorbs into the
            # transcript nor produces any ξ. The verifier gates ``verify_gkr`` on
            # the same ``total_interactions == 0`` (verifier/batch_constraints.rs),
            # so skipping it here keeps both sides' transcript and ξ in lockstep.
            gkr_proof = empty_frac_sumcheck_proof(alpha.dtype)
            xi = []
        transcript, xi = pad_xi(transcript, xi, self._l_skip + shape.n_global)
        return ProveResult(
            LogupClaim(system=claim, alpha=alpha, beta=beta, xi=xi),
            LogupGkrProof(logup_pow_witness, gkr_proof, xi),
            transcript,
        )
