"""End-to-end SWIRL verifier — the five stages checked from a proof + vk.

The structural dual of ``prove`` (``openvm_zorch/prove.py``): ``verify_chain``
builds a ``zorch.round.VerifyChain`` of one verifier Round per prover stage —
``CommitVerifierRound`` / ``GkrVerifierRound`` / ``ZeroCheckVerifierRound`` /
``StackingVerifierRound`` / ``WhirVerifierRound``, the duals of ``prove_chain``'s
``CommitRound`` … ``WhirRound``. Each Round re-derives its stage's Fiat-Shamir
challenges and checks the stage's algebraic relation, threading a witness-free
``VerifyCarry`` (the dual of ``ProveCarry``); the chain consumes the prover's
one-message-per-round proof, so a stage present on one side and not the other is
a structural reject, not a silent Fiat-Shamir desync. The verifier takes only
the verifying key (per-AIR shape + constraint DAG, no traces) and the proof. A
failed check raises ``VerificationError``; returning normally means the proof is
accepted.

The Rounds wrap the per-stage check helpers, which follow the reference verifier
(crates/stark-backend/src/verifier):

- Stage 2 ``_verify_gkr_stage``: GKR fractional-sumcheck verify, ξ padding.
- Stage 3 ``_verify_zerocheck_stage``: the batched ZeroCheck+LogUp sumcheck,
  closed by re-evaluating the constraint/interaction claim at the folded point
  from the column openings. (Stages 2–3 were one ``verify_zerocheck_and_logup``
  before the chain split them at the prover's GKR/ZeroCheck seam.)
- Stage 4 ``_verify_stacked_reduction``: re-derive λ, check s₀ against the
  opening claims, run the sumcheck, close on the stacking-opening claim.
- Stage 5 ``_verify_whir``: μ batching, per-round sumcheck folds + OOD, the
  query phase (Merkle-path verification + k-fold codeword consistency), and
  the final WHIR polynomial constraint.

A verifier Round raises ``VerificationError`` on its stage's check rather than
threading an ``ok`` (openvm's verifier checks were raise-based before the chain,
and keeping that is a pure refactor); each returns ``ok = True`` and the chain's
structural AND is the honest path's verdict.

PoW witnesses are checked, not re-ground. Opened rows and Merkle paths are
verified against the committed roots.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Sequence

import jax.numpy as jnp
from jax import Array, lax

from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.fields import EF, F, MODULUS, f_const, f_to_ef
from openvm_zorch.logup_gkr.prover import FracSumcheckProof
from openvm_zorch.logup_zerocheck import prism
from openvm_zorch.logup_zerocheck.constraints import (
    ConstraintsDag,
    acc_constraints,
    acc_interactions,
    eval_nodes,
)
from openvm_zorch.logup_zerocheck.prover import BatchConstraintProof
from openvm_zorch.prove import GkrStageMsg, Proof, SystemParams
from openvm_zorch.stacked_reduction.prover import StackingProof
from openvm_zorch.transcript import check_witness, sample_ext
from openvm_zorch.whir.prover import WhirProof
from openvm_zorch.whir.scheme import SwirlWhirScheme
from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import Opening
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.pcs.whir.config import WhirParams
from zorch.pcs.whir.config import WhirProof as GenericWhirProof
from zorch.pcs.whir.verifier import WhirVerifier
from zorch.round import Round, VerifyChain
from zorch.transcript import DuplexTranscript


class VerificationError(Exception):
    """Raised when any verifier check fails."""


@dataclass(frozen=True)
class AirVk:
    """Per-AIR verifying-key shape the verifier consumes, in input order."""

    dag: ConstraintsDag
    log_height: int
    width: int  # common-main column count
    public_values: tuple[int, ...]
    constraint_degree: int
    needs_next: bool
    is_required: bool


# --- scalar algebra (poly_common.rs) -------------------------------------

_ZERO = jnp.zeros((), EF)
_ONE = jnp.ones((), EF)
_HALF = f_to_ef(f_const((MODULUS + 1) // 2))  # 2⁻¹
_INV6 = f_to_ef(f_const(pow(6, MODULUS - 2, MODULUS)))
_THREE = f_to_ef(f_const(3))


def _eq(a: Array, b: Array) -> bool:
    """Canonical-u32 equality of two field elements (base or extension),
    independent of any custom-dtype ``__eq__``."""
    au = lax.bitcast_convert_type(jnp.atleast_1d(a), F)
    bu = lax.bitcast_convert_type(jnp.atleast_1d(b), F)
    return bool(jnp.array_equal(au, bu))


def _horner(coeffs: Sequence[Array], x: Array) -> Array:
    acc = _ZERO
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _interp_linear_01(evals: Sequence[Array], x: Array) -> Array:
    return (evals[1] - evals[0]) * x + evals[0]


def _interp_quadratic_012(evals: Sequence[Array], x: Array) -> Array:
    s1 = evals[1] - evals[0]
    s2 = evals[2] - evals[1]
    p = (s2 - s1) * _HALF
    q = s1 - p
    return (p * x + q) * x + evals[0]


def _interp_cubic_0123(evals: Sequence[Array], x: Array) -> Array:
    s1 = evals[1] - evals[0]
    s2 = evals[2] - evals[0]
    s3 = evals[3] - evals[0]
    d3 = s3 - (s2 - s1) * _THREE
    p = d3 * _INV6
    q = (s2 - d3) * _HALF - s1
    r = s1 - p - q
    return ((p * x + q) * x + r) * x + evals[0]


@lru_cache(maxsize=None)
def _inv_factorials(s_deg: int) -> tuple[Array, ...]:
    """``1/0!, 1/1!, …, 1/s_deg!`` as EF constants — fixed per ``s_deg``."""
    out = []
    fval = 1
    for i in range(s_deg + 1):
        if i > 0:
            fval = (fval * i) % MODULUS
        out.append(f_to_ef(f_const(pow(fval, MODULUS - 2, MODULUS))))
    return tuple(out)


def _interp_at_nodes(evals: Sequence[Array], x: Array, s_deg: int) -> Array:
    """Lagrange evaluation at ``x`` of the degree-``s_deg`` polynomial through
    nodes ``{0, 1, …, s_deg}`` (batch_constraints.rs barycentric form)."""
    invfact = _inv_factorials(s_deg)
    pref = [_ONE]
    for i in range(s_deg):
        pref.append(pref[-1] * (x - f_to_ef(f_const(i))))
    suf = [_ONE]
    for i in range(s_deg):
        suf.append(suf[-1] * (f_to_ef(f_const(s_deg - i)) - x))
    acc = _ZERO
    for i in range(s_deg + 1):
        acc = (
            acc + evals[i] * pref[i] * suf[s_deg - i] * invfact[i] * invfact[s_deg - i]
        )
    return acc


def _progression_exp_2(m: Array, l: int) -> Array:
    """``∏_{i=0}^{l-1} (1 + m^{2^i})`` (evaluator.rs ``progression_exp_2``)."""
    acc = _ONE
    pow_m = m
    for _ in range(l):
        acc = acc * (_ONE + pow_m)
        pow_m = pow_m * pow_m
    return acc


def _exp_pow_2(x: Array, k: int) -> Array:
    for _ in range(k):
        x = x * x
    return x


# --- Stage 2: GKR fractional sumcheck ------------------------------------


def _verify_gkr(
    transcript: DuplexTranscript, proof: FracSumcheckProof, total_rounds: int
) -> tuple[DuplexTranscript, Array, Array, list[Array]]:
    if total_rounds == 0:
        if not _eq(proof.q0_claim, _ONE):
            raise VerificationError("GKR zero-round q0 must be 1")
        return transcript, _ZERO, _ONE, []

    transcript = transcript.observe(proof.q0_claim)
    claims = proof.claims_per_layer[0]  # (p0, q0, p1, q1)
    transcript = transcript.observe(claims)
    p_cross = claims[0] * claims[3] + claims[2] * claims[1]
    q_cross = claims[1] * claims[3]
    if not _eq(p_cross, _ZERO):
        raise VerificationError("GKR root zero-check failed")
    if not _eq(q_cross, proof.q0_claim):
        raise VerificationError("GKR root denominator consistency failed")
    transcript, mu = sample_ext(transcript)
    numer = _interp_linear_01([claims[0], claims[2]], mu)
    denom = _interp_linear_01([claims[1], claims[3]], mu)
    gkr_r = [mu]

    for round_ in range(1, total_rounds):
        transcript, lam = sample_ext(transcript)
        claim = numer + lam * denom
        polys = proof.sumcheck_polys[round_ - 1]  # (round_, 3) evals on {1,2,3}
        eq = _ONE
        round_r: list[Array] = []
        for sr in range(round_):
            poly = polys[sr]
            transcript = transcript.observe(poly)
            transcript, ri = sample_ext(transcript)
            round_r.append(ri)
            ev0 = claim - poly[0]
            claim = _interp_cubic_0123([ev0, poly[0], poly[1], poly[2]], ri)
            xi = gkr_r[sr]
            eq = eq * (xi * ri + (_ONE - xi) * (_ONE - ri))
        claims = proof.claims_per_layer[round_]
        transcript = transcript.observe(claims)
        p_cross = claims[0] * claims[3] + claims[2] * claims[1]
        q_cross = claims[1] * claims[3]
        expected = (p_cross + lam * q_cross) * eq
        if not _eq(expected, claim):
            raise VerificationError(f"GKR layer consistency failed at round {round_}")
        transcript, mu = sample_ext(transcript)
        numer = _interp_linear_01([claims[0], claims[2]], mu)
        denom = _interp_linear_01([claims[1], claims[3]], mu)
        gkr_r = [mu] + round_r

    return transcript, numer, denom, gkr_r


# --- Stage 3: batched ZeroCheck + LogUp ----------------------------------


def _by_rot(flat: Array, need_rot: bool) -> list[tuple[Array, Array]]:
    if need_rot:
        return [(flat[2 * i], flat[2 * i + 1]) for i in range(flat.shape[0] // 2)]
    return [(flat[i], _ZERO) for i in range(flat.shape[0])]


def _verify_gkr_stage(
    transcript: DuplexTranscript,
    params: SystemParams,
    sorted_vks: Sequence[AirVk],
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
    protocol-derived sizes ``verify_chain`` binds (the same values the prover's
    ``prove_chain`` derives)."""
    l_skip = params.l_skip
    n_per_trace = [vk.log_height - l_skip for vk in sorted_vks]

    transcript, ok = check_witness(transcript, params.logup_pow_bits, logup_pow_witness)
    if not bool(ok):
        raise VerificationError("invalid LogUp PoW witness")
    transcript, alpha = sample_ext(transcript)
    transcript, beta = sample_ext(transcript)

    total_interactions = sum(
        len(vk.dag.interactions) << (l_skip + max(n, 0))
        for vk, n in zip(sorted_vks, n_per_trace)
    )

    xi: list[Array] = []
    p_xi = _ZERO
    q_xi = alpha
    if total_interactions > 0:
        transcript, p_xi, q_xi, xi = _verify_gkr(
            transcript, gkr_proof, l_skip + n_logup
        )

    while len(xi) != l_skip + n_global:
        transcript, extra = sample_ext(transcript)
        xi.append(extra)

    return transcript, alpha, beta, xi, p_xi, q_xi


def _verify_zerocheck_stage(
    transcript: DuplexTranscript,
    params: SystemParams,
    sorted_vks: Sequence[AirVk],
    n_logup: int,
    n_max: int,
    bcp: BatchConstraintProof,
    alpha: Array,
    beta: Array,
    xi: list[Array],
    p_xi: Array,
    q_xi: Array,
) -> tuple[DuplexTranscript, list[Array]]:
    """Stage 3 verifier — the dual of ``ZeroCheckRound``: the batched ZeroCheck
    + LogUp sumcheck. Consumes the Stage-2 outputs off the carry (α/β, the
    padded point ξ, and the GKR claims ``p_xi`` / ``q_xi``), re-evaluates the
    constraint/interaction claim at the folded point from the proof's column
    openings, and returns the sumcheck point ``r``."""
    l_skip = params.l_skip
    n_per_trace = [vk.log_height - l_skip for vk in sorted_vks]

    transcript, lam = sample_ext(transcript)

    # 3. Observe per-air sum claims; reduce GKR claims to zero / alpha.
    for sum_p, sum_q in zip(bcp.numerator_term_per_air, bcp.denominator_term_per_air):
        p_xi = p_xi - sum_p
        q_xi = q_xi - sum_q
        transcript = transcript.observe(jnp.stack([sum_p, sum_q]))
    if not _eq(p_xi, _ZERO):
        raise VerificationError("GKR numerator claim mismatch")
    if not _eq(q_xi, alpha):
        raise VerificationError("GKR denominator claim mismatch")

    transcript, mu = sample_ext(transcript)
    sum_claim = _ZERO
    cur_mu = _ONE
    for sum_p, sum_q in zip(bcp.numerator_term_per_air, bcp.denominator_term_per_air):
        sum_claim = sum_claim + sum_p * cur_mu
        cur_mu = cur_mu * mu
        sum_claim = sum_claim + sum_q * cur_mu
        cur_mu = cur_mu * mu

    # 5. Univariate round 0.
    s0 = bcp.univariate_round_coeffs
    transcript = transcript.observe(jnp.stack(list(s0)))
    s_deg = params.max_constraint_degree + 1
    transcript, r_0 = sample_ext(transcript)
    size = 1 << l_skip
    s0_sum = _ZERO
    for j in range(0, len(s0), size):
        s0_sum = s0_sum + s0[j]
    s0_sum = s0_sum * f_to_ef(f_const(size))
    if not _eq(sum_claim, s0_sum):
        raise VerificationError("Stage-3 s0 sum mismatch")
    cur_sum = _horner(list(s0), r_0)
    rs = [r_0]

    # 6. Multilinear rounds.
    for round_ in range(n_max):
        evals = bcp.sumcheck_round_polys[round_]  # (s_deg,) at {1..s_deg}
        transcript = transcript.observe(evals)
        s_1 = evals[0]
        s_0v = cur_sum - s_1
        full = [s_0v] + [evals[i] for i in range(s_deg)]
        transcript, r = sample_ext(transcript)
        cur_sum = _interp_at_nodes(full, r, s_deg)
        rs.append(r)

    # Observe the column openings (common main, per trace, (claim, claim_rot)
    # pairs — rot is 0 when the AIR never rotates), matching the prover's
    # closing observes so Stage 4 continues from the same transcript state.
    for trace_idx, vk in enumerate(sorted_vks):
        pairs = _by_rot(bcp.column_openings[trace_idx][0], vk.needs_next)
        flat = jnp.stack([v for pair in pairs for v in pair])
        transcript = transcript.observe(flat)

    # 7. eq_3b per trace (matches the prover's eq_3bs).
    eq_3b_per_trace: list[list[Array]] = []
    stacked_idx = 0
    for trace_idx, vk in enumerate(sorted_vks):
        interactions = vk.dag.interactions
        if not interactions:
            eq_3b_per_trace.append([])
            continue
        n = n_per_trace[trace_idx]
        n_lift = max(n, 0)
        per: list[Array] = []
        n_bits = n_logup - n_lift
        for _ in interactions:
            b_int = stacked_idx >> (l_skip + n_lift)
            bits = [f_to_ef(f_const((b_int >> j) & 1)) for j in range(n_bits)]
            point = xi[l_skip + n_lift : l_skip + n_logup]
            per.append(prism.eval_eq_mle(point, bits))
            stacked_idx += 1 << (l_skip + n_lift)
        eq_3b_per_trace.append(per)

    # 8. eq_ns / eq_sharp_ns.
    eq_ns = [_ONE] * (n_max + 1)
    eq_sharp_ns = [_ONE] * (n_max + 1)
    eq_ns[0] = prism.eval_eq_uni(l_skip, xi[0], r_0)
    eq_sharp_ns[0] = prism.eval_eq_sharp_uni(l_skip, xi[:l_skip], r_0)
    for i in range(1, n_max + 1):
        eq_mle = prism.eval_eq_mle([xi[l_skip + i - 1]], [rs[i]])
        eq_ns[i] = eq_ns[i - 1] * eq_mle
        eq_sharp_ns[i] = eq_sharp_ns[i - 1] * eq_mle
    r_rev = rs[n_max]
    for i in range(n_max - 1, -1, -1):
        eq_ns[i] = eq_ns[i] * r_rev
        eq_sharp_ns[i] = eq_sharp_ns[i] * r_rev
        r_rev = r_rev * rs[i]

    # 9. Re-evaluate the claim from the column openings.
    beta_pows = [_ONE]
    max_msg = max(
        (len(i.message) for vk in sorted_vks for i in vk.dag.interactions), default=0
    )
    for _ in range(max_msg + 1):
        beta_pows.append(beta_pows[-1] * beta)
    lambda_pows = [_ONE]
    max_constraints = max((len(vk.dag.constraint_idx) for vk in sorted_vks), default=1)
    for _ in range(max(max_constraints, 1) - 1):
        lambda_pows.append(lambda_pows[-1] * lam)

    interactions_evals: list[Array] = []
    constraints_evals: list[Array] = []
    for trace_idx, vk in enumerate(sorted_vks):
        n = n_per_trace[trace_idx]
        n_lift = max(n, 0)
        pairs = _by_rot(bcp.column_openings[trace_idx][0], vk.needs_next)
        local = jnp.stack([c for c, _ in pairs])
        nxt = jnp.stack([c_rot for _, c_rot in pairs])
        parts = [(local, nxt)]

        if n < 0:
            l_eff = l_skip + n
            rs_n = [_exp_pow_2(rs[0], -n)]
            norm = f_to_ef(f_const(pow(pow(2, -n, MODULUS), MODULUS - 2, MODULUS)))
        else:
            l_eff = l_skip
            rs_n = rs[: n + 1]
            norm = _ONE
        omega = f_to_ef(f_const(prism.omega_int(l_eff)))
        inv = f_to_ef(f_const(pow(1 << l_eff, MODULUS - 2, MODULUS)))
        prod_lo = _ONE
        prod_hi = _ONE
        for x in rs_n[1:]:
            prod_lo = prod_lo * (_ONE - x)
            prod_hi = prod_hi * x
        is_first = inv * _progression_exp_2(rs_n[0], l_eff) * prod_lo
        is_last = inv * _progression_exp_2(rs_n[0] * omega, l_eff) * prod_hi
        sels = jnp.stack([is_first, _ONE - is_last, is_last])

        node_vals = eval_nodes(vk.dag, sels, parts, vk.public_values)
        expr = acc_constraints(vk.dag, node_vals, lambda_pows)
        constraints_evals.append(eq_ns[n_lift] * expr)

        num, denom = acc_interactions(
            vk.dag, node_vals, beta_pows, eq_3b_per_trace[trace_idx]
        )
        interactions_evals.append(num * norm * eq_sharp_ns[n_lift])
        interactions_evals.append(denom * eq_sharp_ns[n_lift])

    evaluated = _ZERO
    cur_mu = _ONE
    for x in interactions_evals + constraints_evals:
        evaluated = evaluated + x * cur_mu
        cur_mu = cur_mu * mu
    if not _eq(cur_sum, evaluated):
        raise VerificationError("Stage-3 final claim mismatch")

    return transcript, rs


# --- Stage 4: stacked opening reduction ----------------------------------


def _verify_stacked_reduction(
    transcript: DuplexTranscript,
    params: SystemParams,
    proof: StackingProof,
    layout: StackedLayout,
    need_rot: Sequence[bool],
    column_openings: Sequence[Sequence[Array]],
    r: Sequence[Array],
) -> tuple[DuplexTranscript, list[Array]]:
    l_skip = params.l_skip
    n_stack = params.n_stack
    size = 1 << l_skip

    # Order the opening claims exactly as the prover batched them (per commit,
    # per column in layout order); common main only here.
    lambda_count = len(layout.sorted_cols)
    t_claims: list[tuple[Array, Array]] = []
    for trace_idx, vk_need_rot in enumerate(need_rot):
        t_claims.extend(_by_rot(column_openings[trace_idx][0], vk_need_rot))
    if len(t_claims) != lambda_count:
        raise VerificationError("Stage-4 opening-claim count mismatch")

    transcript, lam = sample_ext(transcript)
    lam_sqr = lam * lam
    lam_sqr_pows = [_ONE]
    for _ in range(lambda_count - 1):
        lam_sqr_pows.append(lam_sqr_pows[-1] * lam_sqr)

    s_0 = _ZERO
    for (t0, t1), lam_i in zip(t_claims, lam_sqr_pows):
        s_0 = s_0 + (t0 + t1 * lam) * lam_i
    coeffs = proof.univariate_round_coeffs
    s0_sum = _ZERO
    for j in range(0, coeffs.shape[0], size):
        s0_sum = s0_sum + coeffs[j]
    s0_sum = s0_sum * f_to_ef(f_const(size))
    if not _eq(s_0, s0_sum):
        raise VerificationError("Stage-4 s0 mismatch")
    transcript = transcript.observe(coeffs)

    u = [None] * (n_stack + 1)
    transcript, u[0] = sample_ext(transcript)
    claim = _horner(list(coeffs), u[0])
    for j in range(1, n_stack + 1):
        s_j_1 = proof.sumcheck_round_polys[j - 1][0]
        s_j_2 = proof.sumcheck_round_polys[j - 1][1]
        transcript = transcript.observe(jnp.stack([s_j_1, s_j_2]))
        transcript, u[j] = sample_ext(transcript)
        s_j_0 = claim - s_j_1
        claim = _interp_quadratic_012([s_j_0, s_j_1, s_j_2], u[j])

    # Final: reconstruct the per-column kernel coefficients and close on the
    # stacking-opening claim.
    openings = proof.stacking_openings  # [commit][col]
    q_coeffs = [[_ZERO for _ in openings[c]] for c in range(len(openings))]
    lambda_idx = 0
    for _mat_idx, _col_in_mat, s in layout.sorted_cols:
        col_need_rot = need_rot[_mat_idx]
        n = s.log_height - l_skip
        n_lift = max(n, 0)
        b = [
            f_to_ef(f_const((s.row_idx >> j) & 1))
            for j in range(l_skip + n_lift, l_skip + n_stack)
        ]
        eq_mle = prism.eval_eq_mle(u[n_lift + 1 :], b)
        ind = prism.eval_in_uni(l_skip, n, u[0])
        if n < 0:
            l_eff = l_skip + n
            rs_n = [_exp_pow_2(r[0], -n)]
        else:
            l_eff = l_skip
            rs_n = list(r[: n_lift + 1])
        eq_prism = prism.eval_eq_prism(l_eff, u[: n_lift + 1], rs_n)
        batched = lam_sqr_pows[lambda_idx] * eq_prism
        if col_need_rot:
            rot = prism.eval_rot_kernel_prism(l_eff, u[: n_lift + 1], rs_n)
            batched = batched + lam_sqr_pows[lambda_idx] * lam * rot
        q_coeffs[0][s.col_idx] = q_coeffs[0][s.col_idx] + eq_mle * batched * ind
        lambda_idx += 1

    final_sum = _ZERO
    for c in range(len(openings)):
        for col in range(len(openings[c])):
            q_j = openings[c][col]
            transcript = transcript.observe(q_j)
            final_sum = final_sum + q_coeffs[c][col] * q_j
    if not _eq(claim, final_sum):
        raise VerificationError("Stage-4 final sum mismatch")

    return transcript, [u[j] for j in range(n_stack + 1)]


# --- Stage 5: WHIR -------------------------------------------------------


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


def _verify_whir(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    proof: WhirProof,
    stacking_openings: Sequence[Sequence[Array]],
    commitments: Sequence[Array],
    u: Sequence[Array],
) -> DuplexTranscript:
    """Check Stage 5 over the generic ``zorch.pcs.whir`` ``WhirVerifier``.

    The inverse of ``openvm_zorch.whir.prover.prove_whir_opening``: repackage the
    reference ``WhirProof`` (per-query lists of opened rows and Merkle paths) into
    the generic ``WhirProof`` (``Opening`` pytrees vmapped over the queries),
    rebuild the same ``WhirVerifier`` the prover drove, and replay one ``verify``.
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
    whir = params.whir
    k = whir.k
    m = params.l_skip + params.n_stack
    num_rounds = len(whir.num_queries)

    code = ReedSolomon(message_len=1 << m, blowup=1 << params.log_blowup, dtype=F)
    strided = StridedMerkleTree(sponge, compressor, 1 << k, fuse=True)
    wparams = WhirParams(
        k_whir=k,
        num_queries=tuple(whir.num_queries),
        mu_pow_bits=whir.mu_pow_bits,
        folding_pow_bits=whir.folding_pow_bits,
        query_pow_bits=whir.query_phase_pow_bits,
        rate_increase=True,
    )
    verifier = WhirVerifier(code, strided, wparams, SwirlWhirScheme(params.l_skip))

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


# --- chain ---------------------------------------------------------------


@dataclass(frozen=True)
class VerifyCarry:
    """What flows between the verifier's stage Rounds: the verifying keys (set
    at construction) plus each stage's outputs the next stage reads — the
    witness-free dual of ``prove.ProveCarry``. Stage Rounds return it via
    ``replace`` — a Round writes its own fields and passes the rest through.

    Like ``ProveCarry`` (and for the same reason) this is a plain dataclass, not
    a registered pytree: the verifier's stages jit *internally* (Stage 5's WHIR
    islands), so the carry is host-side Python that never crosses a ``jax.jit``
    boundary and need not flatten.
    """

    # Verifying keys in input (verifying-key) order and stacking order — the
    # dual of ``ProveCarry.airs`` / ``sorted_airs``. The preamble observes per
    # AIR in input order; the stages consume stacking order.
    air_vks: Sequence[AirVk]
    sorted_vks: Sequence[AirVk]
    # Stage 1 (commit) output: the commitment, read by Stage 5 as the opened
    # root — dual of ``ProveCarry.root``.
    common_main_commit: Array | None = None
    # Stage 2 (GKR) outputs read by Stage 3: the sampled α/β, the padded point
    # ξ, and the reduced GKR numerator/denominator claims.
    alpha: Array | None = None
    beta: Array | None = None
    xi: list[Array] | None = None
    gkr_numer: Array | None = None
    gkr_denom: Array | None = None
    # Stage 3 (ZeroCheck) outputs: the sumcheck point read by Stage 4, plus the
    # proof's column openings Stage 4 batches. The prover reads the committed
    # matrix off the carry; the verifier reads its openings off the proof, so
    # they ride the carry from here — dual of ``ProveCarry.bcp_r``.
    r: list[Array] | None = None
    column_openings: Sequence[Sequence[Array]] | None = None
    # Stage 4 (stacking) outputs read by Stage 5: the opening point ``u`` and
    # the stacking openings (WHIR's running claim) — dual of
    # ``ProveCarry.stacking_u``.
    u: list[Array] | None = None
    stacking_openings: Sequence[Sequence[Array]] | None = None


class CommitVerifierRound(Round):
    """Stage 1 dual of ``CommitRound``: replays the preamble absorb stream (vk
    pre-hash, the commitment, then per AIR in *input* order an optional present
    flag, log height, and public values) with the proof's commitment message,
    and writes it onto the carry for the WHIR dual. No local check: the
    commitment is validated downstream by Stage 5's Merkle openings against this
    root (sp1-zorch's ``TraceCommitVerifierRound``)."""

    def __init__(self, *, vk_pre_hash: Sequence[int]) -> None:
        self._vk_pre_hash = vk_pre_hash

    def __call__(
        self, carry: VerifyCarry, msg: Array, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        transcript = transcript.observe(jnp.array(list(self._vk_pre_hash), dtype=F))
        transcript = transcript.observe(msg)
        for vk in carry.air_vks:
            meta: list[int] = [] if vk.is_required else [1]
            meta.append(vk.log_height)
            meta.extend(vk.public_values)
            transcript = transcript.observe(jnp.array(meta, dtype=F))
        carry = replace(carry, common_main_commit=msg)
        return carry, transcript, jnp.bool_(True)


class GkrVerifierRound(Round):
    """Stage 2 dual of ``GkrRound`` over ``_verify_gkr_stage``: writes α/β, the
    padded point ξ, and the GKR claims onto the carry for ZeroCheck. The message
    is the GKR stage's proof contribution (``GkrStageMsg``); its ``xi`` field is
    the prover's record — the verifier re-derives ξ rather than trusting it,
    exactly as the flat verifier did."""

    def __init__(self, *, params: SystemParams, n_logup: int, n_global: int) -> None:
        self._params = params
        self._n_logup = n_logup
        self._n_global = n_global

    def __call__(
        self, carry: VerifyCarry, msg: GkrStageMsg, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        transcript, alpha, beta, xi, p_xi, q_xi = _verify_gkr_stage(
            transcript,
            self._params,
            carry.sorted_vks,
            self._n_logup,
            self._n_global,
            msg.gkr_proof,
            msg.logup_pow_witness,
        )
        carry = replace(
            carry, alpha=alpha, beta=beta, xi=xi, gkr_numer=p_xi, gkr_denom=q_xi
        )
        return carry, transcript, jnp.bool_(True)


class ZeroCheckVerifierRound(Round):
    """Stage 3 dual of ``ZeroCheckRound`` over ``_verify_zerocheck_stage``:
    consumes the Stage-2 outputs off the carry, verifies the batched ZeroCheck +
    LogUp sumcheck, and writes the sumcheck point ``r`` plus the proof's column
    openings (Stage 4 batches them) onto the carry."""

    def __init__(self, *, params: SystemParams, n_logup: int, n_max: int) -> None:
        self._params = params
        self._n_logup = n_logup
        self._n_max = n_max

    def __call__(
        self,
        carry: VerifyCarry,
        msg: BatchConstraintProof,
        transcript: DuplexTranscript,
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        transcript, r = _verify_zerocheck_stage(
            transcript,
            self._params,
            carry.sorted_vks,
            self._n_logup,
            self._n_max,
            msg,
            carry.alpha,
            carry.beta,
            carry.xi,
            carry.gkr_numer,
            carry.gkr_denom,
        )
        carry = replace(carry, r=r, column_openings=msg.column_openings)
        return carry, transcript, jnp.bool_(True)


class StackingVerifierRound(Round):
    """Stage 4 dual of ``StackingRound`` over ``_verify_stacked_reduction``:
    rebuilds the stacked layout from the verifying keys, batches the column
    openings off the carry, and verifies the stacked opening reduction. Writes
    the opening point ``u`` and the proof's stacking openings (WHIR's running
    claim) onto the carry."""

    def __init__(self, *, params: SystemParams) -> None:
        self._params = params

    def __call__(
        self, carry: VerifyCarry, msg: StackingProof, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        sorted_vks = carry.sorted_vks
        layout = StackedLayout.new(
            self._params.l_skip,
            self._params.l_skip + self._params.n_stack,
            [(vk.width, vk.log_height) for vk in sorted_vks],
        )
        need_rot = [vk.needs_next for vk in sorted_vks]
        transcript, u = _verify_stacked_reduction(
            transcript,
            self._params,
            msg,
            layout,
            need_rot,
            carry.column_openings,
            carry.r,
        )
        carry = replace(carry, u=u, stacking_openings=msg.stacking_openings)
        return carry, transcript, jnp.bool_(True)


class WhirVerifierRound(Round):
    """Stage 5 dual of ``WhirRound`` over ``_verify_whir``: forms ``u_cube`` from
    the opening point on the carry (the same Stage-4 → Stage-5 handoff
    ``u_cube = (u₀ squarings over the skip domain) ‖ u[1..]`` the prover does),
    then checks WHIR against the carry's commitment and stacking openings."""

    def __init__(
        self, sponge: Sponge, compressor: Compression, *, params: SystemParams
    ) -> None:
        self._sponge = sponge
        self._compressor = compressor
        self._params = params

    def __call__(
        self, carry: VerifyCarry, msg: WhirProof, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        u = carry.u
        u_cube = [u[0]]
        for _ in range(self._params.l_skip - 1):
            u_cube.append(u_cube[-1] * u_cube[-1])
        u_cube.extend(u[1:])
        transcript = _verify_whir(
            transcript,
            self._sponge,
            self._compressor,
            self._params,
            msg,
            carry.stacking_openings,
            [carry.common_main_commit],
            u_cube,
        )
        return carry, transcript, jnp.bool_(True)


def verify_chain(
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    air_vks: Sequence[AirVk],
) -> tuple[VerifyChain, VerifyCarry]:
    """Build the SWIRL verifier as one ``VerifyChain`` of stage Rounds plus its
    initial carry — the dual of ``prove.prove_chain``. One definition of the
    stage wiring so ``verify`` and any future per-stage verify-timing harness
    cannot drift on it (sp1-zorch's ``verify_shard_chain`` pattern).

    The protocol-derived sizes (stacking order, ``n_logup`` / ``n_max`` /
    ``n_global``) are computed here from the verifying keys — the same values
    ``prove_chain`` derives from the traces — and bound onto the Rounds. Returns
    the carry alongside the chain because the stacking order it derives is also
    the carry's statement.
    """
    l_skip = params.l_skip
    order = sorted(range(len(air_vks)), key=lambda i: (-air_vks[i].log_height, i))
    sorted_vks = [air_vks[i] for i in order]

    n_per_trace = [vk.log_height - l_skip for vk in sorted_vks]
    total_interactions = sum(
        len(vk.dag.interactions) << (l_skip + max(n, 0))
        for vk, n in zip(sorted_vks, n_per_trace)
    )
    n_logup = total_interactions.bit_length() - l_skip if total_interactions else 0
    n_max = max(max(n_per_trace), 0)
    n_global = max(n_max, n_logup)

    chain = VerifyChain(
        [
            CommitVerifierRound(vk_pre_hash=vk_pre_hash),
            GkrVerifierRound(params=params, n_logup=n_logup, n_global=n_global),
            ZeroCheckVerifierRound(params=params, n_logup=n_logup, n_max=n_max),
            StackingVerifierRound(params=params),
            WhirVerifierRound(sponge, compressor, params=params),
        ]
    )
    return chain, VerifyCarry(air_vks=air_vks, sorted_vks=sorted_vks)


# --- driver --------------------------------------------------------------


def verify(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    air_vks: Sequence[AirVk],
    common_main_commit: Array,
    proof: Proof,
) -> None:
    """Verify a SWIRL proof. Raises ``VerificationError`` on any failed check;
    returns ``None`` if the proof is accepted.

    A thin driver over ``verify_chain`` (the dual of ``prove``): deconstruct the
    ``Proof`` into the per-round message list — the inverse of ``prove``'s
    assembly, ``[commit, gkr, bcp, stacking, whir]`` — and replay the chain.
    Each verifier Round raises ``VerificationError`` on its stage's failed
    check, so rejection flows through exactly as the flat verifier's did; ``ok``
    is the chain's structural AND of the rounds, guarded here so a future
    ok-returning check cannot silently pass.
    """
    chain, carry = verify_chain(sponge, compressor, params, vk_pre_hash, air_vks)
    msgs = [
        common_main_commit,
        GkrStageMsg(proof.logup_pow_witness, proof.gkr_proof, proof.xi),
        proof.batch_constraint_proof,
        proof.stacking_proof,
        proof.whir_proof,
    ]
    _, _, ok = chain(carry, msgs, transcript)
    if not bool(ok):
        raise VerificationError("verification failed")
