"""End-to-end SWIRL verifier — the five stages checked from a proof + vk.

The mirror of ``prove`` (``openvm_zorch/prove.py``): it re-derives every
Fiat-Shamir challenge from the same preamble and checks each stage's
algebraic relation, taking only the verifying key (per-AIR shape + constraint
DAG, no traces) and the proof. A failed check raises ``VerificationError``;
returning normally means the proof is accepted.

Structure follows the reference verifier (crates/stark-backend/src/verifier):

- Stage 2-3 ``verify_zerocheck_and_logup``: GKR fractional-sumcheck verify,
  then the batched ZeroCheck+LogUp sumcheck, closed by re-evaluating the
  constraint/interaction claim at the folded point from the column openings.
- Stage 4 ``verify_stacked_reduction``: re-derive λ, check s₀ against the
  opening claims, run the sumcheck, close on the stacking-opening claim.
- Stage 5 ``verify_whir``: μ batching, per-round sumcheck folds + OOD, the
  query phase (Merkle-path verification + k-fold codeword consistency), and
  the final WHIR polynomial constraint.

PoW witnesses are checked, not re-ground. Opened rows and Merkle paths are
verified against the committed roots.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from openvm_zorch.prove import Proof, SystemParams
from openvm_zorch.stacked_reduction.prover import StackingProof
from openvm_zorch.transcript import check_witness, sample_bits, sample_ext
from openvm_zorch.whir.prover import WhirProof
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize


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
        acc = acc + evals[i] * pref[i] * suf[s_deg - i] * invfact[i] * invfact[s_deg - i]
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


def _eval_mobius_eq_mle(u: Sequence[Array], x: Sequence[Array]) -> Array:
    acc = _ONE
    for u_i, x_i in zip(u, x):
        w0 = _ONE - u_i * 2
        acc = acc * (w0 * (_ONE - x_i) + u_i * x_i)
    return acc


def _eval_mle_evals_at_point(evals: list[Array], x: Sequence[Array]) -> Array:
    """Evaluate the MLE given by its ``2^len(x)`` hypercube values at ``x``."""
    buf = list(evals)
    length = len(buf)
    for xj in reversed(list(x)):
        length >>= 1
        buf = [buf[i] * (_ONE - xj) + buf[length + i] * xj for i in range(length)]
    return buf[0]


# --- Merkle verification (hasher.rs / stacked_merkle.py) ------------------


def _tree_compress(compressor: Compression, hashes: list[Array]) -> Array:
    """Balanced 2-ary compression of a power-of-two list of digests
    (hasher.rs ``tree_compress``) — the digest a single query's opened rows
    fold to under the query-strided tree."""
    while len(hashes) > 1:
        hashes = [
            compressor.compress(jnp.stack([hashes[2 * i], hashes[2 * i + 1]]))
            for i in range(len(hashes) // 2)
        ]
    return hashes[0]


def _merkle_verify(
    compressor: Compression, root: Array, idx: int, leaf: Array, path: Array
) -> None:
    """Recompute the root from the query layer up and compare (whir.rs
    ``merkle_verify``): sibling order follows the index bits."""
    cur = leaf
    for sibling in path:
        if idx & 1 == 0:
            cur = compressor.compress(jnp.stack([cur, sibling]))
        else:
            cur = compressor.compress(jnp.stack([sibling, cur]))
        idx >>= 1
    if not _eq(cur, root):
        raise VerificationError("Merkle path does not match the committed root")


def _binary_k_fold(values: list[Array], alphas: Sequence[Array], x_root: int) -> Array:
    """``g_k(x^{2^k})`` from codeword values on the coset ``{x·ω^j}`` and the
    fold points (whir.rs ``binary_k_fold``). ``x_root`` is a host int (a power
    of the RS-domain generator)."""
    n = len(values)
    k = len(alphas)
    if n != 1 << k:
        raise VerificationError("binary_k_fold: values length != 2^k")
    omega_k = prism.omega_int(k) if k > 0 else 1
    omega_k_inv = pow(omega_k, MODULUS - 2, MODULUS)
    half = (MODULUS + 1) // 2
    tw = [pow(omega_k, (i % (1 << max(k - 1, 0))), MODULUS) for i in range(1 << max(k - 1, 0))]
    inv_tw = [pow(omega_k_inv, i, MODULUS) for i in range(1 << max(k - 1, 0))]
    x_pow = x_root
    x_inv = pow(x_root, MODULUS - 2, MODULUS)
    x_pow_j = x_pow
    x_inv_j = x_inv
    vals = list(values)
    for j, alpha in enumerate(alphas):
        m = n >> (j + 1)
        new = list(vals)
        for i in range(m):
            t = (tw[(i << j)] * x_pow_j) % MODULUS
            t_inv = (inv_tw[(i << j)] * x_inv_j) % MODULUS
            coef = f_to_ef(f_const((t_inv * half) % MODULUS))
            alpha_minus_t = alpha - f_to_ef(f_const(t))
            new[i] = vals[i] + alpha_minus_t * (vals[i] - vals[m + i]) * coef
        vals = new
        x_pow_j = (x_pow_j * x_pow_j) % MODULUS
        x_inv_j = (x_inv_j * x_inv_j) % MODULUS
    return vals[0]


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


def _verify_zerocheck_and_logup(
    transcript: DuplexTranscript,
    params: SystemParams,
    sorted_vks: Sequence[AirVk],
    gkr_proof: FracSumcheckProof,
    bcp: BatchConstraintProof,
    logup_pow_bits: int,
    logup_pow_witness: Array,
) -> tuple[DuplexTranscript, list[Array]]:
    l_skip = params.l_skip
    n_per_trace = [vk.log_height - l_skip for vk in sorted_vks]

    transcript, ok = check_witness(transcript, logup_pow_bits, logup_pow_witness)
    if not bool(ok):
        raise VerificationError("invalid LogUp PoW witness")
    transcript, alpha = sample_ext(transcript)
    transcript, beta = sample_ext(transcript)

    total_interactions = sum(
        len(vk.dag.interactions) << (l_skip + max(n, 0))
        for vk, n in zip(sorted_vks, n_per_trace)
    )
    n_logup = total_interactions.bit_length() - l_skip if total_interactions else 0

    xi: list[Array] = []
    p_xi = _ZERO
    q_xi = alpha
    if total_interactions > 0:
        transcript, p_xi, q_xi, xi = _verify_gkr(
            transcript, gkr_proof, l_skip + n_logup
        )

    n_max = max(max(n_per_trace), 0)
    n_global = max(n_max, n_logup)
    while len(xi) != l_skip + n_global:
        transcript, extra = sample_ext(transcript)
        xi.append(extra)

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
    max_msg = max((len(i.message) for vk in sorted_vks for i in vk.dag.interactions),
                  default=0)
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
        b = [f_to_ef(f_const((s.row_idx >> j) & 1))
             for j in range(l_skip + n_lift, l_skip + n_stack)]
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
    whir = params.whir
    k_whir = whir.k
    num_rounds = len(whir.num_queries)
    widths = [len(v) for v in stacking_openings]
    m = params.l_skip + params.n_stack

    transcript, ok = check_witness(transcript, whir.mu_pow_bits, proof.mu_pow_witness)
    if not bool(ok):
        raise VerificationError("invalid WHIR μ PoW witness")
    transcript, mu = sample_ext(transcript)

    total_width = sum(widths)
    mu_pows = [_ONE]
    for _ in range(total_width - 1):
        mu_pows.append(mu_pows[-1] * mu)

    claim = _ZERO
    flat_openings = [o for v in stacking_openings for o in v]
    for opening, mu_pow in zip(flat_openings, mu_pows):
        claim = claim + mu_pow * opening

    log_rs = m + params.log_blowup
    sc_idx = 0
    alphas: list[Array] = []
    z0s: list[Array] = []
    zs: list[list[int]] = []
    gammas: list[Array] = []

    for whir_round in range(num_rounds):
        is_final = whir_round == num_rounds - 1
        alphas_round: list[Array] = []
        for _ in range(k_whir):
            evals = proof.whir_sumcheck_polys[sc_idx]
            transcript = transcript.observe(evals)
            transcript, ok = check_witness(
                transcript, whir.folding_pow_bits, proof.folding_pow_witnesses[sc_idx]
            )
            if not bool(ok):
                raise VerificationError("invalid WHIR folding PoW witness")
            transcript, alpha = sample_ext(transcript)
            alphas_round.append(alpha)
            ev0 = claim - evals[0]
            claim = _interp_quadratic_012([ev0, evals[0], evals[1]], alpha)
            sc_idx += 1

        y0 = None
        if is_final:
            transcript = transcript.observe(proof.final_poly)
        else:
            commit = proof.codeword_commits[whir_round]
            transcript = transcript.observe(commit)
            transcript, z0 = sample_ext(transcript)
            z0s.append(z0)
            y0 = proof.ood_values[whir_round]
            transcript = transcript.observe(y0)

        transcript, ok = check_witness(
            transcript, whir.query_phase_pow_bits,
            proof.query_phase_pow_witnesses[whir_round],
        )
        if not bool(ok):
            raise VerificationError("invalid WHIR query-phase PoW witness")

        num_queries = whir.num_queries[whir_round]
        index_bits = log_rs - k_whir
        omega = prism.omega_int(log_rs)
        ys_round: list[Array] = []
        zs_round: list[int] = []
        for q in range(num_queries):
            transcript, index = sample_bits(transcript, index_bits)
            zi_root = pow(omega, index, MODULUS)
            zi = pow(zi_root, 1 << k_whir, MODULUS)
            if whir_round == 0:
                codeword_vals = [_ZERO] * (1 << k_whir)
                mu_iter = 0
                for com_idx in range(len(commitments)):
                    rows = proof.initial_round_opened_rows[com_idx][q]  # (2^k, width)
                    leaf_hashes = [sponge.hash(rows[j]) for j in range(1 << k_whir)]
                    digest = _tree_compress(compressor, leaf_hashes)
                    path = proof.initial_round_merkle_proofs[com_idx][q]
                    _merkle_verify(compressor, commitments[com_idx], index, digest, path)
                    for c in range(widths[com_idx]):
                        mu_pow = mu_pows[mu_iter]
                        mu_iter += 1
                        for j in range(1 << k_whir):
                            codeword_vals[j] = codeword_vals[j] + mu_pow * f_to_ef(rows[j][c])
                yi = _binary_k_fold(codeword_vals, alphas_round, zi_root)
            else:
                vals = proof.codeword_opened_values[whir_round - 1][q]  # (2^k,) EF
                leaf_hashes = [
                    sponge.hash(_ef_to_limbs(vals[j])) for j in range(1 << k_whir)
                ]
                digest = _tree_compress(compressor, leaf_hashes)
                path = proof.codeword_merkle_proofs[whir_round - 1][q]
                _merkle_verify(
                    compressor, proof.codeword_commits[whir_round - 1], index, digest, path
                )
                yi = _binary_k_fold(list(vals), alphas_round, zi_root)
            ys_round.append(yi)
            zs_round.append(zi)

        transcript, gamma = sample_ext(transcript)
        if y0 is not None:
            claim = claim + y0 * gamma
        gpow = gamma * gamma
        for yi in ys_round:
            claim = claim + yi * gpow
            gpow = gpow * gamma
        gammas.append(gamma)
        zs.append(zs_round)
        alphas.extend(alphas_round)
        log_rs -= 1

    if proof.final_poly.shape[0] != 1 << (m - num_rounds * k_whir):
        raise VerificationError("WHIR final poly degree mismatch")

    # Final WHIR constraint.
    t = k_whir * num_rounds
    final_poly = list(proof.final_poly)
    prefix = _eval_mobius_eq_mle(u[:t], alphas[:t])
    suffix = _eval_mle_evals_at_point(final_poly, u[t:])
    acc = prefix * suffix
    j = k_whir
    for i in range(num_rounds):
        gamma = gammas[i]
        alpha_slc = alphas[j:t]
        slc_len = (t - j) + 1
        if i != num_rounds - 1:
            z0 = z0s[i]
            z0_pow = [z0]
            for _ in range(slc_len - 1):
                z0_pow.append(z0_pow[-1] * z0_pow[-1])
            acc = acc + gamma * prism.eval_eq_mle(alpha_slc, z0_pow[:-1]) * _horner(
                final_poly, z0_pow[-1]
            )
        gpow = gamma * gamma
        for zi in zs[i]:
            zi_pow_hi = f_to_ef(f_const(pow(zi, 1 << (slc_len - 1), MODULUS)))
            zi_pow_left = [f_to_ef(f_const(pow(zi, 1 << p, MODULUS))) for p in range(slc_len - 1)]
            acc = acc + gpow * prism.eval_eq_mle(alpha_slc, zi_pow_left) * _horner(
                final_poly, zi_pow_hi
            )
            gpow = gpow * gamma
        j += k_whir

    if not _eq(acc, claim):
        raise VerificationError("WHIR final constraint failed")
    return transcript


def _ef_to_limbs(x: Array) -> Array:
    """The 4 base-field limbs of one EF element (basis-coefficient order)."""
    return lax.bitcast_convert_type(jnp.atleast_1d(x), F)[0]


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
    returns ``None`` if the proof is accepted."""
    l_skip = params.l_skip
    order = sorted(range(len(air_vks)), key=lambda i: (-air_vks[i].log_height, i))
    sorted_vks = [air_vks[i] for i in order]

    # Preamble.
    transcript = transcript.observe(jnp.array(list(vk_pre_hash), dtype=F))
    transcript = transcript.observe(common_main_commit)
    for vk in air_vks:
        meta: list[int] = [] if vk.is_required else [1]
        meta.append(vk.log_height)
        meta.extend(vk.public_values)
        transcript = transcript.observe(jnp.array(meta, dtype=F))

    transcript, r = _verify_zerocheck_and_logup(
        transcript, params, sorted_vks, proof.gkr_proof,
        proof.batch_constraint_proof, params.logup_pow_bits,
        proof.logup_pow_witness,
    )

    layout = StackedLayout.new(
        l_skip, l_skip + params.n_stack,
        [(vk.width, vk.log_height) for vk in sorted_vks],
    )
    need_rot = [vk.needs_next for vk in sorted_vks]
    transcript, u = _verify_stacked_reduction(
        transcript, params, proof.stacking_proof, layout, need_rot,
        proof.batch_constraint_proof.column_openings, r,
    )

    u_cube = [u[0]]
    for _ in range(l_skip - 1):
        u_cube.append(u_cube[-1] * u_cube[-1])
    u_cube.extend(u[1:])

    _verify_whir(
        transcript, sponge, compressor, params, proof.whir_proof,
        proof.stacking_proof.stacking_openings, [common_main_commit], u_cube,
    )
