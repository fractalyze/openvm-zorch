"""Stage-3 verifier: the dual of ``ZeroCheckStage`` (verifier/zerocheck.rs).

``verify_zerocheck_stage`` replays the batched ZeroCheck + LogUp sumcheck and
closes it by re-evaluating the constraint/interaction claim at the folded
point from the proof's column openings (the ``VerifierConstraintEvaluator``
analogue) — the stage math only. The chain Stage that drives it
(``ZeroCheckVerifierStage``) lives with the other stage duals in
``openvm_zorch/verify.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import frx.numpy as fnp
from frx import Array

from openvm_zorch.fields import MODULUS, f_const, f_inv_const, f_to_ef
from openvm_zorch.logup_zerocheck import prism
from openvm_zorch.logup_zerocheck.constraints import (
    acc_constraints,
    acc_interactions,
    eval_nodes,
)
from openvm_zorch.logup_zerocheck.prover import BatchConstraintProof
from openvm_zorch.poly_common import (
    ONE,
    ZERO,
    VerificationError,
    eq_u32,
    exp_pow_2,
    interp_at_nodes,
    progression_exp_2,
)
from openvm_zorch.transcript import sample_ext
from zorch.poly.univariate import eval_coeffs
from zorch.transcript import DuplexTranscript

if TYPE_CHECKING:
    from openvm_zorch.verify import AirVk


def by_rot(flat: Array, need_rot: bool) -> list[tuple[Array, Array]]:
    """The proof's flat per-trace column openings as (claim, claim_rot) pairs —
    rot is 0 when the AIR never rotates. Stage 4 batches the same pairs."""
    if need_rot:
        return [(flat[2 * i], flat[2 * i + 1]) for i in range(flat.shape[0] // 2)]
    return [(flat[i], ZERO) for i in range(flat.shape[0])]


def verify_zerocheck_stage(
    transcript: DuplexTranscript,
    l_skip: int,
    max_constraint_degree: int,
    sorted_vks: Sequence["AirVk"],
    n_logup: int,
    n_max: int,
    bcp: BatchConstraintProof,
    alpha: Array,
    beta: Array,
    xi: list[Array],
    p_xi: Array,
    q_xi: Array,
) -> tuple[DuplexTranscript, list[Array]]:
    """Stage 3 verifier — the dual of ``ZeroCheckStage``: the batched ZeroCheck
    + LogUp sumcheck. Consumes the Stage-2 outputs off the carry (α/β, the
    padded point ξ, and the GKR claims ``p_xi`` / ``q_xi``), re-evaluates the
    constraint/interaction claim at the folded point from the proof's column
    openings, and returns the sumcheck point ``r``."""
    n_per_trace = [vk.log_height - l_skip for vk in sorted_vks]

    transcript, lam = sample_ext(transcript)

    # 3. Observe per-air sum claims; reduce GKR claims to zero / alpha.
    for sum_p, sum_q in zip(bcp.numerator_term_per_air, bcp.denominator_term_per_air):
        p_xi = p_xi - sum_p
        q_xi = q_xi - sum_q
        transcript = transcript.observe(fnp.stack([sum_p, sum_q]))
    if not eq_u32(p_xi, ZERO):
        raise VerificationError("GKR numerator claim mismatch")
    if not eq_u32(q_xi, alpha):
        raise VerificationError("GKR denominator claim mismatch")

    transcript, mu = sample_ext(transcript)
    sum_claim = ZERO
    cur_mu = ONE
    for sum_p, sum_q in zip(bcp.numerator_term_per_air, bcp.denominator_term_per_air):
        sum_claim = sum_claim + sum_p * cur_mu
        cur_mu = cur_mu * mu
        sum_claim = sum_claim + sum_q * cur_mu
        cur_mu = cur_mu * mu

    # 5. Univariate round 0.
    s0 = bcp.univariate_round_coeffs
    transcript = transcript.observe(fnp.stack(list(s0)))
    s_deg = max_constraint_degree + 1
    transcript, r_0 = sample_ext(transcript)
    size = 1 << l_skip
    s0_sum = ZERO
    for j in range(0, len(s0), size):
        s0_sum = s0_sum + s0[j]
    s0_sum = s0_sum * f_to_ef(f_const(size))
    if not eq_u32(sum_claim, s0_sum):
        raise VerificationError("Stage-3 s0 sum mismatch")
    cur_sum = eval_coeffs(fnp.stack(list(s0)), r_0)
    rs = [r_0]

    # 6. Multilinear rounds.
    for round_ in range(n_max):
        evals = bcp.sumcheck_round_polys[round_]  # (s_deg,) at {1..s_deg}
        transcript = transcript.observe(evals)
        s_1 = evals[0]
        s_0v = cur_sum - s_1
        full = [s_0v] + [evals[i] for i in range(s_deg)]
        transcript, r = sample_ext(transcript)
        cur_sum = interp_at_nodes(full, r, s_deg)
        rs.append(r)

    # Observe the column openings (common main, per trace, (claim, claim_rot)
    # pairs — rot is 0 when the AIR never rotates), matching the prover's
    # closing observes so Stage 4 continues from the same transcript state.
    for trace_idx, vk in enumerate(sorted_vks):
        pairs = by_rot(bcp.column_openings[trace_idx][0], vk.needs_next)
        flat = fnp.stack([v for pair in pairs for v in pair])
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
    eq_ns = [ONE] * (n_max + 1)
    eq_sharp_ns = [ONE] * (n_max + 1)
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
    beta_pows = [ONE]
    max_msg = max(
        (len(i.message) for vk in sorted_vks for i in vk.dag.interactions), default=0
    )
    for _ in range(max_msg + 1):
        beta_pows.append(beta_pows[-1] * beta)
    lambda_pows = [ONE]
    max_constraints = max((len(vk.dag.constraint_idx) for vk in sorted_vks), default=1)
    for _ in range(max(max_constraints, 1) - 1):
        lambda_pows.append(lambda_pows[-1] * lam)

    interactions_evals: list[Array] = []
    constraints_evals: list[Array] = []
    for trace_idx, vk in enumerate(sorted_vks):
        n = n_per_trace[trace_idx]
        n_lift = max(n, 0)
        pairs = by_rot(bcp.column_openings[trace_idx][0], vk.needs_next)
        local = fnp.stack([c for c, _ in pairs])
        nxt = fnp.stack([c_rot for _, c_rot in pairs])
        parts = [(local, nxt)]

        if n < 0:
            l_eff = l_skip + n
            rs_n = [exp_pow_2(rs[0], -n)]
            norm = f_to_ef(f_inv_const(pow(2, -n, MODULUS)))
        else:
            l_eff = l_skip
            rs_n = rs[: n + 1]
            norm = ONE
        omega = f_to_ef(f_const(prism.omega_int(l_eff)))
        inv = f_to_ef(f_inv_const(1 << l_eff))
        prod_lo = ONE
        prod_hi = ONE
        for x in rs_n[1:]:
            prod_lo = prod_lo * (ONE - x)
            prod_hi = prod_hi * x
        is_first = inv * progression_exp_2(rs_n[0], l_eff) * prod_lo
        is_last = inv * progression_exp_2(rs_n[0] * omega, l_eff) * prod_hi
        sels = fnp.stack([is_first, ONE - is_last, is_last])

        node_vals = eval_nodes(vk.dag, sels, parts, vk.public_values)
        expr = acc_constraints(vk.dag, node_vals, lambda_pows)
        constraints_evals.append(eq_ns[n_lift] * expr)

        num, denom = acc_interactions(
            vk.dag, node_vals, beta_pows, eq_3b_per_trace[trace_idx]
        )
        interactions_evals.append(num * norm * eq_sharp_ns[n_lift])
        interactions_evals.append(denom * eq_sharp_ns[n_lift])

    evaluated = ZERO
    cur_mu = ONE
    for x in interactions_evals + constraints_evals:
        evaluated = evaluated + x * cur_mu
        cur_mu = cur_mu * mu
    if not eq_u32(cur_sum, evaluated):
        raise VerificationError("Stage-3 final claim mismatch")

    return transcript, rs
