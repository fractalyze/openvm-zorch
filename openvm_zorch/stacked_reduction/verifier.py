"""Stage-4 verifier: the dual of ``StackingRound`` (verifier/stacked_reduction.rs).

``verify_stacked_reduction`` re-derives λ, checks s₀ against the λ-batched
opening claims, replays the quadratic sumcheck, and closes on the
stacking-opening claim via the per-column eq/κ_rot prism kernels — the stage
math only. The chain Round that drives it (``StackingVerifierRound``) lives
with the other stage duals in ``openvm_zorch/verify.py``.
"""

from __future__ import annotations

from typing import Sequence

import frx.numpy as jnp
from frx import Array

from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.fields import f_const, f_to_ef
from openvm_zorch.logup_zerocheck import prism
from openvm_zorch.logup_zerocheck.verifier import by_rot
from openvm_zorch.poly_common import (
    ONE,
    ZERO,
    VerificationError,
    eq_u32,
    exp_pow_2,
    interp_quadratic_012,
)
from openvm_zorch.stacked_reduction.prover import StackingProof
from openvm_zorch.transcript import sample_ext
from zorch.poly.univariate import eval_coeffs
from zorch.transcript import DuplexTranscript


def verify_stacked_reduction(
    transcript: DuplexTranscript,
    l_skip: int,
    n_stack: int,
    proof: StackingProof,
    layout: StackedLayout,
    need_rot: Sequence[bool],
    column_openings: Sequence[Sequence[Array]],
    r: Sequence[Array],
) -> tuple[DuplexTranscript, list[Array]]:
    size = 1 << l_skip

    # Order the opening claims exactly as the prover batched them (per commit,
    # per column in layout order); common main only here.
    lambda_count = len(layout.sorted_cols)
    t_claims: list[tuple[Array, Array]] = []
    for trace_idx, vk_need_rot in enumerate(need_rot):
        t_claims.extend(by_rot(column_openings[trace_idx][0], vk_need_rot))
    if len(t_claims) != lambda_count:
        raise VerificationError("Stage-4 opening-claim count mismatch")

    transcript, lam = sample_ext(transcript)
    lam_sqr = lam * lam
    lam_sqr_pows = [ONE]
    for _ in range(lambda_count - 1):
        lam_sqr_pows.append(lam_sqr_pows[-1] * lam_sqr)

    s_0 = ZERO
    for (t0, t1), lam_i in zip(t_claims, lam_sqr_pows):
        s_0 = s_0 + (t0 + t1 * lam) * lam_i
    coeffs = proof.univariate_round_coeffs
    s0_sum = ZERO
    for j in range(0, coeffs.shape[0], size):
        s0_sum = s0_sum + coeffs[j]
    s0_sum = s0_sum * f_to_ef(f_const(size))
    if not eq_u32(s_0, s0_sum):
        raise VerificationError("Stage-4 s0 mismatch")
    transcript = transcript.observe(coeffs)

    u = [None] * (n_stack + 1)
    transcript, u[0] = sample_ext(transcript)
    claim = eval_coeffs(coeffs, u[0])
    for j in range(1, n_stack + 1):
        s_j_1 = proof.sumcheck_round_polys[j - 1][0]
        s_j_2 = proof.sumcheck_round_polys[j - 1][1]
        transcript = transcript.observe(jnp.stack([s_j_1, s_j_2]))
        transcript, u[j] = sample_ext(transcript)
        s_j_0 = claim - s_j_1
        claim = interp_quadratic_012([s_j_0, s_j_1, s_j_2], u[j])

    # Final: reconstruct the per-column kernel coefficients and close on the
    # stacking-opening claim.
    openings = proof.stacking_openings  # [commit][col]
    q_coeffs = [[ZERO for _ in openings[c]] for c in range(len(openings))]
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
            rs_n = [exp_pow_2(r[0], -n)]
        else:
            l_eff = l_skip
            rs_n = list(r[: n_lift + 1])
        eq_prism = prism.eval_eq_prism(l_eff, u[: n_lift + 1], rs_n)
        batched = lam_sqr_pows[lambda_idx] * eq_prism
        if col_need_rot:
            rot = prism.eval_rot_kernel_prism(l_eff, u[: n_lift + 1], rs_n)
            batched = batched + lam_sqr_pows[lambda_idx] * lam * rot
        # Commit 0 (the common main) is the only commitment in scope — the
        # layout covers it alone; ``_mat_idx`` is the trace index within the
        # layout, not a commit index.
        q_coeffs[0][s.col_idx] = q_coeffs[0][s.col_idx] + eq_mle * batched * ind
        lambda_idx += 1

    final_sum = ZERO
    for c in range(len(openings)):
        for col in range(len(openings[c])):
            q_j = openings[c][col]
            transcript = transcript.observe(q_j)
            final_sum = final_sum + q_coeffs[c][col] * q_j
    if not eq_u32(claim, final_sum):
        raise VerificationError("Stage-4 final sum mismatch")

    return transcript, [u[j] for j in range(n_stack + 1)]
