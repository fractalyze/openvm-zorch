"""SWIRL Stage 3: batched ZeroCheck + LogUp constraint sumcheck.

The second half of the reference's ``prove_zerocheck_and_logup``
(logup_zerocheck/mod.rs after the ξ padding): one univariate-skip round over
the prism, then ``n_max`` front-loaded MLE rounds batching, per sorted trace,
the zerocheck polynomial and the two logup polynomials under μ-powers, and
finally the column openings at the bound point.

Structure per the reference (driver mod.rs, per-trace math cpu.rs):

- round 0 interpolates each per-trace ``s'_0`` from evaluations on geometric
  cosets ``g^{c+1}·D`` (the zerocheck variant divides by the zerofier and
  re-multiplies in coefficient form), multiplies in the ``eq_D``/``eq♯_D``
  univariate factors, and μ-batches into one ``s_0`` sent in COEFFICIENT
  form; the logup sum claims are read off the product coefficients
  (``Σ_D Z^j = N`` iff ``N | j``).
- MLE rounds send evaluations on ``{1..s_deg}``; ``s'(0)`` is never computed
  directly — the verifier (and prover) derive it from
  ``s_j(0) + s_j(1) = s_{j-1}(r_{j-1})``. Traces exhausted by front-loading
  (``round > ñ_T``) contribute a constant "tilde" term times
  ``r_{j-1}·…`` (cpu.rs ``sumcheck_polys_eval``); the eq(ξ, X) linear factor
  multiplies in coefficient form at the end.
- polynomial helpers stay as Python lists of scalar arrays — every degree
  here is ≤ ``s_0_deg`` (13 coefficients), so unrolled exact arithmetic
  beats array plumbing.

zorch reuse: ``lift_to_domain``/``fold_pair`` (LSB pairing — identical to the
reference's MLE fold), ``expand_eq_to_hypercube`` (fed reversed ξ slices, the
Stage-2 convention), ``eval_eq``, ``compute_inv_vandermonde``, ``eval_coeffs``
(coefficient-form univariate eval, O(1) graph in degree). The
prismalinear/univariate-skip pieces live in ``prism.py``; the constraint DAG
evaluator in ``constraints.py``.

Reference:
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/logup_zerocheck/mod.rs#L184
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/logup_zerocheck/cpu.rs
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax

from openvm_zorch.fields import EF, F, f_const, f_inv_const, f_to_ef
from openvm_zorch.logup_gkr.input_layer import interactions_layout
from openvm_zorch.logup_zerocheck import prism
from openvm_zorch.logup_zerocheck.constraints import (
    ConstraintsDag,
    acc_constraints,
    acc_interactions,
    eval_nodes,
)
from openvm_zorch.transcript import sample_ext
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde, eval_coeffs
from zorch.sumcheck.prover import fold_pair, lift_to_domain
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class AirData:
    """One present AIR, in sorted (stacking) order."""

    trace: Array  # (height, width) base field
    dag: ConstraintsDag
    public_values: tuple[int, ...]
    constraint_degree: int  # this AIR's vk.max_constraint_degree
    needs_next: bool


@dataclass(frozen=True)
class BatchConstraintProof:
    """The reference ``BatchConstraintProof`` plus the sampled challenges
    (λ, μ, r) for transcript-trajectory comparison in tests."""

    numerator_term_per_air: list[Array]
    denominator_term_per_air: list[Array]
    univariate_round_coeffs: list[Array]
    sumcheck_round_polys: list[Array]  # per round, (s_deg,) evals on {1..s_deg}
    column_openings: list[list[Array]]  # per AIR, per part, flat
    lambda_: Array
    mu: Array
    r: list[Array]


def _powers(x: Array, n: int) -> list[Array]:
    out = [f_to_ef(jnp.ones((), F))]
    for _ in range(n - 1):
        out.append(out[-1] * x)
    return out


def _row0(a: Array) -> Array:
    """The first hypercube cell of a ``(s_deg, half)`` round-eval array — the
    fully-folded ``f̂(r⃗)`` once the buffer is frozen. A trace with no zerocheck
    constraints (resp. no interactions) accumulates to a 0-d zero instead, which
    is already that cell's value, so pass it through."""
    return a[0, 0] if a.ndim == 2 else a


def _batched_conv(coeffs: Array, kernel: Array) -> Array:
    """Convolve every row of ``coeffs`` ``(..., La)`` with the shared
    ``kernel`` ``(Lb,)``, returning ``(..., La + Lb - 1)``.

    The batched array form of a schoolbook polynomial convolution for a shared
    short kernel: shift the rows by each kernel tap (static loop over ``Lb``),
    stack the shifts on a trailing axis, broadcast-multiply by ``kernel`` and
    reduce that last axis. Keeping the contracted axis last avoids the mid-axis
    EF reduce fault, and the shift-and-add avoids ``jnp.dot``/``@`` (both
    mis-lower on this fork — see CLAUDE.md). Dispatch-free and jit-fusable,
    unlike a per-scalar coefficient loop. (``conv_test`` pins it against the
    reference scalar convolution.)"""
    la = coeffs.shape[-1]
    lb = kernel.shape[-1]
    lo = la + lb - 1
    lead = coeffs.shape[:-1]
    shifts = [
        jnp.pad(coeffs, ((0, 0),) * len(lead) + ((j, lo - la - j),))
        for j in range(lb)
    ]
    stacked = jnp.stack(shifts, axis=-1)  # (..., lo, lb)
    return (stacked * kernel).sum(axis=-1)


def _pad(coeffs: list[Array], n: int) -> list[Array]:
    """Truncate/zero-pad to length n (the reference's `.take(n)` +
    `unwrap_or(ZERO)` reads; trailing real coefficients past n are zero)."""
    out = list(coeffs[:n])
    while len(out) < n:
        out.append(jnp.zeros((), EF))
    return out


def _eq_table(xi: list[Array]) -> Array:
    """eq(ξ, y) on the hypercube, LSB-first in ξ (Stage-2 convention: the
    MSB-first expand gets the slice reversed)."""
    if not xi:
        return jnp.ones((1,), EF)
    return expand_eq_to_hypercube(jnp.stack(xi[::-1]), jnp.ones((), EF))


def _lift(mat: Array, l_skip: int) -> Array:
    """Cyclic lift of a short trace to height ``2^l_skip`` (the reference's
    ``% height`` indexing)."""
    height = mat.shape[0]
    if height >= 1 << l_skip:
        return mat
    return jnp.tile(mat, ((1 << l_skip) // height, 1))


def _sels(height: int, l_skip: int) -> Array:
    """The lift of [is_first_row, is_transition, is_last_row] (cpu.rs
    ``sels_per_trace_base``)."""
    lifted = max(height, 1 << l_skip)
    rows = jnp.arange(lifted) % height
    table = jnp.stack(
        [rows == 0, rows != height - 1, rows == height - 1], axis=-1
    )
    return table.astype(jnp.uint32).astype(F)


def _view_mats(air: AirData, l_skip: int) -> list[Array]:
    """The (local, rot) matrix list of single.rs ``view_mats`` — common main
    only (no preprocessed/cached traces in scope), lifted."""
    local = _lift(air.trace, l_skip)
    if air.needs_next:
        rot = jnp.concatenate([air.trace[1:], air.trace[:1]], axis=0)
        return [local, _lift(rot, l_skip)]
    return [local]


def _dag_parts(mats: list[Array], needs_next: bool) -> list[tuple[Array, Array | None]]:
    """Group a flat (local, rot, local, rot, ...) mat list into the DAG
    evaluator's (local, next) pairs."""
    if needs_next:
        return [(mats[i], mats[i + 1]) for i in range(0, len(mats), 2)]
    return [(m, None) for m in mats]


def _inv_vandermonde_rows(degree: int) -> list[list[Array]]:
    m = compute_inv_vandermonde(degree, F)
    return [
        [f_to_ef(m[i, j]) for j in range(degree + 1)] for i in range(degree + 1)
    ]


def prove_batch_constraints(
    transcript: DuplexTranscript,
    l_skip: int,
    n_logup: int,
    airs: list[AirData],
    xi: list[Array],
    beta: Array,
    max_constraint_degree: int,
) -> tuple[DuplexTranscript, BatchConstraintProof]:
    """Drive Stage 3 from the post-ξ transcript state; byte-matches
    ``prove_zerocheck_and_logup`` after the ξ padding."""
    num_traces = len(airs)
    n_per_trace = [
        log2_strict_usize(air.trace.shape[0]) - l_skip for air in airs
    ]
    n_max = max(max(n_per_trace), 0)
    s_deg = max_constraint_degree + 1
    sp_0_deg = max_constraint_degree * ((1 << l_skip) - 1)

    zero = jnp.zeros((), EF)
    one_ef = f_to_ef(jnp.ones((), F))

    # --- Per-trace inputs: lifted mats, selectors, eq(ξ_3, b) weights ---
    mats = [_view_mats(air, l_skip) for air in airs]
    sels = [_sels(air.trace.shape[0], l_skip) for air in airs]
    max_msg_len = max(
        (len(i.message) for air in airs for i in air.dag.interactions), default=0
    )
    beta_pows = _powers(beta, max_msg_len + 1)

    sorted_meta = [
        (len(air.dag.interactions), max(n + l_skip, l_skip))
        for air, n in zip(airs, n_per_trace)
    ]
    layout = interactions_layout(l_skip, n_logup, sorted_meta)
    eq_3bs: list[list[Array]] = [
        [zero] * len(air.dag.interactions) for air in airs
    ]
    for trace_idx, int_idx, s in layout.sorted_cols:
        n_lift = max(n_per_trace[trace_idx], 0)
        b_int = s.row_idx >> (l_skip + n_lift)
        n_bits = n_logup - n_lift
        if n_bits == 0:
            eq_3bs[trace_idx][int_idx] = one_ef
            continue
        bits = f_to_ef(
            jnp.array([(b_int >> j) & 1 for j in range(n_bits)], F)
        )
        point = jnp.stack(xi[l_skip + n_lift : l_skip + n_logup])
        eq_3bs[trace_idx][int_idx] = eval_eq(point, bits)

    # --- Batching randomness λ ---
    transcript, lam = sample_ext(transcript)
    max_num_constraints = max(
        (len(air.dag.constraint_idx) for air in airs), default=0
    )
    lambda_pows = _powers(lam, max(max_num_constraints, 1))

    # --- Round 0: per-trace s'_0 polynomials on geometric cosets ---
    sp_zc: list[list[Array]] = []
    sp_logup: list[tuple[list[Array], list[Array]]] = []
    for t, (air, n, trace_mats, trace_sels) in enumerate(
        zip(airs, n_per_trace, mats, sels)
    ):
        n_lift = max(n, 0)
        eq_xi = _eq_table(xi[l_skip : l_skip + n_lift])
        norm = f_inv_const(1 << max(-n, 0))

        def cells_for(num_cosets: int) -> tuple[Array, list[tuple[Array, Array | None]]]:
            sels_cells = prism.coset_evals(l_skip, trace_sels, num_cosets)
            mat_cells = [prism.coset_evals(l_skip, m, num_cosets) for m in trace_mats]
            return sels_cells, _dag_parts(mat_cells, air.needs_next)

        # Zerocheck: q = s'_0 / (Z^N - 1) from constraint_degree - 1 cosets.
        num_cosets = air.constraint_degree - 1
        if num_cosets == 0:
            sp_zc.append([])
        else:
            sels_cells, parts = cells_for(num_cosets)
            node_vals = eval_nodes(air.dag, sels_cells, parts, air.public_values)
            acc = acc_constraints(air.dag, node_vals, lambda_pows)
            weighted = acc * eq_xi[None, None, :]
            q_evals = []
            for c in range(num_cosets):
                zerofier = pow(prism.GENERATOR, (c + 1) << l_skip, prism.MODULUS) - 1
                inv_zerofier = f_to_ef(f_inv_const(zerofier))
                q_evals.append(weighted[c].sum(axis=1) * inv_zerofier)
            q = prism.geometric_cosets_to_coeffs(
                l_skip, jnp.stack(q_evals), num_cosets
            )
            air_sp_0_deg = air.constraint_degree * ((1 << l_skip) - 1)
            q_padded = _pad(q, air_sp_0_deg + 1)
            coeffs = []
            for i in range(air_sp_0_deg + 1):
                c_i = -q_padded[i]
                if i >= 1 << l_skip:
                    c_i = c_i + q_padded[i - (1 << l_skip)]
                coeffs.append(c_i)
            sp_zc.append(coeffs)

        # LogUp: s'_p, s'_q interpolated directly from constraint_degree cosets.
        if not air.dag.interactions:
            sp_logup.append(([], []))
            continue
        sels_cells, parts = cells_for(air.constraint_degree)
        node_vals = eval_nodes(air.dag, sels_cells, parts, air.public_values)
        numer, denom = acc_interactions(air.dag, node_vals, beta_pows, eq_3bs[t])
        p_evals = (numer * eq_xi[None, None, :]).sum(axis=2)
        q_evals = (denom * eq_xi[None, None, :]).sum(axis=2)
        p_coeffs = prism.geometric_cosets_to_coeffs(
            l_skip, p_evals, air.constraint_degree
        )
        q_coeffs = prism.geometric_cosets_to_coeffs(
            l_skip, q_evals, air.constraint_degree
        )
        sp_logup.append(([c * norm for c in p_coeffs], q_coeffs))

    # --- eq♯/eq univariate factors, μ batching, sum claims, s_0 ---
    # The per-trace s'_p/s'_q · eq♯ products and the μ-batched s_0 are degree
    # ≤ s_0_deg (≤ 13–61 coeffs) but there are 2·num_traces of them — a scalar
    # _conv storm under eager dispatch. Stack the per-trace coefficient rows
    # and convolve them in one batched array op instead (issue #3); the eq♯/eq
    # kernels are exactly 1<<l_skip long, so the conv lands at s_0_deg+1 with
    # no padding. Two islands, split by the observe(claims)→sample(μ) seam.
    skip = 1 << l_skip
    eq_sharp = jnp.stack(prism.eq_sharp_uni_poly(l_skip, xi[:l_skip]))
    skip_domain_size = f_to_ef(f_const(skip))

    # Island A (pre-μ): per-trace logup products + sum claims.
    sp_p = jnp.stack([jnp.stack(_pad(p, sp_0_deg + 1)) for p, _ in sp_logup])
    sp_q = jnp.stack([jnp.stack(_pad(q, sp_0_deg + 1)) for _, q in sp_logup])
    p_prods = _batched_conv(sp_p, eq_sharp)  # (num_traces, s_0_deg+1)
    q_prods = _batched_conv(sp_q, eq_sharp)
    # Σ_D Z^j = N iff N | j: read the sum claim off the strided coefficients.
    p_claims = p_prods[:, ::skip].sum(axis=-1) * skip_domain_size
    q_claims = q_prods[:, ::skip].sum(axis=-1) * skip_domain_size

    numerator_term_per_air = []
    denominator_term_per_air = []
    for t in range(num_traces):
        transcript = transcript.observe(jnp.stack([p_claims[t], q_claims[t]]))
        numerator_term_per_air.append(p_claims[t])
        denominator_term_per_air.append(q_claims[t])

    transcript, mu = sample_ext(transcript)
    mu_pows = _powers(mu, 3 * num_traces)

    # Island B (post-μ): μ-batch the zerocheck rows, multiply in eq_D, then add
    # the μ-weighted logup products to form s_0. Contracted axes kept last so
    # the EF reduce stays jit-safe (CLAUDE.md).
    eq_uni = jnp.stack(prism.eq_uni_poly(l_skip, xi[0]))
    sp_zc_rows = jnp.stack(
        [jnp.stack(_pad(coeffs, sp_0_deg + 1)) for coeffs in sp_zc]
    )  # (num_traces, sp_0_deg+1)
    zc_weights = jnp.stack(mu_pows[2 * num_traces : 3 * num_traces])
    zc_batched = (sp_zc_rows.T * zc_weights).sum(axis=-1)  # (sp_0_deg+1,)
    zc_prod = _batched_conv(zc_batched, eq_uni)  # (s_0_deg+1,)

    mu_p = jnp.stack(mu_pows[0 : 2 * num_traces : 2])
    mu_q = jnp.stack(mu_pows[1 : 2 * num_traces : 2])
    s_0_arr = (
        zc_prod + (p_prods.T * mu_p).sum(axis=-1) + (q_prods.T * mu_q).sum(axis=-1)
    )
    transcript = transcript.observe(s_0_arr)
    s_0 = list(s_0_arr)  # the proof field wants list[Array]

    transcript, r_0 = sample_ext(transcript)
    r = [r_0]
    prev_s_eval = eval_coeffs(s_0_arr, r_0)

    # --- Fold the prism at r_0 ---
    mats = [
        [prism.fold_ple_evals(l_skip, m, r_0) for m in trace_mats]
        for trace_mats in mats
    ]
    sels = [prism.fold_ple_evals(l_skip, s, r_0) for s in sels]
    eq_ns = [prism.eval_eq_uni(l_skip, xi[0], r_0)]
    eq_sharp_ns = [prism.eval_eq_sharp_uni(l_skip, xi[:l_skip], r_0)]

    # --- MLE rounds 1..=n_max, as one lax.scan (issue #33) -------------------
    # Eager, the round loop folds the hypercube to half its size each round, so
    # every round is a fresh XLA shape → ~726 one-shot recompiles at 2^16 (#26).
    # Wrapping the unrolled loop in one jit instead *regressed* compile time (one
    # giant module; #33). The fix is a `lax.scan` whose body compiles once,
    # independent of n_max: each trace keeps a fixed-width buffer and stride-pair
    # folds in place, re-padding the dead tail with zeros (mirroring
    # `zorch.sumcheck._prove_scan`'s fixed-width + dynamic-window scheme, but with
    # the reference's LSB-stride pairing, not high/low halves). Front-load
    # exhaustion (round > ñ_t, ñ_t static) becomes a per-trace `jnp.where` on the
    # dynamic scan index instead of a Python branch. The round math is unchanged
    # from the eager body — only the loop carrier moved into the scan carry.
    inv_vdm = _inv_vandermonde_rows(s_deg - 1)
    domain_pts = jnp.stack(
        [f_to_ef(f_const(i)) for i in range(1, s_deg + 1)]
    )  # the {1..s_deg} round-poly sample points
    n_lifts = [max(n, 0) for n in n_per_trace]
    norms = [f_to_ef(f_inv_const(1 << max(-n, 0))) for n in n_per_trace]

    # Per-round, per-trace eq(ξ, ·) weight tables, padded to H_t/2 = 2^(ñ_t-1)
    # and stacked over the n_max scan steps; the live sum `(acc·eq_xi).sum` reads
    # these so the dead lanes (zero) need no separate live mask. Traces with no
    # MLE round (ñ_t = 0) are pure-tilde and carry a 1-wide placeholder.
    eq_xi_xs: list[Array] = []
    for t, n_lift in enumerate(n_lifts):
        half_t = 1 << (n_lift - 1) if n_lift >= 1 else 1
        rows = []
        for round_ in range(1, n_max + 1):
            if n_lift >= 1 and round_ <= n_lift:
                tab = _eq_table(xi[l_skip + round_ : l_skip + n_lift])
                rows.append(
                    jnp.concatenate([tab, jnp.zeros(half_t - tab.shape[0], EF)])
                )
            else:
                rows.append(jnp.zeros(half_t, EF))
        eq_xi_xs.append(  # (n_max, H_t/2)
            jnp.stack(rows) if rows else jnp.zeros((0, half_t), EF)
        )

    xi_cur_xs = (
        jnp.stack([xi[l_skip + round_ - 1] for round_ in range(1, n_max + 1)])
        if n_max >= 1
        else jnp.zeros((0,), EF)
    )

    def step(carry, xs):
        (
            bufs_sels,
            bufs_mats,
            tilde_zc,
            tilde_p,
            tilde_q,
            eq_n,
            eq_sharp_n,
            r_prev,
            transcript,
            prev_s_eval,
            round_idx,
        ) = carry
        eq_xi_row, xi_cur = xs

        sp_head_zc = [zero] * (s_deg - 1)
        sp_head_logup = [zero] * (s_deg - 1)
        sp_tail = zero
        new_tilde_zc = list(tilde_zc)
        new_tilde_p = list(tilde_p)
        new_tilde_q = list(tilde_q)

        for t, (air, n_lift) in enumerate(zip(airs, n_lifts)):
            norm = norms[t]
            mu_zc = mu_pows[2 * num_traces + t]
            mu_p, mu_q = mu_pows[2 * t], mu_pows[2 * t + 1]
            is_head = round_idx <= n_lift  # live this round ⇔ in the head

            if n_lift >= 1:
                # Live evals on {1..s_deg-1}; the body runs every round but its
                # head contribution is gated to round ≤ ñ_t. The fully-folded
                # f̂(r⃗) lands at acc[0,0] once the buffer is frozen (below), so
                # the tilde base reuses it — no second eval_nodes.
                eq_xi = eq_xi_row[t]
                sels_dom = lift_to_domain(
                    bufs_sels[t][0::2], bufs_sels[t][1::2], s_deg - 1
                )
                mats_dom = [
                    lift_to_domain(m[0::2], m[1::2], s_deg - 1) for m in bufs_mats[t]
                ]
                node_vals = eval_nodes(
                    air.dag,
                    sels_dom,
                    _dag_parts(mats_dom, air.needs_next),
                    air.public_values,
                )
                acc = acc_constraints(air.dag, node_vals, lambda_pows)
                zc = (acc * eq_xi[None, :]).sum(axis=1)
                zc0 = eq_n * _row0(acc)
                if air.dag.interactions:
                    numer, denom = acc_interactions(
                        air.dag, node_vals, beta_pows, eq_3bs[t]
                    )
                    p = (numer * eq_xi[None, :]).sum(axis=1) * norm
                    q = (denom * eq_xi[None, :]).sum(axis=1)
                    p0t = eq_sharp_n * _row0(numer) * norm
                    q0t = eq_sharp_n * _row0(denom)
                else:
                    p = jnp.zeros((s_deg,), EF)
                    q = jnp.zeros((s_deg,), EF)
                    p0t = zero
                    q0t = zero
                for i in range(s_deg - 1):
                    sp_head_zc[i] = sp_head_zc[i] + jnp.where(
                        is_head, mu_zc * zc[i + 1], zero
                    )
                    sp_head_logup[i] = sp_head_logup[i] + jnp.where(
                        is_head, mu_p * p[i + 1] + mu_q * q[i + 1], zero
                    )
            else:
                # Pure-tilde trace (height 1): eval over its single row.
                node0 = eval_nodes(
                    air.dag,
                    bufs_sels[t][0],
                    _dag_parts([m[0] for m in bufs_mats[t]], air.needs_next),
                    air.public_values,
                )
                zc0 = eq_n * acc_constraints(air.dag, node0, lambda_pows)
                if air.dag.interactions:
                    numer0, denom0 = acc_interactions(
                        air.dag, node0, beta_pows, eq_3bs[t]
                    )
                    p0t = eq_sharp_n * numer0 * norm
                    q0t = eq_sharp_n * denom0
                else:
                    p0t = zero
                    q0t = zero

            # tilde carry: init f̂-term at round ñ_t+1, then ×r each later round.
            is_init = round_idx == n_lift + 1
            is_accum = round_idx > n_lift + 1
            new_tilde_zc[t] = jnp.where(
                is_init, zc0, jnp.where(is_accum, tilde_zc[t] * r_prev, tilde_zc[t])
            )
            new_tilde_p[t] = jnp.where(
                is_init, p0t, jnp.where(is_accum, tilde_p[t] * r_prev, tilde_p[t])
            )
            new_tilde_q[t] = jnp.where(
                is_init, q0t, jnp.where(is_accum, tilde_q[t] * r_prev, tilde_q[t])
            )
            tail_term = (
                mu_zc * new_tilde_zc[t]
                + mu_p * new_tilde_p[t]
                + mu_q * new_tilde_q[t]
            )
            sp_tail = sp_tail + jnp.where(is_head, zero, tail_term)

        # s'(0) from s_j(0) + s_j(1) = s_{j-1}(r_{j-1}).
        sp_head_evals = [zero] * s_deg
        for i in range(s_deg - 1):
            sp_head_evals[i + 1] = (
                eq_n * sp_head_zc[i] + eq_sharp_n * sp_head_logup[i]
            )
        eq_xi_0 = one_ef - xi_cur
        sp_head_evals[0] = (
            prev_s_eval - xi_cur * sp_head_evals[1] - sp_tail
        ) / eq_xi_0

        sp_head = [
            sum((row[j] * sp_head_evals[j] for j in range(s_deg)), start=zero)
            for row in inv_vdm
        ]
        # batch_s = eq(ξ_cur, X)·s'_head(X) + s'_tail·X, in coefficient form.
        coeffs = sp_head + [zero]
        b = one_ef - xi_cur
        a = xi_cur - b
        for i in reversed(range(s_deg)):
            coeffs[i + 1] = a * coeffs[i] + b * coeffs[i + 1]
        coeffs[0] = coeffs[0] * b
        coeffs[1] = coeffs[1] + sp_tail

        coeffs_arr = jnp.stack(coeffs)
        batch_s_evals = eval_coeffs(coeffs_arr, domain_pts)
        transcript = transcript.observe(batch_s_evals)
        transcript, r_round = sample_ext(transcript)
        new_prev_s_eval = eval_coeffs(coeffs_arr, r_round)

        # Fold MLEs (LSB pairing, re-pad zeros), frozen once the trace exhausts
        # so the fully-folded f̂(r⃗) at index 0 survives for the tilde reads and
        # the column openings.
        new_bufs_sels = []
        new_bufs_mats = []
        for t, n_lift in enumerate(n_lifts):
            if n_lift >= 1:
                live = round_idx <= n_lift
                fs = fold_pair(bufs_sels[t][0::2], bufs_sels[t][1::2], r_round)
                fs = jnp.concatenate([fs, jnp.zeros_like(fs)], axis=0)
                new_bufs_sels.append(jnp.where(live, fs, bufs_sels[t]))
                folded_m = []
                for m in bufs_mats[t]:
                    fm = fold_pair(m[0::2], m[1::2], r_round)
                    fm = jnp.concatenate([fm, jnp.zeros_like(fm)], axis=0)
                    folded_m.append(jnp.where(live, fm, m))
                new_bufs_mats.append(folded_m)
            else:
                new_bufs_sels.append(bufs_sels[t])
                new_bufs_mats.append(list(bufs_mats[t]))

        eq_r = xi_cur * r_round + (one_ef - xi_cur) * (one_ef - r_round)
        new_carry = (
            new_bufs_sels,
            new_bufs_mats,
            new_tilde_zc,
            new_tilde_p,
            new_tilde_q,
            eq_n * eq_r,
            eq_sharp_n * eq_r,
            r_round,
            transcript,
            new_prev_s_eval,
            round_idx + 1,
        )
        return new_carry, (batch_s_evals, r_round)

    init_carry = (
        sels,
        mats,
        [zero] * num_traces,
        [zero] * num_traces,
        [zero] * num_traces,
        eq_ns[0],
        eq_sharp_ns[0],
        r_0,
        transcript,
        prev_s_eval,
        jnp.int32(1),
    )
    final_carry, (round_polys, r_rounds) = lax.scan(
        step, init_carry, (eq_xi_xs, xi_cur_xs), length=n_max
    )
    mats = final_carry[1]
    transcript = final_carry[8]
    sumcheck_round_polys = list(round_polys)
    r = [r_0] + list(r_rounds)

    # --- Column openings: common main first, interleaved with rotations ---
    column_openings: list[list[Array]] = []
    for air, trace_mats in zip(airs, mats):
        if air.needs_next:
            local, rot = trace_mats[-2][0], trace_mats[-1][0]
            part0 = jnp.stack([local, rot], axis=-1).reshape(-1)
        else:
            part0 = trace_mats[-1][0]
        column_openings.append([part0])

    zero_arr = jnp.zeros((1,), EF)
    for air, openings in zip(airs, column_openings):
        part0 = openings[0]
        if air.needs_next:
            transcript = transcript.observe(part0)
        else:
            for j in range(part0.shape[0]):
                transcript = transcript.observe(part0[j : j + 1])
                transcript = transcript.observe(zero_arr)

    proof = BatchConstraintProof(
        numerator_term_per_air=numerator_term_per_air,
        denominator_term_per_air=denominator_term_per_air,
        univariate_round_coeffs=s_0,
        sumcheck_round_polys=sumcheck_round_polys,
        column_openings=column_openings,
        lambda_=lam,
        mu=mu,
        r=r,
    )
    return transcript, proof
