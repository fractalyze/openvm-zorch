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
Stage-2 convention), ``eval_eq``, ``compute_inv_vandermonde``. The
prismalinear/univariate-skip pieces live in ``prism.py``; the constraint DAG
evaluator in ``constraints.py``.

Reference:
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/logup_zerocheck/mod.rs#L184
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/logup_zerocheck/cpu.rs
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

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
from zorch.poly.univariate import compute_inv_vandermonde
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


def _horner(coeffs: list[Array], x: Array) -> Array:
    acc = jnp.zeros((), EF)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _conv(a: list[Array], b: list[Array]) -> list[Array]:
    out: list[Array] = [jnp.zeros((), EF) for _ in range(len(a) + len(b) - 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = out[i + j] + ai * bj
    return out


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
    s_0_deg = s_deg * ((1 << l_skip) - 1)

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
    eq_sharp = prism.eq_sharp_uni_poly(l_skip, xi[:l_skip])
    logup_prods: list[tuple[list[Array], list[Array]]] = [
        (
            _pad(_conv(_pad(p, sp_0_deg + 1), eq_sharp), s_0_deg + 1),
            _pad(_conv(_pad(q, sp_0_deg + 1), eq_sharp), s_0_deg + 1),
        )
        for p, q in sp_logup
    ]
    skip_domain_size = f_to_ef(f_const(1 << l_skip))
    numerator_term_per_air = []
    denominator_term_per_air = []
    for p_prod, q_prod in logup_prods:
        claims = []
        for prod in (p_prod, q_prod):
            acc = jnp.zeros((), EF)
            for j in range(0, s_0_deg + 1, 1 << l_skip):
                acc = acc + prod[j]
            claims.append(acc * skip_domain_size)
        transcript = transcript.observe(jnp.stack(claims))
        numerator_term_per_air.append(claims[0])
        denominator_term_per_air.append(claims[1])

    transcript, mu = sample_ext(transcript)
    mu_pows = _powers(mu, 3 * num_traces)

    zc_batched = [jnp.zeros((), EF) for _ in range(sp_0_deg + 1)]
    for t, coeffs in enumerate(sp_zc):
        for j, c in enumerate(_pad(coeffs, sp_0_deg + 1)):
            zc_batched[j] = zc_batched[j] + mu_pows[2 * num_traces + t] * c
    eq_uni = prism.eq_uni_poly(l_skip, xi[0])
    zc_prod = _pad(_conv(zc_batched, eq_uni), s_0_deg + 1)

    s_0 = []
    for j in range(s_0_deg + 1):
        coeff = zc_prod[j]
        for t, (p_prod, q_prod) in enumerate(logup_prods):
            coeff = coeff + mu_pows[2 * t] * p_prod[j] + mu_pows[2 * t + 1] * q_prod[j]
        s_0.append(coeff)
    transcript = transcript.observe(jnp.stack(s_0))

    transcript, r_0 = sample_ext(transcript)
    r = [r_0]
    prev_s_eval = _horner(s_0, r_0)

    # --- Fold the prism at r_0 ---
    mats = [
        [prism.fold_ple_evals(l_skip, m, r_0) for m in trace_mats]
        for trace_mats in mats
    ]
    sels = [prism.fold_ple_evals(l_skip, s, r_0) for s in sels]
    eq_ns = [prism.eval_eq_uni(l_skip, xi[0], r_0)]
    eq_sharp_ns = [prism.eval_eq_sharp_uni(l_skip, xi[:l_skip], r_0)]

    # --- MLE rounds 1..=n_max ---
    inv_vdm = _inv_vandermonde_rows(s_deg - 1)
    zc_tilde = [zero] * num_traces
    logup_tilde = [(zero, zero)] * num_traces
    sumcheck_round_polys: list[Array] = []
    for round_ in range(1, n_max + 1):
        r_prev = r[round_ - 1]
        sp_zc_evals: list[list[Array]] = []
        sp_lg_evals: list[tuple[list[Array], list[Array]]] = []
        for t, (air, n) in enumerate(zip(airs, n_per_trace)):
            n_lift = max(n, 0)
            norm = f_to_ef(f_inv_const(1 << max(-n, 0)))
            if round_ > n_lift:
                if round_ == n_lift + 1:
                    # Evaluate f̂(r⃗) once; later rounds just multiply r.
                    sels_row = sels[t][0]
                    rows = [m[0] for m in mats[t]]
                    node_vals = eval_nodes(
                        air.dag,
                        sels_row,
                        _dag_parts(rows, air.needs_next),
                        air.public_values,
                    )
                    zc_tilde[t] = eq_ns[round_ - 1] * acc_constraints(
                        air.dag, node_vals, lambda_pows
                    )
                    if air.dag.interactions:
                        numer, denom = acc_interactions(
                            air.dag, node_vals, beta_pows, eq_3bs[t]
                        )
                        logup_tilde[t] = (
                            eq_sharp_ns[round_ - 1] * numer * norm,
                            eq_sharp_ns[round_ - 1] * denom,
                        )
                else:
                    zc_tilde[t] = zc_tilde[t] * r_prev
                    logup_tilde[t] = (
                        logup_tilde[t][0] * r_prev,
                        logup_tilde[t][1] * r_prev,
                    )
                sp_zc_evals.append([zc_tilde[t]])
                sp_lg_evals.append(([logup_tilde[t][0]], [logup_tilde[t][1]]))
            else:
                eq_xi = _eq_table(xi[l_skip + round_ : l_skip + n_lift])
                sels_dom = lift_to_domain(sels[t][0::2], sels[t][1::2], s_deg - 1)
                mats_dom = [
                    lift_to_domain(m[0::2], m[1::2], s_deg - 1) for m in mats[t]
                ]
                node_vals = eval_nodes(
                    air.dag,
                    sels_dom,
                    _dag_parts(mats_dom, air.needs_next),
                    air.public_values,
                )
                acc = acc_constraints(air.dag, node_vals, lambda_pows)
                zc = (acc * eq_xi[None, :]).sum(axis=1)
                sp_zc_evals.append([zc[x] for x in range(1, s_deg)])
                if air.dag.interactions:
                    numer, denom = acc_interactions(
                        air.dag, node_vals, beta_pows, eq_3bs[t]
                    )
                    p = (numer * eq_xi[None, :]).sum(axis=1) * norm
                    q = (denom * eq_xi[None, :]).sum(axis=1)
                    sp_lg_evals.append(
                        ([p[x] for x in range(1, s_deg)], [q[x] for x in range(1, s_deg)])
                    )
                else:
                    zeros = [zero] * (s_deg - 1)
                    sp_lg_evals.append((zeros, zeros))

        # Head/tail combine (mod.rs): front-loaded exhaustion cutoff.
        tail_start = num_traces
        for t, n in enumerate(n_per_trace):
            if round_ > n:
                tail_start = t
                break
        sp_head_zc = [zero] * (s_deg - 1)
        sp_head_logup = [zero] * (s_deg - 1)
        sp_tail = zero
        for t in range(num_traces):
            mu_zc = mu_pows[2 * num_traces + t]
            mu_p, mu_q = mu_pows[2 * t], mu_pows[2 * t + 1]
            p_evals, q_evals = sp_lg_evals[t]
            if t < tail_start:
                for i in range(s_deg - 1):
                    sp_head_zc[i] = sp_head_zc[i] + mu_zc * sp_zc_evals[t][i]
                    sp_head_logup[i] = (
                        sp_head_logup[i] + mu_p * p_evals[i] + mu_q * q_evals[i]
                    )
            else:
                sp_tail = (
                    sp_tail
                    + mu_zc * sp_zc_evals[t][0]
                    + mu_p * p_evals[0]
                    + mu_q * q_evals[0]
                )

        sp_head_evals = [zero] * s_deg
        for i in range(s_deg - 1):
            sp_head_evals[i + 1] = (
                eq_ns[round_ - 1] * sp_head_zc[i]
                + eq_sharp_ns[round_ - 1] * sp_head_logup[i]
            )
        # s'(0) from s_j(0) + s_j(1) = s_{j-1}(r_{j-1}).
        xi_cur = xi[l_skip + round_ - 1]
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

        batch_s_evals = jnp.stack(
            [_horner(coeffs, f_to_ef(f_const(i))) for i in range(1, s_deg + 1)]
        )
        transcript = transcript.observe(batch_s_evals)
        sumcheck_round_polys.append(batch_s_evals)

        transcript, r_round = sample_ext(transcript)
        r.append(r_round)
        prev_s_eval = _horner(coeffs, r_round)

        # Fold MLEs (LSB pairing) and extend the eq accumulators.
        mats = [
            [fold_pair(m[0::2], m[1::2], r_round) if m.shape[0] > 1 else m for m in tm]
            for tm in mats
        ]
        sels = [
            fold_pair(s[0::2], s[1::2], r_round) if s.shape[0] > 1 else s
            for s in sels
        ]
        eq_r = xi_cur * r_round + (one_ef - xi_cur) * (one_ef - r_round)
        eq_ns.append(eq_ns[-1] * eq_r)
        eq_sharp_ns.append(eq_sharp_ns[-1] * eq_r)

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
