"""Stage 4 — stacked opening reduction (``prove_stacked_opening_reduction``).

Batch sumcheck reducing the per-trace column/rotation opening claims at ``r``
(Stage 3's output) to opening claims of the stacked matrix's columns at a new
point ``u``. Per (trace, column) pair the claims enter λ-batched; the kernels
are ``eq``/``κ_rot`` against ``r``, decomposed as (univariate over the skip
domain) × (multilinear over the cube), with short traces entering through the
stride indicator ``in_{D,n_T}`` and the sub-cube position ``b_{T,j}`` of each
column inside its stacked column (the ``eq_ub`` tail factors below).

Round 0 is the univariate skip round (degree ``2·(2^l_skip − 1)``, evaluated
on the cosets ``g·D``, ``g²·D`` and interpolated to coefficients); rounds
``1..=n_stack`` are quadratic MLE sumcheck rounds observed as evaluations at
``{1, 2}``. After the last fold each stacked column has collapsed to a single
value — the stacking openings, observed per commit.

Reference:
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/stacked_reduction.rs
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from openvm_zorch.commit.stacking import StackedLayout, StackedSlice
from openvm_zorch.fields import EF, MODULUS, f_const, f_to_ef
from openvm_zorch.logup_zerocheck import prism
from openvm_zorch.transcript import sample_ext
from zorch.sumcheck.prover import fold_pair
from zorch.transcript import DuplexTranscript


def _rot_prev(table: Array) -> Array:
    """``table[rot_prev(x)]`` for all x: cyclic shift by one (the zkx jnp has
    no ``roll``)."""
    if table.shape[0] == 1:
        return table
    return jnp.concatenate([table[-1:], table[:-1]])


@dataclass(frozen=True)
class StackingProof:
    """The reference ``StackingProof`` plus the sampled challenges."""

    lambda_: Array
    univariate_round_coeffs: Array  # (2·(2^l_skip − 1) + 1,) EF
    sumcheck_round_polys: list[Array]  # n_stack × (2,) EF, evals at {1, 2}
    stacking_openings: list[Array]  # per commit, (stacked_width,) EF
    u: list[Array]  # 1 + n_stack challenges


@dataclass(frozen=True)
class _TraceView:
    """One (trace, column) claim pair: where its sub-column lives in the
    stacked matrix and which λ powers batch its eq / rotation claims."""

    com_idx: int
    slice: StackedSlice
    lam_eq: int
    lam_rot: int | None


def _ef_const(value: int) -> Array:
    return f_to_ef(f_const(value))


def _exp_power_of_2(x: Array, k: int) -> Array:
    for _ in range(k):
        x = x * x
    return x


def _uni_kernel_args(l_skip: int, n: int, omega: int, r_0: Array):
    """The (l, ω, r) triple of the eq_D factor for a trace with ``n =
    log_height − l_skip``: short traces (n < 0) collapse to the order-``2^{l
    + n}`` subgroup (stacked_reduction.rs round-0 / fold_ple_evals match)."""
    if n < 0:
        return (
            l_skip + n,
            pow(omega, 1 << -n, MODULUS),
            _exp_power_of_2(r_0, -n),
        )
    return l_skip, omega, r_0


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _round0_group_contrib(
    l_skip: int,
    l_eff: int,
    n: int,
    ce: Array,
    z_grid: Array,
    r_uni: Array,
    omega_eff_ef: Array,
    eq_const: Array,
    eq_uni_1_grid: Array,
    eq_rs: Array,
    k_rot_rs: Array,
    lam_eq_w: Array,
    lam_rot_w: Array,
) -> Array:
    """One stacking group's round-0 contribution to ``s_evals`` (shape
    ``(num_cosets, 2^l_skip)``), as a single fused kernel.

    Evaluates the eq / κ_rot univariate kernels over the whole ``(coset, z)``
    grid, contracts ``q``'s coset windows (``ce``), then λ-batches the columns.
    Jitted so XLA fuses the per-grid array ops into a handful of kernels
    instead of dispatching the field arithmetic op-by-op (the lever that
    turned GKR/WHIR eager dispatch into a compute win); each distinct group
    shape compiles once. ``coset_evals`` is kept eager and passed in as ``ce``
    — its host-int Lagrange weight construction crashes the zkx backend's
    compiler when traced.
    """
    ind = prism.eval_in_uni(l_skip, n, z_grid)  # (C, S) or scalar 1
    eq_uni_r0 = prism.eval_eq_uni(l_eff, z_grid, r_uni)  # (C, S)
    eq_uni_r0_rot = prism.eval_eq_uni(l_eff, z_grid, r_uni * omega_eff_ef)
    if l_eff == 0:
        # A single-row trace (log_height 0 ⇒ l_eff 0) makes eval_eq_uni loop
        # zero times and return a scalar (eq_D over the trivial size-1
        # subgroup is the constant 1). Restore the (coset, z) grid shape the
        # window broadcast below needs — the value is already correct. l_eff
        # is static, so this never enters the synthetic l_eff>0 kernels.
        eq_uni_r0 = jnp.broadcast_to(eq_uni_r0, z_grid.shape)
        eq_uni_r0_rot = jnp.broadcast_to(eq_uni_r0_rot, z_grid.shape)
    # eq / κ_rot cube vectors over the windows axis, transposed so the window
    # contraction reduces the *trailing* axis (the EF reduce shape the zkx
    # backend lowers cleanly — a strided mid-axis EF reduce crashes codegen).
    eq_vec = eq_uni_r0[..., None] * eq_rs  # (C, S, windows)
    k_rot_vec = eq_uni_r0_rot[..., None] * eq_rs + (
        eq_const * eq_uni_1_grid[..., None] * (k_rot_rs - eq_rs)
    )
    ce_t = jnp.moveaxis(ce, 2, -1)  # (C, S, columns, windows)
    eq_per_col = (ce_t * eq_vec[:, :, None, :]).sum(axis=-1)  # (C, S, columns)
    rot_per_col = (ce_t * k_rot_vec[:, :, None, :]).sum(axis=-1)
    contrib = (lam_eq_w * eq_per_col + lam_rot_w * rot_per_col).sum(axis=-1)
    return contrib * ind  # (C, S)


def prove_stacked_opening_reduction(
    transcript: DuplexTranscript,
    l_skip: int,
    n_stack: int,
    stacked_per_commit: Sequence[tuple[Array, StackedLayout]],
    need_rot_per_commit: Sequence[Sequence[bool]],
    r: Sequence[Array],
) -> tuple[DuplexTranscript, StackingProof]:
    """Drive Stage 4 from the transcript state at ``stage3_end``.

    ``stacked_per_commit`` holds the Stage-1 result per commitment (common
    main first): the stacked matrix (base field, ``(2^(l_skip+n_stack), W)``)
    and its layout. ``need_rot_per_commit[c][m]`` says whether matrix ``m``
    of commit ``c`` carries a rotation claim; ``r`` is Stage 3's challenge
    vector (``r[0]`` univariate).
    """
    views: list[_TraceView] = []
    lam_count = 0
    for com_idx, (_, layout) in enumerate(stacked_per_commit):
        need_rot = need_rot_per_commit[com_idx]
        for mat_idx, _col_in_mat, s in layout.sorted_cols:
            # Every column reserves the rotation power even when unused —
            # mirrors Stage 3 observing (claim, 0) pairs for !need_rot.
            lam_rot = lam_count + 1 if need_rot[mat_idx] else None
            views.append(_TraceView(com_idx, s, lam_count, lam_rot))
            lam_count += 2

    # Runs of equal log_height (views come sorted descending by height).
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(views) + 1):
        if (
            i == len(views)
            or views[i].slice.log_height != views[start].slice.log_height
        ):
            groups.append((start, i))
            start = i

    transcript, lam = sample_ext(transcript)
    lam_pows = [jnp.ones((), EF)]
    for _ in range(lam_count - 1):
        lam_pows.append(lam_pows[-1] * lam)

    one = jnp.ones((), EF)
    omega = prism.omega_int(l_skip)
    r_0 = r[0]
    # eq_D(ω·r_0, 1): the boundary weight of the rotation kernel's cube part.
    eq_const = prism.eval_eq_uni_at_one(l_skip, r_0 * _ef_const(omega))

    # eq(-, r[1..1+ñ_T]) hypercube tables per distinct log_height (LSB-first).
    eq_tables: dict[int, Array] = {}
    for v in views:
        lht = v.slice.log_height
        if lht not in eq_tables:
            n_lift = max(lht - l_skip, 0)
            eq_tables[lht] = prism.eq_cube_table(list(r[1 : 1 + n_lift]))

    # --- Round 0: s_0 from evaluations on the cosets g·D, g²·D ---
    # The whole (coset, z-index) grid is evaluated at once: the per-x kernels
    # (eq_D, κ_rot, in_{D,n}) are elementwise and broadcast over the grid, and
    # the per-column window contraction folds into one batched reduction. This
    # replaces a 2·2^l_skip scalar-op nest per group — eager op-by-op dispatch
    # that XLA cannot fuse — with a handful of array ops over leading
    # (coset, z) batch axes (field arithmetic is exactly associative, so the
    # reduction order does not change s_0).
    num_cosets = 2  # q · (eq or κ_rot) is degree 2 per variable
    size = 1 << l_skip
    # z[c, k] = g^{c+1}·ω^k, the z-index of coset c (host ints → one EF grid).
    z_grid = jnp.stack(
        [
            jnp.stack(
                [
                    _ef_const(
                        pow(prism.GENERATOR, c + 1, MODULUS)
                        * pow(omega, k, MODULUS)
                        % MODULUS
                    )
                    for k in range(size)
                ]
            )
            for c in range(num_cosets)
        ]
    )  # (num_cosets, size) EF
    eq_uni_1_grid = prism.eval_eq_uni_at_one(l_skip, z_grid)  # group-invariant
    s_evals = jnp.zeros((num_cosets, size), EF)
    for g_start, g_end in groups:
        g_views = views[g_start:g_end]
        lht = g_views[0].slice.log_height
        n = lht - l_skip
        eq_rs = eq_tables[lht]
        # κ_rot's cube factor is eq at the rotated-back point: index x − 1.
        k_rot_rs = _rot_prev(eq_rs)
        q_cols = jnp.stack(
            [
                stacked_per_commit[v.com_idx][0][
                    v.slice.row_idx : v.slice.row_idx + v.slice.lifted_len(l_skip),
                    v.slice.col_idx,
                ]
                for v in g_views
            ],
            axis=1,
        )
        lam_eq_w = jnp.stack([lam_pows[v.lam_eq] for v in g_views])
        lam_rot_w = jnp.stack(
            [
                lam_pows[v.lam_rot] if v.lam_rot is not None else jnp.zeros((), EF)
                for v in g_views
            ]
        )
        l_eff, omega_eff, r_uni = _uni_kernel_args(l_skip, n, omega, r_0)
        # (num_cosets, 2^l_skip, 2^ñ_T windows, columns) — coset_evals stays
        # eager; only the kernel-eval + contraction is jitted.
        ce = f_to_ef(prism.coset_evals(l_skip, q_cols, num_cosets))
        s_evals = s_evals + _round0_group_contrib(
            l_skip,
            l_eff,
            n,
            ce,
            z_grid,
            r_uni,
            _ef_const(omega_eff),
            eq_const,
            eq_uni_1_grid,
            eq_rs,
            k_rot_rs,
            lam_eq_w,
            lam_rot_w,
        )  # (C, S)
    s_0_deg = num_cosets * (size - 1)
    s_0 = jnp.stack(
        prism.geometric_cosets_to_coeffs(l_skip, s_evals, num_cosets)[: s_0_deg + 1]
    )
    transcript = transcript.observe(s_0)

    transcript, u_0 = sample_ext(transcript)
    u = [u_0]

    # --- Fold the PLEs (q and both kernels) at u_0 ---
    q_evals = [
        prism.fold_ple_evals(l_skip, mat, u_0) for mat, _ in stacked_per_commit
    ]
    eq_uni_u01 = prism.eval_eq_uni_at_one(l_skip, u_0)
    k_rot_tables: dict[int, Array] = {}
    for lht, eq in eq_tables.items():
        n = lht - l_skip
        l_eff, omega_eff, r_uni = _uni_kernel_args(l_skip, n, omega, r_0)
        ind = prism.eval_in_uni(l_skip, n, u_0)
        eq_uni = prism.eval_eq_uni(l_eff, u_0, r_uni)
        eq_uni_rot = prism.eval_eq_uni(l_eff, u_0, r_uni * _ef_const(omega_eff))
        k_rot_tables[lht] = ind * (
            eq_uni_rot * eq + eq_const * eq_uni_u01 * (_rot_prev(eq) - eq)
        )
        eq_tables[lht] = eq * (ind * eq_uni)

    # --- Rounds 1..=n_stack: quadratic MLE sumcheck, evals at {1, 2} ---
    # eq(u[1+ñ_T..round], b_{T,j}[..round−ñ_T]) accumulator per view: once a
    # trace's cube variables are exhausted its q values stop folding and the
    # remaining rounds bind the column's position bits instead.
    eq_ub = [one] * len(views)
    round_polys: list[Array] = []
    for rnd in range(1, n_stack + 1):
        s_at_1 = jnp.zeros((), EF)
        s_at_2 = jnp.zeros((), EF)
        for g_start, g_end in groups:
            g_views = views[g_start:g_end]
            lht = g_views[0].slice.log_height
            n_lift = max(lht - l_skip, 0)
            hd = max(n_lift - rnd, 0)  # remaining hypercube dim
            eq_rs = eq_tables[lht]
            k_rot_rs = k_rot_tables[lht]
            for gi, v in enumerate(g_views):
                s = v.slice
                if rnd <= n_lift:
                    row_start = (s.row_idx >> lht) << (hd + 1)
                else:
                    row_start = (s.row_idx >> (l_skip + rnd)) << 1
                col = q_evals[v.com_idx][
                    row_start : row_start + (2 << hd), s.col_idx
                ]
                t0, t1 = col[0::2], col[1::2]
                q1, q2 = t1, t1 + t1 - t0
                ub = eq_ub[g_start + gi]
                if rnd > n_lift:
                    # Bind position bit b: eq(X, b) is b at X=1, 3b−1 at X=2.
                    b = (s.row_idx >> (l_skip + rnd - 1)) & 1
                    f1 = ub * (one if b else jnp.zeros((), EF))
                    f2 = ub * (_ef_const(2) if b else _ef_const(MODULUS - 1))
                    eq1, eq2 = eq_rs[0] * f1, eq_rs[0] * f2
                    k1, k2 = k_rot_rs[0] * f1, k_rot_rs[0] * f2
                else:
                    e_lo, e_hi = eq_rs[0::2] * ub, eq_rs[1::2] * ub
                    eq1, eq2 = e_hi, e_hi + e_hi - e_lo
                    k_lo, k_hi = k_rot_rs[0::2] * ub, k_rot_rs[1::2] * ub
                    k1, k2 = k_hi, k_hi + k_hi - k_lo
                s_at_1 = s_at_1 + lam_pows[v.lam_eq] * (q1 * eq1).sum()
                s_at_2 = s_at_2 + lam_pows[v.lam_eq] * (q2 * eq2).sum()
                if v.lam_rot is not None:
                    s_at_1 = s_at_1 + lam_pows[v.lam_rot] * (q1 * k1).sum()
                    s_at_2 = s_at_2 + lam_pows[v.lam_rot] * (q2 * k2).sum()

        batch = jnp.stack([s_at_1, s_at_2])
        transcript = transcript.observe(batch)
        round_polys.append(batch)

        transcript, u_rnd = sample_ext(transcript)
        u.append(u_rnd)

        q_evals = [
            fold_pair(q[0::2], q[1::2], u_rnd) if q.shape[0] > 1 else q
            for q in q_evals
        ]
        for lht, tbl in eq_tables.items():
            if tbl.shape[0] > 1:
                eq_tables[lht] = fold_pair(tbl[0::2], tbl[1::2], u_rnd)
        for lht, tbl in k_rot_tables.items():
            if tbl.shape[0] > 1:
                k_rot_tables[lht] = fold_pair(tbl[0::2], tbl[1::2], u_rnd)
        for t, v in enumerate(views):
            n_lift = max(v.slice.log_height - l_skip, 0)
            if rnd > n_lift:
                b = (v.slice.row_idx >> (l_skip + rnd - 1)) & 1
                eq_ub[t] = eq_ub[t] * (u_rnd if b else one - u_rnd)

    # --- Stacking openings: each stacked column has folded to one value ---
    openings: list[Array] = []
    for q in q_evals:
        assert q.shape[0] == 1
        openings.append(q[0])
        transcript = transcript.observe(q[0])

    return transcript, StackingProof(
        lambda_=lam,
        univariate_round_coeffs=s_0,
        sumcheck_round_polys=round_polys,
        stacking_openings=openings,
        u=u,
    )
