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
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/stacked_reduction.rs
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Sequence

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array, lax

from openvm_zorch.commit.stacking import StackedLayout, StackedSlice
from openvm_zorch.fields import EF, F, MODULUS, f_const, f_to_ef
from openvm_zorch.logup_zerocheck import prism
from openvm_zorch.transcript import sample_ext
from zorch.poly.univariate import powers
from zorch.prove import fold_rounds
from zorch.sumcheck.domain import EvalDomain, fold, natural_domain
from zorch.sumcheck.prover import RoundMsg, StandardRound
from zorch.transcript import DuplexTranscript, Transcript, sample_challenge


def _rot_prev(table: Array) -> Array:
    """``table[rot_prev(x)]`` for all x: cyclic shift by one (the FRX fnp has
    no ``roll``)."""
    if table.shape[0] == 1:
        return table
    return fnp.concatenate([table[-1:], table[:-1]])


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


@functools.cache
def _ef_const(value: int) -> Array:
    # A few distinct host ints (ω and its powers) recur across the groups; each
    # miss builds a device scalar, so cache them.
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


@functools.partial(frx.jit, static_argnums=(0, 1, 2))
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
    — its host-int Lagrange weight construction crashes the XLA backend's
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
        eq_uni_r0 = fnp.broadcast_to(eq_uni_r0, z_grid.shape)
        eq_uni_r0_rot = fnp.broadcast_to(eq_uni_r0_rot, z_grid.shape)
    # eq / κ_rot cube vectors over the windows axis, transposed so the window
    # contraction reduces the *trailing* axis (the EF reduce shape the XLA
    # backend lowers cleanly — a strided mid-axis EF reduce crashes codegen).
    eq_vec = eq_uni_r0[..., None] * eq_rs  # (C, S, windows)
    k_rot_vec = eq_uni_r0_rot[..., None] * eq_rs + (
        eq_const * eq_uni_1_grid[..., None] * (k_rot_rs - eq_rs)
    )
    ce_t = fnp.moveaxis(ce, 2, -1)  # (C, S, columns, windows)
    eq_per_col = (ce_t * eq_vec[:, :, None, :]).sum(axis=-1)  # (C, S, columns)
    rot_per_col = (ce_t * k_rot_vec[:, :, None, :]).sum(axis=-1)
    contrib = (lam_eq_w * eq_per_col + lam_rot_w * rot_per_col).sum(axis=-1)
    return contrib * ind  # (C, S)


@dataclass(frozen=True)
class _StackingSummand:
    """Product summand the sumcheck rounds fold: ``Σ_columns Q·EQW`` per cube
    point. Degree 2 (``q`` × the eq/κ_rot weight), so each round poly is
    quadratic; ``combine`` sums the stacked-column batch axis into the scalar
    round poly while the round folds the cube variable (the trailing axis). No
    loop-invariant scalars — the λ powers are baked into ``EQW``.

    Deliberately not zorch's ``ProductSummand``: ``summand_evals`` reduces the
    combine's output on a single axis, which lands on the cube variable only
    because ``combine`` has already contracted the stacked columns. A plain
    product would leave both contractions to that one reduction, which cannot do
    both — it would sum the columns and hand back an unreduced variable axis."""

    @property
    def degree(self) -> int:
        return 2

    def combine_scalars(self) -> tuple[Array, ...]:
        return ()

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array:
        del scalars  # λ rides inside EQW, not as a loop-invariant scalar
        q, eqw = factors
        return fnp.sum(q * eqw, axis=-2)  # contract the columns; keep the variable

    def _combine(self, *factors: Array) -> Array:
        """The summand bound to its (empty) scalars — the only seam
        ``summand_evals`` reads, so callers stay summand-generic."""
        return self.combine((), *factors)


class _StackingRound(StandardRound):
    """``StandardRound`` that also surfaces the challenge it sampled.

    ``StandardRound`` returns its round poly alone, but Stage 4's proof carries
    the sumcheck point ``u`` too, so this emits a ``RoundMsg`` — the seam zorch's
    own ``LogupSumcheckRound`` exposes for the same reason. Otherwise the base
    round verbatim, fixed to its extension-challenge path (Stage 4 always folds
    in ``EF``)."""

    def __call__(
        self, folded: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, RoundMsg]:
        msg = self._round_poly(folded)
        transcript = transcript.observe(msg)
        transcript, r = sample_challenge(transcript, self.ext_dtype, self.limbs)
        return fold(folded, r), transcript, RoundMsg(msg, r)


_EQW_GATHER_CACHE: dict[tuple, Array] = {}


def _eqw_gather_index(
    l_skip: int, commit_views: Sequence[_TraceView], width: int, height: int
) -> Array:
    """Device gather index placing each view's weight block into the
    ``(height, width)`` eqw matrix; sentinel 0 fills the gaps and zero tail.
    Pure function of the layout — memoized by ``_eqw_columns``, mirroring
    ``commit/stacking.py``'s ``_gather_index``.

    A view's length-``2^ñ`` weight vector lands in stacked column ``col_idx`` at
    cube offset ``row_idx >> l_skip``: the skip domain is round 0's, so the cube
    position is the stacked row shifted past it (equivalently, the running sum of
    the earlier blocks in the column)."""
    dest = []
    for v in commit_views:
        s = v.slice
        blen = 1 << max(s.log_height - l_skip, 0)
        dest.append(((s.row_idx >> l_skip) + np.arange(blen)) * width + s.col_idx)
    gather = np.zeros((height * width,), np.int64)
    if dest:
        dest_flat = np.concatenate(dest)
        gather[dest_flat] = np.arange(dest_flat.size) + 1
    return fnp.asarray(gather)


def _eqw_columns(
    l_skip: int,
    commit_views: Sequence[_TraceView],
    width: int,
    height: int,
    eq_tables: dict[int, Array],
    k_rot_tables: dict[int, Array],
    lam_eq: Array,
    lam_rot: Array,
) -> Array:
    """The λ-batched eq/κ_rot weight matrix for one commit — the full-cube
    companion of the stacked matrix ``Q``, shape ``(2^n_stack, width)``.

    Cell ``[row, col]`` holds the view occupying it,
    ``λ^{lam_eq}·eq + λ^{lam_rot}·κ_rot`` over the view's contiguous cube block
    and zero elsewhere, so folding it round-by-round binds every view uniformly:
    a short trace's block is just a short eq vector, so the high cube bits bind
    its position automatically once the block is exhausted.

    Assembled like ``stacked_matrix``: one broadcast per equal-height run of
    views (a run shares eq/κ_rot tables, so ``lam_eq[run][:, None] *
    eq_tables[lht]`` weights the whole run at once, κ_rot riding the 0-masked
    ``lam_rot``), then one memoized gather scatters the blocks into their
    stacked-column positions. ``lam_eq[i]`` / ``lam_rot[i]`` are the λ powers of
    ``commit_views[i]`` (``lam_rot`` 0 for an absent rotation), pre-sliced by the
    caller. ``EQW`` and ``Q`` share the one row layout the gather encodes."""
    blocks: list[Array] = []
    i, n = 0, len(commit_views)
    while i < n:
        lht = commit_views[i].slice.log_height
        j = i
        while j < n and commit_views[j].slice.log_height == lht:
            j += 1
        weights = (
            lam_eq[i:j][:, None] * eq_tables[lht]
            + lam_rot[i:j][:, None] * k_rot_tables[lht]
        )
        blocks.append(weights.reshape(-1))
        i = j
    key = (
        l_skip,
        width,
        height,
        tuple(
            (v.slice.col_idx, v.slice.row_idx, v.slice.log_height)
            for v in commit_views
        ),
    )
    gather = _EQW_GATHER_CACHE.get(key)
    if gather is None:
        gather = _eqw_gather_index(l_skip, commit_views, width, height)
        _EQW_GATHER_CACHE[key] = gather
    src = fnp.concatenate([fnp.zeros((1,), EF)] + blocks)
    return fnp.take(src, gather).reshape(height, width)


_Q_COLS_GATHER_CACHE: dict[tuple, Array] = {}


def _q_cols_gather_index(
    l_skip: int,
    g_views: Sequence[_TraceView],
    mat_offsets: Sequence[int],
    mat_widths: Sequence[int],
) -> Array:
    """Device gather index building a group's ``(lifted_len, group_size)`` q
    block from the flat stacked-matrix source. Pure function of the layout;
    memoized per group, mirroring ``commit/stacking.py``'s ``_gather_index``.

    A view's round-0 claim is the contiguous column segment
    ``mat[row:row+lifted, col]`` — a stride-``width`` run in the row-major flat
    matrix, shifted to its commit's span in the shared source. Column ``k`` of
    the block is view ``k``, reproducing the axis-1 stack it replaces."""
    lifted = g_views[0].slice.lifted_len(l_skip)
    cols = [
        mat_offsets[v.com_idx]
        + v.slice.col_idx
        + (v.slice.row_idx + np.arange(lifted)) * mat_widths[v.com_idx]
        for v in g_views
    ]
    return fnp.asarray(np.stack(cols, axis=1).reshape(-1))


def _sumcheck_rounds(
    transcript: DuplexTranscript,
    q_evals: Sequence[Array],
    eq_tables: dict[int, Array],
    k_rot_tables: dict[int, Array],
    views: Sequence[_TraceView],
    lam_eq: Array,
    lam_rot: Array,
    u: Sequence[Array],
    n_stack: int,
    l_skip: int,
) -> tuple[DuplexTranscript, list[Array], list[Array], list[Array]]:
    """Rounds 1..=n_stack of Stage 4: the quadratic MLE sumcheck (round-poly
    evals at {1, 2}) folding every stacked column to its opening, then observing
    the openings.

    Driven by zorch's ``StandardRound`` under ``fold_rounds`` (the prover's one
    runtime-dominated stage; #43): the jagged per-view fold is expressed as two
    homogeneous full-cube factors — the stacked matrices ``Q`` and their λ-batched
    eq/κ_rot weights ``EQW`` — stacked into the one ``(factors, columns, cube)``
    state a round takes, with the stacked columns batched and the cube variable
    trailing. The round samples its poly on ``{1, 2}`` (the reference's compressed
    wire form, whose verifier derives ``s(0) = claim − s(1)`` from the running
    claim) and folds at an extension-field challenge (``sample_ext``'s four base
    squeezes). The variable axis is bit-reversed so the round's MSB-first block
    fold matches the reference's LSB-first stride fold, byte-identical to the
    reference prover.

    ``fold_rounds`` unrolls one round per cube variable. The fold halves the
    state each round, so the rounds are not fixed-shape and do not fit a
    ``lax.scan`` without padding every round back to full width and masking the
    dead lanes — which is what this stage did before it moved onto the shared
    round. Byte-identical either way (the masked tail was exactly zero), and
    cheaper here: this stage runs eager, where a ``lax.scan`` re-traces and
    recompiles its body on every call while the unrolled ops hit the dispatch
    cache (~8x at the production ``n_stack`` of 16, ~100x at the suite's 8).
    Jitting the stage would invert that — an unrolled module grows with the
    round count, where one scan body does not."""
    u = list(u)
    widths = [q.shape[1] for q in q_evals]

    if n_stack == 0:
        # No cube variables: each stacked column already holds its single opening.
        openings: list[Array] = []
        for q in q_evals:
            openings.append(q[0])
            transcript = transcript.observe(q[0])
        return transcript, [], u, openings

    height = 1 << n_stack
    q_cols = fnp.concatenate([q.T for q in q_evals], axis=0)  # (Σ width, 2^n_stack)
    # views are commit-major; each commit's contiguous run indexes its slice of
    # the per-view λ vectors alongside its views.
    com_ranges: list[tuple[int, int]] = []
    start = 0
    for c in range(len(q_evals)):
        end = start
        while end < len(views) and views[end].com_idx == c:
            end += 1
        com_ranges.append((start, end))
        start = end
    eqw_cols = fnp.concatenate(
        [
            _eqw_columns(
                l_skip,
                views[s:e],
                widths[c],
                height,
                eq_tables,
                k_rot_tables,
                lam_eq[s:e],
                lam_rot[s:e],
            ).T
            for c, (s, e) in enumerate(com_ranges)
        ],
        axis=0,
    )
    q_cols = lax.bit_reverse(q_cols, dimensions=(1,))
    eqw_cols = lax.bit_reverse(eqw_cols, dimensions=(1,))

    summand = _StackingSummand()
    # {1, 2} — the natural {0, 1, 2} domain minus s(0), which the verifier
    # reconstructs from the running claim rather than reading off the wire.
    domain = EvalDomain(natural_domain(summand.degree, EF).nodes[1:])
    folded, transcript, msgs = fold_rounds(
        _StackingRound(summand, domain, ext_dtype=EF),
        fnp.stack([q_cols, eqw_cols]),
        transcript,
        n_stack,
    )
    round_polys = [m.round_poly for m in msgs]
    u = u + [m.challenge for m in msgs]

    # Split the folded stacked columns back per commit and observe the openings.
    folded_q = folded[0][:, 0]  # (Σ width,)
    openings = []
    start = 0
    for w in widths:
        opening = folded_q[start : start + w]
        openings.append(opening)
        transcript = transcript.observe(opening)
        start += w
    return transcript, round_polys, u, openings


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
    lam_pows = powers(lam, lam_count)
    # Views reserve two powers each, in order (see above), so view i takes
    # λ^{2i} / λ^{2i+1}: the weights are just the even and odd strides, and a
    # group — a contiguous run of views — is a slice of them. No per-view index
    # vector, and nothing rebuilt per group.
    lam_eq_all = lam_pows[0::2]
    lam_rot_all = fnp.where(
        fnp.array([v.lam_rot is not None for v in views]),
        lam_pows[1::2],
        fnp.zeros((), EF),
    )

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
    # z[c, k] = g^{c+1}·ω^k, the z-index of coset c: the host ints become one
    # base-field array, embedded into EF in a single op rather than a scalar
    # `_ef_const` per cell nested in two stacks.
    z_ints = np.array(
        [
            [
                pow(prism.GENERATOR, c + 1, MODULUS) * pow(omega, k, MODULUS) % MODULUS
                for k in range(size)
            ]
            for c in range(num_cosets)
        ],
        dtype=np.int64,
    )
    z_grid = f_to_ef(fnp.array(z_ints, F))  # (num_cosets, size) EF
    eq_uni_1_grid = prism.eval_eq_uni_at_one(l_skip, z_grid)  # group-invariant

    # One flat source for every group's q-column gather: the base-field stacked
    # matrices laid end to end behind a zero sentinel (row-major, so a column
    # segment is a stride-`width` run). Each group then gathers its views'
    # segments in one `fnp.take` instead of a slice-and-stack per view — 723
    # slices + stacks on the real block, ~68 ms of the stage (commit/stacking.py
    # recipe). Each group's index is a pure function of the layout, memoized.
    mat_widths = [mat.shape[1] for mat, _ in stacked_per_commit]
    mat_shapes = tuple((mat.shape[0], mat.shape[1]) for mat, _ in stacked_per_commit)
    mat_offsets: list[int] = []
    off = 1
    for mat, _ in stacked_per_commit:
        mat_offsets.append(off)
        off += mat.shape[0] * mat.shape[1]
    q_src = fnp.concatenate(
        [fnp.zeros((1,), stacked_per_commit[0][0].dtype)]
        + [mat.reshape(-1) for mat, _ in stacked_per_commit]
    )

    s_evals = fnp.zeros((num_cosets, size), EF)
    for g_start, g_end in groups:
        g_views = views[g_start:g_end]
        lht = g_views[0].slice.log_height
        n = lht - l_skip
        eq_rs = eq_tables[lht]
        # κ_rot's cube factor is eq at the rotated-back point: index x − 1.
        k_rot_rs = _rot_prev(eq_rs)
        key = (
            l_skip,
            mat_shapes,
            tuple(
                (v.com_idx, v.slice.row_idx, v.slice.col_idx, v.slice.log_height)
                for v in g_views
            ),
        )
        q_idx = _Q_COLS_GATHER_CACHE.get(key)
        if q_idx is None:
            q_idx = _q_cols_gather_index(l_skip, g_views, mat_offsets, mat_widths)
            _Q_COLS_GATHER_CACHE[key] = q_idx
        q_cols = fnp.take(q_src, q_idx).reshape(
            g_views[0].slice.lifted_len(l_skip), len(g_views)
        )
        # Slice the strided weights: a per-view `stack` here took one operand per
        # view — 723 on the real block, ~104 ms of the stage.
        lam_eq_w = lam_eq_all[g_start:g_end]
        lam_rot_w = lam_rot_all[g_start:g_end]
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
    s_0 = fnp.stack(
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

    # --- Rounds 1..=n_stack: the quadratic MLE sumcheck (split out) ---
    transcript, round_polys, u, openings = _sumcheck_rounds(
        transcript,
        q_evals,
        eq_tables,
        k_rot_tables,
        views,
        lam_eq_all,
        lam_rot_all,
        u,
        n_stack,
        l_skip,
    )

    return transcript, StackingProof(
        lambda_=lam,
        univariate_round_coeffs=s_0,
        sumcheck_round_polys=round_polys,
        stacking_openings=openings,
        u=u,
    )
