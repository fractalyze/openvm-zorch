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
- MLE rounds 1..n_max run on zorch's summand-generic scan driver
  (``zorch.sumcheck.prover.prove``). The reference's jagged front-loaded
  schedule — a trace folds its own ``ñ_t`` MLE variables, then ×r-accumulates a
  constant "tilde" term once exhausted (cpu.rs ``sumcheck_polys_eval``) — is
  reformulated as ONE uniform product sumcheck over ``n_max`` variables by
  embedding every trace into the full ``H = 2^n_max`` cube: its eq weight gains
  high coordinates pinned to ``ξ=1`` (``eq(1, X) = X`` ⇒ the tilde·X tail falls
  out of the eq factor folding), and its data columns are broadcast-constant over
  those high variables. The round poly is ``s_deg`` evals on ``{1..s_deg}``
  (``eval_start=1``; ``s'(0)`` is the verifier-reconstructed ``s_j(0) = claim −
  s_j(1)``) and the fold challenge is an extension element. See
  ``_BatchZerocheckRound`` and ``jagged-sumcheck-as-uniform-fullcube-product``.
- round-0 polynomial helpers stay as Python lists of scalar arrays — every degree
  there is ≤ ``s_0_deg`` (13 coefficients), so unrolled exact arithmetic
  beats array plumbing.

zorch reuse: ``zorch.sumcheck.prover.prove`` (the generic per-variable scan
driver — owns the split / lift / round-poly / Fiat-Shamir / fold),
``expand_eq_to_hypercube`` (fed reversed ξ slices, the Stage-2 convention),
``eval_eq``, ``eval_coeffs`` (coefficient-form univariate eval, O(1) graph in
degree). The prismalinear/univariate-skip pieces live in ``prism.py``; the
constraint DAG evaluator in ``constraints.py``.

Reference:
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/logup_zerocheck/mod.rs#L184
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/logup_zerocheck/cpu.rs
"""

from __future__ import annotations

import operator
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce

import jax
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
from openvm_zorch.transcript import EF_LIMBS, sample_ext
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.univariate import eval_coeffs
from zorch.round import Round
from zorch.sumcheck.prover import prove
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class AirData:
    """One present AIR, in sorted (stacking) order."""

    trace: Array  # (height, width) base field — the common main
    dag: ConstraintsDag
    public_values: tuple[int, ...]
    constraint_degree: int  # this AIR's vk.max_constraint_degree
    needs_next: bool
    # Cached-main partitions (base-field ``(height, width)``, partition order,
    # same height as ``trace``); the partitioned main is ``cached_mains ++
    # [trace]`` so a ``main`` DAG node's ``part_index`` selects among them.
    # Empty for the synthetic fixture.
    cached_mains: tuple[Array, ...] = ()


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
    mis-lower on this fork — see docs/development.md). Dispatch-free and jit-fusable,
    unlike a per-scalar coefficient loop. (``conv_test`` pins it against the
    reference scalar convolution.)"""
    la = coeffs.shape[-1]
    lb = kernel.shape[-1]
    lo = la + lb - 1
    lead = coeffs.shape[:-1]
    shifts = [
        jnp.pad(coeffs, ((0, 0),) * len(lead) + ((j, lo - la - j),)) for j in range(lb)
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


def _bit_reverse_perm(n: int) -> Array:
    """The bit-reversal permutation of ``[0, n)`` as an index array, ``n`` a power
    of two. Used as a gather (``x[perm]``) to reorder a full-cube factor LSB-first
    so the driver's MSB-first block fold reproduces the reference's LSB-first
    stride fold (a host-int gather, not ``lax.bit_reverse``, to dodge any
    extension-dtype dispatch on the permuted values)."""
    bits = n.bit_length() - 1
    rev = [0] * n
    for i in range(n):
        for b in range(bits):
            rev[i] |= ((i >> b) & 1) << (bits - 1 - b)
    return jnp.array(rev, dtype=jnp.int32)


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
    table = jnp.stack([rows == 0, rows != height - 1, rows == height - 1], axis=-1)
    return table.astype(jnp.uint32).astype(F)


def _view_mats(air: AirData, l_skip: int) -> list[Array]:
    """The (local, rot) matrix list of single.rs ``view_mats``, lifted — one
    entry per partitioned-main part in order ``cached_mains ++ [common_main]``,
    with a rotation entry interleaved after each when the AIR rotates. The flat
    ``(local, rot, local, rot, …)`` layout ``_dag_parts`` regroups; the common
    main lands last, so the column-opening read pops it to the front of each
    AIR's ``[common, *cached]`` opening list (reference ``into_column_openings``)."""
    out: list[Array] = []
    for m in (*air.cached_mains, air.trace):
        out.append(_lift(m, l_skip))
        if air.needs_next:
            rot = jnp.concatenate([m[1:], m[:1]], axis=0)
            out.append(_lift(rot, l_skip))
    return out


def _dag_parts(mats: list[Array], needs_next: bool) -> list[tuple[Array, Array | None]]:
    """Group a flat (local, rot, local, rot, ...) mat list into the DAG
    evaluator's (local, next) pairs."""
    if needs_next:
        return [(mats[i], mats[i + 1]) for i in range(0, len(mats), 2)]
    return [(m, None) for m in mats]


def _round0_constraint_fns(dag, needs_next, public_values, l_skip, constraint_degree):
    """jitted per-trace round-0 constraint evaluators (#45).

    The whole round-0 per-trace compute — the prism coset evaluation, the
    ``eval_nodes`` DAG walk, the ``acc_*`` accumulation and the ``eq_D``
    weight/row-sum — is 62% of warm zerocheck on the real block and ran as an
    eager array-op dispatch storm (hundreds of nodes × 19 AIRs). On CPU that is
    FLOP-bound, but on GPU the unfused HBM traffic of the materialized coset
    cells + node values makes it intrinsically ~115s. Fuse each trace's whole
    compute into one kernel: the DAG / rotation flag / public values are static
    (closed over, so the node walk unrolls into the graph), and ``coset_evals``
    runs INSIDE the kernel so the big ``(num_cosets, size, rows, width)`` cells
    never round-trip to HBM. The raw lifted mats/sels, λ/β powers and eq tables
    are traced args. ``prewarm_coset_weights`` must have been called for these
    coset counts so the host-int ω / weight builders are already cached and
    constant-fold (a cold cache faults under trace — ``omega_int``'s
    ``lax.fft`` + ``int(...)`` concretization). Same DAG walk the MLE
    ``lax.scan`` jits, pure EF arithmetic, contracted (row) axis kept LAST per
    docs/development.md ⇒ byte-exact. One compile per AIR (distinct DAGs); warm/GPU reaps
    the fusion.
    """
    num_cosets_zc = constraint_degree - 1

    if num_cosets_zc > 0:
        inv_zerofiers = jnp.stack(
            [
                f_to_ef(
                    f_inv_const(
                        pow(prism.GENERATOR, (c + 1) << l_skip, prism.MODULUS) - 1
                    )
                )
                for c in range(num_cosets_zc)
            ]
        )

        @jax.jit
        def zc_eval(trace_sels, trace_mats, lambda_pows, eq_xi):
            sels_cells = prism.coset_evals(l_skip, trace_sels, num_cosets_zc)
            mat_cells = [
                prism.coset_evals(l_skip, m, num_cosets_zc) for m in trace_mats
            ]
            parts = _dag_parts(mat_cells, needs_next)
            node_vals = eval_nodes(dag, sels_cells, parts, public_values)
            acc = acc_constraints(dag, node_vals, lambda_pows)
            weighted = acc * eq_xi[None, None, :]
            return weighted.sum(axis=2) * inv_zerofiers[:, None]  # (num_cosets, size)
    else:
        zc_eval = None

    @jax.jit
    def lu_eval(trace_sels, trace_mats, beta_pows, eq_3bs_t, eq_xi):
        sels_cells = prism.coset_evals(l_skip, trace_sels, constraint_degree)
        mat_cells = [
            prism.coset_evals(l_skip, m, constraint_degree) for m in trace_mats
        ]
        parts = _dag_parts(mat_cells, needs_next)
        node_vals = eval_nodes(dag, sels_cells, parts, public_values)
        numer, denom = acc_interactions(dag, node_vals, beta_pows, eq_3bs_t)
        p = (numer * eq_xi[None, None, :]).sum(axis=2)
        q = (denom * eq_xi[None, None, :]).sum(axis=2)
        return p, q  # each (num_cosets, size)

    return zc_eval, lu_eval


@dataclass(frozen=True)
class _BatchZerocheckRound(Round):
    """The per-variable summand of the batched ZeroCheck + LogUp MLE sumcheck,
    fed to zorch's generic scan driver (``zorch.sumcheck.prover.prove``).

    The reference (cpu.rs ``sumcheck_polys_eval``) front-loads a jagged per-trace
    schedule: a trace of cube height ``2^ñ_t`` folds its own MLE for ``ñ_t``
    rounds, then contributes a constant "tilde" term times the running
    ``r``-product once exhausted. That jagged schedule is mathematically one
    *uniform* product sumcheck over ``n_max`` variables once every trace is
    embedded into the full ``H = 2^n_max`` cube (see
    ``jagged-sumcheck-as-uniform-fullcube-product``):

    - each trace's eq weight becomes the full-cube ``eq`` whose low ``ñ_t``
      coordinates are ``ξ[l_skip : l_skip+ñ_t]`` and whose high
      ``n_max − ñ_t`` coordinates are the all-ones point ``ξ=1``. Since
      ``eq(1, X) = X``, a trace exhausted after round ``ñ_t`` contributes a pure
      ``tilde·X`` term that ×r-accumulates exactly as the reference's tail, and
      summing the high block preserves the claim;
    - each data column (selectors + matrix columns) is embedded constant
      (broadcast) over the high variables.

    The round summand per trace is ``eq_t · (μ_zc·eq_n·acc + eq♯_n·(μ_p·numer·norm
    + μ_q·denom))`` — the same μ-batched ZeroCheck + LogUp combine the reference
    runs, with ``eq_n``/``eq♯_n`` the round-0 univariate-eq carry-overs (the
    per-round ``eq(ξ_cur, r)`` accumulation is handled by the eq factor folding
    inside the driver). Contributions sum over traces; the driver sums the cube.

    The driver lifts each factor to ``(n_domain, half)`` (cube last) but
    ``eval_nodes`` indexes columns on the *last* axis, so a trace's columns are
    re-stacked into a trailing width axis before the DAG walk (the sp1
    ``_fold_chip`` layout flip). ``degree = s_deg`` (constraint degree + 1 for the
    linear eq factor) and ``eval_start = 1`` make the round poly ``s_deg`` evals on
    ``{1..s_deg}`` — the reference wire form.
    """

    airs: tuple[AirData, ...]
    # Per-trace flat-column layout in the driver state. Each trace contributes,
    # in order, ``[eq, sel_0, sel_1, sel_2, *(view-mat columns)]``; ``part_widths``
    # is the column count of each view-mat (the flat (local, rot, …) list
    # ``_dag_parts`` regroups), so ``combine`` re-slices the flat mat columns back
    # into the DAG evaluator's parts.
    part_widths: tuple[tuple[int, ...], ...]
    # Loop-invariant EF scalars, closed over (the marked driver path is bypassed
    # by ``eval_start=1``, so they need not ride ``combine_scalars``).
    lambda_pows: tuple[Array, ...]
    beta_pows: tuple[Array, ...]
    eq_3bs: tuple[tuple[Array, ...], ...]
    mu_zc: tuple[Array, ...]
    mu_p: tuple[Array, ...]
    mu_q: tuple[Array, ...]
    norms: tuple[Array, ...]
    eq_n: Array
    eq_sharp_n: Array
    degree: int

    def combine_scalars(self) -> tuple[Array, ...]:
        """No marker-threaded scalars: ``eval_start=1`` bypasses the dedicated
        fusion marker, so the summand reads its (closed-over) scalars directly."""
        return ()

    def combine(self, scalars: Sequence[object], *factors: Array) -> Array:
        """``Σ_t eq_t · (μ_zc·eq_n·acc + eq♯_n·(μ_p·numer·norm + μ_q·denom))`` over
        the driver's lifted full-cube factors (each ``(n_domain, half)``).

        ``factors`` arrives flat as ``[eq_t0, sel_t0×3, mat_t0…, eq_t1, …]``; it is
        re-sliced per trace, the per-part mat columns and the selectors stacked
        onto a trailing width axis for the DAG walk (so ``eval_nodes`` reads
        columns on the last axis), then the per-trace contributions summed. The
        single source of the round math — the round-poly reduction routes here."""
        del scalars
        terms = []
        offset = 0
        for t, air in enumerate(self.airs):
            widths = self.part_widths[t]
            n_mat = sum(widths)
            eq_t = factors[offset]
            sel_cols = factors[offset + 1 : offset + 4]
            mat_cols = factors[offset + 4 : offset + 4 + n_mat]
            offset += 4 + n_mat

            sels = jnp.stack(sel_cols, axis=-1)
            parts: list[Array] = []
            c = 0
            for w in widths:
                parts.append(jnp.stack(mat_cols[c : c + w], axis=-1))
                c += w

            node_vals = eval_nodes(
                air.dag,
                sels,
                _dag_parts(parts, air.needs_next),
                air.public_values,
            )
            acc = acc_constraints(air.dag, node_vals, self.lambda_pows)
            contrib = self.mu_zc[t] * self.eq_n * acc
            if air.dag.interactions:
                numer, denom = acc_interactions(
                    air.dag, node_vals, self.beta_pows, self.eq_3bs[t]
                )
                contrib = contrib + self.eq_sharp_n * (
                    self.mu_p[t] * numer * self.norms[t] + self.mu_q[t] * denom
                )
            terms.append(eq_t * contrib)
        return reduce(operator.add, terms)

    def _combine(self, *factors: Array) -> Array:
        return self.combine(self.combine_scalars(), *factors)


_ZC_PROFILE = os.environ.get("OPENVM_ZC_PROFILE") == "1"


class _ZcProfiler:
    """Coarse, env-guarded region timer for Stage-3 localization (#45).

    No-op unless ``OPENVM_ZC_PROFILE=1``. Each ``mark`` blocks on the region's
    output arrays and prints the wall-clock since the previous mark, so a cold
    pass shows compile+run and a warm pass run-only, per region. Off by default
    so ``verify_prove``'s whole-stage ``_TimedRound`` number stays
    block-distortion-free: coarse region blocks sum coherently, but per-element
    blocks inflate badly (the #3 41.1s artifact)."""

    def __init__(self) -> None:
        self._t = time.monotonic()

    def mark(self, label: str, *outputs: object) -> None:
        if not _ZC_PROFILE:
            return
        jax.block_until_ready(outputs)
        now = time.monotonic()
        print(f"  [zc {label}] {now - self._t:.3f}s", flush=True)
        self._t = now


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
    _zc = _ZcProfiler()
    num_traces = len(airs)
    n_per_trace = [log2_strict_usize(air.trace.shape[0]) - l_skip for air in airs]
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
    eq_3bs: list[list[Array]] = [[zero] * len(air.dag.interactions) for air in airs]
    for trace_idx, int_idx, s in layout.sorted_cols:
        n_lift = max(n_per_trace[trace_idx], 0)
        b_int = s.row_idx >> (l_skip + n_lift)
        n_bits = n_logup - n_lift
        if n_bits == 0:
            eq_3bs[trace_idx][int_idx] = one_ef
            continue
        bits = f_to_ef(jnp.array([(b_int >> j) & 1 for j in range(n_bits)], F))
        point = jnp.stack(xi[l_skip + n_lift : l_skip + n_logup])
        eq_3bs[trace_idx][int_idx] = eval_eq(point, bits)

    # --- Batching randomness λ ---
    transcript, lam = sample_ext(transcript)
    max_num_constraints = max((len(air.dag.constraint_idx) for air in airs), default=0)
    lambda_pows = _powers(lam, max(max_num_constraints, 1))

    # Pre-build the host-int prism coset weights eagerly so the jitted round-0
    # evaluators below hit the lru_cache instead of faulting on the construction
    # under trace (#45). Coset counts: constraint_degree (logup) and
    # constraint_degree - 1 (zerocheck).
    for cd in {air.constraint_degree for air in airs}:
        for nc in (cd, cd - 1):
            if nc > 0:
                prism.prewarm_coset_weights(l_skip, nc)
    _zc.mark("setup", mats, sels, eq_3bs, beta_pows, lambda_pows)

    # --- Round 0: per-trace s'_0 polynomials on geometric cosets ---
    sp_zc: list[list[Array]] = []
    sp_logup: list[tuple[list[Array], list[Array]]] = []
    for t, (air, n, trace_mats, trace_sels) in enumerate(
        zip(airs, n_per_trace, mats, sels)
    ):
        n_lift = max(n, 0)
        eq_xi = _eq_table(xi[l_skip : l_skip + n_lift])
        norm = f_inv_const(1 << max(-n, 0))

        # Zerocheck: q = s'_0 / (Z^N - 1) from constraint_degree - 1 cosets.
        num_cosets = air.constraint_degree - 1
        zc_eval, lu_eval = _round0_constraint_fns(
            air.dag, air.needs_next, air.public_values, l_skip, air.constraint_degree
        )
        if num_cosets == 0:
            sp_zc.append([])
        else:
            q_evals = zc_eval(
                trace_sels, trace_mats, lambda_pows, eq_xi
            )  # (num_cosets, size)
            q = prism.geometric_cosets_to_coeffs(l_skip, q_evals, num_cosets)
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
        p_evals, q_evals = lu_eval(trace_sels, trace_mats, beta_pows, eq_3bs[t], eq_xi)
        p_coeffs = prism.geometric_cosets_to_coeffs(
            l_skip, p_evals, air.constraint_degree
        )
        q_coeffs = prism.geometric_cosets_to_coeffs(
            l_skip, q_evals, air.constraint_degree
        )
        sp_logup.append(([c * norm for c in p_coeffs], q_coeffs))
    _zc.mark("round0", sp_zc, sp_logup)

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
    # the EF reduce stays jit-safe (docs/development.md).
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
    _zc.mark("s0_assembly", s_0_arr, p_prods, q_prods)

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
    _zc.mark("r0_fold", mats, sels, eq_ns, eq_sharp_ns, prev_s_eval)

    # --- MLE rounds 1..=n_max, via the generic scan driver (issue #45) -------
    # The reference's jagged front-loaded per-trace schedule (a trace folds its
    # own ``ñ_t`` MLE variables, then ×r-accumulates a constant "tilde" term) is
    # mathematically one *uniform* product sumcheck over ``n_max`` variables once
    # every trace is embedded into the full ``H = 2^n_max`` cube, so it runs on
    # zorch's summand-generic scan driver (``_BatchZerocheckRound`` above; see
    # ``jagged-sumcheck-as-uniform-fullcube-product``). The driver owns the
    # split / lift / round-poly / Fiat-Shamir / fold scan; this block only builds
    # the embedded full-cube factors, threads ``eval_start=1`` /
    # ``challenge_dtype=EF`` so the round poly is ``s_deg`` evals on ``{1..s_deg}``
    # and the fold challenge is an extension element (byte-identical to the
    # reference's compressed wire form + ``sample_ext``), and reconstructs the
    # column openings from the driver's folded state.
    n_lifts = [max(n, 0) for n in n_per_trace]
    mle_norms = [f_to_ef(f_inv_const(1 << max(-n, 0))) for n in n_per_trace]

    if n_max == 0:
        # No MLE variables: every trace is height <= 2^l_skip, so there are no
        # sumcheck rounds and the openings are the round-0-folded single rows.
        sumcheck_round_polys = []
        # ``mats`` already holds the per-trace folded view-mats (each (1, width)).
    else:
        H = 1 << n_max

        def _embed_data(col: Array, n_lift: int) -> Array:
            """Embed a ``(2^n_lift,)`` column into the full ``H`` cube, constant
            (broadcast / tiled) over the high ``n_max - n_lift`` variables."""
            return jnp.tile(col, H >> n_lift)

        def _embed_eq(n_lift: int) -> Array:
            """The full-cube eq factor: low ``n_lift`` coordinates carry
            ``ξ[l_skip:l_skip+n_lift]`` (the ``_eq_table`` LSB convention), the
            high coordinates are ``ξ=1`` (``eq(1, X) = X``), so the ``2^n_lift``-wide
            eq table sits in the top (all-ones-high) block and is zero below."""
            eq_low = _eq_table(xi[l_skip : l_skip + n_lift])  # (2^n_lift,)
            return jnp.concatenate([jnp.zeros(H - eq_low.shape[0], EF), eq_low])

        # The reference folds LSB-first (its [0::2]/[1::2] stride split); the
        # driver folds MSB-first (high/low halves). Bit-reversing the cube axis
        # of every factor makes the driver's MSB-first block fold reproduce the
        # reference's LSB-first stride fold, so the round polys, challenges, and
        # final openings all come out byte-identical.
        br = _bit_reverse_perm(H)

        state: list[Array] = []
        part_widths: list[tuple[int, ...]] = []
        for n_lift, trace_mats, trace_sels in zip(n_lifts, mats, sels):
            eq_t = _embed_eq(n_lift)[br]
            sel_cols = [_embed_data(trace_sels[:, c], n_lift)[br] for c in range(3)]
            widths = []
            mat_cols: list[Array] = []
            for m in trace_mats:
                widths.append(m.shape[1])
                for c in range(m.shape[1]):
                    mat_cols.append(_embed_data(m[:, c], n_lift)[br])
            part_widths.append(tuple(widths))
            state.extend([eq_t, *sel_cols, *mat_cols])

        mu_zc = tuple(mu_pows[2 * num_traces + t] for t in range(num_traces))
        mu_p_t = tuple(mu_pows[2 * t] for t in range(num_traces))
        mu_q_t = tuple(mu_pows[2 * t + 1] for t in range(num_traces))

        round = _BatchZerocheckRound(
            airs=tuple(airs),
            part_widths=tuple(part_widths),
            lambda_pows=tuple(lambda_pows),
            beta_pows=tuple(beta_pows),
            eq_3bs=tuple(tuple(e) for e in eq_3bs),
            mu_zc=mu_zc,
            mu_p=mu_p_t,
            mu_q=mu_q_t,
            norms=tuple(mle_norms),
            eq_n=eq_ns[0],
            eq_sharp_n=eq_sharp_ns[0],
            degree=s_deg,
        )
        folded, transcript, msgs = prove(
            round,
            state,
            transcript,
            eval_start=1,
            challenge_dtype=EF,
            challenge_limbs=EF_LIMBS,
        )
        sumcheck_round_polys = list(msgs.round_poly)
        r = [r_0] + list(msgs.challenge)

        # Reconstruct the per-trace folded view-mats from the flat folded state
        # (each factor is now (1,)): a data column embedded broadcast-constant
        # folds to its real ``mat(r_1..r_{ñ_t})`` value, so ``mats[t][i][0]`` is
        # the column opening the reference reads.
        mats = []
        offset = 0
        for t in range(num_traces):
            widths = part_widths[t]
            n_mat = sum(widths)
            mat_factors = folded[offset + 4 : offset + 4 + n_mat]
            offset += 4 + n_mat
            trace_mats = []
            c = 0
            for w in widths:
                row = jnp.stack([mat_factors[c + j][0] for j in range(w)])  # (width,)
                trace_mats.append(row[None, :])  # (1, width), so [0] reads the row
                c += w
            mats.append(trace_mats)
    _zc.mark("mle_scan", sumcheck_round_polys, mats)

    # --- Column openings: per AIR ``[common, *cached]`` ---
    # Reference ``into_column_openings`` pops the common main to the front; the
    # remaining preprocessed/cached parts follow in view order
    # (``cached.. ++ [common]``). Each part's bound row is ``trace_mats[i][0]``
    # (``mats`` is folded to the sumcheck point); under rotation the (local, rot)
    # pair interleaves per column. The synthetic fixture has no cached parts, so
    # every AIR yields just ``[common]`` and the stream stays byte-identical
    # (A4 #57 / #59 — the cached opening is what ``st.lambda`` onward needs).
    column_openings: list[list[Array]] = []
    for air, trace_mats in zip(airs, mats):
        if air.needs_next:
            # ``mats`` is the flat (local, rot, ...) list; regroup per part.
            parts = [
                jnp.stack([trace_mats[i][0], trace_mats[i + 1][0]], axis=-1).reshape(-1)
                for i in range(0, len(trace_mats), 2)
            ]
        else:
            parts = [m[0] for m in trace_mats]
        *cached, common = parts
        column_openings.append([common, *cached])

    # Observe in two passes (reference ``prove_zerocheck``): every AIR's common
    # opening, then every AIR's remaining (cached/preprocessed) parts.
    # ``column_openings_by_rot``: rotated → (local, rot) already interleaved;
    # un-rotated → each column paired with a zero.
    zero_arr = jnp.zeros((1,), EF)

    def _observe_opening(t, part, needs_next):
        if needs_next:
            return t.observe(part)
        for j in range(part.shape[0]):
            t = t.observe(part[j : j + 1])
            t = t.observe(zero_arr)
        return t

    for air, openings in zip(airs, column_openings):
        transcript = _observe_opening(transcript, openings[0], air.needs_next)
    for air, openings in zip(airs, column_openings):
        for part in openings[1:]:
            transcript = _observe_opening(transcript, part, air.needs_next)
    _zc.mark("openings", column_openings)

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
