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

Reuse: ``natural_domain`` (the round-poly sample domain; the LSB pairing it is
fed is identical to the reference's MLE fold), ``expand_eq_to_hypercube`` (fed
reversed ξ slices, the Stage-2 convention), ``eval_eq``,
``compute_inv_vandermonde``, ``eval_coeffs``
(coefficient-form univariate eval, O(1) graph in degree). The
prismalinear/univariate-skip pieces live in ``prism.py``; the constraint DAG
evaluator in ``constraints.py``.

Reference:
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/logup_zerocheck/mod.rs#L184
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/logup_zerocheck/cpu.rs
"""

from __future__ import annotations

import os
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import frx
import frx.numpy as fnp
from frx import Array, lax

from openvm_zorch.fields import EF, F, f_const, f_inv_const, f_to_ef
from openvm_zorch.logup_gkr.input_layer import interactions_layout
from openvm_zorch.logup_zerocheck import prism
from openvm_zorch.logup_zerocheck._round_composite import zerocheck_round_reduce
from openvm_zorch.logup_zerocheck.constraints import (
    ConstraintsDag,
    _promote,
    acc_constraints,
    acc_interactions,
    eval_nodes,
)
from openvm_zorch.transcript import sample_ext
from zorch.constraint_eval import constraint_eval
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde, eval_coeffs
from zorch.sumcheck.domain import natural_domain
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize


def _fold_pair(p0: Array, p1: Array, r: Array) -> Array:
    """Fold one split pair at challenge ``r``: ``P0 + r*(P1 − P0)``.

    Not zorch's ``sumcheck.domain.fold``, which splits the pair off the trailing
    axis: these buffers carry the cube variable *leading* (the trailing axis
    indexes a trace's selectors / columns), and the caller both splits them
    LSB-first and re-pads the dead tail, so only the combine is shared math."""
    return p0 + r * (p1 - p0)


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
    out = [f_to_ef(fnp.ones((), F))]
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
    EF reduce fault, and the shift-and-add avoids ``fnp.dot``/``@`` (both
    mis-lower on XLA — see docs/development.md). Dispatch-free and jit-fusable,
    unlike a per-scalar coefficient loop. (``conv_test`` pins it against the
    reference scalar convolution.)"""
    la = coeffs.shape[-1]
    lb = kernel.shape[-1]
    lo = la + lb - 1
    lead = coeffs.shape[:-1]
    shifts = [
        fnp.pad(coeffs, ((0, 0),) * len(lead) + ((j, lo - la - j),)) for j in range(lb)
    ]
    stacked = fnp.stack(shifts, axis=-1)  # (..., lo, lb)
    return (stacked * kernel).sum(axis=-1)


def _pad(coeffs: list[Array], n: int) -> list[Array]:
    """Truncate/zero-pad to length n (the reference's `.take(n)` +
    `unwrap_or(ZERO)` reads; trailing real coefficients past n are zero)."""
    out = list(coeffs[:n])
    while len(out) < n:
        out.append(fnp.zeros((), EF))
    return out


def _eq_table(xi: list[Array]) -> Array:
    """eq(ξ, y) on the hypercube, LSB-first in ξ (Stage-2 convention: the
    MSB-first expand gets the slice reversed)."""
    if not xi:
        return fnp.ones((1,), EF)
    return expand_eq_to_hypercube(fnp.stack(xi[::-1]), fnp.ones((), EF))


def _lift(mat: Array, l_skip: int) -> Array:
    """Cyclic lift of a short trace to height ``2^l_skip`` (the reference's
    ``% height`` indexing)."""
    height = mat.shape[0]
    if height >= 1 << l_skip:
        return mat
    return fnp.tile(mat, ((1 << l_skip) // height, 1))


def _sels(height: int, l_skip: int) -> Array:
    """The lift of [is_first_row, is_transition, is_last_row] (cpu.rs
    ``sels_per_trace_base``)."""
    lifted = max(height, 1 << l_skip)
    rows = fnp.arange(lifted) % height
    table = fnp.stack([rows == 0, rows != height - 1, rows == height - 1], axis=-1)
    return table.astype(fnp.uint32).astype(F)


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
            rot = fnp.concatenate([m[1:], m[:1]], axis=0)
            out.append(_lift(rot, l_skip))
    return out


def _dag_parts(mats: list[Array], needs_next: bool) -> list[tuple[Array, Array | None]]:
    """Group a flat (local, rot, local, rot, ...) mat list into the DAG
    evaluator's (local, next) pairs."""
    if needs_next:
        return [(mats[i], mats[i + 1]) for i in range(0, len(mats), 2)]
    return [(m, None) for m in mats]


def _inv_vandermonde_rows(degree: int) -> list[list[Array]]:
    m = compute_inv_vandermonde(degree, F)
    return [[f_to_ef(m[i, j]) for j in range(degree + 1)] for i in range(degree + 1)]


def _pack_cols(sels_cells: Array, mat_cells: list[Array]) -> Array:
    """Pack selector + matrix coset columns into a RANK-2 ``(M, 3 + Σ widths)``
    ``trace`` for ``constraint_eval`` — sp1-zorch's ``_fold_chip`` layout, which
    flattens every leading (coset / size / row) dimension into the row axis and
    stacks the scalar columns on the trailing axis.

    Matching that rank-2 shape is load-bearing on GPU: passing the natural
    higher-rank ``(num_cosets, size, rows, nc)`` trace tripped the XLA
    ``EmitConcat`` codegen (a null cmpi index → ``IntegerType::get`` SIGSEGV,
    fractalyze/zkx#754); the rank-2 form sp1 already compiles cleanly. The caller
    reshapes the folded ``(M,)`` result back to the leading shape. Column order is
    selectors then each part's columns, so the ``eval_fn`` slicing is unchanged."""
    cols = [sels_cells[..., j].reshape(-1) for j in range(sels_cells.shape[-1])]
    cols += [m[..., j].reshape(-1) for m in mat_cells for j in range(m.shape[-1])]
    return fnp.stack(cols, axis=-1)  # (M, nc)


def _stack_promote(node_vals: Sequence[Array], refs: Sequence[int]) -> Array:
    """Stack promoted DAG-node values along a new trailing axis for a fold.

    A DAG node is per-row (trace variables) or row-constant (``constant`` /
    ``public`` nodes evaluate to scalars). The reference ``acc_*`` folds mix the
    two via broadcasting addition; ``fnp.stack`` needs identical shapes, so
    broadcast the row-constants up to the common leading shape first. Byte-
    identical to weighting each node in place, since the fold is linear per
    node."""
    vals = [node_vals[r] for r in refs]
    shape = fnp.broadcast_shapes(*(v.shape for v in vals))
    return fnp.stack([fnp.broadcast_to(v, shape) for v in vals], axis=-1)


def _ceval_folds(
    dag, needs_next, public_values, sels, mats, lambda_pows, beta_pows, eq_3bs_t
):
    """The zerocheck (``Σ λ^k C_k``) and LogUp (numerator, denominator) folds of
    one trace's ``(sels, mats)`` evals via ``zorch.constraint_eval``, sharing one
    packed trace across all three (the MLE rounds evaluate zc and logup off the
    same ``node_vals``; round 0 can't reuse this because its zc and logup use
    different coset counts). Returns ``(acc, numer, denom)`` reshaped to the
    leading shape; ``numer``/``denom`` are ``None`` when the AIR has no
    interactions. Byte-identical to ``acc_constraints`` / ``acc_interactions`` —
    see ``lu_eval`` for the β-RLC flattening the denominator fold uses."""
    mat_ws = [m.shape[-1] for m in mats]
    lead = sels.shape[:-1]
    packed = _pack_cols(sels, mats)

    def _nodes(tr):
        s = tr[..., :3]
        cols, off = [], 3
        for w in mat_ws:
            cols.append(tr[..., off : off + w])
            off += w
        return eval_nodes(dag, s, _dag_parts(cols, needs_next), public_values)

    def _fold(refs, coeffs):
        def eval_fn(tr):
            return _stack_promote(_nodes(tr), refs)

        return constraint_eval(
            eval_fn, packed, fnp.stack(coeffs), live_width=packed.shape[0]
        ).reshape(lead)

    # A lookup-only AIR has no zerocheck constraints; match acc_constraints'
    # empty-fold scalar zero (an empty α-stack has nothing to fold).
    acc = (
        _fold(
            dag.constraint_idx,
            [lambda_pows[k] for k in range(len(dag.constraint_idx))],
        )
        if dag.constraint_idx
        else fnp.zeros((), EF)
    )
    if not dag.interactions:
        return acc, None, None

    numer = _fold([i.count for i in dag.interactions], list(eq_3bs_t))
    denom_refs = [m for intr in dag.interactions for m in intr.message]
    denom_coeffs = [
        eq_3bs_t[i] * beta_pows[j]
        for i, intr in enumerate(dag.interactions)
        for j in range(len(intr.message))
    ]
    bus_const = sum(
        (
            eq_3bs_t[i]
            * beta_pows[len(intr.message)]
            * f_to_ef(f_const(intr.bus_index + 1))
            for i, intr in enumerate(dag.interactions)
        ),
        fnp.zeros((), EF),
    )
    denom = (
        (_fold(denom_refs, denom_coeffs) + bus_const)
        if denom_refs
        else fnp.broadcast_to(bus_const, lead)
    )
    return acc, numer, denom


# Round-0 kernels, keyed by the identity of the AIR's DAG plus every other
# static input the kernels close over. `frx.jit` caches per wrapped callable,
# so handing it a freshly-built closure on each prove defeats that cache: every
# prove would re-trace and re-lower all 19 AIRs' node walks (~4.9k nodes), which
# costs far more than the kernels themselves take to run. `ConstraintsDag` holds
# `dict` nodes, so it is unhashable by value and cannot key a plain `lru_cache`;
# the entry keys on `id(dag)`. The kernels hold the DAG *weakly* (deref'd under
# trace, always live then — the caller holds it to invoke them), and a
# `weakref.finalize` drops the entry when the DAG is collected, so the cache
# neither leaks the DAG's kernels for the process lifetime nor returns a stale
# kernel for a recycled `id()`. Same pattern as the GKR input-layer cache
# (`logup_gkr/input_layer.py` `_air_eval_cache`).
_ROUND0_FNS: dict[tuple, tuple] = {}


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
    ``lax.ntt`` + ``int(...)`` concretization). Same DAG walk the MLE
    ``lax.scan`` jits, pure EF arithmetic, contracted (row) axis kept LAST per
    docs/development.md ⇒ byte-exact. One compile per AIR (distinct DAGs); warm/GPU reaps
    the fusion.

    Build the kernels once per AIR and reuse them across proves (see
    ``_ROUND0_FNS``). Round 0's device work is small — the whole real block is
    ~4M trace cells and its node values ~5GB, single-digit ms of GPU traffic —
    so a per-prove rebuild is not a rounding error but the dominant cost:
    caching cuts warm round 0 on the real block from ~16.7s to ~0.36s.
    """
    key = (id(dag), needs_next, public_values, l_skip, constraint_degree)
    cached = _ROUND0_FNS.get(key)
    if cached is not None:
        return cached

    num_cosets_zc = constraint_degree - 1
    dag_ref = weakref.ref(dag)  # held weakly; deref'd under trace (always live then)

    if num_cosets_zc > 0:
        inv_zerofiers = fnp.stack(
            [
                f_to_ef(
                    f_inv_const(
                        pow(prism.GENERATOR, (c + 1) << l_skip, prism.MODULUS) - 1
                    )
                )
                for c in range(num_cosets_zc)
            ]
        )

        @frx.jit
        def zc_eval(trace_sels, trace_mats, lambda_pows, eq_xi):
            dag_ = dag_ref()
            sels_cells = prism.coset_evals(l_skip, trace_sels, num_cosets_zc)
            mat_cells = [
                prism.coset_evals(l_skip, m, num_cosets_zc) for m in trace_mats
            ]
            # Fold the K constraints under λ via ``zorch.constraint_eval`` (#45):
            # the marker lets a recognizing emitter accumulate ``Σ λ^k C_k`` in
            # one kernel and never materialize the ``[..., K]`` constraint
            # tensor. ``eval_fn`` is opaque, so the coset cells ride in as one
            # packed ``trace`` (selectors ‖ each part's columns); it slices them
            # back, walks the DAG, and returns the K constraint nodes stacked on
            # the trailing axis. Byte-identical to ``eval_nodes`` +
            # ``acc_constraints`` — the composite inlines to the same fold on an
            # unrecognizing backend, and field addition is associative.
            mat_ws = [c.shape[-1] for c in mat_cells]
            lead = sels_cells.shape[:-1]  # (num_cosets, size, rows)
            packed = _pack_cols(sels_cells, mat_cells)  # (M, nc)

            def eval_fn(tr):
                sels = tr[..., :3]
                cols, off = [], 3
                for w in mat_ws:
                    cols.append(tr[..., off : off + w])
                    off += w
                node_vals = eval_nodes(
                    dag_, sels, _dag_parts(cols, needs_next), public_values
                )
                return _stack_promote(node_vals, dag_.constraint_idx)

            alpha = fnp.stack([lambda_pows[k] for k in range(len(dag_.constraint_idx))])
            acc = constraint_eval(
                eval_fn, packed, alpha, live_width=packed.shape[0]
            ).reshape(lead)  # (num_cosets, size, rows)
            weighted = acc * eq_xi[None, None, :]
            return weighted.sum(axis=2) * inv_zerofiers[:, None]  # (num_cosets, size)
    else:
        zc_eval = None

    @frx.jit
    def lu_eval(trace_sels, trace_mats, beta_pows, eq_3bs_t, eq_xi):
        dag_ = dag_ref()
        sels_cells = prism.coset_evals(l_skip, trace_sels, constraint_degree)
        mat_cells = [
            prism.coset_evals(l_skip, m, constraint_degree) for m in trace_mats
        ]
        # The two LogUp folds via ``zorch.constraint_eval`` — the same
        # no-materialize lever as the zerocheck fold above. A composite's
        # ``eval_fn`` may not close over a traced value, so β rides in the fold
        # COEFFICIENTS (the alpha operand), never inside ``eval_fn``:
        #   numerator  ``Σ_i eq_3b_i · count_i`` — eval_fn returns the count
        #     nodes, α = eq_3b.
        #   denominator ``Σ_i eq_3b_i · h_β_i``, with the inner β-RLC
        #     ``h_β_i = β^{|m_i|}·(bus_i+1) + Σ_j β^j·node[m_i[j]]`` FLATTENED into
        #     one RLC: the node terms fold under ``α_{i,j} = eq_3b_i·β^j`` (eval_fn
        #     returns those nodes) and the row-constant bus term
        #     ``Σ_i eq_3b_i·β^{|m_i|}·(bus_i+1)`` adds outside the fold.
        # Both are exact field reassociations of ``acc_interactions``, so byte-
        # identical.
        mat_ws = [c.shape[-1] for c in mat_cells]
        lead = sels_cells.shape[:-1]  # (num_cosets, size, rows)
        packed = _pack_cols(sels_cells, mat_cells)  # (M, nc)

        def _nodes(tr):
            sels = tr[..., :3]
            cols, off = [], 3
            for w in mat_ws:
                cols.append(tr[..., off : off + w])
                off += w
            return eval_nodes(dag_, sels, _dag_parts(cols, needs_next), public_values)

        def count_fn(tr):
            return _stack_promote(_nodes(tr), [i.count for i in dag_.interactions])

        numer = constraint_eval(
            count_fn, packed, fnp.stack(list(eq_3bs_t)), live_width=packed.shape[0]
        ).reshape(lead)

        denom_refs = [m for intr in dag_.interactions for m in intr.message]
        denom_coeffs = [
            eq_3bs_t[i] * beta_pows[j]
            for i, intr in enumerate(dag_.interactions)
            for j in range(len(intr.message))
        ]
        bus_const = sum(
            (
                eq_3bs_t[i]
                * beta_pows[len(intr.message)]
                * f_to_ef(f_const(intr.bus_index + 1))
                for i, intr in enumerate(dag_.interactions)
            ),
            fnp.zeros((), EF),
        )
        if denom_refs:

            def denom_fn(tr):
                return _stack_promote(_nodes(tr), denom_refs)

            denom = (
                constraint_eval(
                    denom_fn,
                    packed,
                    fnp.stack(denom_coeffs),
                    live_width=packed.shape[0],
                ).reshape(lead)
                + bus_const
            )
        else:
            denom = fnp.broadcast_to(bus_const, lead)

        p = (numer * eq_xi[None, None, :]).sum(axis=2)
        q = (denom * eq_xi[None, None, :]).sum(axis=2)
        return p, q  # each (num_cosets, size)

    _ROUND0_FNS[key] = (zc_eval, lu_eval)
    weakref.finalize(dag, _ROUND0_FNS.pop, key, None)
    return zc_eval, lu_eval


_MLE_SCAN_FNS: dict[tuple, object] = {}


def _mle_scan_fn(airs, n_per_trace, s_deg, n_max):
    """The MLE rounds 1..=n_max as one jitted ``lax.scan``, built once per AIR set.

    Eager, the round loop folds the hypercube to half its size each round, so
    every round is a fresh XLA shape → ~726 one-shot recompiles at 2^16 (#26).
    Wrapping the unrolled loop in one jit instead *regressed* compile time (one
    giant module; #33). The fix is a `lax.scan` whose body compiles once,
    independent of n_max: each trace keeps a fixed-width buffer and stride-pair
    folds in place, re-padding the dead tail with zeros — pairing LSB-stride as
    the reference's MLE fold does, not high/low halves. Front-load exhaustion
    (round > ñ_t, ñ_t static) becomes a per-trace `fnp.where` on the dynamic scan
    index instead of a Python branch. The round math is unchanged from the eager
    body — only the loop carrier moved into the scan carry.

    The scan compiles once, but *tracing* it is not free: the body walks every
    AIR's constraint DAG (~4.9k nodes on the real block), and an eagerly-invoked
    `lax.scan` re-traces its body on every call. That cost dominated the stage —
    warm `mle_scan` split as dispatch 1.537s vs block 0.043s, i.e. 97% host — so
    the scan is built here, behind a cache, and reused across proves.

    Reuse demands the body close over statics ONLY: the per-prove challenges
    (``lambda_pows``/``beta_pows``/``eq_3bs``/``mu_pows``) are arguments, not
    captures, or a cached body would replay the first prove's challenges and
    silently break byte-match. It captures ``air_statics`` — a per-AIR
    ``(weakref(dag), needs_next, public_values)`` — NOT the ``AirData`` objects:
    those hold the ``trace`` / ``cached_mains`` device arrays, which a
    module-cached closure would pin for the process lifetime. Same weak-DAG
    +`weakref.finalize` eviction and `id`-keying as ``_ROUND0_FNS``.
    """
    key = (
        tuple((id(a.dag), a.needs_next, a.public_values) for a in airs),
        tuple(n_per_trace),
        s_deg,
        n_max,
    )
    cached = _MLE_SCAN_FNS.get(key)
    if cached is not None:
        return cached

    num_traces = len(airs)
    # Capture only lightweight static per-AIR data (the DAG held weakly), never
    # the `AirData` objects — those carry `trace` / `cached_mains` device arrays,
    # which a module-cached closure would pin for the process lifetime.
    air_statics = [(weakref.ref(a.dag), a.needs_next, a.public_values) for a in airs]
    zero = fnp.zeros((), EF)
    one_ef = f_to_ef(fnp.ones((), F))
    inv_vdm = _inv_vandermonde_rows(s_deg - 1)
    domain_pts = fnp.stack(
        [f_to_ef(f_const(i)) for i in range(1, s_deg + 1)]
    )  # the {1..s_deg} round-poly sample points
    round_dom = natural_domain(s_deg - 1, EF)  # {0..s_deg-1}, the lifted MLE evals
    n_lifts = [max(n, 0) for n in n_per_trace]
    norms = [f_to_ef(f_inv_const(1 << max(-n, 0))) for n in n_per_trace]

    @frx.jit
    def run(
        sels,
        mats,
        eq_n_0,
        eq_sharp_n_0,
        r_0,
        transcript,
        prev_s_eval,
        lambda_pows,
        beta_pows,
        eq_3bs,
        mu_pows,
        eq_xi_xs,
        xi_cur_xs,
    ):
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

            for t, ((dag_ref, nxt, pubs), n_lift) in enumerate(
                zip(air_statics, n_lifts)
            ):
                dag = dag_ref()
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
                    sels_dom = round_dom.sample(bufs_sels[t][0::2], bufs_sels[t][1::2])
                    mats_dom = [
                        round_dom.sample(m[0::2], m[1::2]) for m in bufs_mats[t]
                    ]
                    # Per-round zc + logup folds via the same constraint_eval
                    # markers round 0 uses (shared packed trace — one node walk
                    # feeds both). Byte-identical to acc_constraints /
                    # acc_interactions.
                    acc, numer, denom = _ceval_folds(
                        dag,
                        nxt,
                        pubs,
                        sels_dom,
                        mats_dom,
                        lambda_pows,
                        beta_pows,
                        eq_3bs[t],
                    )
                    if dag.interactions:
                        # The eq-weighted reduce (the per-AIR launch storm) as one
                        # `zorch.sumcheck.round` marker; `_ceval_folds` above stays
                        # its own `zorch.constraint_eval` composite. Byte-identical
                        # to the inline fold when no emitter claims the marker.
                        head_zc_t, head_logup_t, zc0, p0t, q0t = zerocheck_round_reduce(
                            acc,
                            numer,
                            denom,
                            eq_xi,
                            eq_n,
                            eq_sharp_n,
                            mu_zc,
                            mu_p,
                            mu_q,
                            norm,
                            is_head,
                            s_deg=s_deg,
                        )
                        for i in range(s_deg - 1):
                            sp_head_zc[i] = sp_head_zc[i] + head_zc_t[i]
                            sp_head_logup[i] = sp_head_logup[i] + head_logup_t[i]
                    else:
                        # No interactions: only the zc head folds (the logup head
                        # would add field zero, so it is skipped, not marked).
                        zc = (acc * eq_xi[None, :]).sum(axis=1)
                        zc0 = eq_n * _row0(acc)
                        p0t = zero
                        q0t = zero
                        for i in range(s_deg - 1):
                            sp_head_zc[i] = sp_head_zc[i] + fnp.where(
                                is_head, mu_zc * zc[i + 1], zero
                            )
                else:
                    # Pure-tilde trace (height 1): eval over its single row.
                    node0 = eval_nodes(
                        dag,
                        bufs_sels[t][0],
                        _dag_parts([m[0] for m in bufs_mats[t]], nxt),
                        pubs,
                    )
                    zc0 = eq_n * acc_constraints(dag, node0, lambda_pows)
                    if dag.interactions:
                        numer0, denom0 = acc_interactions(
                            dag, node0, beta_pows, eq_3bs[t]
                        )
                        p0t = eq_sharp_n * numer0 * norm
                        q0t = eq_sharp_n * denom0
                    else:
                        p0t = zero
                        q0t = zero

                # tilde carry: init f̂-term at round ñ_t+1, then ×r each later round.
                is_init = round_idx == n_lift + 1
                is_accum = round_idx > n_lift + 1
                new_tilde_zc[t] = fnp.where(
                    is_init, zc0, fnp.where(is_accum, tilde_zc[t] * r_prev, tilde_zc[t])
                )
                new_tilde_p[t] = fnp.where(
                    is_init, p0t, fnp.where(is_accum, tilde_p[t] * r_prev, tilde_p[t])
                )
                new_tilde_q[t] = fnp.where(
                    is_init, q0t, fnp.where(is_accum, tilde_q[t] * r_prev, tilde_q[t])
                )
                tail_term = (
                    mu_zc * new_tilde_zc[t]
                    + mu_p * new_tilde_p[t]
                    + mu_q * new_tilde_q[t]
                )
                sp_tail = sp_tail + fnp.where(is_head, zero, tail_term)

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

            coeffs_arr = fnp.stack(coeffs)
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
                    fs = _fold_pair(bufs_sels[t][0::2], bufs_sels[t][1::2], r_round)
                    fs = fnp.concatenate([fs, fnp.zeros_like(fs)], axis=0)
                    new_bufs_sels.append(fnp.where(live, fs, bufs_sels[t]))
                    folded_m = []
                    for m in bufs_mats[t]:
                        fm = _fold_pair(m[0::2], m[1::2], r_round)
                        fm = fnp.concatenate([fm, fnp.zeros_like(fm)], axis=0)
                        folded_m.append(fnp.where(live, fm, m))
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
            eq_n_0,
            eq_sharp_n_0,
            r_0,
            transcript,
            prev_s_eval,
            fnp.int32(1),
        )
        final_carry, (round_polys, r_rounds) = lax.scan(
            step, init_carry, (eq_xi_xs, xi_cur_xs), length=n_max
        )
        return final_carry[1], final_carry[8], round_polys, r_rounds

    _MLE_SCAN_FNS[key] = run
    # The key embeds every AIR's id(dag); collecting any one invalidates it, so
    # evict when the first is finalized (a later pop of the same key no-ops).
    for a in airs:
        weakref.finalize(a.dag, _MLE_SCAN_FNS.pop, key, None)
    return run


_ZC_PROFILE = os.environ.get("OPENVM_ZC_PROFILE") == "1"


class _ZcProfiler:
    """Coarse, env-guarded region timer for Stage-3 localization (#45).

    No-op unless ``OPENVM_ZC_PROFILE=1``. Each ``mark`` blocks on the region's
    output arrays and prints the wall-clock since the previous mark, so a cold
    pass shows compile+run and a warm pass run-only, per region. Off by default
    so ``verify_prove``'s whole-stage ``_TimedRound`` number stays
    block-distortion-free: coarse region blocks sum coherently, but per-element
    blocks inflate badly (the #3 41.1s artifact).

    The total is split into ``host`` (elapsed before the block — tracing,
    lowering and dispatch) and ``device`` (the block itself — work that was still
    pending). Read the split before optimizing a region: ``device≈0`` means the
    GPU is not the problem and a faster kernel buys nothing. Every region here
    except ``mle_scan`` is currently ~99% host, and the stage's whole device cost
    is ~55ms — so the remaining lever is eliminating per-prove tracing and
    per-scalar dispatch, not arithmetic."""

    def __init__(self) -> None:
        self._t = time.monotonic()

    def mark(self, label: str, *outputs: object) -> None:
        if not _ZC_PROFILE:
            return
        dispatched = time.monotonic()  # host work done; device tail still pending
        frx.block_until_ready(outputs)
        now = time.monotonic()
        host, device = dispatched - self._t, now - dispatched
        print(
            f"  [zc {label}] {now - self._t:.3f}s (host={host:.3f} device={device:.3f})",
            flush=True,
        )
        self._t = now


class _NullProfiler:
    """Profiler that does nothing — passed into the jitted stage body, where the
    per-region ``block_until_ready`` is impossible (its outputs are traces). The
    eager path uses ``_ZcProfiler``; ``OPENVM_ZC_PROFILE=1`` still exercises the
    per-region split by routing the prove through the eager body."""

    def mark(self, label: str, *outputs: object) -> None:
        pass


_NULL_PROFILER = _NullProfiler()


def _prewarm_prism(l_skip: int, airs: list[AirData]) -> None:
    """Eagerly build the host-int prism weight caches this stage touches, so that
    tracing ``_stage_body`` (the whole-stage jit) hits warm ``lru_cache``s instead
    of running the ``omega_int`` / ``pow`` constructions under trace, which fault
    (``ConcretizationTypeError``, #45). Coset counts: ``constraint_degree`` (logup)
    and ``constraint_degree - 1`` (zerocheck). Must run eagerly, never under
    trace — call it before ``_stage_body`` on the eager path and before ``frx.jit``
    on the cached path."""
    for cd in {air.constraint_degree for air in airs}:
        for nc in (cd, cd - 1):
            if nc > 0:
                prism.prewarm_coset_weights(l_skip, nc)
                prism.prewarm_geom_weights(l_skip, nc)


def _stage_body(
    transcript: DuplexTranscript,
    l_skip: int,
    n_logup: int,
    airs: list[AirData],
    xi: list[Array],
    beta: Array,
    max_constraint_degree: int,
    profiler: _ZcProfiler | _NullProfiler,
) -> tuple[DuplexTranscript, tuple]:
    """The Stage-3 compute, factored out of ``prove_batch_constraints`` so it can
    run eagerly (real per-region ``profiler``) or wrapped in one cached jit (null
    profiler). Returns the transcript plus the raw proof arrays; the caller
    assembles ``BatchConstraintProof`` from them — its Python-list fields would
    otherwise bloat the jit's return pytree with per-scalar leaves."""
    num_traces = len(airs)
    n_per_trace = [log2_strict_usize(air.trace.shape[0]) - l_skip for air in airs]
    n_max = max(max(n_per_trace), 0)
    s_deg = max_constraint_degree + 1
    sp_0_deg = max_constraint_degree * ((1 << l_skip) - 1)

    zero = fnp.zeros((), EF)
    one_ef = f_to_ef(fnp.ones((), F))

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
        bits = f_to_ef(fnp.array([(b_int >> j) & 1 for j in range(n_bits)], F))
        point = fnp.stack(xi[l_skip + n_lift : l_skip + n_logup])
        eq_3bs[trace_idx][int_idx] = eval_eq(point, bits)

    # --- Batching randomness λ ---
    transcript, lam = sample_ext(transcript)
    max_num_constraints = max((len(air.dag.constraint_idx) for air in airs), default=0)
    lambda_pows = _powers(lam, max(max_num_constraints, 1))

    profiler.mark("setup", mats, sels, eq_3bs, beta_pows, lambda_pows)

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
    profiler.mark("round0", sp_zc, sp_logup)

    # --- eq♯/eq univariate factors, μ batching, sum claims, s_0 ---
    # The per-trace s'_p/s'_q · eq♯ products and the μ-batched s_0 are degree
    # ≤ s_0_deg (≤ 13–61 coeffs) but there are 2·num_traces of them — a scalar
    # _conv storm under eager dispatch. Stack the per-trace coefficient rows
    # and convolve them in one batched array op instead (issue #3); the eq♯/eq
    # kernels are exactly 1<<l_skip long, so the conv lands at s_0_deg+1 with
    # no padding. Two islands, split by the observe(claims)→sample(μ) seam.
    skip = 1 << l_skip
    eq_sharp = fnp.stack(prism.eq_sharp_uni_poly(l_skip, xi[:l_skip]))
    skip_domain_size = f_to_ef(f_const(skip))

    # Island A (pre-μ): per-trace logup products + sum claims.
    sp_p = fnp.stack([fnp.stack(_pad(p, sp_0_deg + 1)) for p, _ in sp_logup])
    sp_q = fnp.stack([fnp.stack(_pad(q, sp_0_deg + 1)) for _, q in sp_logup])
    p_prods = _batched_conv(sp_p, eq_sharp)  # (num_traces, s_0_deg+1)
    q_prods = _batched_conv(sp_q, eq_sharp)
    # Σ_D Z^j = N iff N | j: read the sum claim off the strided coefficients.
    p_claims = p_prods[:, ::skip].sum(axis=-1) * skip_domain_size
    q_claims = q_prods[:, ::skip].sum(axis=-1) * skip_domain_size

    numerator_term_per_air = []
    denominator_term_per_air = []
    for t in range(num_traces):
        transcript = transcript.observe(fnp.stack([p_claims[t], q_claims[t]]))
        numerator_term_per_air.append(p_claims[t])
        denominator_term_per_air.append(q_claims[t])

    transcript, mu = sample_ext(transcript)
    mu_pows = _powers(mu, 3 * num_traces)

    # Island B (post-μ): μ-batch the zerocheck rows, multiply in eq_D, then add
    # the μ-weighted logup products to form s_0. Contracted axes kept last so
    # the EF reduce stays jit-safe (docs/development.md).
    eq_uni = fnp.stack(prism.eq_uni_poly(l_skip, xi[0]))
    sp_zc_rows = fnp.stack(
        [fnp.stack(_pad(coeffs, sp_0_deg + 1)) for coeffs in sp_zc]
    )  # (num_traces, sp_0_deg+1)
    zc_weights = fnp.stack(mu_pows[2 * num_traces : 3 * num_traces])
    zc_batched = (sp_zc_rows.T * zc_weights).sum(axis=-1)  # (sp_0_deg+1,)
    zc_prod = _batched_conv(zc_batched, eq_uni)  # (s_0_deg+1,)

    mu_p = fnp.stack(mu_pows[0 : 2 * num_traces : 2])
    mu_q = fnp.stack(mu_pows[1 : 2 * num_traces : 2])
    s_0_arr = (
        zc_prod + (p_prods.T * mu_p).sum(axis=-1) + (q_prods.T * mu_q).sum(axis=-1)
    )
    transcript = transcript.observe(s_0_arr)
    profiler.mark("s0_assembly", s_0_arr, p_prods, q_prods)

    transcript, r_0 = sample_ext(transcript)
    prev_s_eval = eval_coeffs(s_0_arr, r_0)

    # --- Fold the prism at r_0 ---
    mats = [
        [prism.fold_ple_evals(l_skip, m, r_0) for m in trace_mats]
        for trace_mats in mats
    ]
    sels = [prism.fold_ple_evals(l_skip, s, r_0) for s in sels]
    eq_ns = [prism.eval_eq_uni(l_skip, xi[0], r_0)]
    eq_sharp_ns = [prism.eval_eq_sharp_uni(l_skip, xi[:l_skip], r_0)]
    profiler.mark("r0_fold", mats, sels, eq_ns, eq_sharp_ns, prev_s_eval)

    # --- MLE rounds 1..=n_max, built once per AIR set (see `_mle_scan_fn`) ----
    n_lifts = [max(n, 0) for n in n_per_trace]

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
                    fnp.concatenate([tab, fnp.zeros(half_t - tab.shape[0], EF)])
                )
            else:
                rows.append(fnp.zeros(half_t, EF))
        eq_xi_xs.append(  # (n_max, H_t/2)
            fnp.stack(rows) if rows else fnp.zeros((0, half_t), EF)
        )

    xi_cur_xs = (
        fnp.stack([xi[l_skip + round_ - 1] for round_ in range(1, n_max + 1)])
        if n_max >= 1
        else fnp.zeros((0,), EF)
    )
    profiler.mark("scan_setup", eq_xi_xs, xi_cur_xs)

    mats, transcript, round_polys, r_rounds = _mle_scan_fn(
        airs, tuple(n_per_trace), s_deg, n_max
    )(
        sels,
        mats,
        eq_ns[0],
        eq_sharp_ns[0],
        r_0,
        transcript,
        prev_s_eval,
        lambda_pows,
        beta_pows,
        eq_3bs,
        mu_pows,
        eq_xi_xs,
        xi_cur_xs,
    )
    profiler.mark("mle_scan", round_polys, mats)

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
                fnp.stack([trace_mats[i][0], trace_mats[i + 1][0]], axis=-1).reshape(-1)
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
    zero_arr = fnp.zeros((1,), EF)

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
    profiler.mark("openings", column_openings)

    return transcript, (
        numerator_term_per_air,
        denominator_term_per_air,
        s_0_arr,
        round_polys,
        column_openings,
        lam,
        mu,
        r_0,
        r_rounds,
    )


# The whole stage as one cached jit (#45). The eager ``_stage_body`` above is
# ~99% host: with round 0 and the MLE scan already jitted, the residual ~1s is
# the per-scalar dispatch of the glue *between* them — ``geometric_cosets_to_coeffs``,
# the coefficient assembly, the prism folds, the batched convs, the opening
# observes. Folding the whole body into one jit collapses that host dispatch to a
# single executable launch (the stage's device work is ~55ms). Fiat–Shamir is no
# barrier: ``observe``/``sample_ext`` already jit inside the MLE scan.
#
# Cached per AIR set, module-global and keyed on ``id(dag)`` — the same discipline
# as ``_ROUND0_FNS`` / ``_MLE_SCAN_FNS`` (which this jit subsumes), so a freshly
# assembled chain over the same verifying key reuses the executable. That reuse is
# load-bearing: ``verify_prove``'s warm pass rebuilds the whole chain, and a
# per-stage-instance cache would miss it and re-trace the stage (~26s vs 0.2s). The
# DAGs are held WEAKLY and the per-AIR device arrays (``trace`` + ``cached_mains``)
# ride in as operands, never closed over, so a cached entry pins neither the traces
# nor (past a weakref) the DAGs.
_STAGE_FNS: dict[tuple, object] = {}


def _stage_cache_key(
    l_skip: int, n_logup: int, max_constraint_degree: int, airs: list[AirData]
) -> tuple:
    # Trace shapes are part of the key: ``_build_stage_jit`` eagerly pre-builds the
    # inner MLE-scan kernel for these heights, and ``_MLE_SCAN_FNS`` keys on the
    # per-trace heights too. Reusing an entry across a height change would let
    # ``_stage_body`` build that kernel lazily under the outer trace — the tracer
    # leak the eager pre-build exists to prevent.
    return (
        tuple(
            (id(a.dag), a.public_values, a.constraint_degree, a.needs_next, a.trace.shape)
            for a in airs
        ),
        l_skip,
        n_logup,
        max_constraint_degree,
    )


def _build_stage_jit(
    l_skip: int, n_logup: int, max_constraint_degree: int, airs: list[AirData]
):
    """Prewarm the prism host-int caches, then jit ``_stage_body`` for this AIR
    set: the AIR structure is closed over statically (DAGs held weakly) and the
    device arrays are threaded through ``run`` as operands."""
    _prewarm_prism(l_skip, airs)
    # Build the round-0 and MLE-scan kernels EAGERLY, before the outer trace.
    # ``_stage_body`` calls their builders (``_ROUND0_FNS`` / ``_MLE_SCAN_FNS``);
    # built lazily under the outer trace instead, their module-cached closures
    # would capture that trace's tracers and — persisting across proves — leak
    # them into the next prove's trace (``UnexpectedTracerError`` on the warm
    # pass). Pre-building here forces eager constants, the same discipline as the
    # prism prewarm above.
    n_per_trace = [log2_strict_usize(a.trace.shape[0]) - l_skip for a in airs]
    n_max = max(max(n_per_trace), 0)
    s_deg = max_constraint_degree + 1
    for a in airs:
        _round0_constraint_fns(
            a.dag, a.needs_next, a.public_values, l_skip, a.constraint_degree
        )
    _mle_scan_fn(airs, tuple(n_per_trace), s_deg, n_max)
    meta = [
        (weakref.ref(a.dag), a.public_values, a.constraint_degree, a.needs_next)
        for a in airs
    ]

    @frx.jit
    def run(transcript, xi, beta, traces, cached_mains):
        airs_ = [
            AirData(
                trace=traces[t],
                dag=dag_ref(),
                public_values=pubs,
                constraint_degree=cd,
                needs_next=nxt,
                cached_mains=tuple(cached_mains[t]),
            )
            for t, (dag_ref, pubs, cd, nxt) in enumerate(meta)
        ]
        return _stage_body(
            transcript,
            l_skip,
            n_logup,
            airs_,
            xi,
            beta,
            max_constraint_degree,
            _NULL_PROFILER,
        )

    return run


def _assemble_proof(arrays: tuple) -> BatchConstraintProof:
    (num, den, s_0_arr, round_polys, column_openings, lam, mu, r_0, r_rounds) = arrays
    return BatchConstraintProof(
        numerator_term_per_air=num,
        denominator_term_per_air=den,
        univariate_round_coeffs=list(s_0_arr),
        sumcheck_round_polys=list(round_polys),
        column_openings=column_openings,
        lambda_=lam,
        mu=mu,
        r=[r_0] + list(r_rounds),
    )


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
    ``prove_zerocheck_and_logup`` after the ξ padding.

    Runs the whole stage as one cached jit, built once per AIR set and reused
    across proves via the module-global ``_STAGE_FNS`` (keyed on ``id(dag)``, so a
    freshly assembled chain over the same verifying key reuses it — the reuse that
    ``verify_prove``'s warm pass and repeated proves both depend on).
    ``OPENVM_ZC_PROFILE=1`` routes through the eager body instead, for the
    per-region host/device split (the jit erases the region boundaries)."""
    if _ZC_PROFILE:
        _prewarm_prism(l_skip, airs)
        transcript, arrays = _stage_body(
            transcript,
            l_skip,
            n_logup,
            airs,
            xi,
            beta,
            max_constraint_degree,
            _ZcProfiler(),
        )
        return transcript, _assemble_proof(arrays)

    key = _stage_cache_key(l_skip, n_logup, max_constraint_degree, airs)
    run = _STAGE_FNS.get(key)
    if run is None:
        run = _build_stage_jit(l_skip, n_logup, max_constraint_degree, airs)
        _STAGE_FNS[key] = run
        # Evict when any AIR's DAG is collected (a later pop of the same key
        # no-ops), so a cached executable never leaks nor serves a recycled id().
        for a in airs:
            weakref.finalize(a.dag, _STAGE_FNS.pop, key, None)
    traces = [a.trace for a in airs]
    cached_mains = [list(a.cached_mains) for a in airs]
    transcript, arrays = run(transcript, xi, beta, traces, cached_mains)
    return transcript, _assemble_proof(arrays)
