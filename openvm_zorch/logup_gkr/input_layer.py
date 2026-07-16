"""LogUp-GKR input layer: interactions → stacked fraction evaluations.

OpenVM flattens every trace's interactions into ONE hypercube
``H_{l_skip + n_logup}`` (the dense layout): per trace row and interaction the
fraction is ``count / h_β(message ‖ bus)`` with

    h_β(σ ‖ b) = β^len(σ)·(bus + 1) + Σ_j β^j·σ_j,

stacked by the same greedy layout as Stage 1 (``StackedLayout`` with striding
threshold 0 and per-interaction width 1). A trace shorter than ``2^l_skip`` is
lifted by cyclic repetition, which multiplies its fraction-sum by the lift
factor — compensated by scaling the numerator with the factor's inverse
(``2^{min(n_T, 0)}``). Off-image slots and the division-by-zero guard share
one move: ``q += α`` over the whole hypercube, so empty slots become the
additive identity ``0/α``.

Reference: ``prove_zerocheck_and_logup`` (logup_zerocheck/mod.rs) and
``EvalHelper::eval_interactions`` (logup_zerocheck/single.rs).

The (count, denom) pair per interaction is whatever its symbolic expression
evaluates to over the trace, so it is built by the shared constraint-DAG
evaluator (``eval_nodes`` + ``eval_interactions``) rather than read off fixed
columns — the same path the ZeroCheck stage uses, run here over the full,
unfolded trace height before the stacked lifting. Only the source of
(count, denom) lives in the DAG; the stacked-layout lifting below is unchanged.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Sequence

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import babybear_mont as F
from zk_dtypes import babybearx4_mont as EF
from zk_dtypes import pfinfo

from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.fields import f_to_ef
from openvm_zorch.logup_zerocheck.constraints import (
    ConstraintsDag,
    eval_interactions,
    eval_nodes,
)
from zorch.utils.bits import log2_strict_usize

# Per-AIR jitted DAG evaluators, keyed by DAG identity. The per-AIR
# eval_nodes + eval_interactions must be jitted (one kernel per AIR) or they run
# as an eager node-by-node dispatch storm — 84% of GKR's warm GPU time (#44).
# The compiled kernel is reused across proves (a fresh ``frx.jit`` per call would
# re-trace the whole node walk — the same dispatch cost). The entry is dropped by
# a ``finalize`` when the DAG is collected, so the cache neither leaks nor returns
# a stale kernel for a recycled ``id()``.
_air_eval_cache: dict[int, Callable[..., Any]] = {}


def interactions_layout(
    l_skip: int, n_logup: int, sorted_meta: Sequence[tuple[int, int]]
) -> StackedLayout:
    """Stacked layout of interaction columns inside ``H_{l_skip + n_logup}``.

    ``sorted_meta`` is ``(num_interactions, log_lifted_height)`` per trace in
    stacking order. Striding threshold is 0 — there is no univariate skip for
    GKR, so lifting is cyclic repetition handled by the caller, not striding.
    """
    return StackedLayout.new(0, l_skip + n_logup, list(sorted_meta))


def _sels(height: int) -> Array:
    """[is_first_row, is_transition, is_last_row] over the full trace height
    (the GKR input layer reads the unfolded trace, so no l_skip lift here —
    cf. ZeroCheck's ``_sels`` which lifts to ``2^l_skip``)."""
    rows = fnp.arange(height)
    table = fnp.stack([rows == 0, rows != height - 1, rows == height - 1], axis=-1)
    return table.astype(fnp.uint32).astype(F)


def _parts(
    trace: Array, cached_mains: Sequence[Array], needs_next: bool
) -> list[tuple[Array, Array | None]]:
    """The DAG evaluator's (local, next) pairs for the partitioned main, in
    order ``cached_mains ++ [common_main]`` — one pair per part, so a ``main``
    DAG node's ``part_index`` selects the right matrix. The GKR input layer
    reads the unfolded trace (no l_skip lift here). ``next`` is the cyclic row
    rotation only when the AIR rotates."""
    parts: list[tuple[Array, Array | None]] = []
    for m in (*cached_mains, trace):
        nxt = fnp.concatenate([m[1:], m[:1]], axis=0) if needs_next else None
        parts.append((m, nxt))
    return parts


def _air_pairs(
    dag: ConstraintsDag, pubs: Sequence[int], nxt: bool
) -> Callable[..., list[tuple[Array, Array]]]:
    """One AIR's ``(count, denom)`` builder — ``eval_nodes`` (the node-by-node
    DAG walk) then ``eval_interactions`` — jitted into a single kernel per AIR
    (cached; see ``_air_eval_cache`` for why).

    ``dag`` / ``pubs`` / ``nxt`` are static, so they are captured in the closure
    (``ConstraintsDag`` has dict-valued nodes ⇒ unhashable ⇒ not a
    ``static_argnum``); ``trace`` / ``cached`` / ``beta_pows`` are the traced
    args. Mirrors ZeroCheck's per-AIR jit (``_round0_constraint_fns``). The DAG is
    held weakly but is always live when the kernel traces — the caller holds it to
    invoke this — so the weak ref never resolves to ``None`` there."""
    key = id(dag)
    hit = _air_eval_cache.get(key)
    if hit is not None:
        return hit

    dag_ref = weakref.ref(dag)

    @frx.jit
    def _run(trace, cached, beta_pows):
        dag_ = dag_ref()
        sels = _sels(trace.shape[0])
        node_vals = eval_nodes(dag_, sels, _parts(trace, cached, nxt), pubs)
        return eval_interactions(dag_, node_vals, beta_pows)

    _air_eval_cache[key] = _run
    weakref.finalize(dag, _air_eval_cache.pop, key, None)
    return _run


def gkr_input_evals(
    l_skip: int,
    n_logup: int,
    traces: Sequence[Array],
    dags: Sequence[ConstraintsDag],
    public_values: Sequence[Sequence[int]],
    needs_next: Sequence[bool],
    cached_mains: Sequence[Sequence[Array]],
    alpha: Array,
    beta: Array,
) -> tuple[Array, Array]:
    """Evaluations of ``(p̂, q̂)`` on ``H_{l_skip + n_logup}``.

    ``traces`` are ``(height, width)`` base-field common-main matrices pre-sorted
    by descending height; ``dags[t]`` / ``public_values[t]`` / ``needs_next[t]``
    / ``cached_mains[t]`` are trace ``t``'s constraint DAG (its interactions
    reference DAG nodes by index), public values, rotation flag, and cached-main
    partitions (``()`` for the synthetic fixture). The partitioned main a ``main``
    node indexes is ``cached_mains[t] ++ [traces[t]]``. Returns ``(num, den)``
    BabyBear⁴ vectors of length ``2^{l_skip + n_logup}``, with the ``q += α``
    guard already applied.
    """
    sorted_meta = [
        (len(dag.interactions), max(log2_strict_usize(t.shape[0]), l_skip))
        for t, dag in zip(traces, dags)
    ]
    layout = interactions_layout(l_skip, n_logup, sorted_meta)

    max_msg_len = max(
        (len(i.message) for dag in dags for i in dag.interactions), default=0
    )
    one = fnp.ones((), EF)
    beta_pows = [one]
    for _ in range(max_msg_len):
        beta_pows.append(beta_pows[-1] * beta)

    # Per-trace (count, h_β(message ‖ bus)) at full trace height, from the
    # shared DAG evaluator over base-field selectors/parts — the same pairs the
    # ZeroCheck stage builds (single.rs ``eval_interactions``).
    pairs: list[list[tuple[Array, Array]]] = []
    for trace, dag, pubs, nxt, cached in zip(
        traces, dags, public_values, needs_next, cached_mains
    ):
        if not dag.interactions:
            pairs.append([])
            continue
        pairs.append(_air_pairs(dag, pubs, nxt)(trace, tuple(cached), beta_pows))

    modulus = pfinfo(F).modulus
    size = 1 << (l_skip + n_logup)

    # Assemble (num, den) on H by a single on-device gather rather than a
    # per-column ``.at[].set()`` scatter loop: scatters serialize on GPU (the
    # gather-not-scatter lesson from #78), and one scatter per interaction
    # column was the eager dispatch storm left in ``input_evals`` after #85.
    #
    # Each column contributes its full-height ``(count, denom)`` once to a flat
    # buffer; a static ``gather_idx`` then maps every hypercube slot to the
    # source row it reads, encoding both the column's ``row_idx`` offset and the
    # cyclic lift (``j % height``). Off-image slots point at an appended zero
    # sentinel, reproducing the additive identity ``0/α`` the old zero-init +
    # ``den += α`` guard produced. The index is built from the static layout
    # (heights / row offsets are Python ints), so it costs no device ops.
    counts: list[Array] = []
    denoms: list[Array] = []
    gather_idx = np.full(size, -1, dtype=np.int32)  # -1 → zero sentinel
    base = 0
    for trace_idx, int_idx, s in layout.sorted_cols:
        count, denom = pairs[trace_idx][int_idx]
        height = traces[trace_idx].shape[0]
        length = 1 << s.log_height

        # eval_interactions returns count in whatever field its expression
        # evaluates to (base when a column/constant product, as the synthetic
        # fixture is; extension once a challenge enters); promote to EF up front
        # so num is uniformly EF and the lift norm stays in one field.
        count = f_to_ef(count) if count.dtype == F else count
        # A count/denom whose expression is a pure constant (no trace column)
        # evaluates to a scalar — a real-block case the synthetic fixture, whose
        # interaction fields were always columns, never hit. The field is the
        # same on every row, so broadcast it to the row axis before the cyclic
        # lift below (which assumes a per-row ``(height,)`` vector).
        if count.ndim == 0:
            count = fnp.broadcast_to(count, (height,))
        if denom.ndim == 0:
            denom = fnp.broadcast_to(denom, (height,))
        reps = length // height
        if length != height:
            # Lifting repeats the rows cyclically; the numerator carries the
            # inverse lift factor so the fraction-sum is unchanged.
            norm = f_to_ef(fnp.array(pow(reps, modulus - 2, modulus), F))
            count = count * norm

        counts.append(count)
        denoms.append(denom)
        # Slot ``row_idx + j`` reads source row ``j % height`` of this column.
        gather_idx[s.row_idx : s.row_idx + length] = base + np.arange(length) % height
        base += height

    # Append the zero sentinel at index ``base`` (the total source height) and
    # route every off-image slot to it.
    zero = fnp.zeros((1,), EF)
    count_flat = fnp.concatenate([*counts, zero])
    denom_flat = fnp.concatenate([*denoms, zero])
    gather_idx[gather_idx < 0] = base

    idx = fnp.asarray(gather_idx)
    num = fnp.take(count_flat, idx, axis=0)
    den = fnp.take(denom_flat, idx, axis=0)
    return num, den + alpha
