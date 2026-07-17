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


# The stacked (count, denom) assembly + gather, cached per AIR set + trace
# heights (``id(dag)``, weakref-evicted — same discipline as ``_air_eval_cache``).
# Run eagerly it was a per-interaction-column op swarm (``f_to_ef`` / broadcast /
# lift-``norm`` / ``concatenate`` / two ``take``), ~68 ms of pure host dispatch at
# ~0 device — 60% of the whole GKR stage (#44). One jit collapses it to a single
# launch. ``gather_idx`` and the lift factors are Python-int-derived — built once
# in the builder, not re-derived per trace; ``pairs`` and ``alpha`` are the only
# device operands. Byte-identical: the same ops in the same order, one jit
# boundary (jit fuses without reassociating field ops).
_assemble_cache: dict[tuple, Callable[..., tuple[Array, Array]]] = {}


def _assemble(
    l_skip: int,
    n_logup: int,
    dags: Sequence[ConstraintsDag],
    trace_heights: Sequence[int],
    sorted_cols: Sequence[tuple[int, int, Any]],
) -> Callable[..., tuple[Array, Array]]:
    """Builder for the assembly kernel; see ``_assemble_cache``. ``pairs`` is the
    per-AIR ``(count, denom)`` list (``pairs[t][i]`` for column ``(t, i)``); the
    rest is static and closed over."""
    key = (tuple((id(d), h) for d, h in zip(dags, trace_heights)), l_skip, n_logup)
    hit = _assemble_cache.get(key)
    if hit is not None:
        return hit

    modulus = pfinfo(F).modulus
    size = 1 << (l_skip + n_logup)

    # gather_idx and the inverse lift factors are static (Python-int-derived);
    # build them once here rather than at every trace. Slot ``row_idx + j`` reads
    # source row ``j % height`` of its column; off-image slots (-1) route to a
    # zero sentinel appended at ``base``. ``norms[c]`` is column c's inverse
    # cyclic-lift factor, or None when its height needs no lift.
    gather_idx = np.full(size, -1, dtype=np.int32)
    norms: list[int | None] = []
    base = 0
    for trace_idx, _, s in sorted_cols:
        height = trace_heights[trace_idx]
        length = 1 << s.log_height
        gather_idx[s.row_idx : s.row_idx + length] = base + np.arange(length) % height
        base += height
        reps = length // height
        norms.append(pow(reps, modulus - 2, modulus) if length != height else None)
    gather_idx[gather_idx < 0] = base

    @frx.jit
    def _run(pairs, alpha):
        counts: list[Array] = []
        denoms: list[Array] = []
        for (trace_idx, int_idx, _), norm_int in zip(sorted_cols, norms):
            count, denom = pairs[trace_idx][int_idx]
            height = trace_heights[trace_idx]

            # eval_interactions returns count in whatever field its expression
            # evaluates to (base for a column/constant product; extension once a
            # challenge enters); promote to EF so num is uniformly EF.
            count = f_to_ef(count) if count.dtype == F else count
            # A pure-constant count/denom (no trace column — a real-block case the
            # column-only synthetic fixture never hit) evaluates to a scalar;
            # broadcast to the row axis the cyclic lift assumes.
            if count.ndim == 0:
                count = fnp.broadcast_to(count, (height,))
            if denom.ndim == 0:
                denom = fnp.broadcast_to(denom, (height,))
            if norm_int is not None:
                # Cyclic lift repeats rows; the numerator carries the inverse lift
                # factor so the fraction-sum is unchanged.
                count = count * f_to_ef(fnp.array(norm_int, F))

            counts.append(count)
            denoms.append(denom)

        zero = fnp.zeros((1,), EF)
        count_flat = fnp.concatenate([*counts, zero])
        denom_flat = fnp.concatenate([*denoms, zero])
        num = fnp.take(count_flat, gather_idx, axis=0)
        den = fnp.take(denom_flat, gather_idx, axis=0)
        return num, den + alpha

    _assemble_cache[key] = _run
    for d in dags:
        weakref.finalize(d, _assemble_cache.pop, key, None)
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

    # Assemble (num, den) on H inside one cached kernel (see ``_assemble``). Each
    # column contributes its full-height ``(count, denom)`` once to a flat buffer;
    # a static ``gather_idx`` maps every hypercube slot to the source row it reads,
    # encoding both the column's ``row_idx`` offset and the cyclic lift
    # (``j % height``) — a single on-device gather, not a per-column scatter
    # (scatters serialize on GPU; the gather-not-scatter lesson from #78). Off-image
    # slots point at an appended zero sentinel, reproducing the additive identity
    # ``0/α`` the old zero-init + ``den += α`` guard produced.
    trace_heights = [t.shape[0] for t in traces]
    return _assemble(l_skip, n_logup, dags, trace_heights, layout.sorted_cols)(
        pairs, alpha
    )
