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

from typing import Sequence

import jax.numpy as jnp
from jax import Array
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
    rows = jnp.arange(height)
    table = jnp.stack([rows == 0, rows != height - 1, rows == height - 1], axis=-1)
    return table.astype(jnp.uint32).astype(F)


def _parts(trace: Array, needs_next: bool) -> list[tuple[Array, Array | None]]:
    """The DAG evaluator's (local, next) pairs for the common main matrix —
    no preprocessed/cached traces in scope, so one part. ``next`` is the
    cyclic row rotation only when the AIR rotates."""
    if needs_next:
        nxt = jnp.concatenate([trace[1:], trace[:1]], axis=0)
        return [(trace, nxt)]
    return [(trace, None)]


def gkr_input_evals(
    l_skip: int,
    n_logup: int,
    traces: Sequence[Array],
    dags: Sequence[ConstraintsDag],
    public_values: Sequence[Sequence[int]],
    needs_next: Sequence[bool],
    alpha: Array,
    beta: Array,
) -> tuple[Array, Array]:
    """Evaluations of ``(p̂, q̂)`` on ``H_{l_skip + n_logup}``.

    ``traces`` are ``(height, width)`` base-field matrices pre-sorted by
    descending height; ``dags[t]`` / ``public_values[t]`` / ``needs_next[t]``
    are trace ``t``'s constraint DAG (its interactions reference DAG nodes by
    index), public values, and rotation flag. Returns ``(num, den)`` BabyBear⁴
    vectors of length ``2^{l_skip + n_logup}``, with the ``q += α`` guard
    already applied.
    """
    sorted_meta = [
        (len(dag.interactions), max(log2_strict_usize(t.shape[0]), l_skip))
        for t, dag in zip(traces, dags)
    ]
    layout = interactions_layout(l_skip, n_logup, sorted_meta)

    max_msg_len = max(
        (len(i.message) for dag in dags for i in dag.interactions), default=0
    )
    one = jnp.ones((), EF)
    beta_pows = [one]
    for _ in range(max_msg_len):
        beta_pows.append(beta_pows[-1] * beta)

    # Per-trace (count, h_β(message ‖ bus)) at full trace height, from the
    # shared DAG evaluator over base-field selectors/parts — the same pairs the
    # ZeroCheck stage builds (single.rs ``eval_interactions``).
    pairs: list[list[tuple[Array, Array]]] = []
    for trace, dag, pubs, nxt in zip(traces, dags, public_values, needs_next):
        if not dag.interactions:
            pairs.append([])
            continue
        sels = _sels(trace.shape[0])
        node_vals = eval_nodes(dag, sels, _parts(trace, nxt), pubs)
        pairs.append(eval_interactions(dag, node_vals, beta_pows))

    modulus = pfinfo(F).modulus
    size = 1 << (l_skip + n_logup)
    num = jnp.zeros(size, EF)
    den = jnp.zeros(size, EF)
    for trace_idx, int_idx, s in layout.sorted_cols:
        count, denom = pairs[trace_idx][int_idx]
        height = traces[trace_idx].shape[0]
        length = 1 << s.log_height

        # eval_interactions returns count in whatever field its expression
        # evaluates to (base when a column/constant product, as the synthetic
        # fixture is; extension once a challenge enters); promote to EF up front
        # so num is uniformly EF and the lift norm stays in one field.
        count = f_to_ef(count) if count.dtype == F else count
        reps = length // height
        if length != height:
            # Lifting repeats the rows cyclically; the numerator carries the
            # inverse lift factor so the fraction-sum is unchanged.
            norm = f_to_ef(jnp.array(pow(reps, modulus - 2, modulus), F))
            count = count * norm

        num = num.at[s.row_idx : s.row_idx + length].set(jnp.tile(count, reps))
        den = den.at[s.row_idx : s.row_idx + length].set(jnp.tile(denom, reps))

    return num, den + alpha
