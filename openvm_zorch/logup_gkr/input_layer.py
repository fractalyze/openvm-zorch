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

Interactions are described positionally (count column with a sign, message
columns) — the shape ``DummyInteractionAir`` pushes — rather than as symbolic
expressions; a general constraint-DAG evaluator is a later stage's concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import Array
from zk_dtypes import babybear_mont as F
from zk_dtypes import babybearx4_mont as EF
from zk_dtypes import pfinfo

from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.fields import f_to_ef  # noqa: F401  (re-exported; Stage-2 API)
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class InteractionSpec:
    """One interaction: ``±count_col`` over ``message_cols`` on ``bus``."""

    bus: int
    count_col: int
    count_neg: bool
    message_cols: tuple[int, ...]


def interactions_layout(
    l_skip: int, n_logup: int, sorted_meta: Sequence[tuple[int, int]]
) -> StackedLayout:
    """Stacked layout of interaction columns inside ``H_{l_skip + n_logup}``.

    ``sorted_meta`` is ``(num_interactions, log_lifted_height)`` per trace in
    stacking order. Striding threshold is 0 — there is no univariate skip for
    GKR, so lifting is cyclic repetition handled by the caller, not striding.
    """
    return StackedLayout.new(0, l_skip + n_logup, list(sorted_meta))


def gkr_input_evals(
    l_skip: int,
    n_logup: int,
    traces: Sequence[Array],
    interactions: Sequence[Sequence[InteractionSpec]],
    alpha: Array,
    beta: Array,
) -> tuple[Array, Array]:
    """Evaluations of ``(p̂, q̂)`` on ``H_{l_skip + n_logup}``.

    ``traces`` are ``(height, width)`` base-field matrices pre-sorted by
    descending height; ``interactions[t]`` are trace ``t``'s specs. Returns
    ``(num, den)`` BabyBear⁴ vectors of length ``2^{l_skip + n_logup}``, with
    the ``q += α`` guard already applied.
    """
    sorted_meta = [
        (len(ints), max(log2_strict_usize(t.shape[0]), l_skip))
        for t, ints in zip(traces, interactions)
    ]
    layout = interactions_layout(l_skip, n_logup, sorted_meta)

    max_msg_len = max(
        (len(i.message_cols) for ints in interactions for i in ints), default=0
    )
    one = jnp.ones((), EF)
    beta_pows = [one]
    for _ in range(max_msg_len):
        beta_pows.append(beta_pows[-1] * beta)

    modulus = pfinfo(F).modulus
    size = 1 << (l_skip + n_logup)
    num = jnp.zeros(size, EF)
    den = jnp.zeros(size, EF)
    for trace_idx, int_idx, s in layout.sorted_cols:
        trace = traces[trace_idx]
        spec = interactions[trace_idx][int_idx]
        height = trace.shape[0]
        length = 1 << s.log_height

        count = trace[:, spec.count_col]
        if spec.count_neg:
            count = -count
        if length != height:
            # Lifting repeats the rows cyclically; the numerator carries the
            # inverse lift factor so the fraction-sum is unchanged.
            norm = jnp.array(pow(length // height, modulus - 2, modulus), F)
            count = count * norm
        denom = beta_pows[len(spec.message_cols)] * jnp.array(spec.bus + 1, F)
        denom = denom + sum(
            beta_pows[j] * trace[:, col] for j, col in enumerate(spec.message_cols)
        )

        reps = length // height
        num = num.at[s.row_idx : s.row_idx + length].set(
            jnp.tile(f_to_ef(count), reps)
        )
        den = den.at[s.row_idx : s.row_idx + length].set(jnp.tile(denom, reps))

    return num, den + alpha
