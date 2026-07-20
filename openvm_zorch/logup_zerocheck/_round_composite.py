"""``zorch.sumcheck.round`` marker, variant=openvm-zerocheck.

Wraps openvm's per-AIR prismalinear ZeroCheck+LogUp round REDUCE -- the
eq-weighted fold of one AIR's constraint/LogUp evals over the current
hypercube -- so a recognizing emitter fuses it into one kernel.
``constraint_eval`` (the DAG walk producing ``acc``/``numer``/``denom``) stays
EXTERNAL, exactly as sp1-zorch's ``zerocheck_round_poly`` keeps its summand
value external: the marker encodes only the generic per-AIR reduce shape and
the summand values ride in as operands.

The reduce is the eq-weighted ``sum(axis=1)`` over the hypercube -- the
per-AIR-per-round launch storm (thousands of tiny select/add/reduce kernels,
GPU ~95% idle) the emitter collapses. The cheap scalar tilde front-load carry
stays OUTSIDE the marker (in ``_mle_scan_fn``), shared across every AIR class,
so the marker body is a pure straight-through reduce.

openvm's round is its own ``variant`` (not sp1's jagged ``variant="zerocheck"``
nor the LogUp-GKR ``variant=dense/jagged``): it combines the ZeroCheck
constraint fold (``acc``) with the LogUp numerator/denominator folds
(``numer``/``denom``) over the l_skip coset structure, at a per-AIR-set degree
``s_deg = max_constraint_degree + 1``.

An unclaimed marker decomposes inline via ``_decomp``, byte-identical to the
eager reduce (field ops are exact, so the reassociation matches lane for lane).
The variant has no GPU emitter yet -- see
``openvm_zorch/logup_zerocheck`` docs / issue #138: the producer here is the
zorch-side half, co-landed with the ``zorch.sumcheck.round`` emitter that
recognizes ``variant="openvm-zerocheck"``.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from zk_dtypes import babybearx4_mont as EF

from zorch._composite import composite
from zorch.sumcheck.prover import (
    SUMCHECK_ROUND_MARKER,
    SUMCHECK_ROUND_MARKER_VERSION,
)


def _row0(a: Array) -> Array:
    """The fully-folded f̂ value at the buffer origin (matches prover._row0): a
    ``(s_deg, half)`` round-eval array's first cell. A trace with no zerocheck
    constraints (resp. no interactions) accumulates to a 0-d zero, already that
    cell's value, so pass it through."""
    return a[0, 0] if a.ndim == 2 else a


def _decomp(
    acc: Array,
    numer: Array,
    denom: Array,
    eq_xi: Array,
    eq_n: Array,
    eq_sharp_n: Array,
    mu_zc: Array,
    mu_p: Array,
    mu_q: Array,
    norm: Array,
    is_head: Array,
    *,
    degree: int,
    **_attrs: object,
) -> tuple[Array, Array, Array, Array, Array]:
    """Byte-exact fallback (the emitter replaces this) for one LIVE
    (``n_lift >= 1``) interaction AIR's reduce: the eq-weighted zc/logup folds
    over the hypercube, plus the origin-cell tilde bases. ``degree`` is
    ``s_deg`` (rides as the marker's ``degree`` attribute).

    Returns ``(head_zc[degree-1], head_logup[degree-1], zc0, p0t, q0t)`` -- the
    round poly's live ``{1..s_deg-1}`` head contributions and the three
    exhausted-trace tilde bases the caller folds into its scan carry.
    """
    zero = fnp.zeros((), EF)
    zc = (acc * eq_xi[None, :]).sum(axis=1)
    zc0 = eq_n * _row0(acc)
    p = (numer * eq_xi[None, :]).sum(axis=1) * norm
    q = (denom * eq_xi[None, :]).sum(axis=1)
    p0t = eq_sharp_n * _row0(numer) * norm
    q0t = eq_sharp_n * _row0(denom)
    head_zc = fnp.stack(
        [fnp.where(is_head, mu_zc * zc[i + 1], zero) for i in range(degree - 1)]
    )
    head_logup = fnp.stack(
        [
            fnp.where(is_head, mu_p * p[i + 1] + mu_q * q[i + 1], zero)
            for i in range(degree - 1)
        ]
    )
    return head_zc, head_logup, zc0, p0t, q0t


def zerocheck_round_reduce(
    acc: Array,
    numer: Array,
    denom: Array,
    eq_xi: Array,
    eq_n: Array,
    eq_sharp_n: Array,
    mu_zc: Array,
    mu_p: Array,
    mu_q: Array,
    norm: Array,
    is_head: Array,
    *,
    s_deg: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Emit the variant=openvm-zerocheck ``zorch.sumcheck.round`` marker around
    one LIVE interaction AIR's round reduce.

    ``acc``/``numer``/``denom`` are the AIR's ``constraint_eval`` folds (shape
    ``(s_deg, size)``: the eval at each of the ``{1..s_deg}`` lifted points over
    the current hypercube of ``size`` cells). Returns ``(head_zc[s_deg-1],
    head_logup[s_deg-1], zc0, p0t, q0t)`` -- the per-AIR contributions the
    caller sums into the round poly and folds into the tilde carry.
    """
    return composite(
        _decomp,
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
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="mid",
        variant="openvm-zerocheck",
        degree=s_deg,
        poly_form="evals",
    )
