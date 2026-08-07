"""``zorch.sumcheck.round`` marker for SWIRL's fractional sumcheck round.

Wraps one dense LogUp-GKR round's ``fold(prev challenge) + round_poly`` in a
``zorch.sumcheck.round`` composite (``variant=dense``, ``poly_form=value_skip0``)
so a recognizing emitter fuses the round into one kernel; when unclaimed (a
pre-#153 xla plugin, or a plain-Python trace) the ``lax.composite`` body
decomposes inline, byte-identical to the eager ``fold``/``_round_poly`` path.

Why ``variant=dense`` and not a new one: the dense round kernel already folds
the four MLE planes by the previous challenge with an **LSB stride-2** pair fold
(``sumcheck.cc`` ``EmitDenseSumcheckRound``) -- the same adjacent-pair fold as
``fold(..., msb=False)`` here -- and carries ``eq`` as a folded fifth plane used
as the LogUp combine weight. openvm's frac round differs from sp1's dense round
on only two axes, both orthogonal knobs (see xla ``sumcheck_scalar.h`` axis map):

- **message form** -- openvm sends the round poly as evals on ``{1, 2, 3}``
  (``value_skip0``: the verifier derives ``s(0)`` from the running claim), not
  sp1's Gruen ``coefficient`` form. ``poly_form="value_skip0"`` selects the
  {1,2,3}-eval finalize (no Gruen), so the eq-factoring scalars
  (``eq_adj``/``pad_adj``/``z_cur``/``claim``) go unread -- passed neutral below.
- **summand** -- openvm batches the *denominator*
  (``eq·((p0·q1 + p1·q0) + λ·q0·q1)``); sp1's ``LogupCombine`` batches the
  numerator. The value_skip0 dense kernel path uses openvm's denominator-λ
  combine (value_skip0 ⇒ SWIRL univariate-skip ⇒ this combine; the two stay
  coupled until a genuinely orthogonal consumer appears, per the closed-by-
  construction enum rule in xla ``round_poly_form.h``).

Reference: ``fractional_sumcheck`` (logup_zerocheck/fractional_sumcheck_gkr.rs).
Mirror of ``openvm_zorch/logup_zerocheck/_round_composite.py`` (the openvm-
zerocheck marker) for the dense frac variant.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from zorch._composite import composite
from zorch.sumcheck.domain import fold
from zorch.sumcheck.prover import (
    SUMCHECK_ROUND_MARKER,
    SUMCHECK_ROUND_MARKER_VERSION,
)

# Round poly degree: eq (deg 1) * projective fraction addition (deg 2) = 3, so
# four evals {0,1,2,3} determine it -- but the prover sends only {1,2,3} (the
# verifier infers s(0) from the running claim s(0)+s(1) = prev). Lifting to the
# SENT domain skips the discarded u=0 across all five MLEs.
_DEGREE = 3
_SENT_US = tuple(range(1, _DEGREE + 1))


def _lift_sent(state: Array) -> Array:
    """Lift the paired state to the SENT eval domain ``{1,2,3}`` (skips u=0).

    ``state`` is the stacked ``(5, W)`` round state; the LSB pairing splits it
    to ``lo``/``hi`` ``(5, W/2)`` and ``f[u] = lo + u*(hi - lo)`` lifts all
    five MLEs in ONE broadcast FMA, shape ``(3, 5, W/2)``. ``us`` uses
    ``fnp.stack`` (not ``fnp.arange``, whose iota is unsupported for extension
    dtypes)."""
    lo, hi = state[:, 0::2], state[:, 1::2]
    us = fnp.stack([fnp.array(u, dtype=lo.dtype) for u in _SENT_US])
    return lo + us.reshape((-1, 1, 1)) * (hi - lo)


def round_poly(state: Array, lam: Array) -> Array:
    """The sent round poly s(1,2,3). λ weights the denominator term -- opposite
    of ``logup_combine``. Binds the LSB: pairs adjacent entries (the reference's
    MLE fold). Field ops are exact, so the batched lift/products are byte-
    identical to a per-MLE form and to any reduction order the emitter picks."""
    f = _lift_sent(state)  # (3, 5, W/2)
    eq, p0, q0, p1, q1 = (f[:, i] for i in range(5))
    return fnp.sum(eq * ((p0 * q1 + p1 * q0) + lam * (q0 * q1)), axis=-1)


def _frac_round_decomp(
    n0: Array,
    n1: Array,
    d0: Array,
    d1: Array,
    eq: Array,
    alpha: Array,
    eq_adj: Array,
    pad_adj: Array,
    z_cur: Array,
    claim: Array,
    lam: Array,
    naturals: Array,
    inv_vand: Array,
    live: Array,
    **_attrs: object,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Byte-exact fallback for one dense frac round (the emitter replaces this).

    Folds the five planes (state order ``[eq, n0, d0, n1, d1]``) by the previous
    round's challenge ``alpha`` (LSB stride-2), then reduces the folded state to
    the sent round poly. ``eq_adj``/``pad_adj``/``z_cur``/``claim`` and
    ``naturals``/``inv_vand`` are unread on the value_skip0 path (they carry the
    Gruen eq-factorization the coefficient form needs); ``**_attrs`` is the
    composite metadata (phase/variant/degree/poly_form) the emitter parses.

    Returns ``(poly, folded n0, n1, d0, d1, eq)`` -- the kernel-order mid-phase
    result tuple; each folded plane is half the input width (openvm's exact
    layout: the reduce covers the whole live prefix, so no zero pad)."""
    del eq_adj, pad_adj, z_cur, claim, naturals, inv_vand, live, _attrs
    state = fnp.stack([eq, n0, d0, n1, d1])
    folded = fold(state, alpha, msb=False)
    poly = round_poly(folded, lam)
    eq_f, n0_f, d0_f, n1_f, d1_f = (folded[i] for i in range(5))
    return poly, n0_f, n1_f, d0_f, d1_f, eq_f


def frac_round(state: Array, lam: Array, alpha: Array) -> tuple[Array, Array]:
    """Emit the dense frac round marker: fold ``state`` (``(5, W)``, order
    ``[eq, n0, d0, n1, d1]``) by ``alpha`` then reduce, returning the sent round
    poly and the folded ``(5, W/2)`` state.

    The marker carries the 14-operand dense ``mid`` ABI so the round kernel
    selects it in place; the eq-factoring scalars ride as neutral operands
    (unread by the value_skip0 finalize). ``live[0]`` is the round's live reduce
    pairs = ``W // 4`` (each thread folds four input elements into one reduce
    pair). Byte-identical to eager ``fold``/``round_poly`` when unclaimed."""
    eq, n0, d0, n1, d1 = (state[i] for i in range(5))
    width = state.shape[1]
    zero = fnp.zeros((), lam.dtype)
    poly, n0_f, n1_f, d0_f, d1_f, eq_f = composite(
        _frac_round_decomp,
        n0,
        n1,
        d0,
        d1,
        eq,
        alpha,
        zero,  # eq_adj  (unread: value_skip0 skips the Gruen eq-factorization)
        zero,  # pad_adj
        zero,  # z_cur
        zero,  # claim
        lam,
        zero,  # naturals (degree-derived, unread by value_skip0)
        zero,  # inv_vand
        fnp.array([width // 4, 0], dtype=fnp.int32),  # live
        name=SUMCHECK_ROUND_MARKER,
        version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="mid",
        variant="dense",
        degree=_DEGREE,
        poly_form="value_skip0",
    )
    return poly, fnp.stack([eq_f, n0_f, d0_f, n1_f, d1_f])
