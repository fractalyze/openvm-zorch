"""Verifier-side scalar algebra shared by the per-stage duals (poly_common.rs).

The per-stage verifier modules (``<stage>/verifier.py``) replay their stage's
transcript with host-side scalar field ops; the small-polynomial evaluation
and interpolation rules they share follow the reference's ``poly_common.rs``
(and ``batch_constraints.rs``'s barycentric form). ``VerificationError`` lives
here too: every stage dual raises it, and ``verify.py`` (which re-exports it)
imports the stage modules — defining it any higher would be an import cycle.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import frx.numpy as fnp
from frx import Array, lax

from openvm_zorch.fields import EF, F, MODULUS, f_const, f_inv_const, f_to_ef


class VerificationError(Exception):
    """Raised when any verifier check fails."""


ZERO = fnp.zeros((), EF)
ONE = fnp.ones((), EF)
_HALF = f_to_ef(f_inv_const(2))
_INV6 = f_to_ef(f_inv_const(6))
_THREE = f_to_ef(f_const(3))


def eq_u32(a: Array, b: Array) -> bool:
    """Canonical-u32 equality of two field elements (base or extension),
    independent of any custom-dtype ``__eq__``."""
    au = lax.bitcast_convert_type(fnp.atleast_1d(a), F)
    bu = lax.bitcast_convert_type(fnp.atleast_1d(b), F)
    return bool(fnp.array_equal(au, bu))


def interp_linear_01(evals: Sequence[Array], x: Array) -> Array:
    return (evals[1] - evals[0]) * x + evals[0]


def interp_quadratic_012(evals: Sequence[Array], x: Array) -> Array:
    s1 = evals[1] - evals[0]
    s2 = evals[2] - evals[1]
    p = (s2 - s1) * _HALF
    q = s1 - p
    return (p * x + q) * x + evals[0]


def interp_cubic_0123(evals: Sequence[Array], x: Array) -> Array:
    s1 = evals[1] - evals[0]
    s2 = evals[2] - evals[0]
    s3 = evals[3] - evals[0]
    d3 = s3 - (s2 - s1) * _THREE
    p = d3 * _INV6
    q = (s2 - d3) * _HALF - s1
    r = s1 - p - q
    return ((p * x + q) * x + r) * x + evals[0]


@lru_cache(maxsize=None)
def _inv_factorials(s_deg: int) -> tuple[Array, ...]:
    """``1/0!, 1/1!, …, 1/s_deg!`` as EF constants — fixed per ``s_deg``."""
    out = []
    fval = 1
    for i in range(s_deg + 1):
        if i > 0:
            fval = (fval * i) % MODULUS
        out.append(f_to_ef(f_inv_const(fval)))
    return tuple(out)


def interp_at_nodes(evals: Sequence[Array], x: Array, s_deg: int) -> Array:
    """Lagrange evaluation at ``x`` of the degree-``s_deg`` polynomial through
    nodes ``{0, 1, …, s_deg}`` (batch_constraints.rs barycentric form)."""
    invfact = _inv_factorials(s_deg)
    pref = [ONE]
    for i in range(s_deg):
        pref.append(pref[-1] * (x - f_to_ef(f_const(i))))
    suf = [ONE]
    for i in range(s_deg):
        suf.append(suf[-1] * (f_to_ef(f_const(s_deg - i)) - x))
    acc = ZERO
    for i in range(s_deg + 1):
        acc = (
            acc + evals[i] * pref[i] * suf[s_deg - i] * invfact[i] * invfact[s_deg - i]
        )
    return acc


def progression_exp_2(m: Array, l: int) -> Array:
    """``∏_{i=0}^{l-1} (1 + m^{2^i})`` (evaluator.rs ``progression_exp_2``)."""
    acc = ONE
    pow_m = m
    for _ in range(l):
        acc = acc * (ONE + pow_m)
        pow_m = pow_m * pow_m
    return acc


def exp_pow_2(x: Array, k: int) -> Array:
    for _ in range(k):
        x = x * x
    return x
