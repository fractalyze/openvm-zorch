"""The dense frac round marker decomposes byte-identically to the eager path.

Two kernel-free checks (no rebuilt xla plugin needed), both canonical-u32:

- ``_frac_round_decomp`` (the marker body a recognizing emitter replaces) equals
  the eager ``fold(prev challenge)`` then ``round_poly``.
- the ``_prove_layer`` restructuring (eager round 0 + deferred-fold markers +
  trailing claim fold) reproduces the original per-round ``round_poly``/``fold``
  sequence, so the deferral is an exact reassociation.
"""

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import babybear_mont as F
from zorch.sumcheck.domain import fold

from openvm_zorch.logup_gkr._round_composite import _frac_round_decomp, round_poly
from openvm_zorch.transcript import ef_from_limbs

_P = 2013265921  # BabyBear prime; any limb in [0, P) is a valid Mont encoding.


def _rand_ef(rng: np.random.Generator, shape: tuple[int, ...]):
    """A random BabyBear⁴ array of the given (element) shape."""
    limbs = rng.integers(0, _P, size=(*shape, 4), dtype=np.uint32)
    return ef_from_limbs(fnp.array(limbs))


def _limbs(x) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(fnp.atleast_1d(x), F).astype(fnp.uint32))


def _decomp(state, alpha, lam):
    """``_frac_round_decomp`` keyed off the ``(5, W)`` stacked state — unstacks to
    the kernel plane order and passes neutral Gruen operands (unread on the
    value_skip0 path). Returns the 6-tuple ``(poly, n0, n1, d0, d1, eq)``."""
    eq, n0, d0, n1, d1 = (state[i] for i in range(5))
    zero = fnp.zeros((), lam.dtype)
    return _frac_round_decomp(
        n0,
        n1,
        d0,
        d1,
        eq,
        alpha,
        zero,
        zero,
        zero,
        zero,
        lam,
        zero,
        zero,
        fnp.array([state.shape[1] // 4, 0], dtype=fnp.int32),
    )


class FracRoundCompositeTest(absltest.TestCase):
    def test_decomp_matches_eager(self) -> None:
        rng = np.random.default_rng(153)
        state = _rand_ef(rng, (5, 16))  # [eq, n0, d0, n1, d1]
        lam = _rand_ef(rng, ())
        alpha = _rand_ef(rng, ())

        # Eager reference: fold by the previous challenge, then reduce.
        folded = fold(state, alpha, msb=False)
        want_poly = round_poly(folded, lam)

        poly, n0_f, n1_f, d0_f, d1_f, eq_f = _decomp(state, alpha, lam)
        np.testing.assert_array_equal(_limbs(poly), _limbs(want_poly))
        # Folded planes match the eager fold, in the returned kernel order.
        for got, i in ((n0_f, 1), (d0_f, 2), (n1_f, 3), (d1_f, 4), (eq_f, 0)):
            np.testing.assert_array_equal(_limbs(got), _limbs(folded[i]))

    def test_deferred_fold_reproduces_original(self) -> None:
        """The eager-first / marker-mid / eager-final schedule == the original
        `for r: s_r = round_poly(state); state = fold(state, ρ_r)` loop, on a
        fixed challenge trajectory (isolates the fold deferral from the
        transcript)."""
        rng = np.random.default_rng(4242)
        rounds = 4
        state0 = _rand_ef(rng, (5, 1 << rounds))
        lam = _rand_ef(rng, ())
        rhos = [_rand_ef(rng, ()) for _ in range(rounds)]

        # Original loop (frx arrays are immutable, so state0 is preserved).
        state = state0
        want_polys = []
        for r in range(rounds):
            want_polys.append(round_poly(state, lam))
            state = fold(state, rhos[r], msb=False)
        want_claims = state[1:, 0]

        # Restructured schedule, marker body inlined (the decomp).
        def marker(st, alpha):
            poly, n0, n1, d0, d1, eq = _decomp(st, alpha, lam)
            return poly, fnp.stack([eq, n0, d0, n1, d1])

        st = state0
        got_polys = [round_poly(st, lam)]  # round 0, eager
        for r in range(1, rounds):
            p, st = marker(st, rhos[r - 1])
            got_polys.append(p)
        st = fold(st, rhos[rounds - 1], msb=False)  # trailing claim fold
        got_claims = st[1:, 0]

        self.assertEqual(len(got_polys), len(want_polys))
        for r, (g, w) in enumerate(zip(got_polys, want_polys)):
            np.testing.assert_array_equal(_limbs(g), _limbs(w), err_msg=f"poly {r}")
        np.testing.assert_array_equal(_limbs(got_claims), _limbs(want_claims))


if __name__ == "__main__":
    absltest.main()
