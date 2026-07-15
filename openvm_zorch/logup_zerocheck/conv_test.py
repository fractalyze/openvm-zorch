"""Unit test: the vectorized ``_batched_conv`` row-wise equals the reference
scalar ``_conv``. Guards the dispatch-storm rewrite of round-0's polynomial
assembly (issue #3) independently of the full byte-match in prover_test."""

import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import babybear_mont as F

from openvm_zorch.fields import EF, f_to_ef
from openvm_zorch.logup_zerocheck.prover import _batched_conv


def _conv(a: list, b: list) -> list:
    """Reference scalar polynomial convolution — the readable oracle
    ``_batched_conv`` must match row-for-row."""
    out = [jnp.zeros((), EF) for _ in range(len(a) + len(b) - 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = out[i + j] + ai * bj
    return out


def _ef_list(vals: list[int]) -> list:
    return [f_to_ef(jnp.array(v, F)) for v in vals]


def _limbs(x) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(jnp.atleast_1d(x), F).astype(jnp.uint32))


class BatchedConvTest(absltest.TestCase):
    def test_matches_scalar_conv(self) -> None:
        kernel_vals = [3, 1, 4, 1, 5]
        rows_vals = [
            [2, 7, 1, 8, 2, 8],
            [1, 1, 2, 3, 5, 8],
            [9, 9, 9, 0, 0, 1],
        ]
        kernel = jnp.stack(_ef_list(kernel_vals))
        rows = jnp.stack([jnp.stack(_ef_list(r)) for r in rows_vals])

        got = _batched_conv(rows, kernel)
        # Output length == len(row) + len(kernel) - 1.
        self.assertEqual(got.shape, (len(rows_vals), len(rows_vals[0]) + len(kernel_vals) - 1))

        for i, r in enumerate(rows_vals):
            want = jnp.stack(_conv(_ef_list(r), _ef_list(kernel_vals)))
            np.testing.assert_array_equal(_limbs(got[i]), _limbs(want))

    def test_rank1_input(self) -> None:
        # _batched_conv is also called on a 1-D coefficient vector (Island B's
        # zc_batched), so cover that rank directly.
        kernel_vals = [7, 0, 2]
        row_vals = [4, 5, 6, 7]
        kernel = jnp.stack(_ef_list(kernel_vals))
        coeffs = jnp.stack(_ef_list(row_vals))
        got = _batched_conv(coeffs, kernel)
        self.assertEqual(got.shape, (len(row_vals) + len(kernel_vals) - 1,))
        want = jnp.stack(_conv(_ef_list(row_vals), _ef_list(kernel_vals)))
        np.testing.assert_array_equal(_limbs(got), _limbs(want))


if __name__ == "__main__":
    absltest.main()
