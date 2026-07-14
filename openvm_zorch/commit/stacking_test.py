"""Stacking layout against the reference's own unit vectors.

The three manual cases are ports of the Rust tests in
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/stacked_pcs.rs#L547
so the layout semantics (head-to-tail packing, striding below ``2^l_skip``,
buffer sizing from lifted cell count) are pinned independently of any hashing.
"""

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from openvm_zorch.commit.stacking import stacked_matrix


def _single_col_traces(*cols: list[int]) -> list[jnp.ndarray]:
    return [jnp.array(c, dtype=F).reshape(-1, 1) for c in cols]


class StackingTest(absltest.TestCase):
    def test_manual_0(self) -> None:
        traces = _single_col_traces([1, 2, 3, 4], [5, 6], [7])
        mat, layout = stacked_matrix(0, 2, traces)
        self.assertEqual(mat.shape, (4, 2))
        expected = jnp.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=F).reshape(2, 4).T
        self.assertTrue(bool(jnp.array_equal(mat, expected)))
        self.assertEqual(layout.mat_starts, [0, 1, 2])

    def test_manual_strided_0(self) -> None:
        traces = _single_col_traces([1, 2, 3, 4], [5, 6], [7])
        mat, _ = stacked_matrix(2, 0, traces)
        self.assertEqual(mat.shape, (4, 3))
        expected = (
            jnp.array([1, 2, 3, 4, 5, 0, 6, 0, 7, 0, 0, 0], dtype=F).reshape(3, 4).T
        )
        self.assertTrue(bool(jnp.array_equal(mat, expected)))

    def test_manual_strided_1(self) -> None:
        traces = _single_col_traces([1, 2, 3, 4], [5, 6], [7])
        mat, _ = stacked_matrix(3, 0, traces)
        self.assertEqual(mat.shape, (8, 3))
        expected = (
            jnp.array(
                [
                    [1, 0, 2, 0, 3, 0, 4, 0],
                    [5, 0, 0, 0, 6, 0, 0, 0],
                    [7, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=F,
            )
            .reshape(3, 8)
            .T
        )
        self.assertTrue(bool(jnp.array_equal(mat, expected)))


if __name__ == "__main__":
    absltest.main()
