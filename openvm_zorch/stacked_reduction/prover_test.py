"""Stage 4 byte-match: stacked opening reduction vs the reference.

Rebuilds the transcript state at ``stage3_end`` by replaying the recorded log
(observes fed back in, samples squeezed and asserted equal), restacks the
sorted traces with the Stage-1 code (asserting the stacked matrix and layout
against the dump), then drives ``prove_stacked_opening_reduction`` and
compares every output against the fixture: λ, the s₀ coefficients, every
round polynomial, the u vector and the stacking openings. The closing check
feeds Stage 5's first observe (the WHIR μ-PoW witness) and asserts the next
squeeze — the transcript state after Stage 4 is pinned, not just its values.
Canonical-u32 equality, no tolerances.
"""

import json
from pathlib import Path

import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import babybear_mont as F

from openvm_zorch.commit.stacking import stacked_matrix
from openvm_zorch.stacked_reduction.prover import prove_stacked_opening_reduction
from openvm_zorch.transcript import ef_from_limbs, new_transcript

_FIXTURE = Path(__file__).parent / "testdata" / "stacking"


def _ef_limbs(x) -> np.ndarray:
    """Canonical-u32 limbs of a BabyBear⁴ array, shape (..., 4)."""
    return np.asarray(lax.bitcast_convert_type(jnp.atleast_1d(x), F).astype(jnp.uint32))


def _to_u32(x) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(x, F).astype(jnp.uint32))


def _replay_log(values: np.ndarray, is_sample: np.ndarray, end: int):
    """Reconstruct the transcript state at log index ``end``: feed observes
    back in, squeeze at samples and assert the squeezed values match."""
    t = new_transcript()
    idx = 0
    while idx < end:
        if is_sample[idx]:
            t, got = t.sample(1)
            got = int(np.asarray(lax.bitcast_convert_type(got, F).astype(jnp.uint32))[0])
            assert got == int(values[idx]), f"sample mismatch at {idx}"
            idx += 1
        else:
            run = idx
            while run < end and not is_sample[run]:
                run += 1
            t = t.observe(jnp.array(values[idx:run], dtype=F))
            idx = run
    return t


class StackedReductionByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.meta = json.loads((_FIXTURE / "meta.json").read_text())
        cls.values = np.load(_FIXTURE / "outputs" / "transcript_values.npy")
        cls.is_sample = np.load(_FIXTURE / "outputs" / "transcript_is_sample.npy")

    def test_stage4_matches(self) -> None:
        meta = self.meta
        l_skip = meta["params"]["l_skip"]
        n_stack = meta["params"]["n_stack"]

        # Restack the sorted traces with the Stage-1 code and pin the result
        # against the reference's stacked matrix and layout.
        traces = [
            jnp.array(np.load(_FIXTURE / "inputs" / f"trace_{air_idx}.npy"), dtype=F)
            for air_idx in meta["sorted_airs"]
        ]
        stacked, layout = stacked_matrix(l_skip, n_stack, traces)
        np.testing.assert_array_equal(
            _to_u32(stacked), np.load(_FIXTURE / "outputs" / "stacked_matrix.npy")
        )
        self.assertEqual(len(layout.sorted_cols), len(meta["layout"]))
        for (mat_idx, col_in_mat, s), want in zip(layout.sorted_cols, meta["layout"]):
            self.assertEqual(mat_idx, want["mat_idx"])
            self.assertEqual(col_in_mat, want["col_in_mat"])
            self.assertEqual(s.col_idx, want["col_idx"])
            self.assertEqual(s.row_idx, want["row_idx"])
            self.assertEqual(s.log_height, want["log_height"])

        r = [
            ef_from_limbs(jnp.array(row, jnp.uint32))
            for row in np.load(_FIXTURE / "inputs" / "r.npy")
        ]

        t = _replay_log(self.values, self.is_sample, meta["stage3_end"])
        t, proof = prove_stacked_opening_reduction(
            t,
            l_skip,
            n_stack,
            [(stacked, layout)],
            [meta["needs_next"]],
            r,
        )

        np.testing.assert_array_equal(
            _ef_limbs(proof.lambda_)[0], np.load(_FIXTURE / "outputs" / "lambda.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(proof.univariate_round_coeffs),
            np.load(_FIXTURE / "outputs" / "s0_coeffs.npy"),
        )
        want_rounds = np.load(_FIXTURE / "outputs" / "round_polys.npy")
        self.assertEqual(len(proof.sumcheck_round_polys), n_stack)
        for j, evals in enumerate(proof.sumcheck_round_polys):
            np.testing.assert_array_equal(
                _ef_limbs(evals), want_rounds[j], err_msg=f"round {j + 1}"
            )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(proof.u)), np.load(_FIXTURE / "outputs" / "u.npy")
        )
        self.assertEqual(len(proof.stacking_openings), 1)
        np.testing.assert_array_equal(
            _ef_limbs(proof.stacking_openings[0]),
            np.load(_FIXTURE / "outputs" / "stacking_openings_c0.npy"),
        )

        # Transcript-state pin: Stage 5 opens with the WHIR μ-PoW grind —
        # witness observe, then the grind's check squeeze.
        idx = meta["stage4_end"]
        self.assertFalse(self.is_sample[idx])
        self.assertTrue(self.is_sample[idx + 1])
        t = t.observe(jnp.array(self.values[idx : idx + 1], dtype=F))
        _, got = t.sample(1)
        self.assertEqual(
            int(np.asarray(lax.bitcast_convert_type(got, F).astype(jnp.uint32))[0]),
            int(self.values[idx + 1]),
        )


if __name__ == "__main__":
    absltest.main()
