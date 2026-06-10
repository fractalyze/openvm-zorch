"""Stage 3 byte-match: batched ZeroCheck + LogUp sumcheck vs the reference.

Rebuilds the transcript state at ``stage2_end`` by replaying the recorded log
(observes fed back in, samples squeezed and asserted equal — the replay
itself re-validates zorch's transcript against the whole Stages-1/2 stream),
then drives ``prove_batch_constraints`` from raw traces + the dumped
constraint DAGs and compares every output against the fixture: λ, per-trace
sum claims, μ, the s₀ coefficients, every round polynomial, the r vector and
the column openings. Canonical-u32 equality, no tolerances.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import lax
from zk_dtypes import babybear_mont as F

from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.logup_zerocheck.prover import AirData, prove_batch_constraints
from openvm_zorch.transcript import ef_from_limbs, new_transcript

_FIXTURE = Path(__file__).parent / "testdata" / "zerocheck"


def _ef_limbs(x) -> np.ndarray:
    """Canonical-u32 limbs of a BabyBear⁴ array, shape (..., 4)."""
    return np.asarray(lax.bitcast_convert_type(jnp.atleast_1d(x), F).astype(jnp.uint32))


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


class ZerocheckByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.meta = json.loads((_FIXTURE / "meta.json").read_text())
        cls.values = np.load(_FIXTURE / "outputs" / "transcript_values.npy")
        cls.is_sample = np.load(_FIXTURE / "outputs" / "transcript_is_sample.npy")

    def test_stage3_matches(self) -> None:
        meta = self.meta
        l_skip = meta["params"]["l_skip"]

        airs = []
        for air_idx in self.meta["sorted_airs"]:
            air = meta["airs"][air_idx]
            trace = jnp.array(
                np.load(_FIXTURE / "inputs" / f"trace_{air_idx}.npy"), dtype=F
            )
            dag = ConstraintsDag.from_json(
                json.loads(
                    (_FIXTURE / "inputs" / f"constraints_{air_idx}.json").read_text()
                )
            )
            airs.append(
                AirData(
                    trace=trace,
                    dag=dag,
                    public_values=tuple(air["public_values"]),
                    constraint_degree=air["constraint_degree"],
                    needs_next=air["needs_next"],
                )
            )

        xi_rows = np.load(_FIXTURE / "outputs" / "xi.npy")
        xi = [ef_from_limbs(jnp.array(row, jnp.uint32)) for row in xi_rows]
        beta = ef_from_limbs(jnp.array(np.load(_FIXTURE / "outputs" / "beta.npy"), jnp.uint32))

        t = _replay_log(self.values, self.is_sample, meta["stage2_end"])
        t, proof = prove_batch_constraints(
            t,
            l_skip,
            meta["n_logup"],
            airs,
            xi,
            beta,
            meta["params"]["max_constraint_degree"],
        )

        np.testing.assert_array_equal(
            _ef_limbs(proof.lambda_)[0], np.load(_FIXTURE / "outputs" / "lambda.npy")
        )
        want_claims = np.load(_FIXTURE / "outputs" / "sum_claims.npy")
        for trace_idx, (p, q) in enumerate(
            zip(proof.numerator_term_per_air, proof.denominator_term_per_air)
        ):
            np.testing.assert_array_equal(
                _ef_limbs(p)[0], want_claims[trace_idx, 0], err_msg=f"sum_p {trace_idx}"
            )
            np.testing.assert_array_equal(
                _ef_limbs(q)[0], want_claims[trace_idx, 1], err_msg=f"sum_q {trace_idx}"
            )
        np.testing.assert_array_equal(
            _ef_limbs(proof.mu)[0], np.load(_FIXTURE / "outputs" / "mu.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(proof.univariate_round_coeffs)),
            np.load(_FIXTURE / "outputs" / "s0_coeffs.npy"),
        )
        want_rounds = np.load(_FIXTURE / "outputs" / "round_polys.npy")
        self.assertEqual(len(proof.sumcheck_round_polys), meta["n_max"])
        for j, evals in enumerate(proof.sumcheck_round_polys):
            np.testing.assert_array_equal(
                _ef_limbs(evals), want_rounds[j], err_msg=f"round {j + 1}"
            )
        want_r = np.load(_FIXTURE / "outputs" / "r.npy")
        np.testing.assert_array_equal(_ef_limbs(jnp.stack(proof.r)), want_r)
        for trace_idx, openings in enumerate(proof.column_openings):
            for part_idx, part in enumerate(openings):
                want = np.load(
                    _FIXTURE
                    / "outputs"
                    / f"column_openings_t{trace_idx}_p{part_idx}.npy"
                )
                np.testing.assert_array_equal(
                    _ef_limbs(part), want, err_msg=f"openings t{trace_idx} p{part_idx}"
                )

        # The advanced transcript must sit exactly at stage3_end: one more
        # squeeze must reproduce the next recorded sample (Stage 4's first).
        end = meta["stage3_end"]
        self.assertTrue(bool(self.is_sample[end]))
        t, nxt = t.sample(1)
        got = int(np.asarray(lax.bitcast_convert_type(nxt, F).astype(jnp.uint32))[0])
        self.assertEqual(got, int(self.values[end]))


if __name__ == "__main__":
    absltest.main()
