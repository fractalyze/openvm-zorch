"""Stage 2 byte-match: LogUp-GKR reproduces the reference prover.

Replays the full pre-GKR transcript (vk pre-hash, commitments, per-AIR
metadata, public values — all recorded by the Rust harness), runs the PoW
check, samples α/β, builds the interaction input layer, runs the fractional
sumcheck and the ξ padding — comparing every value against the fixture:
α, β, the input evaluations, per-layer claims, per-layer round polynomials,
the λ/μ/ρ challenge trajectory, q₀ and the final ξ. Canonical-u32 equality,
no tolerances.
"""

import json
from pathlib import Path

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import babybear_mont as F

from openvm_zorch.logup_gkr.input_layer import gkr_input_evals
from openvm_zorch.logup_gkr.prover import fractional_sumcheck, pad_xi
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.transcript import check_witness, ef_from_limbs, new_transcript, sample_ext

_FIXTURE = Path(__file__).parent / "testdata" / "logup_gkr"
# Interactions are expression-valued (DAG node indices); the constraint DAGs of
# the same deterministic instance live with the ZeroCheck fixture (the GKR
# fixture predates the DAG path and only dumped column-positional interactions).
_DAGS = (
    Path(__file__).parent.parent
    / "logup_zerocheck"
    / "testdata"
    / "zerocheck"
    / "inputs"
)


def _ef_limbs(x) -> np.ndarray:
    """Canonical-u32 limbs of a BabyBear⁴ array, shape (..., 4)."""
    return np.asarray(lax.bitcast_convert_type(fnp.atleast_1d(x), F).astype(fnp.uint32))


class _LogWalk:
    """Structured reader over the recorded (values, is_sample) transcript log."""

    def __init__(self, values: np.ndarray, is_sample: np.ndarray, idx: int):
        self.values = values
        self.is_sample = is_sample
        self.idx = idx

    def take(self, n: int, sampled: bool) -> np.ndarray:
        got = self.is_sample[self.idx : self.idx + n]
        assert (got == sampled).all(), f"flag mismatch at {self.idx}"
        out = self.values[self.idx : self.idx + n]
        self.idx += n
        return out


class LogupGkrByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.meta = json.loads((_FIXTURE / "meta.json").read_text())
        cls.values = np.load(_FIXTURE / "outputs" / "transcript_values.npy")
        cls.is_sample = np.load(_FIXTURE / "outputs" / "transcript_is_sample.npy")
        cls.traces = [
            fnp.array(np.load(_FIXTURE / "inputs" / f"trace_{i}.npy"), dtype=F)
            for i in range(len(cls.meta["airs"]))
        ]
        cls.dags = [
            ConstraintsDag.from_json(
                json.loads((_DAGS / f"constraints_{i}.json").read_text())
            )
            for i in range(len(cls.meta["airs"]))
        ]

    def _prelude_values(self) -> list[int]:
        """The Coordinator's pre-GKR observes, reconstructed from the inputs."""
        meta = self.meta
        out = list(meta["vk_pre_hash"]) + list(meta["common_main_commit"])
        for air in meta["airs"]:
            if not air["is_required"]:
                out.append(1)  # present flag
            # No preprocessed/cached traces in this fixture: log_height only.
            out.append(air["height"].bit_length() - 1)
            out.extend(air["public_values"])
        return out

    def test_stage2_matches(self) -> None:
        meta = self.meta
        params = meta["params"]
        l_skip = params["l_skip"]
        n_logup = meta["n_logup"]
        total_rounds = l_skip + n_logup

        # --- Prelude: reconstructed observes must equal the recorded log ---
        prelude = self._prelude_values()
        self.assertEqual(len(prelude), meta["prelude_len"])
        np.testing.assert_array_equal(
            np.asarray(prelude, dtype=np.uint32), self.values[: len(prelude)]
        )
        self.assertFalse(self.is_sample[: len(prelude)].any())
        t = new_transcript().observe(fnp.array(prelude, dtype=F))

        # --- LogUp PoW + α, β ---
        t, ok = check_witness(
            t, params["logup_pow_bits"], fnp.array(meta["logup_pow_witness"], F)
        )
        self.assertTrue(bool(ok))
        t, alpha = sample_ext(t)
        t, beta = sample_ext(t)
        np.testing.assert_array_equal(
            _ef_limbs(alpha)[0], np.load(_FIXTURE / "outputs" / "alpha.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(beta)[0], np.load(_FIXTURE / "outputs" / "beta.npy")
        )

        # --- Input layer ---
        sorted_airs = meta["sorted_airs"]
        sorted_traces = [self.traces[i] for i in sorted_airs]
        sorted_dags = [self.dags[i] for i in sorted_airs]
        sorted_pubs = [meta["airs"][i]["public_values"] for i in sorted_airs]
        # No interaction in this instance references a next-row (offset=1) node,
        # so the rotation matrix is never read; pass needs_next=False.
        needs_next = [False] * len(sorted_airs)
        # This synthetic instance has no cached-main partitions.
        cached_mains = [() for _ in sorted_airs]
        num, den = gkr_input_evals(
            l_skip,
            n_logup,
            sorted_traces,
            sorted_dags,
            sorted_pubs,
            needs_next,
            cached_mains,
            alpha,
            beta,
        )
        want_evals = np.load(_FIXTURE / "outputs" / "gkr_input_evals.npy")
        np.testing.assert_array_equal(_ef_limbs(num), want_evals[:, 0])
        np.testing.assert_array_equal(_ef_limbs(den), want_evals[:, 1])

        # --- Fractional sumcheck ---
        t, proof, xi = fractional_sumcheck(t, num, den)
        np.testing.assert_array_equal(
            _ef_limbs(proof.q0_claim)[0], np.load(_FIXTURE / "outputs" / "q0_claim.npy")
        )
        want_claims = np.load(_FIXTURE / "outputs" / "claims_per_layer.npy")
        self.assertEqual(len(proof.claims_per_layer), total_rounds)
        for j, claims in enumerate(proof.claims_per_layer):
            np.testing.assert_array_equal(
                _ef_limbs(claims), want_claims[j], err_msg=f"claims of layer {j}"
            )
        for j, polys in enumerate(proof.sumcheck_polys):
            want = np.load(_FIXTURE / "outputs" / f"sumcheck_polys_layer_{j}.npy")
            np.testing.assert_array_equal(
                _ef_limbs(polys), want, err_msg=f"round polys of layer {j}"
            )

        # --- Challenge trajectory (λ/μ/ρ) against the recorded log ---
        walk = _LogWalk(self.values, self.is_sample, meta["prelude_len"])
        walk.take(1, False)  # PoW witness
        walk.take(1, True)  # PoW sample
        walk.take(4, True)  # alpha
        walk.take(4, True)  # beta
        self.assertEqual(walk.idx, meta["idx_after_beta"])
        walk.take(4, False)  # q0
        walk.take(16, False)  # layer-1 claims
        np.testing.assert_array_equal(_ef_limbs(proof.mus[0])[0], walk.take(4, True))
        for round_ in range(1, total_rounds):
            np.testing.assert_array_equal(
                _ef_limbs(proof.lambdas[round_ - 1])[0], walk.take(4, True)
            )
            for sr in range(round_):
                walk.take(12, False)  # round poly evals (compared above)
                np.testing.assert_array_equal(
                    _ef_limbs(proof.rhos[round_ - 1][sr])[0], walk.take(4, True)
                )
            walk.take(16, False)  # claims (compared above)
            np.testing.assert_array_equal(
                _ef_limbs(proof.mus[round_])[0], walk.take(4, True)
            )

        # --- ξ padding ---
        t, xi = pad_xi(t, xi, l_skip + meta["n_global"])
        want_xi = np.load(_FIXTURE / "outputs" / "xi.npy")
        self.assertEqual(len(xi), want_xi.shape[0])
        np.testing.assert_array_equal(_ef_limbs(fnp.stack(xi)), want_xi)
        for _ in range(len(xi) - total_rounds):
            walk.take(4, True)
        self.assertEqual(walk.idx, meta["stage2_end"])

    def test_ef_from_limbs_roundtrip(self) -> None:
        limbs = fnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=fnp.uint32)
        back = lax.bitcast_convert_type(ef_from_limbs(limbs), F).astype(fnp.uint32)
        np.testing.assert_array_equal(np.asarray(back), np.asarray(limbs))


if __name__ == "__main__":
    absltest.main()
