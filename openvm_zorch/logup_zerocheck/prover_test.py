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

import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from frx import lax
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

    def _load_case(self):
        """Build the Stage-3 inputs (sorted AIRs, ξ, β) from the fixture."""
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
        beta = ef_from_limbs(
            jnp.array(np.load(_FIXTURE / "outputs" / "beta.npy"), jnp.uint32)
        )
        return meta, l_skip, airs, xi, beta

    def _prove(self, meta, l_skip, airs, xi, beta):
        t = _replay_log(self.values, self.is_sample, meta["stage2_end"])
        return prove_batch_constraints(
            t,
            l_skip,
            meta["n_logup"],
            airs,
            xi,
            beta,
            meta["params"]["max_constraint_degree"],
        )

    def test_stage3_matches(self) -> None:
        meta, l_skip, airs, xi, beta = self._load_case()
        t, proof = self._prove(meta, l_skip, airs, xi, beta)

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

    def test_stage3_reuse_after_chain_rebuild(self) -> None:
        """A rebuilt chain over the same AIR set must re-trace the whole-stage
        jit without leaking the first trace's tracers into the reused round-0 /
        MLE-scan kernels, and stay byte-identical (#45).

        ``verify_prove``'s warm pass assembles a fresh chain, so the whole-stage
        cache misses and the stage re-traces while ``_ROUND0_FNS`` / ``_MLE_SCAN_FNS``
        are already populated. Building those inner kernels lazily under the first
        trace captured its tracers, which then escaped into the second trace
        (``UnexpectedTracerError``); the fix pre-builds them eagerly. Clearing
        ``_STAGE_FNS`` reproduces the rebuild without touching the inner caches."""
        from openvm_zorch.logup_zerocheck.prover import _STAGE_FNS

        meta, l_skip, airs, xi, beta = self._load_case()
        _, first = self._prove(meta, l_skip, airs, xi, beta)
        _STAGE_FNS.clear()
        _, second = self._prove(meta, l_skip, airs, xi, beta)

        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(first.r)), _ef_limbs(jnp.stack(second.r))
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(first.sumcheck_round_polys)),
            _ef_limbs(jnp.stack(second.sumcheck_round_polys)),
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(first.univariate_round_coeffs)),
            _ef_limbs(jnp.stack(second.univariate_round_coeffs)),
        )


class KernelCacheLifetimeTest(absltest.TestCase):
    """The id(dag)-keyed kernel caches must release their entries — and never
    pin a prove's trace arrays — once the AIR set is collected (#116 review)."""

    def test_round0_cache_evicts_when_dag_collected(self) -> None:
        import gc

        from openvm_zorch.logup_zerocheck.prover import (
            _ROUND0_FNS,
            _round0_constraint_fns,
        )

        dag = ConstraintsDag(nodes=(), constraint_idx=(), interactions=())
        before = len(_ROUND0_FNS)
        _round0_constraint_fns(dag, False, (), 2, 2)
        self.assertEqual(len(_ROUND0_FNS), before + 1)
        key_id = id(dag)
        del dag
        gc.collect()
        self.assertFalse(
            any(k[0] == key_id for k in _ROUND0_FNS),
            "round-0 cache entry survived its DAG's collection",
        )

    def test_mle_scan_cache_does_not_pin_trace(self) -> None:
        import gc
        import weakref

        from openvm_zorch.logup_zerocheck.prover import AirData, _mle_scan_fn

        dag = ConstraintsDag(nodes=(), constraint_idx=(), interactions=())
        trace = jnp.zeros((2, 1), F)
        trace_ref = weakref.ref(trace)
        air = AirData(
            trace=trace,
            dag=dag,
            public_values=(),
            constraint_degree=2,
            needs_next=False,
        )
        _mle_scan_fn([air], [1], 2, 1)
        del air, trace, dag
        gc.collect()
        self.assertIsNone(trace_ref(), "MLE-scan cache pinned the trace array")


if __name__ == "__main__":
    absltest.main()
