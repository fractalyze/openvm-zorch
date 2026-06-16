"""End-to-end byte-match: ``prove`` reproduces the reference prover.

Unlike the per-stage tests — which pin each stage in isolation by replaying
the recorded transcript log up to the stage boundary — this drives the full
five-stage composition from raw inputs only: traces, constraint DAGs,
interaction specs, the vk pre-hash and system params. No log replay anywhere;
the PoW grinds run natively. What it validates is precisely the coordinator
glue the stage tests bypass: the stacking order, the protocol-derived sizes
(n_logup/n_max/n_global), the prelude observes, and every stage-to-stage
handoff (α/β → input layer, ξ → batch constraints, r → stacked reduction,
u → u_cube → WHIR).

All four stage fixtures are generated from the same deterministic instance,
so their inputs/outputs cross-reference freely. Canonical-u32 equality, no
tolerances.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import lax
from zk_dtypes import babybear_mont as F

from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.prove import AirInstance, Proof, SystemParams, prove
from openvm_zorch.transcript import new_transcript
from openvm_zorch.whir.prover import WhirConfig
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_TESTDATA = Path(__file__).parent
_GKR = _TESTDATA / "logup_gkr" / "testdata" / "logup_gkr"
_ZEROCHECK = _TESTDATA / "logup_zerocheck" / "testdata" / "zerocheck"
_STACKING = _TESTDATA / "stacked_reduction" / "testdata" / "stacking"
_WHIR = _TESTDATA / "whir" / "testdata" / "whir"
_PROVE = _TESTDATA / "testdata" / "prove"  # self-contained, production-shaped


def _poseidon2():
    perm = Poseidon2(babybear16_params())
    return (
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _ef_limbs(x) -> np.ndarray:
    """Canonical-u32 limbs of a BabyBear⁴ array, shape (..., 4)."""
    return np.asarray(lax.bitcast_convert_type(jnp.atleast_1d(x), F).astype(jnp.uint32))


def _to_u32(x) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(x, F).astype(jnp.uint32))


class ProveEndToEndTest(absltest.TestCase):
    def test_prove_matches_reference(self) -> None:
        gkr_meta = json.loads((_GKR / "meta.json").read_text())
        zc_meta = json.loads((_ZEROCHECK / "meta.json").read_text())
        whir_meta = json.loads((_WHIR / "meta.json").read_text())
        params_meta = whir_meta["params"]

        airs = []
        for air_idx, (g_air, z_air) in enumerate(
            zip(gkr_meta["airs"], zc_meta["airs"])
        ):
            trace = jnp.array(
                np.load(_GKR / "inputs" / f"trace_{air_idx}.npy"), dtype=F
            )
            dag = ConstraintsDag.from_json(
                json.loads(
                    (_ZEROCHECK / "inputs" / f"constraints_{air_idx}.json").read_text()
                )
            )
            airs.append(
                AirInstance(
                    trace=trace,
                    dag=dag,
                    public_values=tuple(g_air["public_values"]),
                    constraint_degree=z_air["constraint_degree"],
                    needs_next=z_air["needs_next"],
                    is_required=g_air["is_required"],
                )
            )

        params = SystemParams(
            l_skip=params_meta["l_skip"],
            n_stack=params_meta["n_stack"],
            log_blowup=params_meta["log_blowup"],
            logup_pow_bits=gkr_meta["params"]["logup_pow_bits"],
            max_constraint_degree=zc_meta["params"]["max_constraint_degree"],
            whir=WhirConfig(
                k=params_meta["k_whir"],
                num_queries=whir_meta["num_queries"],
                mu_pow_bits=params_meta["mu_pow_bits"],
                folding_pow_bits=params_meta["folding_pow_bits"],
                query_phase_pow_bits=params_meta["query_phase_pow_bits"],
            ),
        )

        sponge, comp = _poseidon2()
        _, proof = prove(
            new_transcript(),
            sponge,
            comp,
            params,
            gkr_meta["vk_pre_hash"],
            airs,
        )

        self._check_stage12(proof, gkr_meta)
        self._check_stage3(proof)
        self._check_stage4(proof)
        self._check_stage5(proof, whir_meta)

    def _check_stage12(self, proof: Proof, gkr_meta: dict) -> None:
        np.testing.assert_array_equal(
            _to_u32(proof.common_main_commit),
            np.asarray(gkr_meta["common_main_commit"], np.uint32),
        )
        self.assertEqual(
            int(_to_u32(jnp.atleast_1d(proof.logup_pow_witness))[0]),
            gkr_meta["logup_pow_witness"],
        )
        np.testing.assert_array_equal(
            _ef_limbs(proof.gkr_proof.q0_claim)[0], np.load(_GKR / "outputs" / "q0_claim.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(proof.xi)), np.load(_GKR / "outputs" / "xi.npy")
        )

    def _check_stage3(self, proof: Proof) -> None:
        bcp = proof.batch_constraint_proof
        np.testing.assert_array_equal(
            _ef_limbs(bcp.lambda_)[0], np.load(_ZEROCHECK / "outputs" / "lambda.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(bcp.univariate_round_coeffs)),
            np.load(_ZEROCHECK / "outputs" / "s0_coeffs.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(bcp.r)), np.load(_ZEROCHECK / "outputs" / "r.npy")
        )

    def _check_stage4(self, proof: Proof) -> None:
        sp = proof.stacking_proof
        np.testing.assert_array_equal(
            _ef_limbs(sp.lambda_)[0], np.load(_STACKING / "outputs" / "lambda.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(sp.u)), np.load(_STACKING / "outputs" / "u.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(sp.stacking_openings[0]),
            np.load(_STACKING / "outputs" / "stacking_openings_c0.npy"),
        )

    def _check_stage5(self, proof: Proof, whir_meta: dict) -> None:
        wp = proof.whir_proof
        out = _WHIR / "outputs"
        np.testing.assert_array_equal(
            _to_u32(jnp.atleast_1d(wp.mu_pow_witness)),
            np.load(out / "mu_pow_witness.npy"),
        )
        np.testing.assert_array_equal(_ef_limbs(wp.mu)[0], np.load(out / "mu.npy"))
        want_sumcheck = np.load(out / "sumcheck_polys.npy")
        for j, evals in enumerate(wp.whir_sumcheck_polys):
            np.testing.assert_array_equal(
                _ef_limbs(evals), want_sumcheck[j], err_msg=f"sumcheck round {j}"
            )
        np.testing.assert_array_equal(
            _to_u32(jnp.stack(wp.codeword_commits)),
            np.load(out / "codeword_commits.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(wp.ood_values)), np.load(out / "ood_values.npy")
        )
        np.testing.assert_array_equal(
            _to_u32(jnp.stack(wp.folding_pow_witnesses)),
            np.load(out / "folding_pow_witnesses.npy"),
        )
        np.testing.assert_array_equal(
            _to_u32(jnp.stack(wp.query_phase_pow_witnesses)),
            np.load(out / "query_phase_pow_witnesses.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(wp.final_poly), np.load(out / "final_poly.npy")
        )
        want_rows = np.load(out / "initial_opened_rows_c0.npy")
        for q, row in enumerate(wp.initial_round_opened_rows[0]):
            np.testing.assert_array_equal(
                _to_u32(row), want_rows[q], err_msg=f"initial rows, query {q}"
            )
        for r, vals in enumerate(wp.codeword_opened_values):
            want_vals = np.load(out / f"codeword_opened_values_r{r + 1}.npy")
            for q, v in enumerate(vals):
                np.testing.assert_array_equal(
                    _ef_limbs(v), want_vals[q], err_msg=f"round {r + 1}, query {q}"
                )

    def test_prove_production_params(self) -> None:
        """Same prover at production-shaped params (l_skip=4, k_whir=4) from a
        self-contained fixture. The per-stage tests only ever ran l_skip=2 /
        k_whir=3, so this is the generality check: every short trace now sits
        below 2^l_skip and takes the lifting/striding path. The fixture pins
        each stage's end-of-chain outputs; matching them all confirms the
        whole Fiat-Shamir transcript agrees under the new params."""
        meta = json.loads((_PROVE / "meta.json").read_text())
        pm = meta["params"]

        airs = []
        for air in meta["airs"]:
            air_idx = air["air_idx"]
            trace = jnp.array(
                np.load(_PROVE / "inputs" / f"trace_{air_idx}.npy"), dtype=F
            )
            dag = ConstraintsDag.from_json(
                json.loads(
                    (_PROVE / "inputs" / f"constraints_{air_idx}.json").read_text()
                )
            )
            airs.append(
                AirInstance(
                    trace=trace,
                    dag=dag,
                    public_values=tuple(air["public_values"]),
                    constraint_degree=air["constraint_degree"],
                    needs_next=air["needs_next"],
                    is_required=air["is_required"],
                )
            )

        params = SystemParams(
            l_skip=pm["l_skip"],
            n_stack=pm["n_stack"],
            log_blowup=pm["log_blowup"],
            logup_pow_bits=pm["logup_pow_bits"],
            max_constraint_degree=pm["max_constraint_degree"],
            whir=WhirConfig(
                k=pm["k_whir"],
                num_queries=meta["num_queries"],
                mu_pow_bits=pm["mu_pow_bits"],
                folding_pow_bits=pm["folding_pow_bits"],
                query_phase_pow_bits=pm["query_phase_pow_bits"],
            ),
        )

        sponge, comp = _poseidon2()
        _, proof = prove(
            new_transcript(), sponge, comp, params, meta["vk_pre_hash"], airs
        )

        out = _PROVE / "outputs"
        bcp = proof.batch_constraint_proof
        sp = proof.stacking_proof
        wp = proof.whir_proof
        # Stage 1 + 2.
        np.testing.assert_array_equal(
            _to_u32(proof.common_main_commit), np.load(out / "common_main_commit.npy")
        )
        self.assertEqual(
            int(_to_u32(jnp.atleast_1d(proof.logup_pow_witness))[0]),
            int(np.load(out / "logup_pow_witness.npy")[0]),
        )
        np.testing.assert_array_equal(
            _ef_limbs(proof.gkr_proof.q0_claim)[0], np.load(out / "q0_claim.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(proof.xi)), np.load(out / "xi.npy")
        )
        # Stage 3.
        np.testing.assert_array_equal(
            _ef_limbs(bcp.lambda_)[0], np.load(out / "zc_lambda.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(bcp.univariate_round_coeffs)),
            np.load(out / "zc_s0_coeffs.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(bcp.r)), np.load(out / "zc_r.npy")
        )
        # Stage 4.
        np.testing.assert_array_equal(
            _ef_limbs(sp.lambda_)[0], np.load(out / "st_lambda.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(sp.u)), np.load(out / "st_u.npy")
        )
        np.testing.assert_array_equal(
            _ef_limbs(sp.stacking_openings[0]), np.load(out / "st_openings_c0.npy")
        )
        # Stage 5.
        np.testing.assert_array_equal(
            _to_u32(jnp.atleast_1d(wp.mu_pow_witness)),
            np.load(out / "whir_mu_pow_witness.npy"),
        )
        np.testing.assert_array_equal(_ef_limbs(wp.mu)[0], np.load(out / "whir_mu.npy"))
        want_sumcheck = np.load(out / "whir_sumcheck_polys.npy")
        for j, evals in enumerate(wp.whir_sumcheck_polys):
            np.testing.assert_array_equal(
                _ef_limbs(evals), want_sumcheck[j], err_msg=f"whir sumcheck {j}"
            )
        np.testing.assert_array_equal(
            _to_u32(jnp.stack(wp.codeword_commits)),
            np.load(out / "whir_codeword_commits.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(jnp.stack(wp.ood_values)), np.load(out / "whir_ood_values.npy")
        )
        np.testing.assert_array_equal(
            _to_u32(jnp.stack(wp.folding_pow_witnesses)),
            np.load(out / "whir_folding_pow_witnesses.npy"),
        )
        np.testing.assert_array_equal(
            _to_u32(jnp.stack(wp.query_phase_pow_witnesses)),
            np.load(out / "whir_query_phase_pow_witnesses.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(wp.final_poly), np.load(out / "whir_final_poly.npy")
        )


if __name__ == "__main__":
    absltest.main()
