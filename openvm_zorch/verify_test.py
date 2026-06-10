"""End-to-end verifier: accept honest proofs, reject tampered ones.

Proves the production-params instance with ``prove`` (no recorded log), then
checks ``verify`` accepts the result. The negative cases perturb one field of
the proof at a time and assert the verifier rejects — exercising each stage's
algebraic check and the Merkle-path verification.
"""

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from openvm_zorch.logup_gkr.input_layer import InteractionSpec
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.prove import AirInstance, SystemParams, prove
from openvm_zorch.transcript import new_transcript
from openvm_zorch.verify import AirVk, VerificationError, verify
from openvm_zorch.whir.prover import WhirConfig
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_PROVE = Path(__file__).parent / "testdata" / "prove"


def _poseidon2():
    perm = Poseidon2(babybear16_params())
    return (
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _load_instance():
    meta = json.loads((_PROVE / "meta.json").read_text())
    pm = meta["params"]
    airs = []
    vks = []
    for air in meta["airs"]:
        air_idx = air["air_idx"]
        trace = jnp.array(np.load(_PROVE / "inputs" / f"trace_{air_idx}.npy"), dtype=F)
        dag = ConstraintsDag.from_json(
            json.loads((_PROVE / "inputs" / f"constraints_{air_idx}.json").read_text())
        )
        interactions = tuple(
            InteractionSpec(
                bus=s["bus"],
                count_col=s["count_col"],
                count_neg=s["count_neg"],
                message_cols=tuple(s["message_cols"]),
            )
            for s in air["interactions"]
        )
        airs.append(
            AirInstance(
                trace=trace,
                dag=dag,
                interactions=interactions,
                public_values=tuple(air["public_values"]),
                constraint_degree=air["constraint_degree"],
                needs_next=air["needs_next"],
                is_required=air["is_required"],
            )
        )
        log_height = int(trace.shape[0]).bit_length() - 1
        vks.append(
            AirVk(
                dag=dag,
                log_height=log_height,
                width=int(trace.shape[1]),
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
    return meta, params, airs, vks


class VerifyTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.meta, cls.params, cls.airs, cls.vks = _load_instance()
        sponge, comp = _poseidon2()
        cls.sponge, cls.comp = sponge, comp
        _, cls.proof = prove(
            new_transcript(), sponge, comp, cls.params, cls.meta["vk_pre_hash"], cls.airs
        )

    def _verify(self, proof) -> None:
        verify(
            new_transcript(),
            self.sponge,
            self.comp,
            self.params,
            self.meta["vk_pre_hash"],
            self.vks,
            proof.common_main_commit,
            proof,
        )

    def test_accepts_honest_proof(self) -> None:
        self._verify(self.proof)  # must not raise

    def test_rejects_tampered_gkr(self) -> None:
        bad_gkr = dataclasses.replace(
            self.proof.gkr_proof, q0_claim=self.proof.gkr_proof.q0_claim + jnp.ones((), self.proof.gkr_proof.q0_claim.dtype)
        )
        bad = dataclasses.replace(self.proof, gkr_proof=bad_gkr)
        with self.assertRaises(VerificationError):
            self._verify(bad)

    def test_rejects_tampered_stage3_opening(self) -> None:
        bcp = self.proof.batch_constraint_proof
        openings = [list(parts) for parts in bcp.column_openings]
        openings[0][0] = openings[0][0] + jnp.ones((), openings[0][0].dtype)
        bad_bcp = dataclasses.replace(bcp, column_openings=openings)
        bad = dataclasses.replace(self.proof, batch_constraint_proof=bad_bcp)
        with self.assertRaises(VerificationError):
            self._verify(bad)

    def test_rejects_tampered_stacking_opening(self) -> None:
        sp = self.proof.stacking_proof
        openings = [list(v) for v in sp.stacking_openings]
        openings[0][0] = openings[0][0] + jnp.ones((), openings[0][0].dtype)
        bad_sp = dataclasses.replace(sp, stacking_openings=openings)
        bad = dataclasses.replace(self.proof, stacking_proof=bad_sp)
        with self.assertRaises(VerificationError):
            self._verify(bad)

    def test_rejects_tampered_whir_final_poly(self) -> None:
        wp = self.proof.whir_proof
        bad_final = wp.final_poly + jnp.ones((), wp.final_poly.dtype)
        bad_wp = dataclasses.replace(wp, final_poly=bad_final)
        bad = dataclasses.replace(self.proof, whir_proof=bad_wp)
        with self.assertRaises(VerificationError):
            self._verify(bad)

    def test_rejects_tampered_whir_opened_row(self) -> None:
        wp = self.proof.whir_proof
        rows = [list(per_commit) for per_commit in wp.initial_round_opened_rows]
        # Perturb the first opened row of the first query of the first commit.
        rows[0][0] = rows[0][0].at[0, 0].add(jnp.ones((), rows[0][0].dtype))
        bad_wp = dataclasses.replace(wp, initial_round_opened_rows=rows)
        bad = dataclasses.replace(self.proof, whir_proof=bad_wp)
        with self.assertRaises(VerificationError):
            self._verify(bad)


if __name__ == "__main__":
    absltest.main()
