"""End-to-end verifier: accept honest proofs, reject tampered ones.

Proves the production-params instance with ``prove`` (no recorded log), then
checks ``verify`` accepts the result. The negative cases perturb one field of
the proof at a time and assert the verifier rejects — exercising each stage's
algebraic check and the Merkle-path verification.
"""

import dataclasses
import json
from pathlib import Path

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import babybear_mont as F

from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_hasher
from openvm_zorch.prove import AirInstance, SystemParams, prove
from openvm_zorch.transcript import new_transcript
from openvm_zorch.verify import AirVk, VerificationError, verify
from openvm_zorch.whir.prover import WhirConfig

_PROVE = Path(__file__).parent / "testdata" / "prove"


def _load_instance():
    meta = json.loads((_PROVE / "meta.json").read_text())
    pm = meta["params"]
    airs = []
    vks = []
    for air in meta["airs"]:
        air_idx = air["air_idx"]
        trace = fnp.array(np.load(_PROVE / "inputs" / f"trace_{air_idx}.npy"), dtype=F)
        dag = ConstraintsDag.from_json(
            json.loads((_PROVE / "inputs" / f"constraints_{air_idx}.json").read_text())
        )
        cached_mains = tuple(
            fnp.array(np.load(_PROVE / "inputs" / f"cached_{air_idx}_{k}.npy"), dtype=F)
            for k in range(air.get("num_cached_mains", 0))
        )
        airs.append(
            AirInstance(
                trace=trace,
                dag=dag,
                public_values=tuple(air["public_values"]),
                constraint_degree=air["constraint_degree"],
                needs_next=air["needs_next"],
                is_required=air["is_required"],
                cached_mains=cached_mains,
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


def _bump(x):
    return x + fnp.ones((), x.dtype)


def _tamper_gkr(p):
    gkr = dataclasses.replace(p.gkr_proof, q0_claim=_bump(p.gkr_proof.q0_claim))
    return dataclasses.replace(p, gkr_proof=gkr)


def _tamper_stage3_opening(p):
    bcp = p.batch_constraint_proof
    openings = [list(parts) for parts in bcp.column_openings]
    openings[0][0] = _bump(openings[0][0])
    return dataclasses.replace(
        p, batch_constraint_proof=dataclasses.replace(bcp, column_openings=openings)
    )


def _tamper_stacking_opening(p):
    sp = p.stacking_proof
    openings = [list(v) for v in sp.stacking_openings]
    openings[0][0] = _bump(openings[0][0])
    return dataclasses.replace(
        p, stacking_proof=dataclasses.replace(sp, stacking_openings=openings)
    )


def _tamper_whir_final_poly(p):
    wp = p.whir_proof
    return dataclasses.replace(
        p, whir_proof=dataclasses.replace(wp, final_poly=_bump(wp.final_poly))
    )


def _tamper_whir_opened_row(p):
    wp = p.whir_proof
    rows = [list(per_commit) for per_commit in wp.initial_round_opened_rows]
    # Perturb the first opened row of the first query of the first commit.
    rows[0][0] = rows[0][0].at[0, 0].add(fnp.ones((), rows[0][0].dtype))
    return dataclasses.replace(
        p, whir_proof=dataclasses.replace(wp, initial_round_opened_rows=rows)
    )


# Each mutator perturbs exactly one proof field; the value is the GKR q0,
# a Stage-3 column opening, a stacking opening, the WHIR final poly, and a
# WHIR opened row respectively — one per stage's verifier check.
_TAMPERS = {
    "gkr_q0": _tamper_gkr,
    "stage3_opening": _tamper_stage3_opening,
    "stacking_opening": _tamper_stacking_opening,
    "whir_final_poly": _tamper_whir_final_poly,
    "whir_opened_row": _tamper_whir_opened_row,
}


class VerifyTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.meta, cls.params, cls.airs, cls.vks = _load_instance()
        sponge, comp = babybear16_hasher()
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

    @parameterized.named_parameters(*_TAMPERS.items())
    def test_rejects_tampered(self, mutate) -> None:
        """Each mutator perturbs one field of an otherwise-honest proof; the
        verifier must reject it at the corresponding stage's check."""
        with self.assertRaises(VerificationError):
            self._verify(mutate(self.proof))


if __name__ == "__main__":
    absltest.main()
