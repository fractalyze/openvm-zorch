"""Stage 5 byte-match: WHIR opening proof vs the reference.

Rebuilds the transcript state at ``stage4_end`` by replaying the recorded log
(observes fed back in, samples squeezed and asserted equal), recommits the
sorted traces with the Stage-1 code (asserting the root against the log's
prelude), then drives ``prove_whir_opening`` — grinds included, run natively —
and compares every ``WhirProof`` field against the fixture: the PoW
witnesses, every sumcheck round polynomial, the per-round codeword commits
and out-of-domain values, every opened row and Merkle path, and the final
polynomial. Stage 5 is the last stage, so the walk consumes the 945-entry
log exactly. Canonical-u32 equality, no tolerances.
"""

import json
from pathlib import Path

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import babybear_mont as F

from openvm_zorch.commit.trace_commit import stacked_commit
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.transcript import ef_from_limbs, new_transcript
from openvm_zorch.whir.prover import WhirConfig, prove_whir_opening
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_FIXTURE = Path(__file__).parent / "testdata" / "whir"


def _ef_limbs(x) -> np.ndarray:
    """Canonical-u32 limbs of a BabyBear⁴ array, shape (..., 4)."""
    return np.asarray(lax.bitcast_convert_type(fnp.atleast_1d(x), F).astype(fnp.uint32))


def _to_u32(x) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(x, F).astype(fnp.uint32))


def _replay_log(values: np.ndarray, is_sample: np.ndarray, end: int):
    """Reconstruct the transcript state at log index ``end``: feed observes
    back in, squeeze at samples and assert the squeezed values match."""
    t = new_transcript()
    idx = 0
    while idx < end:
        if is_sample[idx]:
            t, got = t.sample(1)
            got = int(np.asarray(lax.bitcast_convert_type(got, F).astype(fnp.uint32))[0])
            assert got == int(values[idx]), f"sample mismatch at {idx}"
            idx += 1
        else:
            run = idx
            while run < end and not is_sample[run]:
                run += 1
            t = t.observe(fnp.array(values[idx:run], dtype=F))
            idx = run
    return t


class WhirByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.meta = json.loads((_FIXTURE / "meta.json").read_text())
        cls.values = np.load(_FIXTURE / "outputs" / "transcript_values.npy")
        cls.is_sample = np.load(_FIXTURE / "outputs" / "transcript_is_sample.npy")

    def _check_stage5(self, jit: bool) -> None:
        meta = self.meta
        params = meta["params"]
        l_skip = params["l_skip"]

        # Recommit the sorted traces with the Stage-1 code; the root must
        # reproduce the prelude's common-main commitment (log values [8..16]).
        perm = Poseidon2(babybear16_params())
        sponge = Sponge(perm, SpongeParams(rate=8, out=8))
        comp = Compression(perm, CompressionParams(arity=2, chunk=8))
        traces = [
            fnp.array(np.load(_FIXTURE / "inputs" / f"trace_{air_idx}.npy"), dtype=F)
            for air_idx in meta["sorted_airs"]
        ]
        root, data = stacked_commit(
            sponge,
            comp,
            l_skip,
            params["n_stack"],
            params["log_blowup"],
            params["k_whir"],
            traces,
        )
        np.testing.assert_array_equal(_to_u32(root), self.values[8:16])

        u_cube = [
            ef_from_limbs(fnp.array(row, fnp.uint32))
            for row in np.load(_FIXTURE / "inputs" / "u_cube.npy")
        ]
        config = WhirConfig(
            k=params["k_whir"],
            num_queries=meta["num_queries"],
            mu_pow_bits=params["mu_pow_bits"],
            folding_pow_bits=params["folding_pow_bits"],
            query_phase_pow_bits=params["query_phase_pow_bits"],
        )

        t = _replay_log(self.values, self.is_sample, meta["stage4_end"])
        t, proof = prove_whir_opening(
            t,
            sponge,
            comp,
            l_skip,
            params["log_blowup"],
            config,
            [(data.matrix, data.tree)],
            u_cube,
            jit=jit,
        )

        out = _FIXTURE / "outputs"
        np.testing.assert_array_equal(
            _to_u32(fnp.atleast_1d(proof.mu_pow_witness)),
            np.load(out / "mu_pow_witness.npy"),
        )
        np.testing.assert_array_equal(_ef_limbs(proof.mu)[0], np.load(out / "mu.npy"))
        want_sumcheck = np.load(out / "sumcheck_polys.npy")
        self.assertEqual(len(proof.whir_sumcheck_polys), want_sumcheck.shape[0])
        for j, evals in enumerate(proof.whir_sumcheck_polys):
            np.testing.assert_array_equal(
                _ef_limbs(evals), want_sumcheck[j], err_msg=f"sumcheck round {j}"
            )
        np.testing.assert_array_equal(
            _to_u32(fnp.stack(proof.codeword_commits)),
            np.load(out / "codeword_commits.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(fnp.stack(proof.ood_values)), np.load(out / "ood_values.npy")
        )
        np.testing.assert_array_equal(
            _to_u32(fnp.stack(proof.folding_pow_witnesses)),
            np.load(out / "folding_pow_witnesses.npy"),
        )
        np.testing.assert_array_equal(
            _to_u32(fnp.stack(proof.query_phase_pow_witnesses)),
            np.load(out / "query_phase_pow_witnesses.npy"),
        )
        np.testing.assert_array_equal(
            _ef_limbs(proof.final_poly), np.load(out / "final_poly.npy")
        )

        # Initial-round openings (common main): rows of the Stage-1 codeword
        # and the sibling paths from the query layer up.
        want_rows = np.load(out / "initial_opened_rows_c0.npy")
        self.assertEqual(len(proof.initial_round_opened_rows), 1)
        rows = proof.initial_round_opened_rows[0]
        self.assertEqual(len(rows), want_rows.shape[0])
        for q, row in enumerate(rows):
            np.testing.assert_array_equal(
                _to_u32(row), want_rows[q], err_msg=f"initial rows, query {q}"
            )
        want_proofs = np.load(out / "initial_merkle_proofs_c0.npy")
        for q, path in enumerate(proof.initial_round_merkle_proofs[0]):
            np.testing.assert_array_equal(
                _to_u32(path), want_proofs[q], err_msg=f"initial path, query {q}"
            )

        # Later rounds: opened codeword values (extension field) and paths.
        self.assertEqual(len(proof.codeword_opened_values), meta["num_whir_rounds"] - 1)
        for r, vals in enumerate(proof.codeword_opened_values):
            want_vals = np.load(out / f"codeword_opened_values_r{r + 1}.npy")
            for q, v in enumerate(vals):
                np.testing.assert_array_equal(
                    _ef_limbs(v), want_vals[q], err_msg=f"round {r + 1}, query {q}"
                )
            want_paths = np.load(out / f"codeword_merkle_proofs_r{r + 1}.npy")
            for q, path in enumerate(proof.codeword_merkle_proofs[r]):
                np.testing.assert_array_equal(
                    _to_u32(path),
                    want_paths[q],
                    err_msg=f"round {r + 1} path, query {q}",
                )

    def test_stage5_matches(self) -> None:
        """Eager path byte-matches the reference."""
        self._check_stage5(jit=False)

    def test_stage5_matches_jit(self) -> None:
        """The jit path lowers each device island to one fused kernel and stays
        byte-identical to the reference — the boundary is transparent."""
        self._check_stage5(jit=True)


if __name__ == "__main__":
    absltest.main()
