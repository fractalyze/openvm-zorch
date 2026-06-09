"""Stage 1 byte-match: stacked_commit reproduces the reference prover.

Fixtures come from tools/fixture-gen (openvm-stark-backend v2.0.0-beta.2 CPU
prover on deterministic traces); every pipeline step is compared separately —
stacked matrix, RS codeword matrix, each stored digest layer, root — so a
failure localizes to one transform instead of one opaque root mismatch.
Canonical-u32 equality, no tolerances.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from openvm_zorch.commit.rs_message import rs_code_matrix
from openvm_zorch.commit.stacking import stacked_matrix
from openvm_zorch.commit.trace_commit import stacked_commit
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_FIXTURE = Path(__file__).parent / "testdata" / "stacked_commit"


def _load(rel: str) -> jnp.ndarray:
    return jnp.array(np.load(_FIXTURE / rel), dtype=F)


class TraceCommitByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.meta = json.loads((_FIXTURE / "meta.json").read_text())
        cls.traces = [
            _load(f"inputs/trace_{i}.npy")
            for i in range(len(cls.meta["trace_dims"]))
        ]

    def test_stacked_matrix_matches(self) -> None:
        mat, _ = stacked_matrix(self.meta["l_skip"], self.meta["n_stack"], self.traces)
        self.assertTrue(
            bool(jnp.array_equal(mat, _load("outputs/stacked_matrix.npy")))
        )

    def test_codeword_matches(self) -> None:
        mat = _load("outputs/stacked_matrix.npy")
        codeword = rs_code_matrix(self.meta["l_skip"], self.meta["log_blowup"], mat)
        self.assertTrue(
            bool(jnp.array_equal(codeword, _load("outputs/codeword.npy")))
        )

    def test_commit_matches(self) -> None:
        perm = Poseidon2(babybear16_params())
        sponge = Sponge(perm, SpongeParams(rate=8, out=8))
        comp = Compression(perm, CompressionParams(arity=2, chunk=8))
        root, data = stacked_commit(
            sponge,
            comp,
            self.meta["l_skip"],
            self.meta["n_stack"],
            self.meta["log_blowup"],
            self.meta["k_whir"],
            self.traces,
        )
        for level, layer in enumerate(data.tree.digest_layers):
            self.assertTrue(
                bool(
                    jnp.array_equal(
                        layer, _load(f"outputs/digest_layer_{level}.npy")
                    )
                ),
                msg=f"digest layer {level} diverges",
            )
        self.assertTrue(bool(jnp.array_equal(root, _load("outputs/root.npy"))))


if __name__ == "__main__":
    absltest.main()
