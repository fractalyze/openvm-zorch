"""babybear16 params drive zorch's engine to the reference permutation bytes.

Golden vectors are dumped by tools/fixture-gen from plonky3 =0.4.3's
``default_babybear_poseidon2_16`` — the exact instance openvm-stark-backend
v2.0.0 hashes with. Pinning the bare permutation (plus the sponge and
2-to-1 compress shapes the Merkle tree uses) isolates a parameter mistake from
a tree-structure mistake before any commit test runs.
"""

from pathlib import Path

import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from openvm_zorch.poseidon2.babybear16 import babybear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_FIXTURE = (
    Path(__file__).parent.parent / "commit" / "testdata" / "stacked_commit" / "outputs"
)


def _golden(name: str) -> jnp.ndarray:
    return jnp.array(np.load(_FIXTURE / name), dtype=F)


class BabyBear16Test(absltest.TestCase):
    def test_permutation_matches_reference(self) -> None:
        perm = Poseidon2(babybear16_params())
        out = perm.permute(jnp.arange(16, dtype=F))
        self.assertTrue(bool(jnp.array_equal(out, _golden("perm_0_15.npy"))))

    def test_sponge_matches_reference(self) -> None:
        sponge = Sponge(Poseidon2(babybear16_params()), SpongeParams(rate=8, out=8))
        out = sponge.hash(jnp.arange(32, dtype=F))
        self.assertTrue(bool(jnp.array_equal(out, _golden("sponge_0_31.npy"))))

    def test_compress_matches_reference(self) -> None:
        comp = Compression(
            Poseidon2(babybear16_params()), CompressionParams(arity=2, chunk=8)
        )
        left = jnp.arange(8, dtype=F)
        right = jnp.arange(100, 108, dtype=F)
        out = comp.compress(jnp.stack([left, right]))
        self.assertTrue(bool(jnp.array_equal(out, _golden("compress_pair.npy"))))


if __name__ == "__main__":
    absltest.main()
