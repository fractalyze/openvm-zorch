"""Transcript byte-match: zorch DuplexTranscript == reference DuplexSponge.

Replays the recorded observe/sample script from tools/fixture-gen
(``--transcript-out``): every observed value is absorbed, every sample is
squeezed and compared against the recorded value. The script crosses the
rate-8 boundary mid-observe, drains multiple samples from one squeeze block,
exercises extension-field granularity (4 limbs each way), a digest observe,
and the proof-of-work check — the full surface Stage 2 needs. Canonical-u32
equality, no tolerances.
"""

import json
from pathlib import Path

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from openvm_zorch.transcript import check_witness, new_transcript

_FIXTURE = Path(__file__).parent / "testdata" / "transcript"


class TranscriptByteMatchTest(absltest.TestCase):
    def test_replay_matches(self) -> None:
        values = np.load(_FIXTURE / "transcript_values.npy")
        is_sample = np.load(_FIXTURE / "transcript_is_sample.npy")
        perms = np.load(_FIXTURE / "transcript_perm_results.npy")

        t = new_transcript()
        for value, sampled in zip(values.tolist(), is_sample.tolist()):
            if sampled:
                t, got = t.sample(1)
                self.assertEqual(int(got[0].astype(fnp.uint32)), value)
            else:
                t = t.observe(fnp.array(value, dtype=F))

        # The sponge state after the final permutation must match the
        # reference recorder's last snapshot (the recorder also logs the
        # initial all-zero state, so >= 2 entries are guaranteed here).
        got_state = np.asarray(t.state.sponge_state.astype(fnp.uint32))
        np.testing.assert_array_equal(got_state, perms[-1])

    def test_check_witness_matches_reference_grind(self) -> None:
        meta = json.loads((_FIXTURE / "meta.json").read_text())
        values = np.load(_FIXTURE / "transcript_values.npy")
        is_sample = np.load(_FIXTURE / "transcript_is_sample.npy")
        pow_bits = meta["pow_bits"]
        witness = meta["pow_witness"]

        # Rebuild the transcript up to just before the PoW (witness observe +
        # one sample at the tail, then one final sample).
        t = new_transcript()
        cut = len(values) - 3
        for value, sampled in zip(values[:cut].tolist(), is_sample[:cut].tolist()):
            if sampled:
                t, _ = t.sample(1)
            else:
                t = t.observe(fnp.array(value, dtype=F))

        self.assertEqual(int(values[cut]), witness)
        t, ok = check_witness(t, pow_bits, fnp.array(witness, dtype=F))
        self.assertTrue(bool(ok))
        t, final = t.sample(1)
        self.assertEqual(int(final[0].astype(fnp.uint32)), int(values[-1]))


if __name__ == "__main__":
    absltest.main()
