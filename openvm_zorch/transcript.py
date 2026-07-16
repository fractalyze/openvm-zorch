"""OpenVM's Fiat-Shamir transcript: zorch ``DuplexTranscript`` + BabyBear glue.

The reference is openvm-stark-backend's ``DuplexSponge<BabyBear, Poseidon2,
16, 8>`` (overwrite-mode duplex sponge). zorch's ``DuplexTranscript`` is the
same construction — absorb overwrites the rate prefix preserving the suffix,
squeeze drains the post-permutation rate buffer from the top down — so this
module only pins the OpenVM parameterization (Poseidon2 BabyBear-16, rate 8)
and the BabyBear⁴ challenge conventions:

- ``observe`` of an extension element flattens to its 4 base limbs in basis
  order (zorch's bitcast does this already);
- an extension challenge is 4 consecutive base squeezes reinterpreted as the
  extension element's coefficients (``sample_ext``);
- the LogUp proof-of-work check observes the witness and squeezes one base
  element whose low ``pow_bits`` bits must be zero — except at ``pow_bits ==
  0`` where the reference skips the transcript interaction entirely
  (``FiatShamirTranscript::check_witness`` early-returns), unlike zorch's
  ``check_witness`` which would still absorb the witness.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array, lax
from zk_dtypes import babybear_mont as F
from zk_dtypes import babybearx4_mont as EF

from openvm_zorch.poseidon2.babybear16 import babybear16_params
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.transcript import DuplexTranscript, sample_challenge

RATE = 8
# BabyBear⁴ basis coefficients per extension element.
EF_LIMBS = 4


def new_transcript() -> DuplexTranscript:
    """The OpenVM prover transcript: Poseidon2 BabyBear-16, rate 8."""
    return DuplexTranscript.new(Poseidon2(babybear16_params()), RATE)


def sample_ext(transcript: DuplexTranscript) -> tuple[DuplexTranscript, Array]:
    """Squeeze one BabyBear⁴ challenge as 4 consecutive base squeezes."""
    return sample_challenge(transcript, EF, EF_LIMBS)


def check_witness(
    transcript: DuplexTranscript, pow_bits: int, witness: Array
) -> tuple[DuplexTranscript, Array]:
    """Reference-semantics proof-of-work check.

    Mirrors ``FiatShamirTranscript::check_witness``: at ``pow_bits == 0`` the
    reference returns true WITHOUT touching the transcript, so delegating to
    zorch's ``check_witness`` (which always absorbs) would desynchronize the
    Fiat-Shamir stream.
    """
    if pow_bits == 0:
        return transcript, fnp.bool_(True)
    return transcript.check_witness(pow_bits, witness)


def grind(
    transcript: DuplexTranscript, pow_bits: int
) -> tuple[DuplexTranscript, Array]:
    """Reference-semantics proof-of-work grind.

    Mirrors ``FiatShamirTranscript::grind``: at ``pow_bits == 0`` the
    reference returns witness ZERO without touching the transcript. For
    ``pow_bits > 0`` zorch's lowest-witness search matches the reference's
    serial scan from 0 (the fixture pin: ``default-features = false`` keeps
    the Rust grind serial, so both sides find the same witness).
    """
    if pow_bits == 0:
        return transcript, fnp.zeros((), F)
    return transcript.grind(pow_bits)


def sample_bits(transcript: DuplexTranscript, bits: int) -> tuple[DuplexTranscript, int]:
    """One base squeeze masked to its low ``bits`` canonical bits
    (``FiatShamirTranscript::sample_bits``) — a host int, since it indexes
    Merkle queries."""
    transcript, got = transcript.sample(1)
    canonical = int(
        fnp.asarray(lax.bitcast_convert_type(got, F).astype(fnp.uint32))[0]
    )
    return transcript, canonical & ((1 << bits) - 1)


def ef_from_limbs(limbs: Array) -> Array:
    """Reinterpret a (..., 4) base-limb array as (...,) BabyBear⁴ elements.

    Limb order is the extension basis-coefficient order — the same order
    ``observe`` flattens to and the reference's
    ``as_basis_coefficients_slice`` dumps.
    """
    if limbs.shape[-1] != EF_LIMBS:
        raise ValueError(f"expected trailing dim {EF_LIMBS}, got {limbs.shape}")
    return limbs.astype(F).view(EF)[..., 0]
