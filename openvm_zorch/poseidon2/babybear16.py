"""BabyBear width-16 Poseidon2 parameters — OpenVM's prover instance.

zorch ships the agnostic Poseidon2 engine but no named parameterizations: a
named instance is a consumer concern. This is the permutation every hash byte
in openvm-stark-backend's proofs computes — plonky3's
``default_babybear_poseidon2_16`` at the ``=0.4.3`` pin the reference carries,
https://github.com/Plonky3/Plonky3/blob/v0.4.3/baby-bear/src/poseidon2.rs

- round constants: the HorizenLabs BabyBear instance
  (``BABYBEAR_RC16_EXTERNAL_INITIAL/FINAL``, ``BABYBEAR_RC16_INTERNAL``),
  written in the same hex as the plonky3 source so the arrays diff cleanly;
- internal layer: plonky3's ``1 + Diag(V)`` family — the matrix is
  ``J + Diag(V)`` with the optimized vector
  ``[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8,
  -1/16, -1/2^27]`` reduced to canonical form below. Unlike SP1's vendored
  koalabear kernel there is no Montgomery factor folded into the layer, so
  ``internal_j_scale`` stays at the default 1;
- S-box ``x^7`` (the least D with gcd(p-1, D) = 1 for p = 15·2^27 + 1).
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from zk_dtypes import babybear_mont as F

from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.params import Poseidon2Params
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_WIDTH, _ER, _IR, _ALPHA = 16, 4, 13, 7

_EXTERNAL_INITIAL = [
    [
        0x69CBB6AF, 0x46AD93F9, 0x60A00F4E, 0x6B1297CD, 0x23189AFE, 0x732E7BEF,
        0x72C246DE, 0x2C941900, 0x0557EEDE, 0x1580496F, 0x3A3EA77B, 0x54F3F271,
        0x0F49B029, 0x47872FE1, 0x221E2E36, 0x1AB7202E,
    ],
    [
        0x487779A6, 0x3851C9D8, 0x38DC17C0, 0x209F8849, 0x268DCEE8, 0x350C48DA,
        0x5B9AD32E, 0x0523272B, 0x3F89055B, 0x01E894B2, 0x13DDEDDE, 0x1B2EF334,
        0x7507D8B4, 0x6CEEB94E, 0x52EB6BA2, 0x50642905,
    ],
    [
        0x05453F3F, 0x06349EFC, 0x6922787C, 0x04BFFF9C, 0x768C714A, 0x3E9FF21A,
        0x15737C9C, 0x2229C807, 0x0D47F88C, 0x097E0ECC, 0x27EADBA0, 0x2D7D29E4,
        0x3502AAA0, 0x0F475FD7, 0x29FBDA49, 0x018AFFFD,
    ],
    [
        0x0315B618, 0x6D4497D1, 0x1B171D9E, 0x52861ABD, 0x2E5D0501, 0x3EC8646C,
        0x6E5F250A, 0x148AE8E6, 0x17F5FA4A, 0x3E66D284, 0x0051AA3B, 0x483F7913,
        0x2CFE5F15, 0x023427CA, 0x2CC78315, 0x1E36EA47,
    ],
]

_EXTERNAL_TERMINAL = [
    [
        0x7290A80D, 0x6F7E5329, 0x598EC8A8, 0x76A859A0, 0x6559E868, 0x657B83AF,
        0x13271D3F, 0x1F876063, 0x0AEEAE37, 0x706E9CA6, 0x46400CEE, 0x72A05C26,
        0x2C589C9E, 0x20BD37A7, 0x6A2D3D10, 0x20523767,
    ],
    [
        0x5B8FE9C4, 0x2AA501D6, 0x1E01AC3E, 0x1448BC54, 0x5CE5AD1C, 0x4918A14D,
        0x2C46A83F, 0x4FCF6876, 0x61D8D5C8, 0x6DDF4FF9, 0x11FDA4D3, 0x02933A8F,
        0x170EAF81, 0x5A9C314F, 0x49A12590, 0x35EC52A1,
    ],
    [
        0x58EB1611, 0x5E481E65, 0x367125C9, 0x0EBA33BA, 0x1FC28DED, 0x066399AD,
        0x0CBEC0EA, 0x75FD1AF0, 0x50F5BF4E, 0x643D5F41, 0x6F4FE718, 0x5B3CBBDE,
        0x1E3AFB3E, 0x296FB027, 0x45E1547B, 0x4A8DB2AB,
    ],
    [
        0x59986D19, 0x30BCDFA3, 0x1DB63932, 0x1D7C2824, 0x53B33681, 0x0673B747,
        0x038A98A3, 0x2C5BCE60, 0x351979CD, 0x5008FB73, 0x547BCA78, 0x711AF481,
        0x3F93BF64, 0x644D987B, 0x3C8BCD87, 0x608758B8,
    ],
]

_INTERNAL_RC = [
    0x5A8053C0, 0x693BE639, 0x3858867D, 0x19334F6B, 0x128F0FD8, 0x4E2B1CCB,
    0x61210CE0, 0x3C318939, 0x0B5B2F22, 0x2EDB11D5, 0x213EFFDF, 0x0CAC4606,
    0x241AF16D,
]

# Canonical reductions mod p = 2013265921 of the optimized diagonal V
# (so e.g. -2 -> p-2, 1/2^27 -> p-15 since 2^27 * 15 = p - 1).
_INTERNAL_DIAG = [
    2013265919,  # -2
    1,
    2,
    1006632961,  # 1/2
    3,
    4,
    1006632960,  # -1/2
    2013265918,  # -3
    2013265917,  # -4
    2005401601,  # 1/2^8
    1509949441,  # 1/4
    1761607681,  # 1/8
    2013265906,  # 1/2^27
    7864320,  # -1/2^8
    125829120,  # -1/16
    15,  # -1/2^27
]


def babybear16_params() -> Poseidon2Params:
    """OpenVM's BabyBear width-16 Poseidon2 parameters."""
    internal_rc = np.zeros((_IR, _WIDTH), dtype=np.int64)
    internal_rc[:, 0] = np.array(_INTERNAL_RC, dtype=np.int64)
    return Poseidon2Params(
        width=_WIDTH,
        dtype=F,
        alpha=_ALPHA,
        external_rounds=_ER,
        internal_rounds=_IR,
        external_constants_initial=fnp.array(_EXTERNAL_INITIAL, dtype=F),
        external_constants_terminal=fnp.array(_EXTERNAL_TERMINAL, dtype=F),
        internal_constants=fnp.array(internal_rc, dtype=F),
        internal_diag=fnp.array(_INTERNAL_DIAG, dtype=F),
    )


# The Merkle hasher's shape around that permutation: an 8-element digest,
# absorbed 8 at a time, compressed 2-to-1 — the reference's
# ``Poseidon2Sponge`` / ``Poseidon2Compression`` over the width-16 instance.
_RATE = 8
_DIGEST = 8
_ARITY = 2


def babybear16_hasher() -> tuple[Sponge, Compression]:
    """OpenVM's Merkle hasher: the sponge and the 2-to-1 compression.

    Both wrap ONE permutation — they are the same Poseidon2 instance used two
    ways, and building it twice would only pay for the round constants twice.
    """
    perm = Poseidon2(babybear16_params())
    return (
        Sponge(perm, SpongeParams(rate=_RATE, out=_DIGEST)),
        Compression(perm, CompressionParams(arity=_ARITY, chunk=_DIGEST)),
    )
