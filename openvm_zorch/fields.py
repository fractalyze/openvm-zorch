"""BabyBear / BabyBear⁴ dtype helpers shared across stages.

The reference works in BabyBear (``F``) with BabyBear⁴ challenges (``EF``);
fixtures hold canonical (non-Montgomery) u32. These helpers pin the two
conversions every stage needs: embedding base values into the extension and
building canonical small constants (inverses included) host-side.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from zk_dtypes import babybear_mont as F
from zk_dtypes import babybearx4_mont as EF
from zk_dtypes import pfinfo

MODULUS = pfinfo(F).modulus


def f_to_ef(x: Array) -> Array:
    """Embed base-field values into BabyBear⁴ (limbs ``[x, 0, 0, 0]``)."""
    zero = fnp.zeros_like(x)
    return fnp.stack([x, zero, zero, zero], axis=-1).view(EF)[..., 0]


def f_const(value: int) -> Array:
    """A canonical base-field constant (reduced mod p)."""
    return fnp.array(value % MODULUS, F)


def f_inv_const(value: int) -> Array:
    """The base-field inverse of a host integer, computed host-side."""
    return f_const(pow(value, MODULUS - 2, MODULUS))
