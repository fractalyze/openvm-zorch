"""Prismalinear eval→coeff RS message + codeword — SWIRL's column encoding.

Reference: openvm-stark-backend ``eval_to_coeff_rs_message`` / ``rs_code_matrix``
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/poly.rs#L325
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/stacked_pcs.rs#L341

A stacked column of length ``2^(l_skip + n)`` is read as evaluations of a
prismalinear polynomial on ``D × {0,1}^n`` with ``D`` (the order-``2^l_skip``
two-adic subgroup) varying fastest. The RS message is produced chunk-wise:

1. inverse NTT each contiguous ``2^l_skip`` chunk (one boolean assignment) —
   subgroup evaluations of the Z-variable → Z-monomial coefficients;
2. within each chunk, the MLE coeff→eval transform over the chunk's *bit*
   variables (``a[u + step] += a[u]``, LSB-first) — the Rust
   ``Mle::coeffs_to_evals_inplace``.

The codeword is the message zero-padded by ``2^log_blowup`` and forward-NTT'd
— natural order, no coset shift, no bit-reversal (the Rust side's
``Radix2DitParallel::dft`` on the resized coefficient vector). Both NTTs are
``lax.ntt``, the XLA-native NTT, whose subgroup-generator convention matches
plonky3's (zorch's RS path byte-matches plonky3-derived provers on the same
op).
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array, lax

# The MLE coeff↔eval (zeta / Möbius) transforms are scheme-agnostic and live in
# zorch; re-exported here so this module stays the SWIRL RS-message home and its
# importers are unchanged. Only the prismalinear chunking below is SWIRL-specific.
from zorch.coding.reed_solomon import ReedSolomon
from zorch.poly.multilinear import mle_coeffs_to_evals, mle_evals_to_coeffs
from zorch.utils.bits import log2_strict_usize

__all__ = [
    "mle_coeffs_to_evals",
    "mle_evals_to_coeffs",
    "eval_to_coeff_rs_message",
    "rs_code_matrix",
]


def _coeffs_to_evals_chunk(coeffs: Array) -> Array:
    """Transpose-free MLE coeff→eval (zeta/subset-sum) over the trailing axis of
    ``(..., 2^k)`` — the RS chunk's ``k = l_skip`` bit variables (small, static).

    Same transform as zorch's general ``mle_coeffs_to_evals``, but processes each
    bit in its *natural middle position*: the row-major view ``(.., 2^(k-b-1), 2,
    2^b)`` puts bit ``b`` on the size-2 axis, so the ``(lo, lo+hi)`` pair
    recombines with a plain ``stack(axis=-2)`` and NO ``swapaxes`` rotation.
    Byte-identical — field add commutes, so this LSB-first order equals the
    scan's MSB-first one.

    Why not just call ``mle_coeffs_to_evals``: its general ``lax.scan`` rotates
    bit labels with a per-level ``swapaxes`` (a transpose) to bound *compile* at
    large ``k``. Here ``k`` is small and static and this body is eager, where
    that transpose does not parallelize — it was ~98% of the GPU RS-NTT, and
    removing it is a measured ~13× warm-GPU win on the production block (#46).
    This is NOT a general replacement: a static unroll at large ``k`` lowers to
    ``k`` distinct kernels (the compile explosion the scan exists to avoid)."""
    n = coeffs.shape[-1]
    lead = coeffs.shape[:-1]
    a = coeffs
    for b in range(log2_strict_usize(n)):
        x = a.reshape(lead + (n >> (b + 1), 2, 1 << b))
        lo, hi = x[..., 0, :], x[..., 1, :]
        a = fnp.stack([lo, lo + hi], axis=-2).reshape(lead + (n,))
    return a


def eval_to_coeff_rs_message(l_skip: int, evals: Array) -> Array:
    """Per-column RS message of prismalinear evaluations ``(..., 2^(l_skip+n))``."""
    chunk_len = 1 << l_skip
    # The XLA-native fft accepts at most 2-D, so the chunk batch flattens
    # across all leading axes and reshapes back after the transform.
    chunks = evals.reshape(-1, chunk_len)
    coeffs = lax.ntt(chunks, ntt_type="INTT", ntt_length=chunk_len)
    return _coeffs_to_evals_chunk(coeffs).reshape(evals.shape)


def rs_code_matrix(l_skip: int, log_blowup: int, eval_matrix: Array) -> Array:
    """RS codewords of every column of ``(height, width)``; returns
    ``(height << log_blowup, width)``."""
    # Columns are independent; encode them batched on the leading axis. The
    # zero-pad + forward NTT (natural order, no coset) is zorch
    # ``ReedSolomon.encode`` — the same code object WHIR's rounds re-encode
    # with, so Stage 1 and Stage 5 share one convention by construction.
    messages = eval_to_coeff_rs_message(l_skip, eval_matrix.T)
    code = ReedSolomon(
        message_len=eval_matrix.shape[0],
        blowup=1 << log_blowup,
        dtype=eval_matrix.dtype,
    )
    return code.encode(messages).T
