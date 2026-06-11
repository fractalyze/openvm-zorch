"""Prismalinear eval→coeff RS message + codeword — SWIRL's column encoding.

Reference: openvm-stark-backend ``eval_to_coeff_rs_message`` / ``rs_code_matrix``
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/poly.rs#L325
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/stacked_pcs.rs#L341

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
``lax.fft``, the zkx-native NTT, whose subgroup-generator convention matches
plonky3's (zorch's RS path byte-matches plonky3-derived provers on the same
op).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax

# The MLE coeff↔eval (zeta / Möbius) transforms are scheme-agnostic and live in
# zorch; re-exported here so this module stays the SWIRL RS-message home and its
# importers are unchanged. Only the prismalinear chunking below is SWIRL-specific.
from zorch.poly.multilinear import mle_coeffs_to_evals, mle_evals_to_coeffs

__all__ = [
    "mle_coeffs_to_evals",
    "mle_evals_to_coeffs",
    "eval_to_coeff_rs_message",
    "rs_code_matrix",
]


def eval_to_coeff_rs_message(l_skip: int, evals: Array) -> Array:
    """Per-column RS message of prismalinear evaluations ``(..., 2^(l_skip+n))``."""
    chunk_len = 1 << l_skip
    # The zkx-native fft accepts at most 2-D, so the chunk batch flattens
    # across all leading axes and reshapes back after the transform.
    chunks = evals.reshape(-1, chunk_len)
    coeffs = lax.fft(chunks, "IFFT", chunk_len)
    return mle_coeffs_to_evals(coeffs).reshape(evals.shape)


def rs_code_matrix(l_skip: int, log_blowup: int, eval_matrix: Array) -> Array:
    """RS codewords of every column of ``(height, width)``; returns
    ``(height << log_blowup, width)``."""
    height = eval_matrix.shape[0]
    rs_height = height << log_blowup
    # Columns are independent; encode them batched on the leading axis.
    messages = eval_to_coeff_rs_message(l_skip, eval_matrix.T)
    pad = jnp.zeros(messages.shape[:-1] + (rs_height - height,), eval_matrix.dtype)
    coeffs = jnp.concatenate([messages, pad], axis=-1)
    return lax.fft(coeffs, "FFT", rs_height).T
