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

import os
import time

import jax
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


_COMMIT_PROFILE = os.environ.get("OPENVM_COMMIT_PROFILE") == "1"


class _RsProfiler:
    """Env-guarded sub-op timer splitting ``rs_code_matrix`` into IFFT /
    MLE coeff→eval / forward FFT — localizes the #46 commit-GPU pole (RS-NTT,
    ~0.085s warm) within itself. No-op unless ``OPENVM_COMMIT_PROFILE=1``.

    Mirrors ``trace_commit._StackedProfiler`` one level down. ``rs_code_matrix``
    runs eagerly (only ``stacked_merkle_commit`` jits), so blocking on each
    region's output attributes the three ``lax.fft`` / ``lax.scan`` dispatches
    separately without distorting their fusion."""

    def __init__(self) -> None:
        self._t = time.monotonic()

    def mark(self, label: str, *outputs: object) -> None:
        if not _COMMIT_PROFILE:
            return
        jax.block_until_ready(outputs)
        now = time.monotonic()
        print(f"      [rs {label}] {now - self._t:.3f}s", flush=True)
        self._t = now


# EXPERIMENT (#46): transpose-free coeff->eval candidates for the small static
# RS chunk (k = l_skip = 4). zorch's general _butterfly_scan rotates bit labels
# with a per-level swapaxes (a transpose -- the measured RS-NTT CPU pole); when
# the bit is processed in its NATURAL middle position the pair recombines with no
# transpose. Both are byte-identical to mle_coeffs_to_evals (field add commutes).
# Selected by OPENVM_RS_MLE in {scan (default), unroll, at_add} for A/B timing;
# only valid for SMALL static k (NOT a zorch general-primitive replacement).
_RS_MLE = os.environ.get("OPENVM_RS_MLE", "scan")


def _coeffs_to_evals_unroll(coeffs: Array) -> Array:
    from zorch.utils.bits import log2_strict_usize

    n = coeffs.shape[-1]
    lead = coeffs.shape[:-1]
    a = coeffs
    for b in range(log2_strict_usize(n)):
        x = a.reshape(lead + (n >> (b + 1), 2, 1 << b))
        lo, hi = x[..., 0, :], x[..., 1, :]
        a = jnp.stack([lo, lo + hi], axis=-2).reshape(lead + (n,))
    return a


def _coeffs_to_evals_at(coeffs: Array) -> Array:
    from zorch.utils.bits import log2_strict_usize

    n = coeffs.shape[-1]
    lead = coeffs.shape[:-1]
    a = coeffs
    for b in range(log2_strict_usize(n)):
        x = a.reshape(lead + (n >> (b + 1), 2, 1 << b))
        a = x.at[..., 1, :].add(x[..., 0, :]).reshape(lead + (n,))
    return a


_RS_MLE_FN = {
    "scan": mle_coeffs_to_evals,
    "unroll": _coeffs_to_evals_unroll,
    "at_add": _coeffs_to_evals_at,
}[_RS_MLE]


def eval_to_coeff_rs_message(
    l_skip: int, evals: Array, *, _prof: _RsProfiler | None = None
) -> Array:
    """Per-column RS message of prismalinear evaluations ``(..., 2^(l_skip+n))``."""
    chunk_len = 1 << l_skip
    # The zkx-native fft accepts at most 2-D, so the chunk batch flattens
    # across all leading axes and reshapes back after the transform.
    chunks = evals.reshape(-1, chunk_len)
    coeffs = lax.fft(chunks, "IFFT", chunk_len)
    if _prof is not None:
        _prof.mark("ifft", coeffs)
    message = _RS_MLE_FN(coeffs).reshape(evals.shape)
    if _prof is not None:
        _prof.mark("mle_coeffs_to_evals", message)
    return message


def rs_code_matrix(l_skip: int, log_blowup: int, eval_matrix: Array) -> Array:
    """RS codewords of every column of ``(height, width)``; returns
    ``(height << log_blowup, width)``."""
    prof = _RsProfiler() if _COMMIT_PROFILE else None
    height = eval_matrix.shape[0]
    rs_height = height << log_blowup
    # Columns are independent; encode them batched on the leading axis.
    messages = eval_to_coeff_rs_message(l_skip, eval_matrix.T, _prof=prof)
    pad = jnp.zeros(messages.shape[:-1] + (rs_height - height,), eval_matrix.dtype)
    coeffs = jnp.concatenate([messages, pad], axis=-1)
    codeword = lax.fft(coeffs, "FFT", rs_height).T
    if prof is not None:
        prof.mark("forward_fft", codeword)
    return codeword
