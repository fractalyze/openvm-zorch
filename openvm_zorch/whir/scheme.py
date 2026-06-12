"""SWIRL's WHIR opening scheme — the consumer half of the generic
``zorch.pcs.whir`` seam.

The generic ``WhirProver``/``WhirVerifier`` run the scheme-agnostic WHIR round
driver (sumcheck folds, per-round RS re-encode + out-of-domain sample, strided
query consistency, final constraint) and delegate the SWIRL-specific maps to a
``WhirScheme``. This is that scheme for openvm-stark-backend byte-match:

- the initial message ``f̂`` is the **prismalinear** eval→coeff RS encoding of the
  committed columns, μ-power combined;
- the weight is the **Möbius-adjusted** equality polynomial of ``u_cube`` (so the
  opened claim is ``q̂(u) = Σ_b f̂(b)·mobius_eq(u, b)``), and the final-constraint
  prefix is that weight's multilinear at the fold challenges;
- ``bind`` is a **no-op**: the commitment was already absorbed in Stage 1, so WHIR
  must not re-absorb it (it starts straight at the μ grind).

Bit-order matters for byte-match and is the reference's, NOT the zorch self-test's:
``mobius_eq_table`` is LSB-first (index bit ``i`` ↔ ``u[i]``) via ``concatenate``,
and the prefix pairs ``u`` with the fold challenges **directly** (no reversal),
because table and fold are both LSB-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import Array

from openvm_zorch.commit.rs_message import eval_to_coeff_rs_message, mle_coeffs_to_evals
from openvm_zorch.fields import EF, f_to_ef
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize


def _mobius_eq_table(u: Sequence[Array]) -> Array:
    """``mobius_eq(u, b)`` over the hypercube, LSB-first (index bit ``i`` ↔
    ``u[i]``); per-coordinate kernel ``K(0) = 1 − 2u_i``, ``K(1) = u_i`` — the
    reference ``evals_mobius_eq_hypercube`` (and openvm ``mobius_eq_table``)."""
    table = jnp.ones((1,), EF)
    one = jnp.ones((), EF)
    for u_i in u:
        table = jnp.concatenate([table * (one - u_i - u_i), table * u_i])
    return table


def _eval_mobius_eq_mle(u: Sequence[Array], x: Sequence[Array]) -> Array:
    """The Möbius-eq multilinear ``Π_i (1−2u_i)(1−x_i) + u_i·x_i`` — the closed
    form of ``_mobius_eq_table(u)`` at a bound point ``x``. ``u`` and ``x`` pair
    directly (reference ``_eval_mobius_eq_mle``)."""
    acc = jnp.ones((), EF)
    one = jnp.ones((), EF)
    for u_i, x_i in zip(u, x):
        acc = acc * ((one - u_i - u_i) * (one - x_i) + u_i * x_i)
    return acc


def _unstack(point: Array) -> list[Array]:
    """`(n,)` array → list of `n` scalars. The Möbius kernels iterate per
    coordinate; indexing (vs iterating the array) avoids the ZKX extension-dtype
    `lax.sign` dispatch the eq helpers warn about."""
    return [point[i] for i in range(point.shape[0])]


@dataclass(frozen=True)
class SwirlWhirScheme:
    """SWIRL's WHIR opening over the generic seam. ``l_skip`` selects the
    prismalinear chunk size (the order-``2^l_skip`` two-adic subgroup the skip
    variables live on). Frozen + hashable so it rides the prover/verifier ``@jit``
    static key."""

    l_skip: int

    def bind(
        self, transcript: Transcript, commitment: Array, values: Array
    ) -> Transcript:
        # The common-main commitment was bound in Stage 1; WHIR opens against it
        # and must not re-absorb the root or the (implicit) claimed values.
        return transcript

    def _messages(self, mle: Array) -> Array:
        """Prismalinear RS message of every committed column, re-read as MLE evals
        `(W, 2^m)` — the reference ``f̂`` per column before the μ combine."""
        return mle_coeffs_to_evals(eval_to_coeff_rs_message(self.l_skip, mle.T))

    def claimed_values(self, mle: Array, z: Array) -> Array:
        # Per-column claim ⟨f̂_col, mobius_eq(u)⟩; μ-combining these (the verifier's
        # eval_coeffs(values, μ)) reproduces the sumcheck's Σ f̂·ŵ claim.
        table = _mobius_eq_table(_unstack(z))  # (2^m,) EF
        return (f_to_ef(self._messages(mle)) * table[None, :]).sum(1)  # (W,) EF

    def combined_f_evals(self, mle: Array, mu: Array) -> Array:
        m = log2_strict_usize(mle.shape[0])
        messages = self._messages(mle)
        f_evals = jnp.zeros((1 << m,), EF)
        mu_pow = jnp.ones((), EF)
        for col in range(messages.shape[0]):
            f_evals = f_evals + mu_pow * f_to_ef(messages[col])
            mu_pow = mu_pow * mu
        return f_evals

    def initial_weight(self, z: Array) -> Array:
        return _mobius_eq_table(_unstack(z))

    def final_prefix(self, z: Array, alphas: Array) -> Array:
        return _eval_mobius_eq_mle(_unstack(z), _unstack(alphas))
