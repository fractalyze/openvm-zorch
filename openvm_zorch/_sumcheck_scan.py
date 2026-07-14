# Copyright 2026 Fractalyze Inc. All rights reserved.

"""Vendored homogeneous sumcheck scan (fold_pair / lift_to_domain / prove).

These primitives lived in ``zorch.sumcheck.prover`` through zorch ``f176c86``
(#415) and were removed by ``#422`` ("converge dense rounds on StandardRound +
ProductSummand") on the way to the pin this repo now carries (``3c0a0208``).
openvm-zorch's Stage-3/4/5 provers drive them directly, so the exact
pre-#422 bodies are vendored here verbatim — byte-identical math, no behavior
change — rather than re-expressed against zorch's new Round API. The scan is a
plain ``lax.scan`` (no ``lax.composite`` marker), and its only zorch deps
(``reinterpret_challenge``, ``observe_and_sample``, ``log2_strict_usize``) are
unchanged at ``3c0a0208``, so it composes with the current transcript.

Porting these call sites onto zorch's StandardRound/ProveChain API and dropping
this module is tracked as follow-up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.transcript import DuplexTranscript, Transcript, reinterpret_challenge
from zorch.utils.bits import log2_strict_usize


def fold_pair(p0: Array, p1: Array, r: Array) -> Array:
    """Fold one split pair at challenge `r`: P0 + r*(P1 - P0)."""
    return p0 + r * (p1 - p0)


def lift_to_domain(p0: Array, p1: Array, degree: int, start: int = 0) -> Array:
    """Lift one split pair to the evaluation domain [start..degree]:
    f[u] = P0 + u*(P1 - P0), shape (degree+1-start, *P0.shape).

    `start` defaults to 0 (the natural domain [0..degree]); set it to 1 to omit
    the f[0] = P0 point, the compressed round-poly wire form a verifier
    reconstructs from the running claim (s(0) = claim - s(1)). The whole u-domain
    is built at once so the round poly stays one batched reduction (not
    degree+1 separate ones). `us` uses jnp.stack (not jnp.arange, whose iota is
    unsupported for extension dtypes) and is reshaped to broadcast over any
    leading batch dims of the factor."""
    us = jnp.stack([jnp.array(u, p0.dtype) for u in range(start, degree + 1)])
    return p0 + us.reshape((-1,) + (1,) * p0.ndim) * (p1 - p0)


def zero_extend(arr: Array, width: int) -> Array:
    """Zero-extend the last axis to `width` -- the re-extension a fixed-shape
    round carry pairs with a fold: the live prefix halves each round while the
    buffer width stays put, and the dead tail stays exactly zero so full-width
    reductions match live-prefix-truncated ones byte-for-byte (field zero-adds
    are exact)."""
    pad = width - arr.shape[-1]
    if pad < 0:
        raise ValueError(f"width {width} < last-axis size {arr.shape[-1]}")
    if pad == 0:
        return arr
    return jnp.concatenate([arr, jnp.zeros((*arr.shape[:-1], pad), arr.dtype)], axis=-1)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["round_poly", "challenge"],
    meta_fields=[],
)
@dataclass(frozen=True)
class RoundMsg:
    """One per-variable sumcheck round's message: the round polynomial sent plus
    the Fiat-Shamir challenge it induced. `prove` stacks these over the scan, so
    the returned `msgs.round_poly` is the proof and `msgs.challenge` is the
    evaluation point. The challenge is re-derivable from `round_poly`, so it never
    goes on the wire -- it rides here only to spare the prover a transcript replay.
    A registered pytree because the `lax.scan` stacks it as its output."""

    round_poly: Array
    challenge: Array


class SumcheckSummand(Protocol):
    """The seam the homogeneous `prove` scan driver needs from a per-variable
    round: the round-poly `degree`, and the summand over the lifted factors. The
    driver owns the split / mask / fold / scan, so one scan serves every sumcheck.

    `degree` is a read-only property here so a frozen-dataclass field (product)
    and a `@property` (LogUp) both match. `combine` is the scalar-explicit form of
    the summand and `combine_scalars` the loop-invariant scalars it reads,
    hoisted out of the per-variable scan so they bind once rather than per round."""

    @property
    def degree(self) -> int: ...

    def combine_scalars(self) -> tuple[Array, ...]: ...

    def combine(self, scalars: Sequence[Array], *factors: Array) -> Array: ...

    def _combine(self, *factors: Array) -> Array: ...


def prove(
    round: SumcheckSummand,
    state: Sequence[Array],
    transcript: Transcript,
    *,
    eval_start: int = 0,
    challenge_dtype: object | None = None,
    challenge_limbs: int = 1,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Scan a sumcheck round once per variable; return the folded state, the
    advanced transcript, and the stacked per-round `RoundMsg` (`.round_poly` is the
    proof, `.challenge` is the evaluation point).

    `eval_start` controls the round-poly evaluation domain `[eval_start..degree]`:
    the default 0 sends `degree+1` values; 1 omits `s(0)` and sends `degree`, the
    compressed wire form a verifier reconstructs as `s(0) = claim - s(1)`.
    `challenge_dtype` / `challenge_limbs` pick the per-round fold challenge's
    field: the default (None / 1) folds with one transcript-field squeeze; a
    challenge in an extension reinterprets `challenge_limbs` consecutive squeezes
    as one `challenge_dtype` element (the `sample_challenge` packing, applied
    inside the scan), as a downstream scheme's extension-field sumchecks need."""
    state = list(state)
    if not state:
        raise ValueError("prove requires a non-empty state (one Array per factor)")
    width = state[0].shape[-1]
    if log2_strict_usize(width) == 0:
        raise ValueError(
            f"prove requires a state width >= 2 (at least one round), got width {width}"
        )
    if not 0 <= eval_start <= round.degree:
        raise ValueError(
            f"eval_start must be within [0, degree {round.degree}], got {eval_start}"
        )
    if challenge_limbs < 1:
        raise ValueError(f"challenge_limbs must be >= 1, got {challenge_limbs}")
    if challenge_dtype is None and challenge_limbs != 1:
        # The default squeeze is the identity reinterpret (one transcript-field
        # element). >1 limbs with no dtype to pack them into would advance the
        # transcript past squeezes the fold never consumes — a silent verifier
        # desync; reject it. (The dtype != None case is guarded at the squeeze.)
        raise ValueError(
            "challenge_limbs must be 1 when challenge_dtype is None, got "
            f"{challenge_limbs}"
        )
    if isinstance(transcript, DuplexTranscript) and transcript.fs_on_host:
        # The host sponge is an eager primitive (`device_put` / `.devices()` on its
        # inputs); it cannot run inside `_prove_scan`'s `lax.scan` body. A host-FS
        # transcript must drive the host-relaunch round engine, not this dense
        # scan prove — fail loud rather than abort deep in the scan trace with a
        # ConcretizationError.
        raise NotImplementedError(
            "fs_on_host is unsupported on the dense sumcheck scan path; drive the "
            "host-relaunch round engine instead"
        )
    return _prove_scan(
        round,
        state,
        transcript,
        eval_start=eval_start,
        challenge_dtype=challenge_dtype,
        challenge_limbs=challenge_limbs,
    )


def _prove_scan(
    round: SumcheckSummand,
    state: list[Array],
    transcript: Transcript,
    *,
    eval_start: int = 0,
    challenge_dtype: object | None = None,
    challenge_limbs: int = 1,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """The per-variable sumcheck scan: split / lift / round-poly / Fiat-Shamir /
    fold, one `lax.scan` step per variable. The body always runs, on a validated,
    non-empty `state` (`prove` guards the empty / zero-round cases)."""
    width = state[0].shape[-1]
    rounds = log2_strict_usize(width)
    degree = round.degree
    half_max = width // 2
    scalars = round.combine_scalars()

    def step(
        carry: tuple[list[Array], Transcript, Array], _: None
    ) -> tuple[tuple[list[Array], Transcript, Array], RoundMsg]:
        state, transcript, half = carry
        live = jnp.arange(half_max) < half
        pairs = [
            (buf[..., :half_max], lax.dynamic_slice_in_dim(buf, half, half_max, -1))
            for buf in state
        ]
        lifted = [lift_to_domain(p0, p1, degree, eval_start) for p0, p1 in pairs]
        integrand = round.combine(scalars, *lifted)
        msg = jnp.sum(
            jnp.where(live, integrand, jnp.zeros((), integrand.dtype)), axis=-1
        )
        # One round challenge: `challenge_limbs` squeezes reinterpreted as a single
        # `challenge_dtype` element (the `sample_challenge` packing) — the identity
        # squeeze at the defaults, an extension-field challenge otherwise.
        transcript, raw = transcript.observe_and_sample(msg, challenge_limbs)
        r = (
            raw[0]
            if challenge_dtype is None
            else reinterpret_challenge(raw, challenge_dtype)
        )
        state = [zero_extend(fold_pair(p0, p1, r), width) for p0, p1 in pairs]
        return (state, transcript, half // 2), RoundMsg(msg, r)

    init = (state, transcript, jnp.int32(half_max))
    (state, transcript, _), msgs = lax.scan(step, init, xs=None, length=rounds)
    return [buf[..., :1] for buf in state], transcript, msgs
