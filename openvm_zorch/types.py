# Copyright 2026 The openvm-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The system's claims, witnesses and wire types.

Separate from `prove.py` so a verifier reads a claim or a proof section without
importing the prover that produced it. The two roles of a claim reduction are
separately deployable (`zorch.stage`), which a shared type module is what makes
possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from frx import Array

if TYPE_CHECKING:
    from openvm_zorch.commit.trace_commit import StackedPcsData
    from openvm_zorch.logup_gkr.prover import FracSumcheckProof
    from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
    from openvm_zorch.logup_zerocheck.prover import BatchConstraintProof
    from openvm_zorch.stacked_reduction.prover import StackingProof
    from openvm_zorch.whir.prover import WhirConfig, WhirProof

# --- Vocabulary: per-AIR shape and the commitments a proof carries. -------


@dataclass(frozen=True)
class AirInstance:
    """One AIR with its trace, in input (verifying-key) order."""

    trace: Array  # (height, width) base field — the common main
    dag: ConstraintsDag
    public_values: tuple[int, ...]
    constraint_degree: int
    needs_next: bool
    is_required: bool
    # Cached-main partitions (base-field ``(height, width)`` matrices, in
    # partition order, same height as ``trace``). The prover's partitioned main
    # is ``cached_mains ++ [common_main]``, so a DAG ``main`` node with
    # ``part_index`` k < len(cached_mains) reads a cached part and the last index
    # reads ``trace``. The synthetic fixture has none (``()``), so this only
    # fires on a real openvm block (e.g. ProgramAir's cached columns).
    cached_mains: tuple[Array, ...] = ()
    # Verifying-key position. On a real block the present AIRs are a sparse
    # subset of the pk's AIRs; the gaps (unexercised chips) are absent AIRs the
    # prelude still observes a present=0 flag for. ``None`` ⇒ contiguous
    # all-present (the synthetic fixture), so the prelude sees no gaps.
    air_idx: int | None = None


@dataclass(frozen=True, kw_only=True)
class AirShape:
    """One AIR's public structure for this block.

    Keyword-only: ``log_height`` and ``air_idx`` are both plain ints naming
    different axes, and a positional swap would place a trace height in a
    verifying-key slot without any type error.
    """

    log_height: int
    public_values: tuple[int, ...]
    is_required: bool
    air_idx: int | None


@dataclass(frozen=True)
class AirVk:
    """Per-AIR verifying-key shape the verifier consumes, in input order."""

    dag: ConstraintsDag
    log_height: int
    width: int  # common-main column count
    public_values: tuple[int, ...]
    constraint_degree: int
    needs_next: bool
    is_required: bool


@dataclass(frozen=True)
class SystemParams:
    """The reference ``SystemParams`` fields the prover consumes."""

    l_skip: int
    n_stack: int
    log_blowup: int
    logup_pow_bits: int
    max_constraint_degree: int
    whir: WhirConfig


@dataclass(frozen=True, kw_only=True)
class SystemShape:
    """The protocol sizes the AIR heights determine, in stacking order.

    Derived once and carried on the claim rather than configured onto each
    role: every size here varies per block, and the prover reads the heights
    off traces while the verifier reads them off verifying keys — two sources
    for one fact unless the derivation itself is shared.
    """

    order: tuple[int, ...]  # input index of each AIR, in stacking order
    log_heights: tuple[int, ...]  # stacking order
    total_interactions: int
    n_logup: int
    n_max: int
    n_global: int

    @classmethod
    def of(
        cls,
        *,
        l_skip: int,
        log_heights: Sequence[int],
        interaction_counts: Sequence[int],
    ) -> SystemShape:
        """Stacking order (descending height, ties by input index) and the sizes
        ``calculate_n_logup`` derives, from per-AIR heights in *input* order."""
        order = tuple(
            sorted(range(len(log_heights)), key=lambda i: (-log_heights[i], i))
        )
        sorted_heights = tuple(log_heights[i] for i in order)
        total = sum(interaction_counts[i] << max(log_heights[i], l_skip) for i in order)
        n_logup = total.bit_length() - l_skip if total else 0
        n_max = max(max(h - l_skip, 0) for h in sorted_heights)
        return cls(
            order=order,
            log_heights=sorted_heights,
            total_interactions=total,
            n_logup=n_logup,
            n_max=n_max,
            n_global=max(n_max, n_logup),
        )


@dataclass(frozen=True, kw_only=True)
class CachedMainCommitments:
    """The cached/preprocessed main commitments, committed before any claim
    exists.

    Prover data, not claim data: the prelude observes each root, and the
    stacked reduction and WHIR open the matrices, but no reduced claim carries
    them.
    """

    # Per input-order AIR, that AIR's cached commitments (empty when it has
    # none). The prelude walks this in input order.
    by_air: tuple[tuple[StackedPcsData, ...], ...]
    # The same commitments flattened in stacking order — the order the stacked
    # reduction and WHIR consume them, common main first.
    stacking_order: tuple[StackedPcsData, ...]

    @classmethod
    def none(cls, num_airs: int) -> CachedMainCommitments:
        """No AIR has a cached main — the shape the synthetic fixture and the
        verifier's prelude replay both take."""
        return cls(by_air=((),) * num_airs, stacking_order=())


@dataclass(frozen=True, kw_only=True)
class StackedCommitData:
    """What the stacked PCS retains between its commit and open halves.

    Not prover-only: the common-main root reaches the wire verbatim as the
    proof's commitment, and each Merkle tree's opened rows and paths ride the
    WHIR proof.
    """

    common: StackedPcsData
    cached: CachedMainCommitments


# --- The system statement and the trace that satisfies it. ----------------


@dataclass(frozen=True, kw_only=True)
class SystemClaim:
    """Every present AIR's trace satisfies its constraints, and the LogUp
    interactions across all of them balance.

    The root statement a SWIRL proof discharges; it carries only what varies per
    block, the circuit-fixed properties being role configuration.
    """

    air_shapes: tuple[AirShape, ...]  # input (verifying-key) order
    shape: SystemShape


@dataclass(frozen=True, kw_only=True)
class SystemWitness:
    """The traces the system claim is about, in stacking order, with the
    cached-main commitments already taken."""

    sorted_airs: tuple[AirInstance, ...]
    cached: CachedMainCommitments


# --- Reduction 1: LogUp-GKR — bus balance to column openings. -------------


@dataclass(frozen=True, kw_only=True)
class LogupClaim:
    """The system's LogUp fraction sums collapse, under the batching challenges
    α and β, to a single fraction-sum claim at the point ξ.

    The claimed numerator and denominator are not carried: the verifier
    re-derives them from the reduction proof and the prover recomputes the input
    layer, so neither reads them off a shared field.
    """

    system: SystemClaim
    alpha: Array
    beta: Array
    xi: list[Array]


@dataclass(frozen=True, kw_only=True)
class VerifiedLogupClaim(LogupClaim):
    """``LogupClaim`` plus the fraction sums the verifier derived at ξ.

    The verifier knows strictly more than the prover here: it recovers the
    claimed numerator and denominator by replaying the reduction proof, where
    the prover recomputes the input layer from the witness and never needs
    them. Naming that as its own type keeps the extra data out of the shared
    claim without hiding it in mutable role state.
    """

    numerator: Array
    denominator: Array


@dataclass(frozen=True)
class LogupGkrProof:
    """Discharges the system claim into the LogUp fraction-sum claim at ξ: the
    LogUp PoW witness, the fractional-sumcheck proof, and the padded evaluation
    point.

    ``xi`` is on the reduced claim too — zerocheck reads it there — but it is a
    wire field as well, so it rides the reduction proof for assembly."""

    logup_pow_witness: Array
    gkr_proof: FracSumcheckProof
    xi: list[Array]


# --- Reduction 2: zerocheck — constraints to column openings. -------------


@dataclass(frozen=True, kw_only=True)
class ColumnOpeningClaim:
    """Each AIR's columns open to ``column_openings`` at the zerocheck point
    ``r``."""

    system: SystemClaim
    r: list[Array]
    column_openings: Sequence[Sequence[Array]]


# --- Reduction 3: stacking, then the WHIR PCS open. -----------------------


@dataclass(frozen=True, kw_only=True)
class StackedOpeningClaim:
    """The committed stacked matrix's columns open to ``stacking_openings`` at
    the point ``u`` — the claim the PCS discharges."""

    commitment: Array
    u: list[Array]
    stacking_openings: Sequence[Sequence[Array]]


# --- The composite, naming each reduction's proof. ------------------------


@dataclass(frozen=True)
class Proof:
    """The three reduction proofs plus the commitment and the PCS opening."""

    common_main_commit: Array  # (8,) F
    logup_pow_witness: Array
    gkr_proof: FracSumcheckProof
    xi: list[Array]  # padded to l_skip + n_global
    batch_constraint_proof: BatchConstraintProof
    stacking_proof: StackingProof
    whir_proof: WhirProof
