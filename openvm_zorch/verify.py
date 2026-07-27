"""End-to-end SWIRL verifier — the dual of ``prove``'s reduction chain.

The structural dual of ``openvm_zorch/prove.py``: one verifier role per prover
role, checking the same reductions in the same order against the proof. Each
role re-derives its own Fiat-Shamir challenges and checks its reduction's
algebraic relation, producing the reduced claim its successor consumes — so a
role present on one side and not the other is a structural break, not a silent
Fiat-Shamir desync.

The verifier takes only the verifying key (per-AIR shape + constraint DAG, no
traces) and the proof. A failed check raises ``VerificationError``; returning
normally means the proof is accepted.

The reduction math lives with each protocol, mirroring sp1-zorch's
``verify_shard`` / per-stage ``verifier.py`` split (the shared scalar algebra is
``openvm_zorch/poly_common.py``); this module holds only the roles and the
composite. The duals follow the reference verifier
(crates/stark-backend/src/verifier):

- ``logup_gkr.verifier.verify_gkr_stage``: GKR fractional-sumcheck verify, ξ
  padding.
- ``logup_zerocheck.verifier.verify_zerocheck_stage``: the batched ZeroCheck +
  LogUp sumcheck, closed by re-evaluating the constraint/interaction claim at
  the folded point from the column openings.
- ``stacked_reduction.verifier.verify_stacked_reduction``: re-derive λ, check s₀
  against the opening claims, run the sumcheck, close on the stacking-opening
  claim.
- ``whir.verifier.verify_whir``: μ batching, per-round sumcheck folds + OOD, the
  query phase (Merkle-path verification + k-fold codeword consistency), and the
  final WHIR polynomial constraint.

A verifier role raises ``VerificationError`` on its own check rather than
threading an ``ok`` (openvm's verifier checks were raise-based before the roles
existed, and keeping that is a pure refactor); each returns ``ok = True`` and
the composite's AND is the honest path's verdict.

PoW witnesses are checked, not re-ground. Opened rows and Merkle paths are
verified against the committed roots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import frx.numpy as fnp
from frx import Array
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import DuplexTranscript

from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.logup_gkr.verifier import verify_gkr_stage
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.logup_zerocheck.prover import BatchConstraintProof
from openvm_zorch.logup_zerocheck.verifier import verify_zerocheck_stage
from openvm_zorch.poly_common import VerificationError
from openvm_zorch.prove import (
    AirShape,
    CachedMainCommitments,
    ColumnOpeningClaim,
    LogupClaim,
    LogupGkrProof,
    Proof,
    StackedOpeningClaim,
    SystemClaim,
    SystemParams,
    SystemShape,
    bind_commitment,
)
from openvm_zorch.stacked_reduction.prover import StackingProof
from openvm_zorch.stacked_reduction.verifier import verify_stacked_reduction
from openvm_zorch.whir.prover import WhirProof
from openvm_zorch.whir.verifier import verify_whir


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


class LogupGkrVerifier(
    VerifierStage[SystemClaim, VerifiedLogupClaim, LogupGkrProof, DuplexTranscript]
):
    """The dual of ``LogupGkrProver``: check the LogUp PoW witness, re-derive
    α/β and ξ, and verify the fractional sumcheck."""

    def __init__(self, *, params: SystemParams) -> None:
        self._params = params

    def verify(
        self,
        claim: SystemClaim,
        reduction_proof: LogupGkrProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[VerifiedLogupClaim, DuplexTranscript]:
        shape = claim.shape
        transcript, alpha, beta, xi, p_xi, q_xi = verify_gkr_stage(
            transcript,
            self._params.l_skip,
            self._params.logup_pow_bits,
            shape.total_interactions,
            shape.n_logup,
            shape.n_global,
            reduction_proof.gkr_proof,
            reduction_proof.logup_pow_witness,
        )
        return VerifyResult(
            VerifiedLogupClaim(
                system=claim,
                alpha=alpha,
                beta=beta,
                xi=xi,
                numerator=p_xi,
                denominator=q_xi,
            ),
            transcript,
            fnp.bool_(True),
        )


class ZerocheckVerifier(
    VerifierStage[
        VerifiedLogupClaim,
        ColumnOpeningClaim,
        BatchConstraintProof,
        DuplexTranscript,
    ]
):
    """The dual of ``ZerocheckProver``: verify the batched ZeroCheck + LogUp
    sumcheck and produce the per-column opening claims at ``r``.

    The column openings come off the reduction proof — the prover reads the
    committed matrix, the verifier reads the values the proof claims for it.
    """

    def __init__(self, *, params: SystemParams, air_vks: Sequence[AirVk]) -> None:
        self._params = params
        self._air_vks = air_vks

    def verify(
        self,
        claim: VerifiedLogupClaim,
        reduction_proof: BatchConstraintProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[ColumnOpeningClaim, DuplexTranscript]:
        shape = claim.system.shape
        sorted_vks = [self._air_vks[i] for i in shape.order]
        transcript, r = verify_zerocheck_stage(
            transcript,
            self._params.l_skip,
            self._params.max_constraint_degree,
            sorted_vks,
            shape.n_logup,
            shape.n_max,
            reduction_proof,
            claim.alpha,
            claim.beta,
            claim.xi,
            claim.numerator,
            claim.denominator,
        )
        return VerifyResult(
            ColumnOpeningClaim(
                system=claim.system,
                r=r,
                column_openings=reduction_proof.column_openings,
            ),
            transcript,
            fnp.bool_(True),
        )


class StackingVerifier(
    VerifierStage[
        ColumnOpeningClaim, StackedOpeningClaim, StackingProof, DuplexTranscript
    ]
):
    """The dual of ``StackingProver``: rebuild the stacked layout from the
    verifying keys, batch the incoming column openings, and verify the stacked
    opening reduction."""

    def __init__(
        self,
        *,
        params: SystemParams,
        air_vks: Sequence[AirVk],
        commitment: Array,
    ) -> None:
        self._params = params
        self._air_vks = air_vks
        # The commitment the opening claim is about. It reaches the verifier
        # alongside the proof rather than inside it, so it is bound here.
        self._commitment = commitment

    def verify(
        self,
        claim: ColumnOpeningClaim,
        reduction_proof: StackingProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[StackedOpeningClaim, DuplexTranscript]:
        sorted_vks = [self._air_vks[i] for i in claim.system.shape.order]
        layout = StackedLayout.new(
            self._params.l_skip,
            self._params.l_skip + self._params.n_stack,
            [(vk.width, vk.log_height) for vk in sorted_vks],
        )
        transcript, u = verify_stacked_reduction(
            transcript,
            self._params.l_skip,
            self._params.n_stack,
            reduction_proof,
            layout,
            [vk.needs_next for vk in sorted_vks],
            claim.column_openings,
            claim.r,
        )
        return VerifyResult(
            StackedOpeningClaim(
                commitment=self._commitment,
                u=u,
                stacking_openings=reduction_proof.stacking_openings,
            ),
            transcript,
            fnp.bool_(True),
        )


class StackedWhirPcsVerifier(
    VerifierStage[StackedOpeningClaim, TrivialClaim, WhirProof, DuplexTranscript]
):
    """The open half of the stacked PCS, verifier side: form ``u_cube`` from
    the claim's opening point — the same handoff the prover does — and check
    WHIR against the claim's commitment and opening values.

    Discharges the opening claim into the trivial claim, which is what makes a
    SWIRL proof a complete argument.
    """

    def __init__(
        self, sponge: Sponge, compressor: Compression, *, params: SystemParams
    ) -> None:
        self._sponge = sponge
        self._compressor = compressor
        self._params = params

    def verify(
        self,
        claim: StackedOpeningClaim,
        reduction_proof: WhirProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[TrivialClaim, DuplexTranscript]:
        u_cube = [claim.u[0]]
        for _ in range(self._params.l_skip - 1):
            u_cube.append(u_cube[-1] * u_cube[-1])
        u_cube.extend(claim.u[1:])
        transcript = verify_whir(
            transcript,
            self._sponge,
            self._compressor,
            self._params.l_skip,
            self._params.n_stack,
            self._params.log_blowup,
            self._params.whir,
            reduction_proof,
            claim.stacking_openings,
            [claim.commitment],
            u_cube,
        )
        return VerifyResult(TrivialClaim(), transcript, fnp.bool_(True))


class SwirlVerifier(VerifierStage[SystemClaim, TrivialClaim, Proof, DuplexTranscript]):
    """The SWIRL verifier: bind the commitment, then check three reductions and
    the PCS opening.

    A composite role, the dual of ``SwirlProver``: one definition of the wiring
    so the verifier and any per-role verify-timing harness cannot drift on it.
    """

    def __init__(
        self,
        sponge: Sponge,
        compressor: Compression,
        params: SystemParams,
        vk_pre_hash: Sequence[int],
        air_vks: Sequence[AirVk],
        commitment: Array,
    ) -> None:
        self._vk_pre_hash = vk_pre_hash
        self.gkr = LogupGkrVerifier(params=params)
        self.zerocheck = ZerocheckVerifier(params=params, air_vks=air_vks)
        self.stacking = StackingVerifier(
            params=params, air_vks=air_vks, commitment=commitment
        )
        self.opening = StackedWhirPcsVerifier(sponge, compressor, params=params)

    def verify(
        self,
        claim: SystemClaim,
        reduction_proof: Proof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[TrivialClaim, DuplexTranscript]:
        transcript = bind_commitment(
            transcript,
            claim,
            reduction_proof.common_main_commit,
            # The verifier holds no cached-main commitments: on the shapes this
            # consumer supports there are none, so the prelude observes none. A
            # block carrying them would need their roots on the verifying key.
            CachedMainCommitments.none(len(claim.air_shapes)),
            vk_pre_hash=self._vk_pre_hash,
        )
        gkr = self.gkr.verify(
            claim,
            LogupGkrProof(
                reduction_proof.logup_pow_witness,
                reduction_proof.gkr_proof,
                reduction_proof.xi,
            ),
            transcript,
        )
        zerocheck = self.zerocheck.verify(
            gkr.reduced_claim,
            reduction_proof.batch_constraint_proof,
            gkr.transcript,
        )
        stacking = self.stacking.verify(
            zerocheck.reduced_claim,
            reduction_proof.stacking_proof,
            zerocheck.transcript,
        )
        opening = self.opening.verify(
            stacking.reduced_claim, reduction_proof.whir_proof, stacking.transcript
        )
        ok = gkr.ok & zerocheck.ok & stacking.ok & opening.ok
        return VerifyResult(TrivialClaim(), opening.transcript, ok)


def vk_claim(params: SystemParams, air_vks: Sequence[AirVk]) -> SystemClaim:
    """The root claim the verifying keys state, in input order.

    The dual of ``prove.system_claim``, which reads the same structure off the
    traces. Both go through one ``SystemShape.of``, so the two roles cannot
    disagree on the stacking order or the protocol sizes.
    """
    return SystemClaim(
        air_shapes=tuple(
            AirShape(
                log_height=vk.log_height,
                public_values=vk.public_values,
                is_required=vk.is_required,
                # The verifying key carries no vk position of its own: this
                # consumer supports the contiguous all-present shape, where the
                # prelude sees no absent-AIR gaps.
                air_idx=None,
            )
            for vk in air_vks
        ),
        shape=SystemShape.of(
            l_skip=params.l_skip,
            log_heights=[vk.log_height for vk in air_vks],
            interaction_counts=[len(vk.dag.interactions) for vk in air_vks],
        ),
    )


def verify(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    air_vks: Sequence[AirVk],
    common_main_commit: Array,
    proof: Proof,
) -> None:
    """Verify a SWIRL proof. Raises ``VerificationError`` on any failed check;
    returns ``None`` if the proof is accepted.

    The claim is rebuilt from the verifying keys — the same ``SystemShape``
    derivation the prover runs off the traces, so the two cannot disagree on
    the protocol sizes.
    """
    claim = vk_claim(params, air_vks)
    verifier = SwirlVerifier(
        sponge, compressor, params, vk_pre_hash, air_vks, common_main_commit
    )
    result = verifier.verify(claim, proof, transcript)
    if not bool(result.ok):
        raise VerificationError("verification failed")
