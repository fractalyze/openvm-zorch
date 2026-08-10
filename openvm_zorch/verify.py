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

from typing import Sequence

from frx import Array
from hash_frx.compression import Compression
from hash_frx.sponge import Sponge
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import DuplexTranscript

from openvm_zorch.logup_gkr.verifier import LogupGkrVerifier
from openvm_zorch.logup_zerocheck.verifier import ZerocheckVerifier
from openvm_zorch.poly_common import VerificationError
from openvm_zorch.prove import bind_commitment
from openvm_zorch.stacked_reduction.verifier import StackingVerifier
from openvm_zorch.types import (
    AirShape,
    AirVk,
    CachedMainCommitments,
    LogupGkrProof,
    Proof,
    SystemClaim,
    SystemParams,
    SystemShape,
)
from openvm_zorch.whir.verifier import StackedWhirPcsVerifier


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
