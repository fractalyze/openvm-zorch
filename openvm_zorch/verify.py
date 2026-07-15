"""End-to-end SWIRL verifier — the five stages checked from a proof + vk.

The structural dual of ``prove`` (``openvm_zorch/prove.py``): ``verify_chain``
builds a ``zorch.round.VerifyChain`` of one verifier Stage per prover stage —
``CommitVerifierStage`` / ``GkrVerifierStage`` / ``ZeroCheckVerifierStage`` /
``StackingVerifierStage`` / ``WhirVerifierStage``, the duals of ``prove_chain``'s
``CommitStage`` … ``WhirStage``. Each Stage re-derives its own Fiat-Shamir
challenges and checks the stage's algebraic relation, threading a witness-free
``VerifyCarry`` (the dual of ``ProveCarry``); the chain consumes the prover's
one-message-per-round proof, so a stage present on one side and not the other is
a structural reject, not a silent Fiat-Shamir desync. The verifier takes only
the verifying key (per-AIR shape + constraint DAG, no traces) and the proof. A
failed check raises ``VerificationError``; returning normally means the proof is
accepted.

The stage math lives with each stage, mirroring sp1-zorch's ``verify_shard`` /
per-stage ``verifier.py`` split (the shared scalar algebra is
``openvm_zorch/poly_common.py``); this module holds only the Stages, the carry,
and the driver. The stage duals follow the reference verifier
(crates/stark-backend/src/verifier):

- Stage 2 ``logup_gkr.verifier.verify_gkr_stage``: GKR fractional-sumcheck
  verify, ξ padding.
- Stage 3 ``logup_zerocheck.verifier.verify_zerocheck_stage``: the batched
  ZeroCheck+LogUp sumcheck, closed by re-evaluating the constraint/interaction
  claim at the folded point from the column openings.
- Stage 4 ``stacked_reduction.verifier.verify_stacked_reduction``: re-derive λ,
  check s₀ against the opening claims, run the sumcheck, close on the
  stacking-opening claim.
- Stage 5 ``whir.verifier.verify_whir``: μ batching, per-round sumcheck folds +
  OOD, the query phase (Merkle-path verification + k-fold codeword
  consistency), and the final WHIR polynomial constraint.

A verifier Stage raises ``VerificationError`` on its stage's check rather than
threading an ``ok`` (openvm's verifier checks were raise-based before the chain,
and keeping that is a pure refactor); each returns ``ok = True`` and the chain's
structural AND is the honest path's verdict.

PoW witnesses are checked, not re-ground. Opened rows and Merkle paths are
verified against the committed roots.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import jax.numpy as jnp
from jax import Array

from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.fields import F
from openvm_zorch.logup_gkr.verifier import verify_gkr_stage
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.logup_zerocheck.prover import BatchConstraintProof
from openvm_zorch.logup_zerocheck.verifier import verify_zerocheck_stage
from openvm_zorch.poly_common import VerificationError
from openvm_zorch.prove import GkrStageMsg, Proof, SystemParams
from openvm_zorch.stacked_reduction.prover import StackingProof
from openvm_zorch.stacked_reduction.verifier import verify_stacked_reduction
from openvm_zorch.whir.prover import WhirProof
from openvm_zorch.whir.verifier import verify_whir
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.round import Round, VerifyChain
from zorch.transcript import DuplexTranscript


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
class VerifyCarry:
    """What flows between the verifier's Stages: the verifying keys (set
    at construction) plus each stage's outputs the next stage reads — the
    witness-free dual of ``prove.ProveCarry``. Stages return it via
    ``replace`` — a stage writes its own fields and passes the rest through.

    Like ``ProveCarry`` (and for the same reason) this is a plain dataclass, not
    a registered pytree: the verifier's stages jit *internally* (Stage 5's WHIR
    islands), so the carry is host-side Python that never crosses a ``jax.jit``
    boundary and need not flatten.
    """

    # Verifying keys in input (verifying-key) order and stacking order — the
    # dual of ``ProveCarry.airs`` / ``sorted_airs``. The preamble observes per
    # AIR in input order; the stages consume stacking order.
    air_vks: Sequence[AirVk]
    sorted_vks: Sequence[AirVk]
    # Stage 1 (commit) output: the commitment, read by Stage 5 as the opened
    # root — dual of ``ProveCarry.root``.
    common_main_commit: Array | None = None
    # Stage 2 (GKR) outputs read by Stage 3: the sampled α/β, the padded point
    # ξ, and the reduced GKR numerator/denominator claims.
    alpha: Array | None = None
    beta: Array | None = None
    xi: list[Array] | None = None
    gkr_numer: Array | None = None
    gkr_denom: Array | None = None
    # Stage 3 (ZeroCheck) outputs: the sumcheck point read by Stage 4, plus the
    # proof's column openings Stage 4 batches. The prover reads the committed
    # matrix off the carry; the verifier reads its openings off the proof, so
    # they ride the carry from here — dual of ``ProveCarry.bcp_r``.
    r: list[Array] | None = None
    column_openings: Sequence[Sequence[Array]] | None = None
    # Stage 4 (stacking) outputs read by Stage 5: the opening point ``u`` and
    # the stacking openings (WHIR's running claim) — dual of
    # ``ProveCarry.stacking_u``.
    u: list[Array] | None = None
    stacking_openings: Sequence[Sequence[Array]] | None = None


class CommitVerifierStage(Round):
    """Stage 1 dual of ``CommitStage``: replays the preamble absorb stream (vk
    pre-hash, the commitment, then per AIR in *input* order an optional present
    flag, log height, and public values) with the proof's commitment message,
    and writes it onto the carry for the WHIR dual. No local check: the
    commitment is validated downstream by Stage 5's Merkle openings against this
    root (sp1-zorch's ``TraceCommitVerifierRound``)."""

    def __init__(self, *, vk_pre_hash: Sequence[int]) -> None:
        self._vk_pre_hash = vk_pre_hash

    def __call__(
        self, carry: VerifyCarry, msg: Array, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        transcript = transcript.observe(jnp.array(list(self._vk_pre_hash), dtype=F))
        transcript = transcript.observe(msg)
        for vk in carry.air_vks:
            meta: list[int] = [] if vk.is_required else [1]
            meta.append(vk.log_height)
            meta.extend(vk.public_values)
            transcript = transcript.observe(jnp.array(meta, dtype=F))
        carry = replace(carry, common_main_commit=msg)
        return carry, transcript, jnp.bool_(True)


class GkrVerifierStage(Round):
    """Stage 2 dual of ``GkrStage`` over ``verify_gkr_stage``: writes α/β, the
    padded point ξ, and the GKR claims onto the carry for ZeroCheck. The message
    is the GKR stage's proof contribution (``GkrStageMsg``); its ``xi`` field is
    the prover's record — the verifier re-derives ξ rather than trusting it,
    exactly as the flat verifier did."""

    def __init__(
        self,
        *,
        params: SystemParams,
        total_interactions: int,
        n_logup: int,
        n_global: int,
    ) -> None:
        self._params = params
        self._total_interactions = total_interactions
        self._n_logup = n_logup
        self._n_global = n_global

    def __call__(
        self, carry: VerifyCarry, msg: GkrStageMsg, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        transcript, alpha, beta, xi, p_xi, q_xi = verify_gkr_stage(
            transcript,
            self._params.l_skip,
            self._params.logup_pow_bits,
            self._total_interactions,
            self._n_logup,
            self._n_global,
            msg.gkr_proof,
            msg.logup_pow_witness,
        )
        carry = replace(
            carry, alpha=alpha, beta=beta, xi=xi, gkr_numer=p_xi, gkr_denom=q_xi
        )
        return carry, transcript, jnp.bool_(True)


class ZeroCheckVerifierStage(Round):
    """Stage 3 dual of ``ZeroCheckStage`` over ``verify_zerocheck_stage``:
    consumes the Stage-2 outputs off the carry, verifies the batched ZeroCheck +
    LogUp sumcheck, and writes the sumcheck point ``r`` plus the proof's column
    openings (Stage 4 batches them) onto the carry."""

    def __init__(self, *, params: SystemParams, n_logup: int, n_max: int) -> None:
        self._params = params
        self._n_logup = n_logup
        self._n_max = n_max

    def __call__(
        self,
        carry: VerifyCarry,
        msg: BatchConstraintProof,
        transcript: DuplexTranscript,
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        transcript, r = verify_zerocheck_stage(
            transcript,
            self._params.l_skip,
            self._params.max_constraint_degree,
            carry.sorted_vks,
            self._n_logup,
            self._n_max,
            msg,
            carry.alpha,
            carry.beta,
            carry.xi,
            carry.gkr_numer,
            carry.gkr_denom,
        )
        carry = replace(carry, r=r, column_openings=msg.column_openings)
        return carry, transcript, jnp.bool_(True)


class StackingVerifierStage(Round):
    """Stage 4 dual of ``StackingStage`` over ``verify_stacked_reduction``:
    rebuilds the stacked layout from the verifying keys, batches the column
    openings off the carry, and verifies the stacked opening reduction. Writes
    the opening point ``u`` and the proof's stacking openings (WHIR's running
    claim) onto the carry."""

    def __init__(self, *, params: SystemParams) -> None:
        self._params = params

    def __call__(
        self, carry: VerifyCarry, msg: StackingProof, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        sorted_vks = carry.sorted_vks
        layout = StackedLayout.new(
            self._params.l_skip,
            self._params.l_skip + self._params.n_stack,
            [(vk.width, vk.log_height) for vk in sorted_vks],
        )
        need_rot = [vk.needs_next for vk in sorted_vks]
        transcript, u = verify_stacked_reduction(
            transcript,
            self._params.l_skip,
            self._params.n_stack,
            msg,
            layout,
            need_rot,
            carry.column_openings,
            carry.r,
        )
        carry = replace(carry, u=u, stacking_openings=msg.stacking_openings)
        return carry, transcript, jnp.bool_(True)


class WhirVerifierStage(Round):
    """Stage 5 dual of ``WhirStage`` over ``verify_whir``: forms ``u_cube`` from
    the opening point on the carry (the same Stage-4 → Stage-5 handoff
    ``u_cube = (u₀ squarings over the skip domain) ‖ u[1..]`` the prover does),
    then checks WHIR against the carry's commitment and stacking openings."""

    def __init__(
        self, sponge: Sponge, compressor: Compression, *, params: SystemParams
    ) -> None:
        self._sponge = sponge
        self._compressor = compressor
        self._params = params

    def __call__(
        self, carry: VerifyCarry, msg: WhirProof, transcript: DuplexTranscript
    ) -> tuple[VerifyCarry, DuplexTranscript, Array]:
        u = carry.u
        u_cube = [u[0]]
        for _ in range(self._params.l_skip - 1):
            u_cube.append(u_cube[-1] * u_cube[-1])
        u_cube.extend(u[1:])
        transcript = verify_whir(
            transcript,
            self._sponge,
            self._compressor,
            self._params.l_skip,
            self._params.n_stack,
            self._params.log_blowup,
            self._params.whir,
            msg,
            carry.stacking_openings,
            [carry.common_main_commit],
            u_cube,
        )
        return carry, transcript, jnp.bool_(True)


def verify_chain(
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    air_vks: Sequence[AirVk],
) -> tuple[VerifyChain, VerifyCarry]:
    """Build the SWIRL verifier as one ``VerifyChain`` of Stages plus its
    initial carry — the dual of ``prove.prove_chain``. One definition of the
    stage wiring so ``verify`` and any future per-stage verify-timing harness
    cannot drift on it (sp1-zorch's ``verify_shard_chain`` pattern).

    The protocol-derived sizes (stacking order, ``n_logup`` / ``n_max`` /
    ``n_global``) are computed here from the verifying keys — the same values
    ``prove_chain`` derives from the traces — and bound onto the Stages. Returns
    the carry alongside the chain because the stacking order it derives is also
    the carry's statement.
    """
    l_skip = params.l_skip
    order = sorted(range(len(air_vks)), key=lambda i: (-air_vks[i].log_height, i))
    sorted_vks = [air_vks[i] for i in order]

    n_per_trace = [vk.log_height - l_skip for vk in sorted_vks]
    total_interactions = sum(
        len(vk.dag.interactions) << (l_skip + max(n, 0))
        for vk, n in zip(sorted_vks, n_per_trace)
    )
    n_logup = total_interactions.bit_length() - l_skip if total_interactions else 0
    n_max = max(max(n_per_trace), 0)
    n_global = max(n_max, n_logup)

    chain = VerifyChain(
        [
            CommitVerifierStage(vk_pre_hash=vk_pre_hash),
            GkrVerifierStage(
                params=params,
                total_interactions=total_interactions,
                n_logup=n_logup,
                n_global=n_global,
            ),
            ZeroCheckVerifierStage(params=params, n_logup=n_logup, n_max=n_max),
            StackingVerifierStage(params=params),
            WhirVerifierStage(sponge, compressor, params=params),
        ]
    )
    return chain, VerifyCarry(air_vks=air_vks, sorted_vks=sorted_vks)


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

    A thin driver over ``verify_chain`` (the dual of ``prove``): deconstruct the
    ``Proof`` into the per-round message list — the inverse of ``prove``'s
    assembly, ``[commit, gkr, bcp, stacking, whir]`` — and replay the chain.
    Each verifier Stage raises ``VerificationError`` on its stage's failed
    check, so rejection flows through exactly as the flat verifier's did; ``ok``
    is the chain's structural AND of the stages, guarded here so a future
    ok-returning check cannot silently pass.
    """
    chain, carry = verify_chain(sponge, compressor, params, vk_pre_hash, air_vks)
    msgs = [
        common_main_commit,
        GkrStageMsg(proof.logup_pow_witness, proof.gkr_proof, proof.xi),
        proof.batch_constraint_proof,
        proof.stacking_proof,
        proof.whir_proof,
    ]
    _, _, ok = chain(carry, msgs, transcript)
    if not bool(ok):
        raise VerificationError("verification failed")
