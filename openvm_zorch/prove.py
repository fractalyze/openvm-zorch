"""End-to-end SWIRL prover — three claim reductions bracketed by the PCS.

Mirrors the reference ``Coordinator::prove``. The composition reads as a chain
of reductions: a system claim (every AIR's trace satisfies its constraints and
the LogUp interactions balance) is reduced by LogUp-GKR to a claim about the
fraction sums at ξ, by zerocheck to per-column opening claims at ``r``, and by
the stacked reduction to opening claims on the committed matrix at ``u`` — which
the stacked PCS then discharges, leaving nothing to prove.

The PCS brackets those three: ``StackedWhirPcs.commit`` binds the traces before
any challenge is drawn, and ``.prove`` opens them at a point that only exists
once the reductions have run. Fiat-Shamir is the only reason the halves sit far
apart; ``StackedCommitData`` names what crosses between them.

Cached-main commitments are a **committer**, not a stage: they run before any
claim exists (native commits them during tracegen, outside the timed prove) and
their output is prover data no reduced claim carries.

The protocol-derived sizes the coordinator owns — stacking order, ``n_logup``,
``n_max``, ``n_global`` — are ``SystemShape``, derived once from the AIR heights
so the prover and verifier cannot disagree on them.

Reference: `Coordinator::prove` (prover/coordinator.rs) and
`prove_openings` (prover/cpu_backend.rs) for the stacking → WHIR handoff
``u_cube = (u₀ squarings over the skip domain) ‖ u[1..]``.
"""

from __future__ import annotations

from typing import Sequence

import frx.numpy as fnp
from frx import Array
from zk_dtypes import babybear_mont as F
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize

from openvm_zorch.commit.trace_commit import StackedPcsData, stacked_commit
from openvm_zorch.logup_gkr.prover import LogupGkrProver
from openvm_zorch.logup_zerocheck.prover import ZerocheckProver
from openvm_zorch.stacked_reduction.prover import StackingProver
from openvm_zorch.types import (
    AirInstance,
    AirShape,
    CachedMainCommitments,
    Proof,
    SystemClaim,
    SystemParams,
    SystemShape,
    SystemWitness,
)
from openvm_zorch.whir.prover import StackedWhirPcs


def system_claim(l_skip: int, airs: Sequence[AirInstance]) -> SystemClaim:
    """The root claim a set of traces makes, in input (verifying-key) order."""
    log_heights = [log2_strict_usize(a.trace.shape[0]) for a in airs]
    return SystemClaim(
        air_shapes=tuple(
            AirShape(
                log_height=h,
                public_values=a.public_values,
                is_required=a.is_required,
                air_idx=a.air_idx,
            )
            for a, h in zip(airs, log_heights)
        ),
        shape=SystemShape.of(
            l_skip=l_skip,
            log_heights=log_heights,
            interaction_counts=[len(a.dag.interactions) for a in airs],
        ),
    )


def _log_prelude_obs_diff(obs: Sequence[Array], obs_log: dict) -> None:
    """Diagnostic (issue #59): flatten zorch's prelude observation sequence to
    canonical u32 and diff it element-by-element against the reference
    observation-log prefix the fixture carries. Prints the first divergence (or
    confirms a full match up to the grind boundary), so a prelude mismatch is
    pinned exactly instead of inferred from the cascade of MISMATCH labels
    downstream of the grind (the first real-block divergence,
    ``logup_pow_witness``, is observed right after this prelude)."""
    got: list[int] = []
    for a in obs:
        got.extend(int(v) for v in fnp.atleast_1d(a).astype(fnp.uint32))
    want = [int(v) for v in obs_log["values"]]
    plen = int(obs_log["prelude_len_faithful"])
    n = min(len(got), plen, len(want))
    first = next((i for i in range(n) if got[i] != want[i]), None)
    if first is None and len(got) == plen:
        print(
            f"[prelude obs-diff] MATCH: all {plen} prelude observations agree "
            "with the reference -- the grind input state is byte-identical",
            flush=True,
        )
        return
    if first is None:
        print(
            f"[prelude obs-diff] LENGTH MISMATCH: zorch observed {len(got)} "
            f"prelude elements, reference prelude_len={plen} "
            "(the shared prefix agrees)",
            flush=True,
        )
        return
    lo, hi = max(0, first - 2), min(n, first + 3)
    print(
        f"[prelude obs-diff] FIRST DIVERGENCE at index {first} "
        f"(reference prelude_len={plen}, zorch len={len(got)}):",
        flush=True,
    )
    for i in range(lo, hi):
        mark = "  <-- first diff" if i == first else ""
        print(f"    [{i}] got={got[i]} want={want[i]}{mark}", flush=True)


def prelude_observations(
    *,
    vk_pre_hash: Sequence[int],
    commitment: Array,
    air_shapes: Sequence[AirShape],
    cached_roots: Sequence[Sequence[Array]],
) -> list[Array]:
    """The prelude absorb stream, in order: vk pre-hash, the common-main
    commitment, then per AIR in *input* order a present flag (non-required AIRs
    only), the log height, each cached-main root, and the public values.

    Both roles call this one function, so an ordering edit cannot land on one
    side of the protocol alone. Reference prover/mod.rs:155-175: iterate ALL vk
    AIRs in order; on a real block the present AIRs are a sparse subset of the
    pk, and each unexercised gap still contributes a present=0 flag. The
    synthetic fixture is contiguous and all-present (``air_idx`` ``None``), so
    it sees no gaps and no cached roots.
    """
    obs: list[Array] = [fnp.array(list(vk_pre_hash), dtype=F), commitment]
    prev = -1
    for shape, cached in zip(air_shapes, cached_roots):
        idx = shape.air_idx if shape.air_idx is not None else prev + 1
        # Absent (unexercised) AIRs between the last present AIR and this one
        # are non-required (a required AIR is always present), so each observes
        # a present=0 flag and nothing else.
        for _absent in range(prev + 1, idx):
            obs.append(fnp.array([0], dtype=F))
        prev = idx
        head: list[int] = [] if shape.is_required else [1]
        head.append(shape.log_height)
        if not cached:
            head.extend(shape.public_values)
            obs.append(fnp.array(head, dtype=F))
            continue
        obs.append(fnp.array(head, dtype=F))
        obs.extend(cached)
        if shape.public_values:
            obs.append(fnp.array(list(shape.public_values), dtype=F))
    return obs


def bind_commitment(
    transcript: DuplexTranscript,
    claim: SystemClaim,
    commitment: Array,
    cached: CachedMainCommitments,
    *,
    vk_pre_hash: Sequence[int],
    obs_log: dict | None = None,
) -> DuplexTranscript:
    """Absorb the prelude, binding the commitment and the claim's public
    structure into the transcript before the first challenge is drawn.

    A shared function both roles call between the PCS's halves, not a stage: it
    proves nothing on its own and owns no proof section.
    """
    obs = prelude_observations(
        vk_pre_hash=vk_pre_hash,
        commitment=commitment,
        air_shapes=claim.air_shapes,
        cached_roots=[[cd.commit for cd in cds] for cds in cached.by_air],
    )
    if obs_log is not None:
        _log_prelude_obs_diff(obs, obs_log)
    for o in obs:
        transcript = transcript.observe(o)
    return transcript


def commit_cached_mains(
    sponge: Sponge,
    compressor: Compression,
    *,
    l_skip: int,
    n_stack: int,
    log_blowup: int,
    k: int,
    airs: Sequence[AirInstance],
    order: Sequence[int],
) -> CachedMainCommitments:
    """Commit each cached main as its own stacked commitment (reference
    cpu_backend.rs ``pre_cached_pcs_data_per_commit`` — one PcsData per
    cached/preprocessed trace).

    A committer: it runs before any claim exists, so it is called in build
    scope rather than inside the timed prove, matching native's commit of
    cached mains during tracegen (#46). ``stacked_commit`` is a pure hash of
    the trace and never touches the transcript, so placing it here is
    byte-identical.
    """
    by_air: list[tuple[StackedPcsData, ...]] = []
    for air in airs:
        by_air.append(
            tuple(
                stacked_commit(
                    sponge, compressor, l_skip, n_stack, log_blowup, k, [cm]
                )[1]
                for cm in air.cached_mains
            )
        )
    return CachedMainCommitments(
        by_air=tuple(by_air),
        stacking_order=tuple(cd for i in order for cd in by_air[i]),
    )


class SwirlProver(
    ProverStage[SystemClaim, SystemWitness, TrivialClaim, Proof, DuplexTranscript]
):
    """The SWIRL prover: the stacked PCS commit, then three reductions, then the
    opening.

    One definition of the wiring, so ``prove``, the byte-match runnable and the
    benchmark cannot drift on it. ``StackedCommitData`` is held here between the
    PCS halves because it belongs to neither claim.
    """

    def __init__(
        self,
        sponge: Sponge,
        compressor: Compression,
        params: SystemParams,
        vk_pre_hash: Sequence[int],
        *,
        needs_next: Sequence[bool],
        jit: bool = True,
        obs_log: dict | None = None,
    ) -> None:
        self._vk_pre_hash = vk_pre_hash
        # Reference observation-log prefix (only the verify_prove debug runner
        # supplies it). When set, the prelude is diffed element-by-element
        # against it; prove()/the benchmark leave it ``None`` (issue #59).
        self._obs_log = obs_log
        self.pcs = StackedWhirPcs(
            sponge,
            compressor,
            l_skip=params.l_skip,
            n_stack=params.n_stack,
            log_blowup=params.log_blowup,
            whir=params.whir,
            jit=jit,
        )
        self.gkr = LogupGkrProver(
            l_skip=params.l_skip, logup_pow_bits=params.logup_pow_bits
        )
        self.zerocheck = ZerocheckProver(
            l_skip=params.l_skip,
            max_constraint_degree=params.max_constraint_degree,
        )
        self.stacking = StackingProver(
            l_skip=params.l_skip, n_stack=params.n_stack, needs_next=needs_next
        )

    def prove(
        self,
        claim: SystemClaim,
        witness: SystemWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[TrivialClaim, Proof, DuplexTranscript]:
        commitment, commit_data = self.pcs.commit(witness)
        transcript = bind_commitment(
            transcript,
            claim,
            commitment,
            witness.cached,
            vk_pre_hash=self._vk_pre_hash,
            obs_log=self._obs_log,
        )
        gkr = self.gkr.prove(claim, witness, transcript)
        zerocheck = self.zerocheck.prove(gkr.reduced_claim, witness, gkr.transcript)
        stacking = self.stacking.prove(
            zerocheck.reduced_claim, commit_data, zerocheck.transcript
        )
        opening = self.pcs.prove(
            stacking.reduced_claim, commit_data, stacking.transcript
        )
        return ProveResult(
            TrivialClaim(),
            Proof(
                common_main_commit=commitment,
                logup_pow_witness=gkr.reduction_proof.logup_pow_witness,
                gkr_proof=gkr.reduction_proof.gkr_proof,
                xi=gkr.reduction_proof.xi,
                batch_constraint_proof=zerocheck.reduction_proof,
                stacking_proof=stacking.reduction_proof,
                whir_proof=opening.reduction_proof,
            ),
            opening.transcript,
        )


def build_prover(
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    airs: Sequence[AirInstance],
    *,
    jit: bool = True,
    obs_log: dict | None = None,
) -> tuple[SwirlProver, SystemClaim, SystemWitness]:
    """Build the SWIRL prover role with the claim and witness it acts on.

    One definition of the wiring so ``prove``, the byte-match runnable and the
    benchmark cannot drift on it. The cached-main committer runs here, in build
    scope, because it precedes the first claim.
    """
    claim = system_claim(params.l_skip, airs)
    order = claim.shape.order
    sorted_airs = tuple(airs[i] for i in order)
    cached = commit_cached_mains(
        sponge,
        compressor,
        l_skip=params.l_skip,
        n_stack=params.n_stack,
        log_blowup=params.log_blowup,
        k=params.whir.k,
        airs=airs,
        order=order,
    )
    prover = SwirlProver(
        sponge,
        compressor,
        params,
        vk_pre_hash,
        needs_next=[a.needs_next for a in sorted_airs],
        jit=jit,
        obs_log=obs_log,
    )
    return prover, claim, SystemWitness(sorted_airs=sorted_airs, cached=cached)


def prove(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    airs: Sequence[AirInstance],
) -> tuple[DuplexTranscript, Proof]:
    """Prove the multi-AIR system end-to-end from a fresh transcript."""
    prover, claim, witness = build_prover(sponge, compressor, params, vk_pre_hash, airs)
    result = prover.prove(claim, witness, transcript)
    return result.transcript, result.reduction_proof
