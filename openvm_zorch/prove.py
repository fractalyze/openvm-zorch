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

from dataclasses import dataclass
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
from openvm_zorch.logup_gkr.input_layer import gkr_input_evals
from openvm_zorch.logup_gkr.prover import (
    FracSumcheckProof,
    empty_frac_sumcheck_proof,
    fractional_sumcheck,
    pad_xi,
)
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.logup_zerocheck.prover import (
    AirData,
    BatchConstraintProof,
    prove_batch_constraints,
)
from openvm_zorch.stacked_reduction.prover import (
    StackingProof,
    prove_stacked_opening_reduction,
)
from openvm_zorch.transcript import grind, sample_ext
from openvm_zorch.whir.prover import WhirConfig, WhirProof, prove_whir_opening


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
class SystemClaim:
    """Every present AIR's trace satisfies its constraints, and the LogUp
    interactions across all of them balance.

    The root statement a SWIRL proof discharges. It carries only what varies
    per block — which vk AIRs are present, how tall each is, what public values
    it exposes — while the circuit-fixed properties (constraint DAGs, column
    counts, security parameters) configure the roles.
    """

    air_shapes: tuple[AirShape, ...]  # input (verifying-key) order
    shape: SystemShape


@dataclass(frozen=True, kw_only=True)
class LogupClaim:
    """The system's LogUp fraction sums collapse, under the batching challenges
    α and β, to a single fraction-sum claim at the point ξ.

    What LogUp-GKR leaves for zerocheck: the interaction argument has come down
    to one claim about the input layer, which zerocheck ties back to the traces'
    own columns. The claimed numerator and denominator are not carried — the
    verifier re-derives them from the reduction proof while checking it, and the
    prover recomputes the input layer from the witness, so neither role reads
    them off a shared field.
    """

    system: SystemClaim
    alpha: Array
    beta: Array
    xi: list[Array]


@dataclass(frozen=True, kw_only=True)
class ColumnOpeningClaim:
    """Each AIR's columns open to ``column_openings`` at the zerocheck point
    ``r``.

    What zerocheck leaves for the stacked reduction: the constraint and
    interaction claims have been discharged down to per-column evaluation
    claims, which are statements about committed data rather than about the
    AIRs' algebra.
    """

    system: SystemClaim
    r: list[Array]
    column_openings: Sequence[Sequence[Array]]


@dataclass(frozen=True, kw_only=True)
class StackedOpeningClaim:
    """The committed stacked matrix's columns open to ``stacking_openings`` at
    the point ``u``.

    What the stacked reduction leaves for the PCS: one opening claim per
    stacked column against ``commitment``, which the PCS discharges into the
    trivial claim.
    """

    commitment: Array
    u: list[Array]
    stacking_openings: Sequence[Sequence[Array]]


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
class SystemWitness:
    """The traces the system claim is about, in stacking order, with the
    cached-main commitments already taken."""

    sorted_airs: tuple[AirInstance, ...]
    cached: CachedMainCommitments


@dataclass(frozen=True, kw_only=True)
class StackedCommitData:
    """What the stacked PCS retains between its commit and open halves.

    Not prover-only: the common-main root reaches the wire verbatim as the
    proof's commitment, and each Merkle tree's opened rows and paths ride the
    WHIR proof.
    """

    common: StackedPcsData
    cached: CachedMainCommitments


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


@dataclass(frozen=True)
class GkrStageMsg:
    """The LogUp-GKR reduction proof: the grind witness, the fractional-sumcheck
    proof, and the padded evaluation point. ``xi`` is on the reduced claim too —
    zerocheck reads it there — but it is a proof field as well, so it rides the
    reduction proof for assembly."""

    logup_pow_witness: Array
    gkr_proof: FracSumcheckProof
    xi: list[Array]


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


class StackedWhirPcs:
    """The stacked polynomial commitment scheme: commit the traces, open them
    at a point with WHIR.

    Two halves of one role, held apart by Fiat-Shamir — the commitment must
    bind the transcript before LogUp-GKR draws a challenge, and the opening
    needs the point the stacked reduction produces. ``StackedCommitData`` names
    what crosses between them.

    Reduces a stacked opening claim to the trivial claim: WHIR is terminal, so
    the whole SWIRL proof is a complete argument rather than one link in a
    chain.
    """

    def __init__(
        self,
        sponge: Sponge,
        compressor: Compression,
        *,
        l_skip: int,
        n_stack: int,
        log_blowup: int,
        whir: WhirConfig,
        jit: bool = True,
    ) -> None:
        self._sponge = sponge
        self._compressor = compressor
        self._l_skip = l_skip
        self._n_stack = n_stack
        self._log_blowup = log_blowup
        self._whir = whir
        self._jit = jit

    def commit(self, witness: SystemWitness) -> tuple[Array, StackedCommitData]:
        """Commit the common main. The cached mains were committed earlier, in
        build scope, and ride the witness."""
        root, pcs_data = stacked_commit(
            self._sponge,
            self._compressor,
            self._l_skip,
            self._n_stack,
            self._log_blowup,
            self._whir.k,
            [a.trace for a in witness.sorted_airs],
        )
        return root, StackedCommitData(common=pcs_data, cached=witness.cached)

    def prove(
        self,
        claim: StackedOpeningClaim,
        witness: StackedCommitData,
        transcript: DuplexTranscript,
    ) -> ProveResult[TrivialClaim, WhirProof]:
        """Open the committed matrices at ``u_cube`` — the stacking → WHIR
        handoff ``u_cube = (u₀ squarings over the skip domain) ‖ u[1..]``
        (reference ``prove_openings``)."""
        u_cube = [claim.u[0]]
        for _ in range(self._l_skip - 1):
            u_cube.append(u_cube[-1] * u_cube[-1])
        u_cube.extend(claim.u[1:])
        transcript, whir_proof = prove_whir_opening(
            transcript,
            self._sponge,
            self._compressor,
            self._l_skip,
            self._log_blowup,
            self._whir,
            # Common main first, then each cached/preprocessed commitment (the
            # WHIR μ-batch spans all their columns; round 0 opens each tree). An
            # empty cached prefix (synthetic) leaves the single-commitment path.
            [(witness.common.matrix, witness.common.tree)]
            + [(d.matrix, d.tree) for d in witness.cached.stacking_order],
            u_cube,
            # Lower each WHIR device island to one fused kernel (byte-identical
            # — whir prover_test gates both paths). The strided merkle_commit
            # marker only fuses under jit; eager dispatch decomposes it, so this
            # flip is what turns the strided merkle_commit fusion into an actual
            # compute win.
            jit=self._jit,
        )
        return ProveResult(TrivialClaim(), whir_proof, transcript)


class LogupGkrProver(ProverStage[SystemClaim, SystemWitness, LogupClaim, GkrStageMsg]):
    """Reduce the system's interaction claim to one fraction-sum claim at ξ.

    Grinds the LogUp PoW, samples α/β, builds the GKR input layer from the
    traces, and runs the fractional sumcheck.
    """

    def __init__(self, *, l_skip: int, logup_pow_bits: int) -> None:
        self._l_skip = l_skip
        self._logup_pow_bits = logup_pow_bits

    def prove(
        self,
        claim: SystemClaim,
        witness: SystemWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[LogupClaim, GkrStageMsg]:
        shape = claim.shape
        airs = witness.sorted_airs
        transcript, logup_pow_witness = grind(transcript, self._logup_pow_bits)
        transcript, alpha = sample_ext(transcript)
        transcript, beta = sample_ext(transcript)
        if shape.total_interactions > 0:
            num, den = gkr_input_evals(
                self._l_skip,
                shape.n_logup,
                [a.trace for a in airs],
                [a.dag for a in airs],
                [a.public_values for a in airs],
                [a.needs_next for a in airs],
                [a.cached_mains for a in airs],
                alpha,
                beta,
            )
            transcript, gkr_proof, xi = fractional_sumcheck(transcript, num, den)
        else:
            # No interactions: the reference builds an empty GKR input layer, so
            # ``fractional_sumcheck`` is a no-op — it neither absorbs into the
            # transcript nor produces any ξ. The verifier gates ``verify_gkr`` on
            # the same ``total_interactions == 0`` (verifier/batch_constraints.rs),
            # so skipping it here keeps both sides' transcript and ξ in lockstep.
            gkr_proof = empty_frac_sumcheck_proof(alpha.dtype)
            xi = []
        transcript, xi = pad_xi(transcript, xi, self._l_skip + shape.n_global)
        return ProveResult(
            LogupClaim(system=claim, alpha=alpha, beta=beta, xi=xi),
            GkrStageMsg(logup_pow_witness, gkr_proof, xi),
            transcript,
        )


class ZerocheckProver(
    ProverStage[LogupClaim, SystemWitness, ColumnOpeningClaim, BatchConstraintProof]
):
    """Reduce the constraint and LogUp claims to per-column opening claims at
    ``r``.

    The batched ZeroCheck + LogUp sumcheck over ``prove_batch_constraints``,
    consuming ξ and β off the incoming claim.
    """

    def __init__(self, *, l_skip: int, max_constraint_degree: int) -> None:
        self._l_skip = l_skip
        self._max_constraint_degree = max_constraint_degree

    def prove(
        self,
        claim: LogupClaim,
        witness: SystemWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[ColumnOpeningClaim, BatchConstraintProof]:
        transcript, bcp = prove_batch_constraints(
            transcript,
            self._l_skip,
            claim.system.shape.n_logup,
            [
                AirData(
                    trace=a.trace,
                    dag=a.dag,
                    public_values=a.public_values,
                    constraint_degree=a.constraint_degree,
                    needs_next=a.needs_next,
                    cached_mains=a.cached_mains,
                )
                for a in witness.sorted_airs
            ],
            claim.xi,
            claim.beta,
            self._max_constraint_degree,
        )
        return ProveResult(
            ColumnOpeningClaim(
                system=claim.system,
                r=bcp.r,
                column_openings=bcp.column_openings,
            ),
            bcp,
            transcript,
        )


class StackingProver(
    ProverStage[
        ColumnOpeningClaim, StackedCommitData, StackedOpeningClaim, StackingProof
    ]
):
    """Reduce per-column opening claims at ``r`` to stacked-column opening
    claims at ``u``.

    Its witness is the PCS's retained commit data: the reduction runs over the
    committed matrices themselves, which belong to the scheme rather than to
    any claim.
    """

    def __init__(
        self, *, l_skip: int, n_stack: int, needs_next: Sequence[bool]
    ) -> None:
        self._l_skip = l_skip
        self._n_stack = n_stack
        # Rotation is a property of each AIR's constraints, fixed by the
        # circuit, so it configures the role rather than riding the claim.
        self._needs_next = list(needs_next)

    def prove(
        self,
        claim: ColumnOpeningClaim,
        witness: StackedCommitData,
        transcript: DuplexTranscript,
    ) -> ProveResult[StackedOpeningClaim, StackingProof]:
        # The commit half committed the common main plus each cached main as its
        # own stacked commitment; the opening reduction runs over all of them,
        # common main first (reference ``device.rs`` prove_openings:154-167).
        stacked_per_commit = [(witness.common.matrix, witness.common.layout)] + [
            (d.matrix, d.layout) for d in witness.cached.stacking_order
        ]
        # need_rot for a cached commit is the owning AIR's need_rot — its cached
        # columns share the AIR's rotation claim. An empty cached prefix (the
        # synthetic fixture) leaves this exactly the single-commit call (#59).
        need_rot_per_commit: list[list[bool]] = [list(self._needs_next)]
        for position, air_idx in enumerate(claim.system.shape.order):
            need_rot_per_commit.extend(
                [self._needs_next[position]] for _ in witness.cached.by_air[air_idx]
            )
        transcript, stacking_proof = prove_stacked_opening_reduction(
            transcript,
            self._l_skip,
            self._n_stack,
            stacked_per_commit,
            need_rot_per_commit,
            claim.r,
        )
        return ProveResult(
            StackedOpeningClaim(
                commitment=witness.common.commit,
                u=stacking_proof.u,
                stacking_openings=stacking_proof.stacking_openings,
            ),
            stacking_proof,
            transcript,
        )


class SwirlProver(ProverStage[SystemClaim, SystemWitness, TrivialClaim, Proof]):
    """The SWIRL prover: the stacked PCS commit, then three reductions, then the
    opening.

    A composite role, so the wiring has one definition and ``prove``, the
    byte-match runnable and the benchmark cannot drift on it. LogUp-GKR,
    zerocheck and the stacked reduction each reduce the previous claim; the PCS
    brackets them, binding the traces up front and discharging the final
    opening claim at the end, with ``StackedCommitData`` held here in between
    because it belongs to neither claim.

    Reduces to the trivial claim: the WHIR opening is terminal, so a SWIRL
    proof is a complete argument rather than one link in a chain.
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
    ) -> ProveResult[TrivialClaim, Proof]:
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
