"""End-to-end SWIRL prover — the five stages composed into one ``prove``.

Mirrors the reference ``Coordinator::prove``: each stage module is driven by
the previous stage's Python output, threading one Fiat-Shamir transcript from
the prelude to the final WHIR query — no recorded log anywhere. The
protocol-derived sizes the coordinator owns are computed here:

- stacking order: descending trace height, ties by input AIR index;
- ``n_logup = bit_length(Σ_T num_interactions·2^{lifted log height}) − l_skip``
  (``calculate_n_logup``), ``n_max = max(log height − l_skip, 0)``,
  ``n_global = max(n_max, n_logup)``;
- the prelude observes: vk pre-hash, the Stage-1 commitment, then per AIR in
  *input* order a present flag (only when the AIR is optional), its log
  height, and its public values (no preprocessed/cached commits in scope).

The proof-of-work grinds (LogUp and WHIR's three kinds) run natively.

Reference: `Coordinator::prove` (prover/coordinator.rs) and
`prove_openings` (prover/cpu_backend.rs) for the Stage-4 → Stage-5 handoff
``u_cube = (u₀ squarings over the skip domain) ‖ u[1..]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import jax.numpy as jnp
from jax import Array
from zk_dtypes import babybear_mont as F

from openvm_zorch.commit.trace_commit import stacked_commit
from openvm_zorch.logup_gkr.input_layer import gkr_input_evals
from openvm_zorch.logup_gkr.prover import (
    FracSumcheckProof,
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
from openvm_zorch.commit.trace_commit import StackedPcsData
from openvm_zorch.transcript import grind, sample_ext
from openvm_zorch.whir.prover import WhirConfig, WhirProof, prove_whir_opening
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.round import ProveChain, Round
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize


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


@dataclass(frozen=True)
class Proof:
    """The five stage proofs plus the Stage-1 commitment."""

    common_main_commit: Array  # (8,) F
    logup_pow_witness: Array
    gkr_proof: FracSumcheckProof
    xi: list[Array]  # padded to l_skip + n_global
    batch_constraint_proof: BatchConstraintProof
    stacking_proof: StackingProof
    whir_proof: WhirProof


@dataclass(frozen=True)
class GkrStageMsg:
    """What the LogUp-GKR stage contributes to the final ``Proof``: the grind
    witness, the fractional-sumcheck proof, and the padded evaluation point.
    ``xi`` is also threaded on the carry — ZeroCheck reads it — but it is a
    proof field too, so it rides the message for assembly."""

    logup_pow_witness: Array
    gkr_proof: FracSumcheckProof
    xi: list[Array]


@dataclass(frozen=True)
class ProveCarry:
    """What flows between the prover's stage Rounds: the witness (set at
    construction) plus each stage's outputs the next stage reads. Stage Rounds
    return it via ``replace`` — a stage writes its own fields and passes the
    rest through untouched.

    Unlike sp1-zorch's ``ShardCarry`` this is a plain dataclass, not a
    registered pytree: openvm's stages jit *internally* (the commit tail,
    Stage 4, Stage 5), so the carry is host-side Python that never crosses a
    ``jax.jit`` boundary and need not flatten. ``pcs_data`` (a ``StackedLayout``
    + Merkle tree) isn't a pytree for the same reason it never needs to be one.
    """

    # Witness, in input (verifying-key) order and stacking order. The prelude
    # observes per AIR in input order; the stages consume stacking order.
    airs: Sequence[AirInstance]
    sorted_airs: Sequence[AirInstance]
    # Stage 1 (commit) outputs; ``pcs_data`` read by Stage 4 + Stage 5.
    root: Array | None = None
    pcs_data: StackedPcsData | None = None
    # The preprocessed/cached commitments — each its own stacked commit — in
    # stacking order, read by Stage 4 + Stage 5 alongside ``pcs_data`` (common
    # main first). Empty unless a real block carries cached mains (issue #59).
    pre_cached_pcs_data: Sequence[StackedPcsData] = ()
    # Per-AIR cached commitments (keyed by ``id(air)``), precomputed in
    # ``prove_chain`` so CommitRound only *observes* them in the prelude — native
    # commits cached mains in tracegen, outside the timed prove (#46). Empty
    # unless a real block carries cached mains.
    cached_pcs_data_by_air: dict[int, list[StackedPcsData]] = field(
        default_factory=dict
    )
    # Stage 2 (GKR) outputs; ``beta`` + ``xi`` read by Stage 3.
    beta: Array | None = None
    xi: list[Array] | None = None
    # Stage 3 (ZeroCheck) output; the sumcheck point read by Stage 4.
    bcp_r: Array | None = None
    # Stage 4 (stacking) output; the opening point read by Stage 5.
    stacking_u: list[Array] | None = None


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
        got.extend(int(v) for v in jnp.atleast_1d(a).astype(jnp.uint32))
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


def _commit_cached_mains(
    sponge: Sponge,
    compressor: Compression,
    *,
    l_skip: int,
    n_stack: int,
    log_blowup: int,
    k: int,
    sorted_airs: Sequence[AirInstance],
) -> tuple[dict[int, list[StackedPcsData]], list[StackedPcsData]]:
    """Commit each cached main as its own stacked commitment (reference
    cpu_backend.rs ``pre_cached_pcs_data_per_commit`` — one PcsData per
    cached/preprocessed trace). Returns the per-AIR map (keyed ``id(air)``, for
    the input-order prelude observe) and the flat list in stacking order (read by
    Stage 4/5).

    Hoisted out of ``CommitRound`` into ``prove_chain`` so it lands in
    build/setup scope, not the timed prove: the native prover commits cached
    mains during tracegen and the prove span only *observes* the precomputed
    ``cd.commit`` (#46). ``stacked_commit`` is a pure hash of the trace — it never
    touches the Fiat-Shamir transcript — so hoisting it is byte-identical."""
    cached_by_air: dict[int, list[StackedPcsData]] = {}
    pre_cached: list[StackedPcsData] = []
    for a in sorted_airs:
        if a.cached_mains:
            cds = [
                stacked_commit(
                    sponge, compressor, l_skip, n_stack, log_blowup, k, [cm]
                )[1]
                for cm in a.cached_mains
            ]
            cached_by_air[id(a)] = cds
            pre_cached.extend(cds)
    return cached_by_air, pre_cached


class CommitRound(Round):
    """Stage 1 + prelude: commit the stacked PCS, then absorb the prelude
    stream (vk pre-hash, the commitment, then per AIR in *input* order an
    optional present flag, log height, and public values). The prelude schedule
    lives here once — folded into the commit Round as sp1-zorch folds its
    ``PreambleRound`` into ``TraceCommitRound`` — so an ordering edit cannot
    land in the prover's Fiat-Shamir stream without the byte-match seeing it.
    The message is the structure-bound commitment."""

    def __init__(
        self,
        sponge: Sponge,
        compressor: Compression,
        *,
        l_skip: int,
        n_stack: int,
        log_blowup: int,
        k: int,
        vk_pre_hash: Sequence[int],
        obs_log: dict | None = None,
    ) -> None:
        self._sponge = sponge
        self._compressor = compressor
        self._l_skip = l_skip
        self._n_stack = n_stack
        self._log_blowup = log_blowup
        self._k = k
        self._vk_pre_hash = vk_pre_hash
        # Reference observation-log prefix (only the verify_prove debug runner
        # supplies it). When set, the prelude is diffed element-by-element
        # against it; prove()/the benchmark leave it ``None`` (issue #59).
        self._obs_log = obs_log

    def __call__(
        self, carry: ProveCarry, transcript: DuplexTranscript
    ) -> tuple[ProveCarry, DuplexTranscript, Array]:
        root, pcs_data = stacked_commit(
            self._sponge,
            self._compressor,
            self._l_skip,
            self._n_stack,
            self._log_blowup,
            self._k,
            [a.trace for a in carry.sorted_airs],
        )

        # Cached-main commitments are precomputed in ``prove_chain`` (build
        # scope, like native's tracegen) and ride the carry; the prelude below
        # only observes them. See ``_commit_cached_mains`` (#46).
        cached_by_air = carry.cached_pcs_data_by_air

        # --- Prelude (per AIR in verifying-key order) ---
        # Reference prover/mod.rs:155-175: the common-main commit, then iterate
        # ALL vk AIRs in order. Each non-required AIR observes a present flag
        # (1 present / 0 absent); each PRESENT AIR then observes the log height
        # (or a preprocessed commit — none here), each cached-main commit root,
        # and its public values. On a real block the present AIRs are a sparse
        # subset of the pk (unexercised chips are absent) — those gaps still
        # contribute a present=0 flag. The synthetic fixture is contiguous +
        # all-present (``air_idx`` ``None``), so it sees no gaps and no cached
        # roots — its stream stays byte-identical (#59).
        # Build the ordered observation list first, then absorb it. Keeping the
        # sequence explicit lets the verify_prove debug runner diff it against
        # the reference observation-log element-by-element (issue #59); the
        # absorb order is unchanged, so the Fiat-Shamir stream is byte-identical.
        obs: list[Array] = [
            jnp.array(list(self._vk_pre_hash), dtype=F),
            root,
        ]
        prev = -1
        for air in carry.airs:
            idx = air.air_idx if air.air_idx is not None else prev + 1
            # Absent (unexercised) AIRs between the last present AIR and this one
            # are non-required (a required AIR is always present), so each
            # observes a present=0 flag and nothing else.
            for _absent in range(prev + 1, idx):
                obs.append(jnp.array([0], dtype=F))
            prev = idx
            head: list[int] = [] if air.is_required else [1]
            head.append(log2_strict_usize(air.trace.shape[0]))
            cached = cached_by_air.get(id(air), [])
            if not cached:
                head.extend(air.public_values)
                obs.append(jnp.array(head, dtype=F))
                continue
            obs.append(jnp.array(head, dtype=F))
            for cd in cached:
                obs.append(cd.commit)
            if air.public_values:
                obs.append(jnp.array(list(air.public_values), dtype=F))

        if self._obs_log is not None:
            _log_prelude_obs_diff(obs, self._obs_log)
        # One fused absorb over the whole prelude stream instead of len(obs)
        # separate observe() dispatches. ``observe`` flattens to the base field
        # and absorbs via one ``lax.scan`` with no per-call padding (padding only
        # happens at ``sample``, and the prelude never samples between observes),
        # so absorbing the concatenation is byte-identical to the per-element
        # loop while collapsing ~86 host-dispatched kernels into one.
        transcript = transcript.observe(jnp.concatenate([o.reshape(-1) for o in obs]))

        carry = replace(carry, root=root, pcs_data=pcs_data)
        return carry, transcript, root


class GkrRound(Round):
    """Stage 2: LogUp-GKR. Grinds the LogUp PoW, samples α/β, builds the GKR
    input layer, and runs the fractional sumcheck. Writes β + the padded point
    ξ onto the carry for ZeroCheck."""

    def __init__(
        self, *, l_skip: int, n_logup: int, n_global: int, logup_pow_bits: int
    ) -> None:
        self._l_skip = l_skip
        self._n_logup = n_logup
        self._n_global = n_global
        self._logup_pow_bits = logup_pow_bits

    def __call__(
        self, carry: ProveCarry, transcript: DuplexTranscript
    ) -> tuple[ProveCarry, DuplexTranscript, GkrStageMsg]:
        transcript, logup_pow_witness = grind(transcript, self._logup_pow_bits)
        transcript, alpha = sample_ext(transcript)
        transcript, beta = sample_ext(transcript)
        num, den = gkr_input_evals(
            self._l_skip,
            self._n_logup,
            [a.trace for a in carry.sorted_airs],
            [a.dag for a in carry.sorted_airs],
            [a.public_values for a in carry.sorted_airs],
            [a.needs_next for a in carry.sorted_airs],
            [a.cached_mains for a in carry.sorted_airs],
            alpha,
            beta,
        )
        transcript, gkr_proof, xi = fractional_sumcheck(transcript, num, den)
        transcript, xi = pad_xi(transcript, xi, self._l_skip + self._n_global)
        carry = replace(carry, beta=beta, xi=xi)
        return carry, transcript, GkrStageMsg(logup_pow_witness, gkr_proof, xi)


class ZeroCheckRound(Round):
    """Stage 3: batched ZeroCheck + LogUp sumcheck over
    ``prove_batch_constraints``, consuming ξ and β off the carry. Writes the
    sumcheck point ``r`` for the stacking stage."""

    def __init__(
        self, *, l_skip: int, n_logup: int, max_constraint_degree: int
    ) -> None:
        self._l_skip = l_skip
        self._n_logup = n_logup
        self._max_constraint_degree = max_constraint_degree

    def __call__(
        self, carry: ProveCarry, transcript: DuplexTranscript
    ) -> tuple[ProveCarry, DuplexTranscript, BatchConstraintProof]:
        transcript, bcp = prove_batch_constraints(
            transcript,
            self._l_skip,
            self._n_logup,
            [
                AirData(
                    trace=a.trace,
                    dag=a.dag,
                    public_values=a.public_values,
                    constraint_degree=a.constraint_degree,
                    needs_next=a.needs_next,
                    cached_mains=a.cached_mains,
                )
                for a in carry.sorted_airs
            ],
            carry.xi,
            carry.beta,
            self._max_constraint_degree,
        )
        carry = replace(carry, bcp_r=bcp.r)
        return carry, transcript, bcp


class StackingRound(Round):
    """Stage 4: stacked opening reduction, consuming the committed matrix/layout
    and the ZeroCheck point off the carry. Writes the opening point ``u`` for
    WHIR."""

    def __init__(self, *, l_skip: int, n_stack: int) -> None:
        self._l_skip = l_skip
        self._n_stack = n_stack

    def __call__(
        self, carry: ProveCarry, transcript: DuplexTranscript
    ) -> tuple[ProveCarry, DuplexTranscript, StackingProof]:
        needs_next = [a.needs_next for a in carry.sorted_airs]
        # Stage 1 committed the common main plus each cached main as its own
        # stacked commitment; the opening reduction runs over all of them, common
        # main first (reference ``device.rs`` prove_openings:154-167). need_rot for
        # a cached commit is the owning AIR's need_rot -- its cached columns share
        # the AIR's rotation claim. An empty cached prefix (the synthetic fixture)
        # leaves this exactly the single-commit call (issue #59).
        stacked_per_commit = [(carry.pcs_data.matrix, carry.pcs_data.layout)] + [
            (d.matrix, d.layout) for d in carry.pre_cached_pcs_data
        ]
        need_rot_per_commit = [needs_next] + [
            [a.needs_next] for a in carry.sorted_airs for _ in a.cached_mains
        ]
        transcript, stacking_proof = prove_stacked_opening_reduction(
            transcript,
            self._l_skip,
            self._n_stack,
            stacked_per_commit,
            need_rot_per_commit,
            carry.bcp_r,
        )
        carry = replace(carry, stacking_u=stacking_proof.u)
        return carry, transcript, stacking_proof


class WhirRound(Round):
    """Stage 5: WHIR opening at ``u_cube``, the Stage-4 → Stage-5 handoff
    ``u_cube = (u₀ squarings over the skip domain) ‖ u[1..]``
    (reference ``prove_openings``). Reads the committed matrix/tree and the
    opening point off the carry."""

    def __init__(
        self,
        sponge: Sponge,
        compressor: Compression,
        *,
        l_skip: int,
        log_blowup: int,
        whir: WhirConfig,
        jit: bool = True,
    ) -> None:
        self._sponge = sponge
        self._compressor = compressor
        self._l_skip = l_skip
        self._log_blowup = log_blowup
        self._whir = whir
        self._jit = jit

    def __call__(
        self, carry: ProveCarry, transcript: DuplexTranscript
    ) -> tuple[ProveCarry, DuplexTranscript, WhirProof]:
        u_0 = carry.stacking_u[0]
        u_cube = [u_0]
        for _ in range(self._l_skip - 1):
            u_cube.append(u_cube[-1] * u_cube[-1])
        u_cube.extend(carry.stacking_u[1:])
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
            [(carry.pcs_data.matrix, carry.pcs_data.tree)]
            + [(d.matrix, d.tree) for d in carry.pre_cached_pcs_data],
            u_cube,
            # Lower each Stage-5 device island to one fused kernel (byte-identical
            # — whir_test gates both paths). The strided merkle_commit marker
            # only fuses under jit; eager dispatch decomposes it, so this flip is
            # what turns fuse=True into an actual compute win.
            jit=self._jit,
        )
        return carry, transcript, whir_proof


def prove_chain(
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    airs: Sequence[AirInstance],
    *,
    jit: bool = True,
    obs_log: dict | None = None,
) -> tuple[ProveChain, ProveCarry]:
    """Build the SWIRL prover as one ``ProveChain`` of stage Rounds plus its
    initial carry. One definition of the stage wiring so ``prove`` and the
    benchmark cannot drift on it (sp1-zorch's ``prove_shard_chain`` pattern).

    The protocol-derived sizes the reference ``Coordinator::prove`` owns
    (stacking order, ``n_logup`` / ``n_max`` / ``n_global``) are computed here
    and bound onto the Rounds. Returns the carry alongside the chain because the
    stacking order it derives is also the carry's witness — keeping the
    derivation in one place.
    """
    l_skip = params.l_skip
    order = sorted(range(len(airs)), key=lambda i: (-airs[i].trace.shape[0], i))
    sorted_airs = [airs[i] for i in order]

    # Commit cached mains here (build scope) rather than inside CommitRound,
    # matching native's tracegen-time cached commit — the timed prove's commit
    # stage then covers only the common main (#46). Byte-identical: a pure hash,
    # observed in the prelude exactly as before.
    cached_pcs_data_by_air, pre_cached_pcs_data = _commit_cached_mains(
        sponge,
        compressor,
        l_skip=l_skip,
        n_stack=params.n_stack,
        log_blowup=params.log_blowup,
        k=params.whir.k,
        sorted_airs=sorted_airs,
    )

    # --- Protocol-derived sizes (Coordinator::prove / calculate_n_logup) ---
    log_heights = [log2_strict_usize(a.trace.shape[0]) for a in sorted_airs]
    total_interactions = sum(
        len(a.dag.interactions) << max(lh, l_skip)
        for a, lh in zip(sorted_airs, log_heights)
    )
    n_logup = total_interactions.bit_length() - l_skip if total_interactions else 0
    n_max = max(max(lh - l_skip, 0) for lh in log_heights)
    n_global = max(n_max, n_logup)

    chain = ProveChain(
        [
            CommitRound(
                sponge,
                compressor,
                l_skip=l_skip,
                n_stack=params.n_stack,
                log_blowup=params.log_blowup,
                k=params.whir.k,
                vk_pre_hash=vk_pre_hash,
                obs_log=obs_log,
            ),
            GkrRound(
                l_skip=l_skip,
                n_logup=n_logup,
                n_global=n_global,
                logup_pow_bits=params.logup_pow_bits,
            ),
            ZeroCheckRound(
                l_skip=l_skip,
                n_logup=n_logup,
                max_constraint_degree=params.max_constraint_degree,
            ),
            StackingRound(l_skip=l_skip, n_stack=params.n_stack),
            WhirRound(
                sponge,
                compressor,
                l_skip=l_skip,
                log_blowup=params.log_blowup,
                whir=params.whir,
                jit=jit,
            ),
        ]
    )
    return chain, ProveCarry(
        airs=airs,
        sorted_airs=sorted_airs,
        pre_cached_pcs_data=pre_cached_pcs_data,
        cached_pcs_data_by_air=cached_pcs_data_by_air,
    )


def prove(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    airs: Sequence[AirInstance],
) -> tuple[DuplexTranscript, Proof]:
    """Prove the multi-AIR system end-to-end from a fresh transcript.

    A thin driver over ``prove_chain``: run the chain, then assemble the
    ``Proof`` from the per-stage message list (``[root, gkr, bcp, stacking,
    whir]``, in stage order)."""
    chain, carry = prove_chain(sponge, compressor, params, vk_pre_hash, airs)
    _, transcript, msgs = chain(carry, transcript)
    root, gkr, bcp, stacking_proof, whir_proof = msgs
    return transcript, Proof(
        common_main_commit=root,
        logup_pow_witness=gkr.logup_pow_witness,
        gkr_proof=gkr.gkr_proof,
        xi=gkr.xi,
        batch_constraint_proof=bcp,
        stacking_proof=stacking_proof,
        whir_proof=whir_proof,
    )
