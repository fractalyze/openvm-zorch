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

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import Array
from zk_dtypes import babybear_mont as F

from openvm_zorch.commit.trace_commit import stacked_commit
from openvm_zorch.logup_gkr.input_layer import InteractionSpec, gkr_input_evals
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
from openvm_zorch.transcript import grind, sample_ext
from openvm_zorch.whir.prover import WhirConfig, WhirProof, prove_whir_opening
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class AirInstance:
    """One AIR with its trace, in input (verifying-key) order."""

    trace: Array  # (height, width) base field
    dag: ConstraintsDag
    interactions: tuple[InteractionSpec, ...]
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


def prove(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    params: SystemParams,
    vk_pre_hash: Sequence[int],
    airs: Sequence[AirInstance],
) -> tuple[DuplexTranscript, Proof]:
    """Prove the multi-AIR system end-to-end from a fresh transcript."""
    l_skip = params.l_skip
    order = sorted(
        range(len(airs)), key=lambda i: (-airs[i].trace.shape[0], i)
    )
    sorted_airs = [airs[i] for i in order]
    sorted_traces = [a.trace for a in sorted_airs]

    # --- Protocol-derived sizes (Coordinator::prove / calculate_n_logup) ---
    log_heights = [log2_strict_usize(a.trace.shape[0]) for a in sorted_airs]
    total_interactions = sum(
        len(a.interactions) << max(lh, l_skip)
        for a, lh in zip(sorted_airs, log_heights)
    )
    n_logup = (
        total_interactions.bit_length() - l_skip if total_interactions else 0
    )
    n_max = max(max(lh - l_skip, 0) for lh in log_heights)
    n_global = max(n_max, n_logup)

    # --- Stage 1: stacked PCS commitment ---
    root, pcs_data = stacked_commit(
        sponge,
        compressor,
        l_skip,
        params.n_stack,
        params.log_blowup,
        params.whir.k,
        sorted_traces,
    )

    # --- Prelude (per AIR in input order) ---
    transcript = transcript.observe(jnp.array(list(vk_pre_hash), dtype=F))
    transcript = transcript.observe(root)
    for air in airs:
        meta: list[int] = [] if air.is_required else [1]
        meta.append(log2_strict_usize(air.trace.shape[0]))
        meta.extend(air.public_values)
        transcript = transcript.observe(jnp.array(meta, dtype=F))

    # --- Stage 2: LogUp-GKR ---
    transcript, logup_pow_witness = grind(transcript, params.logup_pow_bits)
    transcript, alpha = sample_ext(transcript)
    transcript, beta = sample_ext(transcript)
    num, den = gkr_input_evals(
        l_skip,
        n_logup,
        sorted_traces,
        [list(a.interactions) for a in sorted_airs],
        alpha,
        beta,
    )
    transcript, gkr_proof, xi = fractional_sumcheck(transcript, num, den)
    transcript, xi = pad_xi(transcript, xi, l_skip + n_global)

    # --- Stage 3: batched ZeroCheck + LogUp sumcheck ---
    transcript, bcp = prove_batch_constraints(
        transcript,
        l_skip,
        n_logup,
        [
            AirData(
                trace=a.trace,
                dag=a.dag,
                public_values=a.public_values,
                constraint_degree=a.constraint_degree,
                needs_next=a.needs_next,
            )
            for a in sorted_airs
        ],
        xi,
        beta,
        params.max_constraint_degree,
    )

    # --- Stage 4: stacked opening reduction ---
    needs_next = [a.needs_next for a in sorted_airs]
    transcript, stacking_proof = prove_stacked_opening_reduction(
        transcript,
        l_skip,
        params.n_stack,
        [(pcs_data.matrix, pcs_data.layout)],
        [needs_next],
        bcp.r,
    )

    # --- Stage 5: WHIR opening at u_cube ---
    u_0 = stacking_proof.u[0]
    u_cube = [u_0]
    for _ in range(l_skip - 1):
        u_cube.append(u_cube[-1] * u_cube[-1])
    u_cube.extend(stacking_proof.u[1:])
    transcript, whir_proof = prove_whir_opening(
        transcript,
        sponge,
        compressor,
        l_skip,
        params.log_blowup,
        params.whir,
        [(pcs_data.matrix, pcs_data.tree)],
        u_cube,
        # Lower each Stage-5 device island to one fused kernel (byte-identical —
        # whir_test gates both paths). The strided merkle_commit marker only
        # fuses under jit; eager dispatch decomposes it, so this flip is what
        # turns fuse=True into an actual compute win.
        jit=True,
    )

    return transcript, Proof(
        common_main_commit=root,
        logup_pow_witness=logup_pow_witness,
        gkr_proof=gkr_proof,
        xi=xi,
        batch_constraint_proof=bcp,
        stacking_proof=stacking_proof,
        whir_proof=whir_proof,
    )
