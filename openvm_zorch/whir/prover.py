"""Stage 5 — WHIR opening proof (``prove_whir_opening``).

Opens the μ-batched committed columns as one MLE at ``u_cube`` (Stage 4's
output mapped onto the cube: ``u₀`` squarings over the skip variables, then
``u[1..]``). The weight starts as the Möbius-adjusted equality polynomial of
``u_cube`` — the eval-to-coeff RS encoding makes ``\\hat q(u) = Σ_b f̂(b)·
mobius_eq(u, b)`` — and each WHIR round folds ``k_whir`` sumcheck variables,
re-encodes the folded MLE as a fresh RS codeword (commit + out-of-domain
sample), then accumulates the in-domain query constraints into the weight
with γ powers.

Folding is ``k_whir`` plain 2-ary sumcheck folds of the hypercube evaluations
(adjacent pairs, LSB bound first) — the reference never folds the codeword
itself; the next round's codeword is a fresh DFT of the folded coefficients.
That is why this stage needs no k-ary fold primitive, in zorch or here.

The proof-of-work grinds run natively (zorch's lowest-witness search matches
the reference's serial scan); opened rows and Merkle paths are hints —
deterministic from index and root — and never enter the transcript.

Reference: openvm-stark-backend ``prove_whir_opening``
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/whir.rs#L78
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import Array, lax

from openvm_zorch.commit.rs_message import (
    eval_to_coeff_rs_message,
    mle_coeffs_to_evals,
    mle_evals_to_coeffs,
)
from openvm_zorch.commit.stacked_merkle import StackedMerkleTree, stacked_merkle_commit
from openvm_zorch.fields import EF, F, MODULUS, f_const, f_to_ef
from openvm_zorch.logup_zerocheck.prism import eq_cube_table, omega_int
from openvm_zorch.transcript import ef_from_limbs, grind, sample_bits, sample_ext
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.sumcheck.prover import fold_pair
from zorch.transcript import DuplexTranscript
from zorch.utils.bits import log2_strict_usize


@dataclass(frozen=True)
class WhirConfig:
    """The reference ``WhirConfig`` fields the prover consumes."""

    k: int
    num_queries: list[int]  # per WHIR round
    mu_pow_bits: int
    folding_pow_bits: int
    query_phase_pow_bits: int


@dataclass(frozen=True)
class WhirProof:
    """The reference ``WhirProof`` plus the sampled challenges."""

    mu_pow_witness: Array
    whir_sumcheck_polys: list[Array]  # num_whir_rounds·k × (2,) EF, evals at {1, 2}
    codeword_commits: list[Array]  # (num_whir_rounds − 1) × (8,) F
    ood_values: list[Array]  # (num_whir_rounds − 1) × () EF
    folding_pow_witnesses: list[Array]
    query_phase_pow_witnesses: list[Array]
    initial_round_opened_rows: list[list[Array]]  # per commit, per query: (2^k, W) F
    initial_round_merkle_proofs: list[list[Array]]  # per commit, per query: (depth, 8) F
    codeword_opened_values: list[list[Array]]  # per later round, per query: (2^k,) EF
    codeword_merkle_proofs: list[list[Array]]
    final_poly: Array  # (2^(m − num_whir_rounds·k),) EF
    mu: Array


def mobius_eq_table(u: Sequence[Array]) -> Array:
    """``mobius_eq_kernel(u, b)`` for ``b`` on the hypercube, LSB-first
    (index bit i ↔ ``u[i]``) — per-coordinate kernel ``K(0) = 1 − 2u_i``,
    ``K(1) = u_i`` (the reference's ``evals_mobius_eq_hypercube``)."""
    table = jnp.ones((1,), EF)
    one = jnp.ones((), EF)
    for u_i in u:
        table = jnp.concatenate([table * (one - u_i - u_i), table * u_i])
    return table


def _uni_eval(coeffs: Array, z: Array) -> Array:
    """Horner evaluation of a coefficient vector at ``z``. The reference's
    ``g_mle.eval_at_point(&z_0_vec)`` over the powers-of-two point
    ``(z, z², z⁴, …)`` is exactly the univariate evaluation at ``z``."""
    acc = jnp.zeros((), coeffs.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * z + coeffs[i]
    return acc


def _pow2_powers(z: Array, dim: int) -> list[Array]:
    """``[z, z², z⁴, …]`` of length ``dim`` (``exp_powers_of_2``)."""
    pows = [z]
    for _ in range(dim - 1):
        pows.append(pows[-1] * pows[-1])
    return pows


def prove_whir_opening(
    transcript: DuplexTranscript,
    sponge: Sponge,
    compressor: Compression,
    l_skip: int,
    log_blowup: int,
    config: WhirConfig,
    committed: Sequence[tuple[Array, StackedMerkleTree]],
    u_cube: Sequence[Array],
) -> tuple[DuplexTranscript, WhirProof]:
    """Drive Stage 5 from the transcript state at ``stage4_end``.

    ``committed`` holds, per commitment (common main first), the stacked
    evaluation matrix (base field, ``(2^m, W)``) and its Stage-1 tree (whose
    backing matrix is the RS codeword the queries open).
    """
    k_whir = config.k
    num_whir_rounds = len(config.num_queries)

    transcript, mu_pow_witness = grind(transcript, config.mu_pow_bits)
    transcript, mu = sample_ext(transcript)

    m = log2_strict_usize(committed[0][0].shape[0])
    # Hypercube evals of \hat{f}: per column, the RS message re-read as MLE
    # coefficients and zeta-transformed over all m bits, then μ-batched.
    f_evals = jnp.zeros((1 << m,), EF)
    mu_pow = jnp.ones((), EF)
    for matrix, _ in committed:
        messages = mle_coeffs_to_evals(eval_to_coeff_rs_message(l_skip, matrix.T))
        for col in range(messages.shape[0]):
            f_evals = f_evals + mu_pow * f_to_ef(messages[col])
            mu_pow = mu_pow * mu
    w_evals = mobius_eq_table(u_cube)

    whir_sumcheck_polys: list[Array] = []
    codeword_commits: list[Array] = []
    ood_values: list[Array] = []
    folding_pow_witnesses: list[Array] = []
    query_phase_pow_witnesses: list[Array] = []
    initial_round_opened_rows: list[list[Array]] = [[] for _ in committed]
    initial_round_merkle_proofs: list[list[Array]] = [[] for _ in committed]
    codeword_opened_values: list[list[Array]] = []
    codeword_merkle_proofs: list[list[Array]] = []
    rs_tree: StackedMerkleTree | None = None
    final_poly: Array | None = None
    log_rs_domain_size = m + log_blowup

    for whir_round, num_queries in enumerate(config.num_queries):
        is_last_round = whir_round == num_whir_rounds - 1

        # k_whir rounds of sumcheck on Σ_x f̂(x)·ŵ(x); s has degree 2, observed
        # as evaluations at {1, 2}.
        for _ in range(k_whir):
            f_0, f_1 = f_evals[0::2], f_evals[1::2]
            w_0, w_1 = w_evals[0::2], w_evals[1::2]
            s_1 = (f_1 * w_1).sum()
            s_2 = ((f_1 + f_1 - f_0) * (w_1 + w_1 - w_0)).sum()
            s_evals = jnp.stack([s_1, s_2])
            transcript = transcript.observe(s_evals)
            whir_sumcheck_polys.append(s_evals)

            transcript, witness = grind(transcript, config.folding_pow_bits)
            folding_pow_witnesses.append(witness)
            transcript, alpha = sample_ext(transcript)
            f_evals = fold_pair(f_0, f_1, alpha)
            w_evals = fold_pair(w_0, w_1, alpha)

        # ĝ = f̂(α⃗, ·): commit RS(ĝ) and answer one out-of-domain point — or,
        # in the last round, send ĝ's coefficients in the clear.
        g_coeffs = mle_evals_to_coeffs(f_evals)
        z_0 = None
        g_tree = None
        if not is_last_round:
            rs_len = 1 << (log_rs_domain_size - 1)
            padded = jnp.concatenate(
                [g_coeffs, jnp.zeros((rs_len - g_coeffs.shape[0],), EF)]
            )
            # lax.fft on extension dtypes accepts only 1-D input (the same
            # constraint zorch's basefold encode works around).
            g_rs = lax.fft(padded, "FFT", rs_len)
            g_tree = stacked_merkle_commit(
                sponge, compressor, lax.bitcast_convert_type(g_rs, F), 1 << k_whir
            )
            transcript = transcript.observe(g_tree.root)
            codeword_commits.append(g_tree.root)

            transcript, z_0 = sample_ext(transcript)
            g_opened_value = _uni_eval(g_coeffs, z_0)
            transcript = transcript.observe(g_opened_value)
            ood_values.append(g_opened_value)
        else:
            transcript = transcript.observe(g_coeffs)
            final_poly = g_coeffs

        # Query phase: grind, sample leaf indices, extract the opened rows and
        # Merkle paths (hints — not observed).
        transcript, witness = grind(transcript, config.query_phase_pow_bits)
        query_phase_pow_witnesses.append(witness)
        index_bits = log_rs_domain_size - k_whir
        indices = []
        for _ in range(num_queries):
            transcript, index = sample_bits(transcript, index_bits)
            indices.append(index)

        if not is_last_round:
            codeword_opened_values.append([])
            codeword_merkle_proofs.append([])
        omega = omega_int(index_bits)
        zs = [pow(omega, index, MODULUS) for index in indices]
        for index in indices:
            if whir_round == 0:
                for com_idx, (_, tree) in enumerate(committed):
                    initial_round_opened_rows[com_idx].append(tree.opened_rows(index))
                    initial_round_merkle_proofs[com_idx].append(
                        tree.query_merkle_proof(index)
                    )
            else:
                assert rs_tree is not None
                codeword_opened_values[whir_round - 1].append(
                    ef_from_limbs(rs_tree.opened_rows(index))
                )
                codeword_merkle_proofs[whir_round - 1].append(
                    rs_tree.query_merkle_proof(index)
                )
        rs_tree = g_tree

        # γ is sampled even in the last round (verifier symmetry); earlier
        # rounds fold the OOD and in-domain constraints into the weight:
        # ŵ += γ·eq(·, pow(z₀)) + Σ_i γ^{i+2}·eq(·, pow(z_i)).
        transcript, gamma = sample_ext(transcript)
        if not is_last_round:
            dim = log2_strict_usize(w_evals.shape[0])
            w_evals = w_evals + gamma * eq_cube_table(_pow2_powers(z_0, dim))
            gamma_pow = gamma * gamma
            for z_int in zs:
                z_pows = [
                    f_to_ef(f_const(pow(z_int, 1 << t, MODULUS))) for t in range(dim)
                ]
                w_evals = w_evals + gamma_pow * eq_cube_table(z_pows)
                gamma_pow = gamma_pow * gamma

        log_rs_domain_size -= 1

    assert final_poly is not None
    return transcript, WhirProof(
        mu_pow_witness=mu_pow_witness,
        whir_sumcheck_polys=whir_sumcheck_polys,
        codeword_commits=codeword_commits,
        ood_values=ood_values,
        folding_pow_witnesses=folding_pow_witnesses,
        query_phase_pow_witnesses=query_phase_pow_witnesses,
        initial_round_opened_rows=initial_round_opened_rows,
        initial_round_merkle_proofs=initial_round_merkle_proofs,
        codeword_opened_values=codeword_opened_values,
        codeword_merkle_proofs=codeword_merkle_proofs,
        final_poly=final_poly,
        mu=mu,
    )
