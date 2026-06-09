# SWIRL pipeline — stages, terminology, file inventory

Reference: openvm-stark-backend `v2.0.0-beta.2` (`f6a84921`). A read-only
worktree is convenient at `$DEVENV_ENVS_DIR/zorch/stark-backend`:

```sh
cd /path/to/stark-backend && git worktree add \
  "$DEVENV_ENVS_DIR/zorch/stark-backend" v2.0.0-beta.2
```

## The five stages

SWIRL proves a multi-AIR system in five stages, each a Round composition
threading one Fiat-Shamir transcript (zorch `DuplexTranscript` ↔ Rust
`DuplexSponge<BabyBear, Poseidon2, 16, 8>`):

| # | Stage | Rust entry point (crates/stark-backend/src/) | This repo |
|---|-------|----------------------------------------------|-----------|
| 1 | Trace commit | `prover/stacked_pcs.rs` `stacked_commit()` | `openvm_zorch/commit` |
| 2 | LogUp-GKR | `prover/logup_zerocheck/mod.rs` `prove_zerocheck_and_logup()` (GKR half) | — |
| 3 | ZeroCheck | same entry, batched-constraint half | — |
| 4 | Stacked reduction | `prover/stacked_reduction.rs` `prove_stacked_opening_reduction()` | — |
| 5 | WHIR opening | `prover/whir.rs` `prove_whir_opening()` | — |

## Stage 1 in detail (implemented)

`stacked_commit(l_skip, n_stack, log_blowup, k_whir, traces)`:

1. **Stack** (`stacked_matrix`): traces, pre-sorted by descending height, are
   stacked column-by-column into one matrix of height `2^(l_skip + n_stack)`.
   A column shorter than `2^l_skip` is lifted to length `2^l_skip` by striding
   (value at `i*stride`, zeros between). Columns are laid head-to-tail down
   each stacked column; a column never straddles two stacked columns because
   every height divides the stacked height.
2. **RS-encode** (`rs_code_matrix`): each stacked column is read as
   evaluations of a *prismalinear* polynomial on `D × {0,1}^n_stack`
   (`D` = the order-`2^l_skip` two-adic subgroup). `eval_to_coeff_rs_message`
   converts to the RS message: per-`2^l_skip`-chunk inverse NTT, then a
   per-chunk MLE coeff→eval transform over the chunk's bit-variables. The
   message is zero-padded by `2^log_blowup` and forward-NTT'd (natural order,
   no coset shift, no bit-reversal).
3. **Merkle** (`MerkleTree::new` with `rows_per_query = 2^k_whir`): hash each
   codeword row (Poseidon2 sponge, rate 8, digest 8), then fold the first
   `k_whir` levels with *query-strided* pairing — level pairs are
   `(2x·s + y, (2x+1)·s + y)` for `s = num_leaves / rows_per_query` — so one
   WHIR query later opens `rows_per_query` rows under a single digest. Levels
   above fold as plain adjacent pairs. The commitment is the root.

## Terminology mapping

| Rust (stark-backend) | Here / zorch |
|----------------------|--------------|
| `ColMajorMatrix<F>` | `(height, width)` row-major `jax.Array` (transpose at the fixture boundary) |
| `PaddingFreeSponge<Perm,16,8,8>` | `zorch.hash.sponge.Sponge(rate=8, out=8)` |
| `TruncatedPermutation<Perm,2,8,16>` | `zorch.hash.compression.Compression(arity=2, chunk=8)` |
| `default_babybear_poseidon2_16` | `openvm_zorch.poseidon2.babybear16` |
| `Radix2Bowers.idft` / `Radix2DitParallel.dft` | `lax.fft(x, "IFFT"/"FFT", n)` (zkx-native NTT) |
| `Mle::coeffs_to_evals_inplace` | `mle_coeffs_to_evals` in `openvm_zorch/commit/rs_message.py` |

## SystemParams cheat sheet

`l_skip` univariate-skip dimension; `n_stack` stacking dimension (stacked
height = `2^(l_skip+n_stack)`); `w_stack` max stacked width; `log_blowup` RS
rate; `k_whir` WHIR folding arity exponent (`rows_per_query = 2^k_whir`).
Production app-VM values: `l_skip=4`, `log_blowup=1`, `k_whir=4`
(`crates/stark-sdk/src/config/mod.rs`).
