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
| 2 | LogUp-GKR | `prover/logup_zerocheck/mod.rs` `prove_zerocheck_and_logup()` (GKR half) | `openvm_zorch/logup_gkr` |
| 3 | ZeroCheck | same entry, batched-constraint half | `openvm_zorch/logup_zerocheck` |
| 4 | Stacked reduction | `prover/stacked_reduction.rs` `prove_stacked_opening_reduction()` | `openvm_zorch/stacked_reduction` |
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

## Stage 2 in detail (implemented)

The GKR half of `prove_zerocheck_and_logup` (everything before the batched
constraint sumcheck), threading the transcript left by Stage 1's commits:

1. **PoW + challenges**: grind witness observe + one masked squeeze
   (`LogUpSecurityParameters.pow_bits`), then α (division-by-zero guard) and
   β (message fingerprint), each 4 base squeezes
   (`openvm_zorch/transcript.py` over zorch `DuplexTranscript`).
2. **Input layer** (`logup_gkr/input_layer.py`): per trace row and
   interaction, the fraction `count / h_β(message ‖ bus)` with
   `h_β(σ‖b) = β^len(σ)·(b+1) + Σ_j β^j·σ_j`, stacked into one hypercube
   `H_{l_skip+n_logup}` by the Stage-1 `StackedLayout` (striding threshold 0).
   Short traces lift by cyclic repetition with the inverse factor on the
   numerator; `q += α` everywhere (off-image slots become `0/α`).
3. **Fractional sumcheck** (`logup_gkr/prover.py`): zorch's dense
   `GkrLayer`/`build_pyramid` is the reference's segment tree byte-for-byte
   (stride-2 pair fold). Per layer j: sample λ_j, run j sumcheck rounds —
   round polys observed as evaluations on `{1,2,3}`, summand
   `eq·((p0q1 + p1q0) + λ·q0q1)`, LSB bound first — then observe the claims
   `(p(0,ρ), q(0,ρ), p(1,ρ), q(1,ρ))` and sample μ_j; `ξ^{(j)} = (μ_j, ρ)`.
   The root observes only q₀ (the numerator must vanish). The per-layer
   driver lives here, not in zorch: the reference's wire format differs from
   zorch's own GKR protocol in form (see `prover.py` docstring).
4. **ξ padding**: extra coordinates sampled until `len(ξ) = l_skip +
   max(n_max, n_logup)` when some AIR out-sizes the interactions hypercube.

The fixture (`tools/fixture-gen --gkr-out`) proves a 5-AIR instance (Fib
without interactions + two bus-balanced `DummyInteractionAir` pairs, heights
64/8/4/2/2) end-to-end with a recording transcript and replays
`fractional_sumcheck` through a `ReadOnlyTranscript` to validate its own
input-layer reconstruction before dumping.

## Stage 3 in detail (implemented)

The batched-constraint half of `prove_zerocheck_and_logup` (everything after
the ξ padding), threading the transcript left by Stage 2. Per *sorted* trace
T three polynomials are batched under μ-powers — the logup numerator
(`μ^{2T}`), the logup denominator (`μ^{2T+1}`) and the λ-folded zerocheck
constraint sum (`μ^{2N+T}`):

1. **λ + constraint DAG** (`logup_zerocheck/constraints.py`): keygen's
   `SymbolicExpressionDag` (nodes topologically ordered; constraints and
   interaction expressions reference nodes by index) is dumped by fixture-gen
   as canonical-u32 JSON and evaluated vectorized — node values carry the
   whole evaluation grid as a leading batch shape.
2. **Univariate round 0** (`logup_zerocheck/prism.py` + driver): each
   per-trace `s'_0` is interpolated from evaluations on geometric cosets
   `g^{c+1}·D` (per-x window iDFT + coset DFT; the zerocheck variant divides
   by the zerofier `Z^{2^l_skip}−1` and re-multiplies in coefficient form).
   The univariate eq factors — `eq_D(ξ_0, ·)` for zerocheck, the ♯-twisted
   `eq♯_D(ξ_{<l_skip}, ·)` for logup — multiply in coefficient form; the
   per-trace sum claims `(Σ p̂, Σ q̂)` are read off the product coefficients
   (`Σ_D Z^j = |D|` iff `|D| divides j`) and observed; then μ batches
   everything into one `s_0`, observed in **coefficient** form (degree
   `(d+1)(2^l_skip−1)`).
3. **MLE rounds 1..n_max**: round polys observed as evaluations on
   `{1..d+1}`; `s'(0)` is derived from `s_j(0)+s_j(1) = s_{j−1}(r_{j−1})`,
   never computed. Front-loaded batching: a trace exhausted at round
   `> ñ_T` degenerates to a constant "tilde" contribution
   (`f̂(r⃗)·r_{…}` products, `eq`/`eq♯` accumulated separately); the
   `eq(ξ_round, X)` linear factor multiplies in coefficient form. Folding
   reuses zorch `fold_pair`/`lift_to_domain` (LSB pairing) after the round-0
   PLE fold (per-window interpolation at r₀).
4. **Column openings**: per trace, common main first as (claim, claim_rot)
   pairs — rotation slots observed as 0 when the AIR never rotates — then
   preprocessed/cached parts (none in scope yet).

The fixture (`tools/fixture-gen --zerocheck-out`) extends the Stage-2 walk
from `meta.json: stage2_end`, asserts the whole observe/sample structure
against the proof, and self-validates by rebuilding the prelude transcript
state and rerunning `prove_zerocheck_and_logup` — the rerun must reproduce
the recorded log byte-for-byte through `stage3_end`.

## Stage 4 in detail (implemented)

`prove_stacked_opening_reduction`, threading the transcript left by Stage 3:
a batch sumcheck reducing every per-trace column opening claim (and rotation
claim) at `r` to opening claims of the *stacked* matrix's columns at a fresh
point `u` of length `1 + n_stack`.

1. **λ batching**: one λ power pair per stacked column — eq claim at
   `λ^{2k}`, rotation claim at `λ^{2k+1}` (the rotation power is reserved
   even when the AIR never rotates, mirroring Stage 3's zero rot openings).
   The summand per column is `q·(eq or κ_rot)·in_{D,n_T}`, where the kernels
   against `r` split as (univariate over the skip domain) × (multilinear over
   the cube): `κ_rot` decomposes as `eq_D(Z, ω·r_0)·eq_cube +
   eq_D(Z,1)·eq_D(ω·r_0,1)·(rot_cube − eq_cube)`, short traces (`n_T < 0`)
   collapse the univariate factor to the order-`2^{l+n_T}` subgroup behind
   the stride indicator `in_{D,n_T}`.
2. **Univariate round 0**: degree `2·(2^l_skip−1)`, evaluated on the cosets
   `g·D`, `g²·D` per column window (`prism.coset_evals`) and interpolated to
   coefficients (`prism.geometric_cosets_to_coeffs`), observed in coefficient
   form; then everything PLE-folds at `u_0` (`prism.fold_ple_evals` for `q`,
   closed-form kernel evaluations for `eq`/`κ_rot` tables).
3. **MLE rounds 1..n_stack**: quadratic round polys observed as evaluations
   at `{1,2}`; a column whose cube variables are exhausted (`round > ñ_T`)
   stops folding and instead binds its position bits `b_{T,j}` (the
   `row_idx` of its `StackedSlice`) through accumulating `eq(u_round, b)`
   factors. Folding is zorch `fold_pair` (LSB pairing) on the stacked
   matrix's columns — all traces at once, per commit.
4. **Stacking openings**: after the last fold each stacked column is a single
   value `q̂(u)`; observed per commit, in column order. WHIR (Stage 5) opens
   these claims against the Stage-1 commitment.

The fixture (`tools/fixture-gen --stacking-out`) extends the Stage-3 walk
from `meta.json: stage3_end`, asserts the observe/sample structure against
`proof.stacking_proof`, and self-validates by rebuilding the common-main
`StackedPcsData` (`device.commit` on the sorted traces, root checked against
the prelude's commitment) and replaying `prove_stacked_opening_reduction`
through a `ReadOnlyTranscript` — possible here because Stage 4 has no PoW
grind. The Python test additionally pins the post-stage transcript state by
feeding Stage 5's first observe (the WHIR μ-PoW witness) and asserting the
next squeeze.

## Terminology mapping

| Rust (stark-backend) | Here / zorch |
|----------------------|--------------|
| `ColMajorMatrix<F>` | `(height, width)` row-major `jax.Array` (transpose at the fixture boundary) |
| `PaddingFreeSponge<Perm,16,8,8>` | `zorch.hash.sponge.Sponge(rate=8, out=8)` |
| `TruncatedPermutation<Perm,2,8,16>` | `zorch.hash.compression.Compression(arity=2, chunk=8)` |
| `default_babybear_poseidon2_16` | `openvm_zorch.poseidon2.babybear16` |
| `Radix2Bowers.idft` / `Radix2DitParallel.dft` | `lax.fft(x, "IFFT"/"FFT", n)` (zkx-native NTT) |
| `Mle::coeffs_to_evals_inplace` | `mle_coeffs_to_evals` in `openvm_zorch/commit/rs_message.py` |
| `DuplexSponge<BabyBear,Poseidon2,16,8>` | `openvm_zorch.transcript.new_transcript()` (zorch `DuplexTranscript`) |
| `transcript.sample_ext()` (4 base squeezes) | `openvm_zorch.transcript.sample_ext` |
| `fractional_sumcheck` segment tree | `zorch.logup_gkr.circuit.build_pyramid` (dense) |
| `evals_eq_hypercube(ξ)` (little-endian) | `expand_eq_to_hypercube(reversed ξ)` (MSB-first) |
| `SymbolicExpressionDag` + `SymbolicEvaluator` | `openvm_zorch/logup_zerocheck/constraints.py` (vectorized) |
| `eval_eq_uni` / `eq♯` / `fold_ple_evals` / round-0 cosets | `openvm_zorch/logup_zerocheck/prism.py` |
| `sumcheck_round_poly_evals` MLE fold (LSB pairs) | `zorch.sumcheck.prover.fold_pair` / `lift_to_domain` |

## SystemParams cheat sheet

`l_skip` univariate-skip dimension; `n_stack` stacking dimension (stacked
height = `2^(l_skip+n_stack)`); `w_stack` max stacked width; `log_blowup` RS
rate; `k_whir` WHIR folding arity exponent (`rows_per_query = 2^k_whir`).
Production app-VM values: `l_skip=4`, `log_blowup=1`, `k_whir=4`
(`crates/stark-sdk/src/config/mod.rs`).
