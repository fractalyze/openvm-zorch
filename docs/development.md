# Development

Pure Python on FRX + the XLA PJRT plugin. Bazel 9 (bzlmod). Tests default to
`FRX_PLATFORMS=cpu`.

```sh
bazel test //...                 # hermetic, sandboxed
# iterative dev outside Bazel:
export PYTHONPATH="$PWD:/abs/path/to/zorch"
```

A read-only worktree of the reference prover is expected at
`$DEVENV_ENVS_DIR/zorch/stark-backend` (see [`architecture.md`](architecture.md)).

## Dependency on zorch

`zorch` is a Bazel module, pinned in `MODULE.bazel` via `git_override` to a
main commit. For dev against a local working copy, add to `.bazelrc.user`
(gitignored — holds an absolute path):

```
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

Bump the pin when you need newer `zorch` blocks; keep it on `main` commits so
CI is reproducible. A pin bump's commit range often also moves `zorch`'s own
frx wheel pins (`frx`/`frxlib`/`frx-cuda12-pjrt`) — that does **not** force a
matching bump here. This repo pins those wheels independently in
`requirements.in`, on a separate pip hub from `zorch`'s, and their **versions**
are allowed to skew — the distribution *name* must match, or the two hubs put
`jax` and `frx` on one interpreter as two separate modules. Only touch
`requirements.in` if a build actually breaks; otherwise the pin bump is a
one-line `MODULE.bazel` change. (To validate the pin the way CI resolves it,
temporarily disable the `.bazelrc.user` `--override_module=zorch` line —
otherwise the build silently uses your local checkout, not the pinned commit.)

## Running on GPU

`//openvm_zorch:verify_prove` is the entry point — the byte-match +
per-stage-timing runnable, openvm's sibling of sp1-zorch's `verify_prove_shard`.

- A target only sees the GPU if it deps **both** `requirement("frx_cuda12_plugin")`
  and `requirement("frx_cuda12_pjrt")`; without them frx **silently falls back to
  CPU**. Run with `FRX_PLATFORMS=cuda` (not `gpu`, which also inits rocm and
  dies) so a missing plugin hard-errors instead of silently using CPU.
- Those plugin `.so`s require **`libcuda` at import**, so a cuda-dep'd target
  cannot even import on a driverless machine. Therefore tests stay
  **backend-agnostic** (no cuda deps) so `bazel test //...` runs on any
  machine; GPU lives only in `bazel run` tools like `verify_prove`.
- `Proof` (and its stage sub-proofs) are plain dataclasses, not registered
  pytrees, so `frx.block_until_ready(proof)` is a **no-op** — walk the tree
  and block on the array leaves to time the device honestly.

## Recurring gotchas

- XLA-native `lax.fft` accepts at most **2-D input** on field dtypes —
  flatten all leading batch axes before any NTT call and reshape after
  (first hit in `openvm_zorch/commit/rs_message.py`; Stages 3/5 are
  DFT-heavy and will hit it again).
- The Rust reference sizes buffers from *lifted cell counts*, not occupied
  extent (e.g. the stacked matrix can end in an all-zero committed column).
  When a byte-match fails at a hash, first suspect a shape/padding delta,
  not the hash params.
- `frx.numpy` is a subset of upstream JAX's: `fnp.roll` does not exist (first hit
  in Stage 4's rotation kernel — use
  `fnp.concatenate([a[-1:], a[:-1]])`), and `fnp.arange` iota is
  unsupported for extension dtypes (zorch builds domains via `fnp.stack`
  of scalars). When an attribute error names a fnp function, reach for a
  concat/stack equivalent before suspecting your logic. Also: `fnp.stack`
  (and `fnp.concatenate`) require each element to ALREADY be an array — they
  do NOT `asarray` a nested Python list (`stack requires ndarray or scalar
  arguments, got list`). A flat `list[Array]` of 0-D scalars stacks directly,
  but to stack rows of scalars (a `list[list[Array]]`) into a matrix you must
  inner-stack each row first: `fnp.stack([fnp.stack(row) for row in rows])`.
  `fnp.pad`, by contrast, DOES work on extension dtypes (verified byte-exact).
- **Perf: a host-int weight loop (a `pow()` nest building a constant
  matrix, contracted into field cells with scalar `acc += w*cell` adds) is
  a dispatch storm, not a FLOP cost.** It dominates eagerly. Fix: build the
  constant weight matrix once (`@lru_cache` keyed on `l_skip`/`num_cosets`,
  as a field array) and replace the scalar nest with one broadcast-multiply
  + a **trailing-axis** `.sum` (mid-axis EF reduce faults under jit; keep
  the contracted axis last). This is eager-fast and jit-fusable. Do NOT use
  `fnp.dot`/`@`/`tensordot` — they mis-lower under `frx.jit` on XLA
  (see `zorch/fusion.py`, `zorch/pcs/whir/_math.py`). And do NOT wrap a
  scalar-list polynomial (`_conv` over 0-D coeffs) in `frx.jit` directly —
  hundreds of pytree-leaf scalars regress; vectorize into arrays first.
  (PR #14 took round-0 prism 4.26→1.0s, whole prove −29%, this way.)

## Benchmarking

The bar is the **native (Rust) reference prover** — openvm-stark-backend
`v2.0.0` (`16d60de7`), the exact prover this repo byte-matches — proving
the same instance at the same params. The timed unit on both sides is the
prove step alone (trace-in → proof-out, `prove_chain`'s scope): keygen,
tracegen, and device transport are one-time setup, recorded separately under
`setup_s`, never folded into the prove number.

### Generating a native baseline

`tools/fixture-gen --baseline-out` times the native prover on the synthetic
instance and writes a JSON under `openvm_zorch/testdata/baseline/` (committed,
keyed by platform + params):

```sh
cd tools/fixture-gen

# CPU (rayon-parallel — the configuration the production native prover runs):
FIB_LOG_HEIGHT=20 N_STACK=16 BENCH_RUNS=5 BENCH_PLATFORM_LABEL="<machine>" \
  cargo run --release --features parallel -- \
  --baseline-out ../../openvm_zorch/testdata/baseline/native_prod_cpu.json

# GPU (openvm-cuda-backend compiles .cu kernels at build time, so this needs
# a CUDA toolchain + GPU on the build box):
FIB_LOG_HEIGHT=20 N_STACK=16 BENCH_RUNS=5 BENCH_PLATFORM_LABEL="<machine>" \
  CUDA_VISIBLE_DEVICES=0 \
  cargo run --release --features cuda -- \
  --baseline-out ../../openvm_zorch/testdata/baseline/native_prod_gpu.json
```

`BENCH_RUNS` (default 3) sets the warm-run count; `BENCH_PLATFORM_LABEL` tags
the machine. `FIB_LOG_HEIGHT` / `L_SKIP` / `N_STACK` / `K_WHIR` are the same
scaling knobs that size `--prove-out` (see
[`tools/fixture-gen/README.md`](../tools/fixture-gen/README.md)).
`--features parallel` is for `--baseline-out` only — the `--*-out` fixture
generators must stay single-threaded or the LogUp PoW grind goes
non-deterministic and fixtures stop being reproducible.

The synthetic instance is **narrow** (tall traces, few columns), so it
under-utilizes a GPU. When the absolute numbers matter, baseline a **real
openvm guest block** instead with
[`tools/real-block-gen`](../tools/real-block-gen)'s `--baseline-out` — same
JSON schema, same timed scope, on the tapped `ProvingContext` (see that
README).

### Comparing zorch against the baseline

`verify_prove --baseline <file>` proves the chain cold once (compiles, byte-
matches), then warm `--runs=N` more times, and prints the zorch per-stage
`_TimedRound` **converged min** (across warm passes) against the native e2e
number, with the delta. Use `--runs=5`: the first warm pass has not settled
(allocator / driver caches) and reads high, so the min over 3–5 passes is the
number worth pinning. The baseline's params must match the fixture's, so pair a
production-scale fixture with a production baseline:

```sh
# 1. Generate the production-scale fixture (multi-MB, not committed):
( cd tools/fixture-gen && FIB_LOG_HEIGHT=20 N_STACK=16 \
    cargo run --release -- --prove-out /tmp/prove_prod )

# 2. Compare (GPU; drop FRX_PLATFORMS for CPU):
FRX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  FRX_COMPILATION_CACHE_DIR=/tmp/frx_cache \
  bazel run //openvm_zorch:verify_prove -- \
    --fixture_dir /tmp/prove_prod --runs=5 \
    --baseline openvm_zorch/testdata/baseline/native_prod_gpu.json
```

Running the committed micro fixture (`testdata/prove`) against a production
baseline warns about the param mismatch — that combination is only a smoke
check of the wiring.

### Per-stage comparison (real fibonacci block, GPU)

The living baseline. Regenerate on every perf change and update the numbers so
the ratios below stay honest. Measured on an **idle RTX 5090** over the real
openvm fibonacci block (19 AIRs, `tools/real-block-gen`), `frx` pin
`0.10.0.dev20260716113241`, byte-match **ALL OK**, `--runs=5` converged min:

```sh
FRX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  FRX_COMPILATION_CACHE_DIR=/tmp/frx_cache \
  bazel run //openvm_zorch:verify_prove -- --fixture_dir /tmp/real_fib --runs=5 \
    --baseline openvm_zorch/testdata/baseline/native_realfib_gpu.json
```

| stage | zorch warm | native openvm | zorch / native |
|---|---|---|---|
| trace commit | 5.7 ms | 4.4 ms | 1.3× |
| LogUp-GKR | 99.3 ms | 7.2 ms | 13.8× |
| zerocheck | 154.4 ms | 10.6 ms | 14.6× |
| stacking | 245.3 ms | 6.2 ms | 39.7× |
| WHIR | 638.4 ms | 6.0 ms | 106.9× |
| **full prove** | **1143 ms** | **34.3 ms** | **33.4×** |

Native GKR is the `fractional_sumcheck` span; native zerocheck is
`prove_zerocheck_and_logup − fractional_sumcheck` (GKR nests inside it). The five
native per-stage bars sum to the `stark_prove_excluding_trace` e2e span. WHIR is
the widest gap (~107×) — zorch's WHIR is 56% of its prove but ~2× the native
stage's *share*.

> **Native per-stage capture.** The native CUDA backend names its phase spans
> differently from the CPU set — `_gpu`-suffixed for zerocheck/GKR,
> `prover.openings.*` for stacking/WHIR — so `tools/real-block-gen/dump_fixture.rs`
> `phase_key` normalizes every backend's name to one canonical `per_stage_s` key
> (openvm-stark-backend `v2.0.0-beta.2`). All five stages are captured; regenerate
> the baseline (`--features cuda`, CUDA box) after any prover change.

**Cold is cache-state-dependent, so it is not in the table.** A true first-ever
cold (empty `FRX_COMPILATION_CACHE_DIR`) compiles zerocheck's whole-stage jit in
one ~259 s XLA pass (`jit(run)`, 95% of the stage cold) plus WHIR ~75 s. With a
populated cache a fresh process **loads** the executables instead of recompiling
(persistent cache hit): zerocheck ~29 s, ~16 s of which is the `jit(run)`
deserialize. So configure `FRX_COMPILATION_CACHE_DIR` in prod/CI and the 259 s is
paid once. See #120 for the cold-compile lever discussion.

### GPU measurement hygiene

- **Idle GPU only.** Shared-GPU contention inflates every stage 2–50× and
  makes the per-stage ranking meaningless (#71).
- **The first GPU `prove()` is compile-dominated, not kernel-dominated.** Each
  stage now lowers as one cached jit, so a true cold pays a single large XLA
  compile per stage — zerocheck's `jit(run)` is ~259 s (95% of the stage cold,
  PR #119), WHIR ~75 s. Two levers:
  - `FRX_COMPILATION_CACHE_DIR=<dir>` persists compiled executables across process
    runs; every run after the first **loads** them (persistent cache hit) rather
    than recompiling — zerocheck drops ~259 s → ~29 s (the residual is the
    `jit(run)` deserialize, not a recompile). Leave it unset for byte-match gates
    — a true cold compile is part of the gate. See #120.
  - `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=<cores>` parallelizes the
    first compile (~25% off; complements, not replaces, the cache).
- The native GPU prover returns before its kernels finish; fixture-gen's timer
  already brackets `prove` with stream syncs, so the baseline JSONs record real
  completion times — don't re-measure with a bare wall-clock around the call.
