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

`verify_prove --baseline <file>` runs a second warm chain pass and prints the
zorch per-stage `_TimedRound` warm sum against the native e2e number, with the
delta. The baseline's params must match the fixture's, so pair a
production-scale fixture with a production baseline:

```sh
# 1. Generate the production-scale fixture (multi-MB, not committed):
( cd tools/fixture-gen && FIB_LOG_HEIGHT=20 N_STACK=16 \
    cargo run --release -- --prove-out /tmp/prove_prod )

# 2. Compare (GPU; drop FRX_PLATFORMS for CPU):
FRX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  bazel run //openvm_zorch:verify_prove -- \
    --fixture_dir /tmp/prove_prod \
    --baseline openvm_zorch/testdata/baseline/native_prod_gpu.json
```

Running the committed micro fixture (`testdata/prove`) against a production
baseline warns about the param mismatch — that combination is only a smoke
check of the wiring.

### GPU measurement hygiene

- **Idle GPU only.** Shared-GPU contention inflates every stage 2–50× and
  makes the per-stage ranking meaningless (#71).
- **The first GPU `prove()` is compile-dominated, not kernel-dominated**: the
  zerocheck constraint DAG unrolls into one giant `jit_scan` kernel that ptxas
  optimizes for minutes. Two levers (#70):
  - `FRX_COMPILATION_CACHE_DIR=<dir>` persists compiled modules across process
    runs, so every run after the first skips the compile. Leave it unset for
    byte-match gates — a true cold compile is part of the gate.
  - `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=<cores>` compiles the
    per-AIR kernels in parallel (~25% off the first compile; complements, not
    replaces, the cache — the dominant cost is a single serial-ptxas kernel).
- The native GPU prover returns before its kernels finish; fixture-gen's timer
  already brackets `prove` with stream syncs, so the baseline JSONs record real
  completion times — don't re-measure with a bare wall-clock around the call.
