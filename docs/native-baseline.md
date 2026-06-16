# Native prover baseline

The single source of truth for "how fast is the native (Rust) openvm prover?",
against which milestone #4's per-stage optimizations are measured. The numbers
below are the wall-clock of the **reference SWIRL prover**
(openvm-stark-backend `v2.0.0-beta.2`, `f6a84921`) — the exact prover this
repo byte-matches — proving the same synthetic instance `prove_chain` consumes,
at the same params, on a **production-scale block** (stacked 2²⁰).

## What is timed

The timed unit is `prover.prove(&d_pk, d_ctx)` alone: trace-in → proof-out.
That is exactly the scope of `prove_chain` (commit → LogUp-GKR → ZeroCheck →
stacking → WHIR), which takes the fixture traces as input. keygen, tracegen,
and the device transport are one-time setup, recorded separately under
`setup_s`, never folded into the prove number — `prove_chain` does none of them.

The CPU baseline uses a non-recording `DuplexSponge` transcript (not the
`DuplexSpongeRecorder` the fixture generator uses); the recorder's per-step log
append is pure overhead unrelated to proving. Both derive identical Fiat-Shamir
challenges, so the proof is byte-identical.

## CPU and GPU

The default build times the CPU reference engine
(`BabyBearPoseidon2RefEngine` / `CpuColMajorBackend`). `--features cuda` swaps in
the CUDA `BabyBearPoseidon2GpuEngine` (`openvm-cuda-backend`) and writes the GPU
baseline. The GPU prover derives identical Fiat-Shamir challenges, so its proof
byte-matches the CPU one (gate that separately with `verify_prove` on CUDA).

Because `openvm-cuda-backend` compiles `.cu` kernels at build time, the cuda
build needs a CUDA toolchain + GPU and must be built and run on a GPU box (a41,
RTX 5090) — it will not build on a driverless machine.

## Recorded numbers

Production-scale block: `FIB_LOG_HEIGHT=20 N_STACK=16` (`l_skip=4 / n_stack=16 /
k_whir=4`, stacked height 2²⁰, 5 WHIR rounds, Fibonacci trace 2²⁰).

| Platform | File | Native prove (warm min) |
|----------|------|-------------------------|
| `cpu-x86_64-32t` | `native_prod_cpu.json` | 7.67 s |
| GPU (a41, RTX 5090) | `native_prod_gpu.json` | _pending — generate on a41_ |

The committed JSONs live in `openvm_zorch/testdata/baseline/` and are keyed by
platform + params.

## Reproducing

```sh
cd tools/fixture-gen

# CPU baseline (this runs anywhere):
FIB_LOG_HEIGHT=20 N_STACK=16 BENCH_RUNS=3 BENCH_PLATFORM_LABEL="<machine>" \
  cargo run --release -- \
  --baseline-out ../../openvm_zorch/testdata/baseline/native_prod_cpu.json

# GPU baseline (a41, with CUDA toolchain + GPU):
FIB_LOG_HEIGHT=20 N_STACK=16 BENCH_RUNS=3 BENCH_PLATFORM_LABEL="a41-rtx5090" \
  cargo run --release --features cuda -- \
  --baseline-out ../../openvm_zorch/testdata/baseline/native_prod_gpu.json
```

`BENCH_RUNS` (default 3) sets the warm-run count; `BENCH_PLATFORM_LABEL` tags the
machine (default `cpu`, or `gpu` under `--features cuda`). The `FIB_LOG_HEIGHT` /
`L_SKIP` / `N_STACK` / `K_WHIR` scaling knobs are the same ones that scale
`--prove-out` (see [`../tools/fixture-gen/README.md`](../tools/fixture-gen/README.md)).

## Comparing zorch against the baseline

`verify_prove --baseline <file>` runs a second warm chain pass and prints the
zorch per-stage `_TimedRound` warm sum against the native e2e number, with the
delta. The baseline params must match the fixture's, so pair a production-scale
fixture with the production baseline:

```sh
# 1. Generate the production-scale fixture (multi-MB, not committed):
( cd tools/fixture-gen && FIB_LOG_HEIGHT=20 N_STACK=16 \
    cargo run --release -- --prove-out /tmp/prove_prod )

# 2. Compare (GPU; drop JAX_PLATFORMS for CPU):
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  bazel run //openvm_zorch:verify_prove -- \
    --fixture_dir /tmp/prove_prod \
    --baseline openvm_zorch/testdata/baseline/native_prod_gpu.json
```

Running the committed micro fixture (`testdata/prove`) against a production
baseline warns about the param mismatch — that comparison is not
apples-to-apples and exists only as a smoke check of the wiring.
