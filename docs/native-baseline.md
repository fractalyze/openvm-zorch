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

On GPU the prover dispatches kernels on the current CUDA stream and returns
before they finish, so the timer brackets the `prove` call with
`current_stream_sync()` (after, to time real completion; before `t0`, to drain
the prior run and the async H2D transport). Without that, the timer catches only
launch overhead — ~11 ms, a meaningless ~100× over the real compute.

## CPU and GPU

The CPU baseline is the **parallel** (rayon, multi-core) reference engine
(`BabyBearPoseidon2RefEngine` / `CpuColMajorBackend`), built with
`--features parallel` — the configuration the production native prover actually
runs. fixture-gen's default build is single-threaded (`default-features = false`,
so the LogUp PoW grind is deterministic and fixtures stay byte-identical); that
is right for the `--*-out` generators but would make a 1-core bar (~6× slower),
so `--baseline-out` is built with `--features parallel` instead.

The GPU baseline is the CUDA `BabyBearPoseidon2GpuEngine` (`openvm-cuda-backend`),
built with `--features cuda`. The GPU prover derives identical Fiat-Shamir
challenges, so its proof byte-matches the CPU one (gate that separately with
`verify_prove` on CUDA). Because `openvm-cuda-backend` compiles `.cu` kernels at
build time, the cuda build needs a CUDA toolchain + GPU and must be built and run
on a GPU box (a41, RTX 5090) — it will not build on a driverless machine.

## Recorded numbers

Production-scale block: `FIB_LOG_HEIGHT=20 N_STACK=16` (`l_skip=4 / n_stack=16 /
k_whir=4`, stacked height 2²⁰, 5 WHIR rounds, Fibonacci trace 2²⁰). Both measured
on **a41** (same machine, fair CPU-vs-GPU delta).

| Platform | File | Native prove (warm min) |
|----------|------|-------------------------|
| `a41-cpu-24t` (parallel) | `native_prod_cpu.json` | 1.28 s |
| `a41-rtx5090` | `native_prod_gpu.json` | 0.011 s |

The committed JSONs live in `openvm_zorch/testdata/baseline/` and are keyed by
platform + params.

**Caveat — the synthetic block is narrow.** The scale-up grows trace *height*
(2²⁰ rows) but not *width*: the AIRs are a 3-column Fibonacci plus tiny dummy
interaction AIRs. Real openvm blocks have many wide AIRs, so the absolute FLOPs
here are small — the GPU is under-utilized (the 0.011 s reflects a starved 5090),
and the CPU-vs-GPU ratio is not what a real block would show. These are useful as
a consistent, byte-matched bar for the per-stage work, but the faithful
real-guest-block path (milestone #3 "Option A") stays deferred. Treat the numbers
as a floor, not a representative production workload.

## Reproducing

```sh
cd tools/fixture-gen

# CPU baseline (parallel; runs anywhere):
FIB_LOG_HEIGHT=20 N_STACK=16 BENCH_RUNS=5 BENCH_PLATFORM_LABEL="<machine>" \
  cargo run --release --features parallel -- \
  --baseline-out ../../openvm_zorch/testdata/baseline/native_prod_cpu.json

# GPU baseline (a41, with CUDA toolchain + GPU):
FIB_LOG_HEIGHT=20 N_STACK=16 BENCH_RUNS=5 BENCH_PLATFORM_LABEL="a41-rtx5090" \
  CUDA_VISIBLE_DEVICES=0 \
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
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  bazel run //openvm_zorch:verify_prove -- \
    --fixture_dir /tmp/prove_prod \
    --baseline openvm_zorch/testdata/baseline/native_prod_gpu.json
```

Running the committed micro fixture (`testdata/prove`) against a production
baseline warns about the param mismatch — that comparison is not
apples-to-apples and exists only as a smoke check of the wiring.
