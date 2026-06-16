# Native prover baseline

The single source of truth for "how fast is the native (Rust) openvm prover?",
against which milestone #4's per-stage optimizations are measured. The numbers
below are the wall-clock of the **reference SWIRL prover**
(openvm-stark-backend `v2.0.0-beta.2`, `f6a84921`) — the exact prover this
repo byte-matches — proving the same synthetic instance `prove_chain` consumes,
at the same params.

## What is timed

The timed unit is `prover.prove(&d_pk, d_ctx)` alone: trace-in → proof-out.
That is exactly the scope of `prove_chain` (commit → LogUp-GKR → ZeroCheck →
stacking → WHIR), which takes the fixture traces as input. keygen, tracegen,
and the device transport are one-time setup, recorded separately under
`setup_s`, never folded into the prove number — `prove_chain` does none of them.

A non-recording `DuplexSponge` transcript is used (not the
`DuplexSpongeRecorder` the fixture generator uses); the recorder's per-step log
append is pure overhead unrelated to proving. Both derive identical Fiat-Shamir
challenges, so the proof is byte-identical.

**CPU only.** `BabyBearPoseidon2RefEngine` is backed by `CpuColMajorBackend`;
the reference prover has no GPU backend at this pin. The GPU comparison lives on
the zorch side (`verify_prove` on CUDA), measured against this CPU bar.

## Recorded numbers (`cpu-x86_64-32t`)

| Scale | Params (`l_skip`/`n_stack`/`k_whir`) | Stacked height | WHIR rounds | Fib trace | Native prove (warm min) |
|-------|--------------------------------------|----------------|-------------|-----------|-------------------------|
| micro | 4 / 8 / 4  | 2¹² | 3 | 2⁶  | 0.021 s |
| prod  | 4 / 16 / 4 | 2²⁰ | 5 | 2²⁰ | 7.67 s  |

- **micro** matches the committed `testdata/prove` fixture, so
  `verify_prove --baseline` works out of the box against the default fixture.
- **prod** is the milestone-#4 bar: a production-scale block (stacked 2²⁰).

The committed JSONs are keyed by platform + params:

- `openvm_zorch/testdata/baseline/native_micro_cpu.json`
- `openvm_zorch/testdata/baseline/native_prod_cpu.json`

## Reproducing

```sh
cd tools/fixture-gen

# micro (matches the committed fixture):
BENCH_RUNS=5 BENCH_PLATFORM_LABEL="<machine>" cargo run --release -- \
  --baseline-out ../../openvm_zorch/testdata/baseline/native_micro_cpu.json

# production-scale (stacked 2^20):
FIB_LOG_HEIGHT=20 N_STACK=16 BENCH_RUNS=3 BENCH_PLATFORM_LABEL="<machine>" \
  cargo run --release -- \
  --baseline-out ../../openvm_zorch/testdata/baseline/native_prod_cpu.json
```

`BENCH_RUNS` (default 3) sets the warm-run count; `BENCH_PLATFORM_LABEL`
(default `cpu`) tags the machine. The `FIB_LOG_HEIGHT` / `L_SKIP` / `N_STACK` /
`K_WHIR` scaling knobs are the same ones that scale `--prove-out` (see
[`../tools/fixture-gen/README.md`](../tools/fixture-gen/README.md)).

## Comparing zorch against the baseline

```sh
bazel run //openvm_zorch:verify_prove -- \
  --baseline openvm_zorch/testdata/baseline/native_micro_cpu.json
```

`verify_prove --baseline <file>` prints the zorch per-stage `_TimedRound` warm
sum against the native e2e number, with the delta. Point `--fixture_dir` at a
generated production-scale fixture and `--baseline` at the matching prod JSON to
compare at scale.
