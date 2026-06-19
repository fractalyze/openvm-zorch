# openvm-zorch docs

Reference documentation for the openvm-zorch SWIRL prover. Start with the
[project README](../README.md) for the overview and quick start.

| Doc | What's in it |
|-----|--------------|
| [`swirl-pipeline.md`](swirl-pipeline.md) | The five SWIRL stages in detail, end-to-end composition, verifier, terminology mapping to openvm-stark-backend, and the `SystemParams` cheat sheet. |
| [`conventions.md`](conventions.md) | Repo conventions on top of zorch's — the OpenVM/SWIRL-specific-only rule, why-not-what comments, pinned reference permalinks, byte-match bar, reference-tracking naming. |
| [`development.md`](development.md) | Dev environment (JAX/Bazel), the `zorch` dependency and pin/wheel-bump rules, running on GPU, and the recurring JAX/fork gotchas. |
| [`byte-match.md`](byte-match.md) | The correctness bar: fixture sources (`fixture-gen`, `real-block-gen`), reading a `MISMATCH` report, and the per-stage fixture-generation pattern. |
| [`native-baseline.md`](native-baseline.md) | The native (Rust) prover wall-clock baseline milestone #4 measures against — what is timed, the CPU/GPU split, recorded numbers, and how to reproduce. |
