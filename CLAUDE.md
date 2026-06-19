# Project context for Claude Code

This file is a thin pointer. The substantive docs live under [`docs/`](docs/README.md);
keep engineering content there, not here.

- **Overview & quick start:** [`README.md`](README.md)
- **Docs hub:** [`docs/README.md`](docs/README.md)
- **Conventions:** [`docs/conventions.md`](docs/conventions.md) — the OpenVM/SWIRL-specific-only non-negotiable, comments (why-not-what), pinned reference permalinks, byte-match bar, reference-tracking naming.
- **Development:** [`docs/development.md`](docs/development.md) — dev environment, the `zorch` dependency and pin/wheel-bump rules, running on GPU, and the recurring JAX/fork gotchas.
- **Byte-match:** [`docs/byte-match.md`](docs/byte-match.md) — fixture sources, reading a `MISMATCH` report, and the per-stage fixture-generation pattern.
- **Pipeline & terminology:** [`docs/swirl-pipeline.md`](docs/swirl-pipeline.md) — SWIRL stages as Round compositions, openvm-stark-backend vocabulary mapping.
- **Native baseline:** [`docs/native-baseline.md`](docs/native-baseline.md) — the native (Rust) prover wall-clock milestone #4 measures against, and the CPU / GPU split.
