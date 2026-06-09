# openvm-zorch

A lean [OpenVM](https://github.com/openvm-org/openvm) prover built on
[zorch](https://github.com/fractalyze/zorch)'s scheme-agnostic SNARK blocks.

```
JAX → zorch (scheme-/zkVM-agnostic blocks) → openvm-zorch (SWIRL glue)
```

OpenVM proves with **SWIRL** — a sumcheck-based proof system composing
LogUp-GKR (interactions), a batched ZeroCheck with univariate skip
(constraints), a stacked opening reduction, and a WHIR polynomial commitment —
as implemented by
[openvm-stark-backend](https://github.com/openvm-org/stark-backend) on the
`develop-v2` lineage. This repo re-implements that prover on zorch, keeping
only the SWIRL-specific surface here and pushing every generic block upstream.

## Status

| Stage | What it proves | Module | Status |
|-------|----------------|--------|--------|
| 1. Trace commit | Stacked PCS: stack traces → RS-encode columns → query-strided Merkle root | `openvm_zorch/commit` | byte-matches reference |
| 2. LogUp-GKR | Interaction fractional sumcheck | — | not started |
| 3. ZeroCheck | Batched constraints, univariate skip + multivariate sumcheck | — | not started |
| 4. Stacked reduction | Column openings → stacked matrix openings | — | not started |
| 5. WHIR opening | k-ary fold + OOD + query phase + grinding | — | not started |

## Quick start

```sh
bazel test //...                              # hermetic, CPU by default
```

Regenerate golden fixtures (requires Rust toolchain; pinned to the reference
tag, so output is reproducible):

```sh
cd tools/fixture-gen
cargo run --release -- --out ../../openvm_zorch/commit/testdata/stacked_commit
```

## Reference pin

`openvm-stark-backend` tag `v2.0.0-beta.2`
(`f6a84921e46a7df9796d41dfdfe69f0658ad74b5`), BabyBear + Poseidon2 width-16
(plonky3 `=0.4.1`). See [`CLAUDE.md`](CLAUDE.md) and
[`docs/swirl-pipeline.md`](docs/swirl-pipeline.md).
