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

## Quick start

```sh
bazel test //...                              # hermetic, CPU by default
```

Regenerate golden fixtures (requires Rust toolchain; pinned to the reference
tag, so output is reproducible):

```sh
cd tools/fixture-gen
cargo run --release -- \
  --out ../../openvm_zorch/commit/testdata/stacked_commit \
  --transcript-out ../../openvm_zorch/testdata/transcript \
  --gkr-out ../../openvm_zorch/logup_gkr/testdata/logup_gkr \
  --zerocheck-out ../../openvm_zorch/logup_zerocheck/testdata/zerocheck \
  --stacking-out ../../openvm_zorch/stacked_reduction/testdata/stacking \
  --whir-out ../../openvm_zorch/whir/testdata/whir \
  --prove-out ../../openvm_zorch/testdata/prove
```

## Reference pin

`openvm-stark-backend` tag `v2.0.0`
(`16d60de724c21dcadfde7d8315a1db507e5832d7`) — the same pin the openvm
`v2.0.0` release consumes. SWIRL lives on stark-backend's `develop-v2`
lineage, NOT `main` (main is still plonky3/FRI). Config: BabyBear base field,
BabyBear⁴ challenges, Poseidon2 width-16 (`default_babybear_poseidon2_16`,
plonky3 `=0.4.3`).

## Documentation

See [`docs/`](docs/README.md) for the full index — pipeline & terminology, and
development & benchmarking.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
