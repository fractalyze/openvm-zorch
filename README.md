# openvm-zorch

A lean [OpenVM](https://github.com/openvm-org/openvm) prover built on
[zorch](https://github.com/fractalyze/zorch)'s scheme-agnostic SNARK blocks.

```
FRX → zorch (scheme-/zkVM-agnostic blocks) → openvm-zorch (SWIRL glue)
```

FRX is Fractalyze's fork of [JAX](https://github.com/jax-ml/jax), and it
compiles through Fractalyze's fork of [XLA](https://github.com/openxla/xla).
Both differ from upstream in the way that matters here: finite fields are
native dtypes, not emulated. Everywhere below, **FRX** and **XLA** mean those
forks.

OpenVM proves with **SWIRL** — a sumcheck-based proof system composing
LogUp-GKR (interactions), a batched ZeroCheck with univariate skip
(constraints), a stacked opening reduction, and a WHIR polynomial commitment —
as implemented by
[openvm-stark-backend](https://github.com/openvm-org/stark-backend) at tag
`v2.0.0`. This repo re-implements that prover on zorch, keeping
only the SWIRL-specific surface here and pushing every generic block upstream.

## Installation

**Python 3.11 on Linux x86_64, or macOS on Apple Silicon.** (`frxlib` ships a
cp311 wheel for those two platforms only — not 3.12/3.13, not Intel Macs.)

### CPU

```sh
pip install openvm-zorch
```

### GPU (CUDA 12)

```sh
pip install openvm-zorch 'frx[cuda12]' \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The extra index carries the CUDA plugin wheels, which are too large for PyPI's
per-file limit. It is not needed for the CPU tier.

### Verify

```sh
python -c "import frx, openvm_zorch; print(frx.devices()); print(openvm_zorch.__version__)"
```

`[CpuDevice(id=0)]` means the CPU tier; a CUDA install prints the GPU devices.

## Development

```sh
bazel test //...                              # hermetic, CPU by default
```

Install the git hooks with both stages named. Plain `pre-commit install` wires
only the `pre-commit` stage, which leaves the commit-message linter inactive —
a malformed commit message then sails through to CI:

```sh
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
a valid type, a lowercase summary with no trailing period, a header of at most
80 characters, and a body on everything but `docs`. The scope is the package the
change lives in — `commit`, `logup_gkr`, `logup_zerocheck`, `poseidon2`,
`stacked_reduction`, `whir` — or one of `prove`, `verify`, `verify_prove`,
`transcript`, `fields`, `poly_common`, `bench_common`, `release` for the modules
directly under `openvm_zorch/`. A change spanning several takes no scope. The
same linter runs in CI over every commit in a pull request and over the PR title.

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
`v2.0.0` release consumes. Config: BabyBear base field,
BabyBear⁴ challenges, Poseidon2 width-16 (`default_babybear_poseidon2_16`,
plonky3 `=0.4.3`).

## Documentation

See [`docs/`](https://github.com/fractalyze/openvm-zorch/blob/main/docs/README.md) for the full index — pipeline & terminology, and
development & benchmarking.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](https://github.com/fractalyze/openvm-zorch/blob/main/LICENSE)).
