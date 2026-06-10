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
| 2. LogUp-GKR | Interaction fractional sumcheck | `openvm_zorch/logup_gkr` | byte-matches reference |
| 3. ZeroCheck | Batched constraints, univariate skip + multivariate sumcheck | `openvm_zorch/logup_zerocheck` | byte-matches reference |
| 4. Stacked reduction | Column openings → stacked matrix openings | `openvm_zorch/stacked_reduction` | byte-matches reference |
| 5. WHIR opening | μ-batched sumcheck folds + per-round RS commits, OOD + query phase, PoW grinds | `openvm_zorch/whir` | byte-matches reference |

All five stages replay the reference prover's full 945-entry transcript log
end-to-end on the shared fixture instance. `openvm_zorch/prove.py` composes
them into a single `prove()` driven by raw inputs only (traces, constraint
DAGs, interaction specs, vk pre-hash, params) — no recorded log, PoW grinds
run natively (`prove_test.py`). That test also runs the prover at
production-shaped params (`l_skip=4`, `k_whir=4`) against a self-contained
fixture, where every short trace takes the lifting/striding path the test
params never exercise.

`openvm_zorch/verify.py` is the matching verifier: from the proof + verifying
key alone (no traces) it re-derives every Fiat-Shamir challenge and checks
each stage's algebraic relation, including WHIR's query-phase Merkle-path
verification. `verify_test.py` confirms an honest proof verifies and that a
proof tampered in any one stage is rejected.

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

`openvm-stark-backend` tag `v2.0.0-beta.2`
(`f6a84921e46a7df9796d41dfdfe69f0658ad74b5`), BabyBear + Poseidon2 width-16
(plonky3 `=0.4.1`). See [`CLAUDE.md`](CLAUDE.md) and
[`docs/swirl-pipeline.md`](docs/swirl-pipeline.md).
