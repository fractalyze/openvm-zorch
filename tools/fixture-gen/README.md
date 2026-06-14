# fixture-gen

Golden-fixture generator for openvm-zorch's byte-match tests. Runs the
openvm-stark-backend reference prover (pinned to tag `v2.0.0-beta.2`,
BabyBear / Poseidon2 width-16) on deterministic inputs and dumps every
intermediate as canonical (non-Montgomery) `u32`, which the Python tests
compare exactly.

## Regenerating the committed fixtures

Each `--*-out` flag writes one stage's vendored `testdata/` directory; they
are independent, so pass only the ones you need. To regenerate all of them:

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

`--prove-out` is the self-contained end-to-end fixture
(`testdata/prove`: `meta.json` + `inputs/` + `outputs/`) that
`prove_test.test_prove_production_params` byte-matches against.

## Scaling the prove fixture

The `--prove-out` fixture is micro-scale by default (Fibonacci 2^6 rows) — too
small to fill a GPU or to give a representative per-stage compute split. Four
environment variables scale it up; **defaults reproduce the committed fixture
byte-for-byte**, and they affect `--prove-out` only (every other generator
keeps the defaults so its golden values are unchanged):

| Var              | Default | Effect                                                       |
| ---------------- | ------- | ------------------------------------------------------------ |
| `FIB_LOG_HEIGHT` | `6`     | log2 of the tallest (Fibonacci) trace height                 |
| `L_SKIP`         | `4`     | skip-domain log size                                         |
| `N_STACK`        | `8`     | stacking knob — `log_stacked = N_STACK + L_SKIP`, so it sets the stacked-matrix size and the WHIR round count |
| `K_WHIR`         | `4`     | WHIR folding arity                                            |

`N_STACK` is the WHIR-size knob: stacked height is `2^(N_STACK + L_SKIP)`
(3 WHIR rounds at the default, 5 at `N_STACK=16` / stacked `2^20`). Grow
`FIB_LOG_HEIGHT` and `N_STACK` together — the scaled trace must still fit the
stacked matrix.

Large fixtures are generated **on demand**, not committed (a production-scale
trace is multi-megabyte). Generate into a scratch directory and point the
benchmark at it with `--fixture_dir`:

```sh
# Production-scale prove fixture (stacked 2^20, ~2^20 trace rows):
FIB_LOG_HEIGHT=20 N_STACK=16 cargo run --release -- --prove-out /tmp/prove_large
```

To confirm byte-match on a scaled fixture, generate over `testdata/prove`,
run `bazel test //openvm_zorch:prove_test`, then `git checkout` the directory
to restore the committed micro fixture.
