# Project context for Claude Code

- **Overview & quick start:** [`README.md`](README.md)
- **Conventions:** [`docs/conventions.md`](docs/conventions.md) — comments (why-not-what), external OpenVM references (pinned permalinks).
- **Pipeline & terminology:** [`docs/swirl-pipeline.md`](docs/swirl-pipeline.md) — SWIRL stages as Round compositions, openvm-stark-backend vocabulary mapping.
- **Native baseline:** [`docs/native-baseline.md`](docs/native-baseline.md) — the native (Rust) prover wall-clock milestone #4 measures against, what is timed, and the CPU / GPU (`--features cuda`) split.

## One non-negotiable

- **OpenVM/SWIRL-specific only.** This repo holds the SWIRL glue (stacked PCS
  layout, query-strided Merkle structure, prismalinear RS message, byte-match
  against openvm-stark-backend). Anything scheme- or zkVM-agnostic belongs
  upstream in `zorch`, not here. If a generic block is missing, add it to
  `zorch` and depend on it — do not fork it into `openvm-zorch`.

## Reference target

The proving scheme is **SWIRL** (LogUp-GKR + batched ZeroCheck + stacked
opening reduction + WHIR), as implemented by `openvm-stark-backend` at tag
**`v2.0.0-beta.2`** (commit `f6a84921e46a7df9796d41dfdfe69f0658ad74b5`) — the
same pin the openvm `v2.0.0-beta.2` release consumes. SWIRL lives on
stark-backend's `develop-v2` branch lineage, NOT `main` (main is still
plonky3/FRI). Config: BabyBear base field, BabyBear⁴ challenges, Poseidon2
width-16 (`default_babybear_poseidon2_16`, plonky3 `=0.4.1`).

A read-only worktree of the reference is expected at
`$DEVENV_ENVS_DIR/zorch/stark-backend` (see `docs/swirl-pipeline.md`).

## Dependency on zorch

`zorch` is a Bazel module, pinned in `MODULE.bazel` via `git_override` to a
main commit. For dev against a local working copy, add to `.bazelrc.user`
(gitignored — holds an absolute path):

```
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

Bump the pin when you need newer `zorch` blocks; keep it on `main` commits so
CI is reproducible. A pin bump's commit range often also moves `zorch`'s own
zkx wheel pins (`jax`/`jaxlib`/`zkx-cuda-pjrt`) — that does **not** force a
matching bump here. This repo pins those wheels independently in
`requirements.in`, on a separate pip hub from `zorch`'s, and the two are
allowed to skew. Only touch `requirements.in` if a build actually breaks;
otherwise the pin bump is a one-line `MODULE.bazel` change. (To validate the
pin the way CI resolves it, temporarily disable the `.bazelrc.user`
`--override_module=zorch` line — otherwise the build silently uses your local
checkout, not the pinned commit.)

## Development environment

Pure Python on JAX + the ZKX PJRT plugin. Bazel 9 (bzlmod). Tests default to
`JAX_PLATFORMS=cpu`.

```sh
bazel test //...                 # hermetic, sandboxed
# iterative dev outside Bazel:
export PYTHONPATH="$PWD:/abs/path/to/zorch"
```

Running on GPU (`//openvm_zorch:verify_prove` is the entry point — the
byte-match + per-stage-timing runnable, openvm's sibling of sp1-zorch's
`verify_prove_shard`):

- A target only sees the GPU if it deps **both** `requirement("jax_cuda12_plugin")`
  and `requirement("zkx_cuda_pjrt")`; without them jax **silently falls back to
  CPU**. Run with `JAX_PLATFORMS=cuda` (not `gpu`, which also inits rocm and
  dies) so a missing plugin hard-errors instead of silently using CPU.
- Those plugin `.so`s require **`libcuda` at import**, so a cuda-dep'd target
  cannot even import on a driverless machine. Therefore tests stay
  **backend-agnostic** (no cuda deps) so `bazel test //...` runs on any
  machine; GPU lives only in `bazel run` tools like `verify_prove`.
- `Proof` (and its stage sub-proofs) are plain dataclasses, not registered
  pytrees, so `jax.block_until_ready(proof)` is a **no-op** — walk the tree
  and block on the array leaves to time the device honestly.

Gotchas that recur across stages:

- zkx-native `lax.fft` accepts at most **2-D input** on field dtypes —
  flatten all leading batch axes before any NTT call and reshape after
  (first hit in `openvm_zorch/commit/rs_message.py`; Stages 3/5 are
  DFT-heavy and will hit it again).
- The Rust reference sizes buffers from *lifted cell counts*, not occupied
  extent (e.g. the stacked matrix can end in an all-zero committed column).
  When a byte-match fails at a hash, first suspect a shape/padding delta,
  not the hash params.
- zkx-native `jax.numpy` is a subset: `jnp.roll` does not exist (first hit
  in Stage 4's rotation kernel — use
  `jnp.concatenate([a[-1:], a[:-1]])`), and `jnp.arange` iota is
  unsupported for extension dtypes (zorch builds domains via `jnp.stack`
  of scalars). When an attribute error names a jnp function, reach for a
  concat/stack equivalent before suspecting your logic. Also: `jnp.stack`
  (and `jnp.concatenate`) require each element to ALREADY be an array — they
  do NOT `asarray` a nested Python list (`stack requires ndarray or scalar
  arguments, got list`). To stack a `list[Array]` of 0-D scalars into a row,
  wrap it: `jnp.stack([jnp.stack(scalar_list) for ...])`. `jnp.pad`, by
  contrast, DOES work on extension dtypes (verified byte-exact).
- **Perf: a host-int weight loop (a `pow()` nest building a constant
  matrix, contracted into field cells with scalar `acc += w*cell` adds) is
  a dispatch storm, not a FLOP cost.** It dominates eagerly. Fix: build the
  constant weight matrix once (`@lru_cache` keyed on `l_skip`/`num_cosets`,
  as a field array) and replace the scalar nest with one broadcast-multiply
  + a **trailing-axis** `.sum` (mid-axis EF reduce faults under jit; keep
  the contracted axis last). This is eager-fast and jit-fusable. Do NOT use
  `jnp.dot`/`@`/`tensordot` — they mis-lower under `jax.jit` on this fork
  (see `zorch/fusion.py`, `zorch/pcs/whir/_math.py`). And do NOT wrap a
  scalar-list polynomial (`_conv` over 0-D coeffs) in `jax.jit` directly —
  hundreds of pytree-leaf scalars regress; vectorize into arrays first.
  (PR #14 took round-0 prism 4.26→1.0s, whole prove −29%, this way.)

## Byte-match

The prover byte-matches the openvm-stark-backend reference prover.
Golden fixtures are generated by the Rust harness in `tools/fixture-gen`
(cargo, pinned to the reference tag) and vendored per module under
`testdata/`. Values are dumped as canonical (non-Montgomery) `u32`; compare
exactly, no tolerances.

`verify_prove`'s byte-match runs only AFTER the full chain completes — the
per-stage `[stage …] Ns` lines are timing/liveness, NOT byte-match: a stage
printing its time means it didn't crash, not that it matched. The
`OK`/`MISMATCH` report appears once at the end and prints no got-values
(only shapes on a shape divergence). To localize a transcript/prelude
divergence (the first `MISMATCH` is the earliest one; everything after
cascades), instrument the value or dump the reference observation-log
prefix and diff element-by-element — inferring from `MISMATCH` labels alone
can't tell "fix had no effect" from "value changed but still wrong".

The first `MISMATCH` typically surfaces one *sample* AFTER the actual bug:
per-column / per-commitment **openings are observed into the transcript but
not individually byte-checked**, so a missing or wrong opening at stage N's
tail only shows up at the first `sample_ext` of stage N+1, while every
checked value of stage N stays green. When a divergence lands on a stage's
first sampled value (`st.lambda`, `whir.mu`/`whir.sumcheck[0]`, …) with the
prior stage fully green, suspect an unchecked observe at the prior stage's
tail (a cached/partition opening), not the stage that reports the mismatch.

Per-stage fixture pattern (established for Stage 2, reuse for 3–5):

- Run the real prover end-to-end with `DuplexSpongeRecorder` and dump the
  full transcript log; per-stage values are extracted by a structural log
  walk that asserts observe/sample flags as it goes — the walk doubles as
  validation of the transcript-sequence understanding.
- Self-validate reconstructions before dumping: rebuild the stage input in
  fixture-gen and replay the stage's `pub` entry point through
  `ReadOnlyTranscript::new(&log, idx)` — a drift fails at generation time,
  not in a Python test. Needs `debug-assertions = true` in release.
- `ReadOnlyTranscript` CANNOT replay across a PoW grind: the witness search
  observes non-matching candidates and trips the log asserts. For a stage
  whose entry point grinds, rebuild the transcript state instead (feed the
  log's observe prefix into a real recorder sponge) and rerun — the serial
  grind re-finds the same witness (Stage-3 pattern in
  `gen_zerocheck_fixture`). Grind-free stages can use `ReadOnlyTranscript`.
- plonky3 field serde emits MONTGOMERY-form u32, not canonical — never
  `serde_json` a struct holding field constants (vk DAGs etc.) into a
  fixture; hand-roll the dump via `as_canonical_u32()`.
- Keep `default-features = false` on the backend deps: the PoW grind uses
  rayon `find_any`, which picks a nondeterministic witness under
  `parallel` and would break fixture reproducibility.
