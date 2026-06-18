# real-block-gen

Dumps a **real openvm guest block** as a golden fixture in the same layout as
[`tools/fixture-gen`](../fixture-gen)'s `--prove-out` (so `verify_prove` /
`prove_test` consume it), but from an actual guest-program execution instead of
the synthetic `Stage2Fixture`. This is the faithful "real block" input that
de-distorts the per-stage prover benchmark (the synthetic Fibonacci has one
narrow AIR and tiny non-scaling interactions, so it cannot rank prover stages —
see milestone #3 issue #49).

## Why it isn't a normal tool here

The dump taps the openvm SDK's per-segment `ProvingContext`
(`VmInstance::prove_continuations`, `crates/vm/src/arch/vm.rs`), so it must build
against the full openvm SDK. `openvm` (at `~/Workspace/openvm`) is **upstream
`openvm-org/openvm`**, not a fractalyze fork, so this bin can't be a normal
in-tree target. The source is vendored here; you apply it to a local openvm
checkout to run it.

## Apply + run

```sh
# 1. Drop the bin into a local openvm checkout:
cp dump_fixture.rs <openvm>/benchmarks/prove/src/bin/dump_fixture.rs
# add a [[bin]] entry (name = "dump_fixture") to benchmarks/prove/Cargo.toml,
# `serde_json.workspace = true` to its [dependencies]; and
# `features = ["test-utils"]` on the `openvm-stark-backend` dependency (the
# only public path to a recording prove + transcript log is
# `TestFixture::prove_from_transcript`, gated behind test-utils; the
# recursion crate already deps it the same way).

# 2. Build + run (SDK build is large the first time):
cd <openvm>
cargo run --profile fast --no-default-features -p openvm-benchmarks-prove --bin dump_fixture -- --out /tmp/real_fib
```

`--no-default-features` is **mandatory** for any run that dumps the byte-match
golden (`--ref-prove`, below). `openvm-benchmarks-prove`'s `default` enables
`parallel`, which unifies up to `openvm-stark-backend/parallel`, making the
LogUp PoW grind use rayon `find_any` — a *nondeterministic* witness. zorch's
grind is serial (smallest nonce), so a parallel reference `logup_pow_witness`
can never byte-match it, and the whole Fiat-Shamir cascade downstream
diverges. Serial (`parallel` off) the grind degrades to `Iterator::find` → the
smallest nonce, deterministic and matching zorch. This is the same reason
[`tools/fixture-gen`](../fixture-gen) builds `default-features = false`.
(`RAYON_NUM_THREADS=1` is **not** a safe substitute — rayon `find_any` doesn't
guarantee ascending search order even single-threaded.)

Output `/tmp/real_fib/`: `meta.json` + `inputs/{trace_<air>.npy, constraints_<air>.json}`.

## Outputs (`--ref-prove`): the byte-match golden

The `inputs/` above drive zorch's `prove()`; to also produce the reference
`outputs/` it byte-matches against, pass `--ref-prove`. It re-proves the same
tapped real ProvingContext with the reference *recording* engine
(`BabyBearPoseidon2RefEngine<DuplexSpongeRecorder>`, reusing the app proving key
directly), then walks the recorded transcript (the `walk_gkr/zerocheck/stacking/
whir_log` functions, vendored from fixture-gen) to extract the sampled
challenges, and dumps `outputs/` byte-for-byte as `gen_prove_fixture` does.

```sh
cargo run --profile fast --no-default-features -p openvm-benchmarks-prove --bin dump_fixture -- \
  --out /tmp/real_fib --ref-prove
```

`--no-default-features` is required here (see the note above) — without it the
reference grind is nondeterministic and `outputs/logup_pow_witness` (plus its
whole downstream cascade) is unreproducible. The serial witness is small (the
smallest valid nonce); zorch reproduces it exactly. The `--ref-prove` outputs
land in `<out>/ref-prove/outputs/`; `verify_prove` reads `<out>/outputs/`, so
symlink or copy them into place (`ln -sfn ref-prove/outputs <out>/outputs`).

The dumped `meta.json` also carries (under `--ref-prove`) the full vk-prelude
structure (`vk_prelude`: per vk position `present` / `is_required` /
`has_preprocessed` / `num_cached_mains` / `n_public_values`) and the raw
reference observation-log prefix (`obs_log`: canonical-u32 `values` + `samples`
through the grind boundary), so zorch's `CommitRound` can diff its prelude
transcript element-by-element instead of inferring divergence from cascaded
`MISMATCH` labels.

Two real-block subtleties the bin handles (vs the synthetic fixture):

- **Presence-gated prelude.** A real block has *absent* AIRs (unexercised chips);
  the Coordinator observes only a 1-entry present-flag for them, so the
  transcript `prelude_len` must gate the preprocessed/log_height/cached/pv terms
  on whether each AIR is present in the ProvingContext — not on its mere
  declaration in `vm_pk.per_air`.
- **Cached mains.** AIRs like `ProgramAir` keep their real columns in a *cached*
  main partition; the tapped cached `CommittedTraceData` can't cross backends
  (`CpuBackend` ≠ `CpuColMajorBackend` PcsData), so each cached main is
  re-committed via `stacked_commit` before the reference prove.

## Fixture format

Mirrors fixture-gen's `--prove-out`, with one change: interactions are
**expression-DAG-referenced**, not column indices (real openvm interactions are
`Interaction<Expr>`; the synthetic ones happened to be bare columns). Each
`meta.json` `airs[].interactions[]` is:

```json
{ "bus": <u16>, "count_weight": <u32>,
  "count_idx": <node index into constraints_<air>.json nodes>,
  "message_idxs": [<node indices into the same node DAG>] }
```

The zorch GKR input layer evaluates these `count`/`message` nodes via the
constraint-DAG evaluator (issue #51). Everything else — `trace_*.npy` (canonical
`<u4`, `(height, width)`), `constraints_*.json`, and the `meta` params/
`sorted_airs`/`vk_pre_hash` — matches fixture-gen byte-for-byte.

Pinned to the same reference as the rest of the repo: openvm-stark-backend
`v2.0.0-beta.2`.
