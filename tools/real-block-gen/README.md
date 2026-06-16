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
# and `serde_json.workspace = true` to its [dependencies].

# 2. Build + run (SDK build is large the first time):
cd <openvm>
cargo run --profile fast -p openvm-benchmarks-prove --bin dump_fixture -- --out /tmp/real_fib
```

Output `/tmp/real_fib/`: `meta.json` + `inputs/{trace_<air>.npy, constraints_<air>.json}`.

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
