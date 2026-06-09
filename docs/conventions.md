# Conventions

The rules below are the deltas this repo adds on top of zorch's
[conventions](https://github.com/fractalyze/zorch/blob/main/docs/conventions.md);
everything there (Protocol seams, frozen dataclass pytrees, full type
annotations, fusion discipline) applies here unchanged.

## Comments say WHY, not WHAT

Code communicates intent on its own; a comment earns its place only by
recording something the code cannot — a protocol obligation, a reference-prover
quirk being matched, a non-obvious ordering constraint.

## External OpenVM references are pinned permalinks

Any claim about what the reference prover does cites a permalink pinned to the
target tag, e.g.

```
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/stacked_pcs.rs#L116
```

Unpinned `main` links rot silently as `develop-v2` moves; a pinned link stays
true forever. The repo-wide pin is `v2.0.0-beta.2` (`f6a84921`).

## Byte-match is the correctness bar

A stage is done when it reproduces the reference prover's bytes (canonical
`u32`), not when a self-consistent round-trip passes. Round-trip tests are
still welcome as fast development guards — they are necessary, not sufficient.

## Naming follows the reference

Module and variable names track openvm-stark-backend's vocabulary (`l_skip`,
`n_stack`, `log_blowup`, `k_whir`, `rows_per_query`, "stacked", "prismalinear")
so a reader can diff this repo against the Rust side without a translation
table. zorch-side names stay agnostic; the mapping lives in
[`swirl-pipeline.md`](swirl-pipeline.md).
