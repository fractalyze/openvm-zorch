# Development

Pure Python on JAX + the ZKX PJRT plugin. Bazel 9 (bzlmod). Tests default to
`JAX_PLATFORMS=cpu`.

```sh
bazel test //...                 # hermetic, sandboxed
# iterative dev outside Bazel:
export PYTHONPATH="$PWD:/abs/path/to/zorch"
```

A read-only worktree of the reference prover is expected at
`$DEVENV_ENVS_DIR/zorch/stark-backend` (see [`swirl-pipeline.md`](swirl-pipeline.md)).

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

A jax wheel bump must move `zk-dtypes` in lockstep: each wheel imports a
specific `zk_dtypes` ABI (e.g. `efinfo.modulus_low_coeffs` from 0.0.6,
`pallas_sf` from 0.0.7), and jax's metadata does not floor the version, so the
lock keeps the old `zk-dtypes` and every target dies at `import jax`. Bump the
`zk-dtypes` pin in `requirements.in` alongside the jax pins and re-lock.

## Running on GPU

`//openvm_zorch:verify_prove` is the entry point — the byte-match +
per-stage-timing runnable, openvm's sibling of sp1-zorch's `verify_prove_shard`.

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

## Recurring gotchas

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
  arguments, got list`). A flat `list[Array]` of 0-D scalars stacks directly,
  but to stack rows of scalars (a `list[list[Array]]`) into a matrix you must
  inner-stack each row first: `jnp.stack([jnp.stack(row) for row in rows])`.
  `jnp.pad`, by contrast, DOES work on extension dtypes (verified byte-exact).
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
- **Perf: an O(N) Python loop *assembling* an array from per-element device
  ops (slice / `.at[].set` / `concatenate`) is also a dispatch storm — and
  the tell is it runs *slower on GPU than CPU* (each op pays host↔device
  launch latency). Vectorize into one on-device op driven by a
  host-precomputed static index map. Use a `jnp.take` **gather**, NOT a
  `.at[idx].set` scatter:** XLA serializes a large GPU scatter into a
  pathologically slow kernel (a real-block commit assembly went 0.55s→**59s**
  as a scatter vs 0.55s→**0.037s** as the gather). Invert the scatter into a
  gather by precomputing, per *output* cell, its source index (a shared
  sentinel-0 source = a zero for unwritten cells). (PR #78 took commit
  stacking ~15× on GPU this way.)
