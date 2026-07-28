# Conventions

How this repo models a protocol: what a claim may say, where instance-varying
data lives, and which steps are roles rather than functions. These were derived
in [sp1-zorch#301](https://github.com/fractalyze/sp1-zorch/pull/301) — each one
from a concrete defect found while porting a prover onto zorch's composition
roles — and are recorded here so they hold without being rediscovered.

zorch's own definitions of stage, round, committer and shared function are
**not** restated here. They live in
[`docs/composition/stage-composition.md`](https://github.com/fractalyze/zorch/blob/main/docs/composition/stage-composition.md);
its "Which one is it?" table is the decision procedure. What follows is only
what is this repo's.

## 1. A claim states a proposition

A `*Claim` docstring opens with a sentence that could be true or false, not an
enumeration of its fields. A `*Witness` says what it is a witness *for*; a
`*Proof` names which claim it discharges and into what. Read down
`openvm_zorch/prove.py`, the composition should state itself as a chain of
reductions: system claim → LogUp fraction-sum claim at ξ → per-column opening
claims at `r` → stacked-column opening claims at `u` → nothing left to prove.

## 2. The claim carries what varies per block; the roles carry what does not

`SystemClaim` holds the per-AIR log heights, public values and presence flags —
they change from block to block and the verifier needs every one of them.
Constraint DAGs, common-main column counts, rotation flags and the security
parameters are fixed by the circuit and configure the roles instead.

Name a type after the axis it holds. `AirShape` (public per-AIR structure) and
`SystemShape` (the protocol sizes the heights determine) are separate types
because they answer different questions; a single "shape" type holding both
would hide the distinction.

**Recorded deviation.** `AirInstance` still carries the trace *and* the
circuit-fixed `dag` / `constraint_degree` / `needs_next`. Splitting it would
churn every fixture loader for no correctness gain, so it stays the witness
carrier; the verifier's `AirVk` is the corresponding vk-side type.

## 3. Hold values, not their serialization

Never store a pre-serialized transcript blob as claim data. The prelude absorb
stream is *derived* from the claim by `prelude_observations`, so the
transcript's view and the structural view cannot disagree.

The same rule governs derived quantities. `SystemShape.of` is the single
derivation of the stacking order and of `n_logup` / `n_max` / `n_global`; the
prover reaches it from trace heights and the verifier from verifying keys, but
they run the same code, so they cannot drift.

## 4. Not everything in the sequence is a stage or a round

See zorch's table for the definitions. In this repo:

- the cached-main commitments are a **committer** — they run before any claim
  exists and their output is prover data no reduced claim carries, which is
  also why they are taken in build scope rather than inside the timed prove;
- `bind_commitment` is a **shared function** both roles call — it only absorbs;
- LogUp-GKR, zerocheck and the stacked reduction are **stages**.

Test: if every call site passes `None` for the carry and throws it away, it is
not a round.

## 5. The PCS's commit and open are two halves of one role

`StackedWhirPcs` owns both. They sit far apart because Fiat-Shamir requires it —
the commitment must bind the transcript before LogUp-GKR draws a challenge, and
the opening needs the point the stacked reduction produces. `StackedCommitData`
names what crosses between them.

That data is **not** prover-only: the common-main root reaches the wire as the
proof's commitment, and each Merkle tree's opened rows and paths ride the WHIR
proof. Do not describe it as prover-side without checking.

Do not add a prover-only output channel to `ProveResult` to make a committer
into a stage — it would have no `VerifyResult` counterpart, and the
prover/verifier symmetry is what makes the composition checkable.

## 6. Point at zorch for zorch's concepts

A doc here covers only what is this repo's: the stacked PCS layout, the
query-strided Merkle structure, the prismalinear RS message, the byte-match.
Anything zorch defines is linked, not copied — local copies drift silently, as
the retired `ProveChain` / `Bridge` vocabulary did.

## 7. Make confusable states unrepresentable

Keyword-only fields on any type whose field shape has a twin. `AirShape` is
`kw_only` because `log_height` and `air_idx` are both plain ints naming
different axes, and a positional swap would put a trace height in a
verifying-key slot with no type error to catch it.

## 8. Do not mirror an upstream name when it collides locally

The reference numbers its protocol "Stage 1..5" and zorch calls a claim
reduction a stage — the same word for two different things, and three of the
five reference stages are zorch stages while the first and last are the halves
of one PCS role. The mapping is recorded in
[`architecture.md`](architecture.md) rather than resolved by renaming, because
[the byte-match](../CLAUDE.md) depends on this repo's names tracking the
reference's.

## 9. Enforce with tooling, and know what the tooling misses

`pre-commit` runs black, ruff and mypy over the whole package. mypy is what
catches a wrong-arity call or a crossed claim type.

**`bazel test` does not execute `py_binary` targets.** `//openvm_zorch:verify_prove`
and `//openvm_zorch/logup_gkr:bench_logup_gkr_phases` are runnables, so the
suite never runs them and a defect can live there indefinitely. Run them
explicitly after changing anything they construct:

```bash
bazel run //openvm_zorch:verify_prove          # must print "byte-match: ALL OK"
bazel run //openvm_zorch/logup_gkr:bench_logup_gkr_phases -- \
  --fixture_dir openvm_zorch/testdata/prove
```
