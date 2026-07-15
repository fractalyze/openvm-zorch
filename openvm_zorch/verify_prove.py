"""Byte-match harness for the assembled ``prove`` chain -- a runnable.

Runs ``prove_chain`` (the ``ProveChain`` of commit -> LogUp-GKR -> zerocheck
-> stacked reduction -> WHIR) over a production-shaped fixture and seals the
assembled proof against the reference: every proof component is byte-matched
(canonical-u32 exact) against the fixture's ``outputs/`` dump, the same
assertion set as ``prove_test.test_prove_production_params``.

Unlike that test this is a GPU-capable **runnable**, not a unit test, and
that is the point: ``prove_test`` is backend-agnostic (no cuda deps, so it
runs on CPU only) and cannot confirm byte-exactness on GPU at production
scale -- the XLA CPU backend currently core-dumps at stacked 2^16 in
zerocheck round-0 (#32), and a cuda-dep'd target cannot even import on a
driverless CI box. This runnable deps the cuda plugin and runs on whatever
backend FRX selects, so it is the way to gate GPU byte-match at scale.

Each Stage is wrapped in a ``_TimedRound`` that prints its wall-clock
on every run, so the compile-vs-runtime split is visible alongside the
byte-match (proof messages are plain dataclasses, opaque to
``block_until_ready``, so block on their array leaves by hand). Wall-clock is
dominated by XLA GPU compiles, not kernel runtime; for the warm split set
``JAX_COMPILATION_CACHE_DIR`` to a per-toolchain directory so every run after
the first skips the compiles (leave it unset for byte-match gates).

    bazel run //openvm_zorch:verify_prove
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        bazel run //openvm_zorch:verify_prove -- --fixture_dir /path/to/fixture

Exits non-zero on any mismatch.
"""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import frx
import frx.numpy as jnp
import numpy as np
from absl import app, flags
from frx import lax
from zk_dtypes import babybear_mont as F

from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.prove import (
    AirInstance,
    CommitStage,
    GkrStage,
    Proof,
    StackingStage,
    SystemParams,
    WhirStage,
    ZeroCheckStage,
    prove_chain,
)
from openvm_zorch.transcript import new_transcript
from openvm_zorch.whir.prover import WhirConfig
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.round import Round

_FIXTURE_DIR = flags.DEFINE_string(
    "fixture_dir",
    None,
    "Directory of a prove fixture (meta.json + inputs/ + outputs/). Defaults "
    "to the committed testdata/prove; point at a generated fixture dir to "
    "byte-match a larger, production-scale instance.",
)

_BASELINE = flags.DEFINE_string(
    "baseline",
    None,
    "Optional native-prover baseline JSON (see docs/development.md). When "
    "set, the chain is run a second (warm) time and its per-stage _TimedRound "
    "sum is printed against the native e2e prove time, with the delta.",
)

_STOP_AFTER = flags.DEFINE_string(
    "stop_after",
    None,
    "Optional stage label (commit/GKR/zerocheck/stacking/WHIR) to truncate the "
    "chain after: run only the stages up to and including it, skip the (now "
    "partial) byte-match, and still time a warm pass. For per-stage profiling "
    "on a backend where a LATER stage's compile is intractable -- e.g. WHIR's "
    "2^22 ptxas blow-up on GPU at real-block scale -- so an earlier stage can "
    "be timed cold+warm in isolation. Pairs with OPENVM_ZC_PROFILE for the "
    "zerocheck sub-region split.",
)

_PROVE = Path(__file__).parent / "testdata" / "prove"

# Friendly per-stage labels, keyed by the stage class itself rather than its
# name: nothing here is covered by a test (this is a cuda-dep'd runnable, and
# tests stay backend-agnostic), so a name-keyed map would let a stage rename
# silently degrade --stop_after to "matched no stage" and drop the timing
# labels back to class names. Keyed by the class, a rename that misses this
# map cannot import.
_STAGE_LABELS = {
    CommitStage: "commit",
    GkrStage: "GKR",
    ZeroCheckStage: "zerocheck",
    StackingStage: "stacking",
    WhirStage: "WHIR",
}


def _rounds_through(rounds, stop_label):
    """The chain rounds up to and including the stage named ``stop_label``
    (``None`` keeps the whole chain). Matched against ``_STAGE_LABELS`` so the
    flag takes friendly names, not class names."""
    if stop_label is None:
        return list(rounds)
    for i, rnd in enumerate(rounds):
        if _STAGE_LABELS.get(type(rnd)) == stop_label:
            return list(rounds[: i + 1])
    choices = sorted(set(_STAGE_LABELS.values()))
    raise ValueError(
        f"--stop_after={stop_label!r} matched no stage; pick from {choices}"
    )


def _array_leaves(obj):
    """Flatten the FRX arrays out of an arbitrary nested structure.

    A stage's output (carry, transcript, message) mixes plain ``@dataclass``
    objects -- ``ProveCarry``, the proof messages -- that are not registered
    FRX pytrees, so ``frx.tree_util`` (and therefore ``frx.block_until_ready``)
    cannot see the arrays inside them; blocking on them directly is a silent
    no-op that would stop the timer at dispatch rather than at compute
    completion. Walk the structure by hand instead.
    """
    if isinstance(obj, frx.Array):
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [
            a
            for f in dataclasses.fields(obj)
            for a in _array_leaves(getattr(obj, f.name))
        ]
    if isinstance(obj, (list, tuple)):
        return [a for x in obj for a in _array_leaves(x)]
    if isinstance(obj, dict):
        return [a for x in obj.values() for a in _array_leaves(x)]
    return []


class _TimedRound(Round):
    """Print each stage's wall-clock so the compile-vs-runtime split is visible
    on every run. Blocking is mandatory -- async dispatch returns before the
    device finishes, so an unblocked timing would attribute this stage's
    compute to the next timed section; the message is a plain dataclass, opaque
    to ``block_until_ready``, so block on its array leaves by hand.

    When a ``record`` dict is passed, each call also stores ``label -> seconds``
    so the caller can sum the per-stage warm runtime for the ``--baseline``
    comparison."""

    def __init__(self, inner: Round, record: dict | None = None) -> None:
        self._inner = inner
        self._record = record

    def __call__(self, carry, transcript):
        t0 = time.monotonic()
        out = self._inner(carry, transcript)
        frx.block_until_ready(_array_leaves(out))
        dt = time.monotonic() - t0
        label = _STAGE_LABELS.get(type(self._inner), type(self._inner).__name__)
        if self._record is not None:
            self._record[label] = dt
        print(f"[stage {label}] {dt:.1f}s", flush=True)
        return out


def _poseidon2():
    perm = Poseidon2(babybear16_params())
    return (
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _load_instance(prove_dir):
    """Mirror prove_test.test_prove_production_params input construction."""
    meta = json.loads((prove_dir / "meta.json").read_text())
    pm = meta["params"]
    airs = []
    for air in meta["airs"]:
        air_idx = air["air_idx"]
        trace = jnp.array(
            np.load(prove_dir / "inputs" / f"trace_{air_idx}.npy"), dtype=F
        )
        dag = ConstraintsDag.from_json(
            json.loads(
                (prove_dir / "inputs" / f"constraints_{air_idx}.json").read_text()
            )
        )
        cached_mains = tuple(
            jnp.array(
                np.load(prove_dir / "inputs" / f"cached_{air_idx}_{k}.npy"), dtype=F
            )
            for k in range(air.get("num_cached_mains", 0))
        )
        airs.append(
            AirInstance(
                trace=trace,
                dag=dag,
                public_values=tuple(air["public_values"]),
                constraint_degree=air["constraint_degree"],
                needs_next=air["needs_next"],
                is_required=air["is_required"],
                cached_mains=cached_mains,
                air_idx=air_idx,
            )
        )
    params = SystemParams(
        l_skip=pm["l_skip"],
        n_stack=pm["n_stack"],
        log_blowup=pm["log_blowup"],
        logup_pow_bits=pm["logup_pow_bits"],
        max_constraint_degree=pm["max_constraint_degree"],
        whir=WhirConfig(
            k=pm["k_whir"],
            num_queries=meta["num_queries"],
            mu_pow_bits=pm["mu_pow_bits"],
            folding_pow_bits=pm["folding_pow_bits"],
            query_phase_pow_bits=pm["query_phase_pow_bits"],
        ),
    )
    return params, meta["vk_pre_hash"], airs, meta.get("obs_log")


def _ef_limbs(x) -> np.ndarray:
    """Canonical-u32 limbs of a BabyBear⁴ array, shape (..., 4)."""
    return np.asarray(lax.bitcast_convert_type(jnp.atleast_1d(x), F).astype(jnp.uint32))


def _to_u32(x) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(x, F).astype(jnp.uint32))


def check_match(label: str, got, want) -> bool:
    """The byte-match runnable's OK/MISMATCH line: compare a value against its
    dump reference and print the verdict (shapes on mismatch)."""
    if isinstance(got, (int, np.integer)):
        ok = int(got) == int(want)
    else:
        # array_equal, not all(got == want): a shape divergence must read as a
        # mismatch, and broadcasting would equate e.g. (1,) with (1, 1).
        ok = bool(np.array_equal(got, want))
    print(f"{'OK ' if ok else 'MISMATCH'} {label}", flush=True)
    if not ok and not isinstance(got, (int, np.integer)):
        print(f"  shapes: got {np.shape(got)} want {np.shape(want)}")
    return ok


def _byte_match(proof: Proof, out: Path) -> bool:
    """Byte-match every proof component against the reference dump in ``out``.

    The component set is kept in lockstep with
    ``prove_test.test_prove_production_params`` so a GPU run here checks exactly
    what the backend-agnostic CPU test checks. Accumulates so every mismatch is
    reported, not just the first."""
    bcp = proof.batch_constraint_proof
    sp = proof.stacking_proof
    wp = proof.whir_proof
    # Stage 1 + 2.
    ok = check_match(
        "common_main_commit",
        _to_u32(proof.common_main_commit),
        np.load(out / "common_main_commit.npy"),
    )
    ok &= check_match(
        "logup_pow_witness",
        int(_to_u32(jnp.atleast_1d(proof.logup_pow_witness))[0]),
        int(np.load(out / "logup_pow_witness.npy")[0]),
    )
    ok &= check_match(
        "gkr.q0_claim",
        _ef_limbs(proof.gkr_proof.q0_claim)[0],
        np.load(out / "q0_claim.npy"),
    )
    ok &= check_match("xi", _ef_limbs(jnp.stack(proof.xi)), np.load(out / "xi.npy"))
    # Stage 3.
    ok &= check_match(
        "zc.lambda", _ef_limbs(bcp.lambda_)[0], np.load(out / "zc_lambda.npy")
    )
    ok &= check_match(
        "zc.s0_coeffs",
        _ef_limbs(jnp.stack(bcp.univariate_round_coeffs)),
        np.load(out / "zc_s0_coeffs.npy"),
    )
    ok &= check_match("zc.r", _ef_limbs(jnp.stack(bcp.r)), np.load(out / "zc_r.npy"))
    # Stage 4.
    ok &= check_match(
        "st.lambda", _ef_limbs(sp.lambda_)[0], np.load(out / "st_lambda.npy")
    )
    ok &= check_match("st.u", _ef_limbs(jnp.stack(sp.u)), np.load(out / "st_u.npy"))
    ok &= check_match(
        "st.openings_c0",
        _ef_limbs(sp.stacking_openings[0]),
        np.load(out / "st_openings_c0.npy"),
    )
    # Stage 5.
    ok &= check_match(
        "whir.mu_pow_witness",
        _to_u32(jnp.atleast_1d(wp.mu_pow_witness)),
        np.load(out / "whir_mu_pow_witness.npy"),
    )
    ok &= check_match("whir.mu", _ef_limbs(wp.mu)[0], np.load(out / "whir_mu.npy"))
    want_sumcheck = np.load(out / "whir_sumcheck_polys.npy")
    for j, evals in enumerate(wp.whir_sumcheck_polys):
        ok &= check_match(f"whir.sumcheck[{j}]", _ef_limbs(evals), want_sumcheck[j])
    ok &= check_match(
        "whir.codeword_commits",
        _to_u32(jnp.stack(wp.codeword_commits)),
        np.load(out / "whir_codeword_commits.npy"),
    )
    ok &= check_match(
        "whir.ood_values",
        _ef_limbs(jnp.stack(wp.ood_values)),
        np.load(out / "whir_ood_values.npy"),
    )
    ok &= check_match(
        "whir.folding_pow_witnesses",
        _to_u32(jnp.stack(wp.folding_pow_witnesses)),
        np.load(out / "whir_folding_pow_witnesses.npy"),
    )
    ok &= check_match(
        "whir.query_phase_pow_witnesses",
        _to_u32(jnp.stack(wp.query_phase_pow_witnesses)),
        np.load(out / "whir_query_phase_pow_witnesses.npy"),
    )
    ok &= check_match(
        "whir.final_poly",
        _ef_limbs(wp.final_poly),
        np.load(out / "whir_final_poly.npy"),
    )
    return ok


def _compare_baseline(baseline_path: str, params, stage_times: dict) -> None:
    """Print the zorch per-stage warm sum against the native e2e baseline.

    ``stage_times`` is the warm-run ``label -> seconds`` map captured by the
    second (compiled) chain pass. The baseline JSON is the native source of
    truth from ``docs/development.md``; its params must match the fixture's
    or the comparison is meaningless, so a mismatch is loud (the run still
    prints, but flagged)."""
    # Resolve a relative path against the workspace root so the documented
    # ``--baseline openvm_zorch/testdata/baseline/...`` works under ``bazel run``
    # (whose cwd is the runfiles tree, not the workspace) -- bazel sets
    # BUILD_WORKSPACE_DIRECTORY. An absolute path or a plain ``python`` run is
    # used as given.
    p = Path(baseline_path)
    if not p.is_absolute() and not p.exists():
        ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if ws and (Path(ws) / p).exists():
            p = Path(ws) / p
    baseline = json.loads(p.read_text())
    bp = baseline["params"]
    fixture = {
        "l_skip": params.l_skip,
        "n_stack": params.n_stack,
        "k_whir": params.whir.k,
    }
    mismatched = {k: (v, bp.get(k)) for k, v in fixture.items() if bp.get(k) != v}

    native = baseline["prove_e2e_s"]["warm_min"]
    zorch_sum = sum(stage_times.values())
    breakdown = " + ".join(f"{k} {t:.2f}" for k, t in stage_times.items())

    print("\n=== baseline comparison ===", flush=True)
    print(f"baseline:     {p}", flush=True)
    print(
        f"  reference:  {baseline['reference']}  platform={baseline['platform']}",
        flush=True,
    )
    if mismatched:
        print(
            "  WARNING: param mismatch (fixture vs baseline): "
            + ", ".join(f"{k}={v[0]}!={v[1]}" for k, v in mismatched.items())
            + " -- comparison is not apples-to-apples",
            flush=True,
        )
    print(f"native e2e prove (warm min): {native:.3f}s", flush=True)
    print(f"zorch per-stage warm sum:    {zorch_sum:.3f}s  ({breakdown})", flush=True)
    if zorch_sum > 0 and native > 0:
        if zorch_sum <= native:
            print(
                f"  zorch is {native / zorch_sum:.2f}x FASTER than native", flush=True
            )
        else:
            print(
                f"  zorch is {zorch_sum / native:.2f}x SLOWER than native", flush=True
            )


def main(argv) -> None:
    del argv
    prove_dir = Path(_FIXTURE_DIR.value) if _FIXTURE_DIR.value else _PROVE
    params, vk_pre_hash, airs, obs_log = _load_instance(prove_dir)
    sponge, comp = _poseidon2()

    heights = [int(a.trace.shape[0]) for a in airs]
    print(f"backend={frx.default_backend()} devices={frx.devices()}")
    print(
        f"fixture={prove_dir}  trace_heights={heights}  "
        f"whir_rounds={len(params.whir.num_queries)}"
    )

    # Per-stage timings print as the chain runs (see _TimedRound). The first
    # (cold) pass carries the reference observation-log so CommitStage diffs the
    # prelude element-by-element (issue #59); the warm pass below omits it.
    chain, carry = prove_chain(sponge, comp, params, vk_pre_hash, airs, obs_log=obs_log)
    chain.rounds = [
        _TimedRound(rnd) for rnd in _rounds_through(chain.rounds, _STOP_AFTER.value)
    ]

    t0 = time.monotonic()
    _, _, msgs = chain(carry, new_transcript())
    print(f"chain run: {time.monotonic() - t0:.1f}s")

    if _STOP_AFTER.value is not None:
        print(f"[stopped after {_STOP_AFTER.value}; byte-match skipped]", flush=True)
    else:
        root, gkr, bcp, stacking_proof, whir_proof = msgs
        # Assemble the Proof exactly as prove() does, then byte-match it.
        proof = Proof(
            common_main_commit=root,
            logup_pow_witness=gkr.logup_pow_witness,
            gkr_proof=gkr.gkr_proof,
            xi=gkr.xi,
            batch_constraint_proof=bcp,
            stacking_proof=stacking_proof,
            whir_proof=whir_proof,
        )
        if not _byte_match(proof, prove_dir / "outputs"):
            sys.exit(1)
        print("prove chain byte-match: ALL OK")

    # The first chain run above pays the XLA compile; for the baseline
    # comparison we want warm per-stage runtime, so run the (now-compiled)
    # chain a second time and capture each stage's wall-clock. --stop_after
    # alone also triggers it (per-stage profiling needs no baseline).
    if _BASELINE.value or _STOP_AFTER.value is not None:
        stage_times: dict = {}
        warm_chain, warm_carry = prove_chain(sponge, comp, params, vk_pre_hash, airs)
        warm_chain.rounds = [
            _TimedRound(rnd, record=stage_times)
            for rnd in _rounds_through(warm_chain.rounds, _STOP_AFTER.value)
        ]
        print("\n[warm pass]", flush=True)
        warm_chain(warm_carry, new_transcript())
        # The e2e sum-vs-native comparison is only meaningful for the full chain.
        if _BASELINE.value and _STOP_AFTER.value is None:
            _compare_baseline(_BASELINE.value, params, stage_times)


if __name__ == "__main__":
    app.run(main)
