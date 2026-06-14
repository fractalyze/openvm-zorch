"""Per-stage wall-clock bench for ``prove`` on a production-shaped fixture.

Drives the full five-stage prover as its ``ProveChain`` (see ``prove_chain``)
and reports, per stage (commit / GKR / zerocheck / stacking / WHIR), the
compile time and the warm runtime separately. Backend is whatever JAX selects
— set ``JAX_PLATFORMS=cuda`` (+ ``CUDA_VISIBLE_DEVICES``) to bench on GPU, the
default ``cpu`` otherwise.

Each stage Round is wrapped in a ``_TimedRound`` that blocks on the stage's
output arrays before reading the clock — async dispatch makes unblocked
timings lie. Unlike the earlier monolithic baseline this DOES block between
stages, on purpose: that is what attributes cost per stage. Each stage is
independently jitted (the commit tail, Stage 4, Stage 5), so a stage's cold
time folds in *that stage's* compile and warm runs are its steady state, giving
per-stage ``compile ≈ cold − warm``. The cost of the attribution is that
serializing the stages removes cross-stage async overlap, so the summed total
here is an upper bound on the true end-to-end wall — do not compare it against
an unblocked baseline total.

    bazel run //openvm_zorch:prove_bench -- --runs 4
    bazel run //openvm_zorch:prove_bench -- --runs 4 --fixture_dir /path/to/fixture
"""

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from zk_dtypes import babybear_mont as F

from openvm_zorch.logup_gkr.input_layer import InteractionSpec
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.prove import AirInstance, SystemParams, prove_chain
from openvm_zorch.transcript import new_transcript
from openvm_zorch.whir.prover import WhirConfig
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.round import Round

_FLAGS = flags.FLAGS
flags.DEFINE_integer("runs", 3, "Total prove() runs; run 1 is cold, 2.. are warm.")
flags.DEFINE_string(
    "fixture_dir",
    None,
    "Directory of a prove fixture (meta.json + inputs/). Defaults to the "
    "committed testdata/prove; point at a generated fixture dir to bench a "
    "larger, production-scale instance.",
)

_PROVE = Path(__file__).parent / "testdata" / "prove"

# Friendly per-stage labels, keyed by the stage Round's class name.
_STAGE_LABELS = {
    "CommitRound": "commit",
    "GkrRound": "GKR",
    "ZeroCheckRound": "zerocheck",
    "StackingRound": "stacking",
    "WhirRound": "WHIR",
}


def _array_leaves(obj):
    """Flatten the jax arrays out of an arbitrary nested structure.

    A stage's output (carry, transcript, message) mixes plain ``@dataclass``
    objects — ``ProveCarry``, the proof messages — that are not registered JAX
    pytrees, so ``jax.tree_util`` (and therefore ``jax.block_until_ready``)
    cannot see the arrays inside them; blocking on them directly is a silent
    no-op that would stop the timer at dispatch rather than at compute
    completion. Walk the structure by hand instead.
    """
    if isinstance(obj, jax.Array):
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [
            a for f in dataclasses.fields(obj)
            for a in _array_leaves(getattr(obj, f.name))
        ]
    if isinstance(obj, (list, tuple)):
        return [a for x in obj for a in _array_leaves(x)]
    if isinstance(obj, dict):
        return [a for x in obj.values() for a in _array_leaves(x)]
    return []


class _TimedRound(Round):
    """Time one stage: run the inner Round, block on its output arrays, record
    the wall-clock under the stage's label. Blocking is mandatory — async
    dispatch returns before the device finishes, so an unblocked timing would
    attribute this stage's compute to the next timed section."""

    def __init__(self, inner: Round, records: dict[str, list[float]]) -> None:
        self._inner = inner
        self._records = records

    def __call__(self, carry, transcript):
        t0 = time.perf_counter()
        out = self._inner(carry, transcript)
        jax.block_until_ready(_array_leaves(out))
        dt = time.perf_counter() - t0
        label = _STAGE_LABELS.get(
            type(self._inner).__name__, type(self._inner).__name__
        )
        self._records.setdefault(label, []).append(dt)
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
        airs.append(
            AirInstance(
                trace=trace,
                dag=dag,
                interactions=tuple(
                    InteractionSpec(
                        bus=spec["bus"],
                        count_col=spec["count_col"],
                        count_neg=spec["count_neg"],
                        message_cols=tuple(spec["message_cols"]),
                    )
                    for spec in air["interactions"]
                ),
                public_values=tuple(air["public_values"]),
                constraint_degree=air["constraint_degree"],
                needs_next=air["needs_next"],
                is_required=air["is_required"],
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
    return params, meta["vk_pre_hash"], airs


def main(argv):
    del argv
    prove_dir = Path(_FLAGS.fixture_dir) if _FLAGS.fixture_dir else _PROVE
    params, vk_pre_hash, airs = _load_instance(prove_dir)
    sponge, comp = _poseidon2()

    backend = jax.default_backend()
    devices = jax.devices()
    heights = [int(a.trace.shape[0]) for a in airs]
    print(f"backend={backend} devices={devices}")
    print(
        f"fixture={prove_dir}  trace_heights={heights}  "
        f"whir_rounds={len(params.whir.num_queries)}"
    )

    # One chain, reused across runs (built from a list, so re-callable); only
    # the transcript is fresh per run. The carry is functional — stages return
    # a new one via ``replace`` — so the same initial carry seeds every run.
    chain, carry = prove_chain(sponge, comp, params, vk_pre_hash, airs)
    records: dict[str, list[float]] = {}
    chain.rounds = [_TimedRound(rnd, records) for rnd in chain.rounds]

    totals = []
    for i in range(_FLAGS.runs):
        t0 = time.perf_counter()
        chain(carry, new_transcript())
        dt = time.perf_counter() - t0
        totals.append(dt)
        tag = "cold" if i == 0 else "warm"
        print(f"run {i}: {dt:8.3f}s ({tag})")

    if _FLAGS.runs > 1:
        print(f"\n{'stage':<12} {'compile(s)':>11} {'warm(s)':>10}")
        for label, ts in records.items():  # insertion order == stage order
            warm_min = min(ts[1:])
            compile_est = ts[0] - warm_min
            print(f"{label:<12} {compile_est:>11.3f} {warm_min:>10.3f}")
        warm = totals[1:]
        print(
            f"\ntotal  cold={totals[0]:.3f}s  warm_min={min(warm):.3f}s  "
            f"warm_mean={sum(warm)/len(warm):.3f}s  "
            f"compile≈{totals[0]-min(warm):.3f}s"
        )


if __name__ == "__main__":
    app.run(main)
