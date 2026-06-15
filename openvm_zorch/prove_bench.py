"""Wall-clock baseline for ``prove`` on the production-shaped fixture.

Drives the full five-stage prover from the self-contained ``testdata/prove``
instance (the same inputs as ``prove_test.test_prove_production_params``) and
reports cold vs warm wall time. The cold run folds in XLA compilation; warm
runs are steady-state runtime. Backend is whatever JAX selects — set
``JAX_PLATFORMS=cuda`` (+ ``CUDA_VISIBLE_DEVICES``) to bench on GPU, the
default ``cpu`` otherwise.

Each run blocks on every array leaf of the proof (``Proof`` is a plain
dataclass, not a registered pytree, so blocking on it directly is a no-op)
so the wall time is honest end-to-end; we deliberately do NOT block between
stages (per-stage blocking serializes compile against compute and distorts
the attribution).

    bazel run //openvm_zorch:prove_bench -- --runs 4
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
from openvm_zorch.prove import AirInstance, SystemParams, prove
from openvm_zorch.transcript import new_transcript
from openvm_zorch.whir.prover import WhirConfig
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_FLAGS = flags.FLAGS
flags.DEFINE_integer("runs", 3, "Total prove() runs; run 1 is cold, 2.. are warm.")
flags.DEFINE_string(
    "fixture",
    None,
    "Directory of a prove fixture (meta.json + inputs/). Defaults to the "
    "committed testdata/prove; point at a fixture-gen --prove-out dir to bench "
    "a larger instance (e.g. N_STACK-scaled).",
)

_PROVE = Path(__file__).parent / "testdata" / "prove"


def _array_leaves(obj):
    """Flatten the jax arrays out of an arbitrary nested structure.

    ``Proof`` and its stage sub-proofs are plain ``@dataclass`` objects, not
    registered JAX pytrees, so ``jax.tree_util`` — and therefore
    ``jax.block_until_ready`` — cannot see the arrays inside them; blocking on
    the proof directly is a silent no-op that would stop the timer at dispatch
    rather than at compute completion. Walk the structure by hand instead.
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
    prove_dir = Path(_FLAGS.fixture) if _FLAGS.fixture else _PROVE
    params, vk_pre_hash, airs = _load_instance(prove_dir)
    sponge, comp = _poseidon2()

    backend = jax.default_backend()
    devices = jax.devices()
    heights = [int(a.trace.shape[0]) for a in airs]
    print(f"backend={backend} devices={devices}")
    print(f"fixture={prove_dir}  trace_heights={heights}  whir_rounds={len(params.whir.num_queries)}")

    times = []
    for i in range(_FLAGS.runs):
        t0 = time.perf_counter()
        _, proof = prove(new_transcript(), sponge, comp, params, vk_pre_hash, airs)
        jax.block_until_ready(_array_leaves(proof))
        dt = time.perf_counter() - t0
        times.append(dt)
        tag = "cold" if i == 0 else "warm"
        print(f"run {i}: {dt:8.3f}s ({tag})")

    if len(times) > 1:
        warm = times[1:]
        print(
            f"\ncold={times[0]:.3f}s  warm_min={min(warm):.3f}s  "
            f"warm_mean={sum(warm)/len(warm):.3f}s  compile≈{times[0]-min(warm):.3f}s"
        )


if __name__ == "__main__":
    app.run(main)
