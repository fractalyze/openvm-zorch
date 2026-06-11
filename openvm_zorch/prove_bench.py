"""Per-stage compile-vs-compute split for ``prove()`` (issue #1, milestone #1).

Runs the production-shaped instance (l_skip=4, n_stack=8, k_whir=4) through
``prove()`` **twice in one process**: the 1st call pays XLA first-compile +
compute, the 2nd hits the warm executable cache (compute only). The delta is
compile-time. Each of the five stages is wrapped so its own compile/compute
split is attributed within the whole-prove number — the warm column localizes
where the dominant *compute* lives.

Not a byte-match test — correctness is pinned by
``prove_test.test_prove_production_params``; this only measures wall time.
Run: ``bazel run //openvm_zorch:prove_bench``.
"""

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from zk_dtypes import babybear_mont as F

import openvm_zorch.prove as prove_mod
from openvm_zorch.logup_gkr.input_layer import InteractionSpec
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.prove import AirInstance, SystemParams, prove
from openvm_zorch.transcript import new_transcript
from openvm_zorch.whir.prover import WhirConfig
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_PROVE = Path(__file__).parent / "testdata" / "prove"


def _poseidon2():
    perm = Poseidon2(babybear16_params())
    return (
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _build_production_instance():
    """Mirror ``prove_test.test_prove_production_params`` instance build."""
    meta = json.loads((_PROVE / "meta.json").read_text())
    pm = meta["params"]

    airs = []
    for air in meta["airs"]:
        air_idx = air["air_idx"]
        trace = jnp.array(np.load(_PROVE / "inputs" / f"trace_{air_idx}.npy"), dtype=F)
        dag = ConstraintsDag.from_json(
            json.loads((_PROVE / "inputs" / f"constraints_{air_idx}.json").read_text())
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


# --- Per-stage timing: wrap each stage fn in prove's namespace. -------------
# Stage 2 (GKR) is two calls — input-layer build + the sumcheck — both charged
# to "2-gkr". Each wrapper blocks on every array leaf of its return so the
# stage's compute is fully realized before the clock stops.
_STAGES = ["1-commit", "2-gkr", "3-zerocheck", "4-stacking", "5-whir"]
_times: dict[str, float] = {}


def _block_all(obj):
    acc: list = []

    def walk(o):
        if isinstance(o, jax.Array):
            acc.append(o)
        elif isinstance(o, (tuple, list)):
            for x in o:
                walk(x)
        elif dataclasses.is_dataclass(o) and not isinstance(o, type):
            for f in dataclasses.fields(o):
                walk(getattr(o, f.name))
        elif hasattr(o, "__dict__"):
            for v in vars(o).values():
                walk(v)

    walk(obj)
    if acc:
        jax.block_until_ready(acc)


def _wrap(attr: str, label: str):
    orig = getattr(prove_mod, attr)

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        _block_all(out)
        _times[label] = _times.get(label, 0.0) + (time.perf_counter() - t0)
        return out

    setattr(prove_mod, attr, timed)


_wrap("stacked_commit", "1-commit")
_wrap("gkr_input_evals", "2-gkr")
_wrap("fractional_sumcheck", "2-gkr")
_wrap("prove_batch_constraints", "3-zerocheck")
_wrap("prove_stacked_opening_reduction", "4-stacking")
_wrap("prove_whir_opening", "5-whir")


def _run_once(params, vk_pre_hash, airs) -> tuple[float, dict[str, float]]:
    """One full prove(); returns (whole_prove_s, {stage: seconds})."""
    _times.clear()
    t0 = time.perf_counter()
    _, proof = prove(new_transcript(), *_poseidon2(), params, vk_pre_hash, airs)
    _block_all(proof)
    whole = time.perf_counter() - t0
    return whole, dict(_times)


def main():
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    params, vk_pre_hash, airs = _build_production_instance()

    whole1, st1 = _run_once(params, vk_pre_hash, airs)
    print(f"\n[call 1 = compile+compute]  prove={whole1:8.1f}s")
    whole2, st2 = _run_once(params, vk_pre_hash, airs)
    print(f"[call 2 = warm compute   ]  prove={whole2:8.1f}s")

    print(f"\n{'stage':<14}{'cold':>9}{'warm':>9}{'compile':>9}{'% warm prove':>14}")
    acc_warm = 0.0
    for s in _STAGES:
        cold, warm = st1.get(s, 0.0), st2.get(s, 0.0)
        acc_warm += warm
        print(f"{s:<14}{cold:>8.1f}s{warm:>8.1f}s{cold - warm:>8.1f}s"
              f"{100 * warm / whole2:>13.0f}%")
    print(f"{'sum(stages)':<14}{'':>9}{acc_warm:>8.1f}s")
    print(f"{'glue/grinds':<14}{'':>9}{whole2 - acc_warm:>8.1f}s  "
          f"(prelude observes, PoW grinds, handoffs)")
    print(f"\nwhole prove() compile (cold-warm): {whole1 - whole2:.1f}s "
          f"({100 * (whole1 - whole2) / whole1:.0f}% of cold)")


if __name__ == "__main__":
    main()
