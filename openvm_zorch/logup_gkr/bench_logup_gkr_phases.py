"""LogUp-GKR stage phase ablation over a prove fixture (zkbench).

Phases mirror GkrStage's body in ``openvm_zorch/prove.py``: ``grind`` (LogUp
PoW), ``input_evals`` (the DAG-built ``(count, denom)`` input layer over all
AIRs), ``frac_sumcheck`` (the fractional-sumcheck round chain), plus ``total``
(the whole GkrStage). ``total`` is the op that joins the milestone-4 per-stage
report; the three sub-phases are this prover's own ablation.

Structure mirrors sp1-zorch's ``bench_logup_gkr_phases.py`` so the per-stage
benches read the same across repos. The harness is zkbench's ``JaxBenchmark``:
it runs ``--warmup`` then ``--iterations`` timed runs and reports warm latency
(GkrStage has no single ``lowered.compile()`` — a host-loop grind plus jit
islands across a Python round loop — so no ``lower`` thunk is given and the op
carries no zkbench compile metric; observe COMPILE out of band, see below).

File loading, the chain build, and every phase's entry state stay outside the
timers. Phase inputs are re-derived from the post-commit transcript (GkrStage
reads only ``carry.sorted_airs`` + that transcript, never CommitStage's carry
outputs) and the run aborts before timing if ``total``'s ``q0_claim`` drifts
from the fixture's ``outputs/q0_claim.npy``, so the phases cannot silently
diverge from what the real prove sees.

    # warm runtime (the standard report):
    JAX_PLATFORMS=cpu bazel run //openvm_zorch/logup_gkr:bench_logup_gkr_phases \
        -- --fixture_dir /tmp/real_fib

    # compile (out of band): zkbench discards warmup, so run a COLD-cache
    # process at --warmup 0 --iterations 1 (each op's one timed call then
    # includes its compile), and subtract the warm latency above. Keep
    # JAX_COMPILATION_CACHE_DIR unset for the cold run.
"""

import argparse
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

import frx
import frx.numpy as jnp
import numpy as np
from frx import lax
from zk_dtypes import babybear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from openvm_zorch.logup_gkr.input_layer import gkr_input_evals
from openvm_zorch.logup_gkr.prover import fractional_sumcheck
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.prove import AirInstance, SystemParams, prove_chain
from openvm_zorch.transcript import grind, new_transcript, sample_ext
from openvm_zorch.whir.prover import WhirConfig
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_OPS = ("grind", "input_evals", "frac_sumcheck", "total")


def _array_leaves(obj):
    """Flatten JAX arrays out of a (possibly dataclass) structure, so the
    harness's ``block_until_ready`` reaches them — stage outputs are plain
    dataclasses, not registered pytrees (block on them is a silent no-op)."""
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


def _poseidon2():
    perm = Poseidon2(babybear16_params())
    return (
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _load_instance(prove_dir):
    """Mirror verify_prove._load_instance / prove_test input construction."""
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
    return params, meta["vk_pre_hash"], airs


def _ef_limbs(x) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(jnp.atleast_1d(x), F).astype(jnp.uint32))


class LogupGkrPhasesBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="openvm-zorch",
            version="0.1.0",
            default_iterations=5,
            default_warmup=1,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--fixture_dir", type=str, required=True, help="Prove fixture dir."
        )
        parser.add_argument("--ops", nargs="+", choices=_OPS, default=list(_OPS))

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        # Everything up to the first yield — file IO, chain build, commit
        # prelude, phase-entry states, and the anchor gate — runs untimed.
        ops = set(args.ops)
        prove_dir = Path(args.fixture_dir)
        params, vk_pre_hash, airs = _load_instance(prove_dir)
        sponge, comp = _poseidon2()

        chain, carry0 = prove_chain(sponge, comp, params, vk_pre_hash, airs)
        commit, gkr = chain.rounds[0], chain.rounds[1]
        c1, t1, _ = commit(carry0, new_transcript())  # advance past the trace commit

        sorted_airs = c1.sorted_airs
        traces = [a.trace for a in sorted_airs]
        dags = [a.dag for a in sorted_airs]
        pubs = [a.public_values for a in sorted_airs]
        nxt = [a.needs_next for a in sorted_airs]
        cached = [a.cached_mains for a in sorted_airs]

        # Phase-entry states (untimed): grind -> sample alpha/beta -> input layer.
        tg, _ = grind(t1, gkr._logup_pow_bits)
        ta, alpha = sample_ext(tg)
        tb, beta = sample_ext(ta)
        num, den = gkr_input_evals(
            gkr._l_skip, gkr._n_logup, traces, dags, pubs, nxt, cached, alpha, beta
        )

        # Anchor: total's q0_claim must match the fixture, or we'd time the
        # wrong computation (mirrors sp1-zorch's check_match gate).
        _, _, msg = gkr(c1, t1)
        got = _ef_limbs(msg.gkr_proof.q0_claim)[0]
        want = np.load(prove_dir / "outputs" / "q0_claim.npy")
        if not np.array_equal(got, want):
            raise SystemExit(
                "q0_claim diverged from the fixture; aborting before timing"
            )

        meta = {
            "fixture": prove_dir.name,
            "field": "babybear",
            "num_airs": str(len(sorted_airs)),
            "n_logup": str(gkr._n_logup),
        }
        total_rows = sum(int(a.trace.shape[0]) for a in sorted_airs)

        def _op(name, fn) -> BenchmarkOp:
            return BenchmarkOp(
                name=name,
                fn=lambda: _array_leaves(fn()),
                metadata=meta,
                throughput_unit="rows/s",
                throughput_count=total_rows,
            )

        if "grind" in ops:
            yield _op("logup_gkr_grind", lambda: grind(t1, gkr._logup_pow_bits))
        if "input_evals" in ops:
            yield _op(
                "logup_gkr_input_evals",
                lambda: gkr_input_evals(
                    gkr._l_skip,
                    gkr._n_logup,
                    traces,
                    dags,
                    pubs,
                    nxt,
                    cached,
                    alpha,
                    beta,
                ),
            )
        if "frac_sumcheck" in ops:
            yield _op(
                "logup_gkr_frac_sumcheck", lambda: fractional_sumcheck(tb, num, den)
            )
        if "total" in ops:
            yield _op("logup_gkr_total", lambda: gkr(c1, t1))


def main() -> int:
    return LogupGkrPhasesBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
