"""LogUp-GKR stage phase ablation over a prove fixture (zkbench).

Phases mirror LogupGkrProver's body in ``openvm_zorch/prove.py``: ``grind`` (LogUp
PoW), ``input_evals`` (the DAG-built ``(count, denom)`` input layer over all
AIRs), ``frac_sumcheck`` (the fractional-sumcheck round chain), plus ``total``
(the whole reduction). ``total`` is the op that joins the milestone-4 per-stage
report; the three sub-phases are this prover's own ablation.

Structure mirrors sp1-zorch's ``bench_logup_gkr_phases.py`` so the per-stage
benches read the same across repos. The harness is zkbench's ``FrxBenchmark``:
it runs ``--warmup`` then ``--iterations`` timed runs and reports warm latency
(the role has no single ``lowered.compile()`` — a host-loop grind plus jit
islands across a Python round loop — so no ``lower`` thunk is given and the op
carries no zkbench compile metric; observe COMPILE out of band, see below).

File loading, the role build, and every phase's entry state stay outside the
timers. Phase inputs are re-derived from the post-commit transcript
(``LogupGkrProver`` reads only the witness traces + that transcript, never the
commit half's data) and the run aborts before timing if ``total``'s ``q0_claim`` drifts
from the fixture's ``outputs/q0_claim.npy``, so the phases cannot silently
diverge from what the real prove sees.

    # warm runtime (the standard report):
    FRX_PLATFORMS=cpu bazel run //openvm_zorch/logup_gkr:bench_logup_gkr_phases \
        -- --fixture_dir /tmp/real_fib

    # compile (out of band): zkbench discards warmup, so run a COLD-cache
    # process at --warmup 0 --iterations 1 (each op's one timed call then
    # includes its compile), and subtract the warm latency above. Keep
    # FRX_COMPILATION_CACHE_DIR unset for the cold run.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import frx.numpy as fnp
import numpy as np
from frx import lax
from zk_dtypes import babybear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, FrxBenchmark

from openvm_zorch.bench_common import array_leaves
from openvm_zorch.logup_gkr.input_layer import gkr_input_evals
from openvm_zorch.logup_gkr.prover import fractional_sumcheck
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag
from openvm_zorch.poseidon2.babybear16 import babybear16_hasher
from openvm_zorch.prove import (
    bind_commitment,
    build_prover,
)
from openvm_zorch.transcript import grind, new_transcript, sample_ext
from openvm_zorch.types import (
    AirInstance,
    SystemParams,
)
from openvm_zorch.whir.prover import WhirConfig

_OPS = ("grind", "input_evals", "frac_sumcheck", "total")


def _load_instance(prove_dir):
    """Mirror verify_prove._load_instance / prove_test input construction."""
    meta = json.loads((prove_dir / "meta.json").read_text())
    pm = meta["params"]
    airs = []
    for air in meta["airs"]:
        air_idx = air["air_idx"]
        trace = fnp.array(
            np.load(prove_dir / "inputs" / f"trace_{air_idx}.npy"), dtype=F
        )
        dag = ConstraintsDag.from_json(
            json.loads(
                (prove_dir / "inputs" / f"constraints_{air_idx}.json").read_text()
            )
        )
        cached_mains = tuple(
            fnp.array(
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
    return np.asarray(lax.bitcast_convert_type(fnp.atleast_1d(x), F).astype(fnp.uint32))


class LogupGkrPhasesBenchmark(FrxBenchmark):
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
        sponge, comp = babybear16_hasher()

        prover, claim, witness = build_prover(sponge, comp, params, vk_pre_hash, airs)
        gkr = prover.gkr
        # Advance past the trace commit: the PCS commit half, then the prelude
        # both roles absorb.
        commitment, _ = prover.pcs.commit(witness)
        t1 = bind_commitment(
            new_transcript(),
            claim,
            commitment,
            witness.cached,
            vk_pre_hash=vk_pre_hash,
        )

        sorted_airs = witness.sorted_airs
        n_logup = claim.shape.n_logup
        traces = [a.trace for a in sorted_airs]
        dags = [a.dag for a in sorted_airs]
        pubs = [a.public_values for a in sorted_airs]
        nxt = [a.needs_next for a in sorted_airs]
        cached = [a.cached_mains for a in sorted_airs]

        # Phase-entry states (untimed): grind -> sample alpha/beta -> input layer.
        tg, _ = grind(t1, params.logup_pow_bits)
        ta, alpha = sample_ext(tg)
        tb, beta = sample_ext(ta)
        num, den = gkr_input_evals(
            params.l_skip, n_logup, traces, dags, pubs, nxt, cached, alpha, beta
        )

        # Anchor: total's q0_claim must match the fixture, or we'd time the
        # wrong computation (mirrors sp1-zorch's check_match gate).
        msg = gkr.prove(claim, witness, t1).reduction_proof
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
            "n_logup": str(n_logup),
        }
        total_rows = sum(int(a.trace.shape[0]) for a in sorted_airs)

        def _op(name, fn) -> BenchmarkOp:
            return BenchmarkOp(
                name=name,
                fn=lambda: array_leaves(fn()),
                metadata=meta,
                throughput_unit="rows/s",
                throughput_count=total_rows,
            )

        if "grind" in ops:
            yield _op("logup_gkr_grind", lambda: grind(t1, params.logup_pow_bits))
        if "input_evals" in ops:
            yield _op(
                "logup_gkr_input_evals",
                lambda: gkr_input_evals(
                    params.l_skip,
                    n_logup,
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
            yield _op("logup_gkr_total", lambda: gkr.prove(claim, witness, t1))


def main() -> int:
    return LogupGkrPhasesBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
