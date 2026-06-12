"""WHIR (Stage 5) phase ablation — eager-vs-jit per device island (#5).

Stage 5's ``prove_whir_opening`` runs entirely eager today, and eager dispatch
*decomposes* each fusable composite (the sumcheck product-sum, the Poseidon2
tree, the NTT) into per-op kernels — the dominant cost. This bench measures the
five device-compute islands the round loop is factored into
(``prover._setup_f_evals`` / ``_round_poly`` / ``_apply_fold`` / ``_encode_commit``
/ ``_weight_update``), each **eager and under ``jax.jit``**, plus ``total`` (the
whole eager stage). zkbench's compile/runtime split (``--phase compile|runtime``)
separates the per-shape recompile cost from the compute win, since WHIR's tables
halve each fold and so retrace per round.

The island inputs are not synthesized: one real ``prove_whir_opening`` over the
vendored fixture is run with the island functions wrapped to record their
first (round-0, largest) call, so what is timed is exactly what the byte-matched
prover produced. Fixture load, trace recommit, and transcript replay stay
outside every timer.

    bazel run //openvm_zorch/whir:bench_whir_phases -- --phase runtime
    bazel run //openvm_zorch/whir:bench_whir_phases -- --phase compile --ops encode_commit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from zk_dtypes import babybear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

import openvm_zorch.whir.prover as wp
from openvm_zorch.commit.trace_commit import stacked_commit
from openvm_zorch.poseidon2.babybear16 import babybear16_params
from openvm_zorch.transcript import ef_from_limbs, new_transcript
from openvm_zorch.whir.prover import WhirConfig, prove_whir_opening
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_FIXTURE = Path(__file__).parent / "testdata" / "whir"

# The five islands, in round order. round-0 inputs are the largest (the tables
# halve each fold), so the first recorded call of each is the headline shape.
_ISLANDS = (
    "_setup_f_evals",
    "_round_poly",
    "_apply_fold",
    "_encode_commit",
    "_weight_update",
)
_OPS = ("setup_f_evals", "round_poly", "apply_fold", "encode_commit", "weight_update")


def _replay_log(values: np.ndarray, is_sample: np.ndarray, end: int):
    """Reconstruct the transcript state at log index ``end`` (whir_test's
    replay): feed observes back in, squeeze at samples and assert they match."""
    t = new_transcript()
    idx = 0
    while idx < end:
        if is_sample[idx]:
            t, got = t.sample(1)
            got = int(np.asarray(lax.bitcast_convert_type(got, F).astype(jnp.uint32))[0])
            assert got == int(values[idx]), f"sample mismatch at {idx}"
            idx += 1
        else:
            run = idx
            while run < end and not is_sample[run]:
                run += 1
            t = t.observe(jnp.array(values[idx:run], dtype=F))
            idx = run
    return t


def _load_stage5_call():
    """Build the byte-anchored ``prove_whir_opening`` call from the fixture:
    returns a zero-arg thunk that runs the whole stage, plus the static
    ``(sponge, compressor)`` and config the islands close over."""
    meta = json.loads((_FIXTURE / "meta.json").read_text())
    params = meta["params"]
    l_skip = params["l_skip"]

    perm = Poseidon2(babybear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))

    traces = [
        jnp.array(np.load(_FIXTURE / "inputs" / f"trace_{air_idx}.npy"), dtype=F)
        for air_idx in meta["sorted_airs"]
    ]
    _, data = stacked_commit(
        sponge, comp, l_skip, params["n_stack"], params["log_blowup"],
        params["k_whir"], traces,
    )
    u_cube = [
        ef_from_limbs(jnp.array(row, jnp.uint32))
        for row in np.load(_FIXTURE / "inputs" / "u_cube.npy")
    ]
    config = WhirConfig(
        k=params["k_whir"],
        num_queries=meta["num_queries"],
        mu_pow_bits=params["mu_pow_bits"],
        folding_pow_bits=params["folding_pow_bits"],
        query_phase_pow_bits=params["query_phase_pow_bits"],
    )
    values = np.load(_FIXTURE / "outputs" / "transcript_values.npy")
    is_sample = np.load(_FIXTURE / "outputs" / "transcript_is_sample.npy")

    # Replay the transcript to stage4_end ONCE, outside every timer. The duplex
    # transcript is functional (observe/sample return fresh instances), so the
    # same t0 drives prove_whir_opening deterministically on each timed call.
    t0 = _replay_log(values, is_sample, meta["stage4_end"])

    def make_run(jit: bool):
        def run_stage():
            return prove_whir_opening(
                t0, sponge, comp, l_skip, params["log_blowup"], config,
                [(data.matrix, data.tree)], u_cube, jit=jit,
            )

        return run_stage

    run_stage = make_run(jit=False)

    meta_out = {
        "fixture": _FIXTURE.name,
        "k_whir": str(params["k_whir"]),
        "num_queries": ",".join(map(str, meta["num_queries"])),
    }
    return run_stage, make_run, sponge, comp, meta_out


def _run_total(run_stage):
    """Run the whole eager stage and block on the observed-path leaves (the
    sumcheck polys, codeword commits, OOD values, final poly) — the dominant
    compute. WhirProof is not a pytree, so the leaves are gathered by field."""
    _, proof = run_stage()
    leaves = [proof.mu_pow_witness, proof.mu, proof.final_poly]
    leaves += list(proof.whir_sumcheck_polys)
    leaves += list(proof.codeword_commits)
    leaves += list(proof.ood_values)
    return jax.block_until_ready(leaves)


def _encode_root(sp, cp, g, rs_len, k_whir):
    """Eager J4 returning just the codeword root (block-able Array)."""
    return wp._encode_commit(sp, cp, g, rs_len, k_whir).root


def _capture_island_inputs(run_stage) -> dict:
    """Run the stage once with each island wrapped to record its first
    (round-0) call. The recorded args are exactly what the byte-matched prover
    fed the island, so no re-derivation can drift."""
    captured: dict = {}
    originals = {name: getattr(wp, name) for name in _ISLANDS}

    def recorder(name):
        orig = originals[name]

        def wrapped(*a, **k):
            captured.setdefault(name, (a, k))  # first call only
            return orig(*a, **k)

        return wrapped

    try:
        for name in _ISLANDS:
            setattr(wp, name, recorder(name))
        jax.block_until_ready(run_stage())
    finally:
        for name, fn in originals.items():
            setattr(wp, name, fn)
    return captured


class WhirPhasesBenchmark(JaxBenchmark):
    """Eager-vs-jit ablation of WHIR's five device islands plus the eager total."""

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="openvm-zorch-whir-phases",
            version="0",
            default_iterations=20,
            default_warmup=3,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ops",
            default="all",
            help=f"comma list of {{{','.join(_OPS)},total}} or 'all'",
        )
        parser.add_argument(
            "--variant",
            default="both",
            choices=("eager", "jit", "both"),
            help="measure the eager island, the jit-wrapped island, or both",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        wanted = (
            set(_OPS) | {"total"} if args.ops == "all" else set(args.ops.split(","))
        )
        run_stage, make_run, sponge, comp, meta = _load_stage5_call()
        captured = _capture_island_inputs(run_stage)

        def op(name: str, fn, lower=None) -> BenchmarkOp:
            return BenchmarkOp(name=name, fn=fn, metadata=meta, lower=lower)

        # Bind args at definition time: get_ops is a generator and the lambdas
        # would otherwise capture loop-reused names by late reference.
        def call(f, *a):
            return lambda: f(*a)

        eager = args.variant in ("eager", "both")
        jit = args.variant in ("jit", "both")

        def emit(op_key, island, build_jit, *call_args):
            """Yield the eager and/or jit op for one island from its captured
            args. ``build_jit`` maps the args to (jitted_callable, jit_args)."""
            if op_key not in wanted or island not in captured:
                return
            if eager:
                yield op(f"{op_key}_eager", call(getattr(wp, island), *call_args))
            if jit:
                j, jargs = build_jit(*call_args)
                yield op(f"{op_key}_jit", call(j, *jargs), lower=call(j.lower, *jargs))

        # total: the whole eager stage (no jit — the per-fold grinds and the
        # host-int query sampling break a single trace). The stage-5 headline
        # for this fixture; blocks on the observed-path leaves (setup + folds +
        # encode + commit), the dominant compute — the query-hint gathers are
        # cheap strided reads and are not separately forced.
        if "total" in wanted:
            yield op("whir_total_eager", call(_run_total, run_stage))
        if "total_jit" in wanted:
            yield op("whir_total_jit", call(_run_total, make_run(jit=True)))

        (mats, l_skip, m, mu), _ = captured.get("_setup_f_evals", ((None,) * 4, None))
        yield from emit(
            "setup_f_evals", "_setup_f_evals",
            lambda *a: (jax.jit(wp._setup_f_evals, static_argnums=(1, 2)), a),
            mats, l_skip, m, mu,
        )

        (fe, we), _ = captured.get("_round_poly", ((None, None), None))
        yield from emit(
            "round_poly", "_round_poly", lambda *a: (jax.jit(wp._round_poly), a), fe, we
        )

        (ffe, fwe, alpha), _ = captured.get("_apply_fold", ((None,) * 3, None))
        yield from emit(
            "apply_fold", "_apply_fold", lambda *a: (jax.jit(wp._apply_fold), a),
            ffe, fwe, alpha,
        )

        # encode_commit: sponge/comp close over (non-pytree hash state),
        # rs_len/k_whir are static, and only the codeword root crosses the jit
        # boundary — the StackedMerkleTree is not a registered pytree.
        (sp, cp, g, rs_len, k_whir), _ = captured.get(
            "_encode_commit", ((None,) * 5, None)
        )
        if "encode_commit" in wanted and "_encode_commit" in captured:
            if eager:
                yield op(
                    "encode_commit_eager",
                    call(_encode_root, sp, cp, g, rs_len, k_whir),
                )
            if jit:
                j = jax.jit(
                    lambda gc: wp._encode_commit(sp, cp, gc, rs_len, k_whir).root
                )
                yield op("encode_commit_jit", call(j, g), lower=call(j.lower, g))

        (uwe, gamma, z0, zi), _ = captured.get("_weight_update", ((None,) * 4, None))
        yield from emit(
            "weight_update", "_weight_update",
            lambda *a: (jax.jit(wp._weight_update), a),
            uwe, gamma, z0, zi,
        )


def main() -> int:
    return WhirPhasesBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
