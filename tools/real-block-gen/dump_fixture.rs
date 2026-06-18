//! Real-guest "fixture dump" benchmark bin.
//!
//! Runs the `fibonacci` guest through the app prover, taps the per-segment
//! [`ProvingContext`] via [`VmInstance::prove_continuations`], and either:
//!   - PRINTS the real block's structure (default, no `--out`): per segment the number of non-empty
//!     AIRs, per-AIR trace dims / public values, and a classification of each interaction
//!     `count`/`message` field (column-ref vs compound expression); or
//!   - DUMPS to disk (`--out <dir>`) a golden fixture in openvm-zorch's fixture-gen `--prove-out`
//!     layout (see below), so zorch's `prove()` can be driven from a real guest block.
//!
//! The disk layout under `<dir>` mirrors `tools/fixture-gen`'s
//! `gen_prove_fixture` (openvm-zorch), for the single fibonacci segment:
//!   - `inputs/trace_<air_idx>.npy`  — each non-empty AIR's `common_main` (`ColMajorMatrix<F>`) as
//!     a `(height, width)` `<u4` row-major array of `as_canonical_u32()` cells.
//!   - `inputs/constraints_<air_idx>.json` — the AIR's `symbolic_constraints` DAG, byte-for-byte in
//!     `constraints_dag_json`'s shape (so zorch's `ConstraintsDag.from_json` parses it).
//!   - `meta.json` — `reference`, real `params`, `num_queries`, `vk_pre_hash`, `sorted_airs`, and
//!     per-AIR `airs[]` (incl. node-index interactions).
//!
//! `<air_idx>` is the GLOBAL proving-key AIR index (the key into
//! `vm_pk.per_air`), used both as the filename key and inside `meta.json`.
//!
//! Run with:
//!   cargo run --profile fast -p openvm-benchmarks-prove --bin dump_fixture
//!   cargo run --profile fast -p openvm-benchmarks-prove --bin dump_fixture -- --out /tmp/real_fib

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs,
    io::Write as _,
    path::{Path, PathBuf},
    time::Instant,
};

use openvm_sdk::{
    config::AppConfig, keygen::AppProvingKey, prover::vm::new_local_prover, CpuSdk, Sdk, StdIn,
};
use openvm_sdk_config::SdkVmConfig;
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_variable::Entry, SymbolicConstraintsDag, SymbolicExpressionNode,
    },
    calculate_n_logup,
    p3_matrix::dense::RowMajorMatrix,
    proof::{column_openings_by_rot, BatchConstraintProof, GkrProof, StackingProof, WhirProof},
    prover::{
        stacked_pcs::stacked_commit, AirProvingContext, ColMajorMatrix, CommittedTraceData,
        CpuColMajorBackend, MatrixDimensions, ProvingContext,
    },
    test_utils::TestFixture,
    AirRef, StarkEngine, StarkProtocolConfig, SystemParams, TranscriptHistory, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2RefEngine, DuplexSponge, DuplexSpongeRecorder,
    EF,
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use p3_field::{BasedVectorSpace, PrimeField32};

type F = openvm_sdk::F;
/// The shared protocol config: `BabyBearPoseidon2Config`. Both the app prover
/// (`CpuBackend`) and the reference recording engine (`CpuColMajorBackend`) are
/// parameterized by this same `SC`, so the app proving key transports directly.
type SC = openvm_sdk::SC;

/// Classification of a single interaction-field expression node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FieldKind {
    /// Bare column reference: a single `Variable` node.
    ColumnRef,
    /// Compound expression: `Add`/`Sub`/`Mul`/`Neg` of multiple nodes.
    Compound,
    /// A literal field constant.
    Constant,
    /// A row selector: `IsFirstRow`/`IsLastRow`/`IsTransition`.
    Selector,
}

impl FieldKind {
    fn label(self) -> &'static str {
        match self {
            FieldKind::ColumnRef => "column-ref",
            FieldKind::Compound => "compound",
            FieldKind::Constant => "constant",
            FieldKind::Selector => "selector",
        }
    }
}

/// Classify the symbolic-expression DAG node referenced by an interaction field
/// (the field is a `usize` index into `nodes`, in topological order).
fn classify(nodes: &[SymbolicExpressionNode<F>], idx: usize) -> FieldKind {
    match &nodes[idx] {
        SymbolicExpressionNode::Variable(_) => FieldKind::ColumnRef,
        SymbolicExpressionNode::Constant(_) => FieldKind::Constant,
        SymbolicExpressionNode::IsFirstRow
        | SymbolicExpressionNode::IsLastRow
        | SymbolicExpressionNode::IsTransition => FieldKind::Selector,
        SymbolicExpressionNode::Add { .. }
        | SymbolicExpressionNode::Sub { .. }
        | SymbolicExpressionNode::Neg { .. }
        | SymbolicExpressionNode::Mul { .. } => FieldKind::Compound,
    }
}

/// Running tally of field-kind counts, split into `count` fields and `message`
/// fields.
#[derive(Default)]
struct Tally {
    count_fields: BTreeMap<FieldKind, usize>,
    message_fields: BTreeMap<FieldKind, usize>,
}

impl Tally {
    fn add_count(&mut self, k: FieldKind) {
        *self.count_fields.entry(k).or_default() += 1;
    }
    fn add_message(&mut self, k: FieldKind) {
        *self.message_fields.entry(k).or_default() += 1;
    }
}

fn fmt_kinds(m: &BTreeMap<FieldKind, usize>) -> String {
    let total: usize = m.values().sum();
    if total == 0 {
        return "(none)".to_string();
    }
    let mut parts: Vec<String> = m
        .iter()
        .map(|(k, n)| {
            let pct = 100.0 * (*n as f64) / (total as f64);
            format!("{}={} ({:.1}%)", k.label(), n, pct)
        })
        .collect();
    parts.push(format!("total={total}"));
    parts.join(", ")
}

// --- .npy serializers (mirror openvm-zorch fixture-gen's `write_npy_u32` /
// `write_matrix` byte-for-byte) ---

/// Minimal .npy v1 writer: little-endian u32, C order.
fn write_npy_u32(path: &Path, shape: &[usize], data: &[u32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let descr = "<u4";
    let elem = 4;
    assert_eq!(shape.iter().product::<usize>() * elem, bytes.len());
    let shape_str = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // Header (incl. magic + 2-byte len) pads with spaces to a multiple of 64,
    // ending in \n.
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let mut out = Vec::with_capacity(unpadded + padding + bytes.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&((header.len() + padding + 1) as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(b' ', padding));
    out.push(b'\n');
    out.extend_from_slice(&bytes);
    fs::write(path, out).unwrap();
}

/// Dump a `RowMajorMatrix` (the CPU backend's `common_main` and cached-main
/// `.trace` type) as a `(height, width)` `<u4` array of canonical u32 cells. Its
/// `values` are already row-major, so write them directly — matching fixture-gen's
/// `write_matrix` output shape (which converts col-major → row-major).
fn write_matrix(path: &Path, m: &RowMajorMatrix<F>) {
    let (h, w) = (m.height(), m.width());
    let data: Vec<u32> = m.values.iter().map(|x| x.as_canonical_u32()).collect();
    write_npy_u32(path, &[h, w], &data);
}

/// Canonical-u32 JSON of one AIR's constraints DAG (nodes in topological
/// order, constraint node indices, interactions in node-index form).
/// Hand-rolled rather than serde: BabyBear's serde emits Montgomery-form u32,
/// which would leak an encoding detail into the fixture. Mirrors
/// openvm-zorch fixture-gen's `constraints_dag_json` byte-for-byte.
fn constraints_dag_json(dag: &SymbolicConstraintsDag<F>) -> serde_json::Value {
    let nodes: Vec<serde_json::Value> = dag
        .constraints
        .nodes
        .iter()
        .map(|node| match *node {
            SymbolicExpressionNode::Variable(var) => {
                let (entry, part_index, offset) = match var.entry {
                    Entry::Preprocessed { offset } => ("preprocessed", None, Some(offset)),
                    Entry::Main { part_index, offset } => ("main", Some(part_index), Some(offset)),
                    Entry::Public => ("public", None, None),
                    _ => panic!("unsupported variable entry in prover DAG"),
                };
                serde_json::json!({
                    "kind": "variable", "entry": entry, "part_index": part_index,
                    "offset": offset, "index": var.index,
                })
            }
            SymbolicExpressionNode::IsFirstRow => serde_json::json!({"kind": "is_first_row"}),
            SymbolicExpressionNode::IsLastRow => serde_json::json!({"kind": "is_last_row"}),
            SymbolicExpressionNode::IsTransition => serde_json::json!({"kind": "is_transition"}),
            SymbolicExpressionNode::Constant(c) => {
                serde_json::json!({"kind": "constant", "value": c.as_canonical_u32()})
            }
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            } => {
                serde_json::json!({"kind": "add", "left": left_idx, "right": right_idx})
            }
            SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            } => {
                serde_json::json!({"kind": "sub", "left": left_idx, "right": right_idx})
            }
            SymbolicExpressionNode::Neg { idx, .. } => {
                serde_json::json!({"kind": "neg", "idx": idx})
            }
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => {
                serde_json::json!({"kind": "mul", "left": left_idx, "right": right_idx})
            }
        })
        .collect();
    let interactions: Vec<serde_json::Value> = dag
        .interactions
        .iter()
        .map(|i| {
            serde_json::json!({
                "bus_index": i.bus_index,
                "message": i.message,
                "count": i.count,
                "count_weight": i.count_weight,
            })
        })
        .collect();
    serde_json::json!({
        "nodes": nodes,
        "constraint_idx": dag.constraints.constraint_idx,
        "interactions": interactions,
    })
}

// --- GKR (Stage 2) transcript-log walk, VENDORED from openvm-zorch
// `tools/fixture-gen/src/main.rs` (the `GkrLogWalk` struct + `walk_gkr_log`) ---
//
// Pure log-walking + arithmetic, no `Stage2Fixture` dependency. It walks the
// recorded transcript through the GKR prelude + rounds, ASSERTING the
// observe/sample structure as it goes (so it doubles as validation), and
// returns the GKR challenges (alpha/beta/xi). It is parameterized by
// `prelude_len`/`pow_bits`/`total_rounds`/`n_global`/`l_skip`, which the caller
// derives from the proving key + traces. This bin runs it on the REAL
// fibonacci block's recorded log to test whether the generic walk survives the
// real-block structure (19 AIRs, cached mains, logup_pow_bits=18) without
// asserting.

/// First base-field limb of an extension element, as a canonical u32. The
/// explicit `&[F]` binding pins `BasedVectorSpace`'s subfield so inference
/// doesn't need a turbofish at each call site.
fn ef_limb0(x: EF) -> u32 {
    let coeffs: &[F] = x.as_basis_coefficients_slice();
    coeffs[0].as_canonical_u32()
}

/// All four base-field limbs of an extension element, as canonical u32. Mirrors
/// openvm-zorch fixture-gen's `ef_limbs` (the `outputs/` dump unit).
fn ef_limbs(x: EF) -> [u32; 4] {
    let coeffs: &[F] = x.as_basis_coefficients_slice();
    core::array::from_fn(|i| coeffs[i].as_canonical_u32())
}

/// Walks the recorded transcript log through the GKR stage, asserting the
/// structure as it goes, and returns the named challenge values.
struct GkrLogWalk {
    alpha: EF,
    beta: EF,
    xi: Vec<EF>,
    idx_after_beta: usize,
    stage2_end: usize,
}

#[allow(clippy::too_many_arguments)]
fn walk_gkr_log(
    log: &TranscriptLog<F, [F; 16]>,
    prelude_len: usize,
    pow_bits: usize,
    pow_witness: F,
    gkr: &GkrProof<SC>,
    l_skip: usize,
    total_rounds: usize,
    n_global: usize,
) -> GkrLogWalk {
    let mut idx = prelude_len;
    let read = |samp: bool, idx: &mut usize| -> F {
        assert_eq!(
            log.samples()[*idx],
            samp,
            "expected {} at transcript index {idx:?}",
            if samp { "sample" } else { "observe" }
        );
        let v = log.values()[*idx];
        *idx += 1;
        v
    };
    let read_ext = |samp: bool, idx: &mut usize| -> EF {
        let limbs: [F; 4] = core::array::from_fn(|_| read(samp, idx));
        EF::from_basis_coefficients_slice(&limbs).unwrap()
    };

    // PoW: witness observe + one masked sample.
    assert_eq!(read(false, &mut idx), pow_witness);
    let pow_sample = read(true, &mut idx);
    assert_eq!(pow_sample.as_canonical_u32() & ((1 << pow_bits) - 1), 0);

    let alpha = read_ext(true, &mut idx);
    let beta = read_ext(true, &mut idx);
    let idx_after_beta = idx;

    // fractional_sumcheck with assert_zero: observes q0 only.
    assert_eq!(read_ext(false, &mut idx), gkr.q0_claim);
    // Layer 1: claims (p0, q0, p1, q1) then mu_1.
    let c = &gkr.claims_per_layer[0];
    for want in [c.p_xi_0, c.q_xi_0, c.p_xi_1, c.q_xi_1] {
        assert_eq!(read_ext(false, &mut idx), want);
    }
    let mu_1 = read_ext(true, &mut idx);
    let mut xi = vec![mu_1];

    for round in 1..total_rounds {
        let _lambda = read_ext(true, &mut idx);
        let mut rho = Vec::with_capacity(round);
        for sumcheck_round in 0..round {
            for k in 0..3 {
                assert_eq!(
                    read_ext(false, &mut idx),
                    gkr.sumcheck_polys[round - 1][sumcheck_round][k]
                );
            }
            rho.push(read_ext(true, &mut idx));
        }
        let c = &gkr.claims_per_layer[round];
        for want in [c.p_xi_0, c.q_xi_0, c.p_xi_1, c.q_xi_1] {
            assert_eq!(read_ext(false, &mut idx), want);
        }
        let mu = read_ext(true, &mut idx);
        xi = [vec![mu], rho].concat();
    }

    // xi padding up to l_skip + n_global.
    while xi.len() != l_skip + n_global {
        xi.push(read_ext(true, &mut idx));
    }

    GkrLogWalk {
        alpha,
        beta,
        xi,
        idx_after_beta,
        stage2_end: idx,
    }
}

// --- Zerocheck (Stage 3), Stacking (Stage 4), WHIR (Stage 5) transcript-log
// walks, VENDORED from openvm-zorch `tools/fixture-gen/src/main.rs`
// (`walk_zerocheck_log` / `walk_stacking_log` / `walk_whir_log` + their result
// structs). They are pure log-walks + struct reads (no synthetic-fixture
// dependency): each asserts the recorded observe/sample structure against the
// proof struct as it goes, starting at the previous stage's end index, so a
// real-block divergence surfaces as an assertion the caller catches. The
// `needs_next` slice (per present AIR, in DESCENDING-height sorted order) is
// derived from the real proving key (`vk.params.need_rot`). ---

/// Stage-3 challenges recovered while walking the recorded log against the
/// `BatchConstraintProof`, asserting the observe/sample structure as it goes.
struct ZerocheckLogWalk {
    lambda: EF,
    #[allow(dead_code)]
    mu: EF,
    r: Vec<EF>,
    stage3_end: usize,
}

fn walk_zerocheck_log(
    log: &TranscriptLog<F, [F; 16]>,
    start: usize,
    bcp: &BatchConstraintProof<SC>,
    needs_next: &[bool],
) -> ZerocheckLogWalk {
    let mut idx = start;
    let read = |samp: bool, idx: &mut usize| -> F {
        assert_eq!(
            log.samples()[*idx],
            samp,
            "expected {} at transcript index {idx:?}",
            if samp { "sample" } else { "observe" }
        );
        let v = log.values()[*idx];
        *idx += 1;
        v
    };
    let read_ext = |samp: bool, idx: &mut usize| -> EF {
        let limbs: [F; 4] = core::array::from_fn(|_| read(samp, idx));
        EF::from_basis_coefficients_slice(&limbs).unwrap()
    };

    let lambda = read_ext(true, &mut idx);
    for (&p, &q) in bcp
        .numerator_term_per_air
        .iter()
        .zip(&bcp.denominator_term_per_air)
    {
        assert_eq!(read_ext(false, &mut idx), p);
        assert_eq!(read_ext(false, &mut idx), q);
    }
    let mu = read_ext(true, &mut idx);
    for &c in &bcp.univariate_round_coeffs {
        assert_eq!(read_ext(false, &mut idx), c);
    }
    let mut r = vec![read_ext(true, &mut idx)];
    for round_polys in &bcp.sumcheck_round_polys {
        for &e in round_polys {
            assert_eq!(read_ext(false, &mut idx), e);
        }
        r.push(read_ext(true, &mut idx));
    }
    // Common main openings first (part 0), then preprocessed/cached parts.
    for (openings, &nn) in bcp.column_openings.iter().zip(needs_next) {
        for (claim, claim_rot) in column_openings_by_rot(&openings[0], nn) {
            assert_eq!(read_ext(false, &mut idx), claim);
            assert_eq!(read_ext(false, &mut idx), claim_rot);
        }
    }
    for (openings, &nn) in bcp.column_openings.iter().zip(needs_next) {
        for part in openings.iter().skip(1) {
            for (claim, claim_rot) in column_openings_by_rot(part, nn) {
                assert_eq!(read_ext(false, &mut idx), claim);
                assert_eq!(read_ext(false, &mut idx), claim_rot);
            }
        }
    }

    ZerocheckLogWalk {
        lambda,
        mu,
        r,
        stage3_end: idx,
    }
}

/// Stage-4 challenges recovered while walking the recorded log against the
/// `StackingProof`.
struct StackingLogWalk {
    lambda: EF,
    u: Vec<EF>,
    stage4_end: usize,
}

fn walk_stacking_log(
    log: &TranscriptLog<F, [F; 16]>,
    start: usize,
    sp: &StackingProof<SC>,
) -> StackingLogWalk {
    let mut idx = start;
    let read = |samp: bool, idx: &mut usize| -> F {
        assert_eq!(
            log.samples()[*idx],
            samp,
            "expected {} at transcript index {idx:?}",
            if samp { "sample" } else { "observe" }
        );
        let v = log.values()[*idx];
        *idx += 1;
        v
    };
    let read_ext = |samp: bool, idx: &mut usize| -> EF {
        let limbs: [F; 4] = core::array::from_fn(|_| read(samp, idx));
        EF::from_basis_coefficients_slice(&limbs).unwrap()
    };

    let lambda = read_ext(true, &mut idx);
    for &c in &sp.univariate_round_coeffs {
        assert_eq!(read_ext(false, &mut idx), c);
    }
    let mut u = vec![read_ext(true, &mut idx)];
    for round_polys in &sp.sumcheck_round_polys {
        for &e in round_polys {
            assert_eq!(read_ext(false, &mut idx), e);
        }
        u.push(read_ext(true, &mut idx));
    }
    for claims_for_com in &sp.stacking_openings {
        for &claim in claims_for_com {
            assert_eq!(read_ext(false, &mut idx), claim);
        }
    }

    StackingLogWalk {
        lambda,
        u,
        stage4_end: idx,
    }
}

/// Stage-5 (WHIR opening) challenge recovered while walking the recorded log
/// against the `WhirProof`. See fixture-gen's `walk_whir_log` for the per-round
/// observe/sample order (μ-PoW grind → μ; per round: k_whir × (2 sumcheck
/// evals, folding grind, α), then either codeword commit/z₀/OOD or the final
/// poly, then the query-phase grind, the query index samples and γ). Opened
/// rows and Merkle proofs are hints — never observed.
struct WhirLogWalk {
    mu: EF,
    /// Per WHIR round, the in-domain query indices.
    #[allow(dead_code)]
    query_indices: Vec<Vec<usize>>,
    stage5_end: usize,
}

fn walk_whir_log(
    log: &TranscriptLog<F, [F; 16]>,
    start: usize,
    params: &SystemParams,
    wp: &WhirProof<SC>,
) -> WhirLogWalk {
    let whir = &params.whir;
    let k_whir = whir.k;
    let num_rounds = whir.num_whir_rounds();
    let mut idx = start;
    let read = |samp: bool, idx: &mut usize| -> F {
        assert_eq!(
            log.samples()[*idx],
            samp,
            "expected {} at transcript index {idx:?}",
            if samp { "sample" } else { "observe" }
        );
        let v = log.values()[*idx];
        *idx += 1;
        v
    };
    let read_ext = |samp: bool, idx: &mut usize| -> EF {
        let limbs: [F; 4] = core::array::from_fn(|_| read(samp, idx));
        EF::from_basis_coefficients_slice(&limbs).unwrap()
    };
    let read_grind = |bits: usize, witness: F, idx: &mut usize| {
        assert_eq!(read(false, idx), witness);
        let check = read(true, idx);
        assert_eq!(check.as_canonical_u32() & ((1 << bits) - 1), 0);
    };

    read_grind(whir.mu_pow_bits, wp.mu_pow_witness, &mut idx);
    let mu = read_ext(true, &mut idx);

    let mut log_rs_domain_size = params.l_skip + params.n_stack + params.log_blowup;
    let mut query_indices = Vec::with_capacity(num_rounds);
    for (whir_round, round_params) in whir.rounds.iter().enumerate() {
        let is_last_round = whir_round == num_rounds - 1;
        for round in 0..k_whir {
            let flat = whir_round * k_whir + round;
            for &eval in &wp.whir_sumcheck_polys[flat] {
                assert_eq!(read_ext(false, &mut idx), eval);
            }
            read_grind(
                whir.folding_pow_bits,
                wp.folding_pow_witnesses[flat],
                &mut idx,
            );
            let _alpha = read_ext(true, &mut idx);
        }
        if !is_last_round {
            for &limb in &wp.codeword_commits[whir_round] {
                assert_eq!(read(false, &mut idx), limb);
            }
            let _z_0 = read_ext(true, &mut idx);
            assert_eq!(read_ext(false, &mut idx), wp.ood_values[whir_round]);
        } else {
            for &coeff in &wp.final_poly {
                assert_eq!(read_ext(false, &mut idx), coeff);
            }
        }
        read_grind(
            whir.query_phase_pow_bits,
            wp.query_phase_pow_witnesses[whir_round],
            &mut idx,
        );
        let bits = log_rs_domain_size - k_whir;
        let indices = (0..round_params.num_queries)
            .map(|_| (read(true, &mut idx).as_canonical_u32() & ((1 << bits) - 1)) as usize)
            .collect();
        query_indices.push(indices);
        let _gamma = read_ext(true, &mut idx);
        log_rs_domain_size -= 1;
    }

    WhirLogWalk {
        mu,
        query_indices,
        stage5_end: idx,
    }
}

/// A [`TestFixture`] wrapping an already-built reference proving context.
///
/// `TestFixture::prove_from_transcript` is the only public path that drives a
/// RECORDING prove AND writes the final transcript back (the `Coordinator`'s
/// transcript field is `pub(crate)`). It needs a fixture supplying the ctx; we
/// hand it the real tapped traces (one-shot, consumed via `RefCell::take`).
/// `airs()` is only used by `keygen`, which we never call (we reuse the app
/// pk), so it returns empty.
struct RealCtxFixture {
    ctx: RefCell<Option<ProvingContext<CpuColMajorBackend<SC>>>>,
}

impl TestFixture<SC> for RealCtxFixture {
    fn airs(&self) -> Vec<AirRef<SC>> {
        vec![]
    }

    fn generate_proving_ctx(&self) -> ProvingContext<CpuColMajorBackend<SC>> {
        self.ctx
            .borrow_mut()
            .take()
            .expect("generate_proving_ctx called more than once")
    }
}

/// The captured real per-AIR raw traces of one segment, in the CPU backend's
/// `RowMajorMatrix` layout: `(air_id, common_main, cached_main_traces,
/// public_values)`. Re-committed into the reference col-major ctx by
/// [`build_ref_ctx`].
type RawAir = (usize, RowMajorMatrix<F>, Vec<RowMajorMatrix<F>>, Vec<F>);

/// Block until the device's current stream drains. The CUDA prover dispatches
/// kernels on the current stream and returns *before* they finish, so a timer
/// around `prove` alone would catch only the launch overhead (~ms), not the
/// compute. Sync after `prove` to time real completion, and before `t0` to
/// drain the prior run + the async H2D ctx transport so neither bleeds into the
/// timed region. A no-op for the CPU engine, whose `prove` is synchronous.
#[cfg(feature = "cuda")]
fn sync_stream() {
    openvm_cuda_common::stream::current_stream_sync().expect("cuda stream sync");
}
#[cfg(not(feature = "cuda"))]
fn sync_stream() {}

/// Build the col-major reference [`ProvingContext`] from the captured real
/// traces, mirroring the `--ref-prove` ctx-build: convert each AIR's
/// `common_main` RowMajor -> ColMajor and RE-COMMIT each cached main with the
/// reference backend's `stacked_commit` (its `PcsData` type differs from the
/// CpuBackend's, so the tapped cached commitment cannot be reused verbatim — it
/// must be recomputed in the reference layout).
///
/// Factored out so the `--ref-prove` recording path and the `--baseline-out`
/// timing path build the ctx identically. The hasher is derived from a
/// reference engine over the same `SystemParams` (the engine carries the SAME
/// `SystemParams` as the app pk, so its config / hasher match the app's vk);
/// building the engine here is cheap and avoids naming the hasher type at call
/// sites.
fn build_ref_ctx(
    ref_raw: Vec<RawAir>,
    params: &SystemParams,
) -> ProvingContext<CpuColMajorBackend<SC>> {
    let hasher_engine = BabyBearPoseidon2RefEngine::<DuplexSpongeRecorder>::new(params.clone());
    let hasher = hasher_engine.config().hasher();
    let whir = &params.whir;

    let mut ref_per_trace: Vec<(usize, AirProvingContext<CpuColMajorBackend<SC>>)> = Vec::new();
    for (air_id, common_main, cached, pvs) in ref_raw {
        let common_main_col = ColMajorMatrix::<F>::from_row_major(&common_main);
        let cached_mains: Vec<CommittedTraceData<CpuColMajorBackend<SC>>> = cached
            .iter()
            .map(|cm| {
                let trace = ColMajorMatrix::<F>::from_row_major(cm);
                let (commitment, data) = stacked_commit(
                    hasher,
                    params.l_skip,
                    params.n_stack,
                    params.log_blowup,
                    whir.k,
                    &[&trace],
                )
                .expect("stacked_commit for cached main");
                CommittedTraceData {
                    commitment,
                    trace,
                    data: std::sync::Arc::new(data),
                }
            })
            .collect();
        ref_per_trace.push((
            air_id,
            AirProvingContext::new(cached_mains, common_main_col, pvs),
        ));
    }
    ProvingContext::new(ref_per_trace)
}

/// Native-prover baseline: time the native SWIRL prover on the tapped real ctx
/// and write a timing JSON keyed by platform + params. The number that matters
/// is `prove_e2e_s` (the `prover.prove` step alone — the apples-to-apples scope
/// vs openvm-zorch's `prove_chain`); setup (the SDK app keygen / tracegen that
/// ran upstream in `prove_continuations`) is not separately timed here.
///
/// The default build times the CPU reference engine
/// (`BabyBearPoseidon2RefEngine` / `CpuColMajorBackend`), using the
/// non-recording `DuplexSponge` (not the `DuplexSpongeRecorder` the fixture path
/// uses — the recorder's per-step log append is pure overhead; the proof is
/// byte-identical either way). `--features cuda` swaps in the CUDA
/// `BabyBearPoseidon2GpuEngine` for the GPU baseline.
///
/// `BENCH_RUNS` (default 3, min 1) sets the warm-run count; `BENCH_PLATFORM_LABEL`
/// overrides the machine tag (default "cpu", or "gpu" under `--features cuda`).
fn gen_baseline(
    out_file: &Path,
    ref_raw: Vec<RawAir>,
    params: &SystemParams,
    vm_pk: &openvm_stark_backend::keygen::types::MultiStarkProvingKey<SC>,
) {
    let runs = std::env::var("BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);

    // Common-main heights of the captured present AIRs, recorded before
    // `ref_raw` is consumed by `build_ref_ctx`.
    let trace_heights: Vec<usize> = ref_raw.iter().map(|(_, cm, _, _)| cm.height()).collect();

    let ctx = build_ref_ctx(ref_raw, params);

    #[cfg(not(feature = "cuda"))]
    let (default_platform, prove_runs) = {
        let engine = BabyBearPoseidon2RefEngine::<DuplexSponge>::new(params.clone());
        let runs_v = measure_baseline(&engine, &ctx, vm_pk, runs);
        ("cpu", runs_v)
    };
    #[cfg(feature = "cuda")]
    let (default_platform, prove_runs) = {
        use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
        let engine = BabyBearPoseidon2GpuEngine::new(params.clone());
        let runs_v = measure_baseline(&engine, &ctx, vm_pk, runs);
        ("gpu", runs_v)
    };

    let platform =
        std::env::var("BENCH_PLATFORM_LABEL").unwrap_or_else(|_| default_platform.to_string());
    let prove_min = prove_runs.iter().copied().fold(f64::INFINITY, f64::min);
    let prove_mean = prove_runs.iter().sum::<f64>() / prove_runs.len() as f64;

    let whir = &params.whir;
    let baseline = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921), real fibonacci guest (n=100)",
        "platform": platform,
        "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "runs": runs,
        "trace_heights": trace_heights,
        "params": {
            "l_skip": params.l_skip,
            "n_stack": params.n_stack,
            "log_blowup": params.log_blowup,
            "k_whir": whir.k,
            "logup_pow_bits": params.logup.pow_bits,
            "max_constraint_degree": params.max_constraint_degree,
            "mu_pow_bits": whir.mu_pow_bits,
            "folding_pow_bits": whir.folding_pow_bits,
            "query_phase_pow_bits": whir.query_phase_pow_bits,
        },
        "num_queries": whir.rounds.iter().map(|r| r.num_queries).collect::<Vec<_>>(),
        // The e2e prove number every per-stage issue compares against. Excludes
        // setup (keygen/tracegen/transport); matches `prove_chain`'s scope.
        "prove_e2e_s": {
            "warm_min": prove_min,
            "warm_mean": prove_mean,
            "runs": prove_runs,
        },
        // Real-block setup (the SDK `app_keygen` + the `prove_continuations`
        // tracegen) runs upstream and is not separately timed here; the
        // apples-to-apples scope vs `prove_chain` is `prove_e2e_s`.
        "setup_s": {
            "keygen": 0.0,
            "tracegen": 0.0,
        },
    });
    if let Some(parent) = out_file.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(out_file, serde_json::to_string_pretty(&baseline).unwrap()).unwrap();
    println!(
        "baseline written to {} (prove warm_min={prove_min:.3}s over {runs} runs, platform={platform})",
        out_file.display()
    );
}

/// Warm prove-timing loop: time `prover.prove(&d_pk, d_ctx)` ALONE — the
/// trace-in -> proof-out step, exactly openvm-zorch `prove_chain`'s scope. A
/// fresh transcript + freshly transported ctx each run (both are consumed by
/// `prove`), but only the `prove` call itself is timed. On GPU `prove` is async
/// on the current stream, so bracket the timer with stream syncs (see
/// [`sync_stream`]); on CPU both syncs are no-ops. Generic over the engine so
/// the CPU reference engine and the CUDA `BabyBearPoseidon2GpuEngine` share one
/// timing path.
fn measure_baseline<E: StarkEngine<SC = SC>>(
    engine: &E,
    ctx: &ProvingContext<CpuColMajorBackend<SC>>,
    vm_pk: &openvm_stark_backend::keygen::types::MultiStarkProvingKey<SC>,
    runs: usize,
) -> Vec<f64> {
    use openvm_stark_backend::prover::{DeviceDataTransporter, Prover};

    let device = engine.device();
    let d_pk = device.transport_pk_to_device(vm_pk);

    let mut prove_runs = Vec::with_capacity(runs);
    for i in 0..runs {
        let d_ctx = device.transport_proving_ctx_to_device(ctx);
        let mut prover = engine.prover_from_transcript(engine.initial_transcript());
        sync_stream();
        let t = Instant::now();
        let proof = prover.prove(&d_pk, d_ctx).unwrap();
        sync_stream();
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(&proof);
        prove_runs.push(dt);
        println!("[baseline] prove run {}/{}: {dt:.3}s", i + 1, runs);
    }
    prove_runs
}

/// Dump the per-stage end-of-chain `outputs/` (canonical-u32 `.npy`), mirroring
/// openvm-zorch fixture-gen's `gen_prove_fixture` byte-for-byte: the same file
/// names, shapes, and value sources, so the epic-integration consumer
/// (`openvm_zorch/verify_prove.py::_byte_match`) can byte-match `prove_chain`'s
/// `Proof` against the real-block dump exactly as it does the synthetic one.
/// Values come from the `Proof<SC>` struct (gkr/bcp/stacking/whir) + the named
/// challenges recovered by the four log walks (xi/lambda/r/u/mu).
#[allow(clippy::too_many_arguments)]
fn dump_ref_prove_outputs(
    outputs: &Path,
    log: &TranscriptLog<F, [F; 16]>,
    gkr: &GkrProof<SC>,
    bcp: &BatchConstraintProof<SC>,
    sp: &StackingProof<SC>,
    wp: &WhirProof<SC>,
    _params: &SystemParams,
    gkr_walk: &GkrLogWalk,
    walk3: &ZerocheckLogWalk,
    walk4: &StackingLogWalk,
    walk5: &WhirLogWalk,
) {
    // Stage 1 + 2.
    write_npy_u32(
        &outputs.join("common_main_commit.npy"),
        &[8],
        &log.values()[8..16]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>(),
    );
    write_npy_u32(
        &outputs.join("logup_pow_witness.npy"),
        &[1],
        &[gkr.logup_pow_witness.as_canonical_u32()],
    );
    write_npy_u32(&outputs.join("q0_claim.npy"), &[4], &ef_limbs(gkr.q0_claim));
    let xi_flat: Vec<u32> = gkr_walk.xi.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("xi.npy"), &[gkr_walk.xi.len(), 4], &xi_flat);
    // Stage 3.
    write_npy_u32(
        &outputs.join("zc_lambda.npy"),
        &[4],
        &ef_limbs(walk3.lambda),
    );
    let s0_flat: Vec<u32> = bcp
        .univariate_round_coeffs
        .iter()
        .flat_map(|&c| ef_limbs(c))
        .collect();
    write_npy_u32(
        &outputs.join("zc_s0_coeffs.npy"),
        &[bcp.univariate_round_coeffs.len(), 4],
        &s0_flat,
    );
    let r_flat: Vec<u32> = walk3.r.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("zc_r.npy"), &[walk3.r.len(), 4], &r_flat);
    // Stage 4.
    write_npy_u32(
        &outputs.join("st_lambda.npy"),
        &[4],
        &ef_limbs(walk4.lambda),
    );
    let u_flat: Vec<u32> = walk4.u.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("st_u.npy"), &[walk4.u.len(), 4], &u_flat);
    let open_flat: Vec<u32> = sp.stacking_openings[0]
        .iter()
        .flat_map(|&e| ef_limbs(e))
        .collect();
    write_npy_u32(
        &outputs.join("st_openings_c0.npy"),
        &[sp.stacking_openings[0].len(), 4],
        &open_flat,
    );
    // Stage 5.
    write_npy_u32(&outputs.join("whir_mu.npy"), &[4], &ef_limbs(walk5.mu));
    write_npy_u32(
        &outputs.join("whir_mu_pow_witness.npy"),
        &[1],
        &[wp.mu_pow_witness.as_canonical_u32()],
    );
    let sumcheck_flat: Vec<u32> = wp
        .whir_sumcheck_polys
        .iter()
        .flat_map(|evals| evals.iter().flat_map(|&e| ef_limbs(e)))
        .collect();
    write_npy_u32(
        &outputs.join("whir_sumcheck_polys.npy"),
        &[wp.whir_sumcheck_polys.len(), 2, 4],
        &sumcheck_flat,
    );
    let commits_flat: Vec<u32> = wp
        .codeword_commits
        .iter()
        .flat_map(|d| d.iter().map(|x| x.as_canonical_u32()))
        .collect();
    write_npy_u32(
        &outputs.join("whir_codeword_commits.npy"),
        &[wp.codeword_commits.len(), 8],
        &commits_flat,
    );
    let ood_flat: Vec<u32> = wp.ood_values.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(
        &outputs.join("whir_ood_values.npy"),
        &[wp.ood_values.len(), 4],
        &ood_flat,
    );
    write_npy_u32(
        &outputs.join("whir_folding_pow_witnesses.npy"),
        &[wp.folding_pow_witnesses.len()],
        &wp.folding_pow_witnesses
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>(),
    );
    write_npy_u32(
        &outputs.join("whir_query_phase_pow_witnesses.npy"),
        &[wp.query_phase_pow_witnesses.len()],
        &wp.query_phase_pow_witnesses
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>(),
    );
    let final_flat: Vec<u32> = wp.final_poly.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(
        &outputs.join("whir_final_poly.npy"),
        &[wp.final_poly.len(), 4],
        &final_flat,
    );
}

fn main() -> eyre::Result<()> {
    // --- arg parse: optional `--out <dir>`, `--ref-prove`, `--baseline-out <file>` ---
    let mut out_dir: Option<PathBuf> = None;
    let mut ref_prove = false;
    let mut baseline_out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => {
                // Fail fast: dumping is this tool's whole purpose, so a bare
                // `--out` (no path) must error, not silently fall back to the
                // print-only path.
                let path = args
                    .next()
                    .ok_or_else(|| eyre::eyre!("--out requires a directory path"))?;
                out_dir = Some(PathBuf::from(path));
            }
            // A3 prototype slice (issue #52): after the tap captures the real
            // per-AIR traces, re-prove the SAME ctx+pk with the reference
            // RECORDING engine to confirm the recording-prove path is viable.
            "--ref-prove" => ref_prove = true,
            // Native-prover baseline: time the native SWIRL prover on the tapped
            // real ctx and write a timing JSON (`gen_baseline`). Same fail-fast
            // rule as `--out`: a bare `--baseline-out` (no path) must error.
            "--baseline-out" => {
                let path = args
                    .next()
                    .ok_or_else(|| eyre::eyre!("--baseline-out requires a file path"))?;
                baseline_out = Some(PathBuf::from(path));
            }
            other => eyre::bail!(
                "unknown arg {other}; usage: [--out <dir>] [--ref-prove] [--baseline-out <file>]"
            ),
        }
    }

    let vm_config = SdkVmConfig::from_toml(include_str!("../../../guest/fibonacci/openvm.toml"))?;

    let elf = Elf::decode(
        include_bytes!("../../../guest/fibonacci/elf/openvm-fibonacci-program.elf"),
        MEM_SIZE as u32,
    )?;

    // A small input so the real block stays small but exercises the full chip
    // inventory of a real guest.
    let n = 100u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    // Mirror `run_app_benchmark`: build the SDK + app proving key.
    let app_config: AppConfig<SdkVmConfig> = AppConfig::new(
        vm_config,
        openvm_benchmarks_prove::default_bench_app_params(),
    );
    let sdk: CpuSdk = Sdk::new(app_config, Default::default())?;
    let (app_pk, _app_vk): (AppProvingKey<SdkVmConfig>, _) = sdk.app_keygen();

    let exe = sdk.convert_to_exe(elf)?;

    // Build a mutable VmInstance so we can tap the ProvingContext. The SDK's
    // `AppProver` only exposes the instance immutably, so build the local prover
    // directly (no lib change needed).
    let mut instance = new_local_prover::<openvm_sdk::DefaultStarkEngine, _>(
        *sdk.app_vm_builder(),
        &app_pk.app_vm_pk,
        exe,
    )?;

    // The proving key holds the per-AIR symbolic constraints (vkey order) plus
    // the real system params and vk pre-hash. We index into it by `air_id`
    // inside the tap closure.
    let vm_pk = app_pk.app_vm_pk.vm_pk.clone();
    let params = vm_pk.params.clone();

    // If dumping, set up the output dirs once up front.
    let inputs_dir = out_dir.as_ref().map(|d| d.join("inputs"));
    if let Some(inputs) = &inputs_dir {
        fs::create_dir_all(inputs)?;
    }

    // Whole-block tally accumulated across every (segment, AIR) tapped.
    let mut block_tally = Tally::default();
    let mut segment_count = 0usize;
    // (height*width, seg_idx, air_id, air_name, height, width) for the largest AIRs.
    let mut largest: Vec<(usize, usize, usize, String, usize, usize)> = Vec::new();

    // Per-AIR meta entries collected for the dumped segment. Each is
    // (air_idx, height, AIR meta json). Heights are kept so we can build the
    // descending-height `sorted_airs` after the tap.
    let mut air_meta: Vec<(usize, usize, serde_json::Value)> = Vec::new();
    let mut dumped_segment: Option<usize> = None;

    // `--ref-prove` / `--baseline-out`: the real per-AIR raw traces of the first
    // segment. We keep them as `RowMajorMatrix` (the CpuBackend layout) and
    // rebuild the col-major reference ctx AFTER the tap (see `build_ref_ctx`),
    // where the reference config/hasher is available to re-commit cached mains.
    // Each entry is a `RawAir` (`(air_id, common_main, cached_mains, pvs)`).
    let mut ref_raw: Vec<RawAir> = Vec::new();
    let mut ref_captured_segment: Option<usize> = None;

    let _proof = instance.prove_continuations(stdin, |seg_idx, ctx| {
        segment_count = segment_count.max(seg_idx + 1);

        // Dump only the first segment (the single fibonacci block). A second
        // segment would overwrite the same files, so guard against it.
        let dump_this = inputs_dir.is_some() && dumped_segment.is_none();

        // `--ref-prove` / `--baseline-out`: capture only the first segment's real
        // per-AIR traces. We MUST preserve `cached_mains` (the partitioned-main
        // structure): some AIRs (e.g. ProgramAir) put their real columns in a
        // cached partition and a width-1 frequency matrix in `common_main`.
        // Collapsing to common-main only mis-indexes the partitioned-main
        // constraints. Same first-segment guard as the dump path (a second
        // segment would clobber).
        let ref_capture_this =
            (ref_prove || baseline_out.is_some()) && ref_captured_segment.is_none();
        if ref_capture_this {
            for (air_id, air_ctx) in ctx.per_trace.iter() {
                let cached: Vec<RowMajorMatrix<F>> = air_ctx
                    .cached_mains
                    .iter()
                    .map(|cd| cd.trace.clone())
                    .collect();
                ref_raw.push((
                    *air_id,
                    air_ctx.common_main.clone(),
                    cached,
                    air_ctx.public_values.clone(),
                ));
            }
            ref_captured_segment = Some(seg_idx);
        }

        // Diagnostics are pure noise when dumping (the same data is serialized
        // to disk), so gate every diagnostic compute + print on the print-only
        // path.
        if out_dir.is_none() {
            println!("================ SEGMENT {seg_idx} ================");
            println!(
                "  AIRs with non-empty trace: {}",
                ctx.per_trace.len()
            );
            if inputs_dir.is_some() && dumped_segment.is_some() && dumped_segment != Some(seg_idx) {
                println!(
                    "  [--out] WARNING: more than one segment; only segment {} was dumped",
                    dumped_segment.unwrap()
                );
            }
        }

        for (air_id, air_ctx) in ctx.per_trace.iter() {
            let air_id = *air_id;
            let height = air_ctx.common_main.height();
            let width = air_ctx.common_main.width();

            let pk = &vm_pk.per_air[air_id];
            let air_name = &pk.air_name;
            let dag = &pk.vk.symbolic_constraints;
            let nodes = &dag.constraints.nodes;
            let interactions = &dag.interactions;

            if out_dir.is_none() {
                let n_pvs = air_ctx.public_values.len();
                largest.push((
                    height * width,
                    seg_idx,
                    air_id,
                    air_name.clone(),
                    height,
                    width,
                ));

                // Per-AIR interaction breakdown.
                let mut air_tally = Tally::default();
                for interaction in interactions {
                    let k = classify(nodes, interaction.count);
                    air_tally.add_count(k);
                    block_tally.add_count(k);
                    for &field_idx in &interaction.message {
                        let k = classify(nodes, field_idx);
                        air_tally.add_message(k);
                        block_tally.add_message(k);
                    }
                }

                println!(
                    "  AIR {air_id:>3} {air_name}: trace {height} x {width}, #public_values={n_pvs}, #interactions={}",
                    interactions.len()
                );
                if !interactions.is_empty() {
                    println!("        count   fields: {}", fmt_kinds(&air_tally.count_fields));
                    println!("        message fields: {}", fmt_kinds(&air_tally.message_fields));
                }
            }

            // --- Disk dump for this AIR ---
            if dump_this {
                let inputs = inputs_dir.as_ref().unwrap();
                write_matrix(
                    &inputs.join(format!("trace_{air_id}.npy")),
                    &air_ctx.common_main,
                );
                // Cached-main partitions: each `cached_mains[k].trace` is a
                // `RowMajorMatrix` here (the app `CpuBackend`'s `Matrix`), so it
                // dumps in the same `(height, width)` layout as `trace_<air>.npy`
                // and zorch loads `cached_<air>_<k>.npy` exactly like the common
                // main. The partitioned main the DAG indexes is `cached_mains ++
                // [common_main]`.
                for (k, cd) in air_ctx.cached_mains.iter().enumerate() {
                    write_matrix(
                        &inputs.join(format!("cached_{air_id}_{k}.npy")),
                        &cd.trace,
                    );
                }
                let dag_json = constraints_dag_json(dag);
                fs::write(
                    inputs.join(format!("constraints_{air_id}.json")),
                    serde_json::to_string_pretty(&dag_json).unwrap(),
                )
                .unwrap();

                // Interactions in node-index form: `count_idx` / `message_idxs`
                // index into this AIR's `constraints_<air_id>.json` node DAG;
                // `count_weight` is the verifier's per-message height weight.
                let interactions_json: Vec<serde_json::Value> = interactions
                    .iter()
                    .map(|i| {
                        serde_json::json!({
                            "bus": i.bus_index,
                            "count_weight": i.count_weight,
                            "count_idx": i.count,
                            "message_idxs": i.message,
                        })
                    })
                    .collect();

                air_meta.push((
                    air_id,
                    height,
                    serde_json::json!({
                        "air_idx": air_id,
                        "is_required": pk.vk.is_required,
                        "needs_next": pk.vk.params.need_rot,
                        "constraint_degree": pk.vk.max_constraint_degree,
                        "num_cached_mains": air_ctx.cached_mains.len(),
                        "public_values": air_ctx
                            .public_values
                            .iter()
                            .map(|x| x.as_canonical_u32())
                            .collect::<Vec<_>>(),
                        "interactions": interactions_json,
                    }),
                ));
            }
        }

        if dump_this {
            dumped_segment = Some(seg_idx);
        }
    })?;

    // --- Native-prover baseline (`--baseline-out <file>`) ---
    //
    // Time the native SWIRL prover on the tapped real ctx and write a timing
    // JSON. Builds its own ctx (via `build_ref_ctx`), so it works WITHOUT
    // `--ref-prove`. When both flags are given, `build_ref_ctx` consumes
    // `ref_raw`, so hand the baseline a clone here and leave the original for the
    // `--ref-prove` block below.
    if let Some(out_file) = &baseline_out {
        if ref_captured_segment.is_none() {
            eyre::bail!("--baseline-out given but no segment was produced");
        }
        println!();
        println!("================ NATIVE BASELINE (--baseline-out) ================");
        let raw_for_baseline = if ref_prove {
            ref_raw.clone()
        } else {
            std::mem::take(&mut ref_raw)
        };
        gen_baseline(out_file, raw_for_baseline, &params, &vm_pk);
    }

    // --- A3 prototype slice (issue #52): reference RECORDING prove ---
    //
    // Re-prove the SAME real ctx + the SAME app proving key with the reference
    // engine's recording transcript. This is the central de-risk: does the
    // recording prover run on the tapped real traces, and what is the
    // proof/transcript-log shape? (The full log-walk + outputs/ dump is NOT
    // wired here — this only proves the path is viable.)
    //
    // Holders for two meta.json fields populated inside this block (they need
    // the present-set + the recorded transcript). Emitted in meta.json so
    // zorch's CommitRound can iterate the real vk prelude and diff its
    // transcript element-by-element against ground truth (issue #59).
    let mut vk_prelude_json: Option<serde_json::Value> = None;
    let mut obs_log_json: Option<serde_json::Value> = None;
    if ref_prove {
        if ref_captured_segment.is_none() {
            eyre::bail!("--ref-prove given but no segment was produced");
        }
        let n_airs = ref_raw.len();
        let n_cached: usize = ref_raw.iter().map(|(_, _, c, _)| c.len()).sum();
        println!();
        println!("================ REF RECORDING PROVE (--ref-prove) ================");
        println!(
            "  captured segment {} with {n_airs} non-empty AIRs ({n_cached} cached-main partitions)",
            ref_captured_segment.unwrap()
        );

        // Capture per-present-AIR shape BEFORE the loop below consumes `ref_raw`,
        // so the protocol-size derivation (prelude_len / n_logup / n_max) can use
        // the real present set, heights, cached counts, and public-value counts.
        // `present[air_id] = (height, num_cached, n_public_values)`.
        let present: BTreeMap<usize, (usize, usize, usize)> = ref_raw
            .iter()
            .map(|(air_id, common_main, cached, pvs)| {
                (*air_id, (common_main.height(), cached.len(), pvs.len()))
            })
            .collect();

        // The reference engine carries the SAME `SystemParams` as the app pk
        // (`vm_pk.params`), so its derived config matches the app's vk.
        let ref_engine = BabyBearPoseidon2RefEngine::<DuplexSpongeRecorder>::new(params.clone());

        // Build the col-major reference ctx from the captured real traces (shared
        // with the `--baseline-out` path; see `build_ref_ctx`).
        let ctx = build_ref_ctx(ref_raw, &params);

        // `prove_from_transcript` transports both the pk and the ctx to the
        // reference device internally; the app proving key is
        // `MultiStarkProvingKey<SC>` over the SAME `SC` the reference engine uses,
        // so it transports directly — no separate keygen needed.
        let fixture = RealCtxFixture {
            ctx: RefCell::new(Some(ctx)),
        };

        // Recording prove: `prove_from_transcript` runs the prover and writes the
        // final recorder back into `recorder` (it has crate-internal access to the
        // `Coordinator`'s transcript), so `into_log` yields the full log.
        let mut recorder = default_duplex_sponge_recorder();
        let proof = fixture.prove_from_transcript(&ref_engine, &vm_pk, &mut recorder);
        let log = recorder.into_log();

        println!("  REF RECORDING PROVE RAN OK");
        println!("  proof.common_main_commit: {:?}", proof.common_main_commit);
        println!(
            "  proof.trace_vdata: {} entries ({} present)",
            proof.trace_vdata.len(),
            proof.trace_vdata.iter().filter(|t| t.is_some()).count()
        );
        println!(
            "  proof.public_values: {} AIRs, {} non-empty",
            proof.public_values.len(),
            proof.public_values.iter().filter(|p| !p.is_empty()).count()
        );
        println!(
            "  gkr_proof: {} layer-claims, {} sumcheck-rounds",
            proof.gkr_proof.claims_per_layer.len(),
            proof.gkr_proof.sumcheck_polys.len()
        );
        println!(
            "  batch_constraint_proof: {} numerator-terms (sorted-AIR order)",
            proof.batch_constraint_proof.numerator_term_per_air.len()
        );
        println!("  stacking_proof present; whir_proof present");
        println!("  recorded transcript log length: {}", log.len());
        println!(
            "  recorded transcript: {} observed / {} sampled",
            log.samples().iter().filter(|s| !**s).count(),
            log.samples().iter().filter(|s| **s).count()
        );

        // --- A3 slice: run the VENDORED generic GKR walk on the REAL log ---
        //
        // Derive the protocol sizes the walk needs from the REAL proving key +
        // present traces, MIRRORING fixture-gen's `prove_instance_with`
        // derivation but sourced from the real `vm_pk` instead of the 5-AIR
        // synthetic specs. The walk asserts the observe/sample structure as it
        // goes, so a real-block mismatch surfaces as an assertion failure (which
        // we catch and report rather than aborting).
        let l_skip = params.l_skip;
        let pow_bits = params.logup.pow_bits;
        let gkr = &proof.gkr_proof;

        // `total_interactions`: per present AIR, (#interactions) << max(log_height,
        // l_skip), summed in DESCENDING-height (sorted) order — same lift as
        // fixture-gen. The lift is height-only and order-independent for the sum,
        // but we follow the sorted iteration for fidelity.
        let mut sorted_present: Vec<usize> = present.keys().copied().collect();
        sorted_present.sort_by_key(|&i| (std::cmp::Reverse(present[&i].0), i));
        let mut total_interactions = 0u64;
        for &air_id in &sorted_present {
            let (height, _, _) = present[&air_id];
            let n_int = vm_pk.per_air[air_id]
                .vk
                .symbolic_constraints
                .interactions
                .len();
            let log_height = height.ilog2() as usize;
            let log_lifted = log_height.max(l_skip);
            total_interactions += (n_int as u64) << log_lifted;
        }
        let n_logup = calculate_n_logup(l_skip, total_interactions);
        let total_rounds = l_skip + n_logup;
        // `n_max`: max over present traces of (log_height - l_skip), saturating.
        let n_max = present
            .values()
            .map(|(height, _, _)| (height.ilog2() as usize).saturating_sub(l_skip))
            .max()
            .unwrap_or(0);
        let n_global = n_max.max(n_logup);

        // --- prelude_len, computed two ways for diagnosis ---
        //
        // (A) NAIVE: fixture-gen's `prove_instance_with` formula applied verbatim
        //     to the real pk. fixture-gen's synthetic block has EVERY pk AIR
        //     present, so it unconditionally adds the preprocessed/log_height and
        //     cached terms for every `per_air` entry, and pulls public-value
        //     counts from the (always-present) specs. On the real block, the pk
        //     has more AIRs than are present.
        //
        // (B) FAITHFUL: mirrors the real Coordinator prelude loop
        //     (`Coordinator::prove`): iterate ALL `per_air`; always observe a
        //     present-flag when `!is_required`; observe preprocessed-commit(8) /
        //     log_height(1) and cached-commits(8 each) ONLY for PRESENT AIRs;
        //     observe public_values for present AIRs (absent AIRs contribute 0).
        //
        // The walk starts at `prelude_len`; if (A) != (B) the naive formula is
        // wrong for the real block and the walk (started at the naive index)
        // would assert. We feed the FAITHFUL value to the walk and report both.
        let mut prelude_len_naive = 16usize;
        let mut prelude_len_faithful = 16usize;
        println!();
        println!("  --- prelude_len derivation (per-AIR) ---");
        println!(
            "    total pk AIRs: {}, present: {}",
            vm_pk.per_air.len(),
            present.len()
        );
        let mut printed = 0usize;
        // Accumulate the FULL vk-prelude structure (one entry per vk position,
        // present AND absent) for meta.json: the array length is the total vk
        // AIR count and `has_preprocessed` disambiguates the 1-elt log_height
        // vs 8-elt preprocessed-commit observe — the two facts the fixture
        // could not previously express (issue #59).
        let mut vk_prelude_entries: Vec<serde_json::Value> =
            Vec::with_capacity(vm_pk.per_air.len());
        let mut any_present_preproc = false;
        for (air_idx, pk_air) in vm_pk.per_air.iter().enumerate() {
            let is_present = present.contains_key(&air_idx);
            let n_cached_vk = pk_air.vk.num_cached_mains();
            let has_pre = pk_air.preprocessed_data.is_some();
            let is_required = pk_air.vk.is_required;

            // (A) naive: matches fixture-gen exactly, using present public-value
            // counts where available (absent AIRs have no captured pvs → 0,
            // which already diverges from fixture-gen's always-present specs).
            let n_pvs = present.get(&air_idx).map(|(_, _, p)| *p).unwrap_or(0);

            vk_prelude_entries.push(serde_json::json!({
                "air_idx": air_idx,
                "present": is_present,
                "is_required": is_required,
                "has_preprocessed": has_pre,
                "num_cached_mains": n_cached_vk,
                "n_public_values": n_pvs,
            }));
            any_present_preproc |= is_present && has_pre;

            let mut naive_delta = 0usize;
            if !is_required {
                naive_delta += 1;
            }
            naive_delta += if has_pre { 8 } else { 1 };
            naive_delta += 8 * n_cached_vk;
            naive_delta += n_pvs;
            prelude_len_naive += naive_delta;

            // (B) faithful: gate the preprocessed/log_height + cached terms on
            // PRESENCE.
            let mut faithful_delta = 0usize;
            if !is_required {
                faithful_delta += 1;
            }
            if is_present {
                faithful_delta += if has_pre { 8 } else { 1 };
                faithful_delta += 8 * present[&air_idx].1; // cached commitments observed = ctx
                                                           // cached_mains
            }
            faithful_delta += n_pvs;
            prelude_len_faithful += faithful_delta;

            // Print the first handful + any AIR where the two formulas diverge.
            if printed < 6 || naive_delta != faithful_delta {
                println!(
                    "    AIR {air_idx:>3} {}: present={is_present} required={is_required} pre={has_pre} cached_vk={n_cached_vk} n_pvs={n_pvs} -> naive +{naive_delta}, faithful +{faithful_delta}",
                    pk_air.air_name
                );
                printed += 1;
            }
        }
        vk_prelude_json = Some(serde_json::Value::Array(vk_prelude_entries));
        if any_present_preproc {
            eprintln!(
                "  WARNING: a present AIR carries a preprocessed trace; its \
                 8-elt preprocessed-commit root is in obs_log but NOT a \
                 structured field — zorch's generation of preprocessed observes \
                 is unverified on this fixture (today's /tmp/real_fib has none)."
            );
        }
        println!(
            "  prelude_len: naive(fixture-gen formula)={prelude_len_naive}, faithful(real-coordinator)={prelude_len_faithful}"
        );
        println!(
            "  derived sizes: l_skip={l_skip}, pow_bits={pow_bits}, total_interactions={total_interactions}, n_logup={n_logup}, total_rounds={total_rounds}, n_max={n_max}, n_global={n_global}"
        );
        println!(
            "  gkr_proof: q0_claim present, claims_per_layer={}, sumcheck_polys layers={}",
            gkr.claims_per_layer.len(),
            gkr.sumcheck_polys.len()
        );

        // Sanity: the faithful prelude_len must point at the logup_pow witness
        // observe (first transcript entry after the prelude). Report what is
        // actually there for both candidate indices.
        let probe = |idx: usize| -> String {
            if idx < log.len() {
                format!(
                    "value={}, is_sample={}",
                    log.values()[idx].as_canonical_u32(),
                    log.samples()[idx]
                )
            } else {
                "out-of-range".to_string()
            }
        };
        println!(
            "  log[naive={prelude_len_naive}]: {} | log[faithful={prelude_len_faithful}]: {} | logup_pow_witness={}",
            probe(prelude_len_naive),
            probe(prelude_len_faithful),
            gkr.logup_pow_witness.as_canonical_u32()
        );

        // Capture the raw reference prelude observation-log prefix (the whole
        // prelude plus a 32-entry margin into the GKR section, for
        // re-convergence context) so zorch can diff its transcript
        // element-by-element against ground truth instead of inferring
        // divergence from MISMATCH labels (issue #59). Values are canonical
        // (non-Montgomery) u32, matching the rest of the fixture.
        let obs_end = (prelude_len_faithful + 32).min(log.len());
        obs_log_json = Some(serde_json::json!({
            "prelude_len_faithful": prelude_len_faithful,
            "len": obs_end,
            "values": log.values()[..obs_end]
                .iter()
                .map(|v| v.as_canonical_u32())
                .collect::<Vec<u32>>(),
            "samples": log.samples()[..obs_end].to_vec(),
        }));

        // --- Run the vendored walk, catching any assert-failure ---
        println!();
        println!("  --- running vendored walk_gkr_log on the REAL log ---");
        let walk_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            walk_gkr_log(
                &log,
                prelude_len_faithful,
                pow_bits,
                gkr.logup_pow_witness,
                gkr,
                l_skip,
                total_rounds,
                n_global,
            )
        }));
        // Helper: extract a panic payload's message (the `assert_eq!` text the
        // walks emit on a transcript mismatch — it carries the log index).
        let panic_msg = |e: &Box<dyn std::any::Any + Send>| -> String {
            e.downcast_ref::<String>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string())
        };

        let gkr_walk = match walk_result {
            Ok(walk) => {
                println!("  GKR WALK COMPLETED on the real block.");
                println!(
                    "    alpha: 1 EF, beta: 1 EF, xi: {} EF elements (expected l_skip+n_global={})",
                    walk.xi.len(),
                    l_skip + n_global
                );
                println!(
                    "    idx_after_beta={}, stage2_end={} (transcript_len={})",
                    walk.idx_after_beta,
                    walk.stage2_end,
                    log.len()
                );
                println!(
                    "    recovered alpha[0]={}, beta[0]={}, xi[0][0]={}",
                    ef_limb0(walk.alpha),
                    ef_limb0(walk.beta),
                    ef_limb0(walk.xi[0]),
                );
                Some(walk)
            }
            Err(e) => {
                println!("  GKR WALK FAILED on the real block.");
                println!("    panic message: {}", panic_msg(&e));
                println!(
                    "    (started walk at faithful prelude_len={prelude_len_faithful}; naive would have been {prelude_len_naive})"
                );
                None
            }
        };

        // --- Stages 3/4/5: run the remaining vendored walks sequentially,
        // each starting at the previous stage's end index, each wrapped in
        // `catch_unwind` so a real-block mismatch reports the stage + the log
        // index (carried in the `assert_eq!` panic text) instead of aborting. ---
        if let Some(gkr_walk) = gkr_walk {
            let bcp = &proof.batch_constraint_proof;
            let sp = &proof.stacking_proof;
            let wp = &proof.whir_proof;

            // `needs_next` per PRESENT AIR, in DESCENDING-height sorted order
            // (the order `BatchConstraintProof::column_openings` /
            // `numerator_term_per_air` use), sourced from the real pk's
            // `vk.params.need_rot`. fixture-gen sources the same from its
            // synthetic `sorted_airs`; here `sorted_present` is the real-block
            // analogue (present AIRs only — absent AIRs are not in the BCP).
            let needs_next: Vec<bool> = sorted_present
                .iter()
                .map(|&air_id| vm_pk.per_air[air_id].vk.params.need_rot)
                .collect();

            println!();
            println!("  --- running vendored walk_zerocheck_log / walk_stacking_log / walk_whir_log on the REAL log ---");

            // Stage 3 (zerocheck / batch-constraint).
            let zc_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                walk_zerocheck_log(&log, gkr_walk.stage2_end, bcp, &needs_next)
            }));
            let walk3 = match zc_result {
                Ok(w) => {
                    println!(
                        "  ZEROCHECK WALK COMPLETED: stage2_end={} -> stage3_end={}",
                        gkr_walk.stage2_end, w.stage3_end
                    );
                    Some(w)
                }
                Err(e) => {
                    println!(
                        "  ZEROCHECK WALK FAILED (started at stage2_end={}).",
                        gkr_walk.stage2_end
                    );
                    println!("    panic message: {}", panic_msg(&e));
                    None
                }
            };

            // Stage 4 (stacked opening reduction).
            let walk4 = walk3.as_ref().and_then(|walk3| {
                let st_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    walk_stacking_log(&log, walk3.stage3_end, sp)
                }));
                match st_result {
                    Ok(w) => {
                        println!(
                            "  STACKING WALK COMPLETED: stage3_end={} -> stage4_end={}",
                            walk3.stage3_end, w.stage4_end
                        );
                        Some(w)
                    }
                    Err(e) => {
                        println!(
                            "  STACKING WALK FAILED (started at stage3_end={}).",
                            walk3.stage3_end
                        );
                        println!("    panic message: {}", panic_msg(&e));
                        None
                    }
                }
            });

            // Stage 5 (WHIR opening).
            let walk5 = walk4.as_ref().and_then(|walk4| {
                let whir_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    walk_whir_log(&log, walk4.stage4_end, &params, wp)
                }));
                match whir_result {
                    Ok(w) => {
                        let landed = w.stage5_end == log.len();
                        println!(
                            "  WHIR WALK COMPLETED: stage4_end={} -> stage5_end={} (transcript_len={}, lands-on-end={landed})",
                            walk4.stage4_end,
                            w.stage5_end,
                            log.len()
                        );
                        if !landed {
                            println!(
                                "    WARNING: stage5_end {} != transcript_len {} (walk did not consume the whole log)",
                                w.stage5_end,
                                log.len()
                            );
                        }
                        Some(w)
                    }
                    Err(e) => {
                        println!(
                            "  WHIR WALK FAILED (started at stage4_end={}).",
                            walk4.stage4_end
                        );
                        println!("    panic message: {}", panic_msg(&e));
                        None
                    }
                }
            });

            // --- Dump `outputs/` (mirror fixture-gen `gen_prove_fixture`) when
            // all four walks pass. Goes under a `--ref-prove`-specific subdir so
            // the default `--out` dump (root `inputs/` + `meta.json`) stays
            // byte-identical. ---
            if let (Some(walk3), Some(walk4), Some(walk5)) =
                (walk3.as_ref(), walk4.as_ref(), walk5.as_ref())
            {
                let ref_root = out_dir
                    .as_ref()
                    .map(|d| d.join("ref-prove"))
                    .unwrap_or_else(|| PathBuf::from("ref-prove-fixture"));
                let outputs = ref_root.join("outputs");
                fs::create_dir_all(&outputs)?;
                dump_ref_prove_outputs(
                    &outputs, &log, gkr, bcp, sp, wp, &params, &gkr_walk, walk3, walk4, walk5,
                );
                println!();
                println!(
                    "  ALL FOUR WALKS PASSED -> outputs/ dumped to {}",
                    outputs.display()
                );
            } else {
                println!();
                println!("  outputs/ NOT dumped (a stage walk failed; see above).");
            }
        }
    }

    // Whole-block summary: largest AIRs by height x width (collected during the
    // tap, since heights are runtime), plus the aggregate interaction breakdown.
    // Only on the print-only path — when dumping, this data is in the fixture.
    if out_dir.is_none() {
        println!();
        println!("================ WHOLE-BLOCK SUMMARY ================");
        println!("  segments: {segment_count}");
        println!("  total AIRs in proving key: {}", vm_pk.per_air.len());

        largest.sort_by(|a, b| b.0.cmp(&a.0));
        println!("  largest AIRs by height x width:");
        for (cells, seg_idx, air_id, air_name, height, width) in largest.iter().take(8) {
            println!(
                "    seg {seg_idx} AIR {air_id:>3} {air_name}: {height} x {width} = {cells} cells"
            );
        }

        println!(
            "  interaction COUNT fields across block:   {}",
            fmt_kinds(&block_tally.count_fields)
        );
        println!(
            "  interaction MESSAGE fields across block: {}",
            fmt_kinds(&block_tally.message_fields)
        );
    }

    // --- Write meta.json if dumping ---
    if let Some(out) = &out_dir {
        if dumped_segment.is_none() {
            eyre::bail!("--out given but no segment was produced");
        }

        // `airs[]` in ascending global air_idx order (deterministic).
        air_meta.sort_by_key(|(air_id, _, _)| *air_id);
        let airs: Vec<serde_json::Value> = air_meta.iter().map(|(_, _, j)| j.clone()).collect();
        // `sorted_airs`: air indices descending by trace height, ties ascending
        // by air index — exactly `ProvingContext::sort_for_stacking`.
        let mut sorted = air_meta
            .iter()
            .map(|(air_id, h, _)| (*air_id, *h))
            .collect::<Vec<_>>();
        sorted.sort_by_key(|(air_id, h)| (std::cmp::Reverse(*h), *air_id));
        let sorted_airs: Vec<usize> = sorted.iter().map(|(air_id, _)| *air_id).collect();

        let whir = &params.whir;
        let mut meta = serde_json::json!({
            "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921), real fibonacci guest (n=100)",
            "params": {
                "l_skip": params.l_skip,
                "n_stack": params.n_stack,
                "log_blowup": params.log_blowup,
                "k_whir": whir.k,
                "logup_pow_bits": params.logup.pow_bits,
                "max_constraint_degree": params.max_constraint_degree,
                "mu_pow_bits": whir.mu_pow_bits,
                "folding_pow_bits": whir.folding_pow_bits,
                "query_phase_pow_bits": whir.query_phase_pow_bits,
            },
            "num_queries": whir.rounds.iter().map(|r| r.num_queries).collect::<Vec<_>>(),
            "num_whir_rounds": whir.num_whir_rounds(),
            "vk_pre_hash": vm_pk.vk_pre_hash.map(|x| x.as_canonical_u32()),
            "sorted_airs": sorted_airs,
            "airs": airs,
        });
        // Full vk-prelude structure + raw reference observation-log prefix
        // (populated only under --ref-prove; absent otherwise). Issue #59.
        if let Some(v) = vk_prelude_json {
            meta["vk_prelude"] = v;
        }
        if let Some(v) = obs_log_json {
            meta["obs_log"] = v;
        }
        let mut f = fs::File::create(out.join("meta.json"))?;
        writeln!(f, "{}", serde_json::to_string_pretty(&meta).unwrap())?;
        println!();
        println!(
            "real-guest prove fixture written to {} ({} non-empty AIRs)",
            out.display(),
            air_meta.len()
        );
    }

    // Assert the field type is BabyBear-shaped (compile-time use of PrimeField32).
    let _ = F::ORDER_U32;

    Ok(())
}
