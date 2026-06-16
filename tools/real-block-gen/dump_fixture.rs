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
    collections::BTreeMap,
    fs,
    io::Write as _,
    path::{Path, PathBuf},
};

use openvm_sdk::{
    config::AppConfig, keygen::AppProvingKey, prover::vm::new_local_prover, CpuSdk, Sdk, StdIn,
};
use openvm_sdk_config::SdkVmConfig;
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_variable::Entry, SymbolicConstraintsDag, SymbolicExpressionNode,
    },
    p3_matrix::dense::RowMajorMatrix,
    prover::MatrixDimensions,
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use p3_field::PrimeField32;

type F = openvm_sdk::F;

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

/// Dump a `RowMajorMatrix` (the CPU backend's `common_main` type) as a
/// `(height, width)` `<u4` array of canonical u32 cells. Its `values` are
/// already row-major, so write them directly — matching fixture-gen's
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

fn main() -> eyre::Result<()> {
    // --- arg parse: optional `--out <dir>` ---
    let mut out_dir: Option<PathBuf> = None;
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
            other => eyre::bail!("unknown arg {other}; usage: [--out <dir>]"),
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

    let _proof = instance.prove_continuations(stdin, |seg_idx, ctx| {
        segment_count = segment_count.max(seg_idx + 1);

        // Dump only the first segment (the single fibonacci block). A second
        // segment would overwrite the same files, so guard against it.
        let dump_this = inputs_dir.is_some() && dumped_segment.is_none();

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
        let meta = serde_json::json!({
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
