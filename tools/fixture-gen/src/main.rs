//! Golden-fixture generator for openvm-zorch's byte-match tests.
//!
//! Runs the reference prover (openvm-stark-backend v2.0.0-beta.2, BabyBear +
//! Poseidon2 width-16) on deterministic inputs and dumps every intermediate as
//! canonical-u32 `.npy` plus a `meta.json`, so the JAX side can compare each
//! pipeline step independently.
//!
//! Sections (each runs only when its flag is given):
//! - `--out <dir>`: Stage 1 `stacked_commit` vectors + Poseidon2 pins.
//! - `--transcript-out <dir>`: transcript-only vector (observe/sample/ext/
//!   grind sequence on `DuplexSpongeRecorder`) to pin zorch's
//!   `DuplexTranscript` against the Rust `DuplexSponge`.
//! - `--gkr-out <dir>`: Stage 2 LogUp-GKR fixture — a multi-AIR instance with
//!   interactions, proven end-to-end with a recording transcript; dumps the
//!   transcript log, the `GkrProof`, named challenges (alpha/beta/xi) and the
//!   reconstructed GKR input layer.

use std::{fs, io::Write as _, path::Path, path::PathBuf};

use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_variable::Entry, SymbolicConstraintsDag, SymbolicExpressionNode,
    },
    any_air_arc_vec, calculate_n_logup,
    hasher::{Hasher, MerkleHasher},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation},
    poly_common::Squarable,
    proof::{column_openings_by_rot, WhirProof},
    prover::{
        fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
        prove_zerocheck_and_logup,
        stacked_pcs::{stacked_commit, StackedLayout},
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        whir::WhirProver,
        AirProvingContext, ColMajorMatrix, DeviceDataTransporter, MatrixDimensions, ProvingContext,
        TraceCommitter,
    },
    test_utils::{
        default_test_params_small, test_system_params_small,
        dummy_airs::{
            fib_air::{air::FibonacciAir, trace::generate_trace_rows},
            interaction::dummy_interaction_air::DummyInteractionAir,
        },
        TestFixture,
    },
    AirRef, FiatShamirTranscript, ReadOnlyTranscript, StarkEngine, SystemParams,
    TranscriptHistory, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2RefEngine, DuplexSpongeRecorder, EF,
};
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};
use p3_field::{BasedVectorSpace, Field};

type F = BabyBear;
type SC = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

const L_SKIP: usize = 2;
const N_STACK: usize = 3;
const LOG_BLOWUP: usize = 1;
const K_WHIR: usize = 2;
/// (width, height) per trace, descending height as `stacked_commit` requires.
/// Height 2 < 2^L_SKIP exercises the striding path.
const TRACE_DIMS: &[(usize, usize)] = &[(3, 16), (2, 8), (5, 2)];

/// Deterministic trace cell; arbitrary but fixed forever (fixtures regenerate
/// bit-identically from the pin).
fn trace_val(mat: usize, row: usize, col: usize) -> F {
    F::from_u32(((mat as u32 + 1) * 65537 + col as u32 * 4097 + row as u32 * 257 + 1) % 0x78000001)
}

/// Minimal .npy v1 writer: little-endian u32, C order.
fn write_npy_u32(path: &Path, shape: &[usize], data: &[u32]) {
    write_npy(
        path,
        shape,
        "<u4",
        &data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
}

/// Minimal .npy v1 writer: u8, C order.
fn write_npy_u8(path: &Path, shape: &[usize], data: &[u8]) {
    write_npy(path, shape, "|u1", data);
}

fn write_npy(path: &Path, shape: &[usize], descr: &str, bytes: &[u8]) {
    let elem = match descr {
        "<u4" => 4,
        "|u1" => 1,
        _ => unreachable!(),
    };
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
    out.extend(std::iter::repeat(b' ').take(padding));
    out.push(b'\n');
    out.extend_from_slice(bytes);
    fs::write(path, out).unwrap();
}

/// Dump a ColMajorMatrix as a row-major (height, width) array.
fn write_matrix(path: &Path, m: &ColMajorMatrix<F>) {
    let (h, w) = (m.height(), m.width());
    let mut data = Vec::with_capacity(h * w);
    for r in 0..h {
        for c in 0..w {
            data.push(m.column(c)[r].as_canonical_u32());
        }
    }
    write_npy_u32(path, &[h, w], &data);
}

fn ef_limbs(x: EF) -> [u32; 4] {
    let coeffs: &[F] = x.as_basis_coefficients_slice();
    core::array::from_fn(|i| coeffs[i].as_canonical_u32())
}

fn write_transcript_log(dir: &Path, log: &TranscriptLog<F, [F; 16]>) {
    let values: Vec<u32> = log.values().iter().map(|x| x.as_canonical_u32()).collect();
    write_npy_u32(&dir.join("transcript_values.npy"), &[values.len()], &values);
    let flags: Vec<u8> = log.samples().iter().map(|&s| s as u8).collect();
    write_npy_u8(
        &dir.join("transcript_is_sample.npy"),
        &[flags.len()],
        &flags,
    );
    let perms: Vec<u32> = log
        .perm_results()
        .iter()
        .flat_map(|st| st.iter().map(|x| x.as_canonical_u32()))
        .collect();
    write_npy_u32(
        &dir.join("transcript_perm_results.npy"),
        &[log.perm_results().len(), 16],
        &perms,
    );
}

fn gen_stage1_fixture(out: &Path) {
    let inputs = out.join("inputs");
    let outputs = out.join("outputs");
    fs::create_dir_all(&inputs).unwrap();
    fs::create_dir_all(&outputs).unwrap();

    let perm = default_babybear_poseidon2_16();
    let hasher = Hasher::<F, [F; 8], _, _>::new(
        PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone()),
        TruncatedPermutation::<_, 2, 8, 16>::new(perm.clone()),
    );

    // --- Poseidon2 vectors: pin the permutation before any tree ---
    let mut state: [F; 16] = core::array::from_fn(|i| F::from_u32(i as u32));
    perm.permute_mut(&mut state);
    write_npy_u32(
        &outputs.join("perm_0_15.npy"),
        &[16],
        &state.map(|x| x.as_canonical_u32()),
    );
    let sponge_in: Vec<F> = (0..32).map(F::from_u32).collect();
    let digest = hasher.hash_slice(&sponge_in);
    write_npy_u32(
        &outputs.join("sponge_0_31.npy"),
        &[8],
        &digest.map(|x| x.as_canonical_u32()),
    );
    let left: [F; 8] = core::array::from_fn(|i| F::from_u32(i as u32));
    let right: [F; 8] = core::array::from_fn(|i| F::from_u32(100 + i as u32));
    let compressed = hasher.compress(left, right);
    write_npy_u32(
        &outputs.join("compress_pair.npy"),
        &[8],
        &compressed.map(|x| x.as_canonical_u32()),
    );

    // --- Stage 1: stacked_commit on deterministic traces ---
    let traces: Vec<ColMajorMatrix<F>> = TRACE_DIMS
        .iter()
        .enumerate()
        .map(|(mat, &(w, h))| {
            let mut values = Vec::with_capacity(w * h);
            for c in 0..w {
                for r in 0..h {
                    values.push(trace_val(mat, r, c));
                }
            }
            ColMajorMatrix::new(values, w)
        })
        .collect();
    for (i, t) in traces.iter().enumerate() {
        write_matrix(&inputs.join(format!("trace_{i}.npy")), t);
    }

    let trace_refs: Vec<&ColMajorMatrix<F>> = traces.iter().collect();
    let (root, data) =
        stacked_commit(&hasher, L_SKIP, N_STACK, LOG_BLOWUP, K_WHIR, &trace_refs).unwrap();

    write_matrix(&outputs.join("stacked_matrix.npy"), &data.matrix);
    write_matrix(&outputs.join("codeword.npy"), data.tree.backing_matrix());
    for (l, layer) in data.tree.digest_layers().iter().enumerate() {
        let flat: Vec<u32> = layer
            .iter()
            .flat_map(|d| d.iter().map(|x| x.as_canonical_u32()))
            .collect();
        write_npy_u32(
            &outputs.join(format!("digest_layer_{l}.npy")),
            &[layer.len(), 8],
            &flat,
        );
    }
    write_npy_u32(
        &outputs.join("root.npy"),
        &[8],
        &root.map(|x| x.as_canonical_u32()),
    );

    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
        "l_skip": L_SKIP,
        "n_stack": N_STACK,
        "log_blowup": LOG_BLOWUP,
        "k_whir": K_WHIR,
        "trace_dims": TRACE_DIMS.iter().map(|&(w, h)| [h, w]).collect::<Vec<_>>(),
        "stacked_height": data.matrix.height(),
        "stacked_width": data.matrix.width(),
    });
    let mut f = fs::File::create(out.join("meta.json")).unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    println!("stage 1 fixtures written to {}", out.display());
}

/// Transcript-only vector: a fixed observe/sample/ext/grind script whose full
/// log (values, sample flags, post-permutation states) pins zorch's
/// `DuplexTranscript` against the Rust `DuplexSponge` byte-for-byte.
fn gen_transcript_fixture(out: &Path) {
    fs::create_dir_all(out).unwrap();
    let mut ts = default_duplex_sponge_recorder();

    // Single observes, then one sample (forces a partial-block flush).
    for v in 1u32..=5 {
        FiatShamirTranscript::<SC>::observe(&mut ts, F::from_u32(v));
    }
    let _ = FiatShamirTranscript::<SC>::sample(&mut ts);
    // 17 observes cross the rate-8 boundary twice mid-stream.
    for v in 100u32..117 {
        FiatShamirTranscript::<SC>::observe(&mut ts, F::from_u32(v));
    }
    // Drain more than one sample from a single squeeze block.
    for _ in 0..3 {
        let _ = FiatShamirTranscript::<SC>::sample(&mut ts);
    }
    // Extension-field granularity: 4 base limbs each way.
    let e = EF::from_basis_coefficients_fn(|i| F::from_u32(7 + i as u32));
    FiatShamirTranscript::<SC>::observe_ext(&mut ts, e);
    let _ = FiatShamirTranscript::<SC>::sample_ext(&mut ts);
    // Digest observe (8 limbs at once).
    FiatShamirTranscript::<SC>::observe_commit(
        &mut ts,
        core::array::from_fn(|i| F::from_u32(i as u32)),
    );
    // PoW: sequential first-match witness so the fixture is deterministic.
    let pow_bits = 2usize;
    let witness = (0u32..)
        .map(F::from_u32)
        .find(|w| FiatShamirTranscript::<SC>::check_witness(&mut ts.clone(), pow_bits, *w))
        .unwrap();
    assert!(FiatShamirTranscript::<SC>::check_witness(
        &mut ts, pow_bits, witness
    ));
    let _ = FiatShamirTranscript::<SC>::sample(&mut ts);

    let log = ts.into_log();
    write_transcript_log(out, &log);
    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
        "pow_bits": pow_bits,
        "pow_witness": witness.as_canonical_u32(),
        "script": "observe 1..=5; sample; observe 100..117; sample x3; observe_ext [7,8,9,10]; sample_ext; observe_commit [0..8]; check_witness(pow_bits, witness); sample",
    });
    fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!("transcript fixtures written to {}", out.display());
}

/// One interaction of a dummy AIR: count is a single (possibly negated)
/// column, the message is a list of columns. This mirrors exactly what
/// `DummyInteractionAir` pushes, so the JAX side can evaluate interactions
/// from raw trace cells without a symbolic-expression evaluator.
struct IntSpec {
    bus: u16,
    count_col: usize,
    count_neg: bool,
    msg_cols: Vec<usize>,
}

struct AirSpec {
    trace: ColMajorMatrix<F>,
    public_values: Vec<F>,
    interactions: Vec<IntSpec>,
}

struct Stage2Fixture;

impl Stage2Fixture {
    /// AIRs in `air_idx` order:
    /// 0. Fibonacci, height 64, no interactions, 3 public values — the tallest
    ///    trace, so `n_max > n_logup` and the xi-padding edge case is hit.
    /// 1. bus-0 sender, height 4 (== 2^l_skip, no lifting), message width 1.
    /// 2. bus-0 receiver, height 8, message width 1.
    /// 3. bus-1 sender, height 2 (< 2^l_skip: exercises lifting + the
    ///    2^{min(n_T,0)} numerator scaling), message width 2.
    /// 4. bus-1 receiver, height 2, message width 2.
    fn specs(&self) -> Vec<AirSpec> {
        let fib_n = 64usize;
        let (a, b) = (0u64, 1u64);
        let fib_trace = ColMajorMatrix::from_row_major(&generate_trace_rows::<F>(a, b, fib_n));
        let f_n = {
            let (mut x, mut y) = (a, b);
            for _ in 0..fib_n - 1 {
                let z = (x + y) % (F::ORDER_U32 as u64);
                x = y;
                y = z;
            }
            y
        };
        let row_major = |vals: &[u32], w: usize| {
            let h = vals.len() / w;
            let mut col_major = Vec::with_capacity(vals.len());
            for c in 0..w {
                for r in 0..h {
                    col_major.push(F::from_u32(vals[r * w + c]));
                }
            }
            ColMajorMatrix::new(col_major, w)
        };
        // bus-0 pair: the InteractionsFixture11 traces (sums balance).
        let sender0 = row_major(&[0, 1, 3, 5, 7, 4, 546, 889], 2);
        let receiver0 = row_major(
            &[1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456],
            2,
        );
        // bus-1 pair: identical send/receive rows, message width 2.
        let sender1 = row_major(&[2, 10, 11, 3, 20, 21], 3);
        let receiver1 = row_major(&[2, 10, 11, 3, 20, 21], 3);
        vec![
            AirSpec {
                trace: fib_trace,
                public_values: [a, b, f_n].map(F::from_u64).to_vec(),
                interactions: vec![],
            },
            AirSpec {
                trace: sender0,
                public_values: vec![],
                interactions: vec![IntSpec {
                    bus: 0,
                    count_col: 0,
                    count_neg: false,
                    msg_cols: vec![1],
                }],
            },
            AirSpec {
                trace: receiver0,
                public_values: vec![],
                interactions: vec![IntSpec {
                    bus: 0,
                    count_col: 0,
                    count_neg: true,
                    msg_cols: vec![1],
                }],
            },
            AirSpec {
                trace: sender1,
                public_values: vec![],
                interactions: vec![IntSpec {
                    bus: 1,
                    count_col: 0,
                    count_neg: false,
                    msg_cols: vec![1, 2],
                }],
            },
            AirSpec {
                trace: receiver1,
                public_values: vec![],
                interactions: vec![IntSpec {
                    bus: 1,
                    count_col: 0,
                    count_neg: true,
                    msg_cols: vec![1, 2],
                }],
            },
        ]
    }
}

impl TestFixture<SC> for Stage2Fixture {
    fn airs(&self) -> Vec<AirRef<SC>> {
        any_air_arc_vec![
            FibonacciAir,
            DummyInteractionAir::new(1, true, 0),
            DummyInteractionAir::new(1, false, 0),
            DummyInteractionAir::new(2, true, 1),
            DummyInteractionAir::new(2, false, 1),
        ]
    }

    fn generate_proving_ctx(
        &self,
    ) -> ProvingContext<openvm_stark_backend::prover::CpuColMajorBackend<SC>> {
        ProvingContext::new(
            self.specs()
                .into_iter()
                .enumerate()
                .map(|(air_idx, spec)| {
                    (
                        air_idx,
                        AirProvingContext::simple(spec.trace, spec.public_values),
                    )
                })
                .collect(),
        )
    }
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
    gkr: &openvm_stark_backend::proof::GkrProof<SC>,
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

/// Everything Stages 2 and 3 share: the proven 5-AIR instance, its recorded
/// transcript log, the protocol-derived sizes, and the GKR challenge walk.
struct ProvenInstance {
    params: openvm_stark_backend::SystemParams,
    specs: Vec<AirSpec>,
    pk: openvm_stark_backend::keygen::types::MultiStarkProvingKey<SC>,
    proof: openvm_stark_backend::proof::Proof<SC>,
    log: TranscriptLog<F, [F; 16]>,
    sorted_airs: Vec<usize>,
    n_logup: usize,
    n_max: usize,
    n_global: usize,
    total_interactions: u64,
    interactions_layout: StackedLayout,
    prelude_len: usize,
    walk: GkrLogWalk,
}

fn prove_instance() -> ProvenInstance {
    // l_skip=2, n_stack=8, k_whir=3, logup pow_bits=2 — the small test params
    // every backend test uses.
    prove_instance_with(default_test_params_small())
}

fn prove_instance_with(params: SystemParams) -> ProvenInstance {
    let engine = BabyBearPoseidon2RefEngine::<DuplexSpongeRecorder>::new(params.clone());
    let fx = Stage2Fixture;
    let specs = fx.specs();
    let (pk, _vk) = fx.keygen(&engine);

    let mut transcript = default_duplex_sponge_recorder();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut transcript);
    let log = transcript.into_log();
    let gkr = &proof.gkr_proof;

    // --- Replicate the protocol-derived sizes ---
    let l_skip = params.l_skip;
    // Traces sorted descending by height, ties by ascending air index.
    let mut sorted_airs: Vec<usize> = (0..specs.len()).collect();
    sorted_airs.sort_by_key(|&i| (std::cmp::Reverse(specs[i].trace.height()), i));
    let mut total_interactions = 0u64;
    let interactions_meta: Vec<(usize, usize)> = sorted_airs
        .iter()
        .map(|&i| {
            let spec = &specs[i];
            let log_height = specs[i].trace.height().ilog2() as usize;
            let log_lifted = log_height.max(l_skip);
            total_interactions += (spec.interactions.len() as u64) << log_lifted;
            (spec.interactions.len(), log_lifted)
        })
        .collect();
    let n_logup = calculate_n_logup(l_skip, total_interactions);
    let total_rounds = l_skip + n_logup;
    let n_max = specs
        .iter()
        .map(|s| (s.trace.height().ilog2() as usize).saturating_sub(l_skip))
        .max()
        .unwrap();
    let n_global = n_max.max(n_logup);
    let interactions_layout = StackedLayout::new(0, total_rounds, interactions_meta).unwrap();

    // --- Prelude length (Coordinator::prove before prove_zerocheck_and_logup) ---
    // vk_pre_hash (8) + common_main_commit (8) + per air: present flag when
    // not required, preprocessed commit (8) or log_height (1), cached commits
    // (8 each), public values.
    let mut prelude_len = 16;
    for (air_idx, pk_air) in pk.per_air.iter().enumerate() {
        if !pk_air.vk.is_required {
            prelude_len += 1;
        }
        prelude_len += if pk_air.preprocessed_data.is_some() {
            8
        } else {
            1
        };
        prelude_len += 8 * pk_air.vk.num_cached_mains();
        prelude_len += specs[air_idx].public_values.len();
    }

    let walk = walk_gkr_log(
        &log,
        prelude_len,
        params.logup.pow_bits,
        gkr.logup_pow_witness,
        gkr,
        l_skip,
        total_rounds,
        n_global,
    );

    ProvenInstance {
        params,
        specs,
        pk,
        proof,
        log,
        sorted_airs,
        n_logup,
        n_max,
        n_global,
        total_interactions,
        interactions_layout,
        prelude_len,
        walk,
    }
}

fn gen_gkr_fixture(out: &Path) {
    let inputs = out.join("inputs");
    let outputs = out.join("outputs");
    fs::create_dir_all(&inputs).unwrap();
    fs::create_dir_all(&outputs).unwrap();

    let inst = prove_instance();
    let ProvenInstance {
        params,
        specs,
        pk,
        proof,
        log,
        sorted_airs,
        n_logup,
        n_max,
        n_global,
        total_interactions,
        interactions_layout,
        prelude_len,
        walk,
    } = &inst;
    let gkr = &proof.gkr_proof;
    let l_skip = params.l_skip;
    let total_rounds = l_skip + n_logup;

    // --- Reconstruct the GKR input layer from raw trace cells ---
    let max_msg_len = specs
        .iter()
        .flat_map(|s| s.interactions.iter().map(|i| i.msg_cols.len()))
        .max()
        .unwrap_or(0);
    let beta_pows: Vec<EF> = walk.beta.powers().take(max_msg_len + 1).collect();
    let mut evals = vec![Frac::new(EF::ZERO, EF::ZERO); 1 << total_rounds];
    for (t_idx, int_idx, s) in interactions_layout.sorted_cols.iter().copied() {
        let spec = &specs[sorted_airs[t_idx]];
        let int = &spec.interactions[int_idx];
        let trace = &spec.trace;
        let h = trace.height();
        let len = s.len(0);
        let norm = F::from_usize(len / h).inverse();
        for c in 0..len / h {
            for r in 0..h {
                let mut count = trace.column(int.count_col)[r];
                if int.count_neg {
                    count = -count;
                }
                let mut denom = beta_pows[int.msg_cols.len()] * F::from_u32(int.bus as u32 + 1);
                for (j, &mc) in int.msg_cols.iter().enumerate() {
                    denom += beta_pows[j] * trace.column(mc)[r];
                }
                evals[s.row_idx + c * h + r] = Frac::new((count * norm).into(), denom);
            }
        }
    }
    for f in &mut evals {
        f.q += walk.alpha;
    }

    // Validate the reconstruction: replay fractional_sumcheck against the
    // recorded log (ReadOnlyTranscript debug_asserts every observe/sample) and
    // compare the proof it produces with the real one.
    {
        let mut ro = ReadOnlyTranscript::new(&log, walk.idx_after_beta);
        let (fsp, xi_gkr) = fractional_sumcheck::<SC, _>(&mut ro, &evals, true).unwrap();
        assert_eq!(fsp.fractional_sum.0, EF::ZERO);
        assert_eq!(fsp.fractional_sum.1, gkr.q0_claim);
        assert_eq!(fsp.claims_per_layer.len(), gkr.claims_per_layer.len());
        for (a, b) in fsp.claims_per_layer.iter().zip(&gkr.claims_per_layer) {
            assert_eq!(a.p_xi_0, b.p_xi_0);
            assert_eq!(a.p_xi_1, b.p_xi_1);
            assert_eq!(a.q_xi_0, b.q_xi_0);
            assert_eq!(a.q_xi_1, b.q_xi_1);
        }
        assert_eq!(fsp.sumcheck_polys, gkr.sumcheck_polys);
        assert_eq!(&xi_gkr[..], &walk.xi[..total_rounds]);
    }

    // --- Dumps ---
    for (air_idx, spec) in specs.iter().enumerate() {
        write_matrix(&inputs.join(format!("trace_{air_idx}.npy")), &spec.trace);
    }
    write_transcript_log(&outputs, &log);
    write_npy_u32(&outputs.join("alpha.npy"), &[4], &ef_limbs(walk.alpha));
    write_npy_u32(&outputs.join("beta.npy"), &[4], &ef_limbs(walk.beta));
    let xi_flat: Vec<u32> = walk.xi.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("xi.npy"), &[walk.xi.len(), 4], &xi_flat);
    write_npy_u32(&outputs.join("q0_claim.npy"), &[4], &ef_limbs(gkr.q0_claim));
    // Claims in observe order (p_xi_0, q_xi_0, p_xi_1, q_xi_1).
    let claims_flat: Vec<u32> = gkr
        .claims_per_layer
        .iter()
        .flat_map(|c| {
            [c.p_xi_0, c.q_xi_0, c.p_xi_1, c.q_xi_1]
                .into_iter()
                .flat_map(ef_limbs)
        })
        .collect();
    write_npy_u32(
        &outputs.join("claims_per_layer.npy"),
        &[gkr.claims_per_layer.len(), 4, 4],
        &claims_flat,
    );
    for (j, layer_polys) in gkr.sumcheck_polys.iter().enumerate() {
        let flat: Vec<u32> = layer_polys
            .iter()
            .flat_map(|evals3| evals3.iter().flat_map(|&e| ef_limbs(e)))
            .collect();
        write_npy_u32(
            &outputs.join(format!("sumcheck_polys_layer_{j}.npy")),
            &[layer_polys.len(), 3, 4],
            &flat,
        );
    }
    let evals_flat: Vec<u32> = evals
        .iter()
        .flat_map(|f| ef_limbs(f.p).into_iter().chain(ef_limbs(f.q)))
        .collect();
    write_npy_u32(
        &outputs.join("gkr_input_evals.npy"),
        &[evals.len(), 2, 4],
        &evals_flat,
    );

    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
        "params": {
            "l_skip": params.l_skip,
            "n_stack": params.n_stack,
            "log_blowup": params.log_blowup,
            "k_whir": params.whir.k,
            "max_constraint_degree": params.max_constraint_degree,
            "logup_pow_bits": params.logup.pow_bits,
        },
        "airs": specs.iter().enumerate().map(|(air_idx, spec)| serde_json::json!({
            "air_idx": air_idx,
            "height": spec.trace.height(),
            "width": spec.trace.width(),
            "is_required": pk.per_air[air_idx].vk.is_required,
            "public_values": spec.public_values.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>(),
            "interactions": spec.interactions.iter().map(|i| serde_json::json!({
                "bus": i.bus,
                "count_col": i.count_col,
                "count_neg": i.count_neg,
                "message_cols": i.msg_cols,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "sorted_airs": sorted_airs,
        "vk_pre_hash": pk.vk_pre_hash.map(|x| x.as_canonical_u32()),
        "common_main_commit": proof.common_main_commit.map(|x| x.as_canonical_u32()),
        "n_logup": n_logup,
        "n_max": n_max,
        "n_global": n_global,
        "total_interactions": total_interactions,
        "logup_pow_witness": gkr.logup_pow_witness.as_canonical_u32(),
        "interactions_layout": interactions_layout.sorted_cols.iter().map(|(t, i, s)| serde_json::json!({
            "sorted_trace_idx": t,
            "interaction_idx": i,
            "row_idx": s.row_idx,
            "log_height": s.log_height(),
        })).collect::<Vec<_>>(),
        "prelude_len": prelude_len,
        "idx_after_beta": walk.idx_after_beta,
        "stage2_end": walk.stage2_end,
        "transcript_len": log.len(),
    });
    fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!("gkr fixtures written to {}", out.display());
}

/// Canonical-u32 JSON of one AIR's constraints DAG (nodes in topological
/// order, constraint node indices, interactions in node-index form).
/// Hand-rolled rather than serde: BabyBear's serde emits Montgomery-form u32,
/// which would leak an encoding detail into the fixture.
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

/// Stage-3 challenges recovered while walking the recorded log against the
/// `BatchConstraintProof`, asserting the observe/sample structure as it goes.
struct ZerocheckLogWalk {
    lambda: EF,
    mu: EF,
    r: Vec<EF>,
    stage3_end: usize,
}

fn walk_zerocheck_log(
    log: &TranscriptLog<F, [F; 16]>,
    start: usize,
    bcp: &openvm_stark_backend::proof::BatchConstraintProof<SC>,
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

fn gen_zerocheck_fixture(out: &Path) {
    let inputs = out.join("inputs");
    let outputs = out.join("outputs");
    fs::create_dir_all(&inputs).unwrap();
    fs::create_dir_all(&outputs).unwrap();

    let inst = prove_instance();
    let params = &inst.params;
    let l_skip = params.l_skip;
    let bcp = &inst.proof.batch_constraint_proof;
    let num_traces = inst.sorted_airs.len();

    let needs_next: Vec<bool> = inst
        .sorted_airs
        .iter()
        .map(|&i| inst.pk.per_air[i].vk.params.need_rot)
        .collect();
    let n_per_trace: Vec<isize> = inst
        .sorted_airs
        .iter()
        .map(|&i| inst.specs[i].trace.height().ilog2() as isize - l_skip as isize)
        .collect();

    let s_deg = inst.pk.max_constraint_degree + 1;
    let s_0_deg = s_deg * ((1 << l_skip) - 1);
    assert_eq!(bcp.univariate_round_coeffs.len(), s_0_deg + 1);
    assert_eq!(bcp.sumcheck_round_polys.len(), inst.n_max);
    for round_polys in &bcp.sumcheck_round_polys {
        assert_eq!(round_polys.len(), s_deg);
    }
    assert_eq!(bcp.numerator_term_per_air.len(), num_traces);

    let walk3 = walk_zerocheck_log(&inst.log, inst.walk.stage2_end, bcp, &needs_next);

    // Self-validate: rebuild the transcript state at the end of the prelude
    // (all observes — `ReadOnlyTranscript` can't replay the PoW grind, whose
    // witness search observes non-matching candidates) and rerun the whole
    // `prove_zerocheck_and_logup`; it must reproduce the proof and the
    // recorded log through stage3_end.
    {
        let engine = BabyBearPoseidon2RefEngine::<DuplexSpongeRecorder>::new(params.clone());
        let d_pk = engine.device().transport_pk_to_device(&inst.pk);
        let ctx = Stage2Fixture.generate_proving_ctx().into_sorted();
        let mut replay = default_duplex_sponge_recorder();
        for &v in &inst.log.values()[..inst.prelude_len] {
            FiatShamirTranscript::<SC>::observe(&mut replay, v);
        }
        let (gkr2, bcp2, r2) = prove_zerocheck_and_logup(&mut replay, &d_pk, &ctx).unwrap();
        assert_eq!(gkr2.q0_claim, inst.proof.gkr_proof.q0_claim);
        assert_eq!(&bcp2, bcp);
        assert_eq!(r2, walk3.r);
        let replay_log = replay.into_log();
        assert_eq!(
            &replay_log.values()[..walk3.stage3_end],
            &inst.log.values()[..walk3.stage3_end],
        );
        assert_eq!(
            &replay_log.samples()[..walk3.stage3_end],
            &inst.log.samples()[..walk3.stage3_end],
        );
    }

    // --- Dumps ---
    for (air_idx, spec) in inst.specs.iter().enumerate() {
        write_matrix(&inputs.join(format!("trace_{air_idx}.npy")), &spec.trace);
        let dag = constraints_dag_json(&inst.pk.per_air[air_idx].vk.symbolic_constraints);
        fs::write(
            inputs.join(format!("constraints_{air_idx}.json")),
            serde_json::to_string_pretty(&dag).unwrap(),
        )
        .unwrap();
    }
    write_transcript_log(&outputs, &inst.log);
    write_npy_u32(&outputs.join("alpha.npy"), &[4], &ef_limbs(inst.walk.alpha));
    write_npy_u32(&outputs.join("beta.npy"), &[4], &ef_limbs(inst.walk.beta));
    let xi_flat: Vec<u32> = inst.walk.xi.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("xi.npy"), &[inst.walk.xi.len(), 4], &xi_flat);
    write_npy_u32(&outputs.join("lambda.npy"), &[4], &ef_limbs(walk3.lambda));
    write_npy_u32(&outputs.join("mu.npy"), &[4], &ef_limbs(walk3.mu));
    let r_flat: Vec<u32> = walk3.r.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("r.npy"), &[walk3.r.len(), 4], &r_flat);
    let claims_flat: Vec<u32> = bcp
        .numerator_term_per_air
        .iter()
        .zip(&bcp.denominator_term_per_air)
        .flat_map(|(&p, &q)| ef_limbs(p).into_iter().chain(ef_limbs(q)))
        .collect();
    write_npy_u32(
        &outputs.join("sum_claims.npy"),
        &[num_traces, 2, 4],
        &claims_flat,
    );
    let s0_flat: Vec<u32> = bcp
        .univariate_round_coeffs
        .iter()
        .flat_map(|&c| ef_limbs(c))
        .collect();
    write_npy_u32(&outputs.join("s0_coeffs.npy"), &[s_0_deg + 1, 4], &s0_flat);
    let rounds_flat: Vec<u32> = bcp
        .sumcheck_round_polys
        .iter()
        .flat_map(|round_polys| round_polys.iter().flat_map(|&e| ef_limbs(e)))
        .collect();
    write_npy_u32(
        &outputs.join("round_polys.npy"),
        &[inst.n_max, s_deg, 4],
        &rounds_flat,
    );
    for (t, openings) in bcp.column_openings.iter().enumerate() {
        for (p, part) in openings.iter().enumerate() {
            let flat: Vec<u32> = part.iter().flat_map(|&e| ef_limbs(e)).collect();
            write_npy_u32(
                &outputs.join(format!("column_openings_t{t}_p{p}.npy")),
                &[part.len(), 4],
                &flat,
            );
        }
    }

    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
        "params": {
            "l_skip": params.l_skip,
            "n_stack": params.n_stack,
            "log_blowup": params.log_blowup,
            "k_whir": params.whir.k,
            "max_constraint_degree": inst.pk.max_constraint_degree,
            "logup_pow_bits": params.logup.pow_bits,
        },
        "airs": inst.specs.iter().enumerate().map(|(air_idx, spec)| serde_json::json!({
            "air_idx": air_idx,
            "height": spec.trace.height(),
            "width": spec.trace.width(),
            "is_required": inst.pk.per_air[air_idx].vk.is_required,
            "public_values": spec.public_values.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>(),
            "constraint_degree": inst.pk.per_air[air_idx].vk.max_constraint_degree,
            "needs_next": inst.pk.per_air[air_idx].vk.params.need_rot,
            "num_constraints": inst.pk.per_air[air_idx].vk.symbolic_constraints.constraints.constraint_idx.len(),
            "num_interactions": inst.pk.per_air[air_idx].vk.symbolic_constraints.interactions.len(),
        })).collect::<Vec<_>>(),
        "sorted_airs": inst.sorted_airs,
        "n_logup": inst.n_logup,
        "n_max": inst.n_max,
        "n_global": inst.n_global,
        "n_per_trace": n_per_trace,
        "total_interactions": inst.total_interactions,
        "s_deg": s_deg,
        "s_0_deg": s_0_deg,
        "interactions_layout": inst.interactions_layout.sorted_cols.iter().map(|(t, i, s)| serde_json::json!({
            "sorted_trace_idx": t,
            "interaction_idx": i,
            "row_idx": s.row_idx,
            "log_height": s.log_height(),
        })).collect::<Vec<_>>(),
        "prelude_len": inst.prelude_len,
        "stage2_end": inst.walk.stage2_end,
        "stage3_end": walk3.stage3_end,
        "transcript_len": inst.log.len(),
    });
    fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!("zerocheck fixtures written to {}", out.display());
}

struct StackingLogWalk {
    lambda: EF,
    u: Vec<EF>,
    stage4_end: usize,
}

fn walk_stacking_log(
    log: &TranscriptLog<F, [F; 16]>,
    start: usize,
    sp: &openvm_stark_backend::proof::StackingProof<SC>,
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

fn gen_stacking_fixture(out: &Path) {
    let inputs = out.join("inputs");
    let outputs = out.join("outputs");
    fs::create_dir_all(&inputs).unwrap();
    fs::create_dir_all(&outputs).unwrap();

    let inst = prove_instance();
    let params = &inst.params;
    let sp = &inst.proof.stacking_proof;
    let bcp = &inst.proof.batch_constraint_proof;

    let needs_next: Vec<bool> = inst
        .sorted_airs
        .iter()
        .map(|&i| inst.pk.per_air[i].vk.params.need_rot)
        .collect();
    let s_0_deg = 2 * ((1 << params.l_skip) - 1);
    assert_eq!(sp.univariate_round_coeffs.len(), s_0_deg + 1);
    assert_eq!(sp.sumcheck_round_polys.len(), params.n_stack);
    // The fixture has no preprocessed/cached commits: common main only.
    assert_eq!(sp.stacking_openings.len(), 1);

    let walk3 = walk_zerocheck_log(&inst.log, inst.walk.stage2_end, bcp, &needs_next);
    let walk4 = walk_stacking_log(&inst.log, walk3.stage3_end, sp);
    // Stage 5 begins with WHIR's μ-PoW grind: a witness observe, then the
    // grind's check sample (default_test_params_small has mu_pow_bits > 0).
    assert!(!inst.log.samples()[walk4.stage4_end]);
    assert_eq!(
        inst.log.values()[walk4.stage4_end],
        inst.proof.whir_proof.mu_pow_witness
    );
    assert!(inst.log.samples()[walk4.stage4_end + 1]);

    // Rebuild the common-main stacked PCS data exactly as the coordinator
    // does (device.commit on the sorted common main traces) and check it
    // reproduces the prelude's commitment.
    let engine = BabyBearPoseidon2RefEngine::<DuplexSpongeRecorder>::new(params.clone());
    let ctx = Stage2Fixture.generate_proving_ctx().into_sorted();
    let traces: Vec<&ColMajorMatrix<F>> =
        ctx.common_main_traces().map(|(_, trace)| trace).collect();
    let (root, data) = engine.device().commit(&traces).unwrap();
    // vk_pre_hash occupies the first 8 log slots, the common main commit the
    // next 8.
    assert_eq!(&root[..], &inst.log.values()[8..16]);

    // Self-validate: Stage 4 has no PoW grind, so the pub entry point replays
    // cleanly on a ReadOnlyTranscript pinned at stage3_end.
    {
        let mut ro = ReadOnlyTranscript::new(&inst.log, walk3.stage3_end);
        let (sp2, u2) = prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpu<SC>>(
            engine.device(),
            &mut ro,
            params.n_stack,
            vec![&data],
            vec![needs_next.clone()],
            &walk3.r,
        );
        assert_eq!(sp2.univariate_round_coeffs, sp.univariate_round_coeffs);
        assert_eq!(sp2.sumcheck_round_polys, sp.sumcheck_round_polys);
        assert_eq!(sp2.stacking_openings, sp.stacking_openings);
        assert_eq!(u2, walk4.u);
    }

    // --- Dumps ---
    for (air_idx, spec) in inst.specs.iter().enumerate() {
        write_matrix(&inputs.join(format!("trace_{air_idx}.npy")), &spec.trace);
    }
    let r_flat: Vec<u32> = walk3.r.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&inputs.join("r.npy"), &[walk3.r.len(), 4], &r_flat);
    write_transcript_log(&outputs, &inst.log);
    write_matrix(&outputs.join("stacked_matrix.npy"), &data.matrix);
    write_npy_u32(&outputs.join("lambda.npy"), &[4], &ef_limbs(walk4.lambda));
    let s0_flat: Vec<u32> = sp
        .univariate_round_coeffs
        .iter()
        .flat_map(|&c| ef_limbs(c))
        .collect();
    write_npy_u32(&outputs.join("s0_coeffs.npy"), &[s_0_deg + 1, 4], &s0_flat);
    let rounds_flat: Vec<u32> = sp
        .sumcheck_round_polys
        .iter()
        .flat_map(|round_polys| round_polys.iter().flat_map(|&e| ef_limbs(e)))
        .collect();
    write_npy_u32(
        &outputs.join("round_polys.npy"),
        &[params.n_stack, 2, 4],
        &rounds_flat,
    );
    let u_flat: Vec<u32> = walk4.u.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("u.npy"), &[walk4.u.len(), 4], &u_flat);
    for (c, claims_for_com) in sp.stacking_openings.iter().enumerate() {
        let flat: Vec<u32> = claims_for_com.iter().flat_map(|&e| ef_limbs(e)).collect();
        write_npy_u32(
            &outputs.join(format!("stacking_openings_c{c}.npy")),
            &[claims_for_com.len(), 4],
            &flat,
        );
    }

    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
        "params": {
            "l_skip": params.l_skip,
            "n_stack": params.n_stack,
            "log_blowup": params.log_blowup,
            "k_whir": params.whir.k,
        },
        "sorted_airs": inst.sorted_airs,
        "needs_next": needs_next,
        "s_0_deg": s_0_deg,
        "stacked_height": data.matrix.height(),
        "stacked_width": data.matrix.width(),
        "layout": data.layout.sorted_cols.iter().map(|(m, j, s)| serde_json::json!({
            "mat_idx": m,
            "col_in_mat": j,
            "col_idx": s.col_idx,
            "row_idx": s.row_idx,
            "log_height": s.log_height(),
        })).collect::<Vec<_>>(),
        "stage3_end": walk3.stage3_end,
        "stage4_end": walk4.stage4_end,
        "transcript_len": inst.log.len(),
    });
    fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!("stacking fixtures written to {}", out.display());
}

struct WhirLogWalk {
    mu: EF,
    /// Per WHIR round, the in-domain query indices (leaf indices of the
    /// query-strided tree, i.e. `sample_bits(log_rs_domain_size - k_whir)`).
    query_indices: Vec<Vec<usize>>,
    stage5_end: usize,
}

/// Walk the Stage-5 (WHIR opening) segment of the transcript log, asserting
/// every observe/sample against `prove_whir_opening`'s order:
/// μ-PoW grind → μ; then per WHIR round: k_whir × (2 sumcheck evals,
/// folding grind, α), then either (codeword commit, z₀, OOD value) or the
/// final-poly coefficients, then the query-phase grind, the query index
/// samples and γ. Opened rows and Merkle proofs are hints — never observed.
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
            read_grind(whir.folding_pow_bits, wp.folding_pow_witnesses[flat], &mut idx);
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

fn gen_whir_fixture(out: &Path) {
    let inputs = out.join("inputs");
    let outputs = out.join("outputs");
    fs::create_dir_all(&inputs).unwrap();
    fs::create_dir_all(&outputs).unwrap();

    let inst = prove_instance();
    let params = &inst.params;
    let whir = &params.whir;
    let wp = &inst.proof.whir_proof;
    let k_whir = whir.k;
    let num_rounds = whir.num_whir_rounds();
    let m = params.l_skip + params.n_stack;

    let needs_next: Vec<bool> = inst
        .sorted_airs
        .iter()
        .map(|&i| inst.pk.per_air[i].vk.params.need_rot)
        .collect();
    let walk3 = walk_zerocheck_log(&inst.log, inst.walk.stage2_end, &inst.proof.batch_constraint_proof, &needs_next);
    let walk4 = walk_stacking_log(&inst.log, walk3.stage3_end, &inst.proof.stacking_proof);

    // Stage-4 → Stage-5 input: u_cube = (u₀ squarings over the skip domain) ‖ u[1..].
    let (&u0, u_rest) = walk4.u.split_first().unwrap();
    let u_cube: Vec<EF> = u0
        .exp_powers_of_2()
        .take(params.l_skip)
        .chain(u_rest.iter().copied())
        .collect();
    assert_eq!(u_cube.len(), m);

    let walk5 = walk_whir_log(&inst.log, walk4.stage4_end, params, wp);
    // Stage 5 is the last stage: the walk must land exactly on the log's end.
    assert_eq!(walk5.stage5_end, inst.log.len());

    // Proof-shape pins (num_rounds = 3, num_queries = [10, 4, 2] for the
    // small params; asserted structurally rather than hardcoded).
    assert_eq!(wp.whir_sumcheck_polys.len(), whir.num_sumcheck_rounds());
    assert_eq!(wp.codeword_commits.len(), num_rounds - 1);
    assert_eq!(wp.ood_values.len(), num_rounds - 1);
    assert_eq!(wp.folding_pow_witnesses.len(), whir.num_sumcheck_rounds());
    assert_eq!(wp.query_phase_pow_witnesses.len(), num_rounds);
    assert_eq!(wp.final_poly.len(), 1 << (m - num_rounds * k_whir));
    assert_eq!(wp.initial_round_opened_rows.len(), 1); // common main only
    assert_eq!(wp.initial_round_merkle_proofs.len(), 1);
    assert_eq!(wp.codeword_opened_values.len(), num_rounds - 1);
    assert_eq!(wp.codeword_merkle_proofs.len(), num_rounds - 1);

    // Self-validate: rebuild a real recorder sponge at stage4_end by replaying
    // the log (observes fed back; samples squeezed and asserted — a
    // `ReadOnlyTranscript` cannot cross Stage 5's grinds), rebuild the
    // common-main `StackedPcsData`, and rerun `prove_whir`. The serial grind
    // (default-features off) re-finds the same witnesses, so the proof and
    // the full log must reproduce.
    {
        let engine = BabyBearPoseidon2RefEngine::<DuplexSpongeRecorder>::new(params.clone());
        let ctx = Stage2Fixture.generate_proving_ctx().into_sorted();
        let traces: Vec<&ColMajorMatrix<F>> =
            ctx.common_main_traces().map(|(_, trace)| trace).collect();
        let (root, data) = engine.device().commit(&traces).unwrap();
        assert_eq!(&root[..], &inst.log.values()[8..16]);

        let mut replay = default_duplex_sponge_recorder();
        for i in 0..walk4.stage4_end {
            if inst.log.samples()[i] {
                let got = FiatShamirTranscript::<SC>::sample(&mut replay);
                assert_eq!(got, inst.log.values()[i], "sample mismatch at {i}");
            } else {
                FiatShamirTranscript::<SC>::observe(&mut replay, inst.log.values()[i]);
            }
        }
        let wp2 = engine
            .device()
            .prove_whir(&mut replay, data, vec![], &u_cube)
            .unwrap();
        assert_eq!(&wp2, wp);
        let replay_log = replay.into_log();
        assert_eq!(replay_log.values(), inst.log.values());
        assert_eq!(replay_log.samples(), inst.log.samples());
    }

    // --- Dumps ---
    for (air_idx, spec) in inst.specs.iter().enumerate() {
        write_matrix(&inputs.join(format!("trace_{air_idx}.npy")), &spec.trace);
    }
    let u_cube_flat: Vec<u32> = u_cube.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&inputs.join("u_cube.npy"), &[u_cube.len(), 4], &u_cube_flat);

    write_transcript_log(&outputs, &inst.log);
    write_npy_u32(&outputs.join("mu.npy"), &[4], &ef_limbs(walk5.mu));
    write_npy_u32(
        &outputs.join("mu_pow_witness.npy"),
        &[1],
        &[wp.mu_pow_witness.as_canonical_u32()],
    );
    let sumcheck_flat: Vec<u32> = wp
        .whir_sumcheck_polys
        .iter()
        .flat_map(|evals| evals.iter().flat_map(|&e| ef_limbs(e)))
        .collect();
    write_npy_u32(
        &outputs.join("sumcheck_polys.npy"),
        &[wp.whir_sumcheck_polys.len(), 2, 4],
        &sumcheck_flat,
    );
    let commits_flat: Vec<u32> = wp
        .codeword_commits
        .iter()
        .flat_map(|d| d.iter().map(|x| x.as_canonical_u32()))
        .collect();
    write_npy_u32(
        &outputs.join("codeword_commits.npy"),
        &[wp.codeword_commits.len(), 8],
        &commits_flat,
    );
    let ood_flat: Vec<u32> = wp.ood_values.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(
        &outputs.join("ood_values.npy"),
        &[wp.ood_values.len(), 4],
        &ood_flat,
    );
    write_npy_u32(
        &outputs.join("folding_pow_witnesses.npy"),
        &[wp.folding_pow_witnesses.len()],
        &wp.folding_pow_witnesses
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>(),
    );
    write_npy_u32(
        &outputs.join("query_phase_pow_witnesses.npy"),
        &[wp.query_phase_pow_witnesses.len()],
        &wp.query_phase_pow_witnesses
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>(),
    );
    let final_flat: Vec<u32> = wp.final_poly.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(
        &outputs.join("final_poly.npy"),
        &[wp.final_poly.len(), 4],
        &final_flat,
    );

    // Initial-round openings (common main, base field): rows are
    // (num_queries, 2^k_whir, width); proofs (num_queries, depth, 8).
    let rows0 = &wp.initial_round_opened_rows[0];
    let width = rows0[0][0].len();
    let rows0_flat: Vec<u32> = rows0
        .iter()
        .flat_map(|q| {
            q.iter()
                .flat_map(|row| row.iter().map(|x| x.as_canonical_u32()))
        })
        .collect();
    write_npy_u32(
        &outputs.join("initial_opened_rows_c0.npy"),
        &[rows0.len(), 1 << k_whir, width],
        &rows0_flat,
    );
    let proofs0 = &wp.initial_round_merkle_proofs[0];
    let depth0 = proofs0[0].len();
    let proofs0_flat: Vec<u32> = proofs0
        .iter()
        .flat_map(|p| p.iter().flat_map(|d| d.iter().map(|x| x.as_canonical_u32())))
        .collect();
    write_npy_u32(
        &outputs.join("initial_merkle_proofs_c0.npy"),
        &[proofs0.len(), depth0, 8],
        &proofs0_flat,
    );

    // Per non-initial round: opened values (num_queries, 2^k_whir, 4) and
    // proofs (num_queries, depth, 8). Query counts differ per round.
    for (r, (vals, proofs)) in wp
        .codeword_opened_values
        .iter()
        .zip(&wp.codeword_merkle_proofs)
        .enumerate()
    {
        let vals_flat: Vec<u32> = vals
            .iter()
            .flat_map(|q| q.iter().flat_map(|&x| ef_limbs(x)))
            .collect();
        write_npy_u32(
            &outputs.join(format!("codeword_opened_values_r{}.npy", r + 1)),
            &[vals.len(), 1 << k_whir, 4],
            &vals_flat,
        );
        let depth = proofs[0].len();
        let proofs_flat: Vec<u32> = proofs
            .iter()
            .flat_map(|p| p.iter().flat_map(|d| d.iter().map(|x| x.as_canonical_u32())))
            .collect();
        write_npy_u32(
            &outputs.join(format!("codeword_merkle_proofs_r{}.npy", r + 1)),
            &[proofs.len(), depth, 8],
            &proofs_flat,
        );
    }

    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
        "params": {
            "l_skip": params.l_skip,
            "n_stack": params.n_stack,
            "log_blowup": params.log_blowup,
            "k_whir": k_whir,
            "mu_pow_bits": whir.mu_pow_bits,
            "folding_pow_bits": whir.folding_pow_bits,
            "query_phase_pow_bits": whir.query_phase_pow_bits,
        },
        "sorted_airs": inst.sorted_airs,
        "num_whir_rounds": num_rounds,
        "num_queries": whir.rounds.iter().map(|r| r.num_queries).collect::<Vec<_>>(),
        "query_indices": walk5.query_indices,
        "stacked_width": width,
        "stage4_end": walk4.stage4_end,
        "stage5_end": walk5.stage5_end,
        "transcript_len": inst.log.len(),
    });
    fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!("whir fixtures written to {}", out.display());
}

/// Self-contained end-to-end fixture under PRODUCTION-shaped params
/// (`l_skip=4`, `n_stack=8`, `k_whir=4` — distinct from the `2/8/3` test
/// params, and still 3 WHIR rounds). The per-stage fixtures pin the test
/// params in isolation; this exercises the same prover at a different
/// `(l_skip, k_whir)`, where every short trace (heights 2/4/8 < 2^4) takes
/// the lifting/striding path — the generality the test params never hit.
///
/// Everything the Python `prove()` consumes (traces, constraint DAGs,
/// interactions, vk pre-hash, params) plus every stage's end-of-chain
/// outputs go into ONE directory; `prove_test.py` drives `prove()` from the
/// inputs and byte-matches the outputs. The generation-time self-check is
/// the full Stage-2..5 log walk, which asserts the transcript observe/sample
/// sequence against the proof struct under the new params — a drift fails
/// here, not in the Python test.
fn gen_prove_fixture(out: &Path) {
    let inputs = out.join("inputs");
    let outputs = out.join("outputs");
    fs::create_dir_all(&inputs).unwrap();
    fs::create_dir_all(&outputs).unwrap();

    let inst = prove_instance_with(test_system_params_small(4, 8, 4));
    let params = &inst.params;
    let whir = &params.whir;
    let gkr = &inst.proof.gkr_proof;
    let bcp = &inst.proof.batch_constraint_proof;
    let sp = &inst.proof.stacking_proof;
    let wp = &inst.proof.whir_proof;

    let needs_next: Vec<bool> = inst
        .sorted_airs
        .iter()
        .map(|&i| inst.pk.per_air[i].vk.params.need_rot)
        .collect();

    // Walk the whole log to validate the transcript sequence under the new
    // params and to recover stage boundaries / the GKR challenge point.
    let walk3 = walk_zerocheck_log(&inst.log, inst.walk.stage2_end, bcp, &needs_next);
    let walk4 = walk_stacking_log(&inst.log, walk3.stage3_end, sp);
    let walk5 = walk_whir_log(&inst.log, walk4.stage4_end, params, wp);
    assert_eq!(walk5.stage5_end, inst.log.len());

    // --- Inputs the Python prove() needs (input/vk order, not sorted) ---
    for (air_idx, spec) in inst.specs.iter().enumerate() {
        write_matrix(&inputs.join(format!("trace_{air_idx}.npy")), &spec.trace);
        let dag = constraints_dag_json(&inst.pk.per_air[air_idx].vk.symbolic_constraints);
        fs::write(
            inputs.join(format!("constraints_{air_idx}.json")),
            serde_json::to_string_pretty(&dag).unwrap(),
        )
        .unwrap();
    }

    // --- End-of-chain outputs per stage (the Fiat-Shamir chain is
    // sequential, so matching each stage boundary covers the whole
    // transcript up to it) ---
    // Stage 1 + 2.
    write_npy_u32(
        &outputs.join("common_main_commit.npy"),
        &[8],
        &inst.log.values()[8..16]
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
    let xi_flat: Vec<u32> = inst.walk.xi.iter().flat_map(|&x| ef_limbs(x)).collect();
    write_npy_u32(&outputs.join("xi.npy"), &[inst.walk.xi.len(), 4], &xi_flat);
    // Stage 3.
    write_npy_u32(&outputs.join("zc_lambda.npy"), &[4], &ef_limbs(walk3.lambda));
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
    write_npy_u32(&outputs.join("st_lambda.npy"), &[4], &ef_limbs(walk4.lambda));
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

    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
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
        "vk_pre_hash": inst.pk.vk_pre_hash.map(|x| x.as_canonical_u32()),
        "sorted_airs": inst.sorted_airs,
        "airs": (0..inst.specs.len()).map(|air_idx| serde_json::json!({
            "air_idx": air_idx,
            "is_required": inst.pk.per_air[air_idx].vk.is_required,
            "needs_next": inst.pk.per_air[air_idx].vk.params.need_rot,
            "constraint_degree": inst.pk.per_air[air_idx].vk.max_constraint_degree,
            "public_values": inst.specs[air_idx].public_values.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>(),
            "interactions": inst.specs[air_idx].interactions.iter().map(|i| serde_json::json!({
                "bus": i.bus,
                "count_col": i.count_col,
                "count_neg": i.count_neg,
                "message_cols": i.msg_cols,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!("prove (production-params) fixture written to {}", out.display());
}

fn main() {
    let mut out_dir: Option<PathBuf> = None;
    let mut transcript_out: Option<PathBuf> = None;
    let mut gkr_out: Option<PathBuf> = None;
    let mut zerocheck_out: Option<PathBuf> = None;
    let mut stacking_out: Option<PathBuf> = None;
    let mut whir_out: Option<PathBuf> = None;
    let mut prove_out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = args.next().map(PathBuf::from),
            "--transcript-out" => transcript_out = args.next().map(PathBuf::from),
            "--gkr-out" => gkr_out = args.next().map(PathBuf::from),
            "--zerocheck-out" => zerocheck_out = args.next().map(PathBuf::from),
            "--stacking-out" => stacking_out = args.next().map(PathBuf::from),
            "--whir-out" => whir_out = args.next().map(PathBuf::from),
            "--prove-out" => prove_out = args.next().map(PathBuf::from),
            other => panic!(
                "unknown arg {other}; usage: [--out <dir>] [--transcript-out <dir>] [--gkr-out <dir>] [--zerocheck-out <dir>] [--stacking-out <dir>] [--whir-out <dir>] [--prove-out <dir>]"
            ),
        }
    }
    if out_dir.is_none()
        && transcript_out.is_none()
        && gkr_out.is_none()
        && zerocheck_out.is_none()
        && stacking_out.is_none()
        && whir_out.is_none()
        && prove_out.is_none()
    {
        panic!(
            "at least one of --out / --transcript-out / --gkr-out / --zerocheck-out / --stacking-out / --whir-out / --prove-out is required"
        );
    }
    if let Some(out) = out_dir {
        gen_stage1_fixture(&out);
    }
    if let Some(out) = transcript_out {
        gen_transcript_fixture(&out);
    }
    if let Some(out) = gkr_out {
        gen_gkr_fixture(&out);
    }
    if let Some(out) = zerocheck_out {
        gen_zerocheck_fixture(&out);
    }
    if let Some(out) = stacking_out {
        gen_stacking_fixture(&out);
    }
    if let Some(out) = whir_out {
        gen_whir_fixture(&out);
    }
    if let Some(out) = prove_out {
        gen_prove_fixture(&out);
    }
}
