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
    any_air_arc_vec, calculate_n_logup,
    hasher::{Hasher, MerkleHasher},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation},
    prover::{
        fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
        stacked_pcs::{stacked_commit, StackedLayout},
        AirProvingContext, ColMajorMatrix, MatrixDimensions, ProvingContext,
    },
    test_utils::{
        default_test_params_small,
        dummy_airs::{
            fib_air::{air::FibonacciAir, trace::generate_trace_rows},
            interaction::dummy_interaction_air::DummyInteractionAir,
        },
        TestFixture,
    },
    AirRef, FiatShamirTranscript, ReadOnlyTranscript, StarkEngine, TranscriptHistory,
    TranscriptLog,
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
    write_npy(path, shape, "<u4", &data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
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
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
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
    write_npy_u8(&dir.join("transcript_is_sample.npy"), &[flags.len()], &flags);
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
    FiatShamirTranscript::<SC>::observe_commit(&mut ts, core::array::from_fn(|i| F::from_u32(i as u32)));
    // PoW: sequential first-match witness so the fixture is deterministic.
    let pow_bits = 2usize;
    let witness = (0u32..)
        .map(F::from_u32)
        .find(|w| FiatShamirTranscript::<SC>::check_witness(&mut ts.clone(), pow_bits, *w))
        .unwrap();
    assert!(FiatShamirTranscript::<SC>::check_witness(&mut ts, pow_bits, witness));
    let _ = FiatShamirTranscript::<SC>::sample(&mut ts);

    let log = ts.into_log();
    write_transcript_log(out, &log);
    let meta = serde_json::json!({
        "reference": "openvm-stark-backend v2.0.0-beta.2 (f6a84921)",
        "pow_bits": pow_bits,
        "pow_witness": witness.as_canonical_u32(),
        "script": "observe 1..=5; sample; observe 100..117; sample x3; observe_ext [7,8,9,10]; sample_ext; observe_commit [0..8]; check_witness(pow_bits, witness); sample",
    });
    fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
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

    fn generate_proving_ctx(&self) -> ProvingContext<openvm_stark_backend::prover::CpuColMajorBackend<SC>> {
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

fn gen_gkr_fixture(out: &Path) {
    let inputs = out.join("inputs");
    let outputs = out.join("outputs");
    fs::create_dir_all(&inputs).unwrap();
    fs::create_dir_all(&outputs).unwrap();

    // l_skip=2, n_stack=8, k_whir=3, logup pow_bits=2 — the small test params
    // every backend test uses.
    let params = default_test_params_small();
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
        prelude_len += if pk_air.preprocessed_data.is_some() { 8 } else { 1 };
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
    fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    println!("gkr fixtures written to {}", out.display());
}

fn main() {
    let mut out_dir: Option<PathBuf> = None;
    let mut transcript_out: Option<PathBuf> = None;
    let mut gkr_out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = args.next().map(PathBuf::from),
            "--transcript-out" => transcript_out = args.next().map(PathBuf::from),
            "--gkr-out" => gkr_out = args.next().map(PathBuf::from),
            other => panic!(
                "unknown arg {other}; usage: [--out <dir>] [--transcript-out <dir>] [--gkr-out <dir>]"
            ),
        }
    }
    if out_dir.is_none() && transcript_out.is_none() && gkr_out.is_none() {
        panic!("at least one of --out / --transcript-out / --gkr-out is required");
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
}
