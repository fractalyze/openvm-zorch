//! Golden-fixture generator for openvm-zorch's Stage 1 byte-match.
//!
//! Runs the reference `stacked_commit` (openvm-stark-backend v2.0.0-beta.2,
//! BabyBear + Poseidon2 width-16) on deterministic traces and dumps every
//! intermediate as canonical-u32 `.npy` plus a `meta.json`, so the JAX side
//! can compare each pipeline step independently. Also dumps Poseidon2
//! permutation / sponge / compress vectors to pin the hash parameterization
//! before any tree enters the picture.
//!
//! Usage: `cargo run --release -- --out <fixture dir>`

use std::{fs, io::Write as _, path::Path, path::PathBuf};

use openvm_stark_backend::{
    hasher::{Hasher, MerkleHasher},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation},
    prover::{stacked_pcs::stacked_commit, ColMajorMatrix, MatrixDimensions},
};
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};

type F = BabyBear;

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
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let shape_str = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let header = format!(
        "{{'descr': '<u4', 'fortran_order': False, 'shape': {shape_str}, }}"
    );
    // Header (incl. magic + 2-byte len) pads with spaces to a multiple of 64,
    // ending in \n.
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let mut out = Vec::with_capacity(unpadded + padding + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&((header.len() + padding + 1) as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat(b' ').take(padding));
    out.push(b'\n');
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
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

fn main() {
    let mut out_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = args.next().map(PathBuf::from),
            other => panic!("unknown arg {other}; usage: --out <dir>"),
        }
    }
    let out = out_dir.expect("--out <dir> is required");
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
    println!("fixtures written to {}", out.display());
}
