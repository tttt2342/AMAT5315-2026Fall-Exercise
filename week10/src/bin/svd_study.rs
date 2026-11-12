use anyhow::Result;
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::{f64::consts::TAU, fs, path::PathBuf};
use thermo::{evidence, linalg};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let Args { out } = Args::parse();
    let n = 512;
    let ranks = [16, 64, 128, 256, 512];
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut matrix = Vec::with_capacity(n * n);
    for _ in 0..n * n / 2 {
        let radius = (-2. * (1. - rng.random::<f64>()).ln()).sqrt();
        let (sin, cos) = (TAU * rng.random::<f64>()).sin_cos();
        matrix.extend([radius * cos, radius * sin]);
    }
    fs::create_dir_all(&out)?;
    evidence::write_f64(&out.join("matrix.bin"), &matrix)?;
    let linalg::Svd { u, s, vt } = linalg::svd(matrix, n, n)?;
    evidence::write_f64(&out.join("singular-values.bin"), &s)?;
    for rank in ranks {
        let mut left = Vec::with_capacity(n * rank);
        for row in u.chunks_exact(n) {
            left.extend(row.iter().zip(&s).take(rank).map(|(u, s)| u * s));
        }
        let approximation = linalg::matmul(&left, &vt[..rank * n], n, rank, n);
        evidence::write_f64(&out.join(format!("rank-{rank}.bin")), &approximation)?;
    }
    evidence::write_json(
        &out.join("matrix.json"),
        &json!({
            "shape": [n, n], "seed": 2026, "generator": "ChaCha8Rng with Box-Muller",
            "ranks": ranks, "format": "row-major little-endian f64"
        }),
    )
}
