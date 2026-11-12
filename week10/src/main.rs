mod boundary;

use anyhow::{Result, ensure};
use clap::{Parser, ValueEnum};
use serde_json::json;
use std::{fs, path::PathBuf};
use thermo::evidence;

#[derive(Clone, Copy, ValueEnum)]
enum Method {
    Boundary,
}

#[derive(Parser)]
#[command(about = "Approximate the zero-field square Ising partition function")]
struct Args {
    #[arg(long, value_enum)]
    method: Method,
    #[arg(long)]
    l: usize,
    #[arg(long)]
    k: f64,
    #[arg(long)]
    chi: usize,
    #[arg(long)]
    out: PathBuf,
    /// Keep only the run summary. Full tensor evidence is written by default.
    #[arg(long)]
    summary_only: bool,
}

fn main() -> Result<()> {
    let Args {
        method,
        l,
        k,
        chi,
        out,
        summary_only,
    } = Args::parse();
    ensure!(l >= 2, "L must be at least 2");
    ensure!(k.is_finite() && k >= 0., "K must be finite and nonnegative");
    ensure!(chi >= 1, "chi must be at least 1");
    let n = l
        .checked_mul(l)
        .ok_or_else(|| anyhow::anyhow!("L squared overflows"))?;
    fs::create_dir_all(&out)?;
    let tensors = (!summary_only).then_some(out.as_path());
    let ((log_z, elapsed), method, geometry) = match method {
        Method::Boundary => (boundary::solve(l, k, chi, tensors)?, "boundary", "open"),
    };
    let elapsed = elapsed.as_secs_f64();
    let result = json!({
        "model": "zero-field square Ising, H=-sum_edges(s_i*s_j), J=k_B=1",
        "geometry": geometry, "method": method, "l": l, "sites": n, "k": k, "chi": chi,
        "log_z": log_z, "g": log_z / n as f64, "seconds": elapsed,
        "tensor_format": "row-major little-endian f64", "full_evidence": !summary_only,
    });
    evidence::write_json(&out.join("run.json"), &result)?;
    println!("{result}");
    Ok(())
}
