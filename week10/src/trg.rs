use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use thermo::{evidence, ising, linalg};

struct Split {
    rank: usize,
    left: Vec<f64>,
    right: Vec<f64>,
    singular: Vec<f64>,
}

fn split(a: &[f64], d: usize, order: [usize; 4], chi: usize) -> Result<Split> {
    let d2 = d * d;
    let mut permuted = Vec::with_capacity(a.len());
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let mut indices = [0; 4];
                    for (axis, index) in order.into_iter().zip([i, j, k, l]) {
                        indices[axis] = index;
                    }
                    let [r, u, l, down] = indices;
                    permuted.push(a[((r * d + u) * d + l) * d + down]);
                }
            }
        }
    }
    let linalg::Svd { u, s, vt } = linalg::svd(permuted, d2, d2)?;
    let rank = chi.min(d2);
    let roots: Vec<_> = s[..rank].iter().map(|value| value.sqrt()).collect();
    let left = u
        .chunks_exact(d2)
        .flat_map(|row| row.iter().zip(&roots).map(|(value, root)| value * root))
        .collect();
    let right = vt
        .chunks_exact(d2)
        .zip(&roots)
        .flat_map(|(row, root)| row.iter().map(move |value| value * root))
        .collect();
    Ok(Split {
        rank,
        left,
        right,
        singular: s,
    })
}

fn regroup(a: &Split, b: &Split, d: usize) -> Vec<f64> {
    let q = a.rank;
    let half = d * q;
    let d2 = d * d;
    let q2 = q * q;
    debug_assert_eq!(b.rank, q);

    // Sum over a in S1[w,a,r] S2[a,b,u], and over gamma in S3[l,b,gamma] S4[d,gamma,w].
    let mut s1 = vec![0.; half * d];
    let mut s4 = vec![0.; d * half];
    for w in 0..d {
        for inner in 0..d {
            for q_index in 0..q {
                s1[(w * q + q_index) * d + inner] = a.left[(w * d + inner) * q + q_index];
                s4[(inner * d + w) * q + q_index] = b.right[(q_index * d + inner) * d + w];
            }
        }
    }
    let upper = linalg::matmul(&s1, &b.left, half, d, half);
    let lower = linalg::matmul(&a.right, &s4, half, d, half);

    // Group the remaining internal pair (w,b) for the final matrix multiplication.
    let mut left = vec![0.; q2 * d2];
    let mut right = vec![0.; d2 * q2];
    for w in 0..d {
        for b in 0..d {
            let pair = w * d + b;
            for i in 0..q {
                for j in 0..q {
                    let external = i * q + j;
                    left[external * d2 + pair] = upper[(w * q + i) * half + b * q + j];
                    right[pair * q2 + external] = lower[(i * d + b) * half + w * q + j];
                }
            }
        }
    }
    linalg::matmul(&left, &right, q2, d2, q2)
}

fn save_split(out: &Path, step: usize, cut: &str, split: &Split, d: usize) -> Result<Value> {
    let left = format!("step-{step:02}-{cut}-left.bin");
    let right = format!("step-{step:02}-{cut}-right.bin");
    let spectrum = format!("step-{step:02}-{cut}-spectrum.bin");
    evidence::write_f64(&out.join(&left), &split.left)?;
    evidence::write_f64(&out.join(&right), &split.right)?;
    evidence::write_f64(&out.join(&spectrum), &split.singular)?;
    let tail: f64 = split.singular[split.rank..].iter().map(|s| s * s).sum();
    Ok(json!({
        "left": {"file": left, "shape": [d, d, split.rank]},
        "right": {"file": right, "shape": [split.rank, d, d]},
        "singular_values": {"file": spectrum, "shape": [d * d]},
        "discarded_squared_weight": tail,
    }))
}

pub fn solve(l: usize, k: f64, chi: usize, out: Option<&Path>) -> Result<(f64, Duration)> {
    ensure!(l.is_power_of_two(), "TRG requires a power-of-two L");
    let start = Instant::now();
    let mut evidence_time = Duration::ZERO;
    let mut sites = l * l;
    let mut log_scale = 0.;
    let mut steps = Vec::new();
    let mut a = ising::site(k).0.to_vec();
    let mut d = 2;
    for step in 1..=2 * l.ilog2() {
        let scale = a.iter().fold(0_f64, |max, &x| max.max(x.abs()));
        ensure!(
            scale.is_finite() && scale > 0.,
            "invalid scale at TRG step {step}"
        );
        for x in &mut a {
            *x /= scale;
        }
        log_scale += sites as f64 * scale.ln();

        let first = split(&a, d, [2, 1, 0, 3], chi)?;
        let second = split(&a, d, [3, 2, 1, 0], chi)?;
        a = regroup(&first, &second, d);
        if let Some(out) = out {
            let writing = Instant::now();
            steps.push(json!({
                "step": step, "sites_before_split": sites, "scale": scale,
                "lu_rd": save_split(out, step as usize, "lu-rd", &first, d)?,
                "dl_ur": save_split(out, step as usize, "dl-ur", &second, d)?,
            }));
            evidence_time += writing.elapsed();
        }
        d = first.rank;
        sites /= 2;
    }
    let mut closure = 0.;
    for i in 0..d {
        for j in 0..d {
            closure += a[((i * d + j) * d + i) * d + j];
        }
    }
    ensure!(closure.is_finite() && closure > 0., "invalid TRG closure");
    let seconds = start.elapsed() - evidence_time;
    if let Some(out) = out {
        let file = "final.bin";
        evidence::write_f64(&out.join(file), &a)?;
        evidence::write_json(
            &out.join("trg.json"),
            &json!({
                "axis_order": ["right", "up", "left", "down"],
                "closure": closure, "steps": steps, "final": {"file": file, "shape": [d, d, d, d]},
            }),
        )?;
    }
    Ok((log_scale + closure.ln(), seconds))
}
