use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
use thermo::{evidence, ising, linalg};

struct Core {
    left: usize,
    right: usize,
    data: Vec<f64>,
}

impl Core {
    fn new(left: usize, right: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), left * 2 * right);
        Self { left, right, data }
    }
}

struct RowSite {
    left: usize,
    right: usize,
    data: Vec<f64>,
}

impl RowSite {
    fn new(a: &[f64; 16], b: &[f64; 2], left_edge: bool, right_edge: bool) -> Self {
        let left = if left_edge { 1 } else { 2 };
        let right = if right_edge { 1 } else { 2 };
        let mut data = vec![0.; left * 2 * 2 * right];
        for r in 0..2 {
            for u in 0..2 {
                for l in 0..2 {
                    for d in 0..2 {
                        let (li, lf) = if left_edge { (0, b[l]) } else { (l, 1.) };
                        let (ri, rf) = if right_edge { (0, b[r]) } else { (r, 1.) };
                        data[((li * 2 + u) * 2 + d) * right + ri] +=
                            lf * rf * a[((r * 2 + u) * 2 + l) * 2 + d];
                    }
                }
            }
        }
        Self { left, right, data }
    }

    fn absorb(&self, core: &Core) -> Core {
        let left = core.left * self.left;
        let right = core.right * self.right;
        let mut data = vec![0.; left * 2 * right];
        for alpha in 0..core.left {
            for l in 0..self.left {
                for d in 0..2 {
                    let dest = ((alpha * self.left + l) * 2 + d) * right;
                    for u in 0..2 {
                        let source = (alpha * 2 + u) * core.right;
                        for beta in 0..core.right {
                            let value = core.data[source + beta];
                            for r in 0..self.right {
                                let weight = self.data[((l * 2 + u) * 2 + d) * self.right + r];
                                data[dest + beta * self.right + r] += value * weight;
                            }
                        }
                    }
                }
            }
        }
        Core::new(left, right, data)
    }
}

fn compress(cores: &mut [Core], chi: usize, discarded: &mut Vec<f64>) -> Result<()> {
    for i in (1..cores.len()).rev() {
        let Core { left, right, data } = &mut cores[i];
        let rows = *left;
        let cols = 2 * *right;
        let linalg::Svd { mut u, s, vt } = linalg::svd(std::mem::take(data), rows, cols)?;
        let q = s.len();
        for row in u.chunks_exact_mut(q) {
            for (value, singular) in row.iter_mut().zip(&s) {
                *value *= singular;
            }
        }
        *data = vt;
        *left = q;

        let previous = &mut cores[i - 1];
        previous.data = linalg::matmul(&previous.data, &u, previous.left * 2, rows, q);
        previous.right = q;
    }

    for i in 0..cores.len() - 1 {
        let Core { left, right, data } = &mut cores[i];
        let rows = *left * 2;
        let cols = *right;
        let linalg::Svd { u, s, mut vt } = linalg::svd(std::mem::take(data), rows, cols)?;
        let q = s.len().min(chi);
        discarded.push(s[q..].iter().map(|s| s * s).sum());
        for (row, singular) in vt.chunks_exact_mut(cols).zip(&s).take(q) {
            for value in row {
                *value *= singular;
            }
        }
        *data = u
            .chunks_exact(s.len())
            .flat_map(|row| row[..q].iter().copied())
            .collect();
        *right = q;

        let next = &mut cores[i + 1];
        next.data = linalg::matmul(&vt[..q * cols], &next.data, q, cols, 2 * next.right);
        next.left = q;
    }
    Ok(())
}

pub fn solve(l: usize, k: f64, chi: usize, out: Option<&Path>) -> Result<(f64, Duration)> {
    let start = Instant::now();
    let (a, b) = ising::site(k);
    ensure!(
        a.iter().all(|value| value.is_finite()),
        "site tensor overflow"
    );
    let row: Vec<_> = (0..l)
        .map(|i| RowSite::new(&a, &b, i == 0, i == l - 1))
        .collect();
    let mut boundary: Vec<_> = (0..l).map(|_| Core::new(1, 1, b.to_vec())).collect();
    let mut scales = Vec::with_capacity(l);
    let mut discarded = Vec::with_capacity(l * (l - 1));
    let mut log_scale = 0.;

    for _ in 0..l {
        for (site, core) in row.iter().zip(&mut boundary) {
            *core = site.absorb(core);
        }
        compress(&mut boundary, chi, &mut discarded)?;
        let last = &mut boundary[l - 1].data;
        let scale = last.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(scale.is_finite() && scale > 0., "invalid boundary norm");
        for value in last {
            *value /= scale;
        }
        scales.push(scale);
        log_scale += scale.ln();
    }

    let mut vector = vec![1.];
    for core in &boundary {
        let mut next = vec![0.; core.right];
        for (left, &coefficient) in vector.iter().enumerate() {
            for (physical, &weight) in b.iter().enumerate() {
                let start = (left * 2 + physical) * core.right;
                for (value, &entry) in next.iter_mut().zip(&core.data[start..start + core.right]) {
                    *value += coefficient * weight * entry;
                }
            }
        }
        vector = next;
    }
    let closure = vector[0];
    ensure!(
        closure.is_finite() && closure > 0.,
        "invalid boundary closure"
    );
    let elapsed = start.elapsed();
    if let Some(out) = out {
        let mut shapes = Vec::with_capacity(l);
        for (i, core) in boundary.iter().enumerate() {
            let file = format!("core-{i:03}.bin");
            evidence::write_f64(&out.join(&file), &core.data)?;
            shapes.push(json!({"file": file, "shape": [core.left, 2, core.right]}));
        }
        evidence::write_json(
            &out.join("boundary.json"),
            &json!({
                "axis_order": ["left", "physical", "right"],
                "edge_vector": b,
                "row_scales": scales,
                "discarded_squared_singular_values_by_row": discarded.chunks_exact(l - 1).collect::<Vec<_>>(),
                "cores": shapes,
                "closure": closure,
            }),
        )?;
    }
    Ok((log_scale + closure.ln(), elapsed))
}
