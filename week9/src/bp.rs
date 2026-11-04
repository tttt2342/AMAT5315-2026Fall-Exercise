use crate::model::{Evidence, Model, decode, normalize};
use anyhow::{Result, ensure};
use serde::Serialize;
#[derive(Clone)]
struct Messages {
    vf: Vec<Vec<Vec<f64>>>,
    fv: Vec<Vec<Vec<f64>>>,
}
#[derive(Serialize)]
pub struct Record {
    pub sweep: usize,
    pub residual: f64,
    pub cumulative_update_work: u64,
    pub node_beliefs: Vec<Vec<f64>>,
}
#[derive(Serialize)]
pub struct Run {
    pub observations: Evidence,
    pub damping: f64,
    pub tolerance: f64,
    pub max_sweeps: usize,
    pub converged: bool,
    pub sweeps: usize,
    pub residual: f64,
    pub node_marginals: Vec<Vec<f64>>,
    pub cumulative_update_work: u64,
    pub history: Vec<Record>,
    pub runtime_seconds: f64,
    pub work_definition: &'static str,
}
type Adj = Vec<Vec<(usize, usize)>>;
fn propose(
    m: &Model,
    msg: &Messages,
    adj: &Adj,
    states: &[Vec<Vec<usize>>],
) -> Result<(Messages, u64)> {
    let mut new = msg.clone();
    let mut work = 0;
    for (f, factor) in m.factors.iter().enumerate() {
        for (axis, &v) in factor.scope.iter().enumerate() {
            let mut vf = vec![1.; m.dims[v]];
            for &(g, b) in &adj[v] {
                if g != f {
                    for (s, x) in vf.iter_mut().enumerate() {
                        *x *= msg.fv[g][b][s];
                        work += 1;
                    }
                }
            }
            normalize(&mut vf)?;
            new.vf[f][axis] = vf;
            let mut fv = vec![0.; m.dims[v]];
            for (k, x) in states[f].iter().enumerate() {
                let mut weight = factor.values[k];
                for (b, &s) in x.iter().enumerate() {
                    if b != axis {
                        weight *= msg.vf[f][b][s];
                        work += 1;
                    }
                }
                fv[x[axis]] += weight;
                work += 1;
            }
            normalize(&mut fv)?;
            new.fv[f][axis] = fv;
        }
    }
    Ok((new, work))
}
fn beliefs(m: &Model, msg: &Messages, adj: &Adj, evidence: &Evidence) -> Result<Vec<Vec<f64>>> {
    (0..m.dims.len())
        .map(|v| {
            let mut b = vec![1.; m.dims[v]];
            if let Some(&s) = evidence.get(&v) {
                b.fill(0.);
                b[s] = 1.;
                return Ok(b);
            }
            for &(f, a) in &adj[v] {
                for (s, x) in b.iter_mut().enumerate() {
                    *x *= msg.fv[f][a][s];
                }
            }
            normalize(&mut b)?;
            Ok(b)
        })
        .collect()
}
fn flattened(m: &Messages) -> impl Iterator<Item = &f64> {
    m.vf.iter().chain(&m.fv).flatten().flatten()
}
pub fn run(
    m: &Model,
    evidence: &Evidence,
    damping: f64,
    tolerance: f64,
    max_sweeps: usize,
) -> Result<Run> {
    ensure!(
        damping > 0. && damping <= 1. && tolerance > 0. && tolerance.is_finite() && max_sweeps > 0,
        "Invalid BP settings"
    );
    ensure!(
        m.factors.iter().all(|f| !f.values.iter().all(|v| *v == 0.)),
        "Impossible evidence: zero factor"
    );
    let start = std::time::Instant::now();
    let mut adj = vec![vec![]; m.dims.len()];
    for (f, factor) in m.factors.iter().enumerate() {
        for (a, &v) in factor.scope.iter().enumerate() {
            adj[v].push((f, a));
        }
    }
    let uniform: Vec<Vec<Vec<f64>>> = m
        .factors
        .iter()
        .map(|f| {
            f.scope
                .iter()
                .map(|&v| vec![1. / m.dims[v] as f64; m.dims[v]])
                .collect()
        })
        .collect();
    let mut msg = Messages {
        vf: uniform.clone(),
        fv: uniform,
    };
    let states: Vec<Vec<Vec<usize>>> = m
        .factors
        .iter()
        .map(|f| {
            (0..f.values.len())
                .map(|k| decode(k, &f.scope, &m.dims))
                .collect()
        })
        .collect();
    let mut history = Vec::new();
    let mut cumulative = 0;
    for sweep in 0..=max_sweeps {
        // F is evaluated at the recorded state, not at the preceding state.
        let (proposal, work) = propose(m, &msg, &adj, &states)?;
        cumulative += work;
        let residual = flattened(&proposal)
            .zip(flattened(&msg))
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let node_beliefs = beliefs(m, &msg, &adj, evidence)?;
        history.push(Record {
            sweep,
            residual,
            cumulative_update_work: cumulative,
            node_beliefs: node_beliefs.clone(),
        });
        if sweep % 100 == 0 || residual < tolerance || sweep == max_sweeps {
            println!(
                "BP sweep={sweep}, undamped residual={residual:.6e}, cumulative update work={cumulative}"
            );
        }
        if residual < tolerance || sweep == max_sweeps {
            return Ok(Run {
                observations: evidence.clone(),
                damping,
                tolerance,
                max_sweeps,
                converged: residual < tolerance,
                sweeps: sweep,
                residual,
                node_marginals: node_beliefs,
                cumulative_update_work: cumulative,
                history,
                runtime_seconds: start.elapsed().as_secs_f64(),
                work_definition: "Scalar multiplies and additions in message proposals; includes initial/final residual evaluations; excludes normalization, beliefs and damping",
            });
        }
        for (old, new) in msg
            .vf
            .iter_mut()
            .chain(&mut msg.fv)
            .flatten()
            .flatten()
            .zip(flattened(&proposal))
        {
            *old = (1. - damping) * *old + damping * new;
        }
    }
    unreachable!()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Factor, enumerate};
    #[test]
    fn nonuniform_tree_matches_enumeration() {
        let m = Model {
            dims: vec![2, 3, 2],
            factors: vec![
                Factor {
                    scope: vec![0],
                    values: vec![0.2, 0.8],
                },
                Factor {
                    scope: vec![0, 1],
                    values: vec![1., 2., 3., 4., 1., 2.],
                },
                Factor {
                    scope: vec![1, 2],
                    values: vec![3., 1., 1., 4., 2., 5.],
                },
            ],
        };
        let obs = Evidence::new();
        let r = run(&m, &obs, 0.5, 1e-12, 1000).unwrap();
        assert!(r.converged);
        for v in 0..3 {
            let exact = enumerate(&m, &obs, &[v]).unwrap().1;
            assert!(
                exact
                    .iter()
                    .zip(&r.node_marginals[v])
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            );
        }
    }
    #[test]
    fn small_damping_does_not_fake_convergence() {
        let m = Model {
            dims: vec![2],
            factors: vec![Factor {
                scope: vec![0],
                values: vec![0.9, 0.1],
            }],
        };
        let r = run(&m, &Evidence::new(), 1e-9, 1e-8, 1).unwrap();
        assert!(!r.converged);
        assert!(r.residual > 0.39);
        assert!((r.node_marginals[0][0] - 0.5).abs() < 1e-8);
    }
    #[test]
    fn zero_message_stops() {
        let m = Model {
            dims: vec![2],
            factors: vec![Factor {
                scope: vec![0],
                values: vec![0., 0.],
            }],
        };
        assert!(run(&m, &Evidence::new(), 0.5, 1e-10, 100).is_err());
    }
}
