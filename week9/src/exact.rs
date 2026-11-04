use crate::model::{Factor, Model, normalize};
use anyhow::{Context, Result, ensure};
use omeco::{EinCode, GreedyMethod, NestedEinsum, optimize_code};
use serde_json::{Value, json};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

pub fn greedy(model: &Model, output: &[usize]) -> Result<NestedEinsum<usize>> {
    let code = EinCode::new(
        model.factors.iter().map(|f| f.scope.clone()).collect(),
        output.to_vec(),
    );
    let sizes = model.dims.iter().copied().enumerate().collect();
    optimize_code(&code, &sizes, &GreedyMethod::default()).context("Ordering failed")
}
// Reverse both shape and labels: UAI's last-fastest buffer is then column-major.
fn tensor(f: &Factor, dims: &[usize]) -> Result<Tensor> {
    Ok(Tensor::from_vec_col_major(
        f.scope.iter().rev().map(|&v| dims[v]).collect::<Vec<_>>(),
        f.values.clone(),
    )?)
}
fn eval(
    tree: &NestedEinsum<usize>,
    model: &Model,
    cpu: &mut CpuBackend,
) -> Result<(Tensor, Vec<usize>, f64)> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let f = model.factors.get(*tensor_index).context("Bad plan leaf")?;
            Ok((
                tensor(f, &model.dims)?,
                f.scope.iter().rev().copied().collect(),
                0.,
            ))
        }
        NestedEinsum::Node { args, eins } => {
            let children = args
                .iter()
                .map(|a| eval(a, model, cpu))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<_> = children.iter().map(|c| &c.0).collect();
            let output: Vec<usize> = eins.iy.iter().rev().copied().collect();
            let sub = EinsumSubscripts {
                inputs: children
                    .iter()
                    .map(|c| c.1.iter().map(|&v| v as u32).collect())
                    .collect(),
                output: output.iter().map(|&v| v as u32).collect(),
            };
            let t = cpu
                .with_backend_session(|session| refs.as_slice().einsum_subscripts(&sub, session))?;
            let mut vals = t.as_slice::<f64>()?.to_vec();
            let scale = vals.iter().copied().fold(0., f64::max);
            ensure!(
                scale > 0. && scale.is_finite(),
                "Zero or nonfinite contraction"
            );
            for x in &mut vals {
                *x /= scale;
            }
            Ok((
                Tensor::from_vec_col_major(t.shape().to_vec(), vals)?,
                output,
                children.iter().map(|c| c.2).sum::<f64>() + scale.ln(),
            ))
        }
    }
}
pub fn contract(
    model: &Model,
    tree: &NestedEinsum<usize>,
    cpu: &mut CpuBackend,
) -> Result<(f64, Vec<f64>)> {
    let (t, _, scale) = eval(tree, model, cpu)?;
    let mut vals = t.as_slice::<f64>()?.to_vec();
    let z = normalize(&mut vals)?;
    Ok((z.ln() + scale, vals))
}
pub fn checks(cpu: &mut CpuBackend) -> Result<Value> {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1., 3., 2., 4.])?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![2., 1., 0., 2.])?;
    let ab = cpu.with_backend_session(|s| [&a, &b].einsum("ij,jk->ik", s))?;
    let tr = cpu.with_backend_session(|s| [&a, &b].einsum("ij,ji->", s))?;
    let k = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![0.5f64.exp(), (-0.5f64).exp(), (-0.5f64).exp(), 0.5f64.exp()],
    )?;
    let ring = cpu
        .with_backend_session(|s| [&k, &k, &k].einsum("ij,jk,ki->", s))?
        .as_slice::<f64>()?[0];
    let mut enumeration = 0.;
    for a in [-1., 1.] {
        for b in [-1., 1.] {
            for c in [-1., 1.] {
                enumeration += (0.5f64 * (a * b + b * c + c * a)).exp();
            }
        }
    }
    let trace = tr.as_slice::<f64>()?[0];
    ensure!((ring - enumeration).abs() < 1e-12, "Ising check failed");
    ensure!((trace - 12.).abs() < 1e-12, "Trace check failed");
    let ab = ab.as_slice::<f64>()?;
    ensure!(
        ab.iter()
            .zip([4., 10., 4., 8.])
            .all(|(a, b)| (a - b).abs() < 1e-12),
        "Matrix check failed"
    );
    let result = json!({"ab_row_major":[ab[0],ab[2],ab[1],ab[3]],"trace":trace,"ising_contraction":ring,"ising_enumeration":enumeration,"ising_absolute_error":(ring-enumeration).abs()});
    println!("Matrix/Ising checks: {result}");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evidence, enumerate};
    #[test]
    fn heterogeneous_hyperedge_and_absolute_scale() {
        let model = Model {
            dims: vec![2, 3, 2],
            factors: vec![
                Factor {
                    scope: vec![0, 1],
                    values: vec![1., 2., 3., 4., 5., 6.],
                },
                Factor {
                    scope: vec![1, 2],
                    values: vec![2., 1., 3., 7., 4., 8.],
                },
                Factor {
                    scope: vec![1],
                    values: vec![0.1, 0.7, 0.2],
                },
                Factor {
                    scope: vec![],
                    values: vec![0.003],
                },
            ],
        };
        let mut cpu = CpuBackend::new();
        let tree = greedy(&model, &[]).unwrap();
        let (z, _) = contract(&model, &tree, &mut cpu).unwrap();
        let e = enumerate(&model, &Evidence::new(), &[]).unwrap().0;
        assert!((z.exp() / e - 1.).abs() < 1e-12);
        for query in [vec![0], vec![1], vec![0, 2], vec![2, 0]] {
            let qt = crate::order::with_output(&tree, &model, &query);
            crate::order::analyze(&qt, &model, &query).unwrap();
            let (qz, p) = contract(&model, &qt, &mut cpu).unwrap();
            let expected = enumerate(&model, &Evidence::new(), &query).unwrap().1;
            assert!((qz - z).abs() < 1e-12);
            assert!(p.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-12));
        }
        let mut scaled = model.clone();
        scaled.factors[3].values[0] *= 10.;
        let (sz, _) = contract(&scaled, &tree, &mut cpu).unwrap();
        assert!((sz - z - 10f64.ln()).abs() < 1e-12);
        let qt = crate::order::with_output(&tree, &model, &[1]);
        let p = contract(&model, &qt, &mut cpu).unwrap().1;
        let sp = contract(&scaled, &qt, &mut cpu).unwrap().1;
        assert!(p.iter().zip(sp).all(|(a, b)| (a - b).abs() < 1e-12));
    }
}
