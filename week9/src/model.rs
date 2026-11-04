use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Factor {
    pub scope: Vec<usize>,
    pub values: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct Model {
    pub dims: Vec<usize>,
    pub factors: Vec<Factor>,
}
pub type Evidence = BTreeMap<usize, usize>;
impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut it = text.split_whitespace();
        let kind = it.next().context("Missing UAI header")?;
        ensure!(
            kind == "MARKOV" || kind == "BAYES",
            "Unsupported UAI header"
        );
        let n: usize = it.next().context("Missing variable count")?.parse()?;
        let dims: Vec<usize> = (0..n)
            .map(|_| {
                it.next()
                    .context("Missing dimension")?
                    .parse()
                    .map_err(Into::into)
            })
            .collect::<Result<_>>()?;
        ensure!(dims.iter().all(|&d| d > 0), "Zero dimension");
        let nf: usize = it.next().context("Missing factor count")?.parse()?;
        let mut scopes = Vec::new();
        for _ in 0..nf {
            let k: usize = it.next().context("Missing scope size")?.parse()?;
            let scope: Vec<usize> = (0..k)
                .map(|_| {
                    it.next()
                        .context("Missing scope variable")?
                        .parse()
                        .map_err(Into::into)
                })
                .collect::<Result<_>>()?;
            ensure!(scope.iter().all(|&v| v < n), "Invalid variable ID");
            ensure!(
                scope
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == scope.len(),
                "Repeated scope variable"
            );
            scopes.push(scope);
        }
        let mut factors = Vec::new();
        for scope in scopes {
            let k: usize = it.next().context("Missing table count")?.parse()?;
            ensure!(
                k == scope.iter().map(|&v| dims[v]).product::<usize>(),
                "Table size mismatch"
            );
            let values: Vec<f64> = (0..k)
                .map(|_| {
                    it.next()
                        .context("Missing table value")?
                        .parse()
                        .map_err(Into::into)
                })
                .collect::<Result<_>>()?;
            ensure!(
                values.iter().all(|v| v.is_finite() && *v >= 0.0),
                "Invalid potential"
            );
            factors.push(Factor { scope, values });
        }
        ensure!(it.next().is_none(), "Unexpected trailing UAI data");
        Ok(Self { dims, factors })
    }
    pub fn validate_evidence(&self, evidence: &Evidence) -> Result<()> {
        ensure!(
            evidence
                .iter()
                .all(|(&v, &s)| v < self.dims.len() && s < self.dims[v]),
            "Invalid evidence"
        );
        Ok(())
    }
    pub fn conditioned(&self, evidence: &Evidence) -> Self {
        let factors = self
            .factors
            .iter()
            .map(|f| {
                let scope: Vec<_> = f
                    .scope
                    .iter()
                    .copied()
                    .filter(|v| !evidence.contains_key(v))
                    .collect();
                let mut values = Vec::new();
                for k in 0..f.values.len() {
                    let x = decode(k, &f.scope, &self.dims);
                    if f.scope
                        .iter()
                        .zip(&x)
                        .all(|(v, s)| evidence.get(v).is_none_or(|e| e == s))
                    {
                        values.push(f.values[k]);
                    }
                }
                Factor { scope, values }
            })
            .collect();
        Self {
            dims: self.dims.clone(),
            factors,
        }
    }
}
pub fn decode(mut k: usize, scope: &[usize], dims: &[usize]) -> Vec<usize> {
    let mut x = vec![0; scope.len()];
    for i in (0..scope.len()).rev() {
        x[i] = k % dims[scope[i]];
        k /= dims[scope[i]];
    }
    x
}
pub fn normalize(x: &mut [f64]) -> Result<f64> {
    let z: f64 = x.iter().sum();
    ensure!(z > 0.0 && z.is_finite(), "Zero or nonfinite normalization");
    for a in x {
        *a /= z;
    }
    Ok(z)
}
pub fn enumerate(model: &Model, evidence: &Evidence, query: &[usize]) -> Result<(f64, Vec<f64>)> {
    let total = model
        .dims
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .context("Enumeration too large")?;
    ensure!(total <= 1 << 24, "Enumeration limited to 2^24 states");
    let mut result = vec![0.; query.iter().map(|&v| model.dims[v]).product()];
    for k in 0..total {
        let x = decode(k, &(0..model.dims.len()).collect::<Vec<_>>(), &model.dims);
        if evidence.iter().any(|(&v, &s)| x[v] != s) {
            continue;
        }
        let weight: f64 = model
            .factors
            .iter()
            .map(|f| {
                let i = f.scope.iter().fold(0, |i, &v| i * model.dims[v] + x[v]);
                f.values[i]
            })
            .product();
        let i = query.iter().fold(0, |i, &v| i * model.dims[v] + x[v]);
        result[i] += weight;
    }
    let z = normalize(&mut result)?;
    Ok((z, result))
}
pub fn observations(label: &str) -> Result<Evidence> {
    Ok(match label {
        "prior" => Evidence::new(),
        "AD" => [(0, 0), (7, 0)].into(),
        "ADX" => [(0, 0), (7, 0), (6, 0)].into(),
        _ => {
            let txt = fs::read_to_string(label)?;
            let x = txt
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<Vec<usize>, _>>()?;
            ensure!(
                !x.is_empty() && x.len() == 1 + 2 * x[0],
                "Invalid evidence file"
            );
            let e: Evidence = x[1..].chunks_exact(2).map(|p| (p[0], p[1])).collect();
            ensure!(e.len() == x[0], "Duplicate evidence variable");
            e
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uai_axis_order_and_asia_masses() {
        let m = Model::load(Path::new("asia.uai")).unwrap();
        assert_eq!(m.factors[1].scope, vec![1, 0]);
        assert_eq!(m.factors[1].values, vec![0.05, 0.01, 0.95, 0.99]);
        assert!((enumerate(&m, &Evidence::new(), &[1]).unwrap().1[0] - 0.0104).abs() < 1e-14);
        for (label, z) in [("prior", 1.), ("AD", 0.004501375), ("ADX", 0.00098822675)] {
            assert!(
                (enumerate(&m, &observations(label).unwrap(), &[]).unwrap().0 / z - 1.).abs()
                    < 1e-12
            );
        }
    }
    #[test]
    fn conditioning_preserves_scalar_weights() {
        let m = Model {
            dims: vec![2, 3],
            factors: vec![
                Factor {
                    scope: vec![0, 1],
                    values: vec![1., 2., 3., 4., 5., 6.],
                },
                Factor {
                    scope: vec![0],
                    values: vec![2., 7.],
                },
            ],
        };
        let e = [(0, 1)].into();
        let c = m.conditioned(&e);
        assert_eq!(c.factors[0].values, vec![4., 5., 6.]);
        assert_eq!(c.factors[1].values, vec![7.]);
        assert_eq!(enumerate(&m, &e, &[1]).unwrap().0, 105.);
    }
}
