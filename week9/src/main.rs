mod bp;
mod exact;
mod model;
mod order;
use anyhow::{Result, ensure};
use clap::Parser;
use model::*;
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};
use tenferro_cpu::CpuBackend;

#[derive(Parser, Debug)]
#[command(about = "CPU tensor-network inference for UAI models")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    evidence: String,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, conflicts_with_all=["plan","analyze"])]
    order: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, conflicts_with = "analyze")]
    plan: Option<PathBuf>,
    #[arg(long)]
    analyze: Option<PathBuf>,
    #[arg(long, default_value = "exact")]
    method: String,
    #[arg(long)]
    damping: Option<f64>,
    #[arg(long)]
    tolerance: Option<f64>,
    #[arg(long)]
    max_sweeps: Option<usize>,
    #[arg(long, value_delimiter = ',')]
    queries: Vec<usize>,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    fs::create_dir_all(&args.out)?;
    let model = Model::load(&args.model)?;
    let obs = observations(&args.evidence)?;
    model.validate_evidence(&obs)?;
    let conditioned = model.conditioned(&obs);
    ensure!(
        args.method == "exact" || args.method == "bp",
        "Unknown inference method"
    );
    if args.method == "bp" {
        ensure!(
            args.order.is_none()
                && args.seed.is_none()
                && args.plan.is_none()
                && args.analyze.is_none(),
            "Ordering options only apply to exact inference"
        );
        let result = bp::run(
            &conditioned,
            &obs,
            args.damping
                .ok_or_else(|| anyhow::anyhow!("BP requires damping"))?,
            args.tolerance
                .ok_or_else(|| anyhow::anyhow!("BP requires tolerance"))?,
            args.max_sweeps
                .ok_or_else(|| anyhow::anyhow!("BP requires max-sweeps"))?,
        )?;
        fs::write(
            args.out.join("run.json"),
            serde_json::to_string_pretty(&result)?,
        )?;
        return Ok(());
    }
    ensure!(
        args.damping.is_none() && args.tolerance.is_none() && args.max_sweeps.is_none(),
        "BP settings only apply to BP"
    );
    ensure!(
        model.dims.len() == 8 || !args.queries.is_empty(),
        "Specify --queries for larger models"
    );
    ensure!(
        args.queries.iter().all(|&q| q < model.dims.len()),
        "Invalid query ID"
    );
    let mut cpu = CpuBackend::new();
    let planning = Instant::now();
    let saved = if let Some(path) = args.plan.as_ref().or(args.analyze.as_ref()) {
        ensure!(args.seed.is_none(), "Seed only applies to random ordering");
        let p: order::Plan = serde_json::from_str(&fs::read_to_string(path)?)?;
        ensure!(
            p.input_scopes
                == conditioned
                    .factors
                    .iter()
                    .map(|f| f.scope.clone())
                    .collect::<Vec<_>>()
                && p.dimensions == model.dims
                && p.observations == obs
                && p.output.is_empty(),
            "Plan/model/evidence mismatch"
        );
        p
    } else {
        let method = args.order.as_deref().unwrap_or("greedy");
        ensure!(
            method == "random" || method == "greedy" || method == "treesa",
            "Unknown order"
        );
        ensure!(
            if method == "random" {
                args.seed.is_some()
            } else if method == "treesa" {
                args.seed == Some(42)
            } else {
                args.seed.is_none()
            },
            "Random requires a seed; TreeSA requires seed 42 (library trial seeds 42 and 43); greedy forbids seed"
        );
        let tree = if method == "random" {
            order::random(&conditioned, &[], args.seed.unwrap())
        } else if method == "treesa" {
            order::treesa(&conditioned, &[])?
        } else {
            exact::greedy(&conditioned, &[])?
        };
        order::Plan {
            method: method.into(),
            seed: args.seed,
            input_scopes: conditioned
                .factors
                .iter()
                .map(|f| f.scope.clone())
                .collect(),
            dimensions: model.dims.clone(),
            observations: obs.clone(),
            output: vec![],
            tree,
            planning_seconds: planning.elapsed().as_secs_f64(),
            settings: if method == "treesa" {
                json!({"trial_seeds":[42,43],"rng":"SmallRng (omeco 0.2.6)","ntrials":2,"niters":20,"betas":(1..=100).map(|i|0.01+0.15*i as f64).collect::<Vec<_>>(),"initializer":"Greedy","tc_weight":1,"sc_weight":0,"rw_weight":0})
            } else {
                json!({})
            },
        }
    };
    let costs = order::analyze(&saved.tree, &conditioned, &[])?;
    fs::write(
        args.out.join("complexity.json"),
        serde_json::to_string_pretty(&costs)?,
    )?;
    fs::write(
        args.out.join("order.json"),
        serde_json::to_string_pretty(&saved)?,
    )?;
    println!(
        "Order {}: work={}, largest tensor={}",
        saved.method, costs.work, costs.largest_tensor
    );
    if args.analyze.is_some() {
        return Ok(());
    }
    let tree = saved.tree;
    let checks = exact::checks(&mut cpu)?;
    let contraction_start = Instant::now();
    let (logz, _) = exact::contract(&conditioned, &tree, &mut cpu)?;
    let scalar_seconds = contraction_start.elapsed().as_secs_f64();
    if model.dims.len() != 8 {
        let mut selected = std::collections::BTreeMap::new();
        let mut query_costs = std::collections::BTreeMap::new();
        for &v in &args.queries {
            let query_start = Instant::now();
            if let Some(&s) = obs.get(&v) {
                let mut p = vec![0.; model.dims[v]];
                p[s] = 1.;
                selected.insert(v, p);
                continue;
            }
            let qtree = order::with_output(&tree, &conditioned, &[v]);
            let qcost = order::analyze(&qtree, &conditioned, &[v])?;
            let (qlog, p) = exact::contract(&conditioned, &qtree, &mut cpu)?;
            ensure!(
                (qlog - logz).abs() < 1e-10,
                "Query partition scale mismatch"
            );
            fs::write(
                args.out.join(format!("query-{v}-order.json")),
                serde_json::to_string_pretty(&order::Plan {
                    method: "saved scalar topology with query axes".into(),
                    seed: args.seed,
                    input_scopes: conditioned
                        .factors
                        .iter()
                        .map(|f| f.scope.clone())
                        .collect(),
                    dimensions: model.dims.clone(),
                    observations: obs.clone(),
                    output: vec![v],
                    tree: qtree,
                    planning_seconds: 0.,
                    settings: json!({"parent":"order.json"}),
                })?,
            )?;
            fs::write(
                args.out.join(format!("query-{v}-complexity.json")),
                serde_json::to_string_pretty(&qcost)?,
            )?;
            query_costs.insert(v,json!({"work":qcost.work,"largest_tensor":qcost.largest_tensor,"runtime_seconds":query_start.elapsed().as_secs_f64()}));
            println!("Query {v}: {p:?}");
            selected.insert(v, p);
        }
        let result = json!({"observations":obs,"log_partition":logz,"log10_partition":logz/std::f64::consts::LN_10,"selected_marginals":selected,"query_costs":query_costs,"scalar_contraction_seconds":scalar_seconds,"planning_seconds":saved.planning_seconds,"runtime_seconds":start.elapsed().as_secs_f64(),"checks":checks});
        fs::write(
            args.out.join("run.json"),
            serde_json::to_string_pretty(&result)?,
        )?;
        println!(
            "log10 evidence partition = {:.15}",
            logz / std::f64::consts::LN_10
        );
        return Ok(());
    }
    let (enumz, _) = enumerate(&model, &obs, &[])?;
    let mut nodes = Vec::new();
    let mut enums = Vec::new();
    for v in 0..model.dims.len() {
        let p = if let Some(&s) = obs.get(&v) {
            let mut p = vec![0.; model.dims[v]];
            p[s] = 1.;
            p
        } else {
            exact::contract(&conditioned, &exact::greedy(&conditioned, &[v])?, &mut cpu)?.1
        };
        nodes.push(p);
        enums.push(enumerate(&model, &obs, &[v])?.1);
    }
    let joint = exact::contract(
        &conditioned,
        &exact::greedy(&conditioned, &[1, 3])?,
        &mut cpu,
    )?
    .1;
    let enumjoint = enumerate(&model, &obs, &[1, 3])?.1;
    let error = nodes
        .iter()
        .flatten()
        .zip(enums.iter().flatten())
        .chain(joint.iter().zip(enumjoint.iter()))
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let mass_error = (logz.exp() / enumz - 1.).abs();
    ensure!(
        error < 1e-10 && mass_error < 1e-10,
        "Exact verification failed"
    );
    let result = json!({"observations":obs,"evidence_mass":logz.exp(),"enumeration_mass":enumz,"node_marginals":nodes,"enumeration_marginals":enums,"joint_TL":joint,"enumeration_joint_TL":enumjoint,"max_absolute_marginal_error":error,"relative_mass_error":mass_error,"checks":checks,"runtime_seconds":start.elapsed().as_secs_f64()});
    fs::write(
        args.out.join("run.json"),
        serde_json::to_string_pretty(&result)?,
    )?;
    println!(
        "{}: mass={:.16}, max marginal error={error:.3e}, relative mass error={mass_error:.3e}",
        args.evidence,
        logz.exp()
    );
    println!(
        "Disease probabilities (T,L,B): {:?}",
        [nodes[1][0], nodes[3][0], nodes[4][0]]
    );
    Ok(())
}
