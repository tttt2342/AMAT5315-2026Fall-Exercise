mod model;
mod exact;
mod order;
use anyhow::{Result,ensure};
use clap::Parser;
use serde_json::json;
use std::{path::PathBuf,fs,time::Instant};
use tenferro_cpu::CpuBackend;
use model::*;

#[derive(Parser,Debug)]
#[command(about="CPU tensor-network inference for UAI models")]
struct Args {
    #[arg(long)] model:PathBuf,
    #[arg(long)] evidence:String,
    #[arg(long)] out:PathBuf,
    #[arg(long, conflicts_with_all=["plan","analyze"])] order:Option<String>,
    #[arg(long)] seed:Option<u64>,
    #[arg(long, conflicts_with="analyze")] plan:Option<PathBuf>,
    #[arg(long)] analyze:Option<PathBuf>,
}
fn main()->Result<()> {
    let args=Args::parse();let start=Instant::now();
    fs::create_dir_all(&args.out)?;
    let model=Model::load(&args.model)?;let obs=observations(&args.evidence)?;model.validate_evidence(&obs)?;
    ensure!(model.dims.len()==8,"This build supports ASIA");
    let conditioned=model.conditioned(&obs);let mut cpu=CpuBackend::new();
    let planning=Instant::now();
    let saved=if let Some(path)=args.plan.as_ref().or(args.analyze.as_ref()) {
        ensure!(args.seed.is_none(),"Seed only applies to random ordering");
        let p:order::Plan=serde_json::from_str(&fs::read_to_string(path)?)?;
        ensure!(p.input_scopes==conditioned.factors.iter().map(|f|f.scope.clone()).collect::<Vec<_>>()&&p.dimensions==model.dims&&p.observations==obs&&p.output.is_empty(),"Plan/model/evidence mismatch");p
    } else {
        let method=args.order.as_deref().unwrap_or("greedy");
        ensure!(method=="random"||method=="greedy","Unknown order");
        ensure!((method=="random")==args.seed.is_some(),"Random requires seed; greedy forbids it");
        let tree=if method=="random" {order::random(&conditioned,&[],args.seed.unwrap())}else{exact::greedy(&conditioned,&[])?};
        order::Plan{method:method.into(),seed:args.seed,input_scopes:conditioned.factors.iter().map(|f|f.scope.clone()).collect(),dimensions:model.dims.clone(),observations:obs.clone(),output:vec![],tree,planning_seconds:planning.elapsed().as_secs_f64(),settings:json!({})}
    };
    let costs=order::analyze(&saved.tree,&conditioned,&[])?;
    fs::write(args.out.join("complexity.json"),serde_json::to_string_pretty(&costs)?)?;
    fs::write(args.out.join("order.json"),serde_json::to_string_pretty(&saved)?)?;
    println!("Order {}: work={}, largest tensor={}",saved.method,costs.work,costs.largest_tensor);
    if args.analyze.is_some(){return Ok(());}
    let tree=saved.tree;
    let checks=exact::checks(&mut cpu)?;
    let (logz,_)=exact::contract(&conditioned,&tree,&mut cpu)?;
    let (enumz,_)=enumerate(&model,&obs,&[])?;
    let mut nodes=Vec::new();let mut enums=Vec::new();
    for v in 0..model.dims.len() {
        let p=if let Some(&s)=obs.get(&v){let mut p=vec![0.;model.dims[v]];p[s]=1.;p}else{
            exact::contract(&conditioned,&exact::greedy(&conditioned,&[v])?,&mut cpu)?.1
        };
        nodes.push(p);enums.push(enumerate(&model,&obs,&[v])?.1);
    }
    let joint=exact::contract(&conditioned,&exact::greedy(&conditioned,&[1,3])?,&mut cpu)?.1;
    let enumjoint=enumerate(&model,&obs,&[1,3])?.1;
    let error=nodes.iter().flatten().zip(enums.iter().flatten()).chain(joint.iter().zip(enumjoint.iter())).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
    let mass_error=(logz.exp()/enumz-1.).abs();ensure!(error<1e-10&&mass_error<1e-10,"Exact verification failed");
    let result=json!({"observations":obs,"evidence_mass":logz.exp(),"enumeration_mass":enumz,"node_marginals":nodes,"enumeration_marginals":enums,"joint_TL":joint,"enumeration_joint_TL":enumjoint,"max_absolute_marginal_error":error,"relative_mass_error":mass_error,"checks":checks,"runtime_seconds":start.elapsed().as_secs_f64()});
    fs::write(args.out.join("run.json"),serde_json::to_string_pretty(&result)?)?;
    println!("{}: mass={:.16}, max marginal error={error:.3e}, relative mass error={mass_error:.3e}",args.evidence,logz.exp());
    println!("Disease probabilities (T,L,B): {:?}",[nodes[1][0],nodes[3][0],nodes[4][0]]);
    Ok(())
}
