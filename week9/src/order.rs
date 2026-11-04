use anyhow::{Result,ensure,Context};
use omeco::{NestedEinsum,EinCode};
use rand::{Rng,SeedableRng,rngs::StdRng};
use serde::{Serialize,Deserialize};
use std::collections::BTreeSet;
use crate::model::Model;

#[derive(Serialize,Deserialize)]
pub struct Plan {
    pub method:String,pub seed:Option<u64>,pub input_scopes:Vec<Vec<usize>>,pub dimensions:Vec<usize>,pub observations:crate::model::Evidence,
    pub output:Vec<usize>,pub tree:NestedEinsum<usize>,pub planning_seconds:f64,pub settings:serde_json::Value,
}
#[derive(Serialize)]
pub struct Step {pub id:usize,pub children:Vec<usize>,pub inputs:Vec<Vec<usize>>,pub union:Vec<usize>,pub retained:Vec<usize>,pub work:u64,pub entries:u64}
#[derive(Serialize)]
pub struct Complexity {pub work:u64,pub largest_tensor:u64,pub largest_generated_tensor:u64,pub largest_step:Option<usize>,pub steps:Vec<Step>}
fn entries(labels:&[usize],dims:&[usize])->Result<u64>{labels.iter().try_fold(1u64,|p,&v|p.checked_mul(dims[v] as u64).context("Complexity overflow"))}
pub fn analyze(tree:&NestedEinsum<usize>,m:&Model,output:&[usize])->Result<Complexity>{
    fn visit(t:&NestedEinsum<usize>,m:&Model,output:&[usize],seen:&mut BTreeSet<usize>,steps:&mut Vec<Step>)->Result<(usize,Vec<usize>,BTreeSet<usize>)>{
        match t {
            NestedEinsum::Leaf{tensor_index:i}=>{ensure!(*i<m.factors.len()&&seen.insert(*i),"Invalid or repeated leaf");Ok((*i,m.factors[*i].scope.clone(),[*i].into()))},
            NestedEinsum::Node{args,eins}=>{
                ensure!(args.len()==2,"Expected binary plan");
                let a=visit(&args[0],m,output,seen,steps)?;let b=visit(&args[1],m,output,seen,steps)?;
                let leaves:BTreeSet<_>=a.2.union(&b.2).copied().collect();
                ensure!(eins.ixs==vec![a.1.clone(),b.1.clone()],"Plan input axes mismatch");
                let union:Vec<_>=a.1.iter().chain(&b.1).copied().collect::<BTreeSet<_>>().into_iter().collect();
                let outside:BTreeSet<_>=m.factors.iter().enumerate().filter(|(i,_)|!leaves.contains(i)).flat_map(|(_,f)|f.scope.iter()).chain(output.iter()).copied().collect();
                let expected:BTreeSet<_>=union.iter().filter(|v|outside.contains(v)).copied().collect();
                ensure!(expected.is_subset(&eins.iy.iter().copied().collect()) && eins.iy.iter().all(|v|union.contains(v)) && eins.iy.iter().collect::<BTreeSet<_>>().len()==eins.iy.len(),"Plan labels {:?}; required {:?}; union {:?}",eins.iy,expected,union);
                let id=m.factors.len()+steps.len();
                steps.push(Step{id,children:vec![a.0,b.0],inputs:vec![a.1,b.1],union:union.clone(),retained:eins.iy.clone(),work:entries(&union,&m.dims)?,entries:entries(&eins.iy,&m.dims)?});
                Ok((id,eins.iy.clone(),leaves))
            }
        }
    }
    let mut steps=Vec::new();let mut seen=BTreeSet::new();let (_,labels,_)=visit(tree,m,output,&mut seen,&mut steps)?;
    ensure!(seen.len()==m.factors.len()&&labels==output,"Incomplete plan or wrong output order");
    let largest_generated_tensor=steps.iter().map(|s|s.entries).max().unwrap_or(1);
    let largest_step=steps.iter().max_by_key(|s|s.entries).map(|s|s.id);
    let largest_input=m.factors.iter().map(|f|f.values.len() as u64).max().unwrap_or(1);
    Ok(Complexity{work:steps.iter().map(|s|s.work).sum(),largest_tensor:largest_input.max(largest_generated_tensor),largest_generated_tensor,largest_step,steps})
}
pub fn random(m:&Model,output:&[usize],seed:u64)->NestedEinsum<usize>{
    let mut rng=StdRng::seed_from_u64(seed);
    let mut active:Vec<_>=m.factors.iter().enumerate().map(|(i,f)|(NestedEinsum::leaf(i),f.scope.clone())).collect();
    while active.len()>1 {
        let i=rng.random_range(0..active.len());let (a,ia)=active.remove(i);
        let j=rng.random_range(0..active.len());let (b,ib)=active.remove(j);
        let outside:BTreeSet<_>=active.iter().flat_map(|(_,s)|s.iter()).chain(output.iter()).copied().collect();
        let union:BTreeSet<_>=ia.iter().chain(&ib).copied().collect();
        let retained=if active.is_empty(){output.to_vec()}else{union.intersection(&outside).copied().collect()};
        active.push((NestedEinsum::node(vec![a,b],EinCode::new(vec![ia,ib],retained.clone())),retained));
    }
    active.remove(0).0
}
#[cfg(test)] mod tests {
    use super::*;use crate::model::Factor;
    #[test]fn chain_costs_and_bad_plan(){
        let m=Model{dims:vec![10;4],factors:vec![Factor{scope:vec![0,1],values:vec![1.;100]},Factor{scope:vec![1,2],values:vec![1.;100]},Factor{scope:vec![2,3],values:vec![1.;100]}]};
        let ab=NestedEinsum::node(vec![NestedEinsum::leaf(0),NestedEinsum::leaf(1)],EinCode::new(vec![vec![0,1],vec![1,2]],vec![0,2]));
        let good=NestedEinsum::node(vec![ab,NestedEinsum::leaf(2)],EinCode::new(vec![vec![0,2],vec![2,3]],vec![0,3]));
        let c=analyze(&good,&m,&[0,3]).unwrap();assert_eq!((c.work,c.largest_tensor),(2000,100));
        let ac=NestedEinsum::node(vec![NestedEinsum::leaf(0),NestedEinsum::leaf(2)],EinCode::new(vec![vec![0,1],vec![2,3]],vec![0,1,2,3]));
        let bad=NestedEinsum::node(vec![ac,NestedEinsum::leaf(1)],EinCode::new(vec![vec![0,1,2,3],vec![1,2]],vec![0,3]));
        let c=analyze(&bad,&m,&[0,3]).unwrap();assert_eq!((c.work,c.largest_tensor),(20000,10000));
        assert!(analyze(&good,&m,&[0]).is_err());
    }
}
