mod physics;
mod npy;
use clap::Parser;
use physics::{Experiment,Kernel};
use serde_json::{json,Value};
use std::{fs,path::PathBuf};
#[derive(Parser)]
struct Args {
 #[arg(long)] self_test:bool,
 #[arg(long)] experiment:Option<PathBuf>,
 #[arg(long,value_parser=["forward","born","adjoint"])] mode:Option<String>,
 #[arg(long)] out:Option<PathBuf>,
 #[arg(long)] every:Option<usize>,
 #[arg(long)] data:Option<PathBuf>,
 #[arg(long,default_value="full",value_parser=["full","treeverse"])] storage:String,
 #[arg(long)] checkpoints:Option<usize>,
}
fn norm(a:&[f64])->f64{a.iter().map(|x|x*x).sum::<f64>().sqrt()}
fn main(){
 let args=Args::parse();if args.self_test{physics::self_test();return;}
 let input=args.experiment.as_ref().expect("--experiment required");let mode=args.mode.as_deref().expect("--mode required");let out=args.out.as_ref().expect("--out required");
 let raw:Value=serde_json::from_slice(&fs::read(input).unwrap()).unwrap();let e:Experiment=serde_json::from_value(raw.clone()).unwrap();e.validate();
 if let Some(v)=args.every{assert!(v>0 && mode!="born");}
 assert_eq!(mode,"forward","derivative modes are implemented in Part 3");
 fs::create_dir_all(out).unwrap();let l=e.len();let c:Vec<_>=e.background.iter().flatten().copied().collect();let m:Vec<_>=e.perturbation.iter().flatten().copied().collect();let cp:Vec<_>=c.iter().zip(&m).map(|(c,m)|c+m).collect();
 let mut traces=Vec::new();let mut frames=Vec::new();let mut echo=Vec::new();let mut frame_steps=Vec::new();
 println!("shot\tmode\tdata L2 norm");
 for shot in 0..e.shots.len(){
  let start=traces.len();let mut k=Kernel::new(&e,shot);let mut s=vec![0.;2*l];let mut next=s.clone();let mut sp=s.clone();let mut nextp=s.clone();
  if shot==0 && args.every.is_some(){frames.extend_from_slice(&s[l..]);echo.extend_from_slice(&s[l..]);frame_steps.push(0);}
  for n in 0..e.steps {
   k.step(n,&s,&c,&mut next);std::mem::swap(&mut s,&mut next);
   for [x,z] in &e.receivers{traces.push(s[l+z*e.nx+x]);}
   if shot==0 && args.every.is_some(){k.step(n,&sp,&cp,&mut nextp);std::mem::swap(&mut sp,&mut nextp);
    if (n+1)%args.every.unwrap()==0{frames.extend_from_slice(&s[l..]);echo.extend(sp[l..].iter().zip(&s[l..]).map(|(p,b)|p-b));frame_steps.push(n+1);}
   }
  }println!("{shot}\t{mode}\t{:.12}",norm(&traces[start..]));
 }
 npy::write(&out.join("traces.npy"),&[e.shots.len(),e.steps,e.receivers.len()],&traces,false);
 if args.every.is_some(){for(name,data)in[("wavefield.npy",&frames),("echo.npy",&echo)]{npy::write(&out.join(name),&[frame_steps.len(),e.nz,e.nx],data,true);}}
 let mut meta=raw;meta.as_object_mut().unwrap().remove("background");meta.as_object_mut().unwrap().remove("perturbation");
 let mut run=json!({"experiment_file":input,"experiment":meta});
 if let Some(every)=args.every{run["recording"]=json!({"every":every,"steps":frame_steps,"times":frame_steps.iter().map(|n|*n as f64*e.dt).collect::<Vec<_>>()});}
 let result=json!({"mode":mode,"nx":e.nx,"nz":e.nz,"dx":e.dx,"dt":e.dt,"steps":e.steps,"shots":e.shots.len(),"receivers":e.receivers.len()});
 for(name,value)in[("run.json",run),("result.json",result)]{fs::write(out.join(name),serde_json::to_string_pretty(&value).unwrap()+"\n").unwrap();}
}
