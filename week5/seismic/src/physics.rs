use serde::{Deserialize,Serialize};
#[derive(Clone,Deserialize,Serialize)]
pub struct Experiment {
 pub nx:usize,pub nz:usize,pub dx:f64,pub dt:f64,pub steps:usize,
 pub source_frequency:f64,pub source_peak_time:f64,pub source_amplitude:f64,
 #[serde(deserialize_with="indices")] pub shots:Vec<[usize;2]>,
 #[serde(deserialize_with="indices")] pub receivers:Vec<[usize;2]>,
 pub sponge_width:f64,pub sponge_strength:f64,
 pub background:Vec<Vec<f64>>,pub perturbation:Vec<Vec<f64>>,
 pub length_unit_m:f64,pub time_unit_s:f64,
}
impl Experiment {
 pub fn len(&self)->usize{self.nx*self.nz}
 pub fn validate(&self){
  assert!(self.nx>2 && self.nz>2 && self.steps>0 && self.dx>0. && self.dt>0. && self.sponge_width>0.);
  for a in [&self.background,&self.perturbation]{assert_eq!(a.len(),self.nz);assert!(a.iter().all(|r|r.len()==self.nx && r.iter().all(|v|v.is_finite())));}
  assert!(self.background.iter().flatten().all(|v|*v>0.));
  for p in self.shots.iter().chain(&self.receivers){assert!(p[0]<self.nx && p[1]<self.nz);}
  let vmax=self.background.iter().flatten().zip(self.perturbation.iter().flatten()).map(|(c,m)|c.abs()+m.abs()).fold(0.0,f64::max);
  assert!(vmax*self.dt/self.dx<=1.0/2.0_f64.sqrt(),"unstable CFL");
 }
}
unsafe extern "C" {
 fn enzyme_cube(x:f64,o:*mut f64);
 fn wave_step(nx:usize,nz:usize,dt2:f64,idx2:f64,s:*const f64,c:*const f64,k:*const f64,o:*mut f64);
 fn wave_jvp(nx:usize,nz:usize,dt2:f64,idx2:f64,s:*const f64,ds:*const f64,c:*const f64,dc:*const f64,k:*const f64,o:*mut f64,do_:*mut f64);
 fn wave_vjp(nx:usize,nz:usize,dt2:f64,idx2:f64,s:*const f64,bs:*mut f64,c:*const f64,bc:*mut f64,k:*const f64,o:*mut f64,bo:*mut f64);
}
pub struct Kernel<'a>{pub e:&'a Experiment,k:Vec<f64>,foot:Vec<f64>}
impl<'a> Kernel<'a>{
 pub fn new(e:&'a Experiment,shot:usize)->Self{
  let l=e.len();let mut k=vec![0.;3*l];let mut foot=vec![0.;l];let [sx,sz]=e.shots[shot];
  for z in 0..e.nz {for x in 0..e.nx {
   let i=z*e.nx+x;let distance=x.min(z).min(e.nx-1-x).min(e.nz-1-z) as f64;
   let sigma=e.sponge_strength*(1.-distance/e.sponge_width).max(0.).powi(2);
   k[i]=1./(1.+sigma*e.dt);k[l+i]=1.-sigma*e.dt;
   foot[i]=(-0.5*((x as f64-sx as f64).powi(2)+(z as f64-sz as f64).powi(2))).exp();
  }} Self{e,k,foot}
 }
 fn prepare(&mut self,n:usize){let e=self.e;let theta=std::f64::consts::PI*e.source_frequency*(n as f64*e.dt-e.source_peak_time);let q=e.source_amplitude*(1.-2.*theta*theta)*(-theta*theta).exp();for i in 0..e.len(){self.k[2*e.len()+i]=q*self.foot[i];}}
 pub fn step(&mut self,n:usize,s:&[f64],c:&[f64],o:&mut[f64]){
  let e=self.e;assert_eq!(s.len(),2*e.len());assert_eq!(o.len(),s.len());assert_eq!(c.len(),e.len());self.prepare(n);
  unsafe{wave_step(e.nx,e.nz,e.dt*e.dt,1./(e.dx*e.dx),s.as_ptr(),c.as_ptr(),self.k.as_ptr(),o.as_mut_ptr())}
 }
 pub fn jvp(&mut self,n:usize,s:&[f64],ds:&[f64],c:&[f64],dc:&[f64],o:&mut[f64],do_:&mut[f64]){
  let e=self.e;for a in [s,ds,&*o,&*do_]{assert_eq!(a.len(),2*e.len());}for a in [c,dc]{assert_eq!(a.len(),e.len());}self.prepare(n);
  unsafe{wave_jvp(e.nx,e.nz,e.dt*e.dt,1./(e.dx*e.dx),s.as_ptr(),ds.as_ptr(),c.as_ptr(),dc.as_ptr(),self.k.as_ptr(),o.as_mut_ptr(),do_.as_mut_ptr())}
 }
 pub fn vjp(&mut self,n:usize,s:&[f64],c:&[f64],bs:&mut[f64],bc:&mut[f64],o:&mut[f64],bo:&mut[f64]){
  let e=self.e;for a in [s,&*bs,&*o,&*bo]{assert_eq!(a.len(),2*e.len());}for a in [c,&*bc]{assert_eq!(a.len(),e.len());}self.prepare(n);bs.fill(0.);o.fill(0.);
  unsafe{wave_vjp(e.nx,e.nz,e.dt*e.dt,1./(e.dx*e.dx),s.as_ptr(),bs.as_mut_ptr(),c.as_ptr(),bc.as_mut_ptr(),self.k.as_ptr(),o.as_mut_ptr(),bo.as_mut_ptr())}
 }
}
pub fn self_test(){
 let mut v=[0.;3];unsafe{enzyme_cube(2.,v.as_mut_ptr())};assert_eq!(v,[8.,12.,12.]);
 let e=Experiment{nx:7,nz:6,dx:1.,dt:0.1,steps:4,source_frequency:0.1,source_peak_time:1.,source_amplitude:1.,shots:vec![[3,2]],receivers:vec![[2,2]],sponge_width:2.,sponge_strength:0.4,background:vec![vec![1.8;7];6],perturbation:vec![vec![0.1;7];6],length_unit_m:100.,time_unit_s:0.1};
 let l=e.len();let s:Vec<_>=(0..2*l).map(|i|(i as f64).sin()).collect();let ds:Vec<_>=(0..2*l).map(|i|(i as f64*0.7).cos()).collect();let c=vec![1.8;l];let dc=vec![0.13;l];let w:Vec<_>=(0..2*l).map(|i|(i as f64*0.3).sin()).collect();
 let mut k=Kernel::new(&e,0);let mut o=vec![0.;2*l];let mut tangent=o.clone();k.jvp(2,&s,&ds,&c,&dc,&mut o,&mut tangent);
 let mut bs=vec![0.;2*l];let mut bc=vec![0.;l];let mut bo=w.clone();k.vjp(2,&s,&c,&mut bs,&mut bc,&mut o,&mut bo);
 let lhs:f64=tangent.iter().zip(&w).map(|(a,b)|a*b).sum();let rhs:f64=bs.iter().zip(&ds).map(|(a,b)|a*b).sum::<f64>()+bc.iter().zip(&dc).map(|(a,b)|a*b).sum::<f64>();assert!((lhs-rhs).abs()<1e-11);
 let h=1e-5;let mut plus=vec![0.;2*l];let mut minus=plus.clone();
 for (sign,out) in [(1.,&mut plus),(-1.,&mut minus)] {let sh:Vec<_>=s.iter().zip(&ds).map(|(a,b)|a+sign*h*b).collect();let ch:Vec<_>=c.iter().zip(&dc).map(|(a,b)|a+sign*h*b).collect();k.step(2,&sh,&ch,out);}
 let error=plus.iter().zip(&minus).zip(&tangent).map(|((p,m),t)|((p-m)/(2.*h)-t).abs()).fold(0.,f64::max);assert!(error<1e-9);
 // Independent interior stencil check, and the fixed outer boundary.
 let x=3;let z=3;let i=z*e.nx+x;let theta=std::f64::consts::PI*0.1*(0.2-1.);let q=(1.-2.*theta*theta)*(-theta*theta).exp()*(-0.5_f64).exp();
 k.step(2,&s,&c,&mut o);let lap=s[l+i-1]+s[l+i+1]+s[l+i-e.nx]+s[l+i+e.nx]-4.*s[l+i];assert!((o[l+i]-(2.*s[l+i]-s[i]+0.01*(3.24*lap+q))).abs()<1e-12);
 assert_eq!(o[l],0.);println!("Enzyme cube and timestep PASS; transpose error={}, FD error={error}",(lhs-rhs).abs());
}

fn indices<'de,D:serde::Deserializer<'de>>(d:D)->Result<Vec<[usize;2]>,D::Error>{
 let v=Vec::<[f64;2]>::deserialize(d)?;v.into_iter().map(|p|{if p.iter().all(|x|x.is_finite() && *x>=0. && x.fract()==0.){Ok([p[0] as usize,p[1] as usize])}else{Err(serde::de::Error::custom("non-integer grid index"))}}).collect()
}
