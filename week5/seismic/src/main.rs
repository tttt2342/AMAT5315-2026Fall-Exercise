mod npy;
mod physics;
mod reverse;
use clap::Parser;
use physics::{Experiment, Kernel};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    self_test: bool,
    #[arg(long)]
    experiment: Option<PathBuf>,
    #[arg(long,value_parser=["forward","born","adjoint"])]
    mode: Option<String>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    every: Option<usize>,
    #[arg(long)]
    data: Option<PathBuf>,
    #[arg(long,default_value="full",value_parser=["full","treeverse"])]
    storage: String,
    #[arg(long)]
    checkpoints: Option<usize>,
}
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn main() {
    let args = Args::parse();
    if args.self_test {
        physics::self_test();
        return;
    }
    let input = args.experiment.as_ref().expect("--experiment required");
    let mode = args.mode.as_deref().expect("--mode required");
    let out = args.out.as_ref().expect("--out required");
    let raw: Value = serde_json::from_slice(&fs::read(input).unwrap()).unwrap();
    let e: Experiment = serde_json::from_value(raw.clone()).unwrap();
    e.validate();
    if let Some(v) = args.every {
        assert!(v > 0 && mode != "born");
    }
    if mode == "adjoint" && args.storage == "full" {
        assert!(
            e.len() * (e.steps + 1) * 16 < 1_000_000_000,
            "full history over 1 GB forbidden; use treeverse"
        );
    }
    fs::create_dir_all(out).unwrap();
    let l = e.len();
    let c: Vec<_> = e.background.iter().flatten().copied().collect();
    let m: Vec<_> = e.perturbation.iter().flatten().copied().collect();
    let cp: Vec<_> = c.iter().zip(&m).map(|(c, m)| c + m).collect();
    let mut traces = Vec::new();
    let mut frames = Vec::new();
    let mut echo = Vec::new();
    let mut frame_steps = Vec::new();
    let mut image = vec![0.; l];
    let mut statistics = Vec::new();
    let weights = if mode == "adjoint" {
        npy::read(
            args.data.as_ref().expect("--data required"),
            &[e.shots.len(), e.steps, e.receivers.len()],
        )
    } else {
        vec![]
    };
    println!("shot\tmode\tdata L2 norm");
    for shot in 0..e.shots.len() {
        if mode == "adjoint" {
            let count = e.steps * e.receivers.len();
            let w = &weights[shot * count..(shot + 1) * count];
            let mut r = reverse::Reverse::new(
                &e,
                shot,
                &c,
                w,
                &mut image,
                if shot == 0 { args.every } else { None },
            );
            if args.storage == "full" {
                r.full();
            } else {
                r.treeverse(args.checkpoints.expect("--checkpoints required"));
            }
            let mut st = json!({"reverse_calls":r.grads,"scheduler_forward_calls":r.calls,"peak_saved_states":r.peak});
            if args.storage == "treeverse" {
                let file = format!("actions-{shot}.json");
                let f = fs::File::create(out.join(&file)).unwrap();
                serde_json::to_writer(std::io::BufWriter::new(f), &r.actions).unwrap();
                st["actions_file"] = json!(file);
            }
            frames.extend_from_slice(&r.frames);
            frame_steps.extend_from_slice(&r.frame_steps);
            statistics.push(st);
            println!("{shot}\t{mode}\t{:.12}", norm(w));
            continue;
        }
        let start = traces.len();
        let mut k = Kernel::new(&e, shot);
        let mut s = vec![0.; 2 * l];
        let mut next = s.clone();
        let mut sp = s.clone();
        let mut nextp = s.clone();
        let mut ds = vec![0.; 2 * l];
        let mut dnext = ds.clone();
        if shot == 0 && args.every.is_some() {
            frames.extend_from_slice(&s[l..]);
            echo.extend_from_slice(&s[l..]);
            frame_steps.push(0);
        }
        for n in 0..e.steps {
            if mode == "born" {
                k.jvp(n, &s, &ds, &c, &m, &mut next, &mut dnext);
                std::mem::swap(&mut ds, &mut dnext);
            } else {
                k.step(n, &s, &c, &mut next);
            }
            std::mem::swap(&mut s, &mut next);
            for [x, z] in &e.receivers {
                traces.push(if mode == "born" {
                    ds[l + z * e.nx + x]
                } else {
                    s[l + z * e.nx + x]
                });
            }
            if shot == 0 && args.every.is_some() {
                k.step(n, &sp, &cp, &mut nextp);
                std::mem::swap(&mut sp, &mut nextp);
                if (n + 1) % args.every.unwrap() == 0 {
                    frames.extend_from_slice(&s[l..]);
                    echo.extend(sp[l..].iter().zip(&s[l..]).map(|(p, b)| p - b));
                    frame_steps.push(n + 1);
                }
            }
        }
        println!("{shot}\t{mode}\t{:.12}", norm(&traces[start..]));
    }
    if mode == "adjoint" {
        npy::write(&out.join("image.npy"), &[e.nz, e.nx], &image, false);
    } else {
        npy::write(
            &out.join(if mode == "born" {
                "born_data.npy"
            } else {
                "traces.npy"
            }),
            &[e.shots.len(), e.steps, e.receivers.len()],
            &traces,
            false,
        );
    }
    if args.every.is_some() {
        npy::write(
            &out.join("wavefield.npy"),
            &[frame_steps.len(), e.nz, e.nx],
            &frames,
            true,
        );
        if mode == "forward" {
            npy::write(
                &out.join("echo.npy"),
                &[frame_steps.len(), e.nz, e.nx],
                &echo,
                true,
            );
        }
    }
    let mut meta = raw;
    meta.as_object_mut().unwrap().remove("background");
    meta.as_object_mut().unwrap().remove("perturbation");
    let mut run = json!({"experiment_file":input,"experiment":meta});
    if let Some(every) = args.every {
        run["recording"] = json!({"every":every,"steps":frame_steps,"times":frame_steps.iter().map(|n|*n as f64*e.dt).collect::<Vec<_>>()});
    }
    let mut result = json!({"mode":mode,"nx":e.nx,"nz":e.nz,"dx":e.dx,"dt":e.dt,"steps":e.steps,"shots":e.shots.len(),"receivers":e.receivers.len()});
    if mode == "adjoint" {
        let peak = statistics
            .iter()
            .map(|s| s["peak_saved_states"].as_u64().unwrap())
            .max()
            .unwrap();
        result["statistics"] = json!({"storage":args.storage,"checkpoints":if args.storage=="full"{None}else{args.checkpoints},"reverse_calls":statistics.iter().map(|s|s["reverse_calls"].as_u64().unwrap()).sum::<u64>(),"scheduler_forward_calls":statistics.iter().map(|s|s["scheduler_forward_calls"].as_u64().unwrap()).sum::<u64>(),"peak_saved_states":peak,"peak_saved_bytes":peak*l as u64*16,"per_shot":statistics});
    }
    for (name, value) in [("run.json", run), ("result.json", result)] {
        fs::write(
            out.join(name),
            serde_json::to_string_pretty(&value).unwrap() + "\n",
        )
        .unwrap();
    }
}
