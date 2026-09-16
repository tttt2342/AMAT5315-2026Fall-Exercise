use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use ising::{temperature_grid, Lattice, WolffWorkspace};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Update {
    Metropolis,
    Wolff,
}

impl Update {
    fn name(self) -> &'static str {
        match self {
            Self::Metropolis => "metropolis",
            Self::Wolff => "wolff",
        }
    }

    fn time_unit(self) -> &'static str {
        match self {
            Self::Metropolis => "sweep",
            Self::Wolff => "cluster_flip",
        }
    }

    fn metric_header(self) -> &'static str {
        match self {
            Self::Metropolis => "acceptance_rate",
            Self::Wolff => "mean_cluster_size",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "ising",
    about = "Sample the Ising model along a temperature ramp; no unstated defaults"
)]
struct Args {
    /// Update rule: metropolis sweeps or Wolff cluster flips
    #[arg(long, value_enum)]
    update: Update,

    /// Integer lattice side, at least 2
    #[arg(long)]
    l: usize,

    /// Lowest temperature; the ramp ascends from here
    #[arg(long)]
    t_from: f64,

    /// Temperature upper bound, included only if reached by the step
    #[arg(long)]
    t_to: f64,

    /// Positive temperature step
    #[arg(long)]
    t_step: f64,

    /// Equilibration steps discarded at each temperature
    #[arg(long)]
    discard: u64,

    /// Measured steps at each temperature
    #[arg(long)]
    measure: u64,

    /// Record a spin frame every N measured steps; zero records none
    #[arg(long, default_value_t = 0)]
    every: u64,

    /// Random seed for the single stream carried through the ramp
    #[arg(long)]
    seed: u64,

    /// Output folder
    #[arg(long)]
    out: PathBuf,
}

#[derive(Serialize)]
struct RunMetadata<'a> {
    #[serde(rename = "L")]
    side: usize,
    update: &'a str,
    t_grid: &'a [f64],
    discard: u64,
    measure: u64,
    seed: u64,
    sample_every: u64,
    time_unit: &'a str,
}

fn validate(args: &Args) -> Result<(), String> {
    if args.l < 2 {
        return Err("--l must be at least 2".into());
    }
    args.l
        .checked_mul(args.l)
        .ok_or_else(|| "--l is too large".to_string())?;
    if !args.t_from.is_finite() || args.t_from <= 0.0 {
        return Err("--t-from must be finite and positive".into());
    }
    if !args.t_to.is_finite() || args.t_to < args.t_from {
        return Err("--t-to must be finite and at least --t-from".into());
    }
    if !args.t_step.is_finite() || args.t_step <= 0.0 {
        return Err("--t-step must be finite and positive".into());
    }
    if args.measure == 0 {
        return Err("--measure must be at least 1 so the requested mean is defined".into());
    }
    Ok(())
}

fn update_step(
    update: Update,
    lattice: &mut Lattice,
    temperature: f64,
    rng: &mut ChaCha8Rng,
    wolff_workspace: &mut WolffWorkspace,
) -> u64 {
    match update {
        Update::Metropolis => lattice.metropolis_sweep(temperature, rng),
        Update::Wolff => lattice.wolff_cluster_flip(temperature, rng, wolff_workspace),
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    validate(&args)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;

    let temperatures = temperature_grid(args.t_from, args.t_to, args.t_step);
    fs::create_dir_all(&args.out)?;

    let metadata = RunMetadata {
        side: args.l,
        update: args.update.name(),
        t_grid: &temperatures,
        discard: args.discard,
        measure: args.measure,
        seed: args.seed,
        sample_every: 1,
        time_unit: args.update.time_unit(),
    };
    let run_file = File::create(args.out.join("run.json"))?;
    serde_json::to_writer_pretty(BufWriter::new(run_file), &metadata)?;

    let mut series = BufWriter::new(File::create(args.out.join("series.jsonl"))?);
    // Always create/truncate the contract file; it remains empty when --every=0.
    let mut frames = BufWriter::new(File::create(args.out.join("spins.jsonl"))?);
    let mut lattice = Lattice::all_up(args.l);
    let mut wolff_workspace = WolffWorkspace::new(args.l * args.l);
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let proposals_per_sweep = (args.l * args.l) as u64;
    let mut global_sweep = 0_u64;

    println!("T\tmean_abs_M\t{}", args.update.metric_header());
    for &temperature in &temperatures {
        let mut update_total = 0_u64;
        for _ in 0..args.discard {
            update_total += update_step(
                args.update,
                &mut lattice,
                temperature,
                &mut rng,
                &mut wolff_workspace,
            );
            global_sweep += 1;
        }

        let mut absolute_magnetization_sum = 0.0;
        for measured_sweep in 1..=args.measure {
            let step_value = update_step(
                args.update,
                &mut lattice,
                temperature,
                &mut rng,
                &mut wolff_workspace,
            );
            update_total += step_value;
            global_sweep += 1;

            let magnetization = lattice.mean_spin();
            let energy = lattice.energy_per_site();
            absolute_magnetization_sum += magnetization.abs();
            match args.update {
                Update::Metropolis => writeln!(
                    series,
                    "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"M\":{:.6},\"E\":{:.6}}}",
                    args.l, temperature, measured_sweep, magnetization, energy
                )?,
                Update::Wolff => writeln!(
                    series,
                    "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"M\":{:.6},\"E\":{:.6},\"cluster_size\":{}}}",
                    args.l, temperature, measured_sweep, magnetization, energy, step_value
                )?,
            }

            if args.every > 0 && measured_sweep % args.every == 0 {
                write!(
                    frames,
                    "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"m\":{:.6},\"spins\":[",
                    args.l, temperature, global_sweep, magnetization
                )?;
                for (index, spin) in lattice.spins().iter().enumerate() {
                    if index > 0 {
                        frames.write_all(b",")?;
                    }
                    write!(frames, "{}", spin)?;
                }
                frames.write_all(b"]}\n")?;
            }
        }

        let mean_absolute_magnetization = absolute_magnetization_sum / args.measure as f64;
        let steps = args.discard + args.measure;
        let update_metric = match args.update {
            Update::Metropolis => update_total as f64 / (steps * proposals_per_sweep) as f64,
            Update::Wolff => update_total as f64 / steps as f64,
        };
        println!("{temperature:.6}\t{mean_absolute_magnetization:.6}\t{update_metric:.6}");
    }

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(error) = run(args) {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
