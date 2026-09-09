use clap::{Args, CommandFactory, Parser, Subcommand};
use md::{FluidResult, ForceMode, IntegratorChoice, RunConfig, check_saved_run};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "md", about = "Lennard-Jones molecular-dynamics tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run equilibration and production, writing run.json and traj.jsonl.
    Run(RunArgs),
    /// Recompute physics from a saved trajectory.
    Check {
        /// Directory containing run.json and traj.jsonl.
        artifacts: PathBuf,
    },
    /// Render the saved trajectory and its radial distribution function.
    Video {
        /// Directory containing run.json and traj.jsonl.
        artifacts: PathBuf,
        /// Output MP4 path.
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Args, Debug)]
struct RunArgs {
    #[arg(long, default_value_t = 100)]
    n: usize,
    #[arg(long, default_value_t = 0.8)]
    rho: f64,
    #[arg(long, default_value_t = 0.5)]
    temperature: f64,
    #[arg(long, default_value_t = 0.01)]
    dt: f64,
    #[arg(long, default_value_t = 2000)]
    eq_steps: usize,
    #[arg(long, default_value_t = 10000)]
    steps: usize,
    #[arg(long, default_value_t = 50)]
    sample_every: usize,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[arg(long, value_enum, default_value = "cells")]
    force: ForceMode,
    #[arg(long)]
    ramp_to: Option<f64>,
    #[arg(long, default_value = "artifacts")]
    out: PathBuf,
}

fn main() -> FluidResult<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Run(args)) => {
            let config = RunConfig::with_parameters_and_options(
                args.n,
                args.rho,
                args.temperature,
                args.dt,
                args.eq_steps,
                args.steps,
                args.sample_every,
                args.seed,
                args.force,
                args.ramp_to,
            )?;
            md::run_simulation(&config, &args.out)?;
            println!(
                "saved {} and {} using {}",
                args.out.join("run.json").display(),
                args.out.join("traj.jsonl").display(),
                IntegratorChoice::VelocityVerlet.as_str()
            );
        }
        Some(Command::Check { artifacts }) => {
            let report = check_saved_run(&artifacts)?;
            println!("frames: {}", report.frame_count);
            println!(
                "energy consistency: {}",
                if report.energy_consistent {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            println!(
                "secular drift: {:.6e} (limit < 2.000e-3)",
                report.energy_drift
            );
            println!(
                "speed temperature: {:.6} (target 0.5 ± 0.05)",
                report.speed_temperature
            );
            println!(
                "speed shape chi2/dof: {:.6} (limit < 2.0)",
                report.speed_shape_chi_squared
            );
            println!("{}", if report.passed { "PASS" } else { "FAIL" });
            if !report.passed {
                return Err("saved trajectory failed one or more physics checks".into());
            }
        }
        Some(Command::Video { artifacts, out }) => {
            md::render_video(&artifacts, &out)?;
            println!("saved {} (under 2 MiB)", out.display());
        }
        None => {
            let mut command = Cli::command();
            command.print_help()?;
            println!();
        }
    }
    Ok(())
}
