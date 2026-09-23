use clap::{Parser, Subcommand};
use continuum::{random_velocity_field, taylor_green_field, InputField};
use std::error::Error;
use std::io;

#[derive(Debug, Parser)]
#[command(
    name = "field",
    about = "write a velocity field to stdout; no unstated defaults"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Taylor-Green exact velocity field.
    TaylorGreen {
        #[arg(long)]
        n: usize,
        #[arg(long, default_value_t = 0.0)]
        t: f64,
        #[arg(long)]
        nu: Option<f64>,
    },
    /// Random equal-amplitude vorticity modes with initial energy 0.5.
    Random {
        #[arg(long)]
        n: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        k_min: usize,
        #[arg(long)]
        k_max: usize,
    },
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn validate_n(n: usize) -> Result<(), io::Error> {
    if n == 0 || !n.is_multiple_of(2) {
        return Err(invalid_input("n must be positive and even"));
    }
    Ok(())
}

fn build_field(command: Command) -> Result<InputField, Box<dyn Error>> {
    match command {
        Command::TaylorGreen { n, t, nu } => {
            validate_n(n)?;
            if t < 0.0 {
                return Err(invalid_input("t must be non-negative").into());
            }
            let viscosity = match (t > 0.0, nu) {
                (true, None) => return Err(invalid_input("--nu is required when --t > 0").into()),
                (_, Some(value)) if value < 0.0 => {
                    return Err(invalid_input("nu must be non-negative").into());
                }
                (_, Some(value)) => value,
                (false, None) => 0.0,
            };
            Ok(taylor_green_field(n, viscosity, t))
        }
        Command::Random {
            n,
            seed,
            k_min,
            k_max,
        } => {
            validate_n(n)?;
            random_velocity_field(n, seed, k_min, k_max)
                .map_err(|message| invalid_input(message).into())
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let field = build_field(Cli::parse().command)?;
    serde_json::to_writer(io::stdout().lock(), &field)?;
    println!();
    Ok(())
}
