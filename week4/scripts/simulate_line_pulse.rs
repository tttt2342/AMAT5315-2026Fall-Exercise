use continuum::{FourierAdvectionDiffusion, Integrator, RungeKutta4};
use std::env;
use std::f64::consts::{FRAC_PI_2, TAU};
use std::io::{self, BufWriter, Write};

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.05;
const SIGMA: f64 = 0.35;
const T_END: f64 = 6.0;

fn periodic_gaussian(x: f64) -> f64 {
    (-4..=4)
        .map(|image| {
            let distance = x - FRAC_PI_2 + image as f64 * TAU;
            (-distance * distance / (2.0 * SIGMA * SIGMA)).exp()
        })
        .sum()
}

fn write_row(output: &mut impl Write, time: f64, state: &[f64]) -> io::Result<()> {
    write!(output, "{time:.15e}")?;
    for value in state {
        write!(output, " {value:.15e}")?;
    }
    writeln!(output)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: line-pulse DT");
        std::process::exit(2);
    }
    let dt: f64 = args[1].parse().expect("DT must be a number");
    assert!(dt > 0.0, "DT must be positive");

    let equation = FourierAdvectionDiffusion::new(N, C, NU);
    let mut state: Vec<f64> = (0..N)
        .map(|j| periodic_gaussian(TAU * j as f64 / N as f64))
        .collect();
    let mut output = BufWriter::new(io::stdout().lock());
    let mut time = 0.0;
    write_row(&mut output, time, &state)?;

    while time < T_END {
        let step_size = dt.min(T_END - time);
        RungeKutta4.step(&mut state, step_size, &|u, du| equation.rate(u, du));
        time += step_size;
        if T_END - time < 32.0 * f64::EPSILON * T_END {
            time = T_END;
        }
        write_row(&mut output, time, &state)?;
    }
    Ok(())
}
