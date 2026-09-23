use clap::{Parser, ValueEnum};
use continuum::{
    diagnostics, ExplicitMidpoint, FlowFields, ForwardEuler, InputField, Integrator, RungeKutta4,
    VorticityEquation,
};
use serde::Serialize;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Method {
    Euler,
    Rk2,
    Rk4,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Rk2 => "rk2",
            Self::Rk4 => "rk4",
        }
    }

    fn integrator(self) -> Box<dyn Integrator> {
        match self {
            Self::Euler => Box::new(ForwardEuler),
            Self::Rk2 => Box::new(ExplicitMidpoint),
            Self::Rk4 => Box::new(RungeKutta4),
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "fluid",
    about = "integrate the field on stdin with a chosen time integrator, record frames; no unstated defaults"
)]
struct Cli {
    #[arg(long, value_enum)]
    method: Method,
    #[arg(long)]
    nu: f64,
    #[arg(long)]
    dt: f64,
    #[arg(long)]
    t_end: f64,
    #[arg(long)]
    every: f64,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Serialize)]
struct RunMetadata<'a> {
    case: &'a str,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[usize; 2]>,
    method: &'a str,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: f64,
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn validate(cli: &Cli, field: &InputField) -> Result<(), io::Error> {
    if field.n == 0 || !field.n.is_multiple_of(2) {
        return Err(invalid_input("input n must be positive and even"));
    }
    if field.u.len() != field.n * field.n || field.v.len() != field.n * field.n {
        return Err(invalid_input("input u and v must each contain n*n values"));
    }
    if cli.nu < 0.0 || cli.dt <= 0.0 || cli.t_end < 0.0 || cli.every <= 0.0 {
        return Err(invalid_input(
            "require nu >= 0, dt > 0, t-end >= 0, and every > 0",
        ));
    }
    Ok(())
}

fn write_array(writer: &mut impl Write, values: &[f64]) -> io::Result<()> {
    write!(writer, "[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{value:.6}")?;
    }
    write!(writer, "]")
}

fn write_frame(
    writer: &mut impl Write,
    time: f64,
    step: usize,
    fields: &FlowFields,
) -> io::Result<()> {
    write!(writer, "{{\"t\":{time:.6},\"step\":{step},\"u\":")?;
    write_array(writer, &fields.u)?;
    write!(writer, ",\"v\":")?;
    write_array(writer, &fields.v)?;
    write!(writer, ",\"omega\":")?;
    write_array(writer, &fields.omega)?;
    writeln!(writer, "}}")
}

fn write_diagnostic(time: f64, energy: f64, enstrophy: f64) {
    println!("{time:.6}\t{energy:.6}\t{enstrophy:.6}");
}

fn save_if_finite(
    fields_writer: &mut impl Write,
    time: f64,
    step: usize,
    fields: &FlowFields,
    is_snapshot: bool,
) -> Result<bool, io::Error> {
    let (energy, enstrophy) = diagnostics(fields);
    if !energy.is_finite() || !enstrophy.is_finite() {
        write_diagnostic(time, energy, enstrophy);
        return Ok(false);
    }
    if is_snapshot {
        write_diagnostic(time, energy, enstrophy);
        write_frame(fields_writer, time, step, fields)?;
    }
    Ok(true)
}

fn write_run_metadata(path: &Path, field: &InputField, cli: &Cli) -> Result<(), Box<dyn Error>> {
    let metadata = RunMetadata {
        case: &field.case,
        n: field.n,
        seed: field.seed,
        k_band: field.k_band,
        method: cli.method.name(),
        nu: cli.nu,
        dt: cli.dt,
        t_end: cli.t_end,
        snapshot_every: cli.every,
    };
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, &metadata)?;
    Ok(())
}

fn run(cli: Cli, input: InputField) -> Result<bool, Box<dyn Error>> {
    validate(&cli, &input)?;
    fs::create_dir_all(&cli.out)?;
    write_run_metadata(&cli.out.join("run.json"), &input, &cli)?;
    let mut fields_writer = BufWriter::new(File::create(cli.out.join("fields.jsonl"))?);

    let equation = VorticityEquation::new(input.n, cli.nu);
    let mut omega = equation.initial_vorticity(&input.u, &input.v);
    equation.project(&mut omega);
    let integrator = cli.method.integrator();

    println!("t\tE\tZ");
    let initial_fields = equation.fields(&omega);
    if !save_if_finite(&mut fields_writer, 0.0, 0, &initial_fields, true)? {
        fields_writer.flush()?;
        return Ok(false);
    }

    let mut time = 0.0;
    let mut step = 0_usize;
    let mut next_snapshot = cli.every;
    while time < cli.t_end {
        let output_target = next_snapshot.min(cli.t_end);
        let step_size = cli.dt.min(output_target - time).min(cli.t_end - time);
        integrator.step(&mut omega, step_size, &|state, rate| {
            equation.rate(state, rate)
        });
        equation.project(&mut omega);
        step += 1;
        time += step_size;
        if cli.t_end - time < 32.0 * f64::EPSILON * cli.t_end.max(1.0) {
            time = cli.t_end;
        }

        let fields = equation.fields(&omega);
        let tolerance = 32.0 * f64::EPSILON * time.max(1.0);
        let at_scheduled_snapshot = (time - next_snapshot).abs() <= tolerance;
        let is_snapshot = at_scheduled_snapshot || time == cli.t_end;
        if !save_if_finite(&mut fields_writer, time, step, &fields, is_snapshot)? {
            fields_writer.flush()?;
            return Ok(false);
        }
        if at_scheduled_snapshot {
            next_snapshot += cli.every;
        }
    }
    fields_writer.flush()?;
    Ok(true)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let input: InputField = serde_json::from_reader(io::stdin().lock())?;
    if !run(cli, input)? {
        io::stdout().flush()?;
        std::process::exit(1);
    }
    Ok(())
}
