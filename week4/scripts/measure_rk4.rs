use continuum::{Integrator, RungeKutta4};
use std::env;
use std::io::{self, BufWriter, Write};

fn parse_arg<T: std::str::FromStr>(args: &[String], index: usize, name: &str) -> T {
    args.get(index)
        .unwrap_or_else(|| panic!("missing argument {name}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid argument {name}"))
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        eprintln!("usage: measure-rk4 NX NY XMIN XMAX YMIN YMAX");
        std::process::exit(2);
    }

    let nx: usize = parse_arg(&args, 1, "NX");
    let ny: usize = parse_arg(&args, 2, "NY");
    let xmin: f64 = parse_arg(&args, 3, "XMIN");
    let xmax: f64 = parse_arg(&args, 4, "XMAX");
    let ymin: f64 = parse_arg(&args, 5, "YMIN");
    let ymax: f64 = parse_arg(&args, 6, "YMAX");
    assert!(nx >= 2 && ny >= 2, "the grid must be at least 2 by 2");
    assert!(xmin < xmax && ymin < ymax, "axis limits must increase");

    let mut output = BufWriter::new(io::stdout().lock());
    for row in 0..ny {
        let imaginary = ymin + (ymax - ymin) * row as f64 / (ny - 1) as f64;
        for column in 0..nx {
            let real = xmin + (xmax - xmin) * column as f64 / (nx - 1) as f64;
            let mut state = [1.0, 0.0];
            RungeKutta4.step(&mut state, 1.0, &|y, dy| {
                dy[0] = real * y[0] - imaginary * y[1];
                dy[1] = imaginary * y[0] + real * y[1];
            });
            let growth = state[0].hypot(state[1]);
            write!(output, "{growth:.15e} ")?;
        }
        writeln!(output)?;
    }
    Ok(())
}
