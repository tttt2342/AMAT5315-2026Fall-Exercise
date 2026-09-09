use plotters::prelude::*;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Copy)]
struct Timing {
    n: f64,
    naive_seconds: f64,
    cells_seconds: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let week2 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let input_path = week2.join("scaling.csv");
    let output_path = week2.join("scaling.png");

    let text = fs::read_to_string(&input_path)?;
    let mut timings = Vec::new();
    for (line_number, line) in text.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<f64> = line
            .split(',')
            .map(str::trim)
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if values.len() != 3 {
            return Err(format!("{} must have three columns", line_number + 1).into());
        }
        timings.push(Timing {
            n: values[0],
            naive_seconds: values[1],
            cells_seconds: values[2],
        });
    }
    if timings.is_empty() {
        return Err("scaling.csv contains no measurements".into());
    }

    let min_n = timings
        .iter()
        .map(|timing| timing.n)
        .fold(f64::INFINITY, f64::min);
    let max_n = timings.iter().map(|timing| timing.n).fold(0.0, f64::max);
    let seconds_per_step: Vec<f64> = timings
        .iter()
        .flat_map(|timing| [timing.naive_seconds, timing.cells_seconds])
        .map(|seconds| seconds / 500.0)
        .collect();
    let min_seconds_per_step = seconds_per_step
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_seconds_per_step = seconds_per_step.iter().copied().fold(0.0, f64::max);

    let root = BitMapBackend::new(&output_path, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Force-path scaling", ("sans-serif", 28))
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(72)
        .build_cartesian_2d(
            (min_n * 0.8..max_n * 1.2).log_scale(),
            (min_seconds_per_step * 0.7..max_seconds_per_step * 1.5).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc("N (atoms)")
        .y_desc("seconds / step")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            timings
                .iter()
                .map(|timing| (timing.n, timing.naive_seconds / 500.0)),
            RED.stroke_width(3),
        ))?
        .label("naive")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], RED.stroke_width(3)));
    chart.draw_series(
        timings
            .iter()
            .map(|timing| Circle::new((timing.n, timing.naive_seconds / 500.0), 5, RED.filled())),
    )?;

    chart
        .draw_series(LineSeries::new(
            timings
                .iter()
                .map(|timing| (timing.n, timing.cells_seconds / 500.0)),
            BLUE.stroke_width(3),
        ))?
        .label("cells")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], BLUE.stroke_width(3)));
    chart.draw_series(
        timings
            .iter()
            .map(|timing| Circle::new((timing.n, timing.cells_seconds / 500.0), 5, BLUE.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.88))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("saved {}", output_path.display());
    Ok(())
}
