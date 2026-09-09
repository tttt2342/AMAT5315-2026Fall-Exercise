use md::{EnergySample, Euler, VelocityVerlet, dimer, run_experiment};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::path::PathBuf;

const DT: f64 = 0.01;
const SHORT_STEPS: usize = 500;
const LONG_STEPS: usize = 5000;

fn draw_error_curve<'a, DB: DrawingBackend>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    samples: &[EnergySample],
    scale: f64,
    color: RGBColor,
    label: &'static str,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    chart
        .draw_series(LineSeries::new(
            samples
                .iter()
                .map(|sample| (sample.time, scale * sample.relative_error)),
            color.stroke_width(2),
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(2)));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dimer.png");

    let initial = dimer();
    let euler_trace = run_experiment(&Euler, &initial, DT, SHORT_STEPS);
    let verlet_trace = run_experiment(&VelocityVerlet, &initial, DT, SHORT_STEPS);
    let long_verlet_trace = run_experiment(&VelocityVerlet, &initial, DT, LONG_STEPS);

    let root = BitMapBackend::new(&output_path, (1400, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    {
        let mut chart = ChartBuilder::on(&panels[0])
            .caption("Dimer total-energy error", ("sans-serif", 24))
            .margin(18)
            .x_label_area_size(42)
            .y_label_area_size(58)
            .build_cartesian_2d(0.0..5.0, -0.05..2.0)?;

        chart
            .configure_mesh()
            .x_desc("time t")
            .y_desc("(E(t) - E0) / |E0|")
            .draw()?;
        draw_error_curve(&mut chart, &euler_trace, 1.0, RED, "forward Euler")?;
        draw_error_curve(&mut chart, &verlet_trace, 1.0, BLUE, "velocity-Verlet")?;
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK)
            .draw()?;
    }

    {
        let mut chart = ChartBuilder::on(&panels[1])
            .caption("Velocity-Verlet over 50 time units", ("sans-serif", 24))
            .margin(18)
            .x_label_area_size(42)
            .y_label_area_size(58)
            .build_cartesian_2d(0.0..50.0, -0.5..0.5)?;

        chart
            .configure_mesh()
            .x_desc("time t")
            .y_desc("relative error × 1000")
            .draw()?;
        draw_error_curve(
            &mut chart,
            &long_verlet_trace,
            1000.0,
            BLUE,
            "velocity-Verlet × 1000",
        )?;
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()?;
    println!("saved {}", output_path.display());
    Ok(())
}
