use md::{energy, force};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::f64::consts::PI;
use std::path::PathBuf;

const HALF_WIDTH: f64 = 3.0;
const GRID_SIZE: usize = 120;
const CORE_RADIUS: f64 = 0.72;

fn equilibrium_distance() -> f64 {
    2.0_f64.powf(1.0 / 6.0)
}

fn energy_color(value: f64) -> HSLColor {
    let normalized = ((value.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0);
    HSLColor(0.66 * (1.0 - normalized), 0.9, 0.52)
}

fn draw_arrow<DB: DrawingBackend>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let end = (x + dx, y + dy);
    let angle = dy.atan2(dx);
    let head_length = 0.10;
    let left = (
        end.0 - head_length * (angle + PI * 0.84).cos(),
        end.1 - head_length * (angle + PI * 0.84).sin(),
    );
    let right = (
        end.0 - head_length * (angle - PI * 0.84).cos(),
        end.1 - head_length * (angle - PI * 0.84).sin(),
    );
    let style = ShapeStyle::from(&BLACK.mix(0.72)).stroke_width(1);

    chart.draw_series(std::iter::once(PathElement::new(vec![(x, y), end], style)))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![left, end, right],
        style,
    )))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("field.png");
    let root = BitMapBackend::new(&output_path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Lennard-Jones pair energy and force field",
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(45)
        .build_cartesian_2d(-HALF_WIDTH..HALF_WIDTH, -HALF_WIDTH..HALF_WIDTH)?;

    chart
        .configure_mesh()
        .x_desc("x / sigma")
        .y_desc("y / sigma")
        .light_line_style(WHITE.mix(0.25))
        .draw()?;

    let width = 2.0 * HALF_WIDTH / GRID_SIZE as f64;
    for ix in 0..GRID_SIZE {
        for iy in 0..GRID_SIZE {
            let x0 = -HALF_WIDTH + ix as f64 * width;
            let y0 = -HALF_WIDTH + iy as f64 * width;
            let x1 = x0 + width;
            let y1 = y0 + width;
            let r = (x0.mul_add(x0, y0 * y0)).sqrt();
            let value = if r < CORE_RADIUS { 1.0 } else { energy(r) };

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x1, y1)],
                energy_color(value).filled(),
            )))?;
        }
    }

    let arrow_spacing = 0.5;
    for ix in -6..=6 {
        for iy in -6..=6 {
            let x = ix as f64 * arrow_spacing;
            let y = iy as f64 * arrow_spacing;
            let r = (x.mul_add(x, y * y)).sqrt();
            if !(CORE_RADIUS + 0.12..=2.75).contains(&r) {
                continue;
            }

            let scalar_force = force(r);
            let arrow_length = 0.34 * (1.0 - (-0.08 * scalar_force.abs()).exp());
            draw_arrow(
                &mut chart,
                x,
                y,
                arrow_length * scalar_force * x / (r * scalar_force.abs()),
                arrow_length * scalar_force * y / (r * scalar_force.abs()),
            )?;
        }
    }

    chart.draw_series(std::iter::once(Circle::new((0.0, 0.0), 6, BLACK.filled())))?;

    let circle_points = (0..=128).map(|i| {
        let theta = 2.0 * PI * i as f64 / 128.0;
        let r0 = equilibrium_distance();
        (r0 * theta.cos(), r0 * theta.sin())
    });
    chart.draw_series(std::iter::once(PathElement::new(
        circle_points.collect::<Vec<_>>(),
        ShapeStyle::from(&BLACK.mix(0.8)).stroke_width(2),
    )))?;

    root.present()?;
    println!("saved {}", output_path.display());
    Ok(())
}
