use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use plotters::prelude::*;
use serde::Deserialize;

const BIN_WIDTH: i32 = 40;
const SITE_COUNT: f64 = 4096.0;
const MIN_SHARED_COUNT: u32 = 5;
const T_COLD: f64 = 3.0;
const T_HOT: f64 = 3.1;

#[derive(Deserialize)]
struct SeriesRow {
    #[serde(rename = "L")]
    side: usize,
    #[serde(rename = "E")]
    energy_per_site: f64,
}

fn read_total_energies(path: &Path) -> Result<Vec<i32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut energies = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let row: SeriesRow = serde_json::from_str(&line?)?;
        if row.side * row.side != SITE_COUNT as usize {
            return Err(format!(
                "{} line {} has L={}, expected L=64",
                path.display(),
                index + 1,
                row.side
            )
            .into());
        }
        energies.push((row.energy_per_site * SITE_COUNT).round() as i32);
    }
    if energies.is_empty() {
        return Err(format!("{} contains no samples", path.display()).into());
    }
    Ok(energies)
}

fn histogram(energies: &[i32]) -> BTreeMap<i32, u32> {
    let mut counts = BTreeMap::new();
    for &energy in energies {
        let bin = energy.div_euclid(BIN_WIDTH);
        *counts.entry(bin).or_insert(0) += 1;
    }
    counts
}

fn main() -> Result<(), Box<dyn Error>> {
    let cold_energies = read_total_energies(Path::new("runs/T3.0/series.jsonl"))?;
    let hot_energies = read_total_energies(Path::new("runs/T3.1/series.jsonl"))?;
    let cold = histogram(&cold_energies);
    let hot = histogram(&hot_energies);

    let first_bin = *cold.keys().chain(hot.keys()).min().unwrap();
    let last_bin = *cold.keys().chain(hot.keys()).max().unwrap();
    let x_min = first_bin * BIN_WIDTH;
    let x_max = (last_bin + 1) * BIN_WIDTH;
    let max_count = cold.values().chain(hot.values()).copied().max().unwrap();
    let y_count_max = max_count.div_ceil(50) * 50;

    // Equal-width bins make the probability ratio the normalized count ratio.
    let ratio_points: Vec<(f64, f64)> = (first_bin..=last_bin)
        .filter_map(|bin| {
            let cold_count = cold.get(&bin).copied().unwrap_or(0);
            let hot_count = hot.get(&bin).copied().unwrap_or(0);
            if cold_count < MIN_SHARED_COUNT || hot_count < MIN_SHARED_COUNT {
                return None;
            }
            let energy = (bin * BIN_WIDTH + BIN_WIDTH / 2) as f64;
            let cold_probability = cold_count as f64 / cold_energies.len() as f64;
            let hot_probability = hot_count as f64 / hot_energies.len() as f64;
            Some((energy, (hot_probability / cold_probability).ln()))
        })
        .collect();
    if ratio_points.is_empty() {
        return Err("no bins have at least five samples in both runs".into());
    }

    // The partition-function ratio is an unknown additive constant. Fix only
    // that intercept; the theoretical slope itself is not fitted.
    let theoretical_slope = 1.0 / T_COLD - 1.0 / T_HOT;
    let theoretical_intercept = ratio_points
        .iter()
        .map(|(energy, ratio)| ratio - theoretical_slope * energy)
        .sum::<f64>()
        / ratio_points.len() as f64;
    let ratio_x_min = ratio_points.first().unwrap().0;
    let ratio_x_max = ratio_points.last().unwrap().0;
    let ratio_y_min = ratio_points
        .iter()
        .map(|(_, ratio)| *ratio)
        .fold(f64::INFINITY, f64::min)
        .floor();
    let ratio_y_max = ratio_points
        .iter()
        .map(|(_, ratio)| *ratio)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil();

    fs::create_dir_all("evidence")?;
    let root = BitMapBackend::new("evidence/boltzmann.png", (1600, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.margin(25, 25, 25, 25).split_evenly((1, 2));
    let blue = RGBColor(40, 105, 180);
    let red = RGBColor(210, 65, 55);

    let mut histogram_chart = ChartBuilder::on(&panels[0])
        .caption("Total-energy histograms (bin width 40)", ("sans-serif", 27))
        .margin(15)
        .x_label_area_size(55)
        .y_label_area_size(65)
        .build_cartesian_2d(x_min..x_max, 0_u32..y_count_max)?;
    histogram_chart
        .configure_mesh()
        .x_desc("total energy E")
        .y_desc("sweeps in bin")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 16))
        .light_line_style(RGBColor(225, 225, 225))
        .draw()?;

    histogram_chart
        .draw_series((first_bin..=last_bin).map(|bin| {
            let left = bin * BIN_WIDTH;
            let count = cold.get(&bin).copied().unwrap_or(0);
            Rectangle::new(
                [(left, 0), (left + BIN_WIDTH, count)],
                blue.mix(0.45).filled(),
            )
        }))?
        .label("T = 3.0")
        .legend(move |(x, y)| Rectangle::new([(x, y - 6), (x + 22, y + 6)], blue.filled()));
    histogram_chart
        .draw_series((first_bin..=last_bin).map(|bin| {
            let left = bin * BIN_WIDTH;
            let count = hot.get(&bin).copied().unwrap_or(0);
            Rectangle::new(
                [(left, 0), (left + BIN_WIDTH, count)],
                red.mix(0.45).filled(),
            )
        }))?
        .label("T = 3.1")
        .legend(move |(x, y)| Rectangle::new([(x, y - 6), (x + 22, y + 6)], red.filled()));
    histogram_chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 17))
        .draw()?;

    let mut ratio_chart = ChartBuilder::on(&panels[1])
        .caption("Boltzmann probability ratio", ("sans-serif", 27))
        .margin(15)
        .x_label_area_size(55)
        .y_label_area_size(75)
        .build_cartesian_2d(x_min as f64..x_max as f64, ratio_y_min..ratio_y_max)?;
    ratio_chart
        .configure_mesh()
        .x_desc("total energy E")
        .y_desc("ln(P3.1(E) / P3.0(E))")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 16))
        .light_line_style(RGBColor(225, 225, 225))
        .draw()?;

    ratio_chart
        .draw_series(
            ratio_points
                .iter()
                .map(|&(energy, ratio)| Circle::new((energy, ratio), 6, BLACK.filled())),
        )?
        .label("bins with both counts >= 5")
        .legend(|(x, y)| Circle::new((x + 11, y), 5, BLACK.filled()));
    ratio_chart
        .draw_series(DashedLineSeries::new(
            [ratio_x_min, ratio_x_max]
                .into_iter()
                .map(|energy| (energy, theoretical_slope * energy + theoretical_intercept)),
            9,
            7,
            red.stroke_width(3),
        ))?
        .label(format!("fixed slope = {theoretical_slope:.7}"))
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], red.stroke_width(3)));
    ratio_chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 17))
        .draw()?;

    root.present()?;
    println!(
        "wrote evidence/boltzmann.png from {} and {} samples; {} shared bins",
        cold_energies.len(),
        hot_energies.len(),
        ratio_points.len()
    );
    Ok(())
}
