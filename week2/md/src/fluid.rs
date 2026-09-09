use crate::{
    DEFAULT_CUTOFF, ForceMode, System, Vec2, VelocityVerlet, advance, kinetic_energy,
    minimum_image, potential_energy, shifted_energy,
};
use clap::ValueEnum;
use plotters::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::Command;

pub type FluidResult<T> = Result<T, Box<dyn Error>>;

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[derive(Clone, Copy, Debug, Deserialize, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum IntegratorChoice {
    VelocityVerlet,
}

impl IntegratorChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VelocityVerlet => "velocity-verlet",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_size: Vec2,
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    #[serde(default)]
    pub force: ForceMode,
    #[serde(default)]
    pub ramp_to: Option<f64>,
}

impl RunConfig {
    pub fn with_parameters(
        n: usize,
        rho: f64,
        temperature: f64,
        dt: f64,
        eq_steps: usize,
        steps: usize,
        sample_every: usize,
        seed: u64,
    ) -> FluidResult<Self> {
        Self::with_parameters_and_options(
            n,
            rho,
            temperature,
            dt,
            eq_steps,
            steps,
            sample_every,
            seed,
            ForceMode::Cells,
            None,
        )
    }

    pub fn with_parameters_and_options(
        n: usize,
        rho: f64,
        temperature: f64,
        dt: f64,
        eq_steps: usize,
        steps: usize,
        sample_every: usize,
        seed: u64,
        force: ForceMode,
        ramp_to: Option<f64>,
    ) -> FluidResult<Self> {
        let (_, box_size) = triangular_lattice(n, rho)?;
        let config = Self {
            n,
            rho,
            box_size,
            dt,
            temperature,
            eq_steps,
            steps,
            sample_every,
            seed,
            integrator: IntegratorChoice::VelocityVerlet.as_str().to_owned(),
            force,
            ramp_to,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> FluidResult<()> {
        if self.n < 2 {
            return Err(invalid("n must be at least 2"));
        }
        if self.rho <= 0.0 || !self.rho.is_finite() {
            return Err(invalid("rho must be finite and positive"));
        }
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err(invalid("temperature must be finite and positive"));
        }
        if let Some(ramp_to) = self.ramp_to {
            if ramp_to <= 0.0 || !ramp_to.is_finite() {
                return Err(invalid("ramp-to temperature must be finite and positive"));
            }
        }
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(invalid("dt must be finite and positive"));
        }
        if self.sample_every == 0 {
            return Err(invalid("sample-every must be positive"));
        }
        if self.steps < self.sample_every {
            return Err(invalid("steps must produce at least one saved frame"));
        }
        if self.integrator != IntegratorChoice::VelocityVerlet.as_str() {
            return Err(invalid(format!(
                "unsupported integrator {:?}; expected velocity-verlet",
                self.integrator
            )));
        }
        let [lx, ly] = self.box_size;
        if !(lx.is_finite() && ly.is_finite() && lx > 0.0 && ly > 0.0) {
            return Err(invalid("box lengths must be finite and positive"));
        }
        let density = self.n as f64 / (lx * ly);
        if (density - self.rho).abs() > 1e-10 * self.rho.max(1.0) {
            return Err(invalid("box and density do not agree"));
        }
        if DEFAULT_CUTOFF >= 0.5 * lx || DEFAULT_CUTOFF >= 0.5 * ly {
            return Err(invalid("the box must be larger than twice the cutoff"));
        }
        Ok(())
    }
}

pub fn default_run_config() -> RunConfig {
    RunConfig::with_parameters(100, 0.8, 0.5, 0.01, 2000, 10000, 50, 2026)
        .expect("the Part 4 default configuration is valid")
}

/// Generate a square triangular lattice and its periodic box.
pub fn triangular_lattice(n: usize, rho: f64) -> FluidResult<(Vec<Vec2>, Vec2)> {
    if n == 0 {
        return Err(invalid("n must be positive"));
    }
    if rho <= 0.0 || !rho.is_finite() {
        return Err(invalid("rho must be finite and positive"));
    }

    let side = (n as f64).sqrt().round() as usize;
    if side * side != n {
        return Err(invalid("n must be a perfect square for the lattice"));
    }
    if side % 2 != 0 {
        return Err(invalid("the number of lattice rows must be even"));
    }

    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 3.0_f64.sqrt() * a / 2.0;
    let box_size = [side as f64 * a, side as f64 * h];
    let mut positions = Vec::with_capacity(n);

    for j in 0..side {
        for i in 0..side {
            let offset = 0.5 * (j % 2) as f64;
            positions.push([(i as f64 + offset) * a, j as f64 * h]);
        }
    }

    Ok((positions, box_size))
}

fn remove_center_of_mass_velocity(velocities: &mut [Vec2]) {
    let n = velocities.len() as f64;
    let mean = velocities.iter().fold([0.0, 0.0], |sum, velocity| {
        [sum[0] + velocity[0], sum[1] + velocity[1]]
    });
    let mean = [mean[0] / n, mean[1] / n];

    for velocity in velocities {
        velocity[0] -= mean[0];
        velocity[1] -= mean[1];
    }
}

pub fn rescale_velocities(velocities: &mut [Vec2], target_temperature: f64) -> FluidResult<()> {
    if velocities.len() < 2 {
        return Err(invalid(
            "at least two atoms are needed for temperature scaling",
        ));
    }
    if target_temperature <= 0.0 || !target_temperature.is_finite() {
        return Err(invalid("target temperature must be finite and positive"));
    }

    remove_center_of_mass_velocity(velocities);
    let kinetic: f64 = velocities
        .iter()
        .map(|velocity| 0.5 * (velocity[0].powi(2) + velocity[1].powi(2)))
        .sum();
    let degrees_of_freedom = 2.0 * (velocities.len() as f64 - 1.0);
    let thermostat_temperature = 2.0 * kinetic / degrees_of_freedom;
    if thermostat_temperature <= 0.0 || !thermostat_temperature.is_finite() {
        return Err(invalid(
            "velocity rescaling encountered zero or non-finite temperature",
        ));
    }

    let scale = (target_temperature / thermostat_temperature).sqrt();
    for velocity in velocities {
        velocity[0] *= scale;
        velocity[1] *= scale;
    }
    Ok(())
}

fn initial_fluid_system(config: &RunConfig) -> FluidResult<System> {
    let (positions, lattice_box) = triangular_lattice(config.n, config.rho)?;
    if lattice_box != config.box_size {
        return Err(invalid("run configuration box does not match its lattice"));
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let standard_deviation = config.temperature.sqrt();
    let velocities = (0..config.n)
        .map(|_| {
            let x: f64 = StandardNormal.sample(&mut rng);
            let y: f64 = StandardNormal.sample(&mut rng);
            [standard_deviation * x, standard_deviation * y]
        })
        .collect();

    let mut system = System::periodic_with_force(
        positions,
        velocities,
        config.box_size,
        DEFAULT_CUTOFF,
        config.force,
    );
    rescale_velocities(&mut system.velocities, config.temperature)?;
    Ok(system)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryFrame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

impl TrajectoryFrame {
    fn from_system(step: usize, dt: f64, system: &System) -> Self {
        Self {
            step,
            t: step as f64 * dt,
            pos: system.positions.clone(),
            vel: system.velocities.clone(),
            e_pot: potential_energy(system),
            e_kin: kinetic_energy(system),
        }
    }
}

pub fn run_simulation(config: &RunConfig, output: &Path) -> FluidResult<()> {
    config.validate()?;
    fs::create_dir_all(output)?;

    let run_file = File::create(output.join("run.json"))?;
    let mut run_writer = BufWriter::new(run_file);
    serde_json::to_writer_pretty(&mut run_writer, config)?;
    writeln!(run_writer)?;

    let trajectory_file = File::create(output.join("traj.jsonl"))?;
    let mut trajectory_writer = BufWriter::new(trajectory_file);
    let mut system = initial_fluid_system(config)?;
    let integrator = VelocityVerlet;

    for step in 1..=config.eq_steps {
        advance(&integrator, &mut system, config.dt);
        if step % 50 == 0 {
            rescale_velocities(&mut system.velocities, config.temperature)?;
        }
    }

    for step in 1..=config.steps {
        advance(&integrator, &mut system, config.dt);
        if let Some(ramp_to) = config.ramp_to {
            if step % 50 == 0 || step == config.steps {
                let fraction = step as f64 / config.steps as f64;
                let target_temperature =
                    config.temperature + fraction * (ramp_to - config.temperature);
                rescale_velocities(&mut system.velocities, target_temperature)?;
            }
        }
        if step % config.sample_every == 0 {
            let frame = TrajectoryFrame::from_system(step, config.dt, &system);
            serde_json::to_writer(&mut trajectory_writer, &frame)?;
            writeln!(trajectory_writer)?;
        }
    }

    trajectory_writer.flush()?;
    run_writer.flush()?;
    Ok(())
}

fn read_artifacts(directory: &Path) -> FluidResult<(RunConfig, Vec<TrajectoryFrame>)> {
    let run_text = fs::read_to_string(directory.join("run.json"))?;
    let config: RunConfig = serde_json::from_str(&run_text)?;
    config.validate()?;

    let trajectory_file = File::open(directory.join("traj.jsonl"))?;
    let reader = BufReader::new(trajectory_file);
    let mut frames = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            return Err(invalid(format!(
                "traj.jsonl line {} is empty",
                line_number + 1
            )));
        }
        let frame: TrajectoryFrame = serde_json::from_str(&line).map_err(|error| {
            invalid(format!(
                "invalid traj.jsonl line {}: {error}",
                line_number + 1
            ))
        })?;
        frames.push(frame);
    }
    if frames.is_empty() {
        return Err(invalid("trajectory contains no production frames"));
    }

    let expected_frames = config.steps / config.sample_every;
    if frames.len() != expected_frames {
        return Err(invalid(format!(
            "expected {expected_frames} saved frames, found {}",
            frames.len()
        )));
    }
    for (index, frame) in frames.iter().enumerate() {
        let expected_step = (index + 1) * config.sample_every;
        if frame.step != expected_step {
            return Err(invalid(format!(
                "frame {} has step {}, expected {expected_step}",
                index, frame.step
            )));
        }
        if (frame.t - frame.step as f64 * config.dt).abs() > 1e-10 * (1.0 + frame.t.abs()) {
            return Err(invalid(format!(
                "frame {} has inconsistent time {}",
                index, frame.t
            )));
        }
        if frame.pos.len() != config.n || frame.vel.len() != config.n {
            return Err(invalid(format!(
                "frame {} has the wrong number of atoms",
                index
            )));
        }
        for (position, velocity) in frame.pos.iter().zip(&frame.vel) {
            if !(position[0].is_finite()
                && position[1].is_finite()
                && velocity[0].is_finite()
                && velocity[1].is_finite())
            {
                return Err(invalid(format!("frame {index} contains non-finite state")));
            }
            if !(0.0 <= position[0]
                && position[0] < config.box_size[0]
                && 0.0 <= position[1]
                && position[1] < config.box_size[1])
            {
                return Err(invalid(format!(
                    "frame {index} contains an unwrapped position"
                )));
            }
        }
        if !(frame.e_pot.is_finite() && frame.e_kin.is_finite()) {
            return Err(invalid(format!("frame {index} contains non-finite energy")));
        }
    }

    Ok((config, frames))
}

fn frame_energies(frame: &TrajectoryFrame, config: &RunConfig) -> FluidResult<(f64, f64)> {
    let mut e_pot = 0.0;
    for i in 0..config.n {
        for j in (i + 1)..config.n {
            let displacement = minimum_image(
                [
                    frame.pos[i][0] - frame.pos[j][0],
                    frame.pos[i][1] - frame.pos[j][1],
                ],
                config.box_size,
            );
            let distance = (displacement[0].powi(2) + displacement[1].powi(2)).sqrt();
            if !distance.is_finite() || distance == 0.0 {
                return Err(invalid("trajectory contains overlapping atoms"));
            }
            e_pot += shifted_energy(distance, DEFAULT_CUTOFF);
        }
    }

    let e_kin = frame
        .vel
        .iter()
        .map(|velocity| 0.5 * (velocity[0].powi(2) + velocity[1].powi(2)))
        .sum();
    Ok((e_pot, e_kin))
}

fn close_enough(expected: f64, actual: f64) -> bool {
    (expected - actual).abs() <= 1e-9 * expected.abs().max(actual.abs()).max(1.0)
}

#[derive(Clone, Copy, Debug)]
pub struct CheckReport {
    pub frame_count: usize,
    pub energy_drift: f64,
    pub speed_temperature: f64,
    pub speed_shape_chi_squared: f64,
    pub energy_consistent: bool,
    pub passed: bool,
}

pub fn check_saved_run(directory: &Path) -> FluidResult<CheckReport> {
    let (config, frames) = read_artifacts(directory)?;
    let mut energies = Vec::with_capacity(frames.len());
    let mut energy_consistent = true;
    let mut speeds = Vec::with_capacity(config.n * frames.len());

    for frame in &frames {
        let (recomputed_pot, recomputed_kin) = frame_energies(frame, &config)?;
        energy_consistent &= close_enough(recomputed_pot, frame.e_pot);
        energy_consistent &= close_enough(recomputed_kin, frame.e_kin);
        energies.push(recomputed_pot + recomputed_kin);
        speeds.extend(
            frame
                .vel
                .iter()
                .map(|velocity| (velocity[0].powi(2) + velocity[1].powi(2)).sqrt()),
        );
    }

    let k = (frames.len() / 10).max(1);
    let first_mean = energies[..k].iter().sum::<f64>() / k as f64;
    let last_mean = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    let energy_drift = (last_mean - first_mean).abs() / energies[0].abs();

    let speed_temperature =
        speeds.iter().map(|speed| speed.powi(2)).sum::<f64>() / (2.0 * speeds.len() as f64);
    let speed_shape_chi_squared = speed_shape_chi_squared(&speeds, speed_temperature)?;
    let passed = energy_consistent
        && energy_drift < 2e-3
        && (speed_temperature - config.temperature).abs() < 0.05
        && speed_shape_chi_squared < 2.0;

    Ok(CheckReport {
        frame_count: frames.len(),
        energy_drift,
        speed_temperature,
        speed_shape_chi_squared,
        energy_consistent,
        passed,
    })
}

fn speed_shape_chi_squared(speeds: &[f64], temperature: f64) -> FluidResult<f64> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(invalid("speed temperature must be finite and positive"));
    }

    let mut observed = [0usize; 24];
    for &speed in speeds {
        let mut bin = 23;
        for k in 1..24 {
            let probability = k as f64 / 24.0;
            let edge = (-2.0 * temperature * (1.0 - probability).ln()).sqrt();
            if speed < edge {
                bin = k - 1;
                break;
            }
        }
        observed[bin] += 1;
    }

    let expected = speeds.len() as f64 / 24.0;
    let chi_squared = observed
        .iter()
        .map(|&count| {
            let difference = count as f64 - expected;
            difference * difference / expected
        })
        .sum::<f64>()
        / 22.0;
    Ok(chi_squared)
}

pub const RDF_BINS: usize = 80;

pub struct RdfAccumulator {
    counts: Vec<f64>,
    frames_seen: usize,
    n_atoms: usize,
    rho: f64,
    box_size: Vec2,
    r_max: f64,
    dr: f64,
}

impl RdfAccumulator {
    pub fn new(config: &RunConfig) -> Self {
        let r_max = 0.5 * config.box_size[0].min(config.box_size[1]);
        Self {
            counts: vec![0.0; RDF_BINS],
            frames_seen: 0,
            n_atoms: config.n,
            rho: config.rho,
            box_size: config.box_size,
            r_max,
            dr: r_max / RDF_BINS as f64,
        }
    }

    pub fn add_frame(&mut self, frame: &TrajectoryFrame) {
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                let displacement = minimum_image(
                    [
                        frame.pos[i][0] - frame.pos[j][0],
                        frame.pos[i][1] - frame.pos[j][1],
                    ],
                    self.box_size,
                );
                let distance = (displacement[0].powi(2) + displacement[1].powi(2)).sqrt();
                if distance < self.r_max {
                    let bin = (distance / self.dr).floor() as usize;
                    if bin < self.counts.len() {
                        self.counts[bin] += 2.0;
                    }
                }
            }
        }
        self.frames_seen += 1;
    }

    pub fn curve(&self) -> Vec<(f64, f64)> {
        let normalization = self.frames_seen as f64 * self.n_atoms as f64 * self.rho;
        self.counts
            .iter()
            .enumerate()
            .map(|(bin, &count)| {
                let r0 = bin as f64 * self.dr;
                let r1 = r0 + self.dr;
                let expected_ring_area = PI * (r1.powi(2) - r0.powi(2));
                let g = if normalization == 0.0 {
                    0.0
                } else {
                    count / (normalization * expected_ring_area)
                };
                (0.5 * (r0 + r1), g)
            })
            .collect()
    }

    pub fn r_max(&self) -> f64 {
        self.r_max
    }
}

pub fn render_video(directory: &Path, output: &Path) -> FluidResult<()> {
    let (config, frames) = read_artifacts(directory)?;
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent)?;
    }

    let temporary_frames = tempfile::tempdir()?;
    let mut rdf = RdfAccumulator::new(&config);
    for (index, frame) in frames.iter().enumerate() {
        rdf.add_frame(frame);
        let frame_path = temporary_frames
            .path()
            .join(format!("frame_{index:05}.png"));
        draw_video_frame(&frame_path, &config, frame, &rdf, index + 1)?;
    }

    let input_pattern = temporary_frames.path().join("frame_%05d.png");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "20", "-i"])
        .arg(&input_pattern)
        .args([
            "-c:v", "libx264", "-preset", "medium", "-crf", "32", "-pix_fmt", "yuv420p",
        ])
        .arg(output)
        .status()
        .map_err(|error| invalid(format!("could not run ffmpeg: {error}")))?;
    if !status.success() {
        return Err(invalid("ffmpeg failed to encode the video"));
    }

    let size = fs::metadata(output)?.len();
    if size >= 2 * 1024 * 1024 {
        return Err(invalid(format!(
            "video is too large: {} bytes (limit is 2 MiB)",
            size
        )));
    }
    Ok(())
}

fn draw_video_frame(
    path: &Path,
    config: &RunConfig,
    frame: &TrajectoryFrame,
    rdf: &RdfAccumulator,
    frame_number: usize,
) -> FluidResult<()> {
    let root = BitMapBackend::new(path, (1000, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    {
        let mut chart = ChartBuilder::on(&panels[0])
            .caption(
                format!("Lennard-Jones fluid  t = {:.2}", frame.t),
                ("sans-serif", 20),
            )
            .margin(16)
            .x_label_area_size(34)
            .y_label_area_size(38)
            .build_cartesian_2d(0.0..config.box_size[0], 0.0..config.box_size[1])?;
        chart.configure_mesh().x_desc("x").y_desc("y").draw()?;
        chart.draw_series(
            frame
                .pos
                .iter()
                .map(|position| Circle::new((position[0], position[1]), 3, BLUE.filled())),
        )?;
    }

    {
        let curve = rdf.curve();
        let observed_max = curve.iter().map(|&(_, value)| value).fold(0.0, f64::max);
        let y_max = (observed_max * 1.2).clamp(2.0, 6.0);
        let mut chart = ChartBuilder::on(&panels[1])
            .caption(
                format!("Radial distribution g(r), frame {frame_number}"),
                ("sans-serif", 20),
            )
            .margin(16)
            .x_label_area_size(34)
            .y_label_area_size(38)
            .build_cartesian_2d(0.0..rdf.r_max(), 0.0..y_max)?;
        chart.configure_mesh().x_desc("r").y_desc("g(r)").draw()?;
        chart.draw_series(LineSeries::new(curve, RED.stroke_width(2)))?;
        chart.draw_series(LineSeries::new(
            [(0.0, 1.0), (rdf.r_max(), 1.0)],
            BLACK.mix(0.35).stroke_width(1),
        ))?;
    }

    root.present()?;
    Ok(())
}
