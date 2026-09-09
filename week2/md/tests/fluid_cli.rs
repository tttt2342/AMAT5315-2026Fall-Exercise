use md::{
    DEFAULT_CUTOFF, RunConfig, check_saved_run, compute_periodic_accelerations,
    default_run_config, run_simulation, shifted_energy,
};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn temporary_output(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("amat5315-md-{name}-{}", std::process::id()))
}

fn remove_output(path: &PathBuf) {
    let _ = fs::remove_dir_all(path);
}

#[test]
fn periodic_pair_forces_sum_to_zero() {
    let positions = vec![[0.1, 1.0], [11.9, 1.0], [6.0, 5.0]];
    let accelerations = compute_periodic_accelerations(&positions, [12.0, 10.0], DEFAULT_CUTOFF);
    let total = accelerations
        .iter()
        .fold([0.0, 0.0], |sum, acceleration| {
            [sum[0] + acceleration[0], sum[1] + acceleration[1]]
        });

    assert!(total[0].abs() < 1e-12, "net x force was {}", total[0]);
    assert!(total[1].abs() < 1e-12, "net y force was {}", total[1]);
}

#[test]
fn shifted_potential_is_continuous_at_cutoff() {
    let just_inside = shifted_energy(DEFAULT_CUTOFF - 1e-7, DEFAULT_CUTOFF);
    let at_cutoff = shifted_energy(DEFAULT_CUTOFF, DEFAULT_CUTOFF);
    let outside = shifted_energy(DEFAULT_CUTOFF + 1e-7, DEFAULT_CUTOFF);

    assert!(just_inside.abs() < 1e-6, "inside value was {just_inside}");
    assert_eq!(at_cutoff, 0.0);
    assert_eq!(outside, 0.0);
}

#[test]
fn default_contract_passes_physics_checks() {
    let output = temporary_output("contract");
    remove_output(&output);

    let config: RunConfig = default_run_config();
    run_simulation(&config, &output).expect("default simulation should run");
    let report = check_saved_run(&output).expect("saved trajectory should be readable");

    assert!(
        report.energy_drift < 2e-3,
        "energy drift was {}",
        report.energy_drift
    );
    assert!(
        (report.speed_temperature - config.temperature).abs() < 0.05,
        "speed temperature was {}",
        report.speed_temperature
    );
    assert!(
        report.speed_shape_chi_squared < 2.0,
        "speed shape statistic was {}",
        report.speed_shape_chi_squared
    );

    remove_output(&output);
}

#[test]
fn cli_run_writes_readable_jsonl_frames() {
    let output = temporary_output("cli-files");
    remove_output(&output);

    let status = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run",
            "--n",
            "16",
            "--rho",
            "0.8",
            "--temperature",
            "0.5",
            "--dt",
            "0.01",
            "--eq-steps",
            "10",
            "--steps",
            "20",
            "--sample-every",
            "10",
            "--seed",
            "2026",
            "--out",
            output.to_str().expect("temporary path is valid UTF-8"),
        ])
        .status()
        .expect("md binary should start");
    assert!(status.success(), "md run exited with {status}");

    let run_json = fs::read_to_string(output.join("run.json")).expect("run.json should exist");
    assert!(run_json.contains("\"integrator\": \"velocity-verlet\""));
    assert!(run_json.contains("\"box\""));

    let trajectory = fs::read_to_string(output.join("traj.jsonl"))
        .expect("traj.jsonl should exist");
    assert_eq!(trajectory.lines().count(), 2);
    assert!(trajectory.lines().all(|line| line.contains("\"E_pot\"") && line.contains("\"E_kin\"")));

    remove_output(&output);
}
