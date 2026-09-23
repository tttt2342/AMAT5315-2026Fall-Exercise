use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn taylor_green_pipeline_writes_expected_outputs() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "16"])
        .output()
        .unwrap();
    assert!(field.status.success());

    let output_directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method",
            "rk4",
            "--nu",
            "0.1",
            "--dt",
            "0.01",
            "--t-end",
            "0.1",
            "--every",
            "0.05",
            "--out",
            output_directory.path().to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&field.stdout)
        .unwrap();
    let fluid = child.wait_with_output().unwrap();
    assert!(fluid.status.success());

    let stdout = String::from_utf8(fluid.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines[0], "t\tE\tZ");
    assert_eq!(lines.len(), 4);
    let final_values: Vec<f64> = lines[3]
        .split('\t')
        .map(|value| value.parse().unwrap())
        .collect();
    assert!((final_values[0] - 0.1).abs() < 1.0e-12);
    assert!((final_values[1] - 0.25 * (-0.04_f64).exp()).abs() < 1.0e-6);
    assert!((final_values[2] - 0.5 * (-0.04_f64).exp()).abs() < 1.0e-6);

    let run: Value = serde_json::from_reader(
        std::fs::File::open(output_directory.path().join("run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(run["case"], "taylor-green");
    assert_eq!(run["method"], "rk4");

    let frames = std::fs::read_to_string(output_directory.path().join("fields.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 3);
    for line in frames.lines() {
        let frame: Value = serde_json::from_str(line).unwrap();
        assert_eq!(frame["u"].as_array().unwrap().len(), 16 * 16);
        assert_eq!(frame["v"].as_array().unwrap().len(), 16 * 16);
        assert_eq!(frame["omega"].as_array().unwrap().len(), 16 * 16);
    }
}

#[test]
fn taylor_green_requires_viscosity_at_positive_time() {
    let output = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "16", "--t", "1"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn snapshots_land_on_requested_times_when_dt_does_not_divide_interval() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "16"])
        .output()
        .unwrap();
    assert!(field.status.success());

    let output_directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method",
            "rk4",
            "--nu",
            "0.1",
            "--dt",
            "0.03",
            "--t-end",
            "0.1",
            "--every",
            "0.05",
            "--out",
            output_directory.path().to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&field.stdout)
        .unwrap();
    let fluid = child.wait_with_output().unwrap();
    assert!(fluid.status.success());

    let stdout = String::from_utf8(fluid.stdout).unwrap();
    let times: Vec<f64> = stdout
        .lines()
        .skip(1)
        .map(|line| line.split('\t').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(times, [0.0, 0.05, 0.1]);
}
