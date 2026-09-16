use std::fs;
use std::process::Command;

use serde_json::Value;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ising"))
}

#[test]
fn writes_contract_files_and_expected_record_counts() {
    let temporary = tempfile::tempdir().unwrap();
    let output = command()
        .args([
            "--update",
            "metropolis",
            "--l",
            "4",
            "--t-from",
            "1.5",
            "--t-to",
            "1.6",
            "--t-step",
            "0.05",
            "--discard",
            "2",
            "--measure",
            "4",
            "--every",
            "2",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(temporary.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 4);
    assert_eq!(
        stdout.lines().next().unwrap(),
        "T\tmean_abs_M\tacceptance_rate"
    );

    let metadata: Value =
        serde_json::from_slice(&fs::read(temporary.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(metadata["L"], 4);
    assert_eq!(metadata["update"], "metropolis");
    assert_eq!(metadata["sample_every"], 1);
    assert_eq!(metadata["time_unit"], "sweep");
    assert_eq!(metadata["t_grid"].as_array().unwrap().len(), 3);

    let series = fs::read_to_string(temporary.path().join("series.jsonl")).unwrap();
    assert_eq!(series.lines().count(), 12);
    for line in series.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        assert!(row.get("M").is_some());
        assert!(row.get("E").is_some());
    }

    let frames = fs::read_to_string(temporary.path().join("spins.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 6);
    let sweeps: Vec<u64> = frames
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["sweep"]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert_eq!(sweeps, vec![4, 6, 10, 12, 16, 18]);
}

#[test]
fn same_seed_produces_identical_series() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    for output_directory in [first.path(), second.path()] {
        let status = command()
            .args([
                "--update",
                "metropolis",
                "--l",
                "8",
                "--t-from",
                "2.3",
                "--t-to",
                "2.3",
                "--t-step",
                "0.1",
                "--discard",
                "5",
                "--measure",
                "10",
                "--seed",
                "99",
                "--out",
            ])
            .arg(output_directory)
            .status()
            .unwrap();
        assert!(status.success());
    }
    assert_eq!(
        fs::read(first.path().join("series.jsonl")).unwrap(),
        fs::read(second.path().join("series.jsonl")).unwrap()
    );
}

#[test]
fn wolff_writes_cluster_metadata_and_measurements() {
    let temporary = tempfile::tempdir().unwrap();
    let output = command()
        .args([
            "--update",
            "wolff",
            "--l",
            "4",
            "--t-from",
            "2.3",
            "--t-to",
            "2.3",
            "--t-step",
            "0.1",
            "--discard",
            "2",
            "--measure",
            "4",
            "--every",
            "2",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(temporary.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 2);
    assert_eq!(
        stdout.lines().next().unwrap(),
        "T\tmean_abs_M\tmean_cluster_size"
    );

    let metadata: Value =
        serde_json::from_slice(&fs::read(temporary.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(metadata["update"], "wolff");
    assert_eq!(metadata["time_unit"], "cluster_flip");

    let series = fs::read_to_string(temporary.path().join("series.jsonl")).unwrap();
    assert_eq!(series.lines().count(), 4);
    for line in series.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let cluster_size = row["cluster_size"].as_u64().unwrap();
        assert!((1..=16).contains(&cluster_size));
    }

    let frames = fs::read_to_string(temporary.path().join("spins.jsonl")).unwrap();
    let sweeps: Vec<u64> = frames
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["sweep"]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert_eq!(sweeps, vec![4, 6]);
}

#[test]
fn same_seed_produces_identical_wolff_series() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    for output_directory in [first.path(), second.path()] {
        let status = command()
            .args([
                "--update",
                "wolff",
                "--l",
                "8",
                "--t-from",
                "2.3",
                "--t-to",
                "2.3",
                "--t-step",
                "0.1",
                "--discard",
                "5",
                "--measure",
                "10",
                "--seed",
                "99",
                "--out",
            ])
            .arg(output_directory)
            .status()
            .unwrap();
        assert!(status.success());
    }
    assert_eq!(
        fs::read(first.path().join("series.jsonl")).unwrap(),
        fs::read(second.path().join("series.jsonl")).unwrap()
    );
}

#[test]
fn rejects_invalid_parameters() {
    let temporary = tempfile::tempdir().unwrap();
    let output = command()
        .args([
            "--update",
            "metropolis",
            "--l",
            "1",
            "--t-from",
            "1.0",
            "--t-to",
            "1.0",
            "--t-step",
            "0.1",
            "--discard",
            "0",
            "--measure",
            "1",
            "--seed",
            "1",
            "--out",
        ])
        .arg(temporary.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("at least 2"));
}
