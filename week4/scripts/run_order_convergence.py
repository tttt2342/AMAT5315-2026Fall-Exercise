#!/usr/bin/env python3
"""Run the Taylor--Green order and random-flow convergence studies."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORDER_ARTIFACTS = ROOT / "artifacts" / "order"
CONVERGENCE_ARTIFACTS = ROOT / "artifacts" / "convergence"
ORDER_FIGURE = ROOT / "evidence" / "order.png"
CONVERGENCE_JSON = ROOT / "evidence" / "convergence.json"
FIELD = ROOT / "target" / "release" / "field"
FLUID = ROOT / "target" / "release" / "fluid"


def build_binaries() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--bins"],
        cwd=ROOT,
        check=True,
    )


def generate_field(arguments: list[str]) -> bytes:
    completed = subprocess.run(
        [str(FIELD), *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def run_fluid(field_bytes: bytes, nu: float, dt: float, output_directory: Path) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(FLUID),
        "--method",
        "rk4",
        "--nu",
        str(nu),
        "--dt",
        str(dt),
        "--t-end",
        "2",
        "--every",
        "2",
        "--out",
        str(output_directory),
    ]
    print(f"running {output_directory.name} ...", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=field_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (output_directory / "diagnostics.tsv").write_bytes(completed.stdout)
    if completed.stderr:
        (output_directory / "stderr.txt").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{output_directory.name} failed:\n{completed.stderr.decode('utf-8')}"
        )
    elapsed = time.perf_counter() - started
    with (output_directory / "fields.jsonl").open(encoding="utf-8") as stream:
        frames = [json.loads(line) for line in stream]
    if len(frames) != 2 or not math.isclose(frames[-1]["t"], 2.0, abs_tol=5.0e-7):
        raise RuntimeError(f"{output_directory.name} did not retain t=0 and t=2 fields")
    print(f"  reached t=2 ({elapsed:.1f} s)", flush=True)
    return frames[-1]


def relative_velocity_error(frame: dict, exact_field: dict) -> float:
    u = np.asarray(frame["u"], dtype=float)
    v = np.asarray(frame["v"], dtype=float)
    exact_u = np.asarray(exact_field["u"], dtype=float)
    exact_v = np.asarray(exact_field["v"], dtype=float)
    numerator = np.linalg.norm(np.concatenate((u - exact_u, v - exact_v)))
    denominator = np.linalg.norm(np.concatenate((exact_u, exact_v)))
    return float(numerator / denominator)


def relative_vorticity_error(frame: dict, reference: dict) -> float:
    omega = np.asarray(frame["omega"], dtype=float)
    reference_omega = np.asarray(reference["omega"], dtype=float)
    return float(np.linalg.norm(omega - reference_omega) / np.linalg.norm(reference_omega))


def fit_log_log(step_sizes: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(step_sizes), np.log(errors), 1)
    return float(slope), float(intercept)


def draw_order(step_sizes: np.ndarray, errors: np.ndarray, slope: float, intercept: float) -> None:
    order = np.argsort(step_sizes)
    ordered_steps = step_sizes[order]
    ordered_errors = errors[order]
    fit_steps = np.geomspace(float(ordered_steps[0]), float(ordered_steps[-1]), 200)
    fit_errors = np.exp(intercept) * fit_steps**slope

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9.5,
        }
    )
    fig, axis = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    axis.loglog(
        ordered_steps,
        ordered_errors,
        linestyle="none",
        marker="o",
        markersize=7,
        color="#3569a8",
        label="RK4 error",
    )
    axis.loglog(
        fit_steps,
        fit_errors,
        color="#d45b38",
        linewidth=1.7,
        label=rf"log--log fit, slope $={slope:.3f}$",
    )
    axis.set_xlabel(r"time step $\Delta t$")
    axis.set_ylabel("relative velocity error")
    axis.set_title(r"Taylor--Green temporal convergence at $t=2$")
    axis.grid(True, which="both", alpha=0.24, linewidth=0.6)
    axis.legend(frameon=False)
    ORDER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ORDER_FIGURE, dpi=220)
    plt.close(fig)
    print(f"saved {ORDER_FIGURE}")


def run_taylor_green() -> None:
    ORDER_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    initial = generate_field(["taylor-green", "--n", "8"])
    exact_bytes = generate_field(
        ["taylor-green", "--n", "8", "--t", "2", "--nu", "0.5"]
    )
    exact = json.loads(exact_bytes)
    (ORDER_ARTIFACTS / "initial.json").write_bytes(initial)
    (ORDER_ARTIFACTS / "exact-t2.json").write_bytes(exact_bytes)

    step_sizes = np.asarray([0.4, 0.25, 0.2])
    errors = []
    for dt in step_sizes:
        output_directory = ORDER_ARTIFACTS / f"rk4-dt{dt:g}"
        final_frame = run_fluid(initial, 0.5, float(dt), output_directory)
        errors.append(relative_velocity_error(final_frame, exact))
    error_array = np.asarray(errors)
    slope, intercept = fit_log_log(step_sizes, error_array)
    summary = {
        "case": "taylor-green",
        "n": 8,
        "nu": 0.5,
        "t_end": 2.0,
        "method": "rk4",
        "runs": [
            {"dt": float(dt), "relative_velocity_error": float(error)}
            for dt, error in zip(step_sizes, error_array)
        ],
        "log_log_fit": {"slope": slope, "intercept": intercept},
    }
    (ORDER_ARTIFACTS / "errors.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Taylor--Green relative velocity errors:")
    for dt, error in zip(step_sizes, error_array):
        print(f"  dt={dt:g}: {error:.16e}")
    print(f"  slope: {slope:.16e}")
    draw_order(step_sizes, error_array, slope, intercept)


def run_random_convergence() -> None:
    CONVERGENCE_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    initial = generate_field(
        [
            "random",
            "--n",
            "128",
            "--seed",
            "2026",
            "--k-min",
            "2",
            "--k-max",
            "6",
        ]
    )
    (CONVERGENCE_ARTIFACTS / "initial.json").write_bytes(initial)
    reference_dt = 0.0025
    reference_directory = CONVERGENCE_ARTIFACTS / f"rk4-dt{reference_dt:g}"
    reference = run_fluid(initial, 0.004, reference_dt, reference_directory)

    step_sizes = np.asarray([0.02, 0.0125, 0.01])
    errors = []
    runs = []
    for dt in step_sizes:
        output_directory = CONVERGENCE_ARTIFACTS / f"rk4-dt{dt:g}"
        final_frame = run_fluid(initial, 0.004, float(dt), output_directory)
        error = relative_vorticity_error(final_frame, reference)
        errors.append(error)
        runs.append(
            {
                "dt": float(dt),
                "relative_omega_error": error,
                "artifact": str(output_directory.relative_to(ROOT)),
                "is_reference": False,
            }
        )
    error_array = np.asarray(errors)
    slope, intercept = fit_log_log(step_sizes, error_array)
    report = {
        "case": "random",
        "n": 128,
        "nu": 0.004,
        "seed": 2026,
        "wavenumbers": [2, 6],
        "method": "rk4",
        "t_end": 2.0,
        "reference": {
            "dt": reference_dt,
            "relative_omega_error": 0.0,
            "artifact": str(reference_directory.relative_to(ROOT)),
            "is_reference": True,
        },
        "runs": runs,
        "log_log_fit": {
            "slope": slope,
            "intercept": intercept,
            "reference_dt": reference_dt,
        },
    }
    CONVERGENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONVERGENCE_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Random-flow relative omega errors:")
    for dt, error in zip(step_sizes, error_array):
        print(f"  dt={dt:g}: {error:.16e}")
    print(f"  dt={reference_dt:g} (reference): {0.0:.16e}")
    print(f"  slope: {slope:.16e}")
    print(f"saved {CONVERGENCE_JSON}")


def main() -> None:
    build_binaries()
    run_taylor_green()
    run_random_convergence()


if __name__ == "__main__":
    main()
