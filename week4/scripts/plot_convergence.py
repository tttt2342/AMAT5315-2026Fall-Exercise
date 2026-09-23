#!/usr/bin/env python3
"""Plot measured convergence and select a step using Richardson estimation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "evidence" / "convergence.json"
OUTPUT = ROOT / "evidence" / "convergence.png"
ARTIFACTS = ROOT / "artifacts" / "convergence"
COARSE_DT = 0.02
FINE_DT = 0.01
ORDER = 4
TOLERANCE = 5.0e-6


def read_final_vorticity(dt: float) -> np.ndarray:
    path = ARTIFACTS / f"rk4-dt{dt:g}" / "fields.jsonl"
    final_frame = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            final_frame = json.loads(line)
    if final_frame is None or not math.isclose(final_frame["t"], 2.0, abs_tol=5.0e-7):
        raise ValueError(f"{path} does not contain a final field at t=2")
    return np.asarray(final_frame["omega"], dtype=float)


def main() -> None:
    with REPORT.open(encoding="utf-8") as stream:
        report = json.load(stream)
    step_sizes = np.asarray([float(run["dt"]) for run in report["runs"]])
    measured_errors = np.asarray(
        [float(run["relative_omega_error"]) for run in report["runs"]]
    )
    if np.any(step_sizes <= 0.0) or np.any(measured_errors <= 0.0):
        raise ValueError("log-log fitting requires positive steps and errors")

    slope, intercept = np.polyfit(np.log(step_sizes), np.log(measured_errors), 1)
    omega_coarse = read_final_vorticity(COARSE_DT)
    omega_fine = read_final_vorticity(FINE_DT)
    richardson_error = float(
        np.linalg.norm(omega_fine - omega_coarse)
        / ((2**ORDER - 1) * np.linalg.norm(omega_fine))
    )
    predicted_errors = richardson_error * (step_sizes / FINE_DT) ** ORDER
    eligible = step_sizes[predicted_errors < TOLERANCE]
    if eligible.size == 0:
        raise RuntimeError("no candidate step satisfies the requested error tolerance")
    chosen_dt = float(np.max(eligible))
    chosen_index = int(np.flatnonzero(np.isclose(step_sizes, chosen_dt))[0])
    chosen_predicted = float(predicted_errors[chosen_index])
    chosen_measured = float(measured_errors[chosen_index])

    sort_order = np.argsort(step_sizes)
    fit_steps = np.geomspace(float(np.min(step_sizes)), float(np.max(step_sizes)), 200)
    fit_errors = np.exp(intercept) * fit_steps**slope

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    axis.loglog(
        step_sizes,
        measured_errors,
        linestyle="none",
        marker="o",
        markersize=7,
        color="#3569a8",
        label="measured error",
        zorder=4,
    )
    axis.loglog(
        fit_steps,
        fit_errors,
        color="#d45b38",
        linewidth=1.7,
        label=rf"log--log fit, slope $={slope:.3f}$",
        zorder=2,
    )
    axis.loglog(
        step_sizes[sort_order],
        predicted_errors[sort_order],
        color="#777777",
        linestyle="--",
        marker="^",
        markersize=5.5,
        linewidth=1.2,
        label="fourth-order prediction",
        zorder=3,
    )
    axis.axhline(
        TOLERANCE,
        color="#555555",
        linestyle=":",
        linewidth=1.1,
        label=rf"tolerance $={TOLERANCE:.0e}$",
    )
    axis.loglog(
        [chosen_dt],
        [chosen_predicted],
        linestyle="none",
        marker="*",
        markersize=15,
        color="#2d8a57",
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=rf"chosen $\Delta t={chosen_dt:g}$",
        zorder=6,
    )
    axis.set_xlabel(r"time step $\Delta t$")
    axis.set_ylabel(r"relative vorticity error at $t=2$")
    axis.set_title("Random-flow temporal convergence")
    axis.grid(True, which="both", alpha=0.24, linewidth=0.6)
    axis.legend(frameon=False, loc="upper left")
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)

    print(f"Richardson error at dt={FINE_DT:g}: {richardson_error:.16e}")
    print("Fourth-order predictions:")
    for dt, error in sorted(zip(step_sizes, predicted_errors), reverse=True):
        print(f"  dt={dt:g}: {error:.16e}")
    print(f"chosen step: {chosen_dt:g}")
    print(f"chosen predicted error: {chosen_predicted:.16e}")
    print(f"chosen measured error: {chosen_measured:.16e}")
    print(f"log-log fit slope: {slope:.16e}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
