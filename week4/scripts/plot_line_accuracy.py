#!/usr/bin/env python3
"""Compare spatial and temporal errors after one pulse circuit."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "line-accuracy.png"


def run_library_comparison() -> np.ndarray:
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--bin", "line-accuracy"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.loadtxt(io.StringIO(result.stdout))


def run_convergence_study() -> np.ndarray:
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--bin", "line-accuracy", "--", "convergence"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.loadtxt(io.StringIO(result.stdout))


def main() -> None:
    data = run_library_comparison()
    x, exact, rk4_fourier, rk4_centred, euler_fourier = data.T
    profiles = [
        ("RK4, Fourier, h=0.02", rk4_fourier, "#2878b5", "-"),
        ("RK4, centred, h=0.02", rk4_centred, "#e68118", "--"),
        ("Euler, Fourier, h=0.005", euler_fourier, "#c83e4d", "-."),
    ]

    print("maximum errors at t = 2 pi")
    for label, profile, _, _ in profiles:
        error = np.max(np.abs(profile - exact))
        print(f"{label}: {error:.8e}")

    convergence = run_convergence_study()
    time_steps = convergence[:, 0]
    convergence_series = [
        ("Euler", convergence[:, 1], "#c83e4d", "o", 1.0),
        ("midpoint", convergence[:, 2], "#e68118", "s", 2.0),
        ("RK4", convergence[:, 3], "#2878b5", "^", 4.0),
        ("RK4, equal weights", convergence[:, 4], "0.35", "x", 2.0),
    ]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )
    fig, (ax, order_axis) = plt.subplots(
        1, 2, figsize=(14.0, 5.0), constrained_layout=True
    )
    ax.plot(x, exact, color="0.25", linewidth=3.0, alpha=0.5, label="exact")
    for label, profile, colour, style in profiles:
        ax.plot(x, profile, color=colour, linestyle=style, linewidth=1.6, label=label)

    all_profiles = np.concatenate([exact, rk4_fourier, rk4_centred, euler_fourier])
    padding = 0.08 * (all_profiles.max() - all_profiles.min())
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(all_profiles.min() - padding, all_profiles.max() + padding)
    ax.set_xticks(
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,2\pi)$")
    ax.set_title("(a) Periodic Gaussian after one lap")
    ax.grid(color="0.88", linewidth=0.7)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.text(
        0.98,
        0.04,
        r"$n=64,\ c=1,\ \nu=0.002,\ \sigma=0.25$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="0.35",
    )

    fit_steps = np.geomspace(time_steps.min(), time_steps.max(), 100)
    print("fitted log-log slopes at t = 1")
    for label, errors, colour, marker, expected_order in convergence_series:
        slope, intercept = np.polyfit(np.log(time_steps), np.log(errors), 1)
        if not np.isclose(slope, expected_order, rtol=0.15):
            raise RuntimeError(f"unexpected convergence slope for {label}: {slope:.4f}")
        fitted_errors = np.exp(intercept) * fit_steps**slope
        order_axis.loglog(
            fit_steps,
            fitted_errors,
            color=colour,
            linewidth=1.2,
            alpha=0.85,
        )
        order_axis.loglog(
            time_steps,
            errors,
            linestyle="none",
            marker=marker,
            markersize=6,
            color=colour,
            label=f"{label}, slope {slope:.2f}",
        )
        print(f"{label}: {slope:.4f}")

    order_axis.set_xlabel(r"time step $h$")
    order_axis.set_ylabel(r"$\max_j |u_j-u_{\mathrm{exact},j}|$")
    order_axis.set_title(r"(b) Temporal convergence at $t=1$")
    order_axis.set_xticks(
        [0.0025, 0.005, 0.01, 0.02],
        ["0.0025", "0.005", "0.01", "0.02"],
    )
    order_axis.minorticks_off()
    order_axis.grid(color="0.88", linewidth=0.7)
    order_axis.legend(loc="lower right", framealpha=0.95)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
