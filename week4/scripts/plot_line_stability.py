#!/usr/bin/env python3
"""Plot measured RK4 growth, stability boundaries, and line spectra."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "line-stability.png"

NX = 501
NY = 501
XMIN, XMAX = -3.2, 0.8
YMIN, YMAX = -3.2, 3.2


def measured_rk4_growth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "measure-rk4",
        "--",
        str(NX),
        str(NY),
        str(XMIN),
        str(XMAX),
        str(YMIN),
        str(YMAX),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    growth = np.fromstring(result.stdout, sep=" ").reshape(NY, NX)
    real = np.linspace(XMIN, XMAX, NX)
    imaginary = np.linspace(YMIN, YMAX, NY)
    return real, imaginary, growth


def stability_functions(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    euler = 1.0 + z
    midpoint = 1.0 + z + z**2 / 2.0
    rk4 = midpoint + z**3 / 6.0 + z**4 / 24.0
    return np.abs(euler), np.abs(midpoint), np.abs(rk4)


def line_modes(step_size: float) -> tuple[np.ndarray, np.ndarray]:
    n = 64
    c = 1.0
    nu = 0.05
    wave_numbers = np.arange(-n // 2, n // 2)
    real = -nu * wave_numbers.astype(float) ** 2 * step_size
    imaginary = -c * wave_numbers.astype(float) * step_size
    imaginary[wave_numbers == -n // 2] = 0.0
    return real, imaginary


def pulse_history(step_size: float) -> tuple[np.ndarray, np.ndarray]:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "line-pulse",
            "--",
            str(step_size),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = np.loadtxt(io.StringIO(result.stdout))
    times = data[:, 0]
    states = data[:, 1:]
    if states.shape[1] != 64 or not np.isclose(times[-1], 6.0):
        raise RuntimeError("line-pulse returned an unexpected grid or final time")
    return times, states


def main() -> None:
    real, imaginary, growth = measured_rk4_growth()
    xx, yy = np.meshgrid(real, imaginary)
    z = xx + 1j * yy
    euler, midpoint, rk4 = stability_functions(z)
    if not np.allclose(growth, rk4, rtol=2.0e-13, atol=2.0e-13):
        raise RuntimeError("measured RK4 growth does not match its stability function")
    stable_times, stable_states = pulse_history(0.045)
    unstable_times, unstable_states = pulse_history(0.056)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )
    fig = plt.figure(figsize=(15.8, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.08, 0.055, 1.0, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    colour_axis = fig.add_subplot(grid[0, 1])
    stable_axis = fig.add_subplot(grid[0, 2])
    unstable_axis = fig.add_subplot(grid[0, 3])
    image = ax.pcolormesh(
        real,
        imaginary,
        growth,
        norm=LogNorm(vmin=3.0e-2, vmax=3.0e1),
        cmap="viridis",
        shading="auto",
        rasterized=True,
    )

    boundary_specs = [
        (euler, "white", ":", "Euler: |R(z)| = 1"),
        (midpoint, "#f4a742", "--", "midpoint: |R(z)| = 1"),
        (rk4, "black", "-", "RK4: |R(z)| = 1"),
    ]
    handles = []
    for values, colour, style, label in boundary_specs:
        contour = ax.contour(
            xx,
            yy,
            values,
            levels=[1.0],
            colors=[colour],
            linewidths=1.8,
            linestyles=[style],
            zorder=3,
        )
        for collection in contour.get_children():
            collection.set_path_effects(
                [path_effects.Stroke(linewidth=3.0, foreground="black"), path_effects.Normal()]
                if colour == "white"
                else []
            )
        handles.append(plt.Line2D([], [], color=colour, linestyle=style, linewidth=1.8, label=label))

    marker_specs = [
        (0.045, "o", "#35c4d8"),
        (0.056, "^", "#e94f9d"),
    ]
    for step_size, marker, colour in marker_specs:
        mode_real, mode_imaginary = line_modes(step_size)
        handle = ax.scatter(
            mode_real,
            mode_imaginary,
            s=24,
            marker=marker,
            facecolors=colour,
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
            label=rf"line modes, $h={step_size:.3f}$",
        )
        handles.append(handle)

    ax.axhline(0.0, color="0.35", linewidth=0.6, zorder=2)
    ax.axvline(0.0, color="0.35", linewidth=0.6, zorder=2)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\operatorname{Re}(z)$")
    ax.set_ylabel(r"$\operatorname{Im}(z)$")
    ax.set_title(r"Measured RK4 growth for $y' = \lambda y$, $h=1$")
    ax.legend(handles=handles, loc="upper right", framealpha=0.94)

    colour_bar = fig.colorbar(image, cax=colour_axis)
    colour_bar.set_label(r"growth per step $|y_1|/|y_0|$")

    pulse_panels = [
        (stable_axis, stable_times, stable_states, 0.045),
        (unstable_axis, unstable_times, unstable_states, 0.056),
    ]
    for pulse_axis, times, states, step_size in pulse_panels:
        pulse_axis.imshow(
            states,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
            extent=(0.0, 2.0 * np.pi, times[-1], times[0]),
        )
        pulse_axis.set_xlim(0.0, 2.0 * np.pi)
        pulse_axis.set_ylim(6.0, 0.0)
        pulse_axis.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        pulse_axis.set_xlabel(r"$x$")
        pulse_axis.set_title(rf"RK4 pulse, $h={step_size:.3f}$")
    stable_axis.set_ylabel(r"$t$")
    unstable_axis.set_ylabel(r"$t$")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
