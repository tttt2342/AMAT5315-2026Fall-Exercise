#!/usr/bin/env python3
"""Plot measured RK4 growth, stability boundaries, and line spectra."""

from __future__ import annotations

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


def main() -> None:
    real, imaginary, growth = measured_rk4_growth()
    xx, yy = np.meshgrid(real, imaginary)
    z = xx + 1j * yy
    euler, midpoint, rk4 = stability_functions(z)
    if not np.allclose(growth, rk4, rtol=2.0e-13, atol=2.0e-13):
        raise RuntimeError("measured RK4 growth does not match its stability function")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
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

    colour_bar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.025)
    colour_bar.set_label(r"growth per step $|y_1|/|y_0|$")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
