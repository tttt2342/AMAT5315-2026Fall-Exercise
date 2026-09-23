#!/usr/bin/env python3
"""Compare and plot the saved Taylor-Green run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "taylor-green"
OUTPUT = ROOT / "evidence" / "taylor-green.png"


def read_frames(path: Path) -> tuple[dict, dict]:
    first = None
    last = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            if first is None:
                first = frame
            last = frame
    if first is None or last is None:
        raise ValueError(f"no frames found in {path}")
    return first, last


def relative_velocity_error(frame: dict, exact: dict) -> float:
    numerical_u = np.asarray(frame["u"], dtype=float)
    numerical_v = np.asarray(frame["v"], dtype=float)
    exact_u = np.asarray(exact["u"], dtype=float)
    exact_v = np.asarray(exact["v"], dtype=float)
    difference_norm = np.sqrt(
        np.sum((numerical_u - exact_u) ** 2 + (numerical_v - exact_v) ** 2)
    )
    exact_norm = np.sqrt(np.sum(exact_u**2 + exact_v**2))
    return float(difference_norm / exact_norm)


def main() -> None:
    first, last = read_frames(ARTIFACTS / "fields.jsonl")
    with (ARTIFACTS / "exact-t1.json").open(encoding="utf-8") as stream:
        exact = json.load(stream)

    n = int(exact["n"])
    if len(last["u"]) != n * n or len(last["v"]) != n * n:
        raise ValueError("last frame dimensions do not match the exact field")
    if not np.isclose(first["t"], 0.0) or not np.isclose(last["t"], 1.0):
        raise ValueError("expected saved frames at t=0 and t=1")

    error = relative_velocity_error(last, exact)
    print(f"relative velocity error: {error:.8e}")
    if error >= 1.0e-5:
        raise RuntimeError("relative velocity error does not meet the 1e-5 requirement")

    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    frames = [first, last]
    omega_arrays = [np.asarray(frame["omega"], dtype=float).reshape(n, n) for frame in frames]
    shared_limit = max(float(np.max(np.abs(omega))) for omega in omega_arrays)
    normalization = Normalize(vmin=-shared_limit, vmax=shared_limit)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), constrained_layout=True)
    image = None
    stride = 4
    for axis, frame, omega in zip(axes, frames, omega_arrays):
        image = axis.imshow(
            omega,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap="RdBu_r",
            norm=normalization,
            interpolation="bilinear",
        )
        u = np.asarray(frame["u"], dtype=float).reshape(n, n)
        v = np.asarray(frame["v"], dtype=float).reshape(n, n)
        axis.quiver(
            xx[::stride, ::stride],
            yy[::stride, ::stride],
            u[::stride, ::stride],
            v[::stride, ::stride],
            color="black",
            angles="xy",
            scale_units="xy",
            scale=3.0,
            width=0.004,
            headwidth=3.5,
            alpha=0.78,
        )
        axis.set_xlim(0.0, 2.0 * np.pi)
        axis.set_ylim(0.0, 2.0 * np.pi)
        axis.set_aspect("equal")
        axis.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        axis.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(
            rf"$t={frame['t']:.0f}$, $\max|\omega|={np.max(np.abs(omega)):.3f}$"
        )

    if image is None:
        raise RuntimeError("no image was drawn")
    colour_bar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.025)
    colour_bar.set_label(r"vorticity $\omega$")
    fig.suptitle(r"Taylor--Green decay, $\nu=0.1$")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
