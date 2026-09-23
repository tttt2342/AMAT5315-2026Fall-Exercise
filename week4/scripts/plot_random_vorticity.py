#!/usr/bin/env python3
"""Plot selected random-flow vorticity snapshots with a shared colour scale."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "random"
OUTPUT = ROOT / "evidence" / "random.png"
REQUESTED_TIMES = (0.0, 2.0, 5.0, 10.0)


def read_selected_frames() -> tuple[int, list[dict]]:
    with (ARTIFACTS / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    n = int(metadata["n"])
    selected: dict[float, dict] = {}
    with (ARTIFACTS / "fields.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            time = float(frame["t"])
            for requested in REQUESTED_TIMES:
                if np.isclose(time, requested, atol=5.0e-7, rtol=0.0):
                    selected[requested] = frame
                    break
    missing = [time for time in REQUESTED_TIMES if time not in selected]
    if missing:
        raise ValueError(f"missing requested snapshot times: {missing}")
    frames = [selected[time] for time in REQUESTED_TIMES]
    if any(len(frame["omega"]) != n * n for frame in frames):
        raise ValueError("a selected vorticity field does not contain n*n values")
    return n, frames


def main() -> None:
    n, frames = read_selected_frames()
    vorticities = [
        np.asarray(frame["omega"], dtype=float).reshape(n, n) for frame in frames
    ]
    shared_limit = max(float(np.max(np.abs(omega))) for omega in vorticities)
    normalization = TwoSlopeNorm(vmin=-shared_limit, vcenter=0.0, vmax=shared_limit)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.35), constrained_layout=True)
    image = None
    for index, (axis, time, omega) in enumerate(
        zip(axes, REQUESTED_TIMES, vorticities)
    ):
        image = axis.imshow(
            omega,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap="RdBu_r",
            norm=normalization,
            interpolation="bilinear",
        )
        axis.set_aspect("equal")
        axis.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        axis.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        axis.set_xlabel(r"$x$")
        if index == 0:
            axis.set_ylabel(r"$y$")
        else:
            axis.tick_params(labelleft=False)
        axis.set_title(rf"$t={time:g}$")

    if image is None:
        raise RuntimeError("no vorticity image was drawn")
    colour_bar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.018)
    colour_bar.set_label(r"vorticity $\omega$")
    fig.suptitle("Random-flow vorticity")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"shared vorticity scale: [{-shared_limit:.8e}, {shared_limit:.8e}]")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
