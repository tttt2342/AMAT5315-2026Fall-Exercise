#!/usr/bin/env python3
"""Plot the first 2,000 absolute-magnetization measurements at two temperatures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from errors import load_selected_series  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "trace.png"
SIDE = 64
TEMPERATURES = (2.3, 3.0)
TRACE_LENGTH = 2000


def main() -> None:
    selected = load_selected_series()
    traces: dict[float, np.ndarray] = {}
    for temperature in TEMPERATURES:
        key = (SIDE, temperature)
        if key not in selected:
            raise ValueError(f"missing Metropolis series for L={SIDE}, T={temperature}")
        values = selected[key]
        if len(values) < TRACE_LENGTH:
            raise ValueError(
                f"L={SIDE}, T={temperature}: found only {len(values)} measurements"
            )
        traces[temperature] = values[:TRACE_LENGTH]

    sweeps = np.arange(1, TRACE_LENGTH + 1)
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    colors = {2.3: "#2864a6", 3.0: "#d1495b"}
    for temperature in TEMPERATURES:
        axis.plot(
            sweeps,
            traces[temperature],
            color=colors[temperature],
            linewidth=1.0,
            alpha=0.9,
            label=rf"$T={temperature:.1f}$",
        )

    axis.set(
        xlabel="measurement sweep",
        ylabel=r"absolute magnetization $|M|$",
        xlim=(1, TRACE_LENGTH),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=True)
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)

    for temperature in TEMPERATURES:
        print(
            f"L={SIDE}, T={temperature:.1f}: first {TRACE_LENGTH} sweeps, "
            f"mean |M|={traces[temperature].mean():.6f}"
        )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
