#!/usr/bin/env python3
"""Plot integrated autocorrelation time over temperature for both sizes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from errors import (  # noqa: E402
    integrated_autocorrelation_time,
    load_selected_series,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "tau.png"
EXACT_CRITICAL_TEMPERATURE = 2.26919
SIZES = (32, 64)


def main() -> None:
    series = load_selected_series()
    by_size: dict[int, list[tuple[float, float, int]]] = {
        side: [] for side in SIZES
    }
    for (side, temperature), values in sorted(series.items()):
        if side not in by_size:
            continue
        tau_int, cutoff_lag = integrated_autocorrelation_time(values)
        by_size[side].append((temperature, tau_int, cutoff_lag))

    for side in SIZES:
        if len(by_size[side]) != 27:
            raise ValueError(
                f"L={side}: expected 27 merged temperatures, found {len(by_size[side])}"
            )

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    colors = {32: "#d1495b", 64: "#2864a6"}
    for side in SIZES:
        temperatures = [row[0] for row in by_size[side]]
        autocorrelation_times = [row[1] for row in by_size[side]]
        axis.plot(
            temperatures,
            autocorrelation_times,
            "o-",
            color=colors[side],
            markersize=4.5,
            linewidth=1.6,
            label=f"L = {side}",
        )

    axis.axvline(
        EXACT_CRITICAL_TEMPERATURE,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label=r"Onsager $T_c=2.26919$",
    )
    axis.set_yscale("log")
    axis.set(
        xlabel="temperature T",
        ylabel=r"integrated autocorrelation time $\tau_{\mathrm{int}}$ (sweeps)",
        xlim=(1.45, 3.55),
    )
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=True)
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)

    for side in SIZES:
        peak = max(by_size[side], key=lambda row: row[1])
        print(
            f"L={side}: max tau_int={peak[1]:.6f} at T={peak[0]:.2f}, "
            f"cutoff_lag={peak[2]}"
        )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
