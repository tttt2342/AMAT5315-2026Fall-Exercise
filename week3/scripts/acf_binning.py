#!/usr/bin/env python3
"""Plot autocorrelation and block-binning diagnostics at L=64, T=2.3."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from errors import (  # noqa: E402
    integrated_autocorrelation_time,
    load_selected_series,
    normalized_autocorrelation,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "acf-binning.png"
SIDE = 64
TEMPERATURE = 2.3
BLOCK_LENGTHS = np.array(
    [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000]
)


def block_standard_error(values: np.ndarray, block_length: int) -> tuple[float, int]:
    sample_count = len(values)
    if sample_count % block_length != 0:
        raise ValueError(
            f"{sample_count} measurements are not divisible by block length "
            f"{block_length}"
        )
    block_count = sample_count // block_length
    if block_count < 2:
        raise ValueError("at least two blocks are required for a standard error")
    block_means = values.reshape(block_count, block_length).mean(axis=1)
    error = block_means.std(ddof=1) / math.sqrt(block_count)
    return float(error), block_count


def main() -> None:
    selected = load_selected_series()
    key = (SIDE, TEMPERATURE)
    if key not in selected:
        raise ValueError(f"missing Metropolis series for L={SIDE}, T={TEMPERATURE}")
    values = selected[key]

    autocorrelation = normalized_autocorrelation(values)
    tau_int, cutoff_lag = integrated_autocorrelation_time(values)
    maximum_lag = min(len(values) - 1, max(3000, cutoff_lag + 100))
    lags = np.arange(maximum_lag + 1)

    errors = []
    block_counts = []
    for block_length in BLOCK_LENGTHS:
        error, block_count = block_standard_error(values, int(block_length))
        errors.append(error)
        block_counts.append(block_count)
    errors = np.array(errors)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].plot(lags, autocorrelation[: maximum_lag + 1], color="#2864a6")
    axes[0].axhline(0.0, color="#555555", linewidth=1.0)
    axes[0].axvline(
        cutoff_lag,
        color="#d1495b",
        linestyle="--",
        linewidth=1.3,
        label=rf"cutoff $t={cutoff_lag}$",
    )
    axes[0].set(
        xlabel="lag t (sweeps)",
        ylabel=r"autocorrelation $\rho(t)$ of $|M|$",
        xlim=(0, maximum_lag),
        ylim=(-0.2, 1.02),
        title=rf"Autocorrelation: $\tau_{{\mathrm{{int}}}}={tau_int:.1f}$ sweeps",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=True)

    axes[1].plot(
        BLOCK_LENGTHS,
        errors,
        "o-",
        color="#d1495b",
        markersize=5,
        linewidth=1.5,
    )
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="block length (sweeps)",
        ylabel=r"standard error of mean $|M|$",
        title="Binning estimate",
    )
    axes[1].grid(alpha=0.22, which="both")
    axes[1].annotate(
        f"{block_counts[-1]} blocks remain",
        xy=(BLOCK_LENGTHS[-1], errors[-1]),
        xytext=(-110, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )

    figure.suptitle(r"Metropolis sampling diagnostics at $L=64$, $T=2.3$")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)

    print(f"tau_int={tau_int:.6f}, cutoff_lag={cutoff_lag}")
    for block_length, error, block_count in zip(
        BLOCK_LENGTHS, errors, block_counts
    ):
        print(
            f"block_length={block_length}: blocks={block_count}, "
            f"standard_error={error:.8f}"
        )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
