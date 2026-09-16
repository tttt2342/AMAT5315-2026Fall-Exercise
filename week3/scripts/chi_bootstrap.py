#!/usr/bin/env python3
"""Block-bootstrap the finite-size susceptibility peak fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from errors import load_selected_series  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "chi-bootstrap.png"
BLOCK_LENGTHS = (2000, 4000, 8000)
REPLICATES = 500
BOOTSTRAP_SEED = 2026
EXACT_CRITICAL_TEMPERATURE = 2.26919
SIZES = (32, 64)


@dataclass(frozen=True)
class QuadraticFit:
    quadratic: float
    linear: float
    constant: float
    center: float
    window_start: float
    window_end: float
    peak_temperature: float

    def evaluate(self, temperatures: np.ndarray) -> np.ndarray:
        shifted = temperatures - self.center
        return self.quadratic * shifted**2 + self.linear * shifted + self.constant


@dataclass(frozen=True)
class BootstrapResult:
    block_length: int
    block_count: int
    discarded: int
    fits: dict[int, list[QuadraticFit]]
    failed_by_size: dict[int, int]
    failed_pairs: int
    critical_temperatures: np.ndarray

    @property
    def critical_temperature_error(self) -> float:
        return float(self.critical_temperatures.std(ddof=1))


def fit_five_points(
    temperatures: np.ndarray, susceptibilities: np.ndarray
) -> QuadraticFit | None:
    maximum_index = int(np.argmax(susceptibilities))
    if maximum_index < 2 or maximum_index + 2 >= len(temperatures):
        return None

    window = slice(maximum_index - 2, maximum_index + 3)
    fit_temperatures = temperatures[window]
    fit_values = susceptibilities[window]
    center = float(temperatures[maximum_index])
    quadratic, linear, constant = np.polyfit(
        fit_temperatures - center, fit_values, 2
    )
    if quadratic >= 0.0:
        return None

    peak_temperature = center - linear / (2.0 * quadratic)
    if not fit_temperatures[0] <= peak_temperature <= fit_temperatures[-1]:
        return None
    return QuadraticFit(
        quadratic=float(quadratic),
        linear=float(linear),
        constant=float(constant),
        center=center,
        window_start=float(fit_temperatures[0]),
        window_end=float(fit_temperatures[-1]),
        peak_temperature=float(peak_temperature),
    )


def window_series() -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
    selected = load_selected_series()
    temperatures = np.array(
        sorted(
            temperature
            for side, temperature in selected
            if side == 32 and 2.0 <= temperature <= 2.6
        )
    )
    if len(temperatures) != 13 or not np.allclose(np.diff(temperatures), 0.05):
        raise ValueError("expected the 13-point window grid from T=2.0 to T=2.6")

    by_size: dict[int, list[np.ndarray]] = {}
    for side in SIZES:
        values = []
        for temperature in temperatures:
            key = (side, round(float(temperature), 10))
            if key not in selected:
                raise ValueError(f"missing window series for L={side}, T={temperature:g}")
            series = selected[key]
            if len(series) != 100_000:
                raise ValueError(
                    f"L={side}, T={temperature:g}: expected 100,000 measurements"
                )
            values.append(series)
        by_size[side] = values
    return temperatures, by_size


def susceptibility(side: int, temperature: float, values: np.ndarray) -> float:
    mean_abs = values.mean()
    return float(side**2 * ((values * values).mean() - mean_abs**2) / temperature)


def central_susceptibilities(
    temperatures: np.ndarray, by_size: dict[int, list[np.ndarray]]
) -> dict[int, np.ndarray]:
    return {
        side: np.array(
            [
                susceptibility(side, float(temperature), values)
                for temperature, values in zip(temperatures, by_size[side])
            ]
        )
        for side in SIZES
    }


def bootstrap(
    temperatures: np.ndarray,
    by_size: dict[int, list[np.ndarray]],
    block_length: int,
    rng: np.random.Generator,
) -> BootstrapResult:
    sample_count = len(by_size[32][0])
    block_count = sample_count // block_length
    used = block_count * block_length
    discarded = sample_count - used
    replicated_chi: dict[int, np.ndarray] = {}

    for side in SIZES:
        replicated_chi[side] = np.empty((REPLICATES, len(temperatures)))
        for temperature_index, (temperature, values) in enumerate(
            zip(temperatures, by_size[side])
        ):
            blocks = values[:used].reshape(block_count, block_length)
            block_abs_means = blocks.mean(axis=1)
            block_m2_means = (blocks * blocks).mean(axis=1)
            draws = rng.integers(0, block_count, size=(REPLICATES, block_count))
            mean_abs = block_abs_means[draws].mean(axis=1)
            mean_m2 = block_m2_means[draws].mean(axis=1)
            replicated_chi[side][:, temperature_index] = (
                side**2 * (mean_m2 - mean_abs**2) / temperature
            )

    fits: dict[int, list[QuadraticFit]] = {side: [] for side in SIZES}
    failed_by_size = {side: 0 for side in SIZES}
    critical_temperatures = []
    failed_pairs = 0
    for replicate in range(REPLICATES):
        pair: dict[int, QuadraticFit] = {}
        for side in SIZES:
            fit = fit_five_points(temperatures, replicated_chi[side][replicate])
            if fit is None:
                failed_by_size[side] += 1
            else:
                fits[side].append(fit)
                pair[side] = fit
        if len(pair) == len(SIZES):
            critical_temperatures.append(
                2.0 * pair[64].peak_temperature - pair[32].peak_temperature
            )
        else:
            failed_pairs += 1

    if len(critical_temperatures) < 2:
        raise ValueError(f"block length {block_length}: too few successful peak pairs")
    return BootstrapResult(
        block_length=block_length,
        block_count=block_count,
        discarded=discarded,
        fits=fits,
        failed_by_size=failed_by_size,
        failed_pairs=failed_pairs,
        critical_temperatures=np.array(critical_temperatures),
    )


def draw_envelope(
    axis: plt.Axes,
    fits: list[QuadraticFit],
    color: str,
    label: str,
) -> None:
    grid = np.linspace(2.15, 2.50, 351)
    curves = np.full((len(fits), len(grid)), np.nan)
    for index, fit in enumerate(fits):
        inside = (grid >= fit.window_start) & (grid <= fit.window_end)
        curves[index, inside] = fit.evaluate(grid[inside])
    covered = np.any(np.isfinite(curves), axis=0)
    lower = np.nanmin(curves[:, covered], axis=0)
    upper = np.nanmax(curves[:, covered], axis=0)
    axis.fill_between(
        grid[covered], lower, upper, color=color, alpha=0.20, linewidth=0, label=label
    )


def draw(
    temperatures: np.ndarray,
    central: dict[int, np.ndarray],
    central_fits: dict[int, QuadraticFit],
    results: list[BootstrapResult],
) -> None:
    colors = {32: "#d1495b", 64: "#2864a6"}
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)

    for panel, result in zip(axes, results):
        for side in SIZES:
            color = colors[side]
            draw_envelope(
                panel,
                result.fits[side],
                color,
                f"L={side} bootstrap envelope",
            )
            panel.plot(
                temperatures,
                central[side],
                "o-",
                color=color,
                markersize=3.5,
                linewidth=1.2,
                label=f"L={side} susceptibility",
            )
            fit = central_fits[side]
            fit_grid = np.linspace(fit.window_start, fit.window_end, 160)
            panel.plot(
                fit_grid,
                fit.evaluate(fit_grid),
                color=color,
                linewidth=2.2,
                label=f"L={side} five-point fit",
            )
        panel.axvline(
            EXACT_CRITICAL_TEMPERATURE,
            color="#444444",
            linestyle="--",
            linewidth=1.4,
            label=r"Onsager $T_c=2.26919$",
        )
        panel.set_title(
            f"block {result.block_length:,}: "
            rf"$\sigma(T_c)={result.critical_temperature_error:.4f}$"
        )
        panel.set_xlim(2.15, 2.50)
        panel.set_ylim(0, 75)
        panel.set_xlabel("temperature T")
        panel.grid(alpha=0.22)

    axes[0].set_ylabel(r"susceptibility $\chi(T)$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        "Metropolis susceptibility: five-point fits and 500-replicate envelopes",
        y=1.075,
        fontsize=15,
    )
    figure.text(
        0.5,
        0.01,
        "Each temperature and size is block-resampled independently; "
        "the 8,000-sweep analysis uses 12 blocks and discards the final 4,000 sweeps.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.92))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    temperatures, by_size = window_series()
    central = central_susceptibilities(temperatures, by_size)
    central_fits: dict[int, QuadraticFit] = {}
    for side in SIZES:
        fit = fit_five_points(temperatures, central[side])
        if fit is None:
            raise ValueError(f"L={side}: central susceptibility fit is invalid")
        central_fits[side] = fit

    seed_sequences = np.random.SeedSequence(BOOTSTRAP_SEED).spawn(len(BLOCK_LENGTHS))
    results = [
        bootstrap(
            temperatures,
            by_size,
            block_length,
            np.random.default_rng(seed_sequence),
        )
        for block_length, seed_sequence in zip(BLOCK_LENGTHS, seed_sequences)
    ]

    draw(temperatures, central, central_fits, results)
    errors = np.array([result.critical_temperature_error for result in results])
    stable = errors.max() - errors.min() <= 0.1 * errors.mean()
    central_tc = (
        2.0 * central_fits[64].peak_temperature
        - central_fits[32].peak_temperature
    )
    print(f"central T_c = {central_tc:.6f}")
    for result in results:
        print(
            f"block_length={result.block_length}: blocks={result.block_count}, "
            f"discarded={result.discarded}, "
            f"T_c mean={result.critical_temperatures.mean():.6f}, "
            f"SE={result.critical_temperature_error:.6f}, "
            f"failed L32={result.failed_by_size[32]}, "
            f"failed L64={result.failed_by_size[64]}, "
            f"failed pairs={result.failed_pairs}"
        )
    print(f"sampling error: {'stable' if stable else 'unresolved'}")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
