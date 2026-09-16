#!/usr/bin/env python3
"""Compare Metropolis and Wolff magnetization and fit the Wolff chi peaks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "evidence" / "magnetization-compare.png"
BLOCK_LENGTHS = (2000, 4000, 8000)
PLOT_BLOCK_LENGTH = 8000
REPLICATES = 500
BOOTSTRAP_SEED = 2026
EXACT_CRITICAL_TEMPERATURE = 2.26919
TEMPERATURES = np.round(np.arange(2.0, 2.6001, 0.05), 10)


@dataclass(frozen=True)
class QuadraticFit:
    quadratic: float
    linear: float
    constant: float
    center: float
    window_start: float
    window_end: float
    peak_temperature: float
    peak_susceptibility: float

    def evaluate(self, temperatures: np.ndarray) -> np.ndarray:
        shifted = temperatures - self.center
        return self.quadratic * shifted**2 + self.linear * shifted + self.constant


def temperature_key(value: float) -> float:
    return round(value, 10)


def load_run(directory: str, update: str, side: int) -> list[np.ndarray]:
    run_file = ARTIFACTS / directory / "run.json"
    metadata = json.loads(run_file.read_text())
    if metadata.get("update") != update or int(metadata.get("L", 0)) != side:
        raise ValueError(f"{run_file}: expected {update}, L={side}")

    run_temperatures = np.array(
        [temperature_key(float(value)) for value in metadata["t_grid"]]
    )
    if not np.array_equal(run_temperatures, TEMPERATURES):
        raise ValueError(f"{run_file}: expected the 13-point window temperature grid")

    measure = int(metadata["measure"])
    series = [np.full(measure, np.nan) for _ in TEMPERATURES]
    temperature_indices = {
        temperature_key(float(temperature)): index
        for index, temperature in enumerate(TEMPERATURES)
    }
    series_file = run_file.with_name("series.jsonl")
    with series_file.open() as rows:
        for line_number, line in enumerate(rows, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{series_file}:{line_number}: incomplete or invalid JSON row"
                ) from error
            if int(row["L"]) != side:
                raise ValueError(
                    f"{series_file}:{line_number}: L disagrees with run.json"
                )
            temperature = temperature_key(float(row["T"]))
            if temperature not in temperature_indices:
                raise ValueError(
                    f"{series_file}:{line_number}: unexpected T={temperature:g}"
                )
            step = int(row["sweep"])
            if not 1 <= step <= measure:
                raise ValueError(
                    f"{series_file}:{line_number}: step {step} is out of range"
                )
            values = series[temperature_indices[temperature]]
            if not math.isnan(values[step - 1]):
                raise ValueError(
                    f"{series_file}:{line_number}: duplicate step {step} "
                    f"at T={temperature:g}"
                )
            values[step - 1] = abs(float(row["M"]))

    for temperature, values in zip(TEMPERATURES, series):
        missing = int(np.isnan(values).sum())
        if missing:
            raise RuntimeError(
                f"{series_file}: T={temperature:g} is missing "
                f"{missing:,} of {measure:,} measurements"
            )
    return series


def block_draws(
    sample_count: int, block_length: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    block_count = sample_count // block_length
    if block_count < 2:
        raise ValueError(
            f"block length {block_length} leaves fewer than two complete blocks"
        )
    return (
        rng.integers(0, block_count, size=(REPLICATES, block_count)),
        block_count * block_length,
    )


def bootstrap_magnetization(
    values: np.ndarray, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    draws, used = block_draws(len(values), block_length, rng)
    block_means = values[:used].reshape(-1, block_length).mean(axis=1)
    return block_means[draws].mean(axis=1)


def susceptibility(side: int, temperature: float, values: np.ndarray) -> float:
    mean_abs = values.mean()
    return float(side**2 * ((values * values).mean() - mean_abs**2) / temperature)


def bootstrap_susceptibility(
    side: int,
    temperature: float,
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    draws, used = block_draws(len(values), block_length, rng)
    blocks = values[:used].reshape(-1, block_length)
    block_abs_means = blocks.mean(axis=1)
    block_m2_means = (blocks * blocks).mean(axis=1)
    mean_abs = block_abs_means[draws].mean(axis=1)
    mean_m2 = block_m2_means[draws].mean(axis=1)
    return side**2 * (mean_m2 - mean_abs**2) / temperature


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
    peak_susceptibility = constant - linear**2 / (4.0 * quadratic)
    return QuadraticFit(
        quadratic=float(quadratic),
        linear=float(linear),
        constant=float(constant),
        center=center,
        window_start=float(fit_temperatures[0]),
        window_end=float(fit_temperatures[-1]),
        peak_temperature=float(peak_temperature),
        peak_susceptibility=float(peak_susceptibility),
    )


def central_susceptibilities(
    wolff: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    return {
        side: np.array(
            [
                susceptibility(side, float(temperature), values)
                for temperature, values in zip(TEMPERATURES, wolff[side])
            ]
        )
        for side in (32, 64)
    }


def bootstrap_peak_temperatures(
    replicated_chi: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, int]]:
    critical_temperatures = []
    failures = {32: 0, 64: 0}
    for replicate in range(REPLICATES):
        fits = {}
        for side in (32, 64):
            fit = fit_five_points(TEMPERATURES, replicated_chi[side][replicate])
            if fit is None:
                failures[side] += 1
            else:
                fits[side] = fit
        if len(fits) == 2:
            critical_temperatures.append(
                2.0 * fits[64].peak_temperature - fits[32].peak_temperature
            )
    if len(critical_temperatures) < 2:
        raise ValueError("too few successful bootstrap peak-fit pairs")
    return np.array(critical_temperatures), failures


def draw(
    magnetizations: dict[str, np.ndarray],
    magnetization_errors: dict[str, np.ndarray],
    central_chi: dict[int, np.ndarray],
    chi_errors: dict[int, np.ndarray],
    central_fits: dict[int, QuadraticFit],
    critical_temperature: float,
    critical_temperature_error: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    colors = {"Metropolis": "#d1495b", "Wolff": "#2864a6"}
    markers = {"Metropolis": "o", "Wolff": "s"}

    for method in ("Metropolis", "Wolff"):
        axes[0].errorbar(
            TEMPERATURES,
            magnetizations[method],
            yerr=magnetization_errors[method],
            color=colors[method],
            marker=markers[method],
            markersize=4.2,
            linewidth=1.4,
            elinewidth=1.0,
            capsize=2.2,
            label=method,
        )
    axes[0].set(
        title=r"Sampler comparison, $L=64$",
        xlabel="temperature T",
        ylabel=r"mean $|M|$",
        xlim=(1.98, 2.62),
    )
    axes[0].legend(frameon=True)
    axes[0].grid(alpha=0.23)

    size_colors = {32: "#d1495b", 64: "#2864a6"}
    size_markers = {32: "o", 64: "s"}
    for side in (32, 64):
        color = size_colors[side]
        fit = central_fits[side]
        axes[1].errorbar(
            TEMPERATURES,
            central_chi[side],
            yerr=chi_errors[side],
            color=color,
            marker=size_markers[side],
            markersize=3.8,
            linewidth=1.2,
            elinewidth=0.9,
            capsize=2.0,
            label=rf"Wolff $L={side}$",
        )
        fit_grid = np.linspace(fit.window_start, fit.window_end, 180)
        axes[1].plot(
            fit_grid,
            fit.evaluate(fit_grid),
            color=color,
            linewidth=2.1,
            linestyle="--",
            label=rf"$L={side}$ five-point fit",
        )
        axes[1].plot(
            fit.peak_temperature,
            fit.peak_susceptibility,
            marker="*",
            color=color,
            markersize=10,
            zorder=5,
        )

    axes[1].axvspan(
        critical_temperature - critical_temperature_error,
        critical_temperature + critical_temperature_error,
        color="#2a9d8f",
        alpha=0.14,
        linewidth=0,
    )
    axes[1].axvline(
        critical_temperature,
        color="#2a9d8f",
        linestyle="-.",
        linewidth=1.6,
        label=rf"extrapolated $T_c={critical_temperature:.4f}$",
    )
    axes[1].axvline(
        EXACT_CRITICAL_TEMPERATURE,
        color="#444444",
        linestyle=":",
        linewidth=1.7,
        label=rf"exact $T_c={EXACT_CRITICAL_TEMPERATURE:.5f}$",
    )
    axes[1].set(
        title="Wolff susceptibility and peak fits",
        xlabel="temperature T",
        ylabel=r"susceptibility $\chi(T)$",
        xlim=(1.98, 2.62),
    )
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=True, fontsize=8.5, loc="upper right")
    axes[1].grid(alpha=0.23)
    axes[1].text(
        0.03,
        0.97,
        "\n".join(
            [
                rf"$T_{{peak}}(32)={central_fits[32].peak_temperature:.4f}$",
                rf"$T_{{peak}}(64)={central_fits[64].peak_temperature:.4f}$",
                rf"$\sigma_{{boot}}(T_c)={critical_temperature_error:.4f}$",
            ]
        ),
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
    )

    figure.suptitle("Ising window runs: Metropolis versus Wolff", fontsize=15)
    figure.text(
        0.5,
        0.01,
        f"Error bars and shaded band: {REPLICATES} block-bootstrap replicates, "
        f"block length {PLOT_BLOCK_LENGTH:,}; full-series central estimates.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)


def main() -> None:
    metropolis_64 = load_run("window-l64", "metropolis", 64)
    wolff = {
        32: load_run("wolff-l32", "wolff", 32),
        64: load_run("wolff-l64", "wolff", 64),
    }
    magnetizations = {
        "Metropolis": np.array([values.mean() for values in metropolis_64]),
        "Wolff": np.array([values.mean() for values in wolff[64]]),
    }
    central_chi = central_susceptibilities(wolff)
    central_fits = {
        side: fit_five_points(TEMPERATURES, central_chi[side])
        for side in (32, 64)
    }
    if any(fit is None for fit in central_fits.values()):
        raise ValueError("a central five-point susceptibility fit is invalid")
    fits = {side: fit for side, fit in central_fits.items() if fit is not None}
    critical_temperature = (
        2.0 * fits[64].peak_temperature - fits[32].peak_temperature
    )

    seed_sequences = np.random.SeedSequence(BOOTSTRAP_SEED).spawn(
        len(BLOCK_LENGTHS)
    )
    magnetization_errors_by_block: dict[int, dict[str, np.ndarray]] = {}
    chi_errors_by_block: dict[int, dict[int, np.ndarray]] = {}
    critical_samples_by_block: dict[int, np.ndarray] = {}
    failures_by_block: dict[int, dict[int, int]] = {}
    for block_length, seed_sequence in zip(BLOCK_LENGTHS, seed_sequences):
        rng = np.random.default_rng(seed_sequence)
        magnetization_replicates = {
            "Metropolis": np.vstack(
                [
                    bootstrap_magnetization(values, block_length, rng)
                    for values in metropolis_64
                ]
            ).T,
            "Wolff": np.vstack(
                [
                    bootstrap_magnetization(values, block_length, rng)
                    for values in wolff[64]
                ]
            ).T,
        }
        magnetization_errors_by_block[block_length] = {
            method: replicates.std(axis=0, ddof=1)
            for method, replicates in magnetization_replicates.items()
        }

        replicated_chi = {
            side: np.vstack(
                [
                    bootstrap_susceptibility(
                        side, float(temperature), values, block_length, rng
                    )
                    for temperature, values in zip(TEMPERATURES, wolff[side])
                ]
            ).T
            for side in (32, 64)
        }
        chi_errors_by_block[block_length] = {
            side: values.std(axis=0, ddof=1)
            for side, values in replicated_chi.items()
        }
        critical_samples, failures = bootstrap_peak_temperatures(replicated_chi)
        critical_samples_by_block[block_length] = critical_samples
        failures_by_block[block_length] = failures

    plot_critical_samples = critical_samples_by_block[PLOT_BLOCK_LENGTH]
    draw(
        magnetizations,
        magnetization_errors_by_block[PLOT_BLOCK_LENGTH],
        central_chi,
        chi_errors_by_block[PLOT_BLOCK_LENGTH],
        fits,
        critical_temperature,
        float(plot_critical_samples.std(ddof=1)),
    )

    target_index = int(np.flatnonzero(np.isclose(TEMPERATURES, 2.3))[0])
    print("T=2.30 sampler comparison (block-bootstrap standard errors):")
    print(
        f"  mean |M|: Metropolis={magnetizations['Metropolis'][target_index]:.6f}, "
        f"Wolff={magnetizations['Wolff'][target_index]:.6f}"
    )
    error_series: dict[str, list[float]] = {"Metropolis": [], "Wolff": []}
    comparison_distances = []
    for block_length in BLOCK_LENGTHS:
        errors = magnetization_errors_by_block[block_length]
        metropolis_error = float(errors["Metropolis"][target_index])
        wolff_error = float(errors["Wolff"][target_index])
        error_series["Metropolis"].append(metropolis_error)
        error_series["Wolff"].append(wolff_error)
        distance = abs(
            magnetizations["Metropolis"][target_index]
            - magnetizations["Wolff"][target_index]
        ) / math.hypot(metropolis_error, wolff_error)
        comparison_distances.append(distance)
        print(
            f"  block={block_length}: SE_metropolis={metropolis_error:.6f}, "
            f"SE_wolff={wolff_error:.6f}, d={distance:.3f}"
        )

    stable = {}
    for method, errors in error_series.items():
        values = np.array(errors)
        stable[method] = bool(values.max() - values.min() <= 0.1 * values.mean())
        print(f"  {method} error stability: {'stable' if stable[method] else 'sensitive'}")
    if max(comparison_distances) > 3.0:
        agreement = "discrepancy needing investigation"
    elif all(stable.values()):
        agreement = "agreement"
    else:
        agreement = "agreement provisional"
    print(f"  conclusion: {agreement}")

    relative_difference = (
        abs(critical_temperature - EXACT_CRITICAL_TEMPERATURE)
        / EXACT_CRITICAL_TEMPERATURE
    )
    print("Wolff susceptibility peaks:")
    for side in (32, 64):
        print(
            f"  L={side}: T_peak={fits[side].peak_temperature:.6f}, "
            f"chi_peak={fits[side].peak_susceptibility:.6f}, "
            f"fit_window=[{fits[side].window_start:.2f}, "
            f"{fits[side].window_end:.2f}]"
        )
    print(
        f"  extrapolated T_c={critical_temperature:.6f}; "
        f"distance from 2.26919={100.0 * relative_difference:.3f}%"
    )
    for block_length in BLOCK_LENGTHS:
        samples = critical_samples_by_block[block_length]
        failures = failures_by_block[block_length]
        print(
            f"  block={block_length}: bootstrap mean={samples.mean():.6f}, "
            f"SE={samples.std(ddof=1):.6f}, successful={len(samples)}/{REPLICATES}, "
            f"fit failures L32={failures[32]}, L64={failures[64]}"
        )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
