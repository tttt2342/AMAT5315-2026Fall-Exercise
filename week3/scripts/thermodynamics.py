#!/usr/bin/env python3
"""Plot magnetization and susceptibility from completed Metropolis ramps."""

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
EVIDENCE = ROOT / "evidence"
CRITICAL_TEMPERATURE = 2.0 / math.log(1.0 + math.sqrt(2.0))


@dataclass(frozen=True)
class TemperatureSummary:
    side: int
    temperature: float
    mean_abs_m: float
    mean_m2: float
    samples: int
    source: Path

    @property
    def susceptibility(self) -> float:
        variance = self.mean_m2 - self.mean_abs_m**2
        return self.side**2 * max(variance, 0.0) / self.temperature


def temperature_key(value: float) -> float:
    """Normalize harmless binary-float differences in metadata temperatures."""
    return round(value, 10)


def summarize_run(run_file: Path) -> tuple[int, int, dict[float, TemperatureSummary]]:
    metadata = json.loads(run_file.read_text())
    if metadata.get("update") != "metropolis":
        raise ValueError(f"{run_file}: expected a metropolis run")

    side = int(metadata["L"])
    measure = int(metadata["measure"])
    temperatures = [temperature_key(float(value)) for value in metadata["t_grid"]]
    moments = {temperature: [0, 0.0, 0.0] for temperature in temperatures}
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
            if temperature not in moments:
                raise ValueError(
                    f"{series_file}:{line_number}: unexpected T={temperature}"
                )
            magnetization = float(row["M"])
            accumulator = moments[temperature]
            accumulator[0] += 1
            accumulator[1] += abs(magnetization)
            accumulator[2] += magnetization * magnetization

    summaries: dict[float, TemperatureSummary] = {}
    for temperature, (count, sum_abs_m, sum_m2) in moments.items():
        if count != measure:
            raise RuntimeError(
                f"{series_file}: ramp is incomplete at T={temperature:g}: "
                f"found {count:,} of {measure:,} rows"
            )
        summaries[temperature] = TemperatureSummary(
            side=side,
            temperature=temperature,
            mean_abs_m=sum_abs_m / count,
            mean_m2=sum_m2 / count,
            samples=count,
            source=series_file,
        )
    return side, measure, summaries


def load_merged_metropolis_runs() -> dict[int, list[TemperatureSummary]]:
    """Use the run with the most measurements at each duplicated (L, T)."""
    selected: dict[tuple[int, float], tuple[int, TemperatureSummary]] = {}
    run_files = sorted(ARTIFACTS.glob("*/run.json"))
    if not run_files:
        raise FileNotFoundError(f"no runs found below {ARTIFACTS}")

    metropolis_runs = 0
    for run_file in run_files:
        metadata = json.loads(run_file.read_text())
        if metadata.get("update") != "metropolis":
            continue
        metropolis_runs += 1
        side, measure, summaries = summarize_run(run_file)
        for temperature, summary in summaries.items():
            key = (side, temperature)
            current = selected.get(key)
            if current is None or measure > current[0]:
                selected[key] = (measure, summary)
            elif measure == current[0]:
                raise ValueError(
                    f"ambiguous duplicate runs for L={side}, T={temperature:g} "
                    f"with {measure:,} measurements"
                )

    if metropolis_runs == 0:
        raise ValueError(f"no metropolis runs found below {ARTIFACTS}")

    merged: dict[int, list[TemperatureSummary]] = {}
    for (_, _), (_, summary) in selected.items():
        merged.setdefault(summary.side, []).append(summary)
    for summaries in merged.values():
        summaries.sort(key=lambda item: item.temperature)
    return merged


def onsager_magnetization(temperatures: np.ndarray) -> np.ndarray:
    values = np.zeros_like(temperatures)
    below = temperatures < CRITICAL_TEMPERATURE
    values[below] = (
        1.0 - np.sinh(2.0 / temperatures[below]) ** -4
    ) ** (1.0 / 8.0)
    return values


def plot_magnetization(summaries: list[TemperatureSummary]) -> Path:
    temperatures = np.array([item.temperature for item in summaries])
    measured = np.array([item.mean_abs_m for item in summaries])
    exact_temperatures = np.linspace(temperatures.min(), temperatures.max(), 800)

    fig, axis = plt.subplots(figsize=(8.0, 5.2))
    axis.plot(
        exact_temperatures,
        onsager_magnetization(exact_temperatures),
        color="#d1495b",
        linewidth=2.2,
        label="Onsager, infinite lattice",
    )
    axis.plot(
        temperatures,
        measured,
        color="#2864a6",
        marker="o",
        markersize=4.5,
        linewidth=1.7,
        label="measured, L = 64",
    )
    axis.axvline(
        CRITICAL_TEMPERATURE,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label=rf"$T_c={CRITICAL_TEMPERATURE:.5f}$",
    )
    axis.set(
        xlabel="temperature T",
        ylabel=r"mean $|M|$",
        xlim=(1.45, 3.55),
        ylim=(-0.02, 1.03),
    )
    axis.legend(frameon=True)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output = EVIDENCE / "magnetization.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_susceptibility(by_size: dict[int, list[TemperatureSummary]]) -> Path:
    fig, axis = plt.subplots(figsize=(8.0, 5.2))
    colors = {32: "#d1495b", 64: "#2864a6"}
    for side in (32, 64):
        summaries = by_size[side]
        temperatures = [item.temperature for item in summaries]
        susceptibility = [item.susceptibility for item in summaries]
        axis.plot(
            temperatures,
            susceptibility,
            color=colors[side],
            marker="o",
            markersize=4.5,
            linewidth=1.7,
            label=f"L = {side}",
        )
    axis.axvline(
        CRITICAL_TEMPERATURE,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label=rf"$T_c={CRITICAL_TEMPERATURE:.5f}$",
    )
    axis.set(
        xlabel="temperature T",
        ylabel=r"susceptibility $\chi(T)$",
        xlim=(1.95, 2.75),
    )
    axis.legend(frameon=True)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output = EVIDENCE / "susceptibility.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    by_size = load_merged_metropolis_runs()
    missing_sizes = {32, 64}.difference(by_size)
    if missing_sizes:
        raise ValueError(f"missing completed runs for L={sorted(missing_sizes)}")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    magnetization = plot_magnetization(by_size[64])
    susceptibility = plot_susceptibility(by_size)
    for side in (32, 64):
        peak = max(by_size[side], key=lambda item: item.susceptibility)
        print(
            f"L={side}: {len(by_size[side])} temperatures, "
            f"chi grid peak at T={peak.temperature:.2f} "
            f"(chi={peak.susceptibility:.6f})"
        )
    print(f"wrote {magnetization.relative_to(ROOT)}")
    print(f"wrote {susceptibility.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
