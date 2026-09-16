#!/usr/bin/env python3
"""Compare work-normalized autocorrelation times for Metropolis and Wolff."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from errors import integrated_autocorrelation_time  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "evidence" / "tau-compare.png"
SIDE = 64
EXACT_CRITICAL_TEMPERATURE = 2.26919
TEMPERATURES = np.round(np.arange(2.0, 2.6001, 0.05), 10)


@dataclass(frozen=True)
class RunSeries:
    magnetizations: list[np.ndarray]
    cluster_sizes: list[np.ndarray] | None


@dataclass(frozen=True)
class AutocorrelationPoint:
    temperature: float
    tau_native: float
    cutoff_lag: int
    mean_cluster_size: float | None
    tau_work: float


def temperature_key(value: float) -> float:
    return round(value, 10)


def load_run(directory: str, expected_update: str) -> RunSeries:
    run_file = ARTIFACTS / directory / "run.json"
    metadata = json.loads(run_file.read_text())
    if metadata.get("update") != expected_update:
        raise ValueError(f"{run_file}: expected update={expected_update}")
    if int(metadata.get("L", 0)) != SIDE:
        raise ValueError(f"{run_file}: expected L={SIDE}")
    expected_time_unit = "sweep" if expected_update == "metropolis" else "cluster_flip"
    if metadata.get("time_unit") != expected_time_unit:
        raise ValueError(f"{run_file}: expected time_unit={expected_time_unit}")

    run_temperatures = np.array(
        [temperature_key(float(value)) for value in metadata["t_grid"]]
    )
    if not np.array_equal(run_temperatures, TEMPERATURES):
        raise ValueError(f"{run_file}: expected the 13-point window temperature grid")

    measure = int(metadata["measure"])
    magnetizations = [np.full(measure, np.nan) for _ in TEMPERATURES]
    cluster_sizes = (
        [np.full(measure, np.nan) for _ in TEMPERATURES]
        if expected_update == "wolff"
        else None
    )
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
            if int(row["L"]) != SIDE:
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
            temperature_index = temperature_indices[temperature]
            values = magnetizations[temperature_index]
            if not math.isnan(values[step - 1]):
                raise ValueError(
                    f"{series_file}:{line_number}: duplicate step {step} "
                    f"at T={temperature:g}"
                )
            values[step - 1] = abs(float(row["M"]))
            if cluster_sizes is not None:
                cluster_size = int(row["cluster_size"])
                if not 1 <= cluster_size <= SIDE**2:
                    raise ValueError(
                        f"{series_file}:{line_number}: invalid cluster size "
                        f"{cluster_size}"
                    )
                cluster_sizes[temperature_index][step - 1] = cluster_size

    for temperature, values in zip(TEMPERATURES, magnetizations):
        missing = int(np.isnan(values).sum())
        if missing:
            raise RuntimeError(
                f"{series_file}: T={temperature:g} is missing "
                f"{missing:,} magnetization measurements"
            )
    if cluster_sizes is not None:
        for temperature, values in zip(TEMPERATURES, cluster_sizes):
            missing = int(np.isnan(values).sum())
            if missing:
                raise RuntimeError(
                    f"{series_file}: T={temperature:g} is missing "
                    f"{missing:,} cluster-size measurements"
                )
    return RunSeries(magnetizations, cluster_sizes)


def analyze_metropolis(run: RunSeries) -> list[AutocorrelationPoint]:
    points = []
    for temperature, magnetizations in zip(TEMPERATURES, run.magnetizations):
        tau_int, cutoff_lag = integrated_autocorrelation_time(magnetizations)
        points.append(
            AutocorrelationPoint(
                temperature=float(temperature),
                tau_native=tau_int,
                cutoff_lag=cutoff_lag,
                mean_cluster_size=None,
                tau_work=tau_int,
            )
        )
    return points


def analyze_wolff(run: RunSeries) -> list[AutocorrelationPoint]:
    if run.cluster_sizes is None:
        raise ValueError("Wolff analysis requires cluster sizes")
    points = []
    for temperature, magnetizations, cluster_sizes in zip(
        TEMPERATURES, run.magnetizations, run.cluster_sizes
    ):
        tau_moves, cutoff_lag = integrated_autocorrelation_time(magnetizations)
        mean_cluster_size = float(cluster_sizes.mean())
        tau_work = tau_moves * mean_cluster_size / SIDE**2
        points.append(
            AutocorrelationPoint(
                temperature=float(temperature),
                tau_native=tau_moves,
                cutoff_lag=cutoff_lag,
                mean_cluster_size=mean_cluster_size,
                tau_work=tau_work,
            )
        )
    return points


def draw(
    metropolis: list[AutocorrelationPoint], wolff: list[AutocorrelationPoint]
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    axis.plot(
        [point.temperature for point in metropolis],
        [point.tau_work for point in metropolis],
        "o-",
        color="#d1495b",
        markersize=5.0,
        linewidth=1.7,
        label="Metropolis (one sweep per row)",
    )
    axis.plot(
        [point.temperature for point in wolff],
        [point.tau_work for point in wolff],
        "s-",
        color="#2864a6",
        markersize=4.8,
        linewidth=1.7,
        label=r"Wolff ($\langle c\rangle/L^2$ sweeps per row)",
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
        title=r"Work-normalized autocorrelation time, $L=64$",
        xlabel="temperature T",
        ylabel=r"$\tau_{\mathrm{work}}$ ($L^2$ spin updates)",
        xlim=(1.98, 2.62),
    )
    axis.grid(alpha=0.23, which="both")
    axis.legend(frameon=True)
    figure.text(
        0.5,
        0.01,
        r"Wolff: $\tau_{\rm work}=\tau_{\rm moves}\langle c\rangle/L^2$; "
        "this compares spin-update counts, not elapsed time.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)


def point_at(
    points: list[AutocorrelationPoint], temperature: float
) -> AutocorrelationPoint:
    return next(point for point in points if math.isclose(point.temperature, temperature))


def main() -> None:
    metropolis = analyze_metropolis(load_run("window-l64", "metropolis"))
    wolff = analyze_wolff(load_run("wolff-l64", "wolff"))
    draw(metropolis, wolff)

    metropolis_critical = point_at(metropolis, 2.3)
    wolff_critical = point_at(wolff, 2.3)
    speedup = metropolis_critical.tau_work / wolff_critical.tau_work
    print("T=2.30 work comparison:")
    print(
        f"  Metropolis: tau_int={metropolis_critical.tau_native:.6f} sweeps, "
        f"cutoff_lag={metropolis_critical.cutoff_lag}"
    )
    print(
        f"  Wolff: tau_moves={wolff_critical.tau_native:.6f}, "
        f"mean_cluster_size={wolff_critical.mean_cluster_size:.6f}, "
        f"tau_work={wolff_critical.tau_work:.6f}, "
        f"cutoff_lag={wolff_critical.cutoff_lag}"
    )
    print(f"  Metropolis/Wolff work ratio={speedup:.3f}")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
