#!/usr/bin/env python3
"""Estimate sampling errors and autocorrelation times for Metropolis runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BLOCK_COUNT = 50
WINDOW_FACTOR = 6.0


@dataclass(frozen=True)
class RunSpec:
    run_file: Path
    side: int
    measure: int
    temperatures: tuple[float, ...]


def temperature_key(value: float) -> float:
    return round(value, 10)


def discover_runs() -> list[RunSpec]:
    runs = []
    for run_file in sorted(ARTIFACTS.glob("*/run.json")):
        metadata = json.loads(run_file.read_text())
        if metadata.get("update") != "metropolis":
            continue
        runs.append(
            RunSpec(
                run_file=run_file,
                side=int(metadata["L"]),
                measure=int(metadata["measure"]),
                temperatures=tuple(
                    temperature_key(float(value)) for value in metadata["t_grid"]
                ),
            )
        )
    if not runs:
        raise FileNotFoundError(f"no metropolis runs found below {ARTIFACTS}")
    return runs


def load_selected_series() -> dict[tuple[int, float], np.ndarray]:
    """Load |M|, preferring the longest run at duplicated (L, T) points."""
    runs = discover_runs()
    selected: dict[tuple[int, float], RunSpec] = {}
    for run in runs:
        for temperature in run.temperatures:
            key = (run.side, temperature)
            current = selected.get(key)
            if current is None or run.measure > current.measure:
                selected[key] = run
            elif run.measure == current.measure:
                raise ValueError(
                    f"ambiguous duplicate runs for L={run.side}, T={temperature:g} "
                    f"with {run.measure:,} measurements"
                )

    series = {
        key: np.full(run.measure, np.nan)
        for key, run in selected.items()
    }
    selected_by_file: dict[Path, set[float]] = {}
    for (side, temperature), run in selected.items():
        if side != run.side:
            raise AssertionError("selected run has inconsistent lattice size")
        selected_by_file.setdefault(run.run_file, set()).add(temperature)

    for run in runs:
        selected_temperatures = selected_by_file.get(run.run_file)
        if not selected_temperatures:
            continue
        series_file = run.run_file.with_name("series.jsonl")
        with series_file.open() as rows:
            for line_number, line in enumerate(rows, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{series_file}:{line_number}: incomplete or invalid JSON row"
                    ) from error
                if int(row["L"]) != run.side:
                    raise ValueError(
                        f"{series_file}:{line_number}: L disagrees with run.json"
                    )
                temperature = temperature_key(float(row["T"]))
                if temperature not in selected_temperatures:
                    continue
                sweep = int(row["sweep"])
                if not 1 <= sweep <= run.measure:
                    raise ValueError(
                        f"{series_file}:{line_number}: sweep {sweep} is out of range"
                    )
                values = series[(run.side, temperature)]
                if not math.isnan(values[sweep - 1]):
                    raise ValueError(
                        f"{series_file}:{line_number}: duplicate sweep {sweep} "
                        f"at T={temperature:g}"
                    )
                values[sweep - 1] = abs(float(row["M"]))

    for (side, temperature), values in series.items():
        missing = int(np.isnan(values).sum())
        if missing:
            raise RuntimeError(
                f"L={side}, T={temperature:g}: ramp is incomplete; "
                f"missing {missing:,} measurements"
            )
    return series


def integrated_autocorrelation_time(values: np.ndarray) -> tuple[float, int]:
    """Return tau_int and its self-consistent six-tau cutoff lag."""
    centered = values - values.mean()
    sample_count = len(centered)
    variance = np.dot(centered, centered) / sample_count
    if variance <= 0.0:
        raise ValueError("autocorrelation is undefined for a constant series")

    # Zero padding prevents the FFT from wrapping the correlation around.
    fft_size = 1 << (2 * sample_count - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(transform * transform.conjugate(), n=fft_size)
    autocovariance = autocovariance[:sample_count]
    autocovariance /= np.arange(sample_count, 0, -1)
    autocorrelation = autocovariance / autocovariance[0]

    tau_int = 0.5
    for lag in range(1, sample_count):
        tau_int += autocorrelation[lag]
        if lag > WINDOW_FACTOR * tau_int:
            if tau_int <= 0.0:
                raise ValueError("autocorrelation window produced non-positive tau_int")
            return float(tau_int), lag
    raise ValueError("autocorrelation window did not close before the series ended")


def standard_errors(values: np.ndarray) -> tuple[float, float]:
    sample_count = len(values)
    if sample_count % BLOCK_COUNT != 0:
        raise ValueError(
            f"{sample_count} measurements cannot be split into {BLOCK_COUNT} equal blocks"
        )
    naive = values.std(ddof=1) / math.sqrt(sample_count)
    block_means = values.reshape(BLOCK_COUNT, sample_count // BLOCK_COUNT).mean(axis=1)
    blocked = block_means.std(ddof=1) / math.sqrt(BLOCK_COUNT)
    return float(naive), float(blocked)


def main() -> None:
    series = load_selected_series()
    print("L\tT\tmean_abs_M\tnaive_se\tblock50_se\tblock50_over_naive\ttau_int")
    for (side, temperature), values in sorted(series.items()):
        naive, blocked = standard_errors(values)
        tau_int, _ = integrated_autocorrelation_time(values)
        print(
            f"{side}\t{temperature:.2f}\t{values.mean():.6f}\t"
            f"{naive:.8f}\t{blocked:.8f}\t{blocked / naive:.3f}\t{tau_int:.3f}"
        )


if __name__ == "__main__":
    main()
