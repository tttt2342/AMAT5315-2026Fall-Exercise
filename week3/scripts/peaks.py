#!/usr/bin/env python3
"""Fit susceptibility peaks and extrapolate the critical temperature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermodynamics import (
    CRITICAL_TEMPERATURE,
    EVIDENCE,
    TemperatureSummary,
    load_merged_metropolis_runs,
)


@dataclass(frozen=True)
class PeakFit:
    temperature: float
    susceptibility: float
    window_start: float
    window_end: float


def fit_peak(summaries: list[TemperatureSummary]) -> PeakFit:
    """Fit a downward quadratic to the grid maximum and two points per side."""
    susceptibilities = np.array([item.susceptibility for item in summaries])
    maximum_index = int(np.argmax(susceptibilities))
    if maximum_index < 2 or maximum_index + 2 >= len(summaries):
        raise ValueError(
            f"L={summaries[0].side}: susceptibility maximum is too near a grid edge"
        )

    window = summaries[maximum_index - 2 : maximum_index + 3]
    temperatures = np.array([item.temperature for item in window])
    values = np.array([item.susceptibility for item in window])

    # Centering the temperatures makes the quadratic coefficients better scaled.
    center = summaries[maximum_index].temperature
    quadratic, linear, constant = np.polyfit(temperatures - center, values, 2)
    if quadratic >= 0.0:
        raise ValueError(f"L={summaries[0].side}: fitted quadratic has no maximum")

    displacement = -linear / (2.0 * quadratic)
    peak_temperature = center + displacement
    if not temperatures.min() <= peak_temperature <= temperatures.max():
        raise ValueError(
            f"L={summaries[0].side}: fitted peak lies outside its five-point window"
        )
    peak_susceptibility = (
        quadratic * displacement**2 + linear * displacement + constant
    )
    return PeakFit(
        temperature=float(peak_temperature),
        susceptibility=float(peak_susceptibility),
        window_start=float(temperatures.min()),
        window_end=float(temperatures.max()),
    )


def main() -> None:
    by_size = load_merged_metropolis_runs()
    missing_sizes = {32, 64}.difference(by_size)
    if missing_sizes:
        raise ValueError(f"missing completed runs for L={sorted(missing_sizes)}")

    fits = {side: fit_peak(by_size[side]) for side in (32, 64)}
    estimated_critical_temperature = (
        2.0 * fits[64].temperature - fits[32].temperature
    )
    relative_error = (
        (estimated_critical_temperature - CRITICAL_TEMPERATURE)
        / CRITICAL_TEMPERATURE
        * 100.0
    )

    lines = []
    for side in (32, 64):
        lowest = by_size[side][0]
        fit = fits[side]
        lines.append(
            f"L={side}: T_peak = {fit.temperature:.6f}, "
            f"chi_peak = {fit.susceptibility:.6f}, "
            f"fit window = [{fit.window_start:.2f}, {fit.window_end:.2f}], "
            f"mean |M|(T={lowest.temperature:.2f}) = {lowest.mean_abs_m:.6f}"
        )
    lines.append(
        f"T_c = {estimated_critical_temperature:.6f} "
        f"(2*T_peak(64) - T_peak(32))"
    )
    lines.append(
        f"Onsager T_c = {CRITICAL_TEMPERATURE:.6f}; "
        f"relative deviation = {relative_error:+.3f}%"
    )
    report = "\n".join(lines) + "\n"

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    output = EVIDENCE / "peaks.txt"
    output.write_text(report)
    print(report, end="")
    print(f"wrote {output.relative_to(EVIDENCE.parent)}")


if __name__ == "__main__":
    main()
