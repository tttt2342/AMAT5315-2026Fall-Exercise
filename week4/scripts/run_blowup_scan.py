#!/usr/bin/env python3
"""Run the requested Taylor--Green and random-flow stability scan."""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "scan"
FIGURE = ROOT / "evidence" / "blowup.png"
FIELD = ROOT / "target" / "release" / "field"
FLUID = ROOT / "target" / "release" / "fluid"


@dataclass(frozen=True)
class RunSpec:
    name: str
    case: str
    method: str
    nu: float
    dt: float
    t_end: float


@dataclass
class RunResult:
    spec: RunSpec
    times: np.ndarray
    energies: np.ndarray
    enstrophies: np.ndarray
    reached_end: bool
    stopping_time: float


def build_binaries() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--bins"],
        cwd=ROOT,
        check=True,
    )


def generate_field(arguments: list[str], destination: Path) -> bytes:
    completed = subprocess.run(
        [str(FIELD), *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    destination.write_bytes(completed.stdout)
    return completed.stdout


def parse_diagnostics(output: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = output.decode("utf-8").strip().splitlines()
    if not lines or lines[0] != "t\tE\tZ":
        raise RuntimeError("fluid produced an unexpected diagnostic header")
    rows = [[float(value) for value in line.split("\t")] for line in lines[1:]]
    if not rows:
        raise RuntimeError("fluid produced no diagnostics")
    values = np.asarray(rows, dtype=float)
    return values[:, 0], values[:, 1], values[:, 2]


def run_case(spec: RunSpec, initial_field: bytes) -> RunResult:
    output_directory = ARTIFACTS / spec.name
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(FLUID),
        "--method",
        spec.method,
        "--nu",
        str(spec.nu),
        "--dt",
        str(spec.dt),
        "--t-end",
        str(spec.t_end),
        "--every",
        "0.5",
        "--out",
        str(output_directory),
    ]
    print(f"running {spec.name} ...", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=initial_field,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - started
    (output_directory / "diagnostics.tsv").write_bytes(completed.stdout)
    if completed.stderr:
        (output_directory / "stderr.txt").write_bytes(completed.stderr)

    times, energies, enstrophies = parse_diagnostics(completed.stdout)
    finite = np.isfinite(energies) & np.isfinite(enstrophies)
    reached_end = (
        completed.returncode == 0
        and finite[-1]
        and math.isclose(times[-1], spec.t_end, abs_tol=5.0e-7)
    )
    if completed.returncode != 0 and finite[-1]:
        raise RuntimeError(
            f"{spec.name} failed without a non-finite diagnostic:\n"
            f"{completed.stderr.decode('utf-8')}"
        )
    stopping_time = spec.t_end if reached_end else float(times[-1])
    outcome = "reached t_end" if reached_end else f"blew up at t={stopping_time:.6f}"
    print(f"  {outcome} ({elapsed:.1f} s)", flush=True)
    return RunResult(
        spec=spec,
        times=times,
        energies=energies,
        enstrophies=enstrophies,
        reached_end=reached_end,
        stopping_time=stopping_time,
    )


def largest_speed(field_bytes: bytes) -> float:
    field = json.loads(field_bytes)
    u = np.asarray(field["u"], dtype=float)
    v = np.asarray(field["v"], dtype=float)
    return float(np.max(np.hypot(u, v)))


def plot_results(
    taylor_results: list[RunResult],
    random_results: list[RunResult],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    colours = ["#3569a8", "#d45b38", "#2d8a57"]
    markers = ["o", "s", "^"]

    for axis, title, results in zip(
        axes,
        [r"Taylor--Green, $n=64$, $\nu=0.1$", r"Random flow, $n=128$, $\nu=0.004$"],
        [taylor_results, random_results],
    ):
        for index, result in enumerate(results):
            finite = np.isfinite(result.energies) & (result.energies > 0.0)
            method = "RK4" if result.spec.method == "rk4" else "Euler"
            label = rf"{method}, $\Delta t={result.spec.dt:g}$"
            colour = colours[index % len(colours)]
            axis.plot(
                result.times[finite],
                result.energies[finite],
                color=colour,
                marker=markers[index % len(markers)],
                markersize=4.2,
                linewidth=1.6,
                label=label,
            )
            if not result.reached_end:
                axis.axvline(
                    result.stopping_time,
                    color=colour,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.8,
                )
                axis.annotate(
                    f"stop {result.stopping_time:.3f}",
                    xy=(result.stopping_time, 0.96),
                    xycoords=("data", "axes fraction"),
                    xytext=(-4, 0),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    rotation=90,
                    color=colour,
                    fontsize=8.5,
                )
        axis.set_yscale("log")
        axis.set_xlabel(r"time $t$")
        axis.set_ylabel(r"kinetic energy $E$")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.22, linewidth=0.6)
        axis.legend(frameon=False, loc="best")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)
    print(f"saved {FIGURE}")


def write_summary(results: list[RunResult], initial_max_speed: float) -> None:
    summary = {
        "random_initial_max_speed": initial_max_speed,
        "runs": [
            {
                "name": result.spec.name,
                "case": result.spec.case,
                "method": result.spec.method,
                "dt": result.spec.dt,
                "reached_end": result.reached_end,
                "stopping_time": result.stopping_time,
            }
            for result in results
        ],
    }
    (ARTIFACTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    build_binaries()
    taylor_field = generate_field(
        ["taylor-green", "--n", "64"], ARTIFACTS / "taylor-green-initial.json"
    )
    random_field = generate_field(
        [
            "random",
            "--n",
            "128",
            "--seed",
            "2026",
            "--k-min",
            "2",
            "--k-max",
            "6",
        ],
        ARTIFACTS / "random-initial.json",
    )
    initial_max_speed = largest_speed(random_field)
    print(f"largest random initial speed: {initial_max_speed:.16e}", flush=True)

    taylor_specs = [
        RunSpec("taylor-green-rk4-dt0.032", "taylor-green", "rk4", 0.1, 0.032, 8.0),
        RunSpec("taylor-green-rk4-dt0.033", "taylor-green", "rk4", 0.1, 0.033, 8.0),
    ]
    random_specs = [
        RunSpec("random-rk4-dt0.038", "random", "rk4", 0.004, 0.038, 10.0),
        RunSpec("random-rk4-dt0.040", "random", "rk4", 0.004, 0.040, 10.0),
        RunSpec("random-euler-dt0.01", "random", "euler", 0.004, 0.01, 10.0),
    ]
    taylor_results = [run_case(spec, taylor_field) for spec in taylor_specs]
    random_results = [run_case(spec, random_field) for spec in random_specs]
    all_results = [*taylor_results, *random_results]
    write_summary(all_results, initial_max_speed)

    rk4_results = [result for result in random_results if result.spec.method == "rk4"]
    stable_rk4 = [result for result in rk4_results if result.reached_end]
    unstable_rk4 = [result for result in rk4_results if not result.reached_end]
    if not stable_rk4 or not unstable_rk4:
        outcomes = ", ".join(
            f"dt={result.spec.dt:g}: {'stable' if result.reached_end else 'unstable'}"
            for result in rk4_results
        )
        raise RuntimeError(
            "the two requested random RK4 steps do not bracket the stability boundary "
            f"({outcomes}); an additional step size must be selected"
        )

    selected_random = [stable_rk4[0], unstable_rk4[0], random_results[-1]]
    plot_results(taylor_results, selected_random)


if __name__ == "__main__":
    main()
