#!/usr/bin/env python3
"""Run paired vorticity-sensitivity experiments and plot their separation."""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "sensitivity"
FIGURE = ROOT / "evidence" / "sensitivity.png"
FIELD = ROOT / "target" / "release" / "field"
FLUID = ROOT / "target" / "release" / "fluid"
DT = 0.01
T_END = 20.0
SNAPSHOT_EVERY = 0.5
RIPPLE_FACTOR = 7.0e-5


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    n: int
    nu: float
    field_arguments: tuple[str, ...]


CASES = (
    Case(
        name="taylor-green",
        label=r"Taylor--Green ($n=64$, $\nu=0.1$)",
        n=64,
        nu=0.1,
        field_arguments=("taylor-green", "--n", "64"),
    ),
    Case(
        name="random",
        label=r"Random ($n=128$, $\nu=0.004$)",
        n=128,
        nu=0.004,
        field_arguments=(
            "random",
            "--n",
            "128",
            "--seed",
            "2026",
            "--k-min",
            "2",
            "--k-max",
            "6",
        ),
    ),
)


def build_binaries() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--bins"],
        cwd=ROOT,
        check=True,
    )


def generate_original(case: Case) -> dict:
    completed = subprocess.run(
        [str(FIELD), *case.field_arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    field = json.loads(completed.stdout)
    (ARTIFACTS / f"{case.name}-original.json").write_bytes(completed.stdout)
    return field


def add_vorticity_ripple(field: dict, case: Case) -> tuple[dict, float]:
    u = np.asarray(field["u"], dtype=float).reshape(case.n, case.n)
    v = np.asarray(field["v"], dtype=float).reshape(case.n, case.n)
    maximum_component = float(max(np.max(np.abs(u)), np.max(np.abs(v))))
    epsilon = RIPPLE_FACTOR * maximum_component

    coordinates = 2.0 * np.pi * np.arange(case.n) / case.n
    x, y = np.meshgrid(coordinates, coordinates)
    # The solver uses omega = d_x v - d_y u.  For
    # delta omega = -epsilon cos(3x) cos(4y), take
    # delta psi = -epsilon cos(3x) cos(4y) / 25,
    # delta u = d_y psi, and delta v = -d_x psi.
    u_perturbation = 4.0 * epsilon / 25.0 * np.cos(3.0 * x) * np.sin(4.0 * y)
    v_perturbation = -3.0 * epsilon / 25.0 * np.sin(3.0 * x) * np.cos(4.0 * y)

    perturbed = dict(field)
    perturbed["u"] = (u + u_perturbation).ravel().tolist()
    perturbed["v"] = (v + v_perturbation).ravel().tolist()
    return perturbed, maximum_component


def encode_field(field: dict) -> bytes:
    return (json.dumps(field, separators=(",", ":")) + "\n").encode("utf-8")


def run_field(case: Case, variant: str, field_bytes: bytes) -> Path:
    output_directory = ARTIFACTS / f"{case.name}-{variant}"
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(FLUID),
        "--method",
        "rk4",
        "--nu",
        str(case.nu),
        "--dt",
        str(DT),
        "--t-end",
        str(T_END),
        "--every",
        str(SNAPSHOT_EVERY),
        "--out",
        str(output_directory),
    ]
    print(f"running {case.name} {variant} ...", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=field_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (output_directory / "diagnostics.tsv").write_bytes(completed.stdout)
    if completed.stderr:
        (output_directory / "stderr.txt").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case.name} {variant} failed:\n{completed.stderr.decode('utf-8')}"
        )
    elapsed = time.perf_counter() - started
    final_time = float(completed.stdout.decode("utf-8").strip().splitlines()[-1].split("\t")[0])
    if not math.isclose(final_time, T_END, abs_tol=5.0e-7):
        raise RuntimeError(f"{case.name} {variant} stopped at t={final_time}")
    print(f"  reached t={final_time:g} ({elapsed:.1f} s)", flush=True)
    return output_directory / "fields.jsonl"


def relative_distances(original_path: Path, perturbed_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times = []
    distances = []
    with original_path.open(encoding="utf-8") as original_stream, perturbed_path.open(
        encoding="utf-8"
    ) as perturbed_stream:
        for original_line, perturbed_line in zip_longest(original_stream, perturbed_stream):
            if original_line is None or perturbed_line is None:
                raise RuntimeError("paired runs saved different numbers of snapshots")
            original = json.loads(original_line)
            perturbed = json.loads(perturbed_line)
            if not math.isclose(original["t"], perturbed["t"], abs_tol=5.0e-7):
                raise RuntimeError("paired snapshot times do not match")
            omega = np.asarray(original["omega"], dtype=float)
            omega_perturbed = np.asarray(perturbed["omega"], dtype=float)
            denominator = np.linalg.norm(omega)
            if denominator == 0.0:
                raise RuntimeError("unperturbed vorticity has zero norm")
            times.append(float(original["t"]))
            distances.append(float(np.linalg.norm(omega_perturbed - omega) / denominator))
    expected_count = round(T_END / SNAPSHOT_EVERY) + 1
    if len(times) != expected_count:
        raise RuntimeError(f"expected {expected_count} snapshots, found {len(times)}")
    return np.asarray(times), np.asarray(distances)


def save_distances(case: Case, times: np.ndarray, distances: np.ndarray) -> None:
    destination = ARTIFACTS / f"{case.name}-relative-distance.tsv"
    with destination.open("w", encoding="utf-8") as stream:
        stream.write("t\trelative_vorticity_distance\n")
        for time_value, distance in zip(times, distances):
            stream.write(f"{time_value:.6f}\t{distance:.17e}\n")


def draw(results: list[tuple[Case, np.ndarray, np.ndarray]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9.5,
        }
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    styles = [
        {"color": "#3569a8", "marker": "o"},
        {"color": "#d45b38", "marker": "s"},
    ]
    for (case, times, distances), style in zip(results, styles):
        positive = distances > 0.0
        axis.plot(
            times[positive],
            distances[positive],
            linewidth=1.7,
            markersize=4.0,
            markevery=2,
            label=case.label,
            **style,
        )
    axis.set_yscale("log")
    axis.set_xlim(0.0, T_END)
    axis.set_xlabel(r"time $t$")
    axis.set_ylabel(r"$\|\omega_\delta-\omega\|_2/\|\omega\|_2$")
    axis.set_title(r"Sensitivity to the initial vorticity ripple")
    axis.grid(True, which="both", alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)
    print(f"saved {FIGURE}")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    build_binaries()
    results = []
    summary = {}
    for case in CASES:
        original = generate_original(case)
        perturbed, maximum_component = add_vorticity_ripple(original, case)
        perturbed_bytes = encode_field(perturbed)
        (ARTIFACTS / f"{case.name}-perturbed.json").write_bytes(perturbed_bytes)
        print(
            f"{case.name}: M={maximum_component:.16e}, "
            f"ripple amplitude={RIPPLE_FACTOR * maximum_component:.16e}",
            flush=True,
        )
        original_path = run_field(case, "original", encode_field(original))
        perturbed_path = run_field(case, "perturbed", perturbed_bytes)
        times, distances = relative_distances(original_path, perturbed_path)
        save_distances(case, times, distances)
        results.append((case, times, distances))
        summary[case.name] = {
            "M": maximum_component,
            "ripple_amplitude": RIPPLE_FACTOR * maximum_component,
            "initial_relative_distance": float(distances[0]),
            "final_relative_distance": float(distances[-1]),
        }
    (ARTIFACTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    draw(results)


if __name__ == "__main__":
    main()
