"""Scan inverse temperature, refine specific-heat peaks, and fit the critical temperature."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify import curvature, replay, run_solver

SIZES = {"boundary": (16, 32, 64)}
EXACT_TC = 2 / math.log(1 + math.sqrt(2))
COARSE_H = 0.0025
FINE_H = 0.00125


def sweep(pool, binary, method, l, chi, ticks, stage, retain=False):
    def point(tick):
        k = tick / 800  # Store grid positions as integers, so repeated points agree exactly.
        if retain:
            out = Path("runs") / f"{method}-peak" / f"L{l}-chi{chi}-k{k:.5f}"
            run = run_solver(binary, method, l, k, chi, out, full=True)
            replay(out)
        else:
            with tempfile.TemporaryDirectory(prefix="thermo-scan-") as tmp:
                run = run_solver(binary, method, l, k, chi, Path(tmp))
        return dict(l=l, k=k, chi=chi, stage=stage, g=run["g"], seconds=run["seconds"])

    return list(pool.map(point, ticks))


def heat_curve(rows, h):
    k = np.array([row["k"] for row in rows[1:-1]])
    g = np.array([row["g"] for row in rows])
    assert np.allclose(np.diff([row["k"] for row in rows]), h, atol=1e-14, rtol=0)
    return k, curvature(k, h, g[:-2], g[1:-1], g[2:])


def peak(rows, h):
    k, cv = heat_curve(rows, h)
    index = int(cv.argmax())
    assert 0 < index < len(k) - 1, f"heat peak at grid edge: {k[index]}"
    left, center, right = cv[index - 1:index + 2]
    offset = (left - right) / (2 * (left - 2 * center + right))
    vertex = k[index] + h * offset
    height = center + (right - left) * offset / 4
    return dict(grid_k=float(k[index]), k_peak=float(vertex), t_peak=float(1 / vertex),
                cv_peak=float(height))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(peaks):
    x = 1 / np.array([row["l"] for row in peaks])
    y = [row["t_peak"] for row in peaks]
    slope, tc = np.polyfit(x, y, 1)
    return dict(tc=float(tc), slope=float(slope), relative_error=float(abs(tc / EXACT_TC - 1)))


def plot(out, method, studies, fits):
    label = {"boundary": "Open square", "trg": "Periodic square"}[method]
    fig, (heat, drift) = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    for i, (coarse, fine) in enumerate(studies):
        l = coarse[0]["l"]
        k, cv = heat_curve(coarse, COARSE_H)
        color = f"C{i}"
        heat.plot(1 / k, cv, color=color, label=f"L = {l}")
        k, cv = heat_curve(fine, FINE_H)
        heat.plot(1 / k, cv, "o", color=color, ms=3)
        p = peak(fine, FINE_H)
        drift.scatter(1 / l, p["t_peak"], color=color, zorder=3)
    heat.axvline(EXACT_TC, color="0.35", linestyle=":", label="Onsager temperature")
    heat.plot([], [], "k-", label=r"$\chi=16$, $h=0.0025$")
    heat.plot([], [], "ko", ms=3, label=r"$\chi=32$, $h=0.00125$")
    heat.set(xlabel="Temperature T", ylabel="Specific heat per spin", title=f"{label} specific heat")
    heat.legend(fontsize=8)
    x = np.linspace(0, 1 / studies[0][0][0]["l"], 100)
    for name, style in [("all_sizes", "-"), ("largest_two", "--")]:
        f = fits[name]
        drift.plot(x, f["tc"] + f["slope"] * x, style,
                   label=f"{name.replace('_', ' ').capitalize()}: {f['tc']:.6f}")
    drift.scatter(0, EXACT_TC, marker="*", color="black", label="Onsager temperature", zorder=4)
    drift.set(xlabel="Inverse side length 1/L", ylabel="Peak temperature",
              title="Extrapolation to infinite size")
    drift.legend(fontsize=8)
    fig.savefig(out / f"{method}-critical.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=SIZES)
    parser.add_argument("--binary", type=Path, default=Path(".tool/bin/thermo"))
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    sizes = SIZES[args.method]
    samples, peaks, studies, changes = [], [], [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for l in sizes:
            print(f"{args.method} L={l}: coarse scan", flush=True)
            coarse = sweep(pool, args.binary, args.method, l, 16, range(320, 401, 2), "coarse")
            broad = peak(coarse, COARSE_H)
            mid = round(broad["grid_k"] * 800)
            print(f"{args.method} L={l}: refining around K={mid / 800:.5f}", flush=True)
            fine = {}
            for chi in (16, 32):
                fine[chi] = sweep(pool, args.binary, args.method, l, chi,
                                  range(mid - 4, mid + 5), "fine", retain=l == sizes[-1])
            estimates = [
                (16, COARSE_H, "coarse", broad),
                (16, FINE_H, "fine", peak(fine[16], FINE_H)),
                (32, COARSE_H, "subsampled", peak(fine[32][::2], COARSE_H)),
                (32, FINE_H, "fine", peak(fine[32], FINE_H)),
            ]
            for chi, h, stage, estimate in estimates:
                rows = coarse if stage == "coarse" else fine[chi]
                if stage == "subsampled":
                    rows = rows[::2]
                peaks.append(dict(l=l, chi=chi, h=h, stage=stage, **estimate,
                                  total_seconds=sum(row["seconds"] for row in rows)))
            p0, p1, p2, p3 = [entry[-1]["t_peak"] for entry in estimates]
            changes.append(dict(l=l, step_at_chi16=abs(p0 / p1 - 1),
                                step_at_chi32=abs(p2 / p3 - 1), rank_at_fine_step=abs(p1 / p3 - 1)))
            samples.extend([*coarse, *fine[16], *fine[32]])
            studies.append((coarse, fine[32]))
            write_csv(out / f"{args.method}-g.csv", samples)
            write_csv(out / f"{args.method}-peaks.csv", peaks)

    final_peaks = [p for p in peaks if p["chi"] == 32 and p["stage"] == "fine"]
    fits = {"all_sizes": fit(final_peaks), "largest_two": fit(final_peaks[1:])}
    plot(out, args.method, studies, fits)
    summary = dict(onsager_tc=EXACT_TC, **fits, relative_peak_changes=changes)
    report = json.dumps(summary, indent=2) + "\n"
    (out / f"{args.method}-fits.json").write_text(report)
    print(report, end="")
    assert all(f["relative_error"] < 0.01 for f in fits.values()), fits
    assert all(value < 0.01 for row in changes for key, value in row.items() if key != "l"), changes


if __name__ == "__main__":
    main()
