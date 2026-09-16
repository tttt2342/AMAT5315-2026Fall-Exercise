# Week 3: reproducibility guide

This guide regenerates the committed Week 3 recording and evidence from a clean
clone. Run every command from `week3/`. The raw `runs/` and `artifacts/`
directories are intentionally ignored by Git; the copied viewer recording
`spins.jsonl` and everything under `evidence/` are committed.

## 1. Install

Install the Rust CLI in release mode and create the Python plotting environment:

```bash
cargo install --path . --quiet
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

All sampler options are written explicitly below. A temperature ramp starts from
an all-up lattice and carries its final configuration into the next temperature.

## 2. Generate the runs

### Single-temperature and seed checks

These runs supply the Boltzmann plot and reproduce the sampler checks:

```bash
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/T1.8

ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0

ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
```

The fixed-seed reproducibility check uses two identical runs and one changed
seed:

```bash
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/a

ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/b

diff runs/a/series.jsonl runs/b/series.jsonl && echo IDENTICAL

ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 \
  --discard 2000 --measure 2000 --seed 2027 --out runs/c
```

### Viewer recording

Record ten frames per temperature and copy the recording to its committed path:

```bash
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 \
  --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp

test -s runs/ramp/spins.jsonl
cp runs/ramp/spins.jsonl spins.jsonl
```

The last command regenerates the committed `spins.jsonl`.

### Metropolis thermodynamic runs

The coarse ramps cover the full temperature interval; the longer window ramps
replace their overlapping points in the analysis scripts.

```bash
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
  --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32

ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
  --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64

ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32

ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

### Wolff thermodynamic runs

One recorded Wolff step is one cluster flip. The two window runs use a longer
discard because cluster moves do not have the same time unit as Metropolis
sweeps.

```bash
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64

ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
```

## 3. Regenerate the committed evidence

Run these steps in order after all runs above have finished.

### Boltzmann energy check

```bash
cargo run --release --example boltzmann
```

Writes `evidence/boltzmann.png` from `runs/T3.0/series.jsonl` and
`runs/T3.1/series.jsonl`. It uses total-energy bins 40 units wide, retains ratio
bins with at least five rows from each run, and overlays the fixed slope
`1/3.0 - 1/3.1`.

### Course-viewer frames

Open the [course viewer](https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html)
and drop `spins.jsonl` onto it. Select each temperature, move to the frame with
the listed stamp, press **Save PNG**, and save or rename the download as shown:

| Viewer selection | File written |
| --- | --- |
| `T=1.8`, `sweep=15400`, `m=0.9458` | `evidence/viewer-T1.8.png` |
| `T=2.3`, `sweep=37400`, `m=0.3442` | `evidence/viewer-T2.3.png` |
| `T=3.0`, `sweep=68200`, `m=0.0571` | `evidence/viewer-T3.0.png` |

These are browser UI actions rather than repository scripts.

### Metropolis thermodynamics and sampling diagnostics

```bash
.venv/bin/python scripts/thermodynamics.py
```

Writes `evidence/magnetization.png` and `evidence/susceptibility.png`.

```bash
.venv/bin/python scripts/peaks.py
```

Writes `evidence/peaks.txt` and also prints the same peak report.

```bash
.venv/bin/python scripts/errors.py > evidence/errors.txt
```

Writes `evidence/errors.txt` with naive, 50-block, and autocorrelation-aware
diagnostics.

```bash
.venv/bin/python scripts/chi_bootstrap.py
```

Writes `evidence/chi-bootstrap.png` using 500 replicates at block lengths 2,000,
4,000, and 8,000 sweeps.

```bash
.venv/bin/python scripts/trace.py
```

Writes `evidence/trace.png` from the first 2,000 recorded sweeps at `T=2.3` and
`T=3.0` for `L=64`.

```bash
.venv/bin/python scripts/acf_binning.py
```

Writes `evidence/acf-binning.png` for the `L=64`, `T=2.3` Metropolis series.

```bash
.venv/bin/python scripts/tau.py
```

Writes `evidence/tau.png` with the integrated autocorrelation time for both
lattice sizes on a logarithmic vertical axis.

### Metropolis and Wolff comparison

```bash
.venv/bin/python scripts/magnetization_compare.py
```

Writes `evidence/magnetization-compare.png`. It compares the `L=64` mean
absolute magnetization, fits the Wolff susceptibility peaks, and uses 500
block-bootstrap replicates at block lengths 2,000, 4,000, and 8,000.

```bash
.venv/bin/python scripts/compare.py
```

Writes `evidence/tau-compare.png`. It converts each Wolff cluster move to spin
update work with `tau_work = tau_moves * mean(cluster_size) / L^2`; Metropolis
time is already measured in sweeps.

## 4. Evidence inventory

Every committed file under `evidence/` is paired with its producer here.

| Committed file | Command or action that writes it |
| --- | --- |
| `evidence/boltzmann.png` | `cargo run --release --example boltzmann` |
| `evidence/viewer-T1.8.png` | Course viewer **Save PNG** at `T=1.8`, sweep 15400 |
| `evidence/viewer-T2.3.png` | Course viewer **Save PNG** at `T=2.3`, sweep 37400 |
| `evidence/viewer-T3.0.png` | Course viewer **Save PNG** at `T=3.0`, sweep 68200 |
| `evidence/magnetization.png` | `.venv/bin/python scripts/thermodynamics.py` |
| `evidence/susceptibility.png` | `.venv/bin/python scripts/thermodynamics.py` |
| `evidence/peaks.txt` | `.venv/bin/python scripts/peaks.py` |
| `evidence/errors.txt` | `.venv/bin/python scripts/errors.py > evidence/errors.txt` |
| `evidence/chi-bootstrap.png` | `.venv/bin/python scripts/chi_bootstrap.py` |
| `evidence/trace.png` | `.venv/bin/python scripts/trace.py` |
| `evidence/acf-binning.png` | `.venv/bin/python scripts/acf_binning.py` |
| `evidence/tau.png` | `.venv/bin/python scripts/tau.py` |
| `evidence/magnetization-compare.png` | `.venv/bin/python scripts/magnetization_compare.py` |
| `evidence/tau-compare.png` | `.venv/bin/python scripts/compare.py` |

## 5. Reproducibility and uncertainty notes

- Fixed seeds and the locked Rust dependencies make the sampled JSONL rows
  deterministic for this implementation. Matplotlib/font versions and browser
  rendering can still cause pixel-level differences in PNG files.
- The viewer images require a manual frame selection. The temperature, sweep,
  and magnetization stamps above identify the committed frames, but a future
  version of the hosted viewer may render them differently.
- Near the critical point, Metropolis measurements are strongly correlated.
  The block-length comparison therefore leaves its sampler-agreement error bars
  provisional.
- The bootstrap estimates sampling uncertainty only. It does not include
  finite-size error or bias from the two-size extrapolation and five-point
  quadratic peak fits.
- Wolff work normalization counts updated spins. It is not a wall-clock speedup,
  because growing a cluster and attempting a single-spin flip have different
  computational costs.
