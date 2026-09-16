# Week 3

## Boltzmann energy check

From `week3/`, redraw the energy histograms and probability-ratio check with:

```bash
cargo run --release --example boltzmann
```

The command reads `runs/T3.0/series.jsonl` and `runs/T3.1/series.jsonl`, uses
40-unit total-energy bins, and writes `evidence/boltzmann.png`. Ratio points are
kept only when both bins contain at least five samples. The dashed line fixes its
slope to `1/3.0 - 1/3.1`; only its arbitrary vertical offset is aligned to the
points.

## Thermodynamic curves

After all four Metropolis ramps under `artifacts/` have finished, draw the
magnetization and susceptibility curves with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/thermodynamics.py
```

When coarse and window runs overlap, the script uses the longer window run. It
checks every input against the measurement count in `run.json` before writing
`evidence/magnetization.png` and `evidence/susceptibility.png`.

Fit the two finite-size susceptibility peaks and estimate the critical
temperature with:

```bash
.venv/bin/python scripts/peaks.py
```

This writes the reported values to `evidence/peaks.txt` as well as stdout.

Compute naive, 50-block, and autocorrelation-aware diagnostics for every
Metropolis temperature with:

```bash
.venv/bin/python scripts/errors.py > evidence/errors.txt
```

The integrated autocorrelation time uses the running-window rule
`lag > 6 * tau_int`, with `tau_int = 1/2 + sum(rho(lag))`. The reported ratio
is `block50_se / naive_se`.

Run the block bootstrap of the susceptibility peaks and draw its three block-
length envelopes with:

```bash
.venv/bin/python scripts/chi_bootstrap.py
```

The command uses 500 replicates and seed 2026, then writes
`evidence/chi-bootstrap.png`.

Draw the first 2,000 measured sweeps of the size-64 magnetization traces at
`T=2.3` and `T=3.0` with:

```bash
.venv/bin/python scripts/trace.py
```

The command writes `evidence/trace.png`.

Draw the size-64, `T=2.3` autocorrelation and block-binning diagnostics with:

```bash
.venv/bin/python scripts/acf_binning.py
```

The command writes `evidence/acf-binning.png`.

Draw the integrated autocorrelation time over temperature for both lattice
sizes with:

```bash
.venv/bin/python scripts/tau.py
```

The command writes `evidence/tau.png` with a logarithmic vertical axis.

## Metropolis and Wolff comparison

After the Wolff window runs have finished, compare their magnetization with the
size-64 Metropolis window run and fit the Wolff susceptibility peaks with:

```bash
.venv/bin/python scripts/magnetization_compare.py
```

The command uses 500 block-bootstrap replicates at block lengths 2,000, 4,000,
and 8,000. It draws the 8,000-step error bars, reports their stability across
all three choices, and writes `evidence/magnetization-compare.png`.
