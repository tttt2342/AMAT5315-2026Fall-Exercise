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
