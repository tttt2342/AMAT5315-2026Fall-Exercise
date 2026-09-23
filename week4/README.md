# Week 4: reproducibility guide

This directory contains a Rust library of explicit time integrators, periodic
one-dimensional advection--diffusion operators, and a two-dimensional
Fourier-pseudospectral vorticity solver. The commands below regenerate every
committed file in `week4/evidence/` from a clean clone.

Run setup commands from `week4/`. Run every Python script from
`week4/scripts/`, as shown below.

## 1. Install and build

The prerequisites are a stable Rust toolchain and Python 3.11 or newer. From
`week4/`, create a virtual environment, install the pinned Python packages, and
build the Rust binaries:

```text
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cargo build --release --locked --bins
```

Optional verification:

```text
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
```

## 2. Generate the two saved flow pipelines

These two pipelines create the inputs used by `plot_taylor_green.py` and
`plot_random_vorticity.py`.

### Taylor--Green pipeline

Values: RK4, `n = 64`, `nu = 0.1`, `dt = 0.01`, `t_end = 1`, and snapshots
every `0.1`.

```text
mkdir -p artifacts/taylor-green
target/release/field taylor-green --n 64 |
  target/release/fluid --method rk4 --nu 0.1 --dt 0.01 \
    --t-end 1 --every 0.1 --out artifacts/taylor-green
target/release/field taylor-green --n 64 --t 1 --nu 0.1 \
  > artifacts/taylor-green/exact-t1.json
```

### Random-flow pipeline

Values: RK4, `n = 128`, `nu = 0.004`, `dt = 0.01`, `t_end = 10`, seed
`2026`, initial wavenumbers `2--6`, and snapshots every `0.1`.

```text
mkdir -p artifacts/random
target/release/field random --n 128 --seed 2026 --k-min 2 --k-max 6 |
  target/release/fluid --method rk4 --nu 0.004 --dt 0.01 \
    --t-end 10 --every 0.1 --out artifacts/random
```

## 3. Regenerate the committed evidence

Change to the scripts directory once:

```text
cd scripts
```

Then run these commands in order. Paths in the output column are relative to
`week4/`.

| Order | Command | Committed file written |
|---:|---|---|
| 1 | `../.venv/bin/python plot_line_stability.py` | `evidence/line-stability.png` |
| 2 | `../.venv/bin/python plot_line_accuracy.py` | `evidence/line-accuracy.png` |
| 3 | `../.venv/bin/python plot_taylor_green.py` | `evidence/taylor-green.png` |
| 4 | `../.venv/bin/python plot_random_vorticity.py` | `evidence/random.png` |
| 5 | `../.venv/bin/python run_blowup_scan.py` | `evidence/blowup.png` |
| 6 | `../.venv/bin/python run_sensitivity.py` | `evidence/sensitivity.png` |
| 7 | `../.venv/bin/python run_order_convergence.py` | `evidence/order.png`, `evidence/convergence.json` |
| 8 | `../.venv/bin/python plot_convergence.py` | `evidence/convergence.png` |

The scripts that launch simulations also populate `week4/artifacts/`. In
particular, command 7 retains the random-flow final fields required by command
8 for the fourth-order Richardson estimate.

The Rust sources in `scripts/` are helper binaries invoked by the Python
drivers: `measure_rk4.rs` and `simulate_line_pulse.rs` support command 1, while
`line_accuracy.rs` supports command 2. `differentiation_comparison.rs` prints
the Fourier-versus-centred-difference error table and does not write a
committed evidence file; run it, if desired, with:

```text
cargo run --quiet --bin differentiation-comparison
```

After the eight numbered commands finish, the complete committed evidence set
is:

```text
evidence/blowup.png
evidence/convergence.json
evidence/convergence.png
evidence/line-accuracy.png
evidence/line-stability.png
evidence/order.png
evidence/random.png
evidence/sensitivity.png
evidence/taylor-green.png
```
