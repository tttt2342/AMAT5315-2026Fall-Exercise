# Week 4: integrators on a periodic line

This Rust library contains one `Integrator` trait implemented by:

- `ForwardEuler`
- `ExplicitMidpoint`
- `RungeKutta4`

Each implementation advances a `&mut [f64]` by one step using an autonomous
rate function `Fn(&[f64], &mut [f64])`. The library also provides the rate of

```text
u_t + c u_x = nu u_xx
```

on the periodic grid `[0, 2 pi)` through `FourierAdvectionDiffusion` and
`CenteredAdvectionDiffusion`. In the Fourier operator, the Nyquist mode's first
derivative is zero but its second derivative is retained.

Example:

```rust
use continuum::{FourierAdvectionDiffusion, Integrator, RungeKutta4};

let equation = FourierAdvectionDiffusion::new(64, 1.0, 0.05);
let mut state = vec![0.0; 64];
RungeKutta4.step(&mut state, 0.01, &|u, du| equation.rate(u, du));
```

Run the tests with:

```text
cargo test
```

Reproduce the line-stability figure after installing `requirements.txt`:

```text
cd scripts
python3 plot_line_stability.py
```

The plotting script calls `measure-rk4`, which measures the colour map by
advancing the complex test equation with the library's `RungeKutta4` stepper.
It also calls `line-pulse` for the stable and unstable Gaussian-pulse panels;
both use the library's Fourier rate and `RungeKutta4` implementation.

The one-lap spatial/temporal comparison is reproduced from the same directory:

```text
python3 plot_line_accuracy.py
```

## Two-dimensional vorticity solver

`field` writes either the Taylor-Green field or a seeded random field as one
JSON object. `fluid` reads that object on standard input and integrates the
vorticity equation with the selected implementation of `Integrator`:

```text
cargo run --quiet --bin field -- taylor-green --n 64 |
  cargo run --quiet --bin fluid -- --method rk4 --nu 0.1 --dt 0.01 \
    --t-end 1 --every 0.1 --out artifacts/taylor-green
```

The solver uses Fourier pseudospectral derivatives and applies the two-thirds
cutoff to both the vorticity and every nonlinear product. Its output contract is
defined by `field.design.toml` and `fluid.design.toml`.

Run the differentiation comparison from the scripts directory:

```text
cd scripts
cargo run --quiet --bin differentiation-comparison
```

Compare and draw the saved Taylor-Green run from the same directory:

```text
python3 plot_taylor_green.py
```

Run the paired initial-vorticity sensitivity experiment from the same directory:

```text
python3 run_sensitivity.py
```

Run the time-step stability scan and draw its energy histories from the same
directory:

```text
python3 run_blowup_scan.py
```

Draw selected snapshots from the saved random-flow run from the same directory:

```text
python3 plot_random_vorticity.py
```

Run the RK4 order and random-flow reference-convergence studies from the same
directory:

```text
python3 run_order_convergence.py
```

Plot the saved random-flow convergence errors and apply the fourth-order
Richardson step selection from the same directory:

```text
python3 plot_convergence.py
```
