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
