# Week 2

## Lennard-Jones field plot

From the `week2/` directory, generate the pair-energy and force-field plot with:

```bash
cargo run --manifest-path md/Cargo.toml --example field
```

The example calls the `energy` and `force` functions from the `md` crate and saves
the image as `week2/field.png`.

The arrow direction uses the sign of the scalar force and the radial unit vector.
Arrow lengths are compressed for display, with a small $10^{-8}$ display cutoff
to avoid turning floating-point noise at $r_0$ into a visible arrow.

## Two-atom dynamics

Run the dimer experiment with both integrators and generate the relative
total-energy error plot with:

```bash
cargo run --manifest-path md/Cargo.toml --example dimer
```

The example saves the image as `week2/dimer.png`. Its left panel compares
forward Euler and velocity-Verlet for 500 steps at `dt = 0.01`; its right panel
shows velocity-Verlet for 5000 steps with the error multiplied by 1000.

## Profiling build

Install the optimized profiling build as the md command:

    cargo install --path md --profile profiling --force --locked

The profiling profile keeps debug information and does not strip symbols, so
sampling profilers can resolve Rust function names.

## Timing

Measured with /usr/bin/time -p using the wall-clock real value. Each
program ran three times sequentially with the default N = 100 contract run.
The Rust release row uses the optimized profiling build installed as md.

| Program | Median (s) | Range: min–max (s) |
| --- | ---: | ---: |
| NumPy week2-sim.py | 3.12 | 3.10–3.19 |
| Rust debug | 1.90 | 1.89–1.94 |
| Rust release | 0.13 | 0.12–0.20 |
