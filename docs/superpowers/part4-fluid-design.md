# Part 4 Lennard-Jones fluid: design

## Goal

Extend the `md` crate from the Part 3 isolated dimer into a reproducible CLI
pipeline:

```text
md run -> artifacts/run.json + artifacts/traj.jsonl
md check artifacts -> recompute and report the physics
md video artifacts --out artifacts/run.mp4 -> atoms + g(r)
```

The Part 3 open-boundary dimer remains available. The fluid run adds a
periodic box, a minimum-image pair calculation, and a shifted cutoff.

## Physical contract

The default run is:

```text
N = 100, rho = 0.8, target temperature = 0.5
dt = 0.01, equilibration = 2000 steps
production = 10000 steps, save every 50 steps
random seed = 2026, integrator = velocity-Verlet
```

The initial positions are a 10 x 10 triangular lattice. With

```text
a = sqrt(2 / (sqrt(3) rho))
h = sqrt(3) a / 2
Lx = 10 a, Ly = 10 h
x(i,j) = (i + 0.5 (j mod 2)) a
y(i,j) = j h
```

the density is `N / (Lx Ly) = rho` and the even number of rows makes the
staggered lattice compatible with the top/bottom periodic boundary.

For each coordinate difference, use the minimum image

```text
d_min = d - L * round(d / L)
```

and wrap positions after every integration step into `0 <= x < Lx` and
`0 <= y < Ly`. Pairs with `r >= rc` do not interact, with `rc = 2.5`. Inside
the cutoff use

```text
U_cut(r) = U(r) - U(rc)
F_cut(r) = F(r)
```

and use zero for both at or beyond the cutoff. The force still obeys Newton's
third law by adding a pair force to one atom and subtracting it from the other.

## Rust design

Keep the Part 3 `Integrator` trait and `VelocityVerlet` implementation. Extend
`System` with a periodic-box configuration while preserving the existing plain
open-boundary constructor for the dimer. The system continues to own positions,
velocities, and cached accelerations; its force calculation chooses either the
plain open-boundary interaction or the periodic shifted-cutoff interaction.

Use small data types for the file boundary:

```text
RunConfig       -> run.json
TrajectoryFrame -> one JSON object in traj.jsonl
```

`run.json` must contain `n`, `rho`, `box` as `[Lx, Ly]`, `dt`,
`temperature`, `eq_steps`, `steps`, `sample_every`, `seed`, and
`integrator` as `"velocity-verlet"`.

Each production frame must contain `step`, `t = step * dt`, wrapped `pos`,
`vel`, `E_pot` from the shifted potential, and `E_kin`. Production step zero is
not saved; the default run therefore writes 200 frames.

Use `clap` for the three CLI subcommands, `serde`/`serde_json` for the JSON
files, and a seeded Gaussian generator for the initial velocities. The existing
Plotters dependency can render video frames; `ffmpeg` can encode them to MP4.

## CLI

```text
md run [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01]
       [--eq-steps 2000] [--steps 10000] [--sample-every 50]
       [--seed 2026] [--out artifacts]
md check artifacts
md video artifacts --out artifacts/run.mp4
```

The `run` command creates its output directory. During equilibration it draws
independent Gaussian velocity components with variance `T`, subtracts the
mean velocity, and rescales initially and every 50 steps using

```text
T_thermo = 2 E_kin / (2 N - 2)
v <- v * sqrt(T_target / T_thermo)
```

The thermostat is disabled for production. The trajectory writer saves only
production steps divisible by `sample_every`.

## Physics check

`md check` reads the files without advancing the simulation. It validates the
schema and recomputes energies from saved positions and velocities using the
periodic shifted-cutoff model. It then reports:

1. secular drift, comparing the means of the first and last
   `k = max(1, floor(frame_count / 10))` frames:
   `abs(mean_last - mean_first) / abs(E0) < 2e-3`;
2. `T_speed = <|v|^2> / 2`, requiring `abs(T_speed - 0.5) < 0.05`;
3. the 24-bin equal-probability Maxwell-Boltzmann speed statistic
   `chi^2_22`, requiring `chi^2_22 < 2`.

The command exits nonzero for malformed files or a failed bound and prints each
measured value next to its limit followed by `PASS` or `FAIL`.

## Video and radial distribution function

Render one frame per saved trajectory frame. The atom panel shows positions in
the periodic box. The second panel shows `g(r)` up to
`0.5 * min(Lx, Ly)`, using minimum-image distances and the annulus normalization

```text
rho * pi * ((r + dr)^2 - r^2)
```

The implementation will use cumulative samples up to the current frame so the
pair-structure estimate becomes smoother as the fluid evolves. Temporary PNG
frames stay outside Git-tracked output, and `ffmpeg` encodes the final MP4 with
a small resolution/quality setting intended to keep it below 2 MB.

## Test-first acceptance

Before implementation, add four red tests:

1. the total force on a periodic configuration is zero within tolerance;
2. the shifted potential is continuous just inside and at `rc`;
3. the default contract run satisfies all three physics bounds;
4. the binary run writes readable `run.json` and `traj.jsonl` frames.

After the implementation these tests must pass in release mode. Then add
`week2/Makefile` with a `reproduce` target that runs

```text
cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts
```

from `week2/`, and ignore `week2/artifacts/` and all build outputs.
