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

## Profile

The naive release run was recorded with:

```bash
samply record md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof
```

In the Firefox Profiler Call Tree, `md::compute_periodic_accelerations` accounts
for 76% of samples (133 of 175). The complete profiled run is shown as 175 ms,
or 0.175 s.

After reinstalling the profiling build with the cell list as the default, I
recorded the same command as:

```bash
samply record md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-cells
```

The cell-list call tree shows `md::compute_periodic_cell_accelerations` in 93%
of samples (99 of 107); the complete run is 107 ms, or 0.107 s. The exported
profiler page is [profile-cells.pdf](profile-cells.pdf).

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 76 | 0.175 |
| Cell list | 93 | 0.107 |

## Benchmark

For each `N` and force path, run the command three times and use the median and
the minimum-maximum range. The speedup values below use paired repetitions:
`naive time / cells time`. The timings include the 100 equilibration steps and
500 production steps, with all other flags at their defaults.

```bash
for n in 100 400 1600; do
  for force in naive cells; do
    for repetition in 1 2 3; do
      /usr/bin/time -p md run --n "$n" --force "$force" \
        --steps 500 --eq-steps 100 \
        --out "/tmp/md-bench-${n}-${force}-${repetition}"
    done
  done
done

cargo run --manifest-path md/Cargo.toml --release --example scaling
```

The last command reads `scaling.csv` and writes `scaling.png`.

| N | Naive (s) | Cells (s) | Speedup (x) |
| ---: | ---: | ---: | ---: |
| 100 | 0.009988 (0.009269-0.013343) | 0.014313 (0.014082-0.014555) | 0.709 (0.648-0.917) |
| 400 | 0.080212 (0.079915-0.081260) | 0.049861 (0.047214-0.050077) | 1.623 (1.603-1.699) |
| 1600 | 1.140852 (1.125055-1.177744) | 0.197550 (0.192760-0.197745) | 5.837 (5.775-5.956) |

The naive pair search grows quadratically because it considers every pair, while
the cell list searches only the nine nearby cells, so its per-step cost grows
much more slowly at fixed density and cutoff.

## Heating and videos

From the repository root, reproduce the public heating trajectory and the two
fixed-temperature videos with:

```bash
md run --n 400 --temperature 0.2 --ramp-to 1.2 \
  --steps 20000 --sample-every 100 --out docs
md run --temperature 0.2 --out /tmp/cold && md video /tmp/cold --out week2/cold.mp4
md run --temperature 1.0 --out /tmp/hot && md video /tmp/hot --out week2/hot.mp4
```

Both videos are below 2 MiB. `docs/run.json` records `ramp_to: 1.2`, and the
published page loads 400 atoms and 200 production frames.

## Final recording

The final screen recording is deliberately not stored in git. Make one take of
at most two minutes: in a fresh terminal run
`make reproduce && cargo run --manifest-path md/Cargo.toml --release -- check artifacts`
until `PASS` appears, then open the
Pages link below. Pause near the cold start (`T` about 0.2) and again near
`T` about 1.0. Explain that the cold `g(r)` has sharp neighbour-shell peaks
persisting to long range, while the hot run keeps mainly the first peak and
approaches `g(r)=1` at larger separation. Upload the recording to a GitHub
Release rather than git, then place its release URL beside the Pages link.

## Pages

[Week 2 molecular-dynamics viewer](https://tttt2342.github.io/AMAT5315-2026Fall-Exercise/)
