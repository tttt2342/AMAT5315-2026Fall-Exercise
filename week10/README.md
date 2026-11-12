# Week 10: Approximate tensor contraction for the Ising model

[Official handout](https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week10-learning-sheet.pdf).
Generated data and figures are local to this directory.

## Gaussian matrix SVD

From this directory, using Rust 1.96.0 and Python with the packages in
`requirements.txt`:

```sh
cargo run --release --locked --bin svd_study -- --out artifacts/svd
python scripts/check_svd.py
```

The Rust generator uses ChaCha8Rng seeded with 2026 and the Box–Muller
transform. The independent Python check reads the original matrix, the singular
spectrum and all five reconstructions from little-endian float64 files. It
compares measured Frobenius errors with the singular-value tail and marks the
storage break-even at rank 256 in `artifacts/svd.png`.

## Open square with boundary MPS

The row tensors follow the handout's square-root bond split. Close each missing
bond with the edge vector. Absorb a row into the MPS, orthogonalize from right to
left with thin SVDs, and truncate from left to right. Record one normalization
scale per row from the last core. Add the sum of log scales to the log of the
bottom-edge contraction.

```sh
cargo install --path . --locked --root .tool
export PATH="$PWD/.tool/bin:$PATH"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
python scripts/verify.py boundary --replay runs/boundary64
python scripts/sweep.py boundary --jobs 4
```

The independent checker enumerates the open 4×4 spin states and compares the
specific heat from energy variance with the finite-difference result. The sweep
records full tensor evidence for both ranks at the nine refined points for
L = 64 and replays each saved contraction. It writes the figure, g(K) table,
peak-refinement and timing table, and fit summary to `artifacts/`.

## Periodic square with TRG

The two diagonal SVD splits and ring contraction follow the
[pinned Julia example](https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl).
The implementation uses the handout's max-absolute normalization, retains up
to the stated rank and uses two independent periodic traces,
`sum(i,j) A[i,j,i,j]`. Each round halves the number of site tensors.
A square of side 1024 therefore takes 20 rounds.

```sh
cargo install --path . --locked --root .tool --force
thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024
python scripts/verify.py trg --replay runs/trg1024
python scripts/sweep.py trg --jobs 4
```

The independent periodic enumeration counts one bond in each positive lattice
direction per site, including both parallel bonds when L = 2. The replay
checks the tensor count, saved spectra, final regrouping, and opposite-leg
closure. To include every retained peak sample in the verification report, run:

```sh
python scripts/verify.py boundary --replay runs/boundary64 runs/boundary-peak/*
python scripts/verify.py trg --replay runs/trg1024 runs/trg-peak/*
cargo test --locked
```

## Numerical settings and evidence

Both geometries use zero field and J = k_B = 1. The coarse grid is
K = 0.40 through 0.50 in steps of 0.0025 at χ = 16. Both χ = 16 and 32
use nine refined samples with spacing 0.00125 around each coarse peak.
The three-point centered difference is applied to g(K), followed by a
three-point parabolic fit to the maximum of the resulting specific heat.
A separate linear fit of peak temperature against 1/L is made for each
geometry, first with all sizes and then with the two largest. The
χ = 32 refined grid is also subsampled every other point to compare steps.

Run summaries record the solver wall time excluding evidence serialization.
Per-size peak tables report the sum of solver times for each set of samples.
Parallel execution and hardware affect these times.

Full evidence consists of JSON metadata and row-major little-endian float64
tensors. The boundary metadata records edge vectors, row normalization
scales, squared discarded weights and final MPS core shapes. The TRG metadata
records scales, site counts, both diagonal splits and singular spectra for
every round, and the final rank-four tensor. Only the transient coarse scans
and smaller-size refinement points use the summary-only CLI option. Their
g(K) values and timings are retained in the CSV tables.

The local artifacts are the three PNGs, two g(K) tables, two peak/timing
tables, fit summaries and the independent checks. The largest-size peak
stencils and the two standalone runs are under `runs/`.

## Environment and sources

The results below were generated on arm64 macOS 26.1 with Rust 1.96.0,
Accelerate's modern LP64 LAPACK/BLAS interface (macOS 13.3 or later), and
Python 3.12.15. The Rust build links system BLAS and LAPACK on non-macOS
systems. Install the plotting and verification packages with
`python -m pip install -r requirements.txt` in a Python environment.

The lecture PDF was downloaded on 2026-11-12. Its SHA-256 is
`d053e28083dce7e199e3a866e65e9da05d78a231614425d2b3c78dbacfbb1e60`.
The Gaussian input is generated locally. The Ising weights and spin-enumeration
references use the model and conventions in the handout.

## Results

The Gaussian matrix's relative error at rank 256 is 0.325165, where factor
storage equals dense storage. The first rank below 10% error is 395 and
requires 1.542969 times as many entries as the original. All five measured
errors agree with the singular-spectrum prediction within 1e-8. The full-rank
reconstruction has relative error 3.63e-15.

| Geometry | L | Refined peak T | Specific heat at peak | Nine χ = 32 solver times summed (s) |
| --- | ---: | ---: | ---: | ---: |
| Open | 16 | 2.093105 | 1.115448 | 0.42 |
| Open | 32 | 2.179676 | 1.387532 | 3.85 |
| Open | 64 | 2.224220 | 1.683394 | 21.93 |
| Periodic | 64 | 2.281602 | 2.251392 | 40.84 |
| Periodic | 256 | 2.272260 | 2.903819 | 60.74 |
| Periodic | 1024 | 2.270012 | 3.162089 | 87.08 |

| Geometry | All sizes T_c | Largest two T_c | All-size relative error |
| --- | ---: | ---: | ---: |
| Open | 2.267506 | 2.268765 | 0.07400% |
| Periodic | 2.269205 | 2.269263 | 0.00086% |

The Onsager temperature is 2.269185314. The maximum relative peak shift from
the rank or step comparisons is 0.00145% for open squares and 0.0983% for
periodic squares. The peak grows in both studies. The open-square peaks rise
toward the bulk temperature from below, and periodic peaks fall toward it
from above. The periodic estimate is the intercept at 1/L = 0.

| Independent check | Absolute error |
| --- | ---: |
| Open 4×4 g at K = 0, 0.44, 0.7 (maximum) | 2.22e-16 |
| Open 4×4 specific heat at K = 0.44 | 3.84e-6 |
| Periodic 2×2 g at K = 0.44 | 0 |
| Periodic 4×4 g at K = 0.44 | 1.06e-4 |
| Periodic 2×2 specific heat at K = 0.44 | 3.70e-6 |

The standalone open 64×64 run gives g = 0.9368851604782735. The periodic
1024×1024 run at K = 0.44 and χ = 16 gives g = 0.9287239625770712 over
1,048,576 sites. Both standalone runs and all 36 retained refinement runs
passed independent replay.
