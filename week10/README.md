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
