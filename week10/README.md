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
