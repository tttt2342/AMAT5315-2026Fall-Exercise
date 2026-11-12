"""Verify saved Rust matrix reconstructions and plot the SVD study."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path("artifacts")
    data = out / "svd"
    metadata = json.loads((data / "matrix.json").read_text())
    rows, cols = metadata["shape"]
    matrix = np.fromfile(data / "matrix.bin", dtype="<f8").reshape(rows, cols)
    spectrum = np.fromfile(data / "singular-values.bin", dtype="<f8")
    assert len(spectrum) == min(rows, cols)
    assert np.all(spectrum[:-1] >= spectrum[1:]) and np.all(spectrum >= 0)
    norm = np.linalg.norm(matrix)
    assert np.isclose(norm, np.linalg.norm(spectrum), atol=1e-8, rtol=0)
    tail = np.r_[np.cumsum(spectrum[::-1] ** 2)[::-1], 0]
    prediction = np.sqrt(tail) / norm
    measured = {}
    for rank in metadata["ranks"]:
        approximation = np.fromfile(data / f"rank-{rank}.bin", dtype="<f8").reshape(rows, cols)
        measured[rank] = float(np.linalg.norm(matrix - approximation) / norm)
        assert abs(measured[rank] - prediction[rank]) < 1e-8
    assert measured[min(rows, cols)] < 1e-8

    equal_storage = rows * cols / (rows + cols)
    crossing = int(np.flatnonzero(prediction <= 0.10)[0])
    ratio = (rows + cols) * crossing / (rows * cols)
    result = {
        "measured": measured,
        "predicted_at_saved_ranks": {rank: float(prediction[rank]) for rank in measured},
        "equal_storage_rank": equal_storage,
        "first_rank_at_10_percent": crossing,
        "storage_ratio_at_10_percent": ratio,
        "storage_saving_at_10_percent": ratio < 1,
    }
    (out / "svd-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    left.plot(np.arange(1, len(spectrum) + 1), spectrum)
    left.set(xlabel="Singular-value index", ylabel="Singular value", title="Gaussian matrix spectrum")
    right.plot(np.arange(len(prediction)), prediction, label="Discarded-spectrum prediction")
    right.scatter(list(measured), list(measured.values()), label="Saved reconstructions", color="tab:orange")
    right.axvspan(equal_storage, len(spectrum), color="0.9", label="Factors use at least dense storage")
    right.axhline(0.10, color="0.4", linestyle="--", label="10% error")
    right.annotate(f"Rank {crossing}", (crossing, 0.10), xytext=(crossing + 10, 0.22),
                   arrowprops={"arrowstyle": "-"})
    right.set(xlabel="Retained rank", ylabel="Relative Frobenius error", title="Error and storage")
    right.legend(fontsize=8)
    fig.savefig(out / "svd.png", dpi=180)


if __name__ == "__main__":
    main()
