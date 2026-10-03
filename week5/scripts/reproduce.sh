#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${TMPDIR:-/tmp}/amat5315-week5-mpl"
if [[ ! -f inputs/reflector.json || ! -f inputs/marmousi.json ]]; then
  curl -fLO https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip
  unzip -o week5-inputs.zip
  rm week5-inputs.zip
fi
uv sync --locked
cargo install --path seismic --locked
cargo test --release --manifest-path seismic/Cargo.toml --locked
.venv/bin/pytest -q tests
.venv/bin/python scripts/ad_evidence.py
seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
.venv/bin/python scripts/forward_evidence.py
seismic --experiment inputs/reflector.json --mode born --out artifacts/born
seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --every 3 --out artifacts/adjoint
.venv/bin/python scripts/adjoint_evidence.py
for b in 1 3 5 10; do
  seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --storage treeverse --checkpoints "$b" --out "artifacts/checkpoint-$b"
done
seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
seismic --experiment inputs/marmousi.json --mode adjoint --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 --out artifacts/marmousi-image
.venv/bin/python scripts/checkpoint_evidence.py
printf '%s\n' 'Numerical checks passed. Export the three course-viewer PNGs as described in README.md.'
