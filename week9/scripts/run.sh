#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/amat5315-week9-mpl"
cargo +1.96.0 install --locked --offline --path . --root ./bin
infer=./bin/bin/infer
for case in prior AD ADX; do
 "$infer" --model asia.uai --evidence "$case" --out "artifacts/$case"
 "$infer" --model asia.uai --evidence "$case" --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out "artifacts/bp-$case"
done
"$infer" --model asia.uai --evidence AD --order random --seed 42 --out runs/random
"$infer" --model asia.uai --evidence AD --order greedy --out runs/greedy
"$infer" --model asia.uai --evidence AD --analyze runs/random/order.json --out runs/analysis
"$infer" --model asia.uai --evidence AD --plan runs/greedy/order.json --out runs/replay
"$infer" --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --queries 59,102,139 --order greedy --out runs/promedus-greedy
"$infer" --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --queries 59,102,139 --order treesa --seed 42 --out runs/promedus-treesa
"$infer" --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/promedus-bp
for script in diagram plot_asia plot_orders plot_bp plot_benchmark verify; do
 .venv/bin/python "scripts/$script.py"
done
