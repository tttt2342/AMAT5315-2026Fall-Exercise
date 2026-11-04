#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week9-learning-sheet.pdf -o week9-learning-sheet.pdf
base=https://raw.githubusercontent.com/TensorBFS/TensorInference.jl
rev=935ed86f8e473f523cac9461c45a1f470e74f0f9
curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 "$base/$rev/examples/asia-network/model.uai" -o asia.uai
base=https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition
rev=22005844423f5bca62798548df808cc418fc0d7d
mkdir -p promedus
for suffix in uai uai.evid uai.PR; do
 curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 "$base/$rev/uai2014/PR/Promedus_14.$suffix" -o "promedus/Promedus_14.$suffix"
done
curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 "$base/$rev/uai2014/MAR/Promedus_14.uai.MAR" -o promedus/Promedus_14.uai.MAR
