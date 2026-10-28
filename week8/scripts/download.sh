#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p data
curl -fL --retry 3 https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week8-learning-sheet.pdf -o week8-learning-sheet.pdf
curl -fL --retry 3 https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/model.safetensors -o data/model.safetensors
curl -fL --retry 3 https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/tokenizer.json -o data/tokenizer.json
curl -fL --retry 3 https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt -o data/TinyStoriesV2-GPT4-valid.txt
curl -fL --retry 3 https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.model -o data/tok512.model
curl -fL --retry 3 https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv -o data/chinchilla-runs.csv
