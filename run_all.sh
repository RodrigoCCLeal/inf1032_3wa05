#!/usr/bin/env bash
# Roda os notebooks desta pasta na ordem numérica, executando no lugar.
# Uso:  ./run_all.sh              roda todos
#       ./run_all.sh 03 06        roda só os que começam com 03 e 06
# Pré-requisito: pip install jupyter nbconvert
set -euo pipefail
cd "$(dirname "$0")"

export ENEM_DIR="${ENEM_DIR:-/content}"          # onde estão os CSVs brutos do ENEM
export ENEM_OUT="${ENEM_OUT:-../data_processed}" # onde ficam os derivados

if [ $# -gt 0 ]; then
  nbs=(); for p in "$@"; do nbs+=("${p}"*.ipynb); done
else
  nbs=([0-9][0-9]_*.ipynb)
fi

for nb in "${nbs[@]}"; do
  if ! grep -q '"cell_type": "code"' "$nb"; then
    echo "-- pulando $nb (sem células de código ainda)"
    continue
  fi
  echo "== $nb"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "$nb"
done
echo "fim"
