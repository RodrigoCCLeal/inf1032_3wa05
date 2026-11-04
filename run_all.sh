#!/usr/bin/env bash
# Roda o notebook único do projeto do início ao fim, gravando as saídas nele.
# Uso:  ./run_all.sh
# Pré-requisito: pip install jupyter nbconvert pyarrow lightgbm shap
set -euo pipefail
cd "$(dirname "$0")"

export ENEM_DIR="${ENEM_DIR:-dados_brutos}"   # CSVs brutos do ENEM (o notebook baixa se faltarem)
export ENEM_OUT="${ENEM_OUT:-data_processed}"  # onde ficam os derivados (o mesmo padrão do notebook)

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 INF1032_G5_projeto.ipynb
echo "fim"
