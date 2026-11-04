# INF1032 · Grupo 5 — Redação do ENEM 2024 × perfil socioeconômico

O projeto inteiro está em **um notebook**, `INF1032_G5_projeto.ipynb`, com o preparo ajustável em
`preparo.py` (importado pelo notebook). A ordem das seções é a ordem de execução.

| Seção | O que faz | Lê | Grava |
|---|---|---|---|
| — | Objetivo, métrica e critério de sucesso | — | — |
| 0 | Setup: caminhos, semente, imports | — | — |
| 1 | Baixa os dados do INEP (se faltarem), carrega e junta participantes + redação | `PARTICIPANTES_2024.csv`, `COMPETENCIAS_REDACAO_2024.csv` | `enem2024_base_bruta.parquet` |
| 2 | Análise exploratória | base bruta | gráficos |
| 3 | Engenharia de features (linha a linha) | base bruta | — |
| 4 | Split treino/teste e gravação | — | `enem2024_features.parquet`, `lista_features.csv` |
| 5–12 | Modelagem (só treino): preparo, referência, diagnóstico, candidatos, tuning, ensemble, interpretabilidade | `enem2024_features.parquet` | — |
| 13 | Avaliação no teste (uma vez só) | — | — |
| 14 | Checklist ético e de viés | — | — |
| 15 | Deploy (Streamlit) | — | — |

## Como rodar

**No Colab:** abra o notebook e rode de cima para baixo. Não é preciso subir nada: o Setup baixa o
`preparo.py` deste repositório e a seção 1 baixa os dois zips do INEP para `dados_brutos/` (~1,8 GB
de disco; leva alguns minutos na primeira vez). As seções 1 a 4 só precisam rodar uma vez; depois,
dá para começar da seção 5 (rode antes a célula de Setup).

Todos os caminhos são relativos à pasta do notebook: `dados_brutos/` (CSVs do INEP) e
`data_processed/` (derivados). Se você já tem os CSVs, coloque-os em `dados_brutos/` e o download é
pulado.

**Pela linha de comando:**

```bash
pip install jupyter nbconvert pyarrow lightgbm shap
export ENEM_DIR=/caminho/dos/csvs_brutos   # opcional; padrão: dados_brutos/
./run_all.sh
```

`ENEM_OUT` (onde ficam os derivados) cai em `data_processed/`.

Dependências: pandas, numpy, scikit-learn, matplotlib e pyarrow (já vêm no Colab); `lightgbm` e
`shap` são opcionais — sem eles o notebook usa o `HistGradientBoosting` e pula o gráfico do SHAP.

Memória: as seções 1 a 4 usam ~5 GB de RAM com a base cheia (4,3 milhões de inscritos). Cabe no
Colab gratuito.

## Arquivos antigos

Os notebooks numerados `01_…` a `12_…` e os da pasta `sprints/` são versões anteriores, por tarefa e
por sprint. O notebook único substitui todos eles.
