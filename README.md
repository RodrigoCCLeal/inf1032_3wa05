# ordem

Os notebooks do projeto separados **por tarefa** e numerados na ordem de execução.
Os arquivos antigos (`INF1032_G5_*.ipynb`, por sprint) continuam na raiz do
repositório, intactos.

| # | notebook | tarefa | lê | grava |
|---|----------|--------|----|-------|
| 01 | `01_carregar_dados.ipynb` | carrega os CSVs brutos e junta participantes + redação | `PARTICIPANTES_2024.csv`, `COMPETENCIAS_REDACAO_2024.csv` | `enem2024_base_bruta.parquet` |
| 02 | `02_analise_exploratoria.ipynb` | EDA: nota da redação × perfil | base bruta | gráficos |
| 03 | `03_features.ipynb` | alvos, codificação, split treino/teste, matriz de features | base bruta | `enem2024_features_ordinal.parquet`, `enem2024_features_amostra_onehot.parquet`, `lista_features.csv` |
| 04 | `04_alvo_sisu.ipynb` | classes de redação a partir das notas de corte do Sisu | `dados_sisu/`, `RESULTADOS_2024.csv` | `sisu2025_cursos_redacao.csv`, `limiares_classe_sisu.csv` |
| 05 | `05_clusterizacao.ipynb` | **vazio** — perfis de participante (KMeans + silhueta) | features de 03 | — |
| 06 | `06_modelo_referencia.ipynb` | regressão linear validada (modelo de referência) | features de 03 | — |
| 07 | `07_classificacao_risco.ipynb` | classificação do risco de zerar a redação | features de 03 | — |
| 08 | `08_ensemble_tuning_shap.ipynb` | ensemble, tuning e SHAP | features de 03 | — |
| 09 | `09_visualizacoes_finais.ipynb` | **vazio** — gráficos de fechamento do G2 | 02 e 06–08 | — |
| 10 | `10_avaliacao_teste.ipynb` | **vazio** — conjunto de teste, uma vez só | `split == "teste"` | — |
| 11 | `11_checklist_etico_vies.ipynb` | **vazio** — equidade consolidada de 06 e 07 | — | — |
| 12 | `12_deploy_streamlit.ipynb` | **vazio** — empacotar o modelo e o app | — | — |

Os cinco "vazios" são os itens que a Sprint 6 deixou em aberto: existem só para
segurar o lugar na ordem. `run_all.sh` pula qualquer notebook sem célula de código.

## Como rodar

```bash
export ENEM_DIR=/caminho/dos/csvs_brutos   # padrão: /content (Colab)
./run_all.sh          # todos, na ordem
./run_all.sh 03 06    # só esses
```

`ENEM_OUT` já cai em `../data_processed`. A execução é *in-place*: as saídas
ficam gravadas no próprio notebook.

Dependências: 01 → 02 e 03; 03 → 05, 06, 07, 08, 10. 04 é independente (lê os
próprios arquivos do Sisu, que nenhum outro notebook usa).

O que mudou em relação aos notebooks da raiz: a leitura dos CSVs brutos saiu de
02 e de 03 e virou o notebook 01. O resto é cópia.
