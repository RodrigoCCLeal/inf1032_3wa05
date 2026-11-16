# INF1032 · Grupo 5 — Redação do ENEM 2024 × perfil socioeconômico

**Pergunta:** quanto o perfil socioeconômico do participante explica a nota da redação do ENEM 2024 — e
para quem o modelo erra mais?

**Resposta curta:** pouco, e de forma desigual. O modelo final (boosting ajustado) erra, em média,
**111,3 pontos** no conjunto de teste (contra 114,0 da regressão linear de referência) e explica **28,5%**
da variação da nota. O erro é parecido entre tipos de escola, sexo e regiões, mas o modelo **superestima
a nota dos participantes indígenas** em cerca de 41 pontos. Detalhes nas seções 13 e 14 do notebook.

**App:** _link do Streamlit Community Cloud — preencher depois do deploy (seção 15.3 do notebook)._

## Arquivos

| Arquivo | O que é |
|---|---|
| `INF1032_G5_projeto.ipynb` | o projeto inteiro, da leitura dos dados ao modelo final; a ordem das seções é a ordem de execução |
| `preparo.py` | o preparo ajustável (porte do município, winsorização, imputação, índices), importado pelo notebook e pelo app |
| `app.py` | o app Streamlit: formulário do questionário → nota que o modelo associa ao perfil |
| `perfil.py` | as transformações linha a linha da seção 3, para uma pessoa (usado pelo app) |
| `modelo_redacao_g5.joblib` | o modelo final ajustado, gravado pela seção 15.1 |
| `requirements.txt` | dependências do app, nas mesmas versões com que o `.joblib` foi gravado |
| `limiares_classe_sisu.csv` | cortes das classes Baixa / Intermediária / Competitiva (seção 13.1) |
| `INF1032_G5_alvo_sisu.ipynb` | estudo do Sisu 2025 que gera `limiares_classe_sisu.csv`, a partir de `dados_sisu/` |
| `run_all.sh` | roda o notebook do início ao fim pela linha de comando |
| `antigos/` | notebooks das sprints anteriores, substituídos pelo notebook único |

## Seções do notebook

| Seção | O que faz | Lê | Grava |
|---|---|---|---|
| — | Objetivo, métrica e critério de sucesso | — | — |
| 0 | Setup: caminhos, semente, imports | — | — |
| 1 | Baixa os dados do INEP (se faltarem), carrega e junta participantes + redação | `PARTICIPANTES_2024.csv`, `COMPETENCIAS_REDACAO_2024.csv` | `enem2024_base_bruta.parquet` |
| 2 | Análise exploratória | base bruta | gráficos |
| 3 | Engenharia de features (linha a linha) | base bruta | — |
| 4 | Split treino/teste e gravação | — | `enem2024_features.parquet`, `lista_features.csv` |
| 5–12 | Modelagem (só treino): preparo, referência, diagnóstico, candidatos, tuning, ensemble, interpretabilidade e escolha do modelo | `enem2024_features.parquet` | — |
| 13 | Avaliação no teste (uma vez só) e erro em classes do Sisu | linhas de teste do parquet, `limiares_classe_sisu.csv` | — |
| 14 | Erro por grupo no teste e checklist ético | — | — |
| 15 | Exportar o modelo, conferir o app e publicar | `enem2024_base_bruta.parquet` (conferência do app) | `modelo_redacao_g5.joblib` |

## Como rodar o notebook

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

**Memória e tempo.** As seções 1 a 4 usam ~5 GB de RAM com a base cheia (4,3 milhões de inscritos):
cabe no Colab gratuito. Numa máquina com menos de 8 GB, rode as seções 1 a 4, reinicie o kernel e
continue da seção 5 (com a célula de Setup antes). Com `RAPIDO = False` (a versão do G2), a modelagem leva
cerca de 20 minutos em 2 núcleos.

## Como rodar o app

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app precisa, na mesma pasta, de `modelo_redacao_g5.joblib`, `preparo.py` e `perfil.py`. Se a seção 15.1
for rodada de novo com outras versões das bibliotecas, atualize o `requirements.txt` com as versões que ela
imprime — o app avisa quando elas não batem.

## Arquivos antigos

A pasta `antigos/` guarda os notebooks numerados `01_…` a `12_…` e os da pasta `sprints/`: versões
anteriores, por tarefa e por sprint. O notebook único substitui todos eles.
