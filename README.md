# INF1032 · Grupo 5 — Redação do ENEM 2024 × perfil socioeconômico

**Pergunta:** quanto o perfil socioeconômico do participante explica a nota da redação do ENEM 2024 — e
para quem o modelo erra mais?

**Resposta curta:** pouco para cada pessoa, bastante para cada município. O modelo final (boosting
ajustado) erra, em média, **110,5 pontos** no conjunto de teste (contra 114,0 da regressão linear de
referência) e explica **29,3%** da variação da nota (R² 0,293). Isso **não** é limite do ruído entre
corretores: a concordância entre eles permitiria R² até 0,88 (seção 13.2). O que falta é variação real de
escrita que o perfil não carrega. Na **média do município**, o mesmo modelo chega a R² 0,68.

O erro é parecido entre tipos de escola, sexo e regiões, mas o modelo **superestima a nota dos
participantes indígenas** em cerca de 39 pontos (seção 14).

**Decisão (seção 16):** com orçamento para atender 20% dos municípios num programa de apoio à redação,
priorizar pela nota média prevista acerta **78%** dos atendidos (precisão) e alcança **63%** dos municípios
que precisam (recall), contra 52% e 42% da melhor regra ingênua (% de escola pública) e 25% e 20% do
sorteio. Se o falso negativo custa o dobro do falso positivo, o orçamento ideal é 30% (recall 79%). A lista
está em `municipios_priorizados.csv` e na aba de municípios do app.

**App:** [inf10323wa05-7p9a4prmgkj2vdyyvp8mmf.streamlit.app](https://inf10323wa05-7p9a4prmgkj2vdyyvp8mmf.streamlit.app/) — formulário do questionário → nota que o modelo associa ao perfil (seção 15 do notebook).

## Arquivos

| Arquivo | O que é |
|---|---|
| `INF1032_G5_projeto.ipynb` | o projeto inteiro, da leitura dos dados ao modelo final; a ordem das seções é a ordem de execução |
| `preparo.py` | o preparo ajustável (porte do município, winsorização, imputação, índices), importado pelo notebook e pelo app |
| `app.py` | o app Streamlit: formulário do questionário → nota que o modelo associa ao perfil |
| `perfil.py` | as transformações linha a linha da seção 3, para uma pessoa (usado pelo app) |
| `modelo_redacao_g5.joblib` | o modelo final ajustado, gravado pela seção 15.1 |
| `municipios_priorizados.csv` | a lista de municípios da decisão da seção 16.2 (o app mostra numa aba) |
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
| 1 | Baixa os dados do INEP (se faltarem), carrega e junta participantes + redação | `PARTICIPANTES_2024.csv`, `COMPETENCIAS_REDACAO_2024.csv` (e extrai o `RESULTADOS_2024.csv` para a 13.2) | `enem2024_base_bruta.parquet` |
| 2 | Análise exploratória | base bruta | gráficos |
| 3 | Engenharia de features (linha a linha) | base bruta | — |
| 4 | Split treino/teste e gravação | — | `enem2024_features.parquet`, `lista_features.csv` |
| 5–12 | Modelagem (só treino): preparo, referência, diagnóstico, candidatos (com a comparação de florestas da 9.1), tuning, ensemble, interpretabilidade e escolha do modelo | `enem2024_features.parquet` | — |
| 13 | Avaliação no teste (uma vez só), erro em classes do Sisu e por que o R² é baixo (13.2) | linhas de teste do parquet, `limiares_classe_sisu.csv`, `RESULTADOS_2024.csv` | — |
| 14 | Erro por grupo no teste e checklist ético | — | — |
| 15 | Exportar o modelo, conferir o app e publicar | `enem2024_base_bruta.parquet` (conferência do app) | `modelo_redacao_g5.joblib` |
| 16 | Decisão: quais municípios priorizar com um orçamento fixo (regra no treino, avaliação no teste, custo e equidade) | previsões das seções 13 e 16.1 | `municipios_priorizados.csv` |

## Como rodar o notebook

**No Colab:** abra o notebook e rode de cima para baixo. Não é preciso subir nada: o Setup baixa o
`preparo.py` deste repositório e a seção 1 baixa os dois zips do INEP para `dados_brutos/` (~3,5 GB
de disco, com o `RESULTADOS_2024.csv`; leva alguns minutos na primeira vez). As seções 1 a 4 só precisam rodar uma vez; depois,
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
continue da seção 5 (com a célula de Setup antes).

Com `RAPIDO = False` e `AMOSTRA = 600_000` (a rodada do G2), as seções 5 a 16 levaram **cerca de 6 horas**
em 2 núcleos e 8 GB de RAM. As células mais lentas foram o Stacking (seção 11, ~2,6 h), o tuning (seção
10.2, ~70 min), a comparação de florestas (9.1, ~35 min) e a permutation importance (seção 12, ~30 min).
O kernel fica em ~3 GB, mas o tuning e o Stacking abrem processos paralelos que copiam os dados: com 8 GB,
eles estouraram a memória. Para rodar numa máquina assim, defina `LOKY_MAX_CPU_COUNT=1` antes de rodar
(no Colab: `import os; os.environ["LOKY_MAX_CPU_COUNT"] = "1"` na primeira célula) ou use um ambiente
com mais RAM. Com `RAPIDO = True` o notebook roda em poucos minutos, com os números da rodada rápida.

## Como rodar o app

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app precisa, na mesma pasta, de `modelo_redacao_g5.joblib`, `preparo.py` e `perfil.py` (e de
`municipios_priorizados.csv` para a aba de municípios). Se a seção 15.1
for rodada de novo com outras versões das bibliotecas, atualize o `requirements.txt` com as versões que ela
imprime — o app avisa quando elas não batem.

## Arquivos antigos

A pasta `antigos/` guarda os notebooks numerados `01_…` a `12_…` e os da pasta `sprints/`: versões
anteriores, por tarefa e por sprint. O notebook único substitui todos eles.
