# Dicionário de features — ENEM 2024 (redação × perfil)

Documentação do que é produzido por `INF1032_G5_features.ipynb`.
Entrada: merge de `PARTICIPANTES_2024` × `COMPETENCIAS_REDACAO_2024` por `NU_INSCRICAO` (4.332.944 linhas).
Saída: `data_processed/enem2024_features_ordinal.parquet` (base cheia, 71 colunas) e
`data_processed/enem2024_features_amostra_onehot.parquet` (500 mil linhas, 103 colunas).

---

## 1. Alvos

| Coluna | Definição | Linhas válidas | Para que serve |
|---|---|---|---|
| `y_fez_redacao` | 1 se o inscrito tem nota de redação | 4.332.944 | Prever abstenção; usa todos os inscritos |
| `y_nota` | Nota final da redação (0–1000) | 3.167.955 | Alvo principal de regressão |
| `y_zerada` | 1 se `TP_STATUS_REDACAO ≠ 1` (anulada, em branco, fuga ao tema…) | 3.167.955 | Classificação do "risco de zerar" |
| `y_nota_ok` | Nota entre as redações corrigidas sem problema | 3.002.710 | Regressão "limpa", sem a massa de zeros |
| `y_alto` | 1 se `y_nota ≥ 700` | 3.167.955 | Classificação binária, mais fácil de apresentar |

**Por que cinco alvos:** a nota tem duas naturezas misturadas. Os 165 mil zeros vêm de redações com
problema de forma, e não de desempenho na escrita. Separar `y_zerada` de `y_nota_ok` evita que o modelo
gaste capacidade tentando explicar zeros com renda e escolaridade. `y_fez_redacao` existe porque a
ausência não é aleatória: quem já concluiu o EM falta muito mais, então quem some da base de notas é
sistematicamente diferente de quem fica.

---

## 2. Renda

| Feature | Definição | Motivo |
|---|---|---|
| `renda_familiar` | Ponto médio em R$ da faixa da Q007 (A=0 … Q=33.888) | Transforma a faixa em valor numérico, permitindo razões e logaritmos. A última faixa é aberta, e usamos 1,2× o limite inferior |
| `renda_pc` | `renda_familiar / pessoas_casa` | Renda per capita separa "R$ 4.000 para 2 pessoas" de "R$ 4.000 para 7 pessoas" |
| `log_renda_pc` | `log(1 + renda_pc)` | O efeito da renda é decrescente: R$ 500 a mais pesa muito na base e pouco no topo. O log lineariza essa relação |
| `renda_sm` | Renda em salários mínimos de 2024 (R$ 1.412) | Mesma escala usada nos editais e comparável entre anos, já que as faixas do ENEM seguem o salário mínimo |
| `sem_renda` | 1 se a família declarou nenhuma renda | Grupo de 261 mil pessoas com média 100 pontos abaixo; o zero em escala de log merece uma marca própria |
| `tem_renda` | Q006: o próprio participante tem renda | Indica trabalho, que concorre com o tempo de estudo |

---

## 3. Capital cultural dos pais

| Feature | Definição | Motivo |
|---|---|---|
| `esc_pai`, `esc_mae` | Escolaridade da Q001/Q002 em escala 0–6 | As categorias são naturalmente ordenadas |
| `esc_max_pais` | Maior escolaridade entre os dois | Costuma ser o melhor resumo isolado do capital cultural do domicílio |
| `esc_media_pais` | Média dos dois | Captura o caso em que só um dos pais estudou |
| `pais_superior` | 1 se algum tem ensino superior completo | Limiar com salto grande na nota (média 718 contra 499 no extremo oposto) |
| `nao_sabe_esc` | 1 se respondeu "não sei" (opção H) | "Não sei" não é escolaridade média: costuma indicar ausência do pai/mãe, e é informativo por si só |
| `ocup_pai`, `ocup_mae` | Grupo ocupacional da Q003/Q004 (1–5) | Proxy de classe ocupacional, complementar à escolaridade |
| `ocup_max_pais` | Maior entre os dois | Mesma lógica de `esc_max_pais` |
| `nao_sabe_ocup` | 1 se respondeu "não sei" | Mesmo raciocínio do `nao_sabe_esc` |

---

## 4. Bens e condições do domicílio

| Feature | Definição | Motivo |
|---|---|---|
| `n_q008` … `n_q022` | 15 itens do questionário em escala numérica (0–3, 0–4 ou 0/1) | Guarda o detalhe item a item para os modelos de árvore |
| `indice_bens` | Média dos z-scores dos 15 itens | Resume a posse de bens em uma variável estável e reduz a redundância entre itens altamente correlacionados |
| `comodos_por_pessoa` | `(banheiros + quartos) / moradores` | Mede densidade domiciliar, um indicador clássico de condição de estudo em casa |
| `computador_por_pessoa` | `computadores / moradores` | Um computador dividido por 6 pessoas não é o mesmo que um por pessoa |
| `celular_por_pessoa` | `celulares / moradores` | Mesma lógica, para acesso individual |
| `tem_computador` | 1 se há ao menos um computador | Limiar simples e forte: correlação de 0,26 com a nota |
| `acesso_digital` | 1 se há computador **e** wi-fi | Interação explícita: computador sem internet é bem menos útil para estudar |

---

## 5. Índice socioeconômico composto

| Feature | Definição | Motivo |
|---|---|---|
| `inse_proxy` | Média dos z-scores de `log_renda_pc`, `esc_max_pais` e `indice_bens` | Inspirado no INSE do INEP: junta renda, escolaridade e bens em um só número. É a feature com maior correlação isolada com a nota (0,32) e serve de resumo interpretável para gráficos e relatório |

---

## 6. Escola

| Feature | Definição | Motivo |
|---|---|---|
| `escola_publica` | Q023 = A (só pública) | A Q023 não é ordinal (D e E não são "mais" que B), por isso viram indicadores |
| `escola_privada` | Q023 ∈ {D, E} | Diferença de 170 pontos para a pública, e ela persiste dentro da mesma faixa de renda |
| `escola_mista` | Q023 ∈ {B, C} | Trajetória mista tem média intermediária |
| `bolsista` | Q023 ∈ {C, E} | Separa quem está na privada por bolsa de quem paga; perfis socioeconômicos bem diferentes |
| `sem_ensino_medio` | Q023 = F | Grupo pequeno (0,3%) e atípico |

---

## 7. Trajetória escolar e demografia

| Feature | Definição | Motivo |
|---|---|---|
| `idade` | Ponto médio da faixa etária | Converte 20 faixas em uma variável contínua |
| `atraso_idade` | `idade − 17` para quem conclui em 2024 | Mede defasagem série-idade, associada a repetência. Fica ausente para os demais, e a flag marca isso |
| `anos_desde_conclusao` | Anos desde a conclusão do EM (0 para quem ainda cursa) | Distância do conteúdo escolar; treineiros e recém-formados não são comparáveis |
| `treineiro` | Fez a prova só para treinar | Comparece mais e tem motivação diferente |
| `concluinte`, `ja_concluiu` | Situação do ensino médio | Os dois grupos têm taxas de presença muito diferentes (80% contra 61%) |
| `sexo_fem` | 1 se feminino | Mulheres têm média 45 pontos maior na redação |
| `cor_raca` | Categórica com 7 níveis | Mantida como categoria porque não há ordem. Serve para análise de equidade, não como variável causal |

---

## 8. Geografia

| Feature | Definição | Motivo |
|---|---|---|
| `uf` | UF do local da prova (categórica) | Diferença de 124 pontos entre a maior e a menor média |
| `regiao` | UF agregada em 5 regiões | Versão com menos níveis, útil quando a UF fragmenta demais os grupos |
| `log_porte_municipio` | Log do número de inscritos no município da prova | Proxy de porte do município (capital, cidade média, interior). Usa só contagem de linhas, sem tocar no alvo |
| `te_municipio` | Nota média do município, suavizada | *Target encoding*: resume centenas de municípios em um número. Suavização bayesiana com m = 200 puxa municípios pequenos para a média geral, evitando que 5 alunos definam o valor |
| `te_uf` | Nota média da UF, suavizada | Mesma ideia, em nível mais agregado |

⚠️ **As duas últimas são calculadas apenas com as linhas de treino.** Se o split mudar, o passo 6 do
notebook precisa ser refeito. Em validação cruzada, o ideal é recalcular dentro de cada fold.

---

## 9. Tratamento de ausentes

Regra: só imputamos onde o vazio **não** tem significado.

| Situação | Tratamento |
|---|---|
| Notas ausentes (26,9%) | Não imputadas — a linha simplesmente não entra no alvo de regressão |
| Escolaridade/ocupação com "não sei" | Imputadas pela **mediana do treino** + flag `*_ausente` |
| `atraso_idade` (só faz sentido para concluintes) | Imputada pela mediana do treino + flag `atraso_idade_ausente` |
| Demais features | Sem ausentes |

As medianas vêm **só do conjunto de treino**, para que nenhuma informação do teste vaze para o pré-processamento.

---

## 10. Split e codificações

- **Split:** 80/20, estratificado por UF, com semente fixa (`SEED = 42`). A coluna `split` (`treino`/`teste`)
  viaja junto com a base, então todos do grupo usam exatamente a mesma divisão.
- **Ordinal** (`enem2024_features_ordinal.parquet`): todas as categóricas viram números ordenados.
  Formato para árvores de decisão, random forest, LightGBM e XGBoost, que não precisam de one-hot.
- **One-hot** (`enem2024_features_amostra_onehot.parquet`): `cor_raca`, `regiao` e `uf` expandidas em
  indicadores, com `drop_first=True` para evitar colinearidade perfeita. Formato para regressão linear
  e logística. Gerado sobre uma amostra de 500 mil linhas, porque a base cheia em one-hot ficaria
  grande demais para o Colab; para o modelo final, é só regerar com a base inteira.

---

## 11. Limitações conhecidas

1. **Sem dados da escola real:** em 2024 não dá para ligar `PARTICIPANTES` a `RESULTADOS`, então o tipo
   de escola vem da declaração do participante (Q023), e não do Censo Escolar.
2. **Features redundantes:** renda, bens, escolaridade dos pais e `inse_proxy` medem, em boa parte, a
   mesma coisa. Para modelos lineares, vale escolher um subconjunto ou usar regularização.
3. **Pontos médios de faixa:** `renda_familiar` e `idade` são aproximações da faixa declarada, não valores reais.
4. **Associação, não causa:** nenhuma dessas features explica *por que* a nota é diferente; elas descrevem
   com quem a nota está associada.
5. **Target encoding:** reintroduz o alvo no conjunto de features. Se for usada validação cruzada, recalcule
   `te_municipio` e `te_uf` dentro de cada fold para não superestimar o desempenho.
