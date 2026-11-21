"""App do G5 · INF1032 — nota prevista da redação do ENEM 2024 a partir do perfil.

Rodar localmente:   streamlit run app.py
Precisa, na mesma pasta: modelo_redacao_g5.joblib (seção 15.1 do notebook), preparo.py e perfil.py.
"""

from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st

import joblib
from perfil import PORTES, municipio_representativo, montar_entrada

PASTA = Path(__file__).parent
MEDIA_NACIONAL = 659   # média das redações corrigidas sem problema de status (seção 2.6)

st.set_page_config(page_title="Redação do ENEM × perfil · G5", page_icon="📝", layout="centered")


@st.cache_resource
def carregar():
    return joblib.load(PASTA / "modelo_redacao_g5.joblib")


pacote = carregar()
pipeline = pacote["pipeline"]
mae = pacote["metricas_teste"]["MAE"]
c_inter, c_comp = pacote["cortes_sisu"]["Intermediária"], pacote["cortes_sisu"]["Competitiva"]

if sklearn.__version__ != pacote["versoes"]["scikit-learn"]:
    st.warning(f"O modelo foi salvo com scikit-learn {pacote['versoes']['scikit-learn']} e este "
               f"ambiente tem {sklearn.__version__}. Fixe a mesma versão no requirements.txt.")

# ----------------------------------------------------------------- cabeçalho
st.title("📝 Redação do ENEM 2024 × perfil socioeconômico")
st.caption("INF1032 · Introdução à Ciência de Dados · PUC-Rio · Turma 3WA · Grupo 5")
st.info("**O que este app faz:** mostra a nota de redação que o modelo do grupo **associa** a um "
        "perfil de participante do ENEM 2024, a partir das respostas do questionário "
        "socioeconômico. Serve para entender **grupos**, não para prever a nota de uma pessoa: o "
        f"modelo erra, em média, **{mae:.0f} pontos** para mais ou para menos. "
        "As respostas não são gravadas.", icon="ℹ️")

# ----------------------------------------------------------------- opções
UFS = sorted(["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA",
              "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"])
FAIXA_ETARIA = {"1": "Menos de 17 anos", "2": "17 anos", "3": "18 anos", "4": "19 anos",
                "5": "20 anos", "6": "21 anos", "7": "22 anos", "8": "23 anos", "9": "24 anos",
                "10": "25 anos", "11": "26 a 30 anos", "12": "31 a 35 anos", "13": "36 a 40 anos",
                "14": "41 a 45 anos", "15": "46 a 50 anos", "16": "51 a 55 anos",
                "17": "56 a 60 anos", "18": "61 a 65 anos", "19": "66 a 70 anos",
                "20": "Mais de 70 anos"}
CONCLUSAO = {"2": "Estou cursando e concluo o Ensino Médio em 2024",
             "1": "Já concluí o Ensino Médio",
             "3": "Estou cursando e concluo depois de 2024",
             "4": "Não concluí e não estou cursando"}
ANO_CONCLUIU = {str(i): str(2024 - i) for i in range(1, 18)} | {"18": "Antes de 2007"}
ESCOLA = {"A": "Somente em escola pública",
          "B": "Parte pública, parte privada sem bolsa integral",
          "C": "Parte pública, parte privada com bolsa integral",
          "D": "Somente em escola privada sem bolsa integral",
          "E": "Somente em escola privada com bolsa integral",
          "F": "Não frequentei o Ensino Médio"}
ESCOLARIDADE = {"A": "Nunca estudou", "B": "Não completou o 5º ano do Fundamental",
                "C": "Completou o 5º ano, mas não o 9º", "D": "Completou o 9º ano, mas não o Médio",
                "E": "Completou o Médio, mas não a faculdade",
                "F": "Completou a faculdade, mas não a pós", "G": "Completou a pós-graduação",
                "H": "Não sei"}
OCUPACAO = {"A": "Grupo 1 — lavoura, criação de animais, pesca, extrativismo",
            "B": "Grupo 2 — serviços domésticos, vendedor, auxiliar, motorista particular…",
            "C": "Grupo 3 — ofícios: padeiro, mecânico, eletricista, motorista profissional…",
            "D": "Grupo 4 — professor do básico, técnico, policial, pequeno empresário…",
            "E": "Grupo 5 — médico, engenheiro, advogado, professor universitário, gerente…",
            "F": "Não sei"}
RENDA = {"A": "Nenhuma renda", "B": "Até R$ 1.412", "C": "R$ 1.412 a R$ 2.118",
         "D": "R$ 2.118 a R$ 2.824", "E": "R$ 2.824 a R$ 3.530", "F": "R$ 3.530 a R$ 4.236",
         "G": "R$ 4.236 a R$ 5.648", "H": "R$ 5.648 a R$ 7.060", "I": "R$ 7.060 a R$ 8.472",
         "J": "R$ 8.472 a R$ 9.884", "K": "R$ 9.884 a R$ 11.296", "L": "R$ 11.296 a R$ 12.708",
         "M": "R$ 12.708 a R$ 14.120", "N": "R$ 14.120 a R$ 16.944", "O": "R$ 16.944 a R$ 21.180",
         "P": "R$ 21.180 a R$ 28.240", "Q": "Acima de R$ 28.240"}
QTD_4 = {"A": "Não", "B": "Um(a)", "C": "Dois/duas", "D": "Três ou mais"}
QTD_5 = {"A": "Não", "B": "Um", "C": "Dois", "D": "Três", "E": "Quatro ou mais"}
SIM_NAO = {"A": "Não", "B": "Sim"}


def escolha(rotulo, opcoes, padrao=None, **kw):
    chaves = list(opcoes)
    return st.selectbox(rotulo, chaves, index=chaves.index(padrao) if padrao else 0,
                        format_func=opcoes.get, **kw)


aba_nota, aba_mun = st.tabs(["📝 Nota pelo perfil", "📍 Municípios priorizados"])

with aba_nota:
    # ----------------------------------------------------------------- formulário
    with st.form("perfil"):
        st.subheader("Você")
        c1, c2 = st.columns(2)
        with c1:
            faixa = escolha("Idade", FAIXA_ETARIA, "3")
            sexo = st.radio("Sexo (como no cadastro do ENEM)", ["F", "M"], horizontal=True,
                            format_func={"F": "Feminino", "M": "Masculino"}.get)
        with c2:
            uf = st.selectbox("UF onde fez a prova", UFS, index=UFS.index("RJ"))
            porte = st.selectbox("Cidade onde fez a prova", list(PORTES),
                                 help="O modelo usa o porte da cidade (quantos participantes fizeram "
                                      "a prova lá), não o nome dela.")

        situacao = escolha("Ensino Médio", CONCLUSAO, "2")
        c1, c2 = st.columns(2)
        with c1:
            ano = escolha("Se já concluiu: em que ano?", ANO_CONCLUIU, "1",
                          help="Só conta se você marcou que já concluiu o Ensino Médio.")
        with c2:
            treineiro = st.checkbox("Fiz a prova só para treinar (treineiro)")
        escola = escolha("Em que tipo de escola você fez o Ensino Médio?", ESCOLA, "A")

        st.subheader("Família")
        c1, c2 = st.columns(2)
        with c1:
            esc_pai = escolha("Até onde seu pai (ou responsável) estudou?", ESCOLARIDADE, "E")
            ocup_pai = escolha("Ocupação do pai (ou responsável)", OCUPACAO, "C")
        with c2:
            esc_mae = escolha("Até onde sua mãe (ou responsável) estudou?", ESCOLARIDADE, "E")
            ocup_mae = escolha("Ocupação da mãe (ou responsável)", OCUPACAO, "C")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            pessoas = st.number_input("Pessoas na casa (com você)", 1, 20, 4)
        with c2:
            renda = escolha("Renda mensal da família", RENDA, "D")
        with c3:
            tem_renda = escolha("Você tem renda?", SIM_NAO, "A")

        st.subheader("Casa")
        c1, c2, c3 = st.columns(3)
        with c1:
            q009 = escolha("Banheiros", QTD_4, "B")
            q010 = escolha("Quartos", QTD_4, "C")
            q018 = escolha("TVs", QTD_4, "B")
            q008 = escolha("Empregada doméstica", {"A": "Não", "B": "1–2 dias/sem.",
                                                   "C": "3–4 dias/sem.", "D": "5+ dias/sem."}, "A")
        with c2:
            q011 = escolha("Carros", QTD_4, "A")
            q012 = escolha("Motos", QTD_4, "A")
            q013 = escolha("Geladeiras", QTD_4, "B")
            q021 = escolha("Computadores/notebooks", QTD_5, "A")
        with c3:
            q022 = escolha("Celulares na casa", QTD_5, "D")
            q020 = escolha("Internet wi-fi", SIM_NAO, "B")
            q019 = escolha("TV por assinatura", SIM_NAO, "A")
        c1, c2, c3, c4 = st.columns(4)
        q014 = c1.selectbox("Freezer", list(SIM_NAO), format_func=SIM_NAO.get)
        q015 = c2.selectbox("Máq. de lavar", list(SIM_NAO), index=1, format_func=SIM_NAO.get)
        q016 = c3.selectbox("Micro-ondas", list(SIM_NAO), format_func=SIM_NAO.get)
        q017 = c4.selectbox("Aspirador", list(SIM_NAO), format_func=SIM_NAO.get)

        enviar = st.form_submit_button("Estimar nota", type="primary", width="stretch")

    # ----------------------------------------------------------------- resultado
    if enviar:
        prop = pipeline.named_steps["preparo"].prop_municipio_
        respostas = {
            "TP_FAIXA_ETARIA": faixa, "TP_SEXO": sexo, "TP_ST_CONCLUSAO": situacao,
            "TP_ANO_CONCLUIU": ano if situacao == "1" else "0", "IN_TREINEIRO": "1" if treineiro else "0",
            "SG_UF_PROVA": uf, "CO_MUNICIPIO_PROVA": municipio_representativo(prop, uf, porte),
            "Q001": esc_pai, "Q002": esc_mae, "Q003": ocup_pai, "Q004": ocup_mae, "Q005": pessoas,
            "Q006": tem_renda, "Q007": renda, "Q023": escola,
            "Q008": q008, "Q009": q009, "Q010": q010, "Q011": q011, "Q012": q012, "Q013": q013,
            "Q018": q018, "Q021": q021, "Q022": q022,
            "Q014": q014, "Q015": q015, "Q016": q016, "Q017": q017, "Q019": q019, "Q020": q020,
        }
        linha = montar_entrada(respostas, pacote["features"])
        nota = float(pipeline.predict(linha)[0])
        classe = "Competitiva" if nota >= c_comp else "Intermediária" if nota >= c_inter else "Baixa"

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Nota associada ao perfil", f"{nota:.0f}", f"{nota - MEDIA_NACIONAL:+.0f} vs. média",
                  help=f"Média nacional das redações corrigidas sem problema: {MEDIA_NACIONAL}.")
        c2.metric("Faixa do erro típico", f"{max(nota - mae, 0):.0f} – {min(nota + mae, 1000):.0f}",
                  help=f"Nota prevista ± o erro médio do modelo no teste ({mae:.0f} pontos).")
        c3.metric("Classe do Sisu", classe,
                  help=f"Baixa: abaixo de {c_inter:.0f} · Intermediária: {c_inter:.0f} a "
                       f"{c_comp - 1:.0f} · Competitiva: {c_comp:.0f} ou mais.")

        st.warning(
            "**Como ler.** Este número é a nota **média** que o modelo associa a pessoas com um perfil "
            "parecido no ENEM 2024 — não a nota que você vai tirar. A previsão tende a ficar perto da "
            "média: quem escreve muito bem quase nunca aparece como *Competitiva*, mesmo sendo. "
            "Associação não é causa: mudar uma resposta não muda a sua redação.", icon="⚠️")

        with st.expander("O que o modelo recebeu"):
            st.caption("Colunas de entrada do pipeline, depois da seção 3. `cor_raca` vai como "
                       "não declarada: o modelo não a usa.")
            st.dataframe(linha.T.rename(columns={0: "valor"}).astype(str), width="stretch")

# ----------------------------------------------------------------- municípios (seção 16)
with aba_mun:
    caminho_lista = PASTA / "municipios_priorizados.csv"
    if not caminho_lista.exists():
        st.info("A lista ainda não foi gerada: rode a seção 16.2 do notebook.")
    else:
        lista = pd.read_csv(caminho_lista)
        n_prior = int(lista["priorizado"].sum())
        st.markdown(
            f"**A decisão da seção 16 do notebook.** Um programa de reforço em redação com orçamento para "
            f"atender **{n_prior / len(lista):.0%} dos municípios** ordena os municípios pela nota média que "
            f"o modelo **prevê** a partir do perfil dos participantes e atende os de nota prevista mais baixa. "
            f"Aqui estão os **{len(lista)} municípios** do conjunto de teste com pelo menos 50 participantes; "
            f"**{n_prior}** ficam na lista de prioridade.")
        acertos = lista.loc[lista["priorizado"], "precisa"].mean()
        c1, c2 = st.columns(2)
        c1.metric("Priorizados que de fato precisavam", f"{acertos:.0%}",
                  help="Precisão: municípios da lista com nota média real no quartil de baixo.")
        c2.metric("Dos que precisavam, quantos entraram",
                  f"{lista.loc[lista['precisa'], 'priorizado'].mean():.0%}",
                  help="Recall: entre os municípios com nota média real no quartil de baixo.")
        c1, c2 = st.columns([1, 2])
        with c1:
            uf_lista = st.selectbox("UF", ["Todas"] + sorted(lista["uf"].unique()))
        with c2:
            so_prior = st.toggle("Só os priorizados", value=True)
        ver = lista if uf_lista == "Todas" else lista[lista["uf"] == uf_lista]
        if so_prior:
            ver = ver[ver["priorizado"]]
        st.dataframe(
            ver.rename(columns={"ordem": "ordem", "nome": "município", "uf": "UF",
                                "participantes": "participantes (teste)",
                                "previsto": "nota média prevista", "real": "nota média real",
                                "priorizado": "priorizado", "precisa": "precisava de apoio"})
               .drop(columns="municipio"),
            hide_index=True, width="stretch")
        st.caption("Município = onde a pessoa fez a prova, não onde mora. \"Precisava de apoio\" = nota "
                   "média real abaixo do 1º quartil das médias municipais do treino. A lista orienta "
                   "**onde** levar apoio a grupos; ela não diz nada sobre nenhuma pessoa.")

# ----------------------------------------------------------------- rodapé
with st.expander("Sobre o modelo"):
    m = pacote["metricas_teste"]
    n_treino = pacote.get("linhas_treino")
    amostra_txt = f"{n_treino / 1000:.0f} mil" if n_treino else "uma amostra de"
    st.markdown(
        f"- **Modelo:** {pacote['nome']} (`HistGradientBoostingRegressor` com tuning), treinado "
        f"em {amostra_txt} redações do ENEM 2024 corrigidas sem problema de status.\n"
        f"- **No conjunto de teste** (600 mil redações que o modelo não viu): MAE "
        f"{m['MAE']:.1f} pontos, RMSE {m['RMSE']:.1f}, R² {m['R2']:.3f}.\n"
        f"- **Cor/raça não entra no modelo**; ela só foi usada para auditar o erro por grupo.\n"
        f"- **Não use para:** selecionar ou ranquear alunos, avaliar escolas, prever outros anos.\n"
        f"- Gerado em {pacote['gerado_em']} · scikit-learn {pacote['versoes']['scikit-learn']}.")
    st.caption(pacote["aviso"])
