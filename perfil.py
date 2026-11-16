"""Do questionário do ENEM às colunas de entrada do modelo (uma pessoa por vez).

Repete, para uma linha, as transformações "linha a linha" da seção 3 do notebook
(`INF1032_G5_projeto.ipynb`). As transformações que olham o conjunto (porte do
município, winsorização, imputação, índices por z-score) NÃO estão aqui: elas
fazem parte do pipeline salvo no `.joblib` e foram aprendidas só no treino.

As respostas usam os mesmos códigos do dicionário dos microdados do ENEM 2024
(letras do questionário, códigos de faixa etária etc.), para que a seção 15.2 do
notebook possa conferir este arquivo contra a base de verdade.

Se a seção 3 do notebook mudar, este arquivo tem de mudar junto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ seção 3.3
ESCOLARIDADE = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}   # H = não sei
OCUPACAO = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}                       # F = não sei
BIN = {"A": 0, "B": 1}
QTD_4 = {"A": 0, "B": 1, "C": 2, "D": 3}
QTD_5 = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
RENDA_MEIO = {"A": 0, "B": 706, "C": 1765, "D": 2471, "E": 3177, "F": 3883, "G": 4942, "H": 6354,
              "I": 7766, "J": 9178, "K": 10590, "L": 12002, "M": 13414, "N": 15532, "O": 19062,
              "P": 24710, "Q": 33888}
IDADE_MEIO = {"1": 16, "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23, "9": 24,
              "10": 25, "11": 28, "12": 33, "13": 38, "14": 43, "15": 48, "16": 53, "17": 58,
              "18": 63, "19": 68, "20": 73}
ANOS_CONCL = {str(k): k for k in range(1, 18)} | {"18": 19}                # 0 = não informado
BENS_QTD = {"Q008": QTD_4, "Q009": QTD_4, "Q010": QTD_4, "Q011": QTD_4, "Q012": QTD_4,
            "Q013": QTD_4, "Q018": QTD_4, "Q021": QTD_5, "Q022": QTD_5}
BENS_BIN = {"Q014": BIN, "Q015": BIN, "Q016": BIN, "Q017": BIN, "Q019": BIN, "Q020": BIN}
SM_2024 = 1412.0

REGIAO = {**{uf: "Norte" for uf in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
          **{uf: "Nordeste" for uf in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
          **{uf: "Centro-Oeste" for uf in ["DF", "GO", "MT", "MS"]},
          **{uf: "Sudeste" for uf in ["ES", "MG", "RJ", "SP"]},
          **{uf: "Sul" for uf in ["PR", "RS", "SC"]}}

# código IBGE da UF = 2 primeiros dígitos do código do município
UF_IBGE = {"RO": 11, "AC": 12, "AM": 13, "RR": 14, "PA": 15, "AP": 16, "TO": 17, "MA": 21,
           "PI": 22, "CE": 23, "RN": 24, "PB": 25, "PE": 26, "AL": 27, "SE": 28, "BA": 29,
           "MG": 31, "ES": 32, "RJ": 33, "SP": 35, "PR": 41, "SC": 42, "RS": 43, "MS": 50,
           "MT": 51, "GO": 52, "DF": 53}

COR_RACA = {"0": "nao_declarado", "1": "branca", "2": "preta", "3": "parda", "4": "amarela",
            "5": "indigena", "6": "sem_info"}

#: campos de entrada, com os códigos do dicionário do INEP
CAMPOS = (["TP_FAIXA_ETARIA", "TP_SEXO", "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "IN_TREINEIRO",
           "SG_UF_PROVA", "CO_MUNICIPIO_PROVA", "Q001", "Q002", "Q003", "Q004", "Q005", "Q006",
           "Q007", "Q023"] + list(BENS_QTD) + list(BENS_BIN))


def _num(dic, codigo):
    v = dic.get(str(codigo))
    return np.nan if v is None else float(v)


def montar_entrada(r: dict, colunas: list[str]) -> pd.DataFrame:
    """Respostas de uma pessoa (códigos do INEP) -> DataFrame de 1 linha nas `colunas` do modelo.

    `TP_COR_RACA` é opcional: o modelo não usa cor/raça (ela só serve à auditoria de equidade);
    sem ela, a coluna vai como "nao_declarado".
    """
    faltando = [c for c in CAMPOS if c not in r]
    if faltando:
        raise ValueError(f"faltam respostas: {faltando}")
    f = {}

    # bens (contagens e sim/não)
    for q, dic in {**BENS_QTD, **BENS_BIN}.items():
        f[f"n_{q.lower()}"] = _num(dic, r[q])

    # escolaridade, ocupação, casa e renda
    f["esc_pai"], f["esc_mae"] = _num(ESCOLARIDADE, r["Q001"]), _num(ESCOLARIDADE, r["Q002"])
    f["ocup_pai"], f["ocup_mae"] = _num(OCUPACAO, r["Q003"]), _num(OCUPACAO, r["Q004"])
    f["pessoas_casa"] = float(r["Q005"])
    f["tem_renda"] = _num(BIN, r["Q006"])
    f["renda_familiar"] = _num(RENDA_MEIO, r["Q007"])
    f["idade"] = _num(IDADE_MEIO, r["TP_FAIXA_ETARIA"])
    anos = _num(ANOS_CONCL, r["TP_ANO_CONCLUIU"])

    # seção 3.4: razões e flags
    f["renda_pc"] = f["renda_familiar"] / f["pessoas_casa"]
    f["log_renda_pc"] = float(np.log1p(f["renda_pc"]))
    f["renda_sm"] = f["renda_familiar"] / SM_2024
    f["sem_renda"] = float(f["renda_familiar"] == 0)

    pais = pd.Series([f["esc_pai"], f["esc_mae"]], dtype="float64")
    f["esc_max_pais"] = pais.max()                       # ignora NaN, como df.max(axis=1)
    f["esc_media_pais"] = pais.mean()
    f["pais_superior"] = float(f["esc_max_pais"] >= 5)    # NaN >= 5 é False, como no notebook
    f["nao_sabe_esc"] = float(r["Q001"] == "H" or r["Q002"] == "H")
    f["ocup_max_pais"] = pd.Series([f["ocup_pai"], f["ocup_mae"]], dtype="float64").max()
    f["nao_sabe_ocup"] = float(r["Q003"] == "F" or r["Q004"] == "F")

    f["comodos_por_pessoa"] = (f["n_q009"] + f["n_q010"]) / f["pessoas_casa"]
    f["computador_por_pessoa"] = f["n_q021"] / f["pessoas_casa"]
    f["celular_por_pessoa"] = f["n_q022"] / f["pessoas_casa"]
    f["tem_computador"] = float(f["n_q021"] > 0)
    f["acesso_digital"] = float(f["n_q021"] > 0 and f["n_q020"] > 0)

    q23 = r["Q023"]
    f["escola_publica"] = float(q23 == "A")
    f["escola_privada"] = float(q23 in ("D", "E"))
    f["escola_mista"] = float(q23 in ("B", "C"))
    f["bolsista"] = float(q23 in ("C", "E"))
    f["sem_ensino_medio"] = float(q23 == "F")

    f["treineiro"] = float(int(r["IN_TREINEIRO"]))
    f["concluinte"] = float(str(r["TP_ST_CONCLUSAO"]) == "2")
    f["ja_concluiu"] = float(str(r["TP_ST_CONCLUSAO"]) == "1")
    f["atraso_idade"] = f["idade"] - 17 if f["concluinte"] == 1 else np.nan
    f["anos_desde_conclusao"] = 0.0 if np.isnan(anos) else anos

    f["sexo_fem"] = float(r["TP_SEXO"] == "F")
    f["cor_raca"] = COR_RACA.get(str(r.get("TP_COR_RACA", "0")), "nao_declarado")
    f["regiao"] = REGIAO[r["SG_UF_PROVA"]]
    f["uf"] = r["SG_UF_PROVA"]
    f["CO_MUNICIPIO_PROVA"] = int(r["CO_MUNICIPIO_PROVA"])

    return pd.DataFrame([f])[colunas]


# ------------------------------------------------------------------ município
PORTES = {"Maior cidade do estado": 0.0,
          "Cidade de porte médio": 0.5,
          "Cidade pequena": 0.9}


def municipio_representativo(prop_municipio: pd.Series, uf: str, porte: str) -> int:
    """Código de um município da UF que represente o porte escolhido.

    O app não pergunta o município (são mais de 5 mil, e o modelo só usa o porte dele: a
    proporção de participantes que fez a prova ali, aprendida no treino). Ordenamos os
    municípios da UF do maior para o menor e pegamos aquele em que a soma acumulada de
    participantes passa da fração do `porte`: 0 = o maior, 0,5 = o do meio da distribuição
    de participantes, 0,9 = um dos menores.
    """
    prefixo = UF_IBGE[uf]
    da_uf = prop_municipio[prop_municipio.index.astype(int) // 100000 == prefixo]
    da_uf = da_uf.sort_values(ascending=False)
    acumulada = da_uf.cumsum() / da_uf.sum()
    alvo = PORTES[porte]
    i = int((acumulada <= alvo).sum())
    return int(da_uf.index[min(i, len(da_uf) - 1)])
