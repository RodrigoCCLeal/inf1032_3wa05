"""Preparo ajustável da base do ENEM 2024.

Separa a engenharia de features em duas metades:

* **linha a linha** (seção 3 do notebook) — razões, flags, codificação. Cada linha
  é calculada só com os próprios valores, então não há nada para "aprender": pode
  rodar uma vez e ser gravada no parquet.
* **ajustável** (este arquivo) — porte do município, winsorização, imputação e os
  dois índices por z-score (`indice_bens` e `inse_proxy`). Essas transformações
  olham o *conjunto* de linhas (uma proporção, um percentil, uma mediana, uma média
  e um desvio). Se forem calculadas na base inteira, as linhas de validação — e as
  de teste — ajudam a calcular o número que depois é aplicado a elas mesmas: é
  vazamento, o mesmo que a seção 8.1 mede nos target encodings.

Por isso o preparo é um transformador do scikit-learn, com `fit` e `transform`
separados: dentro de um `Pipeline`, a validação cruzada chama o `fit` **só** com
as linhas de treino de cada rodada.

Uso::

    from sklearn.pipeline import Pipeline
    from preparo import preparo, NUMERICAS, CAT_MODELO

    Pipeline([("preparo", preparo()),
              ("colunas", ColumnTransformer(...)),
              ("modelo", LinearRegression())])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------- #
# colunas
# --------------------------------------------------------------------------- #

COL_MUNICIPIO = "CO_MUNICIPIO_PROVA"

BENS = ["n_q008", "n_q009", "n_q010", "n_q011", "n_q012", "n_q013", "n_q018",
        "n_q021", "n_q022", "n_q014", "n_q015", "n_q016", "n_q017", "n_q019", "n_q020"]

#: numéricas que já chegam prontas da seção 3 (linha a linha)
NUM_ENTRADA = [
    "renda_familiar", "renda_pc", "log_renda_pc", "renda_sm", "sem_renda", "tem_renda",
    "pessoas_casa", "esc_pai", "esc_mae", "esc_max_pais", "esc_media_pais",
    "pais_superior", "nao_sabe_esc",
    "ocup_pai", "ocup_mae", "ocup_max_pais", "nao_sabe_ocup",
    "comodos_por_pessoa", "computador_por_pessoa",
    "celular_por_pessoa", "tem_computador", "acesso_digital",
    "escola_publica", "escola_privada", "escola_mista", "bolsista", "sem_ensino_medio",
    "idade", "atraso_idade", "anos_desde_conclusao", "treineiro", "concluinte", "ja_concluiu",
    "sexo_fem",
] + BENS

#: target encodings da seção 8.1; entram só quando `target_encodings=True` (demonstração de vazamento)
TE = ["te_municipio", "te_uf"]

#: categóricas que passam pelo preparo
CATEGORICAS = ["cor_raca", "regiao", "uf"]

#: categóricas que entram no MODELO. `cor_raca` passa pelo preparo só para a auditoria de
#: equidade (erro por grupo) e não é feature; `regiao` é função exata da UF.
CAT_MODELO = ["uf"]

# quocientes com denominador pequeno estouram: winsorizamos em vez de remover a linha
WINSOR = ["renda_familiar", "renda_pc", "log_renda_pc", "renda_sm",
          "comodos_por_pessoa", "computador_por_pessoa", "celular_por_pessoa"]
P_BAIXO, P_ALTO = 0.005, 0.995

# NaN sem significado próprio: imputamos pela mediana e guardamos a flag
AUSENTES = ["esc_pai", "esc_mae", "esc_max_pais", "esc_media_pais",
            "ocup_pai", "ocup_mae", "ocup_max_pais", "atraso_idade"]

# componentes do índice socioeconômico composto (proxy do INSE do INEP)
INSE = ["log_renda_pc", "esc_max_pais", "indice_bens"]

#: criadas aqui dentro
DERIVADAS = (["prop_municipio", "outlier_renda", "indice_bens", "inse_proxy"]
             + [f"{c}_ausente" for c in AUSENTES])

#: numéricas na saída do preparo (sem os target encodings)
NUMERICAS = NUM_ENTRADA + DERIVADAS

#: tudo o que o preparo entrega, na ordem
COLUNAS = NUMERICAS + CATEGORICAS

#: colunas que o preparo exige na entrada
ENTRADA = NUM_ENTRADA + CATEGORICAS + [COL_MUNICIPIO]


# --------------------------------------------------------------------------- #
# transformador
# --------------------------------------------------------------------------- #

class Preparo(BaseEstimator, TransformerMixin):
    """Porte do município, winsorização, imputação e índices, ajustados só no treino.

    Parameters
    ----------
    target_encodings : bool, default False
        Se ``True``, deixa ``te_municipio`` e ``te_uf`` passarem para a saída.
        Eles foram calculados fora do pipeline, na base de treino inteira, e
        vazam na validação cruzada — só servem para *medir* esse vazamento.
    """

    def __init__(self, target_encodings: bool = False):
        self.target_encodings = target_encodings

    # -- ajuste ------------------------------------------------------------- #
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self._conferir(X)

        # porte do município como PROPORÇÃO das linhas de treino que fizeram prova nele.
        # Proporção e não contagem: a contagem depende do tamanho da amostra, então o
        # mesmo município vale 12.000 na base cheia e 900 numa amostra de 50 mil — e o
        # número muda de rodada para rodada da CV só porque a dobra é menor.
        self.prop_municipio_ = X[COL_MUNICIPIO].value_counts(normalize=True)

        self.limites_ = X[WINSOR].astype("float64").quantile([P_BAIXO, P_ALTO])

        # medianas depois do corte, para que um extremo não puxe o valor imputado
        Xw = self._winsorizar(X)
        self.medianas_ = Xw[NUM_ENTRADA].astype("float64").median()

        # média e desvio dos índices, sobre o treino já cortado e imputado
        Xi = self._imputar(Xw)
        self.bens_media_, self.bens_dp_ = self._media_dp(Xi[BENS])
        Xi["indice_bens"] = self._media_z(Xi, BENS, self.bens_media_, self.bens_dp_)
        self.inse_media_, self.inse_dp_ = self._media_dp(Xi[INSE])

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    # -- aplicação ---------------------------------------------------------- #
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        self._conferir(X)

        # município nunca visto no treino desta rodada: proporção 0 (nenhuma linha de treino lá)
        X["prop_municipio"] = (X[COL_MUNICIPIO].map(self.prop_municipio_)
                               .fillna(0.0).astype("float64"))

        # a flag é marcada ANTES do corte, para o modelo poder tratar o grupo à parte
        baixo, alto = self.limites_.loc[P_BAIXO, "renda_pc"], self.limites_.loc[P_ALTO, "renda_pc"]
        X["outlier_renda"] = ((X["renda_pc"] < baixo) | (X["renda_pc"] > alto)).astype("float64")

        X = self._winsorizar(X)

        for c in AUSENTES:
            X[f"{c}_ausente"] = X[c].isna().astype("float64")
        # float64 de propósito: em float32 a regressão linear perde precisão com a
        # renda em reais (o R² caiu de ~0,27 para ~0,08 antes da G1, sem erro nenhum)
        X = self._imputar(X)

        X["indice_bens"] = self._media_z(X, BENS, self.bens_media_, self.bens_dp_)
        X["inse_proxy"] = self._media_z(X, INSE, self.inse_media_, self.inse_dp_)

        X[CATEGORICAS] = X[CATEGORICAS].astype(str)
        saida = COLUNAS + (TE if self.target_encodings else [])
        return X[saida]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(COLUNAS + (TE if self.target_encodings else []), dtype=object)

    # -- apoio -------------------------------------------------------------- #
    def _winsorizar(self, X):
        X = X.copy()
        for c in WINSOR:
            X[c] = X[c].astype("float64").clip(self.limites_.loc[P_BAIXO, c],
                                               self.limites_.loc[P_ALTO, c])
        return X

    def _imputar(self, X):
        X = X.copy()
        for c in NUM_ENTRADA:
            X[c] = X[c].astype("float64").fillna(self.medianas_[c])
        return X

    @staticmethod
    def _media_dp(tabela):
        # desvio 0 (coluna constante na amostra) vira 1 para não dividir por zero
        return tabela.mean(), tabela.std(ddof=0).replace(0, 1.0)

    @staticmethod
    def _media_z(X, colunas, media, dp):
        return ((X[colunas] - media) / dp).mean(axis=1).astype("float64")

    def _conferir(self, X):
        exigidas = ENTRADA + (TE if self.target_encodings else [])
        faltando = [c for c in exigidas if c not in X.columns]
        if faltando:
            raise ValueError(f"preparo: faltam colunas na entrada: {faltando}")


def preparo(target_encodings: bool = False) -> Preparo:
    """Devolve o passo de preparo, para entrar como primeiro passo do `Pipeline`."""
    return Preparo(target_encodings=target_encodings)
