"""
Testes para ``models/gbm_predictor.py`` (modelo de gradient boosting).

Usa um dataset sintético em memória (sem BD nem rede) com a mesma estrutura de
colunas que ``forecasting_dataset.csv``. Cobre:
  - criar_modelo (tipo e reprodutibilidade)
  - avaliar_gbm_por_produto (estrutura, filtro de história, métricas, determinismo)
  - casos limite (dataset vazio / sem história suficiente)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from models.gbm_predictor import (
    FEATURE_COLS,
    TARGET_COL,
    avaliar_gbm_por_produto,
    criar_modelo,
    importancia_features,
)


# ---------------------------------------------------------------------------
# Gerador de dataset sintético
# ---------------------------------------------------------------------------

def _dataset_sintetico(n_produtos: int = 6, n_dias: int = 30, seed: int = 0) -> pd.DataFrame:
    """Constrói um dataset com as colunas que o GBM espera.

    Os preços seguem um passeio aleatório suave; lags e médias móveis são
    calculados de forma consistente. ``target_t_plus_1`` é o preço do dia
    seguinte (NaN no último dia de cada produto, como no dataset real).
    """
    rng = np.random.default_rng(seed)
    linhas: list[dict] = []
    for pid in range(1, n_produtos + 1):
        base = rng.uniform(1.0, 5.0)
        ruido = rng.normal(0, 0.03, n_dias).cumsum()
        precos = np.clip(base + ruido, 0.2, None)
        for i in range(n_dias):
            janela3 = precos[max(0, i - 2): i + 1]
            janela7 = precos[max(0, i - 6): i + 1]
            linhas.append({
                "dia":                    pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "id_produto_loja":        pid,
                "preco":                  precos[i],
                "promo_flag":             0,
                "weekday":                i % 7,
                "weekofyear":             (i // 7) % 52 + 1,
                "month":                  1,
                "dayofmonth":             (i % 28) + 1,
                "cadeia_id":              pid % 3,
                "dias_desde_ultima_obs":  1,
                "lag_1":                  precos[i - 1] if i >= 1 else precos[i],
                "lag_2":                  precos[i - 2] if i >= 2 else precos[i],
                "lag_3":                  precos[i - 3] if i >= 3 else precos[i],
                "lag_7":                  precos[i - 7] if i >= 7 else precos[i],
                "media_movel_3":          float(janela3.mean()),
                "media_movel_7":          float(janela7.mean()),
                "desvio_movel_7":         float(janela7.std()),
                "target_t_plus_1":        precos[i + 1] if i + 1 < n_dias else np.nan,
            })
    return pd.DataFrame(linhas)


COLS_ESPERADAS = ["produto_id", "modelo", "n_amostras",
                  "rmse_euros", "mae_euros", "mape_pct"]


# ---------------------------------------------------------------------------
# criar_modelo
# ---------------------------------------------------------------------------

class TestCriarModelo:
    def test_devolve_hgbr(self):
        assert isinstance(criar_modelo(), HistGradientBoostingRegressor)

    def test_random_state_propagado(self):
        assert criar_modelo(random_state=123).random_state == 123


# ---------------------------------------------------------------------------
# FEATURE_COLS / TARGET_COL — sincronizados com o LSTM
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_15_features(self):
        assert len(FEATURE_COLS) == 15

    def test_target_e_t_plus_1(self):
        assert TARGET_COL == "target_t_plus_1"

    def test_features_iguais_ao_lstm(self):
        # Garante a comparação apples-to-apples com o LSTM.
        from models.price_predictor import FEATURE_COLS as LSTM_FEATURES
        assert FEATURE_COLS == LSTM_FEATURES


# ---------------------------------------------------------------------------
# avaliar_gbm_por_produto
# ---------------------------------------------------------------------------

class TestAvaliarGbm:
    def test_estrutura_de_colunas(self):
        df = _dataset_sintetico()
        out = avaliar_gbm_por_produto(df, janela=7)
        assert list(out.columns) == COLS_ESPERADAS

    def test_modelo_e_gbm(self):
        out = avaliar_gbm_por_produto(_dataset_sintetico(), janela=7)
        assert (out["modelo"] == "gbm").all()

    def test_avalia_produtos_com_historia_suficiente(self):
        df = _dataset_sintetico(n_produtos=6, n_dias=30)
        out = avaliar_gbm_por_produto(df, janela=7)
        # Todos os 6 produtos têm 29 linhas válidas (>= 15) → avaliados.
        assert set(out["produto_id"]) == {1, 2, 3, 4, 5, 6}

    def test_produto_curto_excluido(self):
        # Produto longo (avaliável) + produto curto (história insuficiente).
        longo = _dataset_sintetico(n_produtos=1, n_dias=30, seed=1)
        curto = _dataset_sintetico(n_produtos=1, n_dias=10, seed=2)
        curto["id_produto_loja"] = 99
        df = pd.concat([longo, curto], ignore_index=True)
        out = avaliar_gbm_por_produto(df, janela=7)
        assert 1 in set(out["produto_id"])
        assert 99 not in set(out["produto_id"])  # curto não é avaliado

    def test_n_amostras_igual_a_janela(self):
        out = avaliar_gbm_por_produto(_dataset_sintetico(), janela=7)
        assert (out["n_amostras"] == 7).all()

    def test_metricas_finitas_e_nao_negativas(self):
        out = avaliar_gbm_por_produto(_dataset_sintetico(), janela=7)
        assert out["rmse_euros"].notna().all()
        assert (out["rmse_euros"] >= 0).all()
        assert (out["mae_euros"] >= 0).all()

    def test_determinista(self):
        df = _dataset_sintetico()
        a = avaliar_gbm_por_produto(df, janela=7, random_state=42)
        b = avaliar_gbm_por_produto(df, janela=7, random_state=42)
        pd.testing.assert_frame_equal(a, b)

    def test_max_produtos_limita(self):
        df = _dataset_sintetico(n_produtos=6)
        out = avaliar_gbm_por_produto(df, janela=7, max_produtos=3)
        assert out["produto_id"].nunique() <= 3

    def test_dataset_vazio_devolve_vazio(self):
        vazio = pd.DataFrame(columns=["dia", "id_produto_loja", *FEATURE_COLS, TARGET_COL])
        out = avaliar_gbm_por_produto(vazio, janela=7)
        assert out.empty
        assert list(out.columns) == COLS_ESPERADAS

    def test_sem_historia_suficiente_devolve_vazio(self):
        # Todos os produtos demasiado curtos → nada a avaliar.
        df = _dataset_sintetico(n_produtos=3, n_dias=8)
        out = avaliar_gbm_por_produto(df, janela=7)
        assert out.empty


# ---------------------------------------------------------------------------
# importancia_features
# ---------------------------------------------------------------------------

class TestImportanciaFeatures:
    def test_estrutura_e_cobre_todas_as_features(self):
        imp = importancia_features(_dataset_sintetico(n_produtos=8, n_dias=40), random_state=0)
        assert list(imp.columns) == ["feature", "importancia"]
        assert set(imp["feature"]) == set(FEATURE_COLS)
        assert len(imp) == len(FEATURE_COLS)

    def test_ordenado_por_importancia_desc(self):
        imp = importancia_features(_dataset_sintetico(n_produtos=8, n_dias=40), random_state=0)
        assert imp["importancia"].is_monotonic_decreasing
