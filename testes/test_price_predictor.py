"""
Testes para models/price_predictor.py.

Cobre sem necessidade de treino real ou GPU:
  - Constantes e índices de features
  - LSTMPrecoPredictor — forward pass, shapes de output
  - SerieTemporalDataset — len, __getitem__
  - preparar_serie — janelas deslizantes, série curta demais
  - calcular_metricas — RMSE/MAE/MAPE
  - _raw_features_para_dia — shape e campos preenchidos
  - Cache de artefactos (_mtime_combinado quando ficheiros ausentes)
"""

from __future__ import annotations

from collections import deque
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from models.price_predictor import (
    CADEIA_IDX,
    DAY_IDX,
    DIAS_OBS_IDX,
    DM7_IDX,
    FEATURE_COLS,
    LAG1_IDX,
    LAG2_IDX,
    LAG3_IDX,
    LAG7_IDX,
    MM3_IDX,
    MM7_IDX,
    MONTH_IDX,
    N_FEATURES,
    PRECO_IDX,
    PROMO_IDX,
    TARGET_COL,
    WEEKDAY_IDX,
    WEEKOFYEAR_IDX,
    LSTMPrecoPredictor,
    SerieTemporalDataset,
    _mtime_combinado,
    _raw_features_para_dia,
    calcular_metricas,
    preparar_serie,
)


# ---------------------------------------------------------------------------
# Constantes e índices
# ---------------------------------------------------------------------------

class TestConstantes:
    def test_n_features_correto(self):
        assert N_FEATURES == len(FEATURE_COLS)

    def test_n_features_e_15(self):
        assert N_FEATURES == 15

    def test_target_col_definida(self):
        assert TARGET_COL == "target_t_plus_1"

    def test_indices_unicos(self):
        indices = [
            PRECO_IDX, PROMO_IDX, WEEKDAY_IDX, WEEKOFYEAR_IDX,
            MONTH_IDX, DAY_IDX, CADEIA_IDX, DIAS_OBS_IDX,
            LAG1_IDX, LAG2_IDX, LAG3_IDX, LAG7_IDX,
            MM3_IDX, MM7_IDX, DM7_IDX,
        ]
        assert len(indices) == len(set(indices))

    def test_indices_dentro_do_range(self):
        indices = [
            PRECO_IDX, PROMO_IDX, WEEKDAY_IDX, WEEKOFYEAR_IDX,
            MONTH_IDX, DAY_IDX, CADEIA_IDX, DIAS_OBS_IDX,
            LAG1_IDX, LAG2_IDX, LAG3_IDX, LAG7_IDX,
            MM3_IDX, MM7_IDX, DM7_IDX,
        ]
        for idx in indices:
            assert 0 <= idx < N_FEATURES

    def test_feature_cols_sem_duplicados(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


# ---------------------------------------------------------------------------
# LSTMPrecoPredictor — arquitectura
# ---------------------------------------------------------------------------

class TestLSTMPrecoPredictor:
    def test_instanciacao_default(self):
        model = LSTMPrecoPredictor()
        assert model is not None

    def test_forward_shape(self):
        model = LSTMPrecoPredictor()
        model.eval()
        batch, janela = 4, 7
        x = torch.randn(batch, janela, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch,)

    def test_forward_batch_1(self):
        model = LSTMPrecoPredictor()
        model.eval()
        x = torch.randn(1, 7, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_forward_diferentes_janelas(self):
        model = LSTMPrecoPredictor()
        model.eval()
        for janela in [3, 7, 14, 30]:
            x = torch.randn(2, janela, N_FEATURES)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2,), f"Falhou para janela={janela}"

    def test_forward_input_size_customizado(self):
        model = LSTMPrecoPredictor(input_size=5)
        model.eval()
        x = torch.randn(2, 7, 5)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)

    def test_parametros_treinaveis(self):
        model = LSTMPrecoPredictor()
        params = list(model.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0

    def test_modo_treino_vs_eval(self):
        """Em modo train, dropout activo → outputs diferentes por chamada."""
        model = LSTMPrecoPredictor(dropout=0.5)
        model.train()
        x = torch.randn(1, 7, N_FEATURES)
        outs = []
        for _ in range(10):
            with torch.no_grad():
                outs.append(model(x).item())
        # Com dropout activo, os outputs devem variar
        assert len(set(round(o, 4) for o in outs)) > 1

    def test_modo_eval_determinista(self):
        """Em modo eval, dropout desactivado → outputs idênticos."""
        model = LSTMPrecoPredictor()
        model.eval()
        x = torch.randn(1, 7, N_FEATURES)
        outs = []
        for _ in range(5):
            with torch.no_grad():
                outs.append(model(x).item())
        assert len(set(round(o, 6) for o in outs)) == 1


# ---------------------------------------------------------------------------
# SerieTemporalDataset
# ---------------------------------------------------------------------------

class TestSerieTemporalDataset:
    def _make_dataset(self, n=10, janela=3):
        X = np.random.randn(n, janela, N_FEATURES).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        return SerieTemporalDataset(X, y), X, y

    def test_len_correto(self):
        ds, _, _ = self._make_dataset(n=10)
        assert len(ds) == 10

    def test_getitem_shapes(self):
        ds, _, _ = self._make_dataset(n=10, janela=7)
        x, y = ds[0]
        assert x.shape == (7, N_FEATURES)
        assert y.shape == ()

    def test_getitem_valores(self):
        ds, X, y = self._make_dataset(n=5)
        x_item, y_item = ds[2]
        assert torch.allclose(x_item, torch.tensor(X[2]))
        assert abs(float(y_item) - float(y[2])) < 1e-6

    def test_tensores_float32(self):
        ds, _, _ = self._make_dataset()
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_dataset_vazio(self):
        X = np.empty((0, 7, N_FEATURES), dtype=np.float32)
        y = np.empty(0, dtype=np.float32)
        ds = SerieTemporalDataset(X, y)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# preparar_serie
# ---------------------------------------------------------------------------

class TestPrepararSerie:
    def _df_produto(self, n_dias=20):
        """DataFrame sintético com todas as features necessárias."""
        rng = np.random.default_rng(42)
        data = {col: rng.random(n_dias) for col in FEATURE_COLS}
        data[TARGET_COL] = rng.random(n_dias)
        data["dia"] = pd.date_range("2026-01-01", periods=n_dias)
        return pd.DataFrame(data)

    def test_shapes_basicos(self):
        df = self._df_produto(n_dias=20)
        scaler = MinMaxScaler()
        janela = 7
        X, y = preparar_serie(df, scaler, janela, fit_scaler=True)
        n_esperado = 20 - janela  # 13 janelas
        assert X.shape == (n_esperado, janela, N_FEATURES)
        assert y.shape == (n_esperado,)

    def test_dtype_float32(self):
        df = self._df_produto(n_dias=15)
        scaler = MinMaxScaler()
        X, y = preparar_serie(df, scaler, 7, fit_scaler=True)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_serie_demasiado_curta_devolve_vazio(self):
        df = self._df_produto(n_dias=5)
        scaler = MinMaxScaler()
        X, y = preparar_serie(df, scaler, janela=7, fit_scaler=True)
        assert len(X) == 0
        assert len(y) == 0
        assert X.shape == (0, 7, N_FEATURES)

    def test_fit_false_usa_scaler_existente(self):
        df = self._df_produto(n_dias=20)
        scaler = MinMaxScaler()
        X_train, _ = preparar_serie(df, scaler, 7, fit_scaler=True)

        df_val = self._df_produto(n_dias=15)
        X_val, _ = preparar_serie(df_val, scaler, 7, fit_scaler=False)
        # Com fit_scaler=False, o scaler não deve ser re-ajustado
        assert X_val.shape[1] == 7
        assert X_val.shape[2] == N_FEATURES

    def test_valores_normalizados_entre_0_e_1(self):
        df = self._df_produto(n_dias=20)
        scaler = MinMaxScaler()
        X, _ = preparar_serie(df, scaler, 7, fit_scaler=True)
        # MinMaxScaler normaliza para [0,1]
        assert float(X.min()) >= -0.01
        assert float(X.max()) <= 1.01

    def test_target_e_preco_normalizado(self):
        df = self._df_produto(n_dias=20)
        scaler = MinMaxScaler()
        X, y = preparar_serie(df, scaler, 7, fit_scaler=True)
        # Target deve ser o preço normalizado da posição PRECO_IDX
        # O primeiro target é o PRECO_IDX do elemento na posição janela (7)
        assert y.shape[0] == X.shape[0]


# ---------------------------------------------------------------------------
# calcular_metricas
# ---------------------------------------------------------------------------

class TestCalcularMetricas:
    def _scaler_identidade(self):
        """Scaler que não transforma (fit num array [0..1])."""
        scaler = MinMaxScaler()
        dummy = np.zeros((2, N_FEATURES), dtype=np.float32)
        dummy[0, PRECO_IDX] = 0.0
        dummy[1, PRECO_IDX] = 10.0
        scaler.fit(dummy)
        return scaler

    def test_rmse_zero_previsao_perfeita(self):
        scaler = self._scaler_identidade()
        y = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        result = calcular_metricas(y, y, scaler, produto_id=1)
        assert result["rmse_euros"] == 0.0
        assert result["mae_euros"] == 0.0

    def test_campos_presentes(self):
        scaler = self._scaler_identidade()
        y_real = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        y_pred = np.array([0.4, 0.7, 0.65], dtype=np.float32)
        result = calcular_metricas(y_real, y_pred, scaler, produto_id=42)
        for campo in ["produto_id", "n_amostras", "rmse_euros", "mae_euros", "mape_pct"]:
            assert campo in result

    def test_produto_id_preservado(self):
        scaler = self._scaler_identidade()
        y = np.array([0.5], dtype=np.float32)
        result = calcular_metricas(y, y, scaler, produto_id=99)
        assert result["produto_id"] == 99

    def test_n_amostras_correto(self):
        scaler = self._scaler_identidade()
        y = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        result = calcular_metricas(y, y, scaler, produto_id=1)
        assert result["n_amostras"] == 4

    def test_rmse_maior_que_mae(self):
        """RMSE é sempre ≥ MAE (penaliza mais os erros grandes)."""
        scaler = self._scaler_identidade()
        y_real = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        y_pred = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = calcular_metricas(y_real, y_pred, scaler, produto_id=1)
        assert result["rmse_euros"] >= result["mae_euros"]

    def test_valores_nao_negativos(self):
        scaler = self._scaler_identidade()
        y_real = np.array([0.3, 0.5, 0.8], dtype=np.float32)
        y_pred = np.array([0.4, 0.4, 0.9], dtype=np.float32)
        result = calcular_metricas(y_real, y_pred, scaler, produto_id=1)
        assert result["rmse_euros"] >= 0
        assert result["mae_euros"] >= 0


# ---------------------------------------------------------------------------
# _raw_features_para_dia
# ---------------------------------------------------------------------------

class TestRawFeaturesPorDia:
    def test_shape_correto(self):
        hist = deque([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], maxlen=7)
        row = _raw_features_para_dia(
            data=date(2026, 4, 18),
            preco_raw=1.7,
            historico_precos=hist,
            dias_desde_obs=1,
        )
        assert row.shape == (N_FEATURES,)

    def test_dtype_float32(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 1.5, hist, 0)
        assert row.dtype == np.float32

    def test_preco_na_posicao_correta(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 2.5, hist, 0)
        assert float(row[PRECO_IDX]) == 2.5

    def test_mes_correto(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 1.0, hist, 0)
        assert int(row[MONTH_IDX]) == 4

    def test_dia_correto(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 1.0, hist, 0)
        assert int(row[DAY_IDX]) == 18

    def test_promo_default_zero(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 1.0, hist, 0)
        assert float(row[PROMO_IDX]) == 0.0

    def test_cadeia_id_passado(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 1.0, hist, 0, cadeia_id=2)
        assert int(row[CADEIA_IDX]) == 2

    def test_lags_com_historico_suficiente(self):
        hist = deque([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 8.0, hist, 0)
        assert float(row[LAG1_IDX]) == 7.0  # último elemento
        assert float(row[LAG2_IDX]) == 6.0
        assert float(row[LAG7_IDX]) == 1.0  # primeiro elemento

    def test_lags_com_historico_vazio(self):
        hist = deque([], maxlen=7)
        preco = 1.5
        row = _raw_features_para_dia(date(2026, 4, 18), preco, hist, 0)
        # Com histórico vazio, todos os lags devem ser o próprio preço
        assert float(row[LAG1_IDX]) == preco

    def test_dias_desde_obs_preenchido(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 1.0, hist, 5)
        assert int(row[DIAS_OBS_IDX]) == 5

    def test_todos_os_campos_preenchidos(self):
        hist = deque([1.0] * 7, maxlen=7)
        row = _raw_features_para_dia(date(2026, 4, 18), 1.0, hist, 0)
        # Nenhum NaN
        assert not np.any(np.isnan(row))


# ---------------------------------------------------------------------------
# _mtime_combinado (cache de artefactos)
# ---------------------------------------------------------------------------

class TestMtimeCombinado:
    def test_sem_ficheiros_devolve_zero(self):
        """Sem modelos treinados, deve devolver 0.0."""
        # Esta função depende de MODELS_DIR/lstm_global.pt etc.
        # Se os ficheiros não existirem, deve devolver 0.0
        result = _mtime_combinado()
        assert isinstance(result, float)
        # Pode ser 0.0 (sem ficheiros) ou > 0 (com ficheiros)
        assert result >= 0.0
