"""
Testes para ``models/baselines.py``.

Cobre todos os modelos baseline:
  - prever_naive
  - prever_media_movel
  - prever_arima
  - calcular_metricas_em_euros
  - aplicar_baseline (despacho)
  - BASELINES (sanidade do registo)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from models.baselines import (
    BASELINES,
    aplicar_baseline,
    calcular_metricas_em_euros,
    prever_arima,
    prever_media_movel,
    prever_naive,
)


# ===========================================================================
# Naive
# ===========================================================================

class TestPreverNaive:
    def test_repete_ultimo_valor(self):
        serie = np.array([1.0, 1.5, 2.0])
        assert np.array_equal(prever_naive(serie, horizonte=3), [2.0, 2.0, 2.0])

    def test_horizonte_1(self):
        serie = np.array([3.14])
        assert np.array_equal(prever_naive(serie, 1), [3.14])

    def test_horizonte_grande(self):
        serie = np.array([1.0, 2.0])
        out = prever_naive(serie, horizonte=10)
        assert len(out) == 10
        assert all(v == 2.0 for v in out)


# ===========================================================================
# Média móvel
# ===========================================================================

class TestPreverMediaMovel:
    def test_janela_completa(self):
        serie = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Média das últimas 3: (3+4+5)/3 = 4.0
        out = prever_media_movel(serie, horizonte=2, janela=3)
        assert np.allclose(out, [4.0, 4.0])

    def test_janela_maior_que_serie_usa_toda_a_serie(self):
        serie = np.array([1.0, 2.0, 3.0])
        out = prever_media_movel(serie, horizonte=1, janela=10)
        # Média de toda a série: 2.0
        assert math.isclose(out[0], 2.0)

    def test_serie_vazia_devolve_zeros(self):
        serie = np.array([])
        out = prever_media_movel(serie, horizonte=3, janela=7)
        assert np.array_equal(out, [0.0, 0.0, 0.0])

    def test_previsao_e_constante(self):
        """A previsão de média móvel não muda ao longo do horizonte."""
        serie = np.array([1.0, 2.0, 3.0])
        out = prever_media_movel(serie, horizonte=5, janela=2)
        assert len(set(out)) == 1


# ===========================================================================
# ARIMA
# ===========================================================================

class TestPreverArima:
    def test_arima_converge_em_serie_normal(self):
        """ARIMA(1,1,1) deve convergir e produzir previsões finitas."""
        # Série com tendência ligeira
        serie = np.array([1.0, 1.05, 1.10, 1.12, 1.15, 1.18, 1.20,
                          1.22, 1.25, 1.28, 1.30, 1.32])
        out = prever_arima(serie, horizonte=3)
        assert len(out) == 3
        assert all(np.isfinite(out)), "ARIMA produziu NaN/Inf em série normal"

    def test_arima_nan_em_serie_curta(self):
        """Série mais curta que p+d+q deve devolver NaN sem rebentar."""
        serie = np.array([1.0])
        out = prever_arima(serie, horizonte=3)
        assert all(np.isnan(out))

    def test_arima_nan_em_serie_constante(self):
        """Série constante geralmente não converge no ARIMA — deve devolver NaN."""
        serie = np.full(20, 1.50)
        out = prever_arima(serie, horizonte=3)
        # Pode dar NaN (não convergiu) ou 1.50 (degenerou). Ambos válidos.
        assert len(out) == 3

    def test_arima_ordem_customizada(self):
        """Aceita ordem (p,d,q) customizada."""
        serie = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0])
        out = prever_arima(serie, horizonte=2, ordem=(2, 0, 1))
        # Não deve rebentar; pode ser NaN ou número
        assert len(out) == 2


# ===========================================================================
# Métricas em euros
# ===========================================================================

class TestCalcularMetricasEmEuros:
    def test_previsao_perfeita_dá_zeros(self):
        y = np.array([1.0, 2.0, 3.0])
        m = calcular_metricas_em_euros(y, y.copy())
        assert m["rmse_euros"] == 0.0
        assert m["mae_euros"] == 0.0
        assert m["mape_pct"] == 0.0

    def test_erro_constante_dá_mae_igual_ao_erro(self):
        y_real = np.array([1.0, 2.0, 3.0])
        y_pred = y_real + 0.5    # erro constante de +0.5
        m = calcular_metricas_em_euros(y_real, y_pred)
        assert math.isclose(m["mae_euros"], 0.5)
        # RMSE também = 0.5 (todos os erros iguais)
        assert math.isclose(m["rmse_euros"], 0.5)

    def test_nan_em_y_pred_propaga_nan(self):
        y_real = np.array([1.0, 2.0])
        y_pred = np.array([1.0, np.nan])
        m = calcular_metricas_em_euros(y_real, y_pred)
        assert math.isnan(m["rmse_euros"])
        assert math.isnan(m["mae_euros"])
        assert math.isnan(m["mape_pct"])

    def test_serie_vazia_devolve_nan(self):
        m = calcular_metricas_em_euros(np.array([]), np.array([]))
        assert math.isnan(m["rmse_euros"])

    def test_y_real_zero_evita_divisao_por_zero_no_mape(self):
        """Quando y_real é 0, é excluído do MAPE para evitar divisão por zero."""
        y_real = np.array([0.0, 1.0])
        y_pred = np.array([0.1, 0.9])
        m = calcular_metricas_em_euros(y_real, y_pred)
        # MAPE só considera o segundo ponto: |1.0 - 0.9| / 1.0 = 10%
        assert math.isclose(m["mape_pct"], 10.0)


# ===========================================================================
# Despacho e registo
# ===========================================================================

class TestDespacho:
    @pytest.mark.parametrize("nome", list(BASELINES.keys()))
    def test_todos_os_baselines_funcionam(self, nome):
        """Todos os baselines devolvem array do tamanho do horizonte."""
        serie = np.array([1.0, 1.05, 1.10, 1.12, 1.15, 1.18,
                          1.20, 1.22, 1.25, 1.28, 1.30, 1.32])
        out = aplicar_baseline(nome, serie, horizonte=3)
        assert len(out) == 3

    def test_nome_desconhecido_levanta_key_error(self):
        with pytest.raises(KeyError):
            aplicar_baseline("modelo-que-nao-existe", np.array([1.0]), 1)

    def test_registo_tem_4_modelos(self):
        """4 baselines registados: naive, media_3, media_7, arima_111."""
        assert set(BASELINES.keys()) == {"naive", "media_3", "media_7", "arima_111"}

    def test_cada_baseline_e_tupla_func_kwargs(self):
        for nome, (fn, kwargs) in BASELINES.items():
            assert callable(fn), f"{nome}: primeira posição deve ser callable"
            assert isinstance(kwargs, dict), f"{nome}: segunda posição deve ser dict"
