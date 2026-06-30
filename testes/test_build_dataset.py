"""
Testes para scripts/build_forecasting_dataset.py.

Cobre todas as transformações do pipeline (sem BD):
  - CADEIA_MAP (completude)
  - aggregate_daily
  - enforce_dtypes
  - expand_daily_panel
  - add_time_features
  - add_cadeia_id
  - add_lag_features
  - add_target
"""

from __future__ import annotations

import warnings

import pandas as pd

from scripts.build_forecasting_dataset import (
    CADEIA_MAP,
    add_cadeia_id,
    add_lag_features,
    add_target,
    add_time_features,
    aggregate_daily,
    enforce_dtypes,
    expand_daily_panel,
)


# ---------------------------------------------------------------------------
# CADEIA_MAP
# ---------------------------------------------------------------------------

class TestCadeiaMap:
    def test_tem_continente(self):
        assert "Continente" in CADEIA_MAP

    def test_tem_pingo_doce(self):
        assert "Pingo Doce" in CADEIA_MAP

    def test_tem_auchan(self):
        assert "Auchan" in CADEIA_MAP

    def test_ids_unicos(self):
        ids = list(CADEIA_MAP.values())
        assert len(ids) == len(set(ids))

    def test_ids_inteiros_sequenciais(self):
        ids = sorted(CADEIA_MAP.values())
        assert ids == list(range(len(ids)))


# ---------------------------------------------------------------------------
# aggregate_daily
# ---------------------------------------------------------------------------

class TestAggregateDaily:
    def test_shape_basica(self, df_historico_minimo):
        result = aggregate_daily(df_historico_minimo)
        # 2 produtos × 3 dias = 6 linhas únicas (sem duplicados por dia)
        assert len(result) == 6

    def test_colunas_presentes(self, df_historico_minimo):
        result = aggregate_daily(df_historico_minimo)
        for col in ["preco", "promo_flag", "observacoes_no_dia"]:
            assert col in result.columns

    def test_preco_e_media(self, df_historico_minimo):
        # Injectar duplicado: produto 1, dia 1, dois preços
        dup = df_historico_minimo.copy()
        extra = dup[dup["id_produto_loja"] == 1].iloc[[0]].copy()
        extra["preco_atual"] = 1.99
        extra["id_historico"] = 99
        df_aug = pd.concat([dup, extra], ignore_index=True)

        result = aggregate_daily(df_aug)
        dia1_prod1 = result[(result["id_produto_loja"] == 1) & (result["dia"] == pd.Timestamp("2026-01-01"))]
        assert len(dia1_prod1) == 1
        # média de 1.69 e 1.99 = 1.84
        assert abs(float(dia1_prod1["preco"].iloc[0]) - 1.84) < 0.01

    def test_promo_flag_true_se_algum(self, df_historico_minimo):
        result = aggregate_daily(df_historico_minimo)
        # produto 1, dia 3 tem em_promocao=True
        linha = result[(result["id_produto_loja"] == 1) & (result["dia"] == pd.Timestamp("2026-01-03"))]
        assert bool(linha["promo_flag"].iloc[0]) is True

    def test_promo_flag_false_sem_promo(self, df_historico_minimo):
        result = aggregate_daily(df_historico_minimo)
        linha = result[(result["id_produto_loja"] == 1) & (result["dia"] == pd.Timestamp("2026-01-01"))]
        assert bool(linha["promo_flag"].iloc[0]) is False


# ---------------------------------------------------------------------------
# enforce_dtypes
# ---------------------------------------------------------------------------

class TestEnforceDtypes:
    def test_ids_sao_int64(self, df_historico_minimo):
        daily = aggregate_daily(df_historico_minimo)
        result = enforce_dtypes(daily)
        assert str(result["id_produto_loja"].dtype) == "Int64"
        assert str(result["id_loja"].dtype) == "Int64"

    def test_preco_e_float64(self, df_historico_minimo):
        daily = aggregate_daily(df_historico_minimo)
        result = enforce_dtypes(daily)
        assert result["preco"].dtype == float

    def test_cadeia_e_string(self, df_historico_minimo):
        daily = aggregate_daily(df_historico_minimo)
        result = enforce_dtypes(daily)
        assert str(result["cadeia"].dtype) == "string"


# ---------------------------------------------------------------------------
# expand_daily_panel
# ---------------------------------------------------------------------------

class TestExpandDailyPanel:
    def test_dias_continuos(self, df_historico_minimo):
        daily = aggregate_daily(df_historico_minimo)
        result = expand_daily_panel(daily)
        # Para cada produto, os dias devem ser contínuos
        for pid, grupo in result.groupby("id_produto_loja"):
            dias = sorted(grupo["dia"].values)
            for i in range(1, len(dias)):
                diff = (pd.Timestamp(dias[i]) - pd.Timestamp(dias[i - 1])).days
                assert diff == 1, f"Produto {pid}: gap de {diff} dias"

    def test_coluna_foi_observado(self, df_historico_minimo):
        daily = aggregate_daily(df_historico_minimo)
        result = expand_daily_panel(daily)
        assert "foi_observado_no_dia" in result.columns
        assert set(result["foi_observado_no_dia"].unique()).issubset({0, 1})

    def test_coluna_dias_desde_ultima_obs(self, df_historico_minimo):
        daily = aggregate_daily(df_historico_minimo)
        result = expand_daily_panel(daily)
        assert "dias_desde_ultima_obs" in result.columns
        assert (result["dias_desde_ultima_obs"] >= 0).all()

    def test_forward_fill_preco(self):
        """Verifica que dias interpolados têm o último preço propagado."""
        df = pd.DataFrame({
            "id_historico":       [1, 2],
            "dia":                pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "id_produto_loja":    [1, 1],
            "id_produto_mestre":  [10, 10],
            "id_loja":            [1, 1],
            "cadeia":             ["Continente", "Continente"],
            "canal":              ["online", "online"],
            "nome_produto":       ["Arroz", "Arroz"],
            "marca":              ["X", "X"],
            "categoria_geral":    ["alim", "alim"],
            "quantidade_valor":   [1000.0, 1000.0],
            "quantidade_unidade": ["g", "g"],
            "preco_unitario_valor":   [1.69, 1.75],
            "preco_unitario_unidade": ["kg", "kg"],
            "preco_atual":        [1.69, 1.75],
            "preco_original":     [None, None],
            "em_promocao":        [False, False],
        })
        daily = aggregate_daily(df)
        result = expand_daily_panel(daily)
        # Deve ter 3 dias: 1, 2, 3
        assert len(result) == 3
        # Dia 2 (interpolado) deve ter preço = 1.69 (forward fill do dia 1)
        dia2 = result[result["dia"] == pd.Timestamp("2026-01-02")]
        assert abs(float(dia2["preco"].iloc[0]) - 1.69) < 0.001

    def test_n_linhas_correto(self, df_historico_minimo):
        daily = aggregate_daily(df_historico_minimo)
        result = expand_daily_panel(daily)
        # Ambos os produtos têm 3 dias contínuos → 6 linhas (sem gaps)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# add_time_features
# ---------------------------------------------------------------------------

class TestAddTimeFeatures:
    def _df_simples(self):
        return pd.DataFrame({
            "dia": pd.to_datetime(["2026-01-01", "2026-04-15", "2026-12-31"]),
        })

    def test_colunas_adicionadas(self):
        df = add_time_features(self._df_simples())
        for col in ["weekday", "month", "weekofyear", "dayofmonth"]:
            assert col in df.columns

    def test_weekday_correto(self):
        df = add_time_features(self._df_simples())
        # 2026-01-01 é quinta-feira (3)
        assert int(df["weekday"].iloc[0]) == 3

    def test_month_correto(self):
        df = add_time_features(self._df_simples())
        assert int(df["month"].iloc[0]) == 1
        assert int(df["month"].iloc[1]) == 4
        assert int(df["month"].iloc[2]) == 12

    def test_dayofmonth_correto(self):
        df = add_time_features(self._df_simples())
        assert int(df["dayofmonth"].iloc[0]) == 1
        assert int(df["dayofmonth"].iloc[2]) == 31

    def test_weekofyear_range(self):
        df = add_time_features(self._df_simples())
        assert df["weekofyear"].between(1, 53).all()

    def test_nao_modifica_original(self):
        df = self._df_simples()
        _ = add_time_features(df)
        assert "weekday" not in df.columns


# ---------------------------------------------------------------------------
# add_cadeia_id
# ---------------------------------------------------------------------------

class TestAddCadeiaId:
    def _df_cadeias(self, cadeias):
        return pd.DataFrame({"cadeia": pd.array(cadeias, dtype="string")})

    def test_continente(self):
        df = add_cadeia_id(self._df_cadeias(["Continente"]))
        assert int(df["cadeia_id"].iloc[0]) == CADEIA_MAP["Continente"]

    def test_pingo_doce(self):
        df = add_cadeia_id(self._df_cadeias(["Pingo Doce"]))
        assert int(df["cadeia_id"].iloc[0]) == CADEIA_MAP["Pingo Doce"]

    def test_auchan(self):
        df = add_cadeia_id(self._df_cadeias(["Auchan"]))
        assert int(df["cadeia_id"].iloc[0]) == CADEIA_MAP["Auchan"]

    def test_desconhecida_da_minus1(self):
        df = add_cadeia_id(self._df_cadeias(["SuperMercadoXPTO"]))
        assert int(df["cadeia_id"].iloc[0]) == -1

    def test_desconhecida_emite_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            add_cadeia_id(self._df_cadeias(["CadeiaDesconhecida"]))
            assert len(w) == 1
            assert "cadeia_id=-1" in str(w[0].message)

    def test_todas_cadeias_conhecidas_sem_warning(self):
        cadeias = list(CADEIA_MAP.keys())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            add_cadeia_id(self._df_cadeias(cadeias))
            # Não deve emitir warnings
            assert len(w) == 0

    def test_nao_modifica_original(self):
        df = self._df_cadeias(["Continente"])
        _ = add_cadeia_id(df)
        assert "cadeia_id" not in df.columns


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------

class TestAddLagFeatures:
    def _df_serie(self):
        return pd.DataFrame({
            "id_produto_loja": [1] * 10,
            "preco": [float(i) for i in range(1, 11)],
        })

    def test_colunas_adicionadas(self):
        df = add_lag_features(self._df_serie())
        for col in ["lag_1", "lag_2", "lag_3", "lag_7", "media_movel_3", "media_movel_7", "desvio_movel_7"]:
            assert col in df.columns

    def test_lag_1_correto(self):
        df = add_lag_features(self._df_serie())
        # Índice 1: lag_1 do preço 2 deve ser 1
        assert float(df["lag_1"].iloc[1]) == 1.0

    def test_lag_7_nan_no_inicio(self):
        df = add_lag_features(self._df_serie())
        # Primeiros 7 elementos não têm lag_7 completo
        assert df["lag_7"].isna().sum() == 7

    def test_media_movel_3_calculada(self):
        df = add_lag_features(self._df_serie())
        # No índice 3 (preço=4): media_movel_3 = média de [1,2,3] = 2.0
        assert abs(float(df["media_movel_3"].iloc[3]) - 2.0) < 0.001

    def test_desvio_movel_7_nao_negativo(self):
        df = add_lag_features(self._df_serie())
        assert (df["desvio_movel_7"].fillna(0) >= 0).all()

    def test_lags_customizados(self):
        df = add_lag_features(self._df_serie(), lags=(1, 5))
        assert "lag_1" in df.columns
        assert "lag_5" in df.columns
        assert "lag_2" not in df.columns
        assert "lag_3" not in df.columns

    def test_multiplos_produtos_independentes(self):
        df = pd.DataFrame({
            "id_produto_loja": [1, 1, 1, 2, 2, 2],
            "preco": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
        })
        result = add_lag_features(df, lags=(1,))
        # Produto 2, índice 3 deve ter lag_1=NaN (primeiro da sua série)
        assert pd.isna(result.loc[result["id_produto_loja"] == 2, "lag_1"].iloc[0])

    def test_nao_modifica_original(self):
        df = self._df_serie()
        _ = add_lag_features(df)
        assert "lag_1" not in df.columns


# ---------------------------------------------------------------------------
# add_target
# ---------------------------------------------------------------------------

class TestAddTarget:
    def _df_serie(self):
        return pd.DataFrame({
            "id_produto_loja": [1] * 5,
            "preco": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

    def test_coluna_adicionada(self):
        df = add_target(self._df_serie())
        assert "target_t_plus_1" in df.columns

    def test_target_e_preco_seguinte(self):
        df = add_target(self._df_serie())
        # Índice 0: preco=1.0 → target = 2.0
        assert float(df["target_t_plus_1"].iloc[0]) == 2.0

    def test_ultimo_elemento_e_nan(self):
        df = add_target(self._df_serie())
        assert pd.isna(df["target_t_plus_1"].iloc[-1])

    def test_horizonte_personalizado(self):
        df = add_target(self._df_serie(), horizonte=3)
        assert "target_t_plus_3" in df.columns
        # Índice 0: target_t_plus_3 = preco do índice 3 = 4.0
        assert float(df["target_t_plus_3"].iloc[0]) == 4.0
        # Últimos 3 elementos são NaN
        assert df["target_t_plus_3"].iloc[-3:].isna().all()

    def test_multiplos_produtos_independentes(self):
        df = pd.DataFrame({
            "id_produto_loja": [1, 1, 2, 2],
            "preco": [10.0, 20.0, 1.0, 2.0],
        })
        result = add_target(df)
        # Produto 1, último elemento (índice 1): NaN
        assert pd.isna(result.loc[result["id_produto_loja"] == 1, "target_t_plus_1"].iloc[-1])
        # Produto 2, último elemento: NaN
        assert pd.isna(result.loc[result["id_produto_loja"] == 2, "target_t_plus_1"].iloc[-1])

    def test_nao_modifica_original(self):
        df = self._df_serie()
        _ = add_target(df)
        assert "target_t_plus_1" not in df.columns
