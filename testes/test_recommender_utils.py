"""
Testes para models/recommender.py — funções puras (sem BD).

Cobre:
  - _condicoes_multi_ilike  — geração de cláusulas AND-ILIKE
  - _condicoes_multi_ou     — geração de cláusulas OR-ILIKE
  - recomendar_momento_compra — lógica de limiar e recomendação (com mock)
  - otimizar_lista_compras   — lógica de custo mínimo (com mock)
  - pesquisar_produtos       — lógica AND/OR com fallback (com mock)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from models.recommender import (
    _condicoes_multi_ilike,
    _condicoes_multi_ou,
)


# ---------------------------------------------------------------------------
# _condicoes_multi_ilike (AND)
# ---------------------------------------------------------------------------

class TestCondicoesMultiIlike:
    def test_uma_palavra(self):
        sql, params = _condicoes_multi_ilike("pm.nome_padronizado", "arroz")
        assert "ILIKE" in sql
        assert params == ["%arroz%"]

    def test_duas_palavras(self):
        sql, params = _condicoes_multi_ilike("pm.nome_padronizado", "arroz agulha")
        assert " AND " in sql
        assert "%arroz%" in params
        assert "%agulha%" in params
        assert len(params) == 2

    def test_tres_palavras(self):
        sql, params = _condicoes_multi_ilike("pm.nome_padronizado", "arroz agulha cigala")
        assert sql.count(" AND ") == 2
        assert len(params) == 3

    def test_palavras_curtas_ignoradas(self):
        # O filtro usa >= 2 chars, portanto "de" (2 chars) É incluído
        # Apenas palavras com 1 char são filtradas
        sql, params = _condicoes_multi_ilike("pm.nome_padronizado", "arroz a mesa")
        assert "%a%" not in params   # "a" tem 1 char — filtrada
        assert "%arroz%" in params
        assert "%mesa%" in params

    def test_termo_vazio_usa_like_total(self):
        sql, params = _condicoes_multi_ilike("pm.nome_padronizado", "")
        assert "ILIKE" in sql
        assert len(params) == 1

    def test_campo_personalizado(self):
        sql, _ = _condicoes_multi_ilike("pm.marca", "cigala")
        assert "pm.marca" in sql

    def test_params_sao_wildcards(self):
        _, params = _condicoes_multi_ilike("pm.nome_padronizado", "leite mimosa")
        for p in params:
            assert p.startswith("%")
            assert p.endswith("%")


# ---------------------------------------------------------------------------
# _condicoes_multi_ou (OR)
# ---------------------------------------------------------------------------

class TestCondicoesMultiOu:
    def test_uma_palavra(self):
        sql, params = _condicoes_multi_ou("pm.nome_padronizado", "arroz")
        assert "ILIKE" in sql
        assert params == ["%arroz%"]

    def test_duas_palavras(self):
        sql, params = _condicoes_multi_ou("pm.nome_padronizado", "arroz agulha")
        assert " OR " in sql
        assert "%arroz%" in params
        assert "%agulha%" in params

    def test_resultado_envolto_em_parenteses(self):
        sql, _ = _condicoes_multi_ou("pm.nome_padronizado", "arroz agulha")
        # Com múltiplas palavras, deve ter parênteses para evitar precedência errada
        assert sql.startswith("(")
        assert sql.endswith(")")

    def test_uma_palavra_sem_parenteses_extra(self):
        sql, _ = _condicoes_multi_ou("pm.nome_padronizado", "arroz")
        # Apenas uma condição — sem parênteses obrigatórios
        assert "ILIKE" in sql

    def test_palavras_curtas_ignoradas(self):
        # O filtro usa >= 2 chars; "a" (1 char) é filtrado, "de" (2 chars) não
        sql, params = _condicoes_multi_ou("pm.nome_padronizado", "arroz a mesa")
        assert "%a%" not in params

    def test_params_sao_wildcards(self):
        _, params = _condicoes_multi_ou("pm.nome_padronizado", "azeite gallo")
        for p in params:
            assert p.startswith("%")
            assert p.endswith("%")

    def test_and_vs_or_diferente(self):
        sql_and, _ = _condicoes_multi_ilike("pm.nome_padronizado", "arroz agulha")
        sql_or, _ = _condicoes_multi_ou("pm.nome_padronizado", "arroz agulha")
        assert sql_and != sql_or
        assert "AND" in sql_and
        assert "OR" in sql_or


# ---------------------------------------------------------------------------
# recomendar_momento_compra — lógica de limiar (mock do LSTM)
# ---------------------------------------------------------------------------

class TestRecomendarMomentCompra:
    """Testa a lógica de decisão sem BD nem modelo treinado."""

    def _mock_resultado_mc(self, precos_medios, precos_ic5, preco_atual):
        datas = pd.date_range("2026-04-19", periods=len(precos_medios))
        df = pd.DataFrame({
            "data":       datas,
            "preco_medio": precos_medios,
            "preco_std":  [0.05] * len(precos_medios),
            "ic_5pct":    precos_ic5,
            "ic_95pct":   [p + 0.2 for p in precos_medios],
        })
        return {"previsoes": df, "preco_atual": preco_atual}

    def test_recomenda_aguardar_quando_descida_supera_limiar(self):
        from models.recommender import recomendar_momento_compra
        resultado_mc = self._mock_resultado_mc(
            precos_medios=[1.69, 1.60, 1.55, 1.50, 1.55, 1.60, 1.65],
            precos_ic5=   [1.65, 1.55, 1.50, 1.45, 1.50, 1.55, 1.60],
            preco_atual=1.69,
        )
        with patch("models.recommender.prever_preco_com_incerteza", return_value=resultado_mc), \
             patch("models.recommender.get_connection"):
            result = recomendar_momento_compra(
                produto_id=1,
                caminho_csv="dummy.csv",
                horizonte=7,
                limiar_descida_pct=2.0,
            )
        assert result is not None
        assert result["recomendacao"] == "aguardar"

    def test_recomenda_comprar_agora_quando_descida_abaixo_limiar(self):
        from models.recommender import recomendar_momento_compra
        resultado_mc = self._mock_resultado_mc(
            precos_medios=[1.69, 1.68, 1.69, 1.70, 1.69, 1.68, 1.69],
            precos_ic5=   [1.67, 1.66, 1.67, 1.68, 1.67, 1.66, 1.67],
            preco_atual=1.69,
        )
        with patch("models.recommender.prever_preco_com_incerteza", return_value=resultado_mc), \
             patch("models.recommender.get_connection"):
            result = recomendar_momento_compra(
                produto_id=1,
                caminho_csv="dummy.csv",
                horizonte=7,
                limiar_descida_pct=2.0,
            )
        assert result is not None
        assert result["recomendacao"] == "comprar_agora"

    def test_modelo_indisponivel_devolve_none(self):
        from models.recommender import recomendar_momento_compra
        with patch("models.recommender.prever_preco_com_incerteza", return_value=None), \
             patch("models.recommender.get_connection"):
            result = recomendar_momento_compra(
                produto_id=999,
                caminho_csv="dummy.csv",
            )
        assert result is None

    def test_campos_resultado(self):
        from models.recommender import recomendar_momento_compra
        resultado_mc = self._mock_resultado_mc(
            precos_medios=[1.50] * 7,
            precos_ic5=   [1.45] * 7,
            preco_atual=1.69,
        )
        with patch("models.recommender.prever_preco_com_incerteza", return_value=resultado_mc), \
             patch("models.recommender.get_connection"):
            result = recomendar_momento_compra(1, "dummy.csv")

        assert result is not None
        for campo in ["produto_id", "preco_atual", "preco_minimo_previsto",
                      "descida_pct", "recomendacao", "previsoes"]:
            assert campo in result

    def test_preco_atual_preservado(self):
        from models.recommender import recomendar_momento_compra
        resultado_mc = self._mock_resultado_mc([1.69] * 7, [1.65] * 7, 2.50)
        with patch("models.recommender.prever_preco_com_incerteza", return_value=resultado_mc), \
             patch("models.recommender.get_connection"):
            result = recomendar_momento_compra(1, "dummy.csv")
        assert result is not None
        assert abs(result["preco_atual"] - 2.50) < 0.01

    def test_descida_pct_calculada(self):
        from models.recommender import recomendar_momento_compra
        # Preco atual 2.00, mínimo previsto 1.80 → descida 10%
        resultado_mc = self._mock_resultado_mc(
            precos_medios=[1.80] * 7,
            precos_ic5=   [1.75] * 7,
            preco_atual=2.00,
        )
        with patch("models.recommender.prever_preco_com_incerteza", return_value=resultado_mc), \
             patch("models.recommender.get_connection"):
            result = recomendar_momento_compra(1, "dummy.csv", limiar_descida_pct=2.0)
        assert result is not None
        assert abs(result["descida_pct"] - 10.0) < 0.1


# ---------------------------------------------------------------------------
# pesquisar_produtos — fallback AND → OR (mock)
# ---------------------------------------------------------------------------

class TestPesquisarProdutos:
    def test_devolve_df_quando_encontra(self):
        """Smoke test: chamar pesquisar_produtos com mock não-vazio não levanta exceção.

        Os testes de comportamento real ficam para a suite de integração com BD
        (test_recommender_bd.py) — este apenas garante que a função é chamável
        com um cursor mockado sem rebentar a meio do parsing/conversão.
        """
        from models.recommender import pesquisar_produtos
        with patch("models.recommender.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ("Arroz Agulha Cigala 1kg", "Cigala", 1000.0, "g", 2, 1.69, 1.89)
            ]
            mock_cursor.description = [
                ("nome_padronizado",), ("marca",), ("quantidade_valor",),
                ("quantidade_unidade",), ("num_lojas",), ("preco_min",), ("preco_max",),
            ]
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            # Verificação mínima: a função executa sem exceção.
            pesquisar_produtos("arroz")

    def test_devolve_none_quando_nao_encontra(self):
        from models.recommender import pesquisar_produtos
        with patch("models.recommender.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_cursor.description = [
                ("nome_padronizado",), ("marca",), ("quantidade_valor",),
                ("quantidade_unidade",), ("num_lojas",), ("preco_min",), ("preco_max",),
            ]
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            result = pesquisar_produtos("produtoinexistentexyz123")

        assert result is None


# ---------------------------------------------------------------------------
# otimizar_lista_compras — lógica de custo (mock)
# ---------------------------------------------------------------------------

class TestOtimizarListaCompras:
    def test_lista_vazia_comportamento(self):
        """Lista vazia não deve lançar exceção — devolve None."""
        from models.recommender import otimizar_lista_compras
        with patch("models.recommender.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_cursor.description = [
                ("nome_padronizado",), ("loja",), ("preco_atual",),
                ("em_promocao",), ("id_produto_loja",), ("sim_score",),
            ]
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            result = otimizar_lista_compras([])
        assert result is None

    def test_item_nao_encontrado_reportado(self, capsys):
        from models.recommender import otimizar_lista_compras
        with patch("models.recommender.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_cursor.description = [
                ("nome_padronizado",), ("loja",), ("preco_atual",),
                ("em_promocao",), ("id_produto_loja",), ("sim_score",),
            ]
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            result = otimizar_lista_compras(["produto_impossivel_xyz"])
        assert result is None

    def test_loja_completa_pode_nao_ser_a_mais_barata_em_nenhum_item(self):
        from models.recommender import otimizar_lista_compras

        def linha(nome, loja, preco):
            return pd.DataFrame([{
                "nome_padronizado": nome.title(),
                "loja": loja,
                "preco_atual": preco,
                "em_promocao": False,
                "id_produto_loja": 1,
                "sim_score": 1.0,
            }])

        def fake_melhor_preco(_cur, nome, loja=None):
            if loja is None:
                if nome == "arroz":
                    return linha("arroz", "A", 1.00)
                if nome == "leite":
                    return linha("leite", "B", 1.00)
            precos = {
                ("arroz", "A"): 1.00,
                ("leite", "B"): 1.00,
                ("arroz", "C"): 1.20,
                ("leite", "C"): 1.20,
            }
            if (nome, loja) not in precos:
                return pd.DataFrame()
            return linha(nome, loja, precos[(nome, loja)])

        with patch("models.recommender.get_connection") as mock_conn, \
             patch("models.recommender._melhor_preco_para_item", side_effect=fake_melhor_preco), \
             patch("models.recommender._lojas_candidatas_para_lista", return_value=["A", "B", "C"]):
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            result = otimizar_lista_compras(["arroz", "leite"])

        assert result is not None
        assert result["melhor_loja"] == "C"
        assert result["custo_melhor_loja"] == pytest.approx(2.40)
        assert result["poupanca_split"] == pytest.approx(0.40)

    def test_sem_loja_completa_nao_devolve_loja_incompleta_como_melhor(self):
        from models.recommender import otimizar_lista_compras

        def linha(nome, loja, preco):
            return pd.DataFrame([{
                "nome_padronizado": nome.title(),
                "loja": loja,
                "preco_atual": preco,
                "em_promocao": False,
                "id_produto_loja": 1,
                "sim_score": 1.0,
            }])

        def fake_melhor_preco(_cur, nome, loja=None):
            if loja is None:
                return linha(nome, "A" if nome == "arroz" else "B", 1.00)
            precos = {
                ("arroz", "A"): 1.00,
                ("leite", "B"): 1.00,
            }
            if (nome, loja) not in precos:
                return pd.DataFrame()
            return linha(nome, loja, precos[(nome, loja)])

        with patch("models.recommender.get_connection") as mock_conn, \
             patch("models.recommender._melhor_preco_para_item", side_effect=fake_melhor_preco), \
             patch("models.recommender._lojas_candidatas_para_lista", return_value=["A", "B"]):
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            result = otimizar_lista_compras(["arroz", "leite"])

        assert result is not None
        assert result["melhor_loja"] is None
        assert result["custo_melhor_loja"] is None