"""
Testes para a função GPS-aware do recommender: ``otimizar_lista_compras_geo``.

Usa ``unittest.mock`` para isolar a função da BD e do scraper de lojas físicas,
testando exclusivamente a lógica de:
  - Cálculo de custo_total = custo_produtos + 2 × distância × custo_km
  - Seleção da melhor cadeia (custo_total mínimo entre alcançáveis)
  - Tratamento de cadeias sem loja física no raio
  - Tratamento de cadeias com lista incompleta
  - Resolução de presets de custo_km
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures partilhadas
# ---------------------------------------------------------------------------

@pytest.fixture
def base_otimizar_lista():
    """Resultado típico de otimizar_lista_compras (sem GPS) para mockar."""
    return {
        "itens": [
            {"item_pesquisado": "arroz", "nome_padronizado": "Arroz Cigala 1kg",
             "loja": "Auchan", "preco_atual": 1.38, "em_promocao": False,
             "id_produto_loja": 100},
            {"item_pesquisado": "azeite", "nome_padronizado": "Azeite Gallo 750ml",
             "loja": "Pingo Doce", "preco_atual": 7.99, "em_promocao": False,
             "id_produto_loja": 200},
        ],
        "custo_minimo": 9.37,
        "melhor_loja": "Auchan",
        "custo_melhor_loja": 10.10,
        "poupanca_split": 0.73,
        "detalhe_por_loja": {
            "Auchan":     {"total": 10.10, "em_falta": []},
            "Pingo Doce": {"total": 11.50, "em_falta": []},
            "Continente": {"total": 12.20, "em_falta": []},
        },
    }


@pytest.fixture
def distancias_mock():
    """Distâncias típicas a partir da UMinho (Braga)."""
    return {
        "Auchan":     {"id_loja_fisica": 1, "nome_loja": "Auchan Braga",
                       "morada": "R. X", "cidade": "Braga",
                       "latitude": 41.559, "longitude": -8.414,
                       "distancia_km": 1.42},
        "Pingo Doce": {"id_loja_fisica": 2, "nome_loja": "PD Braga Hiper",
                       "morada": "R. Y", "cidade": "Braga",
                       "latitude": 41.557, "longitude": -8.405,
                       "distancia_km": 0.78},
        "Continente": {"id_loja_fisica": 3, "nome_loja": "Continente Braga",
                       "morada": "R. Z", "cidade": "Braga",
                       "latitude": 41.541, "longitude": -8.400,
                       "distancia_km": 2.25},
    }


# ---------------------------------------------------------------------------
# Casos de teste — usando patch para mockar BD
# ---------------------------------------------------------------------------

class TestOtimizarListaComprasGeo:

    def _patch_dependencias(self, base, distancias):
        """Aplica os dois patches necessários para isolar da BD."""
        return (
            patch("models.recommender.otimizar_lista_compras", return_value=base),
            patch("models.recommender.distancia_minima_por_insignia", return_value=distancias),
        )

    def test_calcula_custo_total_correto(self, base_otimizar_lista, distancias_mock):
        from models.recommender import otimizar_lista_compras_geo

        p1, p2 = self._patch_dependencias(base_otimizar_lista, distancias_mock)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz", "azeite"],
                user_lat=41.561, user_lon=-8.397,
                custo_km="equilibrado",   # 0.20 €/km
            )

        # Verificar custo_total = produtos + 2 × distância × custo_km
        # Auchan: 10.10 + 2*1.42*0.20 = 10.10 + 0.568 = 10.67 (≈ 10.668 → 10.67)
        auchan = result["detalhe_por_cadeia"]["Auchan"]
        assert auchan["custo_produtos"] == 10.10
        assert auchan["distancia_km"]   == 1.42
        assert auchan["custo_deslocacao"] == 0.57   # round(2*1.42*0.20, 2) = 0.57
        # 10.10 + 0.57 = 10.67
        assert auchan["custo_total"]    == 10.67
        assert auchan["alcancavel"] is True

    def test_seleciona_cadeia_com_custo_total_minimo(self, base_otimizar_lista, distancias_mock):
        """O Auchan tem produtos mais baratos e a sua deslocação (1.42 km) é
        compensada pelos preços. Deve ser a recomendação."""
        from models.recommender import otimizar_lista_compras_geo

        p1, p2 = self._patch_dependencias(base_otimizar_lista, distancias_mock)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz", "azeite"],
                user_lat=41.561, user_lon=-8.397,
                custo_km=0.20,
            )

        # Cálculos esperados:
        # Auchan:     10.10 + 0.57 = 10.67
        # Pingo Doce: 11.50 + 0.31 = 11.81
        # Continente: 12.20 + 0.90 = 13.10
        # → Auchan ganha
        assert result["melhor_opcao"]["insignia"] == "Auchan"
        assert result["melhor_opcao"]["custo_total"] == 10.67

    def test_preset_caro_pode_mudar_a_recomendacao(self, base_otimizar_lista, distancias_mock):
        """Cenário construído onde Pingo Doce tem produtos mais caros que
        Auchan mas está MUITO mais perto. Com tarifa_at (0.36€/km) podia
        compensar? Construindo manualmente:

        Auchan:     10.10 + 2*1.42*0.36 = 10.10 + 1.02 = 11.12
        Pingo Doce: 11.50 + 2*0.78*0.36 = 11.50 + 0.56 = 12.06

        Mesmo com tarifa_at, Auchan ainda ganha aqui (diferença produtos
        é 1.40€, diferença deslocação só 0.46€). Confirmamos isto.
        """
        from models.recommender import otimizar_lista_compras_geo

        p1, p2 = self._patch_dependencias(base_otimizar_lista, distancias_mock)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz", "azeite"],
                user_lat=41.561, user_lon=-8.397,
                custo_km="tarifa_at",   # 0.36
            )

        # Verificar custo_km efectivo registado
        assert result["custo_km"] == 0.36
        # Auchan ainda ganha — diferença de produtos não é compensável pela distância
        assert result["melhor_opcao"]["insignia"] == "Auchan"

    def test_cadeia_sem_loja_no_raio_marcada_como_inalcancavel(self, base_otimizar_lista):
        """Se a distancia_minima_por_insignia devolver None para uma cadeia,
        essa cadeia não entra na escolha mas aparece em detalhe_por_cadeia."""
        from models.recommender import otimizar_lista_compras_geo

        # Auchan sem loja no raio
        distancias = {
            "Auchan":     None,
            "Pingo Doce": {"id_loja_fisica": 2, "nome_loja": "PD",
                           "latitude": 41.557, "longitude": -8.405,
                           "distancia_km": 0.78},
            "Continente": {"id_loja_fisica": 3, "nome_loja": "Cont",
                           "latitude": 41.541, "longitude": -8.400,
                           "distancia_km": 2.25},
        }
        p1, p2 = self._patch_dependencias(base_otimizar_lista, distancias)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz"],
                user_lat=41.561, user_lon=-8.397,
                custo_km=0.20,
            )

        assert result["detalhe_por_cadeia"]["Auchan"]["alcancavel"] is False
        assert result["detalhe_por_cadeia"]["Auchan"]["custo_total"] is None
        # Recomendação não deve ser Auchan
        assert result["melhor_opcao"]["insignia"] in ("Pingo Doce", "Continente")

    def test_lista_incompleta_exclui_cadeia_da_recomendacao(self, distancias_mock):
        """Cadeia com em_falta != [] não deve ser recomendada, mesmo que
        tenha custo_total mais baixo."""
        from models.recommender import otimizar_lista_compras_geo

        # Auchan barato mas com 1 item em falta — deve ser excluído
        base = {
            "itens": [],
            "custo_minimo": 8.0,
            "melhor_loja": "Pingo Doce",
            "custo_melhor_loja": 11.50,
            "poupanca_split": 0.0,
            "detalhe_por_loja": {
                "Auchan":     {"total": 8.00,  "em_falta": ["atum"]},   # incompleto
                "Pingo Doce": {"total": 11.50, "em_falta": []},
                "Continente": {"total": 12.20, "em_falta": []},
            },
        }

        p1, p2 = self._patch_dependencias(base, distancias_mock)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz", "azeite", "atum"],
                user_lat=41.561, user_lon=-8.397,
                custo_km=0.20,
            )

        # Auchan tem o menor custo_total mas está incompleto
        # → Pingo Doce ganha (11.50 + 0.31 = 11.81)
        assert result["melhor_opcao"]["insignia"] == "Pingo Doce"
        # Auchan continua em detalhe_por_cadeia (para visualização)
        assert "Auchan" in result["detalhe_por_cadeia"]
        assert result["detalhe_por_cadeia"]["Auchan"]["em_falta"] == ["atum"]

    def test_nenhuma_cadeia_completa_devolve_melhor_opcao_none(self, distancias_mock):
        """Se TODAS as cadeias tiverem lista incompleta, melhor_opcao é None."""
        from models.recommender import otimizar_lista_compras_geo

        base = {
            "itens": [],
            "custo_minimo": 5.0,
            "melhor_loja": None,
            "custo_melhor_loja": None,
            "poupanca_split": 0.0,
            "detalhe_por_loja": {
                "Auchan":     {"total": 5.0, "em_falta": ["item-raro"]},
                "Pingo Doce": {"total": 6.0, "em_falta": ["item-raro"]},
            },
        }
        distancias = {k: distancias_mock[k] for k in ("Auchan", "Pingo Doce")}

        p1, p2 = self._patch_dependencias(base, distancias)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["item-raro"], user_lat=41.561, user_lon=-8.397,
                custo_km=0.20,
            )
        assert result["melhor_opcao"] is None

    def test_preserva_dados_da_base(self, base_otimizar_lista, distancias_mock):
        """itens e custo_minimo_split_multi_loja vêm da base sem alteração."""
        from models.recommender import otimizar_lista_compras_geo

        p1, p2 = self._patch_dependencias(base_otimizar_lista, distancias_mock)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz"], user_lat=41.561, user_lon=-8.397,
            )

        assert result["itens"] == base_otimizar_lista["itens"]
        assert result["custo_minimo_split_multi_loja"] == base_otimizar_lista["custo_minimo"]

    def test_resolve_custo_km_default_se_none(self, base_otimizar_lista, distancias_mock):
        """custo_km=None deve usar o preset 'equilibrado' (0.20)."""
        from models.recommender import otimizar_lista_compras_geo

        p1, p2 = self._patch_dependencias(base_otimizar_lista, distancias_mock)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz"], user_lat=41.561, user_lon=-8.397,
                custo_km=None,
            )
        assert result["custo_km"] == 0.20

    def test_eco_da_localizacao_e_raio_no_resultado(self, base_otimizar_lista, distancias_mock):
        from models.recommender import otimizar_lista_compras_geo

        p1, p2 = self._patch_dependencias(base_otimizar_lista, distancias_mock)
        with p1, p2:
            result = otimizar_lista_compras_geo(
                lista=["arroz"], user_lat=38.7223, user_lon=-9.1393,
                raio_km=50.0,
            )

        assert result["localizacao_utilizador"] == {"lat": 38.7223, "lon": -9.1393}
        assert result["raio_km"] == 50.0

    def test_base_none_propaga_none(self, distancias_mock):
        """Se nenhum item for encontrado pela função base, devolvemos None."""
        from models.recommender import otimizar_lista_compras_geo

        with patch("models.recommender.otimizar_lista_compras", return_value=None):
            result = otimizar_lista_compras_geo(
                lista=["item-inexistente"], user_lat=41.5, user_lon=-8.4,
            )
        assert result is None
