"""
Testes para scrapers/lojas_fisicas_scraper.py.

Cobre a normalização de lojas SFCC (Continente/Pingo Doce) e Auchan, validação
de coordenadas e parsing do snippet HTML do Auchan — tudo sem fazer pedidos
de rede (usando dicts/HTML sintético).
"""

from __future__ import annotations

import pytest

from scrapers.lojas_fisicas_scraper import (
    _has_valid_coords,
    _normalize_auchan_store,
    _normalize_sfcc_store,
    _parse_auchan_info_window,
    CADEIAS_DISPONIVEIS,
)


# ---------------------------------------------------------------------------
# _has_valid_coords
# ---------------------------------------------------------------------------

class TestHasValidCoords:
    def test_coords_validas_em_pt_continental(self):
        assert _has_valid_coords({"latitude": 41.56, "longitude": -8.40}) is True

    def test_coords_validas_madeira(self):
        assert _has_valid_coords({"latitude": 32.65, "longitude": -16.91}) is True

    def test_coords_validas_acores(self):
        assert _has_valid_coords({"latitude": 37.74, "longitude": -25.66}) is True

    def test_coords_none_rejeitadas(self):
        assert _has_valid_coords({"latitude": None, "longitude": None}) is False
        assert _has_valid_coords({"latitude": 41.5, "longitude": None}) is False
        assert _has_valid_coords({}) is False

    def test_coords_string_numerica_aceite(self):
        # Algumas APIs SFCC devolvem coordenadas como string — o scraper aceita.
        assert _has_valid_coords({"latitude": "41.56", "longitude": "-8.40"}) is True

    def test_coords_nao_numericas_rejeitadas(self):
        assert _has_valid_coords({"latitude": "abc", "longitude": "-8.40"}) is False

    def test_coords_fora_de_pt_rejeitadas(self):
        # Madrid: dentro do range global mas fora do bbox de PT
        assert _has_valid_coords({"latitude": 40.4168, "longitude": -3.7038}) is False
        # Polo Norte
        assert _has_valid_coords({"latitude": 80.0, "longitude": 0.0}) is False


# ---------------------------------------------------------------------------
# _normalize_sfcc_store
# ---------------------------------------------------------------------------

class TestNormalizeSfccStore:
    @pytest.fixture
    def store_sfcc_completa(self):
        """Exemplo de loja SFCC (Continente/Pingo Doce) com todos os campos."""
        return {
            "ID": "Zu-5475-store",
            "name": "Zu Morais Soares",
            "address1": "R. Morais Soares",
            "address2": "Loja 1",
            "city": "Lisboa",
            "postalCode": "1900-349",
            "latitude": 38.7334,
            "longitude": -9.131994,
            "phone": "210112701",
            "stateCode": "Lisboa",
            "storeHours": "Seg a Sexta: 09h00/20h00",
        }

    def test_campos_obrigatorios(self, store_sfcc_completa):
        norm = _normalize_sfcc_store(
            store_sfcc_completa,
            insignia="Continente",
            fonte="scraper:continente",
        )
        assert norm["insignia"] == "Continente"
        assert norm["nome_loja"] == "Zu Morais Soares"
        assert norm["latitude"] == 38.7334
        assert norm["longitude"] == -9.131994
        assert norm["external_id"] == "Zu-5475-store"
        assert norm["fonte"] == "scraper:continente"

    def test_morada_compoe_address1_e_address2(self, store_sfcc_completa):
        norm = _normalize_sfcc_store(store_sfcc_completa, "Continente", "scraper:continente")
        assert norm["morada"] == "R. Morais Soares, Loja 1"

    def test_morada_sem_address2_apenas_address1(self):
        store = {
            "ID": "X", "name": "X", "address1": "Rua A", "address2": None,
            "latitude": 41.5, "longitude": -8.4,
        }
        norm = _normalize_sfcc_store(store, "Continente", "scraper:continente")
        assert norm["morada"] == "Rua A"

    def test_morada_vazia_devolve_none(self):
        store = {
            "ID": "X", "name": "X", "address1": "", "address2": "",
            "latitude": 41.5, "longitude": -8.4,
        }
        norm = _normalize_sfcc_store(store, "Pingo Doce", "scraper:pingo_doce")
        assert norm["morada"] is None

    def test_telefone_e_horario_opcionais(self):
        store = {
            "ID": "X", "name": "X", "address1": "Rua A",
            "latitude": 41.5, "longitude": -8.4,
            "phone": None, "storeHours": None,
        }
        norm = _normalize_sfcc_store(store, "Pingo Doce", "scraper:pingo_doce")
        assert norm["telefone"] is None
        assert norm["horario"] is None

    def test_distrito_vem_de_state_code(self, store_sfcc_completa):
        norm = _normalize_sfcc_store(store_sfcc_completa, "Continente", "scraper:continente")
        assert norm["distrito"] == "Lisboa"

    def test_nome_default_quando_em_falta(self):
        store = {
            "ID": "X", "name": None,
            "latitude": 41.5, "longitude": -8.4,
        }
        norm = _normalize_sfcc_store(store, "Continente", "scraper:continente")
        assert norm["nome_loja"] == "(sem nome)"


# ---------------------------------------------------------------------------
# _parse_auchan_info_window
# ---------------------------------------------------------------------------

class TestParseAuchanInfoWindow:
    def test_extrai_morada_cp_cidade_e_id(self):
        info_html = """
        <div class="store-details" data-store-id="B19">
            <div class="store-name">My Auchan D Dinis - Odivelas</div>
            <address>
                Rua D. Dinis, n° 39 B
                Odivelas, Odivelas  2675-332
            </address>
        </div>
        """
        morada, cp, cidade, ext_id = _parse_auchan_info_window(info_html)
        assert ext_id == "B19"
        assert cp == "2675-332"
        # Morada e cidade: a heurística separa pela última vírgula
        assert cidade == "Odivelas"
        assert "Rua D. Dinis" in morada

    def test_html_sem_address_devolve_nones(self):
        info_html = '<div data-store-id="X1"><div>Sem endereço</div></div>'
        morada, cp, cidade, ext_id = _parse_auchan_info_window(info_html)
        assert ext_id == "X1"
        assert morada is None
        assert cp is None
        assert cidade is None

    def test_html_completamente_vazio_devolve_tudo_none(self):
        morada, cp, cidade, ext_id = _parse_auchan_info_window("")
        assert ext_id == ""
        assert morada is None
        assert cp is None
        assert cidade is None

    def test_codigo_postal_pt_corretamente_extraido(self):
        info_html = """
        <div data-store-id="A1">
            <address>Rua X, Vila Y 4700-100</address>
        </div>
        """
        morada, cp, cidade, ext_id = _parse_auchan_info_window(info_html)
        assert cp == "4700-100"


# ---------------------------------------------------------------------------
# _normalize_auchan_store
# ---------------------------------------------------------------------------

class TestNormalizeAuchanStore:
    @pytest.fixture
    def store_auchan_completa(self):
        """Loja Auchan no formato do data-locations."""
        return {
            "name": "My Auchan D Dinis - Odivelas",
            "latitude": 38.788,
            "longitude": -9.1788,
            "type": "Auchan",
            "infoWindowHtml": (
                '<div class="store-details" data-store-id="B19">'
                '<div class="store-name">My Auchan D Dinis - Odivelas</div>'
                '<address>Rua D. Dinis, n° 39 B Odivelas, Odivelas  2675-332</address>'
                '</div>'
            ),
            "pickAndGoMarker": False,
        }

    def test_loja_completa_normalizada(self, store_auchan_completa):
        norm = _normalize_auchan_store(store_auchan_completa)
        assert norm is not None
        assert norm["insignia"] == "Auchan"
        assert norm["nome_loja"] == "My Auchan D Dinis - Odivelas"
        assert norm["latitude"] == 38.788
        assert norm["longitude"] == -9.1788
        assert norm["external_id"] == "B19"
        assert norm["codigo_postal"] == "2675-332"
        assert norm["fonte"] == "scraper:auchan"

    def test_telefone_e_horario_sempre_none(self, store_auchan_completa):
        """Auchan não expõe estes campos no data-locations."""
        norm = _normalize_auchan_store(store_auchan_completa)
        assert norm["telefone"] is None
        assert norm["horario"] is None
        assert norm["distrito"] is None

    def test_loja_sem_coords_rejeitada(self):
        store = {
            "name": "Sem Coords",
            "latitude": None,
            "longitude": None,
            "infoWindowHtml": "",
        }
        assert _normalize_auchan_store(store) is None


# ---------------------------------------------------------------------------
# CADEIAS_DISPONIVEIS — sanidade do registo
# ---------------------------------------------------------------------------

class TestCadeiasDisponiveis:
    def test_tres_cadeias_registadas(self):
        assert set(CADEIAS_DISPONIVEIS.keys()) == {"continente", "pingo_doce", "auchan"}

    def test_todas_as_funcoes_sao_callable(self):
        for fn in CADEIAS_DISPONIVEIS.values():
            assert callable(fn)
