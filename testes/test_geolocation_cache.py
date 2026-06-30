"""
Testes para a cache TTL de ``models/geolocation.lojas_proximas``.

Foca os aspectos puros da cache (sem precisar de chamar a função real):
  - Construção da chave (arredondamento de coordenadas).
  - TTL: entradas expiram após ``CACHE_TTL_SEGUNDOS``.
  - ``limpar_cache_lojas_proximas`` esvazia a cache.

Os testes de cache hit/miss end-to-end com BD ficam em ``test_geolocation_bd.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _limpar_cache():
    """Garante que cada teste começa com cache vazia."""
    from models.geolocation import limpar_cache_lojas_proximas
    limpar_cache_lojas_proximas()
    yield
    limpar_cache_lojas_proximas()


class TestCacheKey:
    def test_lat_lon_arredondam_para_4_casas(self):
        from models.geolocation import _cache_key_lojas_proximas
        # 0.00005 abaixo da precisão (5ª casa decimal) — deve arredondar igual
        k1 = _cache_key_lojas_proximas(41.5610, -8.3970, 10.0, None, 50)
        k2 = _cache_key_lojas_proximas(41.56099, -8.39701, 10.0, None, 50)
        assert k1 == k2, "Coords sub-métricas devem cair na mesma chave"

    def test_lat_diferentes_em_4a_casa_dao_chaves_distintas(self):
        from models.geolocation import _cache_key_lojas_proximas
        k1 = _cache_key_lojas_proximas(41.5610, -8.3970, 10.0, None, 50)
        k2 = _cache_key_lojas_proximas(41.5611, -8.3970, 10.0, None, 50)
        assert k1 != k2

    def test_raio_diferente_chaves_distintas(self):
        from models.geolocation import _cache_key_lojas_proximas
        k1 = _cache_key_lojas_proximas(41.561, -8.397, 10.0, None, 50)
        k2 = _cache_key_lojas_proximas(41.561, -8.397, 20.0, None, 50)
        assert k1 != k2

    def test_insignia_diferente_chaves_distintas(self):
        from models.geolocation import _cache_key_lojas_proximas
        k1 = _cache_key_lojas_proximas(41.561, -8.397, 10.0, None, 50)
        k2 = _cache_key_lojas_proximas(41.561, -8.397, 10.0, "Continente", 50)
        assert k1 != k2


class TestCacheHitMiss:
    """Testa a lógica de cache hit/miss mockando a query à BD."""

    def test_segunda_chamada_nao_toca_na_bd(self):
        """2 chamadas com mesmos argumentos → BD chamada só 1 vez."""
        from models.geolocation import lojas_proximas

        # Mockar psycopg2.connect dentro do módulo para evitar ir à BD
        with patch("models.geolocation.psycopg2.connect") as mock_connect:
            cur = mock_connect.return_value.__enter__.return_value.cursor.return_value
            cur_ctx = cur.__enter__.return_value
            cur_ctx.description = [
                ("id_loja_fisica",), ("insignia",), ("nome_loja",), ("morada",),
                ("cidade",), ("codigo_postal",), ("latitude",), ("longitude",),
            ]
            cur_ctx.fetchall.return_value = [
                (1, "Continente", "Cont Braga", "R. X", "Braga", "4700-001",
                 41.5409, -8.4004),
            ]

            # 1ª chamada → cache miss → vai à BD
            r1 = lojas_proximas(41.561, -8.397, raio_km=10.0)
            # 2ª chamada → cache hit → NÃO vai à BD
            r2 = lojas_proximas(41.561, -8.397, raio_km=10.0)

            assert r1 == r2
            # psycopg2.connect só foi chamado uma vez
            assert mock_connect.call_count == 1

    def test_argumentos_diferentes_resultam_em_2_misses(self):
        from models.geolocation import lojas_proximas

        with patch("models.geolocation.psycopg2.connect") as mock_connect:
            cur = mock_connect.return_value.__enter__.return_value.cursor.return_value
            cur_ctx = cur.__enter__.return_value
            cur_ctx.description = [
                ("id_loja_fisica",), ("insignia",), ("nome_loja",), ("morada",),
                ("cidade",), ("codigo_postal",), ("latitude",), ("longitude",),
            ]
            cur_ctx.fetchall.return_value = []

            lojas_proximas(41.561, -8.397, raio_km=10.0)
            lojas_proximas(41.561, -8.397, raio_km=20.0)   # raio diferente
            assert mock_connect.call_count == 2

    def test_cache_devolve_copia_e_nao_referencia(self):
        """Mutar o resultado não deve poluir a cache."""
        from models.geolocation import lojas_proximas

        with patch("models.geolocation.psycopg2.connect") as mock_connect:
            cur = mock_connect.return_value.__enter__.return_value.cursor.return_value
            cur_ctx = cur.__enter__.return_value
            cur_ctx.description = [
                ("id_loja_fisica",), ("insignia",), ("nome_loja",), ("morada",),
                ("cidade",), ("codigo_postal",), ("latitude",), ("longitude",),
            ]
            cur_ctx.fetchall.return_value = [
                (1, "Continente", "Cont Braga", "R. X", "Braga", "4700-001",
                 41.5409, -8.4004),
            ]

            r1 = lojas_proximas(41.561, -8.397, raio_km=10.0)
            # Mutar agressivamente o resultado
            r1[0]["nome_loja"] = "ALTERADO"
            r1.clear()

            # 2ª chamada deve devolver o original, não a versão mutada
            r2 = lojas_proximas(41.561, -8.397, raio_km=10.0)
            assert len(r2) == 1
            assert r2[0]["nome_loja"] == "Cont Braga"


class TestTTL:
    def test_entrada_expira_apos_ttl(self):
        """Forçando o tempo a avançar, a entrada deve expirar e re-querer a BD."""
        from models.geolocation import lojas_proximas

        with patch("models.geolocation.psycopg2.connect") as mock_connect, \
             patch("models.geolocation.time.monotonic") as mock_time:
            cur = mock_connect.return_value.__enter__.return_value.cursor.return_value
            cur_ctx = cur.__enter__.return_value
            cur_ctx.description = [
                ("id_loja_fisica",), ("insignia",), ("nome_loja",), ("morada",),
                ("cidade",), ("codigo_postal",), ("latitude",), ("longitude",),
            ]
            cur_ctx.fetchall.return_value = []

            # 1ª chamada @ t=0 → miss
            mock_time.return_value = 0.0
            lojas_proximas(41.561, -8.397, raio_km=10.0)
            assert mock_connect.call_count == 1

            # 2ª chamada @ t=299s → ainda dentro do TTL → hit
            mock_time.return_value = 299.0
            lojas_proximas(41.561, -8.397, raio_km=10.0)
            assert mock_connect.call_count == 1

            # 3ª chamada @ t=301s → fora do TTL (300s) → miss
            mock_time.return_value = 301.0
            lojas_proximas(41.561, -8.397, raio_km=10.0)
            assert mock_connect.call_count == 2


class TestLimparCache:
    def test_limpar_cache_forca_proxima_chamada_a_ir_a_bd(self):
        from models.geolocation import lojas_proximas, limpar_cache_lojas_proximas

        with patch("models.geolocation.psycopg2.connect") as mock_connect:
            cur = mock_connect.return_value.__enter__.return_value.cursor.return_value
            cur_ctx = cur.__enter__.return_value
            cur_ctx.description = [
                ("id_loja_fisica",), ("insignia",), ("nome_loja",), ("morada",),
                ("cidade",), ("codigo_postal",), ("latitude",), ("longitude",),
            ]
            cur_ctx.fetchall.return_value = []

            lojas_proximas(41.561, -8.397, raio_km=10.0)   # miss
            lojas_proximas(41.561, -8.397, raio_km=10.0)   # hit
            assert mock_connect.call_count == 1

            limpar_cache_lojas_proximas()

            lojas_proximas(41.561, -8.397, raio_km=10.0)   # miss novamente
            assert mock_connect.call_count == 2

    def test_estado_inicial_da_cache_e_vazio(self):
        from models.geolocation import _cache_lojas_proximas
        assert _cache_lojas_proximas == {}
