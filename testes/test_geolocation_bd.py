"""
Testes de integração com BD para models/geolocation.py.

Cobre:
  - lojas_proximas (filtragem por bbox + haversine, ordenação por distância)
  - distancia_minima_por_insignia (loja mais próxima por cadeia)

Requer a BD ``products_db_test`` configurada (ver instruções em conftest.py).
Usa a fixture ``db_seed_lojas_fisicas`` que insere 4 lojas em coordenadas
conhecidas (2 em Braga, 1 em Porto, 1 em Lisboa).
"""

from __future__ import annotations


# Coordenadas de referência para os testes:
UMINHO_LAT, UMINHO_LON = 41.561, -8.397   # Campus Gualtar
LISBOA_LAT, LISBOA_LON = 38.7259, -9.1500  # Marquês de Pombal
PORTO_LAT, PORTO_LON   = 41.1579, -8.6291  # Boavista


class TestLojasProximas:
    """Testes para ``lojas_proximas`` — versão com BD real."""

    def test_devolve_lojas_dentro_do_raio(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        # UMinho com raio 5 km: deve apanhar as 2 lojas de Braga
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=5.0)
        assert len(proximas) == 2
        cidades = {loja["cidade"] for loja in proximas}
        assert cidades == {"Braga"}

    def test_ordenadas_por_distancia_ascendente(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=500.0)
        # Devem aparecer todas (3 cobertas pelo raio: Braga × 2 + Porto)
        # Lisboa fica fora porque está a >300 km
        distancias = [loja["distancia_km"] for loja in proximas]
        assert distancias == sorted(distancias), "Não ordenado por distância"

    def test_raio_grande_apanha_lisboa(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        # 500 km a partir da UMinho cobre Lisboa (~330 km)
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=500.0)
        cidades = {loja["cidade"] for loja in proximas}
        assert "Lisboa" in cidades

    def test_filtragem_por_insignia(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        # Continente: tem 2 lojas no seed (Braga + Lisboa)
        cont = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=500.0, insignia="Continente")
        assert all(loja["insignia"] == "Continente" for loja in cont)
        assert len(cont) == 2

    def test_filtragem_por_insignia_inexistente(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=500.0,
                                   insignia="LidlInventado")
        assert proximas == []

    def test_limite_corta_resultados(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=500.0, limite=2)
        assert len(proximas) == 2

    def test_raio_pequeno_devolve_vazio_quando_longe(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        # Faro: nenhuma loja do seed está perto (~500 km até à mais próxima)
        proximas = lojas_proximas(37.0179, -7.9304, raio_km=10.0)
        assert proximas == []

    def test_lojas_inativas_excluidas(self, db_clean, db_seed_lojas_fisicas):
        """Lojas com ativa=FALSE não devem aparecer."""
        from models.geolocation import lojas_proximas
        # Marcar todas as lojas como inativas
        with db_clean.cursor() as cur:
            cur.execute("UPDATE lojas_fisicas SET ativa = FALSE")
        db_clean.commit()

        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=500.0)
        assert proximas == []

    def test_campos_devolvidos(self, db_seed_lojas_fisicas):
        from models.geolocation import lojas_proximas
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=5.0)
        assert proximas, "Esperava pelo menos 1 loja"
        loja = proximas[0]
        for campo in ("id_loja_fisica", "insignia", "nome_loja", "morada",
                      "cidade", "codigo_postal", "latitude", "longitude",
                      "distancia_km"):
            assert campo in loja, f"Campo '{campo}' em falta"


class TestDistanciaMinimaPorInsignia:
    """Testes para ``distancia_minima_por_insignia`` — versão com BD real."""

    def test_loja_mais_proxima_de_cada_cadeia(self, db_seed_lojas_fisicas):
        from models.geolocation import distancia_minima_por_insignia
        # A partir de UMinho:
        # - Continente Braga é mais perto que Continente Lisboa
        # - Pingo Doce Braga é a única → é a mais próxima
        # - Auchan Porto é a única → é a mais próxima
        result = distancia_minima_por_insignia(
            UMINHO_LAT, UMINHO_LON,
            insignias=["Continente", "Pingo Doce", "Auchan"],
            raio_km=500.0,
        )
        assert result["Continente"]["nome_loja"] == "Continente Braga"
        assert result["Pingo Doce"]["nome_loja"] == "Pingo Doce Braga Hiper"
        assert result["Auchan"]["nome_loja"] == "Auchan Porto Boavista"

    def test_cadeia_sem_loja_no_raio_devolve_none(self, db_seed_lojas_fisicas):
        from models.geolocation import distancia_minima_por_insignia
        # Faro, raio 10 km: nenhuma cadeia tem loja no seed perto
        result = distancia_minima_por_insignia(
            37.0179, -7.9304,
            insignias=["Continente", "Pingo Doce", "Auchan"],
            raio_km=10.0,
        )
        assert all(v is None for v in result.values())

    def test_insignia_inexistente_devolve_none(self, db_seed_lojas_fisicas):
        from models.geolocation import distancia_minima_por_insignia
        result = distancia_minima_por_insignia(
            UMINHO_LAT, UMINHO_LON,
            insignias=["Lidl"],   # não existe no seed
            raio_km=500.0,
        )
        assert result == {"Lidl": None}

    def test_distancias_coerentes(self, db_seed_lojas_fisicas):
        """A partir de UMinho, Pingo Doce Braga (~0.8 km) deve estar mais perto
        que Continente Braga (~2.2 km) — confirma cálculo haversine."""
        from models.geolocation import distancia_minima_por_insignia
        result = distancia_minima_por_insignia(
            UMINHO_LAT, UMINHO_LON,
            insignias=["Pingo Doce", "Continente"],
            raio_km=500.0,
        )
        assert result["Pingo Doce"]["distancia_km"] < result["Continente"]["distancia_km"]


class TestBboxFiltragem:
    """Verifica que o pré-filtro por bounding box em SQL não exclui falsos negativos."""

    def test_loja_na_fronteira_da_bbox_e_incluida(self, db_clean):
        """Loja exatamente na fronteira do raio deve ser incluída."""
        from models.geolocation import lojas_proximas, haversine_km

        # Inserir loja a ~10 km exactos da UMinho
        # 0.09° lat ≈ 10 km
        lat_loja = UMINHO_LAT + 0.0901
        lon_loja = UMINHO_LON
        dist_real = haversine_km(UMINHO_LAT, UMINHO_LON, lat_loja, lon_loja)

        with db_clean.cursor() as cur:
            cur.execute("""
                INSERT INTO lojas_fisicas
                    (insignia, nome_loja, latitude, longitude, fonte, external_id)
                VALUES ('Continente', 'Fronteira', %s, %s, 'test', 'edge-1')
            """, (lat_loja, lon_loja))
        db_clean.commit()

        # Raio ligeiramente acima da distância real → deve ser incluída
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=dist_real + 0.5)
        assert len(proximas) == 1
        assert proximas[0]["nome_loja"] == "Fronteira"

        # Raio ligeiramente abaixo → não deve ser incluída
        proximas = lojas_proximas(UMINHO_LAT, UMINHO_LON, raio_km=dist_real - 0.5)
        assert proximas == []
