"""
Testes BD para :func:`models.recommender.produto_perto_de_mim`.

Cobre:
  - Caso típico: produto em todas as cadeias, ordenação por custo total.
  - Cadeia sem loja física no raio fica excluída do resultado.
  - Produto inexistente devolve ``None``.
  - Top N limita resultados.
  - Estrutura da resposta (campos obrigatórios).
"""

from __future__ import annotations

import pytest


UMINHO_LAT, UMINHO_LON = 41.561, -8.397


@pytest.fixture
def db_seed_produto_perto(db_clean):
    """1 produto em 3 cadeias com preços distintos e lojas físicas em Braga."""
    conn = db_clean
    cadeias_precos_loja = [
        # (insignia, preco, lat, lon)
        ("Continente", 1.69, 41.5409, -8.4004),
        ("Pingo Doce", 1.04, 41.5572, -8.4050),
        ("Auchan",     1.23, 41.5591, -8.4141),
    ]
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO produtos_mestre (chave_mestre, nome_padronizado, "
            "categoria_geral) VALUES ('ean:arr-1', 'Arroz Agulha Cigala 1000g', "
            "'mercearia') RETURNING id_produto_mestre"
        )
        id_pm = cur.fetchone()[0]
        for i, (ins, preco, lat, lon) in enumerate(cadeias_precos_loja, start=1):
            cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                        "VALUES (%s, 'Online', 'Nacional', 'online') RETURNING id_loja",
                        (ins,))
            id_l = cur.fetchone()[0]
            cur.execute("INSERT INTO produtos_loja (id_produto_mestre, id_loja, "
                        "sku_loja, nome_na_loja) VALUES (%s, %s, %s, 'Arroz') "
                        "RETURNING id_produto_loja", (id_pm, id_l, f"SKU-{i}"))
            id_pl = cur.fetchone()[0]
            cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, "
                        "em_promocao, data_recolha, agente_origem) "
                        "VALUES (%s, %s, FALSE, '2026-05-15', 'test')",
                        (id_pl, preco))
            cur.execute("INSERT INTO lojas_fisicas (insignia, nome_loja, latitude, "
                        "longitude, fonte, external_id) "
                        "VALUES (%s, %s, %s, %s, 'test', %s)",
                        (ins, f"{ins} Braga", lat, lon, f"test-{i}"))
    conn.commit()


class TestProdutoPertoDeMim:

    def test_devolve_3_cadeias_ordenadas_por_custo_total(self, db_seed_produto_perto):
        from models.recommender import produto_perto_de_mim
        result = produto_perto_de_mim(
            "arroz agulha", user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km=0.20, raio_km=30.0,
        )
        assert result is not None
        assert len(result) == 3

        # Verificar ordenação ascendente
        custos = [r["custo_total"] for r in result]
        assert custos == sorted(custos), "Deve estar ordenado por custo_total"

    def test_estrutura_da_resposta(self, db_seed_produto_perto):
        from models.recommender import produto_perto_de_mim
        result = produto_perto_de_mim(
            "arroz agulha", user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
        )
        assert result, "Esperava pelo menos 1 resultado"
        primeiro = result[0]
        for campo in ("insignia", "produto", "id_produto_loja",
                       "preco_atual", "em_promocao",
                       "loja_fisica", "distancia_km",
                       "custo_deslocacao", "custo_total"):
            assert campo in primeiro, f"Campo {campo!r} em falta"

        # loja_fisica deve ser dict
        assert isinstance(primeiro["loja_fisica"], dict)
        for sub in ("nome_loja", "latitude", "longitude"):
            assert sub in primeiro["loja_fisica"]

    def test_custo_total_e_preco_mais_deslocacao(self, db_seed_produto_perto):
        from models.recommender import produto_perto_de_mim
        result = produto_perto_de_mim(
            "arroz agulha", user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km=0.20,
        )
        for r in result:
            esperado = round(r["preco_atual"] + r["custo_deslocacao"], 2)
            assert abs(r["custo_total"] - esperado) < 0.01

    def test_cadeia_sem_loja_fisica_no_raio_excluida(self, db_clean):
        """Inserir 2 cadeias mas só 1 com loja física no raio → resultado tem 1 só."""
        from models.recommender import produto_perto_de_mim

        with db_clean.cursor() as cur:
            cur.execute("INSERT INTO produtos_mestre (chave_mestre, nome_padronizado, "
                        "categoria_geral) VALUES ('ean:x', 'Arroz Agulha 1kg', 'm') "
                        "RETURNING id_produto_mestre")
            id_pm = cur.fetchone()[0]

            for ins, preco, lat, lon in [
                ("Continente", 1.69, 41.5409, -8.4004),   # Braga, perto da UMinho
                ("Pingo Doce", 1.04, 37.0179, -7.9304),   # Faro, muito longe
            ]:
                cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                            "VALUES (%s, 'Online', 'Nacional', 'online') RETURNING id_loja",
                            (ins,))
                id_l = cur.fetchone()[0]
                cur.execute("INSERT INTO produtos_loja (id_produto_mestre, id_loja, "
                            "sku_loja, nome_na_loja) VALUES (%s, %s, %s, 'Arroz') "
                            "RETURNING id_produto_loja",
                            (id_pm, id_l, f"SKU-{ins}"))
                id_pl = cur.fetchone()[0]
                cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, "
                            "em_promocao, data_recolha, agente_origem) "
                            "VALUES (%s, %s, FALSE, '2026-05-15', 'test')",
                            (id_pl, preco))
                cur.execute("INSERT INTO lojas_fisicas (insignia, nome_loja, latitude, "
                            "longitude, fonte, external_id) "
                            "VALUES (%s, %s, %s, %s, 'test', %s)",
                            (ins, f"{ins} Loja", lat, lon, f"test-{ins}"))
        db_clean.commit()

        # Raio 30 km da UMinho — Pingo Doce em Faro fica fora
        result = produto_perto_de_mim(
            "arroz agulha", user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            raio_km=30.0,
        )
        assert result is not None
        insignias_encontradas = {r["insignia"] for r in result}
        assert insignias_encontradas == {"Continente"}

    def test_produto_inexistente_devolve_none(self, db_clean):
        from models.recommender import produto_perto_de_mim
        result = produto_perto_de_mim(
            "produto-inexistente-xpto",
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
        )
        assert result is None

    def test_top_n_limita_resultados(self, db_seed_produto_perto):
        from models.recommender import produto_perto_de_mim
        result = produto_perto_de_mim(
            "arroz agulha", user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            top_n=2,
        )
        assert len(result) == 2

    def test_preset_tarifa_at_aumenta_custo_deslocacao(self, db_seed_produto_perto):
        """Mudar preset de equilibrado para tarifa_at deve aumentar custo desl."""
        from models.recommender import produto_perto_de_mim
        result_eq = produto_perto_de_mim(
            "arroz agulha", user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km="equilibrado",   # 0.20
        )
        result_at = produto_perto_de_mim(
            "arroz agulha", user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km="tarifa_at",     # 0.36
        )
        for r_eq, r_at in zip(result_eq, result_at):
            # Mesma cadeia, mesma distância — só o custo_deslocacao muda
            if r_eq["insignia"] == r_at["insignia"]:
                assert r_at["custo_deslocacao"] > r_eq["custo_deslocacao"]
                assert r_at["custo_total"] > r_eq["custo_total"]
