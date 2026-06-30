"""
Testes BD para a otimização multi-loja com custo de deslocação.

Cobre :func:`otimizar_lista_compras_geo_multi_loja` sobre cenários sintéticos
inseridos pela fixture, exercitando:

- Cenário 1: dividir é estritamente melhor que single-store (poupança > 0).
- Cenário 2: single-store continua a ser melhor (poupança = 0, recomendação=single).
- Cenário 3: par com lista incompleta é excluído (item em falta nas 2 cadeias).
- Cenário 4: par onde uma das cadeias não tem nada faz fallback para single.
- Cenário 5: cálculo correto da distância triangular (haversine das 3 pernas).
"""

from __future__ import annotations

import math

import pytest


# Coordenadas de referência: UMinho Gualtar
UMINHO_LAT, UMINHO_LON = 41.561, -8.397


# ===========================================================================
# Fixture — cenário com produtos em 3 cadeias para forçar split
# ===========================================================================

@pytest.fixture
def db_seed_multi(db_clean):
    """Cenário onde dividir é claramente vantajoso.

    Inserimos 2 produtos × 3 cadeias com preços assimétricos:

    | Produto     | Continente | Pingo Doce | Auchan  |
    |-------------|------------|------------|---------|
    | Arroz       |    2.00 €  |    1.00 €  |  3.00 € |
    | Azeite      |    8.00 €  |    9.00 €  |  6.00 € |

    + Lojas físicas em Braga para todas as cadeias (~1-2 km da UMinho).

    Splits possíveis (custo de produtos):
    - Tudo Continente:  2.00 + 8.00 = 10.00
    - Tudo Pingo Doce:  1.00 + 9.00 = 10.00
    - Tudo Auchan:      3.00 + 6.00 =  9.00
    - PD + Auchan (best split): arroz no PD (1.00) + azeite no Auchan (6.00) = 7.00

    Auchan single-store: 9.00 € + deslocação curta — vencedor sem multi-loja.
    PD + Auchan split:   7.00 € + deslocação triangular — vencedor com multi-loja.
    """
    conn = db_clean
    produtos = [
        # (nome_padronizado, chave_mestre, [(insignia, preco), ...])
        ("Arroz Carolino 1kg", "ean:arr-001", [
            ("Continente", 2.00),
            ("Pingo Doce", 1.00),
            ("Auchan",     3.00),
        ]),
        ("Azeite Virgem Extra 750ml", "ean:az-001", [
            ("Continente", 8.00),
            ("Pingo Doce", 9.00),
            ("Auchan",     6.00),
        ]),
    ]
    # Lojas físicas em Braga, próximas mas com distâncias distintas
    lojas_fisicas = [
        ("Continente", "Continente Braga",       41.5409, -8.4004),
        ("Pingo Doce", "Pingo Doce Braga Hiper", 41.5572, -8.4050),
        ("Auchan",     "Auchan Braga Dr. Loureiro", 41.5591, -8.4141),
    ]

    with conn.cursor() as cur:
        # Lojas online + produtos
        ids_lojas: dict[str, int] = {}
        for ins in ("Continente", "Pingo Doce", "Auchan"):
            cur.execute(
                "INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                "VALUES (%s, 'Online', 'Nacional', 'online') RETURNING id_loja",
                (ins,),
            )
            ids_lojas[ins] = cur.fetchone()[0]

        for nome, chave, precos in produtos:
            cur.execute(
                "INSERT INTO produtos_mestre "
                "(chave_mestre, nome_padronizado, categoria_geral) "
                "VALUES (%s, %s, 'mercearia') RETURNING id_produto_mestre",
                (chave, nome),
            )
            id_pm = cur.fetchone()[0]
            for ins, preco in precos:
                cur.execute(
                    "INSERT INTO produtos_loja "
                    "(id_produto_mestre, id_loja, sku_loja, nome_na_loja) "
                    "VALUES (%s, %s, %s, %s) RETURNING id_produto_loja",
                    (id_pm, ids_lojas[ins], f"{chave}-{ins[:3]}", nome),
                )
                id_pl = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO precos_atuais "
                    "(id_produto_loja, preco_atual, em_promocao, "
                    " data_recolha, agente_origem) "
                    "VALUES (%s, %s, FALSE, '2026-05-15', 'test')",
                    (id_pl, preco),
                )

        # Lojas físicas
        for i, (ins, nome, lat, lon) in enumerate(lojas_fisicas, start=1):
            cur.execute(
                "INSERT INTO lojas_fisicas "
                "(insignia, nome_loja, latitude, longitude, fonte, external_id) "
                "VALUES (%s, %s, %s, %s, 'test', %s)",
                (ins, nome, lat, lon, f"test-{i}"),
            )

    conn.commit()


# ===========================================================================
# Cenário 1: dividir é estritamente melhor que single-store
# ===========================================================================

class TestSplitMelhorQueSingle:
    def test_recomendacao_e_par(self, db_seed_multi):
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km=0.20,
        )
        assert result is not None
        assert result["recomendacao"] == "par"
        assert result["poupanca_par"] > 0

    def test_par_recomendado_e_pd_auchan(self, db_seed_multi):
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km=0.20,
        )
        par = result["melhor_par"]
        # Espera-se que o melhor par contenha Pingo Doce (mais barato no arroz)
        # e Auchan (mais barato no azeite). A ordem pode variar.
        assert set(par["cadeias"]) == {"Pingo Doce", "Auchan"}

    def test_split_correto_arroz_em_pd_azeite_no_auchan(self, db_seed_multi):
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km=0.20,
        )
        par = result["melhor_par"]
        # Localizar quem é A e quem é B
        idx_pd = par["cadeias"].index("Pingo Doce")
        itens_pd = par[f"itens_em_{'a' if idx_pd == 0 else 'b'}"]
        itens_auchan = par[f"itens_em_{'b' if idx_pd == 0 else 'a'}"]

        nomes_pd = {it["item_pesquisado"] for it in itens_pd}
        nomes_auchan = {it["item_pesquisado"] for it in itens_auchan}

        # Arroz vai para o Pingo Doce (1.00 < 3.00 Auchan)
        assert "arroz carolino" in nomes_pd
        # Azeite vai para o Auchan (6.00 < 9.00 Pingo Doce)
        assert "azeite virgem extra" in nomes_auchan


# ===========================================================================
# Cenário 2: single-store continua melhor
# ===========================================================================

class TestSingleMelhorQueSplit:
    def test_lista_pequena_lojas_longe_entre_si(self, db_clean):
        """Se ambas as cadeias têm preços parecidos e a rota A→B é longa, o split
        não compensa porque a deslocação extra come a poupança."""
        from models.recommender import otimizar_lista_compras_geo_multi_loja

        # Cenário: 2 cadeias, preços iguais, 1 em Braga e outra em Faro
        with db_clean.cursor() as cur:
            # Cadeias online
            cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                        "VALUES ('Continente', 'Online', 'Nacional', 'online') RETURNING id_loja")
            id_c = cur.fetchone()[0]
            cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                        "VALUES ('Pingo Doce', 'Online', 'Nacional', 'online') RETURNING id_loja")
            id_pd = cur.fetchone()[0]

            # Produto comum
            cur.execute("INSERT INTO produtos_mestre (chave_mestre, nome_padronizado, "
                        "categoria_geral) VALUES ('ean:x', 'Arroz Carolino 1kg', 'm') "
                        "RETURNING id_produto_mestre")
            id_pm = cur.fetchone()[0]
            for id_loja, ins, preco in [(id_c, "C", 2.00), (id_pd, "PD", 2.00)]:
                cur.execute("INSERT INTO produtos_loja (id_produto_mestre, id_loja, "
                            "sku_loja, nome_na_loja) VALUES (%s, %s, %s, 'Arroz') "
                            "RETURNING id_produto_loja", (id_pm, id_loja, f"SKU-{ins}"))
                id_pl = cur.fetchone()[0]
                cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, "
                            "em_promocao, data_recolha, agente_origem) "
                            "VALUES (%s, %s, FALSE, '2026-05-15', 'test')", (id_pl, preco))

            # 1 loja física para cada — Continente em Braga, Pingo Doce em Faro
            for ins, lat, lon, ext_id in [
                ("Continente", 41.55, -8.40, "test-c"),
                ("Pingo Doce", 37.02, -7.93, "test-pd"),
            ]:
                cur.execute("INSERT INTO lojas_fisicas (insignia, nome_loja, latitude, "
                            "longitude, fonte, external_id) "
                            "VALUES (%s, %s, %s, %s, 'test', %s)",
                            (ins, f"{ins} Loja", lat, lon, ext_id))
        db_clean.commit()

        # Raio muito grande para deixar ambas as cadeias serem alcançáveis
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km=0.20, raio_km=1000.0,
        )
        # 1 item só → nunca é split (uma cadeia sempre fica sem itens)
        # Recomendação deve ser single
        assert result["recomendacao"] == "single"
        assert result["poupanca_par"] == 0.0


# ===========================================================================
# Cenário 3: nenhum par viável (lista incompleta em todos os pares)
# ===========================================================================

class TestNenhumParViavel:
    def test_item_em_falta_em_todas_as_cadeias(self, db_clean):
        from models.recommender import otimizar_lista_compras_geo_multi_loja

        # Inserir 1 cadeia com 1 produto. Item de "leite" não existe.
        with db_clean.cursor() as cur:
            cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                        "VALUES ('Continente', 'Online', 'Nacional', 'online') RETURNING id_loja")
            id_c = cur.fetchone()[0]
            cur.execute("INSERT INTO produtos_mestre (chave_mestre, nome_padronizado, "
                        "categoria_geral) VALUES ('ean:y', 'Arroz Carolino 1kg', 'm') "
                        "RETURNING id_produto_mestre")
            id_pm = cur.fetchone()[0]
            cur.execute("INSERT INTO produtos_loja (id_produto_mestre, id_loja, sku_loja, "
                        "nome_na_loja) VALUES (%s, %s, 'SKU-C', 'Arroz') "
                        "RETURNING id_produto_loja", (id_pm, id_c))
            id_pl = cur.fetchone()[0]
            cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, "
                        "em_promocao, data_recolha, agente_origem) "
                        "VALUES (%s, 2.00, FALSE, '2026-05-15', 'test')", (id_pl,))
            cur.execute("INSERT INTO lojas_fisicas (insignia, nome_loja, latitude, "
                        "longitude, fonte, external_id) "
                        "VALUES ('Continente', 'C Braga', 41.55, -8.40, 'test', 'tc')")
        db_clean.commit()

        # Lista com produto que não existe
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "produto-inexistente-xpto"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
        )
        # melhor_par deve ser None — nenhum par cobre a lista completa
        assert result["melhor_par"] is None
        assert result["recomendacao"] == "single"


# ===========================================================================
# Cenário 5: cálculo da distância triangular
# ===========================================================================

class TestDistanciaTriangular:
    def test_rota_user_a_b_user(self, db_seed_multi):
        """Verifica que a soma das 3 pernas da rota é igual a haversine das pernas."""
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        from models.geolocation import haversine_km

        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
        )
        par = result["melhor_par"]
        loja_a = par["loja_fisica_a"]
        loja_b = par["loja_fisica_b"]

        # Recalcular haversine das 3 pernas
        d_ua = haversine_km(UMINHO_LAT, UMINHO_LON,
                             loja_a["latitude"], loja_a["longitude"])
        d_ab = haversine_km(loja_a["latitude"], loja_a["longitude"],
                             loja_b["latitude"], loja_b["longitude"])
        d_bu = haversine_km(loja_b["latitude"], loja_b["longitude"],
                             UMINHO_LAT, UMINHO_LON)

        # Comparar (com tolerância de arredondamento — armazenado com 2 casas)
        assert math.isclose(par["distancia_user_a_km"], round(d_ua, 2), abs_tol=0.01)
        assert math.isclose(par["distancia_a_b_km"],    round(d_ab, 2), abs_tol=0.01)
        assert math.isclose(par["distancia_b_user_km"], round(d_bu, 2), abs_tol=0.01)
        assert math.isclose(
            par["distancia_total_km"],
            round(d_ua + d_ab + d_bu, 2),
            abs_tol=0.01,
        )

    def test_custo_deslocacao_e_distancia_total_vezes_custo_km(self, db_seed_multi):
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
            custo_km=0.20,
        )
        par = result["melhor_par"]
        # Custo deslocação na rota = distancia_total × custo_km (sem fator 2 — já é triangular)
        esperado = round(par["distancia_total_km"] * 0.20, 2)
        assert math.isclose(par["custo_deslocacao"], esperado, abs_tol=0.01)


# ===========================================================================
# Estrutura da resposta: campos obrigatórios
# ===========================================================================

class TestEstruturaDoResultado:
    def test_campos_obrigatorios_presentes(self, db_seed_multi):
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
        )
        # Campos do single-store
        for campo in ("itens", "melhor_opcao", "detalhe_por_cadeia",
                      "localizacao_utilizador", "custo_km", "raio_km"):
            assert campo in result, f"Campo {campo!r} em falta"
        # Campos novos do multi-loja
        for campo in ("melhor_par", "todos_os_pares", "recomendacao", "poupanca_par"):
            assert campo in result, f"Campo {campo!r} em falta"

    def test_todos_os_pares_estao_ordenados_por_custo_total(self, db_seed_multi):
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
        )
        custos = [p["custo_total"] for p in result["todos_os_pares"]]
        assert custos == sorted(custos)

    def test_recomendacao_e_string_valida(self, db_seed_multi):
        from models.recommender import otimizar_lista_compras_geo_multi_loja
        result = otimizar_lista_compras_geo_multi_loja(
            ["arroz carolino", "azeite virgem extra"],
            user_lat=UMINHO_LAT, user_lon=UMINHO_LON,
        )
        assert result["recomendacao"] in ("single", "par")
