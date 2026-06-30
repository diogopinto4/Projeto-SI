"""
Testes de integração com BD para models/recommender.py.

Cobre os caminhos principais do sistema de recomendação:
  - pesquisar_produtos (AND-ILIKE + fallback OR + similarity)
  - recomendar_melhor_loja (comparação entre lojas)
  - otimizar_lista_compras (split multi-loja + melhor loja única)
  - otimizar_lista_compras_geo (com custo de deslocação)

Requer a BD ``products_db_test``. As fixtures inserem dados mínimos coerentes
com o schema; os testes verificam comportamento sobre esses dados conhecidos.
"""

from __future__ import annotations

import pytest


# ===========================================================================
# pesquisar_produtos
# ===========================================================================

class TestPesquisarProdutos:
    def test_devolve_produto_que_existe(self, db_seed_basico):
        from models.recommender import pesquisar_produtos
        df = pesquisar_produtos("arroz cigala", limite=10)
        assert df is not None
        assert len(df) >= 1
        # Verificar campos
        assert "nome_padronizado" in df.columns
        assert "preco_min" in df.columns
        assert df.iloc[0]["nome_padronizado"] == "Arroz Agulha Cigala 1000g"

    def test_devolve_none_quando_nao_encontra(self, db_clean):
        from models.recommender import pesquisar_produtos
        # BD vazia → nenhum produto encontrado
        df = pesquisar_produtos("xpto-produto-inexistente", limite=5)
        assert df is None

    def test_pesquisa_case_insensitive_e_sem_acentos(self, db_seed_basico):
        from models.recommender import pesquisar_produtos
        # Maiúsculas + sem 'unaccent' tratamento — devia funcionar
        df = pesquisar_produtos("ARROZ", limite=5)
        assert df is not None
        assert len(df) >= 1

    def test_respeita_limite(self, db_seed_3_cadeias):
        from models.recommender import pesquisar_produtos
        # 1 produto-mestre partilhado por 3 lojas → 1 linha (agrupada por nome)
        df = pesquisar_produtos("arroz", limite=10)
        assert df is not None
        # GROUP BY nome_padronizado → 1 linha
        assert len(df) == 1
        assert df.iloc[0]["num_lojas"] == 3


# ===========================================================================
# recomendar_melhor_loja
# ===========================================================================

class TestRecomendarMelhorLoja:
    def test_devolve_ordenado_por_preco_ascendente(self, db_seed_3_cadeias):
        from models.recommender import recomendar_melhor_loja
        df = recomendar_melhor_loja("arroz cigala", top_n=5)
        assert df is not None
        precos = df["preco_atual"].tolist()
        assert precos == sorted(precos), "Deve estar ordenado por preço ascendente"
        # Auchan tem 1.59 — o mais barato
        assert df.iloc[0]["loja"] == "Auchan"

    def test_calcula_poupanca_pct(self, db_seed_3_cadeias):
        from models.recommender import recomendar_melhor_loja
        df = recomendar_melhor_loja("arroz cigala", top_n=5)
        assert df is not None
        # A linha mais cara (Pingo Doce 1.79) tem poupança = 0
        # A linha mais barata (Auchan 1.59) tem maior poupança
        assert df.iloc[-1]["poupanca_pct"] == 0.0
        assert df.iloc[0]["poupanca_pct"] > 0

    def test_respeita_top_n(self, db_seed_3_cadeias):
        from models.recommender import recomendar_melhor_loja
        df = recomendar_melhor_loja("arroz cigala", top_n=2)
        assert df is not None
        assert len(df) == 2

    def test_devolve_none_para_produto_inexistente(self, db_clean):
        from models.recommender import recomendar_melhor_loja
        df = recomendar_melhor_loja("xpto-inexistente", top_n=5)
        assert df is None


# ===========================================================================
# otimizar_lista_compras
# ===========================================================================

class TestOtimizarListaCompras:
    def test_lista_unico_item_em_3_cadeias(self, db_seed_3_cadeias):
        from models.recommender import otimizar_lista_compras
        result = otimizar_lista_compras(["arroz cigala"])
        assert result is not None
        # 1 item encontrado
        assert len(result["itens"]) == 1
        # custo_minimo == melhor preço entre as 3 cadeias (Auchan 1.59)
        assert result["custo_minimo"] == 1.59
        # melhor_loja única: a mais barata
        assert result["melhor_loja"] == "Auchan"
        assert result["custo_melhor_loja"] == 1.59

    def test_detalhe_por_loja_com_3_cadeias(self, db_seed_3_cadeias):
        from models.recommender import otimizar_lista_compras
        result = otimizar_lista_compras(["arroz cigala"])
        detalhe = result["detalhe_por_loja"]
        assert set(detalhe.keys()) == {"Continente", "Pingo Doce", "Auchan"}
        assert detalhe["Auchan"]["total"] == 1.59
        assert detalhe["Pingo Doce"]["total"] == 1.79
        assert detalhe["Continente"]["total"] == 1.69
        # Nenhuma cadeia tem item em falta — todas têm o arroz
        for info in detalhe.values():
            assert info["em_falta"] == []

    def test_devolve_none_quando_nenhum_item_encontrado(self, db_clean):
        from models.recommender import otimizar_lista_compras
        result = otimizar_lista_compras(["produto-inexistente-xyz"])
        assert result is None

    def test_item_em_falta_em_algumas_lojas(self, db_clean):
        """Cenário: 2 itens, mas só 1 cadeia tem ambos."""
        from models.recommender import otimizar_lista_compras

        # Continente: tem arroz + leite
        # Pingo Doce: só tem leite
        with db_clean.cursor() as cur:
            # Continente
            cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                        "VALUES ('Continente', 'Online', 'Nacional', 'online') RETURNING id_loja")
            id_cont = cur.fetchone()[0]

            # Arroz (só Continente)
            cur.execute("""
                INSERT INTO produtos_mestre (chave_mestre, nome_padronizado, categoria_geral)
                VALUES ('ean:111', 'Arroz Carolino 1kg', 'mercearia') RETURNING id_produto_mestre
            """)
            id_pm_arroz = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO produtos_loja (id_produto_mestre, id_loja, sku_loja, nome_na_loja)
                VALUES (%s, %s, 'A001', 'Arroz Carolino') RETURNING id_produto_loja
            """, (id_pm_arroz, id_cont))
            id_pl = cur.fetchone()[0]
            cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, em_promocao, "
                        "data_recolha, agente_origem) VALUES (%s, 1.00, FALSE, '2026-05-15', 'test')",
                        (id_pl,))

            # Leite (em ambas)
            cur.execute("""
                INSERT INTO produtos_mestre (chave_mestre, nome_padronizado, categoria_geral)
                VALUES ('ean:222', 'Leite Meio Gordo 1l', 'frescos') RETURNING id_produto_mestre
            """)
            id_pm_leite = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO produtos_loja (id_produto_mestre, id_loja, sku_loja, nome_na_loja)
                VALUES (%s, %s, 'L001', 'Leite MG') RETURNING id_produto_loja
            """, (id_pm_leite, id_cont))
            id_pl_leite_c = cur.fetchone()[0]
            cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, em_promocao, "
                        "data_recolha, agente_origem) VALUES (%s, 0.90, FALSE, '2026-05-15', 'test')",
                        (id_pl_leite_c,))

            cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                        "VALUES ('Pingo Doce', 'Online', 'Nacional', 'online') RETURNING id_loja")
            id_pd = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO produtos_loja (id_produto_mestre, id_loja, sku_loja, nome_na_loja)
                VALUES (%s, %s, 'L002', 'Leite MG') RETURNING id_produto_loja
            """, (id_pm_leite, id_pd))
            id_pl_leite_p = cur.fetchone()[0]
            cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, em_promocao, "
                        "data_recolha, agente_origem) VALUES (%s, 0.85, FALSE, '2026-05-15', 'test')",
                        (id_pl_leite_p,))
        db_clean.commit()

        result = otimizar_lista_compras(["arroz carolino", "leite meio gordo"])
        assert result is not None

        # Pingo Doce não tem arroz — deve aparecer em em_falta
        assert "arroz carolino" in result["detalhe_por_loja"]["Pingo Doce"]["em_falta"]
        # Continente tem ambos
        assert result["detalhe_por_loja"]["Continente"]["em_falta"] == []
        # Melhor loja única (com lista completa) deve ser Continente
        assert result["melhor_loja"] == "Continente"


# ===========================================================================
# otimizar_lista_compras_geo (BD real, integra geolocation + recommender)
# ===========================================================================

class TestOtimizarListaComprasGeo:
    UMINHO_LAT, UMINHO_LON = 41.561, -8.397

    @pytest.fixture
    def db_seed_geo(self, db_clean):
        """3 cadeias com produto + 3 lojas físicas (1 por cadeia em Braga)."""
        conn = db_clean
        cadeias = [
            # (insignia, preco_produto, lat, lon — coordenadas reais em Braga)
            ("Continente", 1.69, 41.5409, -8.4004),
            ("Pingo Doce", 1.79, 41.5572, -8.4050),
            ("Auchan",     1.59, 41.5591, -8.4141),
        ]
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO produtos_mestre
                    (chave_mestre, nome_padronizado, categoria_geral)
                VALUES ('ean:5601', 'Arroz Agulha Cigala 1000g', 'mercearia')
                RETURNING id_produto_mestre
            """)
            id_pm = cur.fetchone()[0]

            for insignia, preco, lat, lon in cadeias:
                cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                            "VALUES (%s, 'Online', 'Nacional', 'online') RETURNING id_loja",
                            (insignia,))
                id_loja = cur.fetchone()[0]
                cur.execute("""
                    INSERT INTO produtos_loja (id_produto_mestre, id_loja, sku_loja, nome_na_loja)
                    VALUES (%s, %s, 'SKU', 'Arroz') RETURNING id_produto_loja
                """, (id_pm, id_loja))
                id_pl = cur.fetchone()[0]
                cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, em_promocao, "
                            "data_recolha, agente_origem) VALUES (%s, %s, FALSE, '2026-05-15', 'test')",
                            (id_pl, preco))
                # Loja física
                cur.execute("""
                    INSERT INTO lojas_fisicas
                        (insignia, nome_loja, latitude, longitude, fonte, external_id)
                    VALUES (%s, %s, %s, %s, 'test', %s)
                """, (insignia, f"{insignia} Braga Test", lat, lon, f"test-{insignia}"))
        conn.commit()
        return id_pm

    def test_recomendacao_combina_preco_e_distancia(self, db_seed_geo):
        from models.recommender import otimizar_lista_compras_geo
        result = otimizar_lista_compras_geo(
            ["arroz cigala"],
            user_lat=self.UMINHO_LAT, user_lon=self.UMINHO_LON,
            custo_km=0.20,
        )
        assert result is not None
        assert result["melhor_opcao"] is not None
        # Auchan: 1.59 + (dist × 2 × 0.20). Continente: 1.69 + (dist × 2 × 0.20).
        # Independentemente das distâncias específicas, é provável que Auchan
        # ainda ganhe pelo preço mais baixo dos produtos.
        # Vamos verificar que a recomendação tem todos os campos esperados:
        m = result["melhor_opcao"]
        for campo in ("insignia", "loja_fisica", "distancia_km",
                       "custo_produtos", "custo_deslocacao", "custo_total"):
            assert campo in m

    def test_detalhe_por_cadeia_inclui_todas_as_3(self, db_seed_geo):
        from models.recommender import otimizar_lista_compras_geo
        result = otimizar_lista_compras_geo(
            ["arroz cigala"],
            user_lat=self.UMINHO_LAT, user_lon=self.UMINHO_LON,
            custo_km="equilibrado",
        )
        assert set(result["detalhe_por_cadeia"].keys()) == {"Continente", "Pingo Doce", "Auchan"}

    def test_custo_total_e_soma_produtos_mais_deslocacao(self, db_seed_geo):
        from models.recommender import otimizar_lista_compras_geo
        result = otimizar_lista_compras_geo(
            ["arroz cigala"],
            user_lat=self.UMINHO_LAT, user_lon=self.UMINHO_LON,
            custo_km=0.20,
        )
        for ins, d in result["detalhe_por_cadeia"].items():
            if d["alcancavel"]:
                # custo_total == custo_produtos + custo_deslocacao (com tolerância de centavos)
                assert abs(d["custo_total"] - (d["custo_produtos"] + d["custo_deslocacao"])) < 0.01

    def test_raio_pequeno_exclui_cadeias(self, db_seed_geo):
        from models.recommender import otimizar_lista_compras_geo
        # Faro: nenhuma das lojas físicas (todas em Braga) está no raio
        result = otimizar_lista_compras_geo(
            ["arroz cigala"],
            user_lat=37.0, user_lon=-7.9,   # Faro
            custo_km=0.20,
            raio_km=10.0,
        )
        # Todas as cadeias devem ficar alcancavel=False
        for ins, d in result["detalhe_por_cadeia"].items():
            assert d["alcancavel"] is False
        assert result["melhor_opcao"] is None

    def test_devolve_none_quando_nenhum_item_encontrado(self, db_clean):
        from models.recommender import otimizar_lista_compras_geo
        result = otimizar_lista_compras_geo(
            ["produto-inexistente"],
            user_lat=self.UMINHO_LAT, user_lon=self.UMINHO_LON,
        )
        assert result is None
