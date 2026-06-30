"""
Testes de integração com BD para scripts/price_monitor.py.

Cobre:
  - detetar_mudancas_preco: window functions SQL para comparar preço antigo vs recente
  - detetar_mudancas_produto: histórico cronológico de um produto

A função ``gerar_alertas`` (pura sobre DataFrame) e ``classificar_alerta`` já são
cobertas em testes de unidade — aqui validamos o caminho SQL.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _inserir_historico(cur, id_produto_loja: int, observacoes: list[tuple]):
    """Insere observações ``(preco, em_promocao, horas_atras)`` no histórico.

    ``horas_atras`` é o número de horas a subtrair ao timestamp actual.
    """
    agora = datetime.now(timezone.utc)
    for preco, em_promo, horas_atras in observacoes:
        ts = agora - timedelta(hours=horas_atras)
        cur.execute("""
            INSERT INTO historico_precos
                (id_produto_loja, preco_atual, em_promocao,
                 data_recolha, agente_origem)
            VALUES (%s, %s, %s, %s, 'test')
        """, (id_produto_loja, preco, em_promo, ts))


# ===========================================================================
# detetar_mudancas_preco
# ===========================================================================

class TestDetetarMudancasPreco:
    def test_devolve_vazio_quando_sem_mudancas(self, db_seed_basico):
        """Sem histórico (só preco_atual) → window não tem 2 pontos → sem mudanças."""
        from scripts.price_monitor import detetar_mudancas_preco
        df = detetar_mudancas_preco(janela_horas=24)
        assert df.empty

    def test_deteta_subida_de_preco(self, db_clean, db_seed_basico):
        from scripts.price_monitor import detetar_mudancas_preco

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _inserir_historico(cur, id_pl, [
                (1.50, False, 20),   # 20h atrás: 1.50€
                (1.80, False,  2),   # 2h atrás:  1.80€ (subiu)
            ])
        db_clean.commit()

        df = detetar_mudancas_preco(janela_horas=24)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["preco_antigo"] == 1.50
        assert row["preco_novo"]   == 1.80
        assert row["variacao_abs"] > 0
        assert row["variacao_pct"] > 0   # +20%

    def test_deteta_descida_de_preco(self, db_clean, db_seed_basico):
        from scripts.price_monitor import detetar_mudancas_preco

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _inserir_historico(cur, id_pl, [
                (2.00, False, 20),
                (1.60, False,  2),
            ])
        db_clean.commit()

        df = detetar_mudancas_preco(janela_horas=24)
        assert len(df) == 1
        assert df.iloc[0]["variacao_pct"] < 0   # -20%

    def test_deteta_mudanca_de_promocao(self, db_clean, db_seed_basico):
        """Mesmo sem mudança de preço, mudança de estado de promoção é detectada."""
        from scripts.price_monitor import detetar_mudancas_preco

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _inserir_historico(cur, id_pl, [
                (1.69, False, 20),   # antes: sem promoção
                (1.69, True,   2),   # agora: em promoção (mesmo preço)
            ])
        db_clean.commit()

        df = detetar_mudancas_preco(janela_horas=24)
        # A query inclui WHERE preco_diferente OR promo_diferente → deve apanhar
        assert len(df) == 1

    def test_ignora_observacoes_fora_da_janela(self, db_clean, db_seed_basico):
        """Observações antigas (fora da janela) não entram na comparação."""
        from scripts.price_monitor import detetar_mudancas_preco

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _inserir_historico(cur, id_pl, [
                (1.00, False, 200),   # 200h atrás (fora da janela 24h)
                (1.50, False,  20),
                (1.50, False,   2),   # sem mudança nos últimos 24h
            ])
        db_clean.commit()

        df = detetar_mudancas_preco(janela_horas=24)
        # Dentro da janela só há 1.50 → 1.50, sem mudança
        assert df.empty

    def test_ordena_por_variacao_absoluta_decrescente(self, db_clean):
        """Vários produtos com mudanças — devem ser ordenados pela maior variação primeiro."""
        from scripts.price_monitor import detetar_mudancas_preco

        with db_clean.cursor() as cur:
            # Loja única para todos
            cur.execute("INSERT INTO lojas (insignia, formato_loja, localizacao, canal) "
                        "VALUES ('Continente', 'Online', 'Nacional', 'online') RETURNING id_loja")
            id_loja = cur.fetchone()[0]

            # 2 produtos, 1 com variação grande, 1 com pequena
            ids = []
            for i, (nome, var) in enumerate([("Pequena", 0.10), ("Grande", 2.00)], start=1):
                cur.execute("INSERT INTO produtos_mestre (chave_mestre, nome_padronizado, "
                            "categoria_geral) VALUES (%s, %s, 'm') RETURNING id_produto_mestre",
                            (f"ean:{i}", nome))
                pm = cur.fetchone()[0]
                cur.execute("INSERT INTO produtos_loja (id_produto_mestre, id_loja, sku_loja, "
                            "nome_na_loja) VALUES (%s, %s, %s, %s) RETURNING id_produto_loja",
                            (pm, id_loja, f"SKU{i}", nome))
                id_pl = cur.fetchone()[0]
                ids.append(id_pl)
                cur.execute("INSERT INTO precos_atuais (id_produto_loja, preco_atual, em_promocao, "
                            "data_recolha, agente_origem) VALUES (%s, 1.00, FALSE, '2026-05-15', 't')",
                            (id_pl,))
                _inserir_historico(cur, id_pl, [
                    (1.00, False, 20),
                    (1.00 + var, False, 2),
                ])
        db_clean.commit()

        df = detetar_mudancas_preco(janela_horas=24)
        assert len(df) == 2
        # Maior variação aparece primeiro
        assert df.iloc[0]["nome_padronizado"] == "Grande"
        assert df.iloc[1]["nome_padronizado"] == "Pequena"


# ===========================================================================
# detetar_mudancas_produto — histórico de 1 produto
# ===========================================================================

class TestDetetarMudancasProduto:
    def test_devolve_cronologia_do_produto(self, db_clean, db_seed_basico):
        from scripts.price_monitor import detetar_mudancas_produto

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _inserir_historico(cur, id_pl, [
                (1.00, False, 100),
                (1.20, False,  50),
                (1.50, True,    5),
            ])
        db_clean.commit()

        df = detetar_mudancas_produto(id_pl, janela_horas=168)
        assert len(df) == 3
        # Ordenado por data ASC (mais antigo primeiro)
        precos = df["preco_atual"].tolist()
        assert precos == [1.00, 1.20, 1.50]

    def test_devolve_vazio_para_produto_inexistente(self, db_clean):
        from scripts.price_monitor import detetar_mudancas_produto
        df = detetar_mudancas_produto(99999, janela_horas=24)
        assert df.empty

    def test_filtra_pela_janela(self, db_clean, db_seed_basico):
        from scripts.price_monitor import detetar_mudancas_produto

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _inserir_historico(cur, id_pl, [
                (1.00, False, 500),    # fora de qualquer janela razoável
                (2.00, False,  10),
            ])
        db_clean.commit()

        df = detetar_mudancas_produto(id_pl, janela_horas=24)
        assert len(df) == 1
        assert df.iloc[0]["preco_atual"] == 2.00
