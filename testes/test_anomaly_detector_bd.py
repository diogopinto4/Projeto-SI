"""
Testes de integração com BD para scripts/anomaly_detector.py.

Cobre:
  - carregar_historico (query SQL ao histórico)
  - detetar_anomalias é puro sobre DataFrame e já é testado em unit tests
    (test_recommender_utils ou similar), mas re-validamos aqui o fluxo completo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _insere_serie(cur, id_produto_loja, precos_por_dia: list[tuple]):
    """Insere observações ``(preco, dias_atras)`` no histórico."""
    agora = datetime.now(timezone.utc)
    for preco, dias in precos_por_dia:
        ts = agora - timedelta(days=dias)
        cur.execute("""
            INSERT INTO historico_precos
                (id_produto_loja, preco_atual, em_promocao,
                 data_recolha, agente_origem)
            VALUES (%s, %s, FALSE, %s, 'test')
        """, (id_produto_loja, preco, ts))


# ===========================================================================
# carregar_historico
# ===========================================================================

class TestCarregarHistorico:
    def test_devolve_vazio_quando_bd_vazia(self, db_clean):
        from scripts.anomaly_detector import carregar_historico
        df = carregar_historico()
        assert df.empty

    def test_devolve_observacoes_com_joins(self, db_clean, db_seed_basico):
        from scripts.anomaly_detector import carregar_historico
        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _insere_serie(cur, id_pl, [(1.00, 5), (1.10, 3), (1.20, 1)])
        db_clean.commit()

        df = carregar_historico()
        assert len(df) == 3
        # Verificar joins
        assert (df["nome_padronizado"] == "Arroz Agulha Cigala 1000g").all()
        assert (df["cadeia"] == "Continente").all()
        # Ordenado por id_produto_loja, data_recolha
        assert list(df["preco_atual"]) == [1.00, 1.10, 1.20]


# ===========================================================================
# detetar_anomalias — fluxo end-to-end (carregar + analisar)
# ===========================================================================

class TestDetetarAnomaliasEndToEnd:
    def test_serie_estavel_sem_anomalias(self, db_clean, db_seed_basico):
        """Preços muito próximos uns dos outros → IQR pequeno → sem anomalias."""
        from scripts.anomaly_detector import carregar_historico, detetar_anomalias

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            _insere_serie(cur, id_pl, [(1.00, 10), (1.02, 8), (1.01, 6),
                                       (1.03, 4), (1.00, 2), (1.02, 1)])
        db_clean.commit()

        df = carregar_historico()
        anomalias = detetar_anomalias(df, desvios_iqr=3.0, variacao_max_pct=0.5)
        assert anomalias.empty

    def test_outlier_iqr_detectado(self, db_clean, db_seed_basico):
        """Um valor muito longe dos outros → fora de [Q1 - k·IQR, Q3 + k·IQR]."""
        from scripts.anomaly_detector import carregar_historico, detetar_anomalias

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            # Série estável 1.00€ com 1 outlier de 10€
            _insere_serie(cur, id_pl, [(1.00, 10), (1.01, 9), (1.00, 8),
                                       (1.01, 7), (10.00, 6),  # outlier
                                       (1.00, 5), (1.01, 4), (1.00, 3),
                                       (1.01, 2), (1.00, 1)])
        db_clean.commit()

        df = carregar_historico()
        anomalias = detetar_anomalias(df, desvios_iqr=3.0)
        # O 10.00€ deve ser detectado pelo IQR
        assert not anomalias.empty
        criterios = set(anomalias["criterio"])
        assert "iqr" in criterios

    def test_variacao_abrupta_detectada(self, db_clean, db_seed_basico):
        """Salto brusco (>50%) entre observações próximas no tempo."""
        from scripts.anomaly_detector import carregar_historico, detetar_anomalias

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            # 1.00 → 5.00 num dia (+400%) — variação abrupta
            _insere_serie(cur, id_pl, [(1.00, 2), (5.00, 1)])
        db_clean.commit()

        df = carregar_historico()
        anomalias = detetar_anomalias(df, variacao_max_pct=0.5, max_gap_dias=3)
        assert not anomalias.empty
        criterios = set(anomalias["criterio"])
        assert "variacao_abrupta" in criterios

    def test_gap_grande_nao_aplica_variacao_abrupta(self, db_clean, db_seed_basico):
        """Se o gap entre observações for > max_gap_dias, não conta como variação abrupta."""
        from scripts.anomaly_detector import carregar_historico, detetar_anomalias

        id_pl = db_seed_basico["id_produto_loja"]
        with db_clean.cursor() as cur:
            # 1.00 hoje, 5.00 há 60 dias → gap=60 > 3 → ignora variação abrupta
            _insere_serie(cur, id_pl, [(1.00, 60), (5.00, 1)])
        db_clean.commit()

        df = carregar_historico()
        anomalias = detetar_anomalias(df, variacao_max_pct=0.5, max_gap_dias=3)
        # Pode haver IQR (depende de quantas observações), mas não variação abrupta
        if not anomalias.empty:
            assert "variacao_abrupta" not in set(anomalias["criterio"])
