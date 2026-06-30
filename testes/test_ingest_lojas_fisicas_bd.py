"""
Testes de integração com BD para scripts/ingest_lojas_fisicas.py.

Cobre:
  - upsert_loja_fisica: INSERT inicial + UPDATE em conflito por (insignia, external_id)
  - ingestao(): orquestração completa com BD
  - Idempotência: re-correr o mesmo input não duplica registos
  - Trigger de data_atualizacao em UPDATE

Requer a BD ``products_db_test`` configurada.
"""

from __future__ import annotations


def _loja_exemplo(**overrides) -> dict:
    """Constrói uma loja de exemplo no schema do scraper, com overrides opcionais."""
    base = {
        "insignia":      "Continente",
        "nome_loja":     "Continente Braga Teste",
        "morada":        "Rua de Teste, 1",
        "codigo_postal": "4700-001",
        "cidade":        "Braga",
        "distrito":      None,
        "latitude":      41.5409,
        "longitude":     -8.4004,
        "telefone":      "253123456",
        "horario":       "Seg-Dom 9h-22h",
        "fonte":         "scraper:continente",
        "external_id":   "test-001",
    }
    base.update(overrides)
    return base


# ===========================================================================
# upsert_loja_fisica — INSERT inicial
# ===========================================================================

class TestUpsertInicial:
    def test_insere_loja_nova(self, db_clean):
        from scripts.ingest_lojas_fisicas import upsert_loja_fisica
        with db_clean.cursor() as cur:
            upsert_loja_fisica(cur, _loja_exemplo())
        db_clean.commit()

        with db_clean.cursor() as cur:
            cur.execute("SELECT count(*), insignia, nome_loja, latitude::float, longitude::float "
                        "FROM lojas_fisicas GROUP BY insignia, nome_loja, latitude, longitude")
            row = cur.fetchone()
        assert row[0] == 1
        assert row[1] == "Continente"
        assert row[2] == "Continente Braga Teste"
        assert row[3] == 41.5409
        assert row[4] == -8.4004

    def test_loja_inserida_fica_ativa_por_default(self, db_clean):
        from scripts.ingest_lojas_fisicas import upsert_loja_fisica
        with db_clean.cursor() as cur:
            upsert_loja_fisica(cur, _loja_exemplo())
            cur.execute("SELECT ativa FROM lojas_fisicas")
            assert cur.fetchone()[0] is True
        db_clean.commit()


# ===========================================================================
# upsert_loja_fisica — UPDATE em conflito
# ===========================================================================

class TestUpsertConflito:
    def test_actualiza_loja_em_conflito(self, db_clean):
        """Re-correr com a mesma (insignia, external_id) faz UPDATE."""
        from scripts.ingest_lojas_fisicas import upsert_loja_fisica
        with db_clean.cursor() as cur:
            upsert_loja_fisica(cur, _loja_exemplo(latitude=41.0))
            upsert_loja_fisica(cur, _loja_exemplo(latitude=42.0, nome_loja="Renomeada"))
        db_clean.commit()

        with db_clean.cursor() as cur:
            cur.execute("SELECT count(*), nome_loja, latitude::float FROM lojas_fisicas "
                        "GROUP BY nome_loja, latitude")
            row = cur.fetchone()
        # Continua a haver 1 registo, com os novos valores
        assert row[0] == 1
        assert row[1] == "Renomeada"
        assert row[2] == 42.0

    def test_actualiza_so_loja_da_mesma_insignia_e_external_id(self, db_clean):
        """Insignias diferentes com mesmo external_id NÃO conflitam."""
        from scripts.ingest_lojas_fisicas import upsert_loja_fisica
        with db_clean.cursor() as cur:
            upsert_loja_fisica(cur, _loja_exemplo(insignia="Continente", external_id="X1"))
            upsert_loja_fisica(cur, _loja_exemplo(insignia="Pingo Doce", external_id="X1"))
        db_clean.commit()

        with db_clean.cursor() as cur:
            cur.execute("SELECT count(*) FROM lojas_fisicas")
            assert cur.fetchone()[0] == 2

    def test_em_update_campos_opcionais_nao_apagam_dados_existentes(self, db_clean):
        """Se vier NULL em telefone/horario num update, mantém-se o valor anterior
        (COALESCE no UPDATE preserva os dados)."""
        from scripts.ingest_lojas_fisicas import upsert_loja_fisica

        with db_clean.cursor() as cur:
            # 1º insert: tem telefone
            upsert_loja_fisica(cur, _loja_exemplo(telefone="111", horario="Aberto"))
            # 2º insert (mesmo external_id): sem telefone/horario
            loja_sem_extra = _loja_exemplo()
            loja_sem_extra["telefone"] = None
            loja_sem_extra["horario"] = None
            upsert_loja_fisica(cur, loja_sem_extra)

            cur.execute("SELECT telefone, horario FROM lojas_fisicas")
            row = cur.fetchone()
        db_clean.commit()
        # O COALESCE preservou os dados originais
        assert row[0] == "111"
        assert row[1] == "Aberto"

    def test_em_update_lat_lon_sao_sempre_actualizados(self, db_clean):
        """lat/lon são SEMPRE actualizados — não usam COALESCE."""
        from scripts.ingest_lojas_fisicas import upsert_loja_fisica
        with db_clean.cursor() as cur:
            upsert_loja_fisica(cur, _loja_exemplo(latitude=41.0, longitude=-8.0))
            upsert_loja_fisica(cur, _loja_exemplo(latitude=41.5, longitude=-8.5))
            cur.execute("SELECT latitude::float, longitude::float FROM lojas_fisicas")
            row = cur.fetchone()
        db_clean.commit()
        assert row[0] == 41.5
        assert row[1] == -8.5


# ===========================================================================
# ingestao() — orquestração completa
# ===========================================================================

class TestIngestao:
    def test_ingere_lista_de_lojas(self, db_clean):
        from scripts.ingest_lojas_fisicas import ingestao
        lojas = [
            _loja_exemplo(external_id="a", nome_loja="A"),
            _loja_exemplo(external_id="b", nome_loja="B"),
            _loja_exemplo(external_id="c", nome_loja="C"),
        ]
        total_ok, falhas = ingestao(lojas)
        assert total_ok == 3
        assert falhas == []

        with db_clean.cursor() as cur:
            cur.execute("SELECT count(*) FROM lojas_fisicas")
            assert cur.fetchone()[0] == 3

    def test_idempotencia_re_correr_nao_duplica(self, db_clean):
        from scripts.ingest_lojas_fisicas import ingestao
        lojas = [
            _loja_exemplo(external_id="a"),
            _loja_exemplo(external_id="b"),
        ]
        # 1ª passagem
        total_ok_1, _ = ingestao(lojas)
        # 2ª passagem
        total_ok_2, _ = ingestao(lojas)

        assert total_ok_1 == 2
        assert total_ok_2 == 2   # ambas passam (UPDATE não cria nova linha)

        with db_clean.cursor() as cur:
            cur.execute("SELECT count(*) FROM lojas_fisicas")
            assert cur.fetchone()[0] == 2

    def test_loja_invalida_falha_mas_nao_para_pipeline(self, db_clean):
        """Loja sem campos obrigatórios falha, mas restantes são ingeridas."""
        from scripts.ingest_lojas_fisicas import ingestao
        invalid = _loja_exemplo()
        invalid["latitude"] = None    # campo obrigatório
        valid = _loja_exemplo(external_id="ok")

        total_ok, falhas = ingestao([invalid, valid])
        assert total_ok == 1
        assert len(falhas) == 1
        assert falhas[0]["nome_loja"] == valid["nome_loja"] or "latitude" in falhas[0]["erro"]

        with db_clean.cursor() as cur:
            cur.execute("SELECT count(*) FROM lojas_fisicas")
            assert cur.fetchone()[0] == 1

    def test_dry_run_nao_escreve_na_bd(self, db_clean):
        from scripts.ingest_lojas_fisicas import ingestao
        ingestao([_loja_exemplo()], dry_run=True)
        with db_clean.cursor() as cur:
            cur.execute("SELECT count(*) FROM lojas_fisicas")
            assert cur.fetchone()[0] == 0


# ===========================================================================
# Trigger de data_atualizacao
# ===========================================================================

class TestTriggerDataAtualizacao:
    def test_data_atualizacao_e_diferente_de_data_criacao_apos_update(self, db_clean):
        """O trigger trg_lojas_fisicas_data_atualizacao deve actualizar o campo."""
        import time
        from scripts.ingest_lojas_fisicas import upsert_loja_fisica

        with db_clean.cursor() as cur:
            upsert_loja_fisica(cur, _loja_exemplo())
            cur.execute("SELECT data_criacao, data_atualizacao FROM lojas_fisicas")
            criacao_1, atualizacao_1 = cur.fetchone()
        db_clean.commit()

        # Aguarda 1.1s para garantir que o timestamp avança (resolução em segundos)
        time.sleep(1.1)

        with db_clean.cursor() as cur:
            upsert_loja_fisica(cur, _loja_exemplo(nome_loja="Alterada"))
            cur.execute("SELECT data_criacao, data_atualizacao FROM lojas_fisicas")
            criacao_2, atualizacao_2 = cur.fetchone()
        db_clean.commit()

        # data_criacao preserva-se, data_atualizacao avança
        assert criacao_1 == criacao_2
        assert atualizacao_2 > atualizacao_1
