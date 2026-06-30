"""
Diagnóstico rápido dos dados na base de dados.

Imprime no terminal um conjunto de contagens e amostras que permitem avaliar
rapidamente o estado da BD após uma ingestão: quantos produtos, dias de histórico,
produtos sem quantidade, distribuição por loja, etc.

Uso::

    python scripts/data_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG


#: Contagens simples executadas no início do diagnóstico.
_COUNT_QUERIES: dict[str, str] = {
    "lojas":                       "SELECT COUNT(*) FROM lojas",
    "produtos_mestre":             "SELECT COUNT(*) FROM produtos_mestre",
    "produtos_loja":               "SELECT COUNT(*) FROM produtos_loja",
    "precos_atuais":               "SELECT COUNT(*) FROM precos_atuais",
    "historico_precos":            "SELECT COUNT(*) FROM historico_precos",
    "produtos_sem_quantidade":     """
        SELECT COUNT(*) FROM produtos_mestre
        WHERE quantidade_valor IS NULL OR quantidade_unidade IS NULL
    """,
    "produtos_em_promocao":        "SELECT COUNT(*) FROM precos_atuais WHERE em_promocao = TRUE",
    "promocoes_incoerentes":       """
        SELECT COUNT(*) FROM precos_atuais
        WHERE em_promocao = TRUE
          AND (preco_original IS NULL OR preco_original <= preco_atual)
    """,
    "produtos_sem_preco_unitario": "SELECT COUNT(*) FROM precos_atuais WHERE preco_unitario_valor IS NULL",
    "dias_distintos_historico":    "SELECT COUNT(DISTINCT data_recolha::date) FROM historico_precos",
    "primeiro_dia_historico":      "SELECT MIN(data_recolha::date) FROM historico_precos",
    "ultimo_dia_historico":        "SELECT MAX(data_recolha::date) FROM historico_precos",
}


def _header(title: str) -> None:
    """Imprime um cabeçalho de secção no terminal."""
    print(f"\n=== {title} ===")


def main() -> None:
    """Executa o diagnóstico completo e imprime os resultados."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:

            _header("CONTAGENS GERAIS")
            for nome, query in _COUNT_QUERIES.items():
                cur.execute(query)
                print(f"  {nome}: {cur.fetchone()[0]}")

            _header("PRODUTOS POR LOJA")
            cur.execute("""
                SELECT l.insignia, COUNT(*) AS total
                FROM produtos_loja pl
                JOIN lojas l ON l.id_loja = pl.id_loja
                GROUP BY l.insignia
                ORDER BY total DESC
            """)
            for insignia, total in cur.fetchall():
                print(f"  {insignia}: {total}")

            _header("OBSERVACOES POR DIA")
            cur.execute("""
                SELECT data_recolha::date AS dia, COUNT(*) AS total
                FROM historico_precos
                GROUP BY dia
                ORDER BY dia
            """)
            for dia, total in cur.fetchall():
                print(f"  {dia}: {total}")

            _header("PRODUTOS COM MAIS HISTORICO (top 20)")
            cur.execute("""
                SELECT
                    pl.id_produto_loja,
                    pm.nome_padronizado,
                    l.insignia,
                    COUNT(*) AS total_observacoes,
                    COUNT(DISTINCT hp.data_recolha::date) AS dias_distintos
                FROM historico_precos hp
                JOIN produtos_loja  pl  ON pl.id_produto_loja   = hp.id_produto_loja
                JOIN produtos_mestre pm ON pm.id_produto_mestre  = pl.id_produto_mestre
                JOIN lojas l            ON l.id_loja             = pl.id_loja
                GROUP BY pl.id_produto_loja, pm.nome_padronizado, l.insignia
                ORDER BY dias_distintos DESC, total_observacoes DESC
                LIMIT 20
            """)
            for row in cur.fetchall():
                print(f"  {row}")

            _header("AMOSTRA — PRODUTOS SEM QUANTIDADE (top 20)")
            cur.execute("""
                SELECT nome_padronizado
                FROM produtos_mestre
                WHERE quantidade_valor IS NULL OR quantidade_unidade IS NULL
                ORDER BY nome_padronizado
                LIMIT 20
            """)
            for (nome,) in cur.fetchall():
                print(f"  {nome}")

            _header("AMOSTRA — PRECOS ATUAIS (top 20)")
            cur.execute("""
                SELECT
                    l.insignia,
                    pm.nome_padronizado,
                    pm.quantidade_valor,
                    pm.quantidade_unidade,
                    pa.preco_atual,
                    pa.preco_original,
                    pa.em_promocao,
                    pa.data_recolha::date
                FROM precos_atuais pa
                JOIN produtos_loja  pl  ON pl.id_produto_loja   = pa.id_produto_loja
                JOIN produtos_mestre pm ON pm.id_produto_mestre  = pl.id_produto_mestre
                JOIN lojas l            ON l.id_loja             = pl.id_loja
                ORDER BY l.insignia, pm.nome_padronizado
                LIMIT 20
            """)
            for row in cur.fetchall():
                print(f"  {row}")


if __name__ == "__main__":
    main()
