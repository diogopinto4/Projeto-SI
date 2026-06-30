"""
Auditoria de qualidade dos dados na base de dados.

Complementa ``data_diagnostics.py`` com verificações orientadas à qualidade:
nomes padronizados duplicados, produtos-mestre mapeados para múltiplas lojas,
distribuição do histórico por produto, e produtos com maior variação de preço.

Uso::

    python scripts/audit_data_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG


def _header(title: str) -> None:
    """Imprime um cabeçalho de secção no terminal."""
    print(f"\n=== {title} ===")


def main() -> None:
    """Executa a auditoria completa e imprime os resultados."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:

            _header("RESUMO")
            for tabela in ("produtos_mestre", "produtos_loja", "historico_precos"):
                cur.execute(f"SELECT COUNT(*) FROM {tabela}")
                print(f"  {tabela}: {cur.fetchone()[0]}")

            _header("PRODUTOS SEM QUANTIDADE")
            cur.execute("""
                SELECT COUNT(*) FROM produtos_mestre
                WHERE quantidade_valor IS NULL OR quantidade_unidade IS NULL
            """)
            print(f"  total sem quantidade: {cur.fetchone()[0]}")

            cur.execute("""
                SELECT nome_padronizado, marca, categoria_geral
                FROM produtos_mestre
                WHERE quantidade_valor IS NULL OR quantidade_unidade IS NULL
                ORDER BY nome_padronizado
                LIMIT 50
            """)
            for row in cur.fetchall():
                print(f"  {row}")

            _header("NOMES PADRONIZADOS REPETIDOS (top 50)")
            cur.execute("""
                SELECT nome_padronizado, COUNT(*) AS total
                FROM produtos_mestre
                GROUP BY nome_padronizado
                HAVING COUNT(*) > 1
                ORDER BY total DESC, nome_padronizado
                LIMIT 50
            """)
            for row in cur.fetchall():
                print(f"  {row}")

            _header("PRODUTOS MESTRE COM MAIS DE UMA LOJA (top 50)")
            cur.execute("""
                SELECT
                    pm.nome_padronizado,
                    pm.marca,
                    COUNT(*) AS total_lojas
                FROM produtos_loja pl
                JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
                GROUP BY pm.id_produto_mestre, pm.nome_padronizado, pm.marca
                HAVING COUNT(*) > 1
                ORDER BY total_lojas DESC, pm.nome_padronizado
                LIMIT 50
            """)
            for row in cur.fetchall():
                print(f"  {row}")

            _header("DISTRIBUICAO DE HISTORICO POR PRODUTO")
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE total_dias = 1) AS produtos_com_1_dia,
                    COUNT(*) FILTER (WHERE total_dias = 2) AS produtos_com_2_dias,
                    COUNT(*) FILTER (WHERE total_dias >= 3) AS produtos_com_3_ou_mais_dias
                FROM (
                    SELECT id_produto_loja, COUNT(DISTINCT data_recolha::date) AS total_dias
                    FROM historico_precos
                    GROUP BY id_produto_loja
                ) t
            """)
            row = cur.fetchone()
            print(f"  1 dia: {row[0]}  |  2 dias: {row[1]}  |  3+ dias: {row[2]}")

            _header("PRODUTOS COM MAIOR VARIACAO DE PRECO (top 30)")
            cur.execute("""
                SELECT
                    pm.nome_padronizado,
                    l.insignia,
                    MIN(hp.preco_atual)                            AS preco_min,
                    MAX(hp.preco_atual)                            AS preco_max,
                    COUNT(DISTINCT hp.data_recolha::date)          AS dias
                FROM historico_precos hp
                JOIN produtos_loja  pl  ON pl.id_produto_loja   = hp.id_produto_loja
                JOIN produtos_mestre pm ON pm.id_produto_mestre  = pl.id_produto_mestre
                JOIN lojas l            ON l.id_loja             = pl.id_loja
                GROUP BY pm.nome_padronizado, l.insignia, hp.id_produto_loja
                HAVING COUNT(DISTINCT hp.data_recolha::date) >= 2
                ORDER BY (MAX(hp.preco_atual) - MIN(hp.preco_atual)) DESC, dias DESC
                LIMIT 30
            """)
            for row in cur.fetchall():
                print(f"  {row}")


if __name__ == "__main__":
    main()
