"""
Monitor de preços — Sensorização e Ambiente.

Deteta mudanças de preço e estado de promoção ocorridas dentro de uma janela
temporal configurável, classifica-as por tipo de alerta (promoção, subida,
descida) e gera relatório no terminal ou ficheiro CSV.

Concebido para correr após cada ciclo de scraping + ingestão.

Uso::

    python scripts/price_monitor.py
    python scripts/price_monitor.py --horas 48 --threshold 3.0
    python scripts/price_monitor.py --produto-id 42
    python scripts/price_monitor.py --stats
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

#: Directório onde são guardados os CSV de alertas gerados com --guardar.
ALERTAS_DIR = Path(__file__).parent.parent / "data" / "alertas"
ALERTAS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilitários de acesso à BD
# ---------------------------------------------------------------------------

def get_connection() -> psycopg2.extensions.connection:
    """Abre uma nova ligação à base de dados usando :data:`DB_CONFIG`.

    Returns:
        Ligação psycopg2 pronta a usar.

    Raises:
        psycopg2.OperationalError: Se a ligação falhar.
    """
    return psycopg2.connect(**DB_CONFIG)


def _query_to_df(cur: psycopg2.extensions.cursor, query: str, params: tuple = ()) -> pd.DataFrame:
    """Executa uma query e devolve o resultado como DataFrame.

    Args:
        cur: Cursor psycopg2 aberto.
        query: SQL a executar.
        params: Parâmetros posicionais para a query (evita SQL injection).

    Returns:
        DataFrame com colunas inferidas dos descritores do cursor.
    """
    cur.execute(query, params)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Deteção de mudanças de preço
# ---------------------------------------------------------------------------

def detetar_mudancas_preco(janela_horas: int = 24) -> pd.DataFrame:
    """Compara o preço mais recente com o mais antigo dentro da janela temporal.

    Para cada produto-loja, usa window functions SQL para encontrar o registo
    mais recente e o mais antigo dentro de ``janela_horas`` e calcula a variação
    absoluta e percentual.

    Args:
        janela_horas: Número de horas a analisar para trás a partir de agora.

    Returns:
        DataFrame com uma linha por produto-loja que registou mudança de preço
        ou de estado de promoção dentro da janela. Inclui colunas:
        ``id_produto_loja``, ``nome_padronizado``, ``marca``, ``loja``,
        ``preco_antigo``, ``preco_novo``, ``variacao_abs``, ``variacao_pct``,
        ``promo_antiga``, ``promo_nova``, ``data_antiga``, ``data_nova``.
    """
    corte = datetime.now(timezone.utc) - timedelta(hours=janela_horas)

    query = """
        WITH ranked AS (
            SELECT
                hp.id_produto_loja,
                hp.preco_atual,
                hp.em_promocao,
                hp.data_recolha,
                ROW_NUMBER() OVER (
                    PARTITION BY hp.id_produto_loja
                    ORDER BY hp.data_recolha DESC
                ) AS rn_recente,
                ROW_NUMBER() OVER (
                    PARTITION BY hp.id_produto_loja
                    ORDER BY hp.data_recolha ASC
                ) AS rn_antigo
            FROM historico_precos hp
            WHERE hp.data_recolha >= %s
        ),
        mais_recente AS (
            SELECT id_produto_loja, preco_atual AS preco_novo,
                   em_promocao AS promo_nova, data_recolha AS data_nova
            FROM ranked WHERE rn_recente = 1
        ),
        mais_antigo AS (
            SELECT id_produto_loja, preco_atual AS preco_antigo,
                   em_promocao AS promo_antiga, data_recolha AS data_antiga
            FROM ranked WHERE rn_antigo = 1
        )
        SELECT
            mr.id_produto_loja,
            pm.nome_padronizado,
            pm.marca,
            l.insignia                                              AS loja,
            ma.preco_antigo::float,
            mr.preco_novo::float,
            (mr.preco_novo - ma.preco_antigo)::float               AS variacao_abs,
            CASE WHEN ma.preco_antigo > 0
                 THEN ((mr.preco_novo - ma.preco_antigo) / ma.preco_antigo * 100)::float
                 ELSE 0
            END                                                     AS variacao_pct,
            ma.promo_antiga,
            mr.promo_nova,
            ma.data_antiga,
            mr.data_nova
        FROM mais_recente mr
        JOIN mais_antigo ma        USING (id_produto_loja)
        JOIN produtos_loja  pl     ON pl.id_produto_loja   = mr.id_produto_loja
        JOIN produtos_mestre pm    ON pm.id_produto_mestre  = pl.id_produto_mestre
        JOIN lojas           l     ON l.id_loja             = pl.id_loja
        WHERE mr.preco_novo IS DISTINCT FROM ma.preco_antigo
           OR mr.promo_nova  IS DISTINCT FROM ma.promo_antiga
        ORDER BY ABS((mr.preco_novo - ma.preco_antigo)::float) DESC
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            return _query_to_df(cur, query, (corte,))


def detetar_mudancas_produto(produto_id: int, janela_horas: int = 168) -> pd.DataFrame:
    """Devolve o histórico cronológico de preços de um produto-loja específico.

    Args:
        produto_id: ``id_produto_loja`` a consultar.
        janela_horas: Janela de análise em horas (default: 168 = 1 semana).

    Returns:
        DataFrame ordenado por ``data_recolha`` com colunas:
        ``data_recolha``, ``preco_atual``, ``preco_original``,
        ``em_promocao``, ``agente_origem``.
    """
    corte = datetime.now(timezone.utc) - timedelta(hours=janela_horas)

    query = """
        SELECT
            hp.data_recolha,
            hp.preco_atual::float,
            hp.preco_original::float,
            hp.em_promocao,
            hp.agente_origem
        FROM historico_precos hp
        WHERE hp.id_produto_loja = %s
          AND hp.data_recolha >= %s
        ORDER BY hp.data_recolha ASC
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            return _query_to_df(cur, query, (produto_id, corte))


# ---------------------------------------------------------------------------
# Geração de alertas
# ---------------------------------------------------------------------------

def classificar_alerta(variacao_pct: float, promo_antiga: bool, promo_nova: bool) -> str:
    """Classifica o tipo de alerta com base na variação e estado de promoção.

    Hierarquia de classificação:
    1. Mudança de estado de promoção (PROMOÇÃO_INICIADA / PROMOÇÃO_TERMINADA).
    2. Variação >= 5% (DESCIDA_SIGNIFICATIVA / SUBIDA_SIGNIFICATIVA).
    3. Qualquer variação (DESCIDA / SUBIDA).
    4. Sem alteração.

    Args:
        variacao_pct: Variação percentual do preço (negativa = descida).
        promo_antiga: Estado de promoção antes da mudança.
        promo_nova: Estado de promoção depois da mudança.

    Returns:
        String com o tipo de alerta (ex: ``"PROMOÇÃO_INICIADA"``).
    """
    if not promo_antiga and promo_nova:
        return "PROMOÇÃO_INICIADA"
    if promo_antiga and not promo_nova:
        return "PROMOÇÃO_TERMINADA"
    if variacao_pct <= -5.0:
        return "DESCIDA_SIGNIFICATIVA"
    if variacao_pct >= 5.0:
        return "SUBIDA_SIGNIFICATIVA"
    if variacao_pct < 0:
        return "DESCIDA"
    if variacao_pct > 0:
        return "SUBIDA"
    return "SEM_ALTERAÇÃO"


def gerar_alertas(df_mudancas: pd.DataFrame, threshold_pct: float = 2.0) -> pd.DataFrame:
    """Filtra as mudanças relevantes e classifica-as como alertas.

    Considera relevante qualquer mudança com variação absoluta >= ``threshold_pct``
    ou com alteração de estado de promoção (independentemente da variação).

    Args:
        df_mudancas: DataFrame de :func:`detetar_mudancas_preco`.
        threshold_pct: Variação percentual mínima para incluir no alerta.

    Returns:
        Subconjunto do DataFrame com a coluna ``tipo_alerta`` adicionada,
        ou DataFrame vazio se não houver mudanças relevantes.
    """
    if df_mudancas.empty:
        return pd.DataFrame()

    df = df_mudancas.copy()
    df["tipo_alerta"] = df.apply(
        lambda r: classificar_alerta(r["variacao_pct"], r["promo_antiga"], r["promo_nova"]),
        axis=1,
    )

    mascara = (
        (df["variacao_pct"].abs() >= threshold_pct)
        | df["tipo_alerta"].isin(["PROMOÇÃO_INICIADA", "PROMOÇÃO_TERMINADA"])
    )
    return df[mascara].copy()


def guardar_alertas(alertas: pd.DataFrame) -> Path | None:
    """Guarda o DataFrame de alertas em CSV com timestamp no nome.

    O ficheiro é escrito em :data:`ALERTAS_DIR` com o formato
    ``alertas_YYYYMMDD_HHMMSS.csv``.

    Args:
        alertas: DataFrame de :func:`gerar_alertas`.

    Returns:
        ``Path`` do ficheiro criado, ou ``None`` se ``alertas`` estiver vazio.
    """
    if alertas.empty:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = ALERTAS_DIR / f"alertas_{ts}.csv"
    alertas.to_csv(caminho, index=False)
    return caminho


# ---------------------------------------------------------------------------
# Relatório no terminal
# ---------------------------------------------------------------------------

def imprimir_relatorio(alertas: pd.DataFrame, janela_horas: int) -> None:
    """Imprime relatório formatado de alertas agrupado por tipo.

    Args:
        alertas: DataFrame de :func:`gerar_alertas`.
        janela_horas: Janela de análise usada (apenas para o cabeçalho).
    """
    agora = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  MONITOR DE PRECOS — {agora}")
    print(f"  Janela de analise: ultimas {janela_horas}h")
    print(f"{'='*60}")

    if alertas.empty:
        print("\n  Sem alteracoes de preco relevantes neste periodo.\n")
        return

    ordem_tipos = [
        "PROMOÇÃO_INICIADA",
        "DESCIDA_SIGNIFICATIVA",
        "DESCIDA",
        "SUBIDA",
        "SUBIDA_SIGNIFICATIVA",
        "PROMOÇÃO_TERMINADA",
    ]
    icone_map = {
        "PROMOÇÃO_INICIADA":     "[+]",
        "DESCIDA_SIGNIFICATIVA": "[v]",
        "DESCIDA":               "[v]",
        "SUBIDA":                "[^]",
        "SUBIDA_SIGNIFICATIVA":  "[^]",
        "PROMOÇÃO_TERMINADA":    "[x]",
    }

    for tipo in ordem_tipos:
        subset = alertas[alertas["tipo_alerta"] == tipo]
        if subset.empty:
            continue

        icone = icone_map.get(tipo, "[ ]")
        print(f"\n  {icone} {tipo} ({len(subset)} produto(s)):")
        print(f"  {'-'*56}")

        for _, row in subset.iterrows():
            sinal = "v" if row["variacao_abs"] < 0 else "^"
            print(
                f"  {sinal} {str(row['nome_padronizado'])[:45]:<45} "
                f"[{row['loja']}]"
            )
            print(
                f"    {row['preco_antigo']:.2f}EUR -> {row['preco_novo']:.2f}EUR  "
                f"({row['variacao_pct']:+.1f}%)"
            )

    print(f"\n  Total de alertas: {len(alertas)}")
    print(f"{'='*60}\n")


def imprimir_historico_produto(df: pd.DataFrame, produto_id: int) -> None:
    """Imprime o histórico cronológico de preços de um produto no terminal.

    Args:
        df: DataFrame de :func:`detetar_mudancas_produto`.
        produto_id: ID do produto (usado apenas no cabeçalho).
    """
    if df.empty:
        print(f"\n[MONITOR] Sem dados recentes para o produto {produto_id}.")
        return

    print(f"\n[MONITOR] Historico de precos — Produto {produto_id}:")
    print(f"  {'Data/hora':<22} {'Preco':>8} {'Original':>10} {'Promocao':>9} {'Agente':<25}")
    print(f"  {'-'*78}")

    preco_anterior = None
    for _, row in df.iterrows():
        preco = row["preco_atual"]
        sinal = ""
        if preco_anterior is not None:
            sinal = " v" if preco < preco_anterior else (" ^" if preco > preco_anterior else "")
        original = f"{row['preco_original']:.2f}EUR" if row["preco_original"] else "—"
        promo = "sim" if row["em_promocao"] else "—"
        print(
            f"  {str(row['data_recolha'])[:19]:<22} "
            f"{preco:>7.2f}EUR{sinal:<2} "
            f"{original:>10} "
            f"{promo:>9} "
            f"{str(row['agente_origem'] or ''):<25}"
        )
        preco_anterior = preco


# ---------------------------------------------------------------------------
# Estatísticas gerais da BD
# ---------------------------------------------------------------------------

def imprimir_estatisticas_bd() -> None:
    """Imprime um resumo rápido do estado atual da base de dados."""
    queries: dict[str, str] = {
        "Produtos monitorizados": "SELECT COUNT(DISTINCT id_produto_loja) FROM precos_atuais",
        "Lojas ativas":           "SELECT COUNT(*) FROM lojas",
        "Registos historicos":    "SELECT COUNT(*) FROM historico_precos",
        "Em promocao agora":      "SELECT COUNT(*) FROM precos_atuais WHERE em_promocao = TRUE",
        "Ultimo scraping":        "SELECT MAX(data_recolha)::text FROM historico_precos",
        "Dias de historico":      "SELECT COUNT(DISTINCT data_recolha::date) FROM historico_precos",
    }

    print("\n[ESTADO DA BASE DE DADOS]")
    with get_connection() as conn:
        with conn.cursor() as cur:
            for label, query in queries.items():
                cur.execute(query)
                valor = cur.fetchone()[0]
                print(f"  {label:<28}: {valor}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI do monitor de preços."""
    parser = argparse.ArgumentParser(
        description="Monitor de precos — deteta mudancas e gera alertas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Monitorizar mudancas nas ultimas 24h (default)
  python scripts/price_monitor.py

  # Mudancas nas ultimas 48h com threshold de 3%
  python scripts/price_monitor.py --horas 48 --threshold 3.0

  # Historico de um produto especifico (ultima semana)
  python scripts/price_monitor.py --produto-id 42

  # So estatisticas da BD
  python scripts/price_monitor.py --stats

  # Guardar alertas em CSV
  python scripts/price_monitor.py --guardar
        """,
    )
    parser.add_argument(
        "--horas",
        type=int,
        default=24,
        help="Janela temporal de analise em horas (default: 24).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Variacao minima em %% para gerar alerta (default: 2.0).",
    )
    parser.add_argument(
        "--produto-id",
        type=int,
        help="Ver historico de um produto-loja especifico (id_produto_loja).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar apenas estatisticas resumidas da BD.",
    )
    parser.add_argument(
        "--guardar",
        action="store_true",
        help="Guardar alertas gerados em CSV (data/alertas/).",
    )
    args = parser.parse_args()

    if args.stats:
        imprimir_estatisticas_bd()
        return

    if args.produto_id:
        df_hist = detetar_mudancas_produto(args.produto_id, janela_horas=args.horas)
        imprimir_historico_produto(df_hist, args.produto_id)
        imprimir_estatisticas_bd()
        return

    # Monitorização geral
    print(f"[MONITOR] A analisar mudancas de preco nas ultimas {args.horas}h...")
    df_mudancas = detetar_mudancas_preco(janela_horas=args.horas)
    alertas = gerar_alertas(df_mudancas, threshold_pct=args.threshold)

    imprimir_relatorio(alertas, args.horas)
    imprimir_estatisticas_bd()

    if args.guardar and not alertas.empty:
        caminho = guardar_alertas(alertas)
        print(f"\n[MONITOR] Alertas guardados em: {caminho}")


if __name__ == "__main__":
    main()
