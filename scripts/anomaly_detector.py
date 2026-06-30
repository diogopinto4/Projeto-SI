"""
Detetor de anomalias de preços — Sensorização e Ambiente.

Identifica preços suspeitos em ``historico_precos`` usando dois critérios
independentes:

1. **IQR por produto** — preços fora de ``[Q1 - k·IQR, Q3 + k·IQR]``
   (com k configurável via ``--desvios``). Aplicado apenas a produtos com
   IQR > 0 (i.e., com alguma variação histórica).

2. **Variação abrupta** — variação relativa superior a ``--variacao`` entre
   dias consecutivos. Apanha erros de scraping pontuais (ex: 1€ → 100€).

Deve correr após cada ingestão para sinalizar dados potencialmente corrompidos
antes de entrarem no treino do LSTM.

Uso::

    python scripts/anomaly_detector.py
    python scripts/anomaly_detector.py --desvios 4.0 --variacao 0.4
    python scripts/anomaly_detector.py --output data/anomalias.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Query que carrega o histórico completo com metadados de produto e cadeia.
_QUERY_HISTORICO = """
    SELECT
        hp.id_historico,
        hp.id_produto_loja,
        pm.nome_padronizado,
        l.insignia          AS cadeia,
        hp.preco_atual::float,
        hp.data_recolha::date AS dia
    FROM historico_precos hp
    JOIN produtos_loja   pl ON pl.id_produto_loja   = hp.id_produto_loja
    JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
    JOIN lojas            l  ON l.id_loja            = pl.id_loja
    ORDER BY hp.id_produto_loja, hp.data_recolha
"""


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

def carregar_historico() -> pd.DataFrame:
    """Carrega o histórico de preços da BD para um DataFrame.

    Returns:
        DataFrame com colunas ``id_historico``, ``id_produto_loja``,
        ``nome_padronizado``, ``cadeia``, ``preco_atual``, ``dia``.

    Raises:
        psycopg2.OperationalError: Se a ligação à BD falhar.
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(_QUERY_HISTORICO)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    df["dia"] = pd.to_datetime(df["dia"])
    return df


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------

def detetar_anomalias(
    df: pd.DataFrame,
    desvios_iqr: float = 3.0,
    variacao_max_pct: float = 0.5,
    max_gap_dias: int = 3,
) -> pd.DataFrame:
    """Deteta preços anómalos usando critérios IQR e variação abrupta.

    Args:
        df: DataFrame de :func:`carregar_historico`.
        desvios_iqr: Multiplicador do IQR para os limites (default: 3.0).
            Valores mais baixos são mais restritivos.
        variacao_max_pct: Variação relativa máxima permitida entre dias
            consecutivos (default: 0.5 = 50%). Acima deste valor o registo
            é considerado uma variação abrupta.
        max_gap_dias: Gap máximo em dias entre observações consecutivas para
            aplicar o critério de variação abrupta (default: 3). Gaps maiores
            indicam ausência de scraping — a variação pode ser legítima.

    Returns:
        DataFrame com as linhas suspeitas, deduplicadas por ``id_historico``,
        com as colunas originais mais ``motivo`` (string descritiva) e
        ``criterio`` (``"iqr"`` ou ``"variacao_abrupta"``).
        Devolve DataFrame vazio se não houver anomalias.

    Note:
        Os limites IQR são arredondados a 4 casas decimais para evitar falsos
        positivos por imprecisão de vírgula flutuante (ex: ``3.39 < 3.39``
        a avaliar como ``True`` por acumulação de erros de representação).
    """
    anomalias: list[dict] = []

    for _, grupo in df.groupby("id_produto_loja"):
        g = grupo.sort_values("dia").copy()
        precos = g["preco_atual"]

        # Critério 1: IQR
        q1 = precos.quantile(0.25)
        q3 = precos.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            # Arredondamento a 4 casas para evitar falsos positivos por
            # imprecisão de vírgula flutuante (ex: lim_inf = 3.390000000001)
            lim_inf = round(q1 - desvios_iqr * iqr, 4)
            lim_sup = round(q3 + desvios_iqr * iqr, 4)
            for _, row in g[(precos < lim_inf) | (precos > lim_sup)].iterrows():
                anomalias.append({
                    **row.to_dict(),
                    "motivo":   f"IQR — {row['preco_atual']:.2f}EUR fora de [{lim_inf:.2f}, {lim_sup:.2f}]",
                    "criterio": "iqr",
                })

        # Critério 2: variação abrupta entre observações consecutivas próximas.
        # Só se aplica quando o gap temporal é <= max_gap_dias para evitar
        # falsos positivos em produtos com longos períodos sem scraping
        # (ex: scraped em março e novamente em abril — qualquer variação seria
        # flaggada mesmo sendo legítima).
        variacao = precos.pct_change().abs()
        gap_dias = g["dia"].diff().dt.days.fillna(0)
        mascara = (variacao > variacao_max_pct) & (gap_dias <= max_gap_dias)
        for idx, row in g[mascara].iterrows():
            anomalias.append({
                **row.to_dict(),
                "motivo":   f"Variacao abrupta — {variacao[idx]*100:.1f}% face ao dia anterior",
                "criterio": "variacao_abrupta",
            })

    if not anomalias:
        return pd.DataFrame()

    return (
        pd.DataFrame(anomalias)
        .drop_duplicates(subset=["id_historico"])
        .sort_values(["id_produto_loja", "dia"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI do detetor de anomalias."""
    parser = argparse.ArgumentParser(
        description="Deteta preços anómalos em historico_precos por IQR e variação abrupta."
    )
    parser.add_argument(
        "--desvios",
        type=float,
        default=3.0,
        help="Multiplicador IQR para os limites (default: 3.0). Menor = mais restritivo.",
    )
    parser.add_argument(
        "--variacao",
        type=float,
        default=0.5,
        help="Variacao relativa maxima entre dias consecutivos (default: 0.5 = 50%%).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho CSV de saida com as anomalias (opcional).",
    )
    args = parser.parse_args()

    print("[ANOMALIAS] A carregar historico de precos da BD...")
    df = carregar_historico()
    print(f"  {len(df)} registos | {df['id_produto_loja'].nunique()} produtos")

    print(f"\n[ANOMALIAS] A analisar (IQR x{args.desvios}, variacao max {args.variacao*100:.0f}%)...")
    df_anomalias = detetar_anomalias(df, desvios_iqr=args.desvios, variacao_max_pct=args.variacao)

    if df_anomalias.empty:
        print("[ANOMALIAS] Nenhuma anomalia detectada — dados com boa qualidade.")
        return

    print(f"\n[ANOMALIAS] {len(df_anomalias)} registos suspeitos:")
    cols = ["nome_padronizado", "cadeia", "preco_atual", "dia", "motivo"]
    print(df_anomalias[cols].to_string(index=False))

    iqr_n = (df_anomalias["criterio"] == "iqr").sum()
    var_n = (df_anomalias["criterio"] == "variacao_abrupta").sum()
    print(f"\n  Por criterio: IQR={iqr_n} | Variacao abrupta={var_n}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_anomalias.to_csv(out, index=False)
        print(f"  Guardado em: {out}")


if __name__ == "__main__":
    main()
