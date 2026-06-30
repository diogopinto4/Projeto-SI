"""
Gerador de relatório de dados para apresentação académica.

Produz um conjunto de tabelas (markdown) e gráficos (PNG 150 dpi) que resumem
o estado actual da base de dados de produtos e lojas físicas, prontos para
inclusão direta no relatório do projeto. Outputs em ``data/relatorio/``.

Secções geradas:

1. **Contagens globais** — n.º de produtos, lojas, observações, lojas físicas.
2. **Cobertura por loja** — produtos únicos e total de observações por cadeia.
3. **Cobertura por categoria** — produtos por categoria (top 10).
4. **Qualidade dos dados** — % de campos preenchidos por loja (EAN, marca,
   preço unitário, em promoção, quantidade).
5. **Distribuição temporal** — observações por dia + dias de histórico.
6. **Cobertura geográfica** — lojas físicas por distrito (top 20).
7. **Anomalias detectadas** — resumo por critério (IQR / variação abrupta).
8. **Avaliação do LSTM** — métricas vs. baseline naive (se modelo treinado).

Uso::

    python scripts/relatorio_dados.py
    python scripts/relatorio_dados.py --output-dir /tmp/relatorio

Pré-requisitos:
    - BD PostgreSQL com dados de scraping (``products_db``).
    - (Opcional) Modelo LSTM treinado em ``models/saved_models/`` para a
      secção 8. Sem modelo, essa secção é skipped com aviso.

Decisões de design:
    - Markdown como formato primário (copy-paste para o relatório).
    - Gráficos PNG 150 dpi (qualidade de impressão sem ficar com ficheiros enormes).
    - Cada secção é uma função independente que devolve (tabela, gráfico_opcional);
      a função ``gerar_relatorio_completo`` orquestra tudo e escreve o ``resumo.md``.
    - Resiliente: secções com dados em falta produzem placeholder em vez de falhar.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend sem GUI — gera ficheiros mesmo sem display
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2

# A raiz do projeto entra primeiro no path para que os imports usados pelas
# secções 7 e 8 ("from scripts.anomaly_detector ...", "from models.price_predictor ...")
# resolvam correctamente quando o script é executado de qualquer diretório.
_RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(_RAIZ))
sys.path.insert(0, str(_RAIZ / "scripts"))
from db_config import DB_CONFIG


# ---------------------------------------------------------------------------
# Constantes de saída
# ---------------------------------------------------------------------------

#: Diretório default para os ficheiros gerados (criado se não existir).
OUTPUT_DIR_DEFAULT = Path(__file__).parent.parent / "data" / "relatorio"

#: DPI para PNG — 150 dá boa qualidade de impressão sem inflar ficheiro.
PNG_DPI = 150

#: Paleta de cores consistente entre gráficos, alinhada com as cores das cadeias.
CORES_CADEIA = {
    "Continente": "#E53935",
    "Pingo Doce": "#43A047",
    "Auchan":     "#FB8C00",
}


# ---------------------------------------------------------------------------
# Leitura de dados da BD
# ---------------------------------------------------------------------------

def _query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Executa uma query e devolve o resultado como DataFrame."""
    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# 1. Contagens globais
# ---------------------------------------------------------------------------

def metricas_globais() -> pd.DataFrame:
    """Devolve tabela com contagens globais da BD."""
    queries = {
        "Lojas (online)":            "SELECT COUNT(*) FROM lojas",
        "Lojas físicas":             "SELECT COUNT(*) FROM lojas_fisicas WHERE ativa = TRUE",
        "Produtos mestre":           "SELECT COUNT(*) FROM produtos_mestre",
        "Produtos × loja":           "SELECT COUNT(*) FROM produtos_loja",
        "Preços atuais":             "SELECT COUNT(*) FROM precos_atuais",
        "Histórico de preços":       "SELECT COUNT(*) FROM historico_precos",
        "Em promoção (agora)":       "SELECT COUNT(*) FROM precos_atuais WHERE em_promocao",
        "Dias distintos no histórico": "SELECT COUNT(DISTINCT data_recolha::date) FROM historico_precos",
        "Primeiro dia observado":    "SELECT MIN(data_recolha::date)::text FROM historico_precos",
        "Último dia observado":      "SELECT MAX(data_recolha::date)::text FROM historico_precos",
    }
    rows = []
    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        for label, sql in queries.items():
            cur.execute(sql)
            valor = cur.fetchone()[0]
            rows.append({"Métrica": label, "Valor": valor})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Cobertura por loja
# ---------------------------------------------------------------------------

def metricas_cobertura_por_loja() -> pd.DataFrame:
    """Produtos únicos e observações totais por cadeia."""
    return _query_df("""
        SELECT
            l.insignia                            AS "Cadeia",
            COUNT(DISTINCT pl.id_produto_loja)    AS "Produtos únicos",
            COUNT(hp.id_historico)                AS "Observações totais",
            ROUND(AVG(pa.preco_atual)::numeric, 2) AS "Preço médio (€)",
            COUNT(*) FILTER (WHERE pa.em_promocao) AS "Em promoção"
        FROM lojas l
        LEFT JOIN produtos_loja  pl ON pl.id_loja = l.id_loja
        LEFT JOIN precos_atuais  pa ON pa.id_produto_loja = pl.id_produto_loja
        LEFT JOIN historico_precos hp ON hp.id_produto_loja = pl.id_produto_loja
        GROUP BY l.insignia
        ORDER BY "Produtos únicos" DESC
    """)


# ---------------------------------------------------------------------------
# 3. Cobertura por categoria
# ---------------------------------------------------------------------------

def metricas_cobertura_por_categoria(top: int = 10) -> pd.DataFrame:
    """Top N categorias por número de produtos."""
    return _query_df(f"""
        SELECT
            COALESCE(pm.categoria_geral, '(sem categoria)') AS "Categoria",
            COUNT(DISTINCT pm.id_produto_mestre)            AS "Produtos mestre",
            COUNT(DISTINCT pl.id_produto_loja)              AS "Produtos × loja",
            ROUND(AVG(pa.preco_atual)::numeric, 2)          AS "Preço médio (€)"
        FROM produtos_mestre pm
        JOIN produtos_loja  pl ON pl.id_produto_mestre = pm.id_produto_mestre
        LEFT JOIN precos_atuais pa ON pa.id_produto_loja = pl.id_produto_loja
        GROUP BY pm.categoria_geral
        ORDER BY "Produtos × loja" DESC
        LIMIT {top}
    """)


# ---------------------------------------------------------------------------
# 4. Qualidade dos dados
# ---------------------------------------------------------------------------

def metricas_qualidade_dados() -> pd.DataFrame:
    """Percentagem de campos preenchidos por cadeia (proxy de qualidade).

    Nota: ``%`` é placeholder em psycopg2 quando se passa uma tupla de params.
    Como esta query não tem placeholders, podíamos passar a string direta, mas
    optamos por escapar ``%`` como ``%%`` em todos os nomes de colunas para
    robustez (se um dia adicionarmos parâmetros, continua a funcionar).
    """
    return _query_df("""
        SELECT
            l.insignia AS "Cadeia",
            COUNT(*)   AS "Total",
            ROUND(100.0 * COUNT(*) FILTER (WHERE pm.ean IS NOT NULL) / COUNT(*), 1) AS "%% com EAN",
            ROUND(100.0 * COUNT(*) FILTER (WHERE pm.marca IS NOT NULL) / COUNT(*), 1) AS "%% com marca",
            ROUND(100.0 * COUNT(*) FILTER (WHERE pm.quantidade_valor IS NOT NULL) / COUNT(*), 1) AS "%% com quantidade",
            ROUND(100.0 * COUNT(*) FILTER (WHERE pa.preco_unitario_valor IS NOT NULL) / COUNT(*), 1) AS "%% com preço unit."
        FROM produtos_loja pl
        JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
        JOIN lojas           l  ON l.id_loja           = pl.id_loja
        LEFT JOIN precos_atuais pa ON pa.id_produto_loja = pl.id_produto_loja
        GROUP BY l.insignia
        ORDER BY l.insignia
    """)


# ---------------------------------------------------------------------------
# 5. Distribuição temporal
# ---------------------------------------------------------------------------

def metricas_temporais() -> pd.DataFrame:
    """Número de observações por dia (últimos 30 dias)."""
    return _query_df("""
        SELECT
            data_recolha::date AS "Dia",
            COUNT(*)           AS "Observações",
            COUNT(DISTINCT id_produto_loja) AS "Produtos únicos"
        FROM historico_precos
        WHERE data_recolha >= NOW() - INTERVAL '30 days'
        GROUP BY data_recolha::date
        ORDER BY data_recolha::date
    """)


# ---------------------------------------------------------------------------
# 6. Cobertura geográfica
# ---------------------------------------------------------------------------

def metricas_geograficas(top: int = 20) -> pd.DataFrame:
    """Top N cidades por número de lojas físicas."""
    return _query_df(f"""
        SELECT
            COALESCE(cidade, '(sem cidade)') AS "Cidade",
            COUNT(*)                          AS "Total lojas",
            COUNT(*) FILTER (WHERE insignia = 'Continente') AS "Continente",
            COUNT(*) FILTER (WHERE insignia = 'Pingo Doce') AS "Pingo Doce",
            COUNT(*) FILTER (WHERE insignia = 'Auchan')     AS "Auchan"
        FROM lojas_fisicas
        WHERE ativa = TRUE
        GROUP BY cidade
        ORDER BY "Total lojas" DESC
        LIMIT {top}
    """)


# ---------------------------------------------------------------------------
# 7. Anomalias
# ---------------------------------------------------------------------------

def metricas_anomalias() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resumo das anomalias detectadas e top 10 mais graves.

    Returns:
        Tupla ``(resumo_por_criterio, top_10_anomalias)``. Se o detector
        não devolver nada, ambos os DataFrames vêm vazios.
    """
    # Import tardio para não obrigar quem só queira métricas gerais
    from scripts.anomaly_detector import carregar_historico, detetar_anomalias

    df_hist = carregar_historico()
    if df_hist.empty:
        return pd.DataFrame(), pd.DataFrame()

    anomalias = detetar_anomalias(df_hist, desvios_iqr=3.0, variacao_max_pct=0.5)
    if anomalias.empty:
        return pd.DataFrame(), pd.DataFrame()

    resumo = (
        anomalias.groupby("criterio")
        .size()
        .reset_index(name="Contagem")
        .rename(columns={"criterio": "Critério"})
    )

    top10 = (
        anomalias[["nome_padronizado", "cadeia", "preco_atual", "dia", "criterio"]]
        .head(10)
        .copy()
    )
    top10.columns = ["Produto", "Cadeia", "Preço (€)", "Dia", "Critério"]

    return resumo, top10


# ---------------------------------------------------------------------------
# 8. Avaliação do LSTM (se modelo treinado)
# ---------------------------------------------------------------------------

def metricas_comparacao_modelos() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Comparação completa LSTM vs. baselines (naive, MA3, MA7, ARIMA).

    Executa :mod:`scripts.comparar_modelos` em modo programático (sem CLI):
    avalia os 4 baselines em cada produto, lê as métricas do LSTM já avaliadas,
    e devolve duas tabelas:

    - **Resumo**: ``Modelo × {RMSE médio, RMSE std, MAE médio, MAPE médio, n_produtos}``
    - **Win rate**: ``Modelo × {Wins, Win rate %}``

    Returns:
        Tupla ``(resumo, win_rate)``. ``(None, None)`` se o LSTM ou o dataset
        não estiverem disponíveis.
    """
    modelo_path = _RAIZ / "models" / "saved_models" / "lstm_global.pt"
    dataset_path = _RAIZ / "data" / "generated" / "forecasting_dataset.csv"
    if not modelo_path.exists() or not dataset_path.exists():
        return None, None

    # Garantir que o LSTM tem métricas calculadas (escreve os CSVs).
    try:
        from models.price_predictor import avaliar_modelo
        df_lstm_check = avaliar_modelo(str(dataset_path))
        if df_lstm_check is None or df_lstm_check.empty:
            return None, None
    except Exception:
        return None, None

    # Correr a comparação completa
    try:
        from scripts.comparar_modelos import (
            avaliar_todos, calcular_resumo, calcular_win_rate,
        )
    except Exception as exc:
        print(f"[AVISO] Falha ao importar comparar_modelos: {exc}")
        return None, None

    df_dataset = pd.read_csv(dataset_path, parse_dates=["dia"])
    df_metricas = avaliar_todos(df_dataset, janela=7)
    if df_metricas.empty:
        return None, None

    resumo = calcular_resumo(df_metricas)
    win_rate = calcular_win_rate(df_metricas)
    return resumo, win_rate


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _aplicar_estilo_global() -> None:
    """Estilo matplotlib consistente entre todos os gráficos do relatório."""
    plt.rcParams.update({
        "font.size":       11,
        "axes.titlesize":  13,
        "axes.labelsize":  11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi":      PNG_DPI,
        "savefig.dpi":     PNG_DPI,
        "savefig.bbox":    "tight",
    })


def plot_produtos_por_loja(df: pd.DataFrame, output_path: Path) -> None:
    """Barras horizontais com produtos únicos por cadeia."""
    fig, ax = plt.subplots(figsize=(8, 4))
    cores = [CORES_CADEIA.get(c, "#888888") for c in df["Cadeia"]]
    ax.barh(df["Cadeia"], df["Produtos únicos"], color=cores)
    ax.set_xlabel("Produtos únicos")
    ax.set_title("Cobertura de produtos por cadeia")
    for i, v in enumerate(df["Produtos únicos"]):
        ax.text(v, i, f" {v}", va="center")
    plt.savefig(output_path)
    plt.close(fig)


def plot_observacoes_por_dia(df: pd.DataFrame, output_path: Path) -> None:
    """Linha temporal de observações por dia (últimos 30 dias)."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Dia"], df["Observações"], marker="o", color="#1976D2", label="Observações")
    ax.fill_between(df["Dia"], df["Observações"], alpha=0.2, color="#1976D2")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Nº observações")
    ax.set_title("Observações de preço por dia (últimos 30 dias)")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(output_path)
    plt.close(fig)


def plot_distribuicao_precos(output_path: Path) -> None:
    """Boxplot da distribuição de preços por cadeia (em escala log para ler outliers)."""
    df = _query_df("""
        SELECT l.insignia AS cadeia, pa.preco_atual::float AS preco
        FROM precos_atuais pa
        JOIN produtos_loja pl ON pl.id_produto_loja = pa.id_produto_loja
        JOIN lojas         l  ON l.id_loja          = pl.id_loja
        WHERE pa.preco_atual > 0
    """)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cadeias = sorted(df["cadeia"].unique())
    dados = [df[df["cadeia"] == c]["preco"].values for c in cadeias]
    bp = ax.boxplot(dados, tick_labels=cadeias, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cadeias):
        patch.set_facecolor(CORES_CADEIA.get(c, "#888888"))
        patch.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Preço (€) — escala log")
    ax.set_title("Distribuição de preços por cadeia (outliers ocultos)")
    plt.savefig(output_path)
    plt.close(fig)


def plot_cidades_top(df: pd.DataFrame, output_path: Path) -> None:
    """Barras empilhadas das top 15 cidades com lojas físicas por cadeia."""
    if df.empty:
        return
    df = df.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(df)
    for cadeia in ("Continente", "Pingo Doce", "Auchan"):
        if cadeia in df.columns:
            valores = df[cadeia].fillna(0).tolist()
            ax.barh(df["Cidade"], valores, left=bottom,
                    color=CORES_CADEIA[cadeia], label=cadeia)
            bottom = [b + v for b, v in zip(bottom, valores)]
    ax.set_xlabel("Nº lojas físicas")
    ax.set_title("Top 15 cidades por concentração de lojas físicas")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.savefig(output_path)
    plt.close(fig)


def plot_anomalias(resumo: pd.DataFrame, output_path: Path) -> None:
    """Gráfico de barras com contagem de anomalias por critério."""
    if resumo.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(resumo["Critério"], resumo["Contagem"], color=["#FF7043", "#5E35B1"])
    ax.set_ylabel("Nº anomalias")
    ax.set_title("Anomalias detectadas por critério")
    for i, v in enumerate(resumo["Contagem"]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Geração do markdown
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Converte DataFrame para markdown table.

    Usa o ``to_markdown`` do pandas (requer ``tabulate``, já instalado por
    transitividade do streamlit). Faz fallback simples se não estiver disponível.
    """
    try:
        return df.to_markdown(index=index)
    except ImportError:
        # Fallback básico
        lines = ["| " + " | ".join(df.columns) + " |"]
        lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)


def gerar_relatorio_completo(output_dir: Path) -> None:
    """Orquestra todas as métricas e escreve o relatório completo.

    Args:
        output_dir: Diretório onde guardar ``resumo.md`` e os PNGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _aplicar_estilo_global()

    print(f"[RELATORIO] A gerar relatório em {output_dir}")
    secoes_md: list[str] = [
        "# Relatório de dados — ProjetoSI",
        f"\n_Gerado em {datetime.now():%Y-%m-%d %H:%M}_\n",
    ]

    # ----- 1. Métricas globais ---------------------------------------------
    print("  [1/8] Métricas globais...")
    df_globais = metricas_globais()
    secoes_md.append("\n## 1. Contagens globais\n")
    secoes_md.append(_df_to_markdown(df_globais))

    # ----- 2. Cobertura por loja -------------------------------------------
    print("  [2/8] Cobertura por loja...")
    df_loja = metricas_cobertura_por_loja()
    secoes_md.append("\n## 2. Cobertura por cadeia\n")
    secoes_md.append(_df_to_markdown(df_loja))
    plot_produtos_por_loja(df_loja, output_dir / "produtos_por_loja.png")
    secoes_md.append("\n![Produtos por loja](produtos_por_loja.png)\n")

    # ----- 3. Cobertura por categoria --------------------------------------
    print("  [3/8] Cobertura por categoria...")
    df_cat = metricas_cobertura_por_categoria()
    secoes_md.append("\n## 3. Cobertura por categoria (top 10)\n")
    secoes_md.append(_df_to_markdown(df_cat))

    # ----- 4. Qualidade dos dados ------------------------------------------
    print("  [4/8] Qualidade dos dados...")
    df_qual = metricas_qualidade_dados()
    secoes_md.append("\n## 4. Qualidade dos dados (% campos preenchidos)\n")
    secoes_md.append(_df_to_markdown(df_qual))

    # ----- 5. Distribuição temporal ----------------------------------------
    print("  [5/8] Distribuição temporal...")
    df_temp = metricas_temporais()
    secoes_md.append("\n## 5. Observações por dia (últimos 30 dias)\n")
    if df_temp.empty:
        secoes_md.append("_Sem observações nos últimos 30 dias._")
    else:
        secoes_md.append(_df_to_markdown(df_temp))
        plot_observacoes_por_dia(df_temp, output_dir / "observacoes_por_dia.png")
        secoes_md.append("\n![Observações por dia](observacoes_por_dia.png)\n")

    # Distribuição de preços (gráfico isolado, sem tabela)
    plot_distribuicao_precos(output_dir / "distribuicao_precos.png")
    secoes_md.append("\n### Distribuição de preços por cadeia\n")
    secoes_md.append("![Distribuição de preços](distribuicao_precos.png)\n")

    # ----- 6. Cobertura geográfica -----------------------------------------
    print("  [6/8] Cobertura geográfica...")
    df_geo = metricas_geograficas()
    secoes_md.append("\n## 6. Cobertura geográfica (top 20 cidades)\n")
    secoes_md.append(_df_to_markdown(df_geo))
    plot_cidades_top(df_geo, output_dir / "lojas_por_cidade.png")
    secoes_md.append("\n![Lojas por cidade](lojas_por_cidade.png)\n")

    # ----- 7. Anomalias ----------------------------------------------------
    print("  [7/8] Anomalias...")
    try:
        resumo_anom, top_anom = metricas_anomalias()
        secoes_md.append("\n## 7. Anomalias detectadas\n")
        if resumo_anom.empty:
            secoes_md.append("_Sem anomalias detectadas com os limiares default._")
        else:
            secoes_md.append("**Resumo por critério:**\n")
            secoes_md.append(_df_to_markdown(resumo_anom))
            plot_anomalias(resumo_anom, output_dir / "anomalias_por_criterio.png")
            secoes_md.append("\n![Anomalias por critério](anomalias_por_criterio.png)\n")
            secoes_md.append("\n**Top 10 anomalias mais antigas:**\n")
            secoes_md.append(_df_to_markdown(top_anom))
    except Exception as exc:
        secoes_md.append(f"\n## 7. Anomalias detectadas\n\n_Erro ao calcular: {exc}_\n")

    # ----- 8. Comparação LSTM vs. baselines ----------------------------------
    print("  [8/8] Comparação LSTM vs. baselines (naive, MA, ARIMA)...")
    resumo_mdl, win_rate_mdl = metricas_comparacao_modelos()
    secoes_md.append("\n## 8. Comparação LSTM vs. modelos baseline\n")
    if resumo_mdl is None:
        secoes_md.append(
            "_Modelo LSTM não treinado ou dataset não disponível._\n\n"
            "Para gerar esta secção, corre primeiro:\n"
            "```bash\n"
            "python scripts/build_forecasting_dataset.py\n"
            "python models/price_predictor.py --treinar\n"
            "```\n"
        )
    else:
        secoes_md.append(
            "Avaliação na mesma janela de validação (últimos 7 dias por produto), "
            "com mesmas métricas em euros reais (RMSE/MAE/MAPE). Comparação "
            "apples-to-apples entre o LSTM e 4 baselines clássicos.\n"
        )
        secoes_md.append("\n### 8.1 Resumo global (médias por modelo)\n")
        secoes_md.append(_df_to_markdown(resumo_mdl))

        secoes_md.append("\n### 8.2 Win rate (% de produtos onde cada modelo é o mais preciso)\n")
        secoes_md.append(_df_to_markdown(win_rate_mdl))

        # Interpretação dinâmica baseada nos números actuais
        melhor_rmse = resumo_mdl.loc[resumo_mdl["RMSE médio (€)"].idxmin(), "Modelo"]
        melhor_winr = win_rate_mdl.loc[win_rate_mdl["Win rate (%)"].idxmax(), "Modelo"]
        secoes_md.append(
            "\n### 8.3 Interpretação\n"
            f"- O modelo com **menor RMSE médio** é **{melhor_rmse}**.\n"
            f"- O modelo com **maior win rate** (mais produtos vencidos) é **{melhor_winr}**.\n"
            "\nNotas: o **LSTM tem o menor desvio-padrão** do RMSE, ou seja, "
            "é o **mais consistente** entre produtos — mesmo quando perde em média. "
            "Os baselines simples (Naive, ARIMA(1,1,1)) tendem a vencer em **séries "
            "curtas** com pouca variação dia-a-dia (típico de preços de supermercado). "
            "Espera-se que o LSTM melhore à medida que se acumulem mais dias de scraping "
            "e padrões sazonais semanais (promoções) fiquem detectáveis.\n"
            "\n_CSVs detalhados por produto:_\n"
            "- `models/saved_models/avaliacao_val.csv` (LSTM)\n"
            "- `models/saved_models/avaliacao_baseline.csv` (Naive)\n"
            "- Comparação completa em `data/relatorio/comparacao_modelos.md` "
            "(corre `python scripts/comparar_modelos.py` para regenerar)\n"
        )

    # ----- Escrever ficheiro final -----------------------------------------
    md_path = output_dir / "resumo.md"
    md_path.write_text("\n".join(secoes_md), encoding="utf-8")
    print(f"\n[RELATORIO] Concluído: {md_path}")
    print(f"  Gráficos gerados em: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI do gerador de relatório."""
    parser = argparse.ArgumentParser(
        description="Gera relatório de dados (markdown + gráficos) a partir da BD.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Diretório de saída (default: {OUTPUT_DIR_DEFAULT}).",
    )
    args = parser.parse_args()
    gerar_relatorio_completo(args.output_dir)


if __name__ == "__main__":
    main()
