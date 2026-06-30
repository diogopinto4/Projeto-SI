"""
Comparação entre modelos de previsão de preços.

Avalia 6 modelos (naive, média móvel 3 dias, média móvel 7 dias, ARIMA(1,1,1),
gradient boosting e LSTM) no mesmo conjunto de validação por produto (últimos
``janela=7`` dias) e produz:

1. Tabela comparativa global: RMSE/MAE/MAPE médios + std + win rate.
2. Tabela de win rate por modelo (% de produtos onde cada um é melhor).
3. Boxplot do RMSE por modelo.
4. Gráfico de barras com win rate por modelo.
5. Ficheiro markdown ``data/relatorio/comparacao_modelos.md``.

Pré-requisitos:
    - Dataset de forecasting: ``python scripts/build_forecasting_dataset.py``
    - LSTM treinado: ``python models/price_predictor.py --treinar``

Uso::

    python scripts/comparar_modelos.py
    python scripts/comparar_modelos.py --max-produtos 100   # subset rápido
    python scripts/comparar_modelos.py --output-dir /tmp/comp

Decisões (justificadas no relatório):

* **Mesma janela de validação que o LSTM** (últimos 7 dias). Comparação
  apples-to-apples — se mudássemos o split, o LSTM seria penalizado.

* **Win rate como métrica secundária**: a média global pode esconder
  heterogeneidade (modelo A melhor para 80% dos produtos mas RMSE muito alto
  em 20%). O win rate complementa as médias.

* **ARIMA pode falhar** em produtos com séries constantes ou muito curtas;
  esses são contados como "ARIMA não convergiu" e excluídos da média ARIMA
  (mas continuam nos outros modelos).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Raiz do projeto no path para imports limpos
_RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(_RAIZ))
sys.path.insert(0, str(_RAIZ / "scripts"))

from models.baselines import BASELINES, aplicar_baseline, calcular_metricas_em_euros
from models.gbm_predictor import avaliar_gbm_por_produto


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

OUTPUT_DIR_DEFAULT = _RAIZ / "data" / "relatorio"
DATASET_DEFAULT = _RAIZ / "data" / "generated" / "forecasting_dataset.csv"
PNG_DPI = 150

#: Ordem das colunas/labels usados nas tabelas e gráficos. Baselines primeiro,
#: depois os modelos treinados (gradient boosting e LSTM) ao final.
ORDEM_MODELOS = ["naive", "media_3", "media_7", "arima_111", "gbm", "lstm"]

#: Labels "legíveis" para apresentação. Mapeiam chaves técnicas → texto.
LABELS_MODELOS = {
    "naive":     "Naive",
    "media_3":   "Média móvel (3d)",
    "media_7":   "Média móvel (7d)",
    "arima_111": "ARIMA(1,1,1)",
    "gbm":       "Gradient Boosting",
    "lstm":      "LSTM",
}

#: Cores consistentes nos gráficos. Modelos treinados destacados a cor própria.
CORES_MODELOS = {
    "naive":     "#9E9E9E",
    "media_3":   "#42A5F5",
    "media_7":   "#1976D2",
    "arima_111": "#FB8C00",
    "gbm":       "#43A047",
    "lstm":      "#E53935",
}


# ---------------------------------------------------------------------------
# Avaliação por produto
# ---------------------------------------------------------------------------

def _avaliar_baselines_por_produto(
    df_dataset: pd.DataFrame,
    janela: int,
    max_produtos: int | None,
) -> pd.DataFrame:
    """Aplica cada baseline a cada produto e devolve métricas detalhadas.

    Usa o mesmo split temporal que o ``avaliar_modelo`` do LSTM: últimos
    ``janela`` dias = validação; tudo o anterior = treino.

    Args:
        df_dataset: DataFrame do CSV de forecasting (``forecasting_dataset.csv``).
        janela: Dias de validação por produto.
        max_produtos: Se fornecido, limita a este número de produtos (útil
            para iterações rápidas).

    Returns:
        DataFrame "long": uma linha por (produto, modelo) com colunas
        ``produto_id``, ``modelo``, ``rmse_euros``, ``mae_euros``, ``mape_pct``,
        ``n_amostras``.
    """
    produtos = df_dataset["id_produto_loja"].unique()
    if max_produtos is not None:
        produtos = produtos[:max_produtos]

    rows: list[dict] = []
    ignorados = 0

    for produto_id in produtos:
        df_p = (
            df_dataset[df_dataset["id_produto_loja"] == produto_id]
            .sort_values("dia")
            .dropna(subset=["preco"])
            .copy()
        )
        if len(df_p) < janela * 2 + 1:
            ignorados += 1
            continue

        # Split temporal: últimos `janela` dias = validação
        serie = df_p["preco"].to_numpy()
        treino = serie[:-janela]
        validacao = serie[-janela:]

        # Avaliar cada baseline
        for nome in BASELINES:
            pred = aplicar_baseline(nome, treino, horizonte=janela)
            metricas = calcular_metricas_em_euros(validacao, pred)
            rows.append({
                "produto_id":   int(produto_id),
                "modelo":       nome,
                "n_amostras":   len(validacao),
                **metricas,
            })

    print(f"[BASELINES] Produtos avaliados: {len(produtos) - ignorados} "
          f"(ignorados {ignorados} com história demasiado curta)")
    return pd.DataFrame(rows)


def _ler_metricas_lstm(janela: int) -> pd.DataFrame:
    """Lê os CSVs de avaliação do LSTM já produzidos por ``avaliar_modelo``.

    Não retreina nem re-avalia o LSTM aqui — assume que o utilizador já correu
    ``python models/price_predictor.py --avaliar`` ou que o pipeline auto-correu
    durante o ``relatorio_dados.py``. Se não houver CSV, devolve DataFrame vazio.
    """
    csv_lstm = _RAIZ / "models" / "saved_models" / "avaliacao_val.csv"
    if not csv_lstm.exists():
        print(f"[LSTM] CSV não encontrado em {csv_lstm}.")
        print("       Corre primeiro: python models/price_predictor.py --avaliar")
        return pd.DataFrame()

    df = pd.read_csv(csv_lstm)
    df["modelo"] = "lstm"
    # Garantir as mesmas colunas que os baselines
    cols = ["produto_id", "modelo", "n_amostras",
            "rmse_euros", "mae_euros", "mape_pct"]
    return df[cols]


def avaliar_todos(
    df_dataset: pd.DataFrame,
    janela: int = 7,
    max_produtos: int | None = None,
) -> pd.DataFrame:
    """Avalia todos os modelos (baselines + gradient boosting + LSTM).

    O gradient boosting é treinado e avaliado aqui (rápido); o LSTM é lido de
    CSV pré-computado. A comparação é restrita aos produtos comuns a **todos**
    os modelos presentes, para ser justa.

    Args:
        df_dataset: DataFrame do ``forecasting_dataset.csv``.
        janela: Dias de validação. Default 7 — igual ao usado pelo LSTM.
        max_produtos: Limite (para testes rápidos).

    Returns:
        DataFrame "long" combinando todos os modelos (mesma estrutura). O LSTM
        só entra se o CSV de avaliação existir.
    """
    df_base = _avaliar_baselines_por_produto(df_dataset, janela, max_produtos)
    df_gbm = avaliar_gbm_por_produto(df_dataset, janela=janela, max_produtos=max_produtos)
    df_lstm = _ler_metricas_lstm(janela)

    frames = [df_base, df_gbm]
    if df_lstm.empty:
        print("[AVISO] LSTM ausente da comparação — só baselines + GBM serão avaliados.")
    else:
        frames.append(df_lstm)

    # Interseção de produtos comuns a TODOS os modelos presentes (comparação justa).
    conjuntos = [set(f["produto_id"].unique()) for f in frames if not f.empty]
    if not conjuntos:
        return pd.DataFrame()
    intersecao = set.intersection(*conjuntos)
    print(f"[COMPARAÇÃO] Produtos comuns a todos os modelos: {len(intersecao)}")

    return pd.concat(
        [f[f["produto_id"].isin(intersecao)] for f in frames if not f.empty],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Análise agregada
# ---------------------------------------------------------------------------

def calcular_resumo(df_metricas: pd.DataFrame) -> pd.DataFrame:
    """Resumo por modelo: média, std, mediana, n_produtos avaliados.

    Args:
        df_metricas: DataFrame "long" devolvido por :func:`avaliar_todos`.

    Returns:
        DataFrame com 1 linha por modelo, ordenado por :data:`ORDEM_MODELOS`.
    """
    rows = []
    for modelo in ORDEM_MODELOS:
        sub = df_metricas[df_metricas["modelo"] == modelo]
        if sub.empty:
            continue
        # Excluir NaN das médias (e.g. ARIMA que não convergiu)
        rows.append({
            "Modelo":          LABELS_MODELOS[modelo],
            "Produtos":        int(sub["rmse_euros"].notna().sum()),
            "RMSE médio (€)":  round(sub["rmse_euros"].mean(skipna=True), 4),
            "RMSE std":        round(sub["rmse_euros"].std(skipna=True), 4),
            "MAE médio (€)":   round(sub["mae_euros"].mean(skipna=True), 4),
            "MAPE médio (%)":  round(sub["mape_pct"].mean(skipna=True), 2),
        })
    return pd.DataFrame(rows)


def calcular_win_rate(df_metricas: pd.DataFrame) -> pd.DataFrame:
    """Para cada produto, identifica o modelo com menor RMSE e calcula win rate.

    Args:
        df_metricas: DataFrame "long" devolvido por :func:`avaliar_todos`.

    Returns:
        DataFrame com colunas ``Modelo``, ``Wins`` (n produtos onde o modelo
        ganha), ``Win rate (%)``.
    """
    # Pivot: produto × modelo → RMSE
    pivot = df_metricas.pivot(index="produto_id", columns="modelo", values="rmse_euros")
    # Para cada linha, identificar o modelo com menor RMSE (excluindo NaN)
    vencedores = pivot.idxmin(axis=1, skipna=True)
    total = vencedores.notna().sum()

    rows = []
    for modelo in ORDEM_MODELOS:
        if modelo not in pivot.columns:
            continue
        wins = int((vencedores == modelo).sum())
        rows.append({
            "Modelo":       LABELS_MODELOS[modelo],
            "Wins":         wins,
            "Win rate (%)": round(100 * wins / total, 1) if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def plot_boxplot_rmse(df_metricas: pd.DataFrame, output_path: Path) -> None:
    """Boxplot do RMSE por modelo — mostra mediana e dispersão."""
    fig, ax = plt.subplots(figsize=(8, 5))

    dados, labels, cores = [], [], []
    for modelo in ORDEM_MODELOS:
        sub = df_metricas[df_metricas["modelo"] == modelo]["rmse_euros"].dropna()
        if not sub.empty:
            dados.append(sub.to_numpy())
            labels.append(LABELS_MODELOS[modelo])
            cores.append(CORES_MODELOS[modelo])

    bp = ax.boxplot(dados, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cores):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel("RMSE (€) — outliers ocultos")
    ax.set_title("Distribuição do RMSE por modelo (validação por produto)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PNG_DPI)
    plt.close(fig)


def plot_win_rate(df_win_rate: pd.DataFrame, output_path: Path) -> None:
    """Barras com win rate por modelo."""
    fig, ax = plt.subplots(figsize=(8, 4))
    # Mapear cores a partir dos labels
    labels_ordenados = df_win_rate["Modelo"].tolist()
    chaves = [k for k in ORDEM_MODELOS if LABELS_MODELOS[k] in labels_ordenados]
    cores = [CORES_MODELOS[k] for k in chaves]

    ax.bar(labels_ordenados, df_win_rate["Win rate (%)"], color=cores)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Percentagem de produtos onde cada modelo é o mais preciso (menor RMSE)")
    for i, v in enumerate(df_win_rate["Win rate (%)"]):
        ax.text(i, v + 0.5, f"{v}%", ha="center")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PNG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Geração do relatório markdown
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Conversão segura para markdown (fallback simples se tabulate ausente)."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        lines = ["| " + " | ".join(df.columns) + " |"]
        lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)


def escrever_relatorio_md(
    output_dir: Path,
    resumo: pd.DataFrame,
    win_rate: pd.DataFrame,
    df_metricas: pd.DataFrame,
) -> Path:
    """Escreve o relatório completo em markdown.

    Args:
        output_dir: Diretório de saída (criado se não existir).
        resumo: Tabela de :func:`calcular_resumo`.
        win_rate: Tabela de :func:`calcular_win_rate`.
        df_metricas: DataFrame completo (para análise top/worst por modelo).

    Returns:
        ``Path`` do ficheiro escrito.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    caminho = output_dir / "comparacao_modelos.md"

    secoes: list[str] = [
        "# Comparação de modelos de previsão de preços\n",
        "Avaliação de baselines (Naive, médias móveis, ARIMA), de um modelo de "
        "**gradient boosting** e do **LSTM**, na **mesma janela de validação** "
        "(últimos 7 dias por produto) e com as **mesmas métricas** em euros "
        "reais (RMSE/MAE/MAPE).\n",
        "\n## 1. Resumo global\n",
        _df_to_markdown(resumo),
        "\n![Distribuição do RMSE](boxplot_rmse_modelos.png)\n",
        "\n## 2. Win rate — % de produtos onde cada modelo ganha (menor RMSE)\n",
        _df_to_markdown(win_rate),
        "\n![Win rate](win_rate_modelos.png)\n",
        "\n## 3. Interpretação\n",
    ]

    # Texto interpretativo dinâmico baseado nos números
    melhor_rmse = resumo.loc[resumo["RMSE médio (€)"].idxmin()]
    melhor_winr = win_rate.loc[win_rate["Win rate (%)"].idxmax()]
    secoes.append(
        f"- O modelo com **menor RMSE médio** é **{melhor_rmse['Modelo']}** "
        f"({melhor_rmse['RMSE médio (€)']} €).\n"
        f"- O modelo com **maior win rate** é **{melhor_winr['Modelo']}** "
        f"({melhor_winr['Win rate (%)']}%).\n"
    )

    # Nota quando o melhor erro médio e o melhor win rate são modelos diferentes.
    if melhor_rmse["Modelo"] != melhor_winr["Modelo"]:
        secoes.append(
            f"\n> **Nota:** o **{melhor_rmse['Modelo']}** tem o menor erro "
            f"*médio*, mas o **{melhor_winr['Modelo']}** ganha em mais produtos. "
            f"Isto indica que o {melhor_rmse['Modelo']} é muito melhor nos "
            f"produtos difíceis (promoções, séries voláteis) — onde os outros "
            f"falham mais — enquanto o {melhor_winr['Modelo']} ganha por margem "
            f"pequena nas muitas séries quase constantes. Para a qualidade "
            f"global, o erro médio (e o seu desvio-padrão) é a métrica mais "
            f"relevante.\n"
        )

    # Discussão quando o gradient boosting lidera (resultado esperado com poucos
    # dados: árvores sobre features tabulares > rede recorrente sub-treinada).
    if "Gradient Boosting" in (melhor_rmse["Modelo"], melhor_winr["Modelo"]):
        secoes.append(
            "\nO **gradient boosting** liderar é consistente com a literatura de "
            "*forecasting* tabular de curto prazo: com poucas dezenas de dias por "
            "produto, um modelo de árvores sobre features (lags, médias móveis, "
            "dia-da-semana, cadeia, promoção) generaliza melhor que uma rede "
            "recorrente, treina em segundos e não precisa de normalização.\n"
        )

    # Discussão padrão (válida quando o naive vence — situação esperada com
    # séries curtas)
    if "Naive" in (melhor_rmse["Modelo"], melhor_winr["Modelo"]):
        secoes.append(
            "\nO **baseline naive vencer** modelos mais complexos é um resultado "
            "**esperado** em séries temporais curtas:\n"
            "- Os preços de supermercado mudam pouco dia-a-dia → '\"amanhã = hoje\"' "
            "é uma aproximação local muito boa.\n"
            "- O LSTM precisa de muitos dias para aprender padrões — com menos de "
            "60 dias de histórico fica sub-treinado.\n"
            "- O ARIMA tem dificuldade em convergir para séries muito constantes "
            "(NaN em parte dos produtos).\n"
            "\nÀ medida que o scraping acumulares mais dias, espera-se que o LSTM "
            "ultrapasse o naive — sobretudo em produtos com sazonalidade semanal "
            "(promoções).\n"
        )

    caminho.write_text("\n".join(secoes), encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI da comparação de modelos."""
    parser = argparse.ArgumentParser(
        description="Compara LSTM com modelos baseline (naive, MA, ARIMA)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
        help=f"CSV de forecasting (default: {DATASET_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Diretório de saída (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--janela", type=int, default=7,
        help="Dias de validação por produto (default: 7, igual ao LSTM).",
    )
    parser.add_argument(
        "--max-produtos", type=int,
        help="Limitar a este número de produtos (útil para testes rápidos).",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"[ERRO] Dataset não encontrado em {args.dataset}.")
        print("       Corre: python scripts/build_forecasting_dataset.py")
        sys.exit(1)

    print(f"[CARREGAR] {args.dataset}")
    df_dataset = pd.read_csv(args.dataset, parse_dates=["dia"])
    print(f"  {len(df_dataset):,} linhas, "
          f"{df_dataset['id_produto_loja'].nunique()} produtos.")

    # ARIMA emite avisos verbosos com séries curtas; silenciamos a nível global
    warnings.filterwarnings("ignore")

    print("\n[AVALIAR] A correr modelos...")
    df_metricas = avaliar_todos(
        df_dataset, janela=args.janela, max_produtos=args.max_produtos,
    )
    if df_metricas.empty:
        print("[ERRO] Sem métricas para reportar.")
        sys.exit(1)

    print("\n[RESUMO] A calcular tabelas...")
    resumo = calcular_resumo(df_metricas)
    win_rate = calcular_win_rate(df_metricas)

    print("\n=== Resumo global ===")
    print(resumo.to_string(index=False))
    print("\n=== Win rate ===")
    print(win_rate.to_string(index=False))

    print("\n[GRÁFICOS] A gerar PNGs...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_boxplot_rmse(df_metricas, args.output_dir / "boxplot_rmse_modelos.png")
    plot_win_rate(win_rate, args.output_dir / "win_rate_modelos.png")

    print("[MARKDOWN] A escrever relatório...")
    caminho = escrever_relatorio_md(args.output_dir, resumo, win_rate, df_metricas)

    print("\n[OK] Comparação concluída.")
    print(f"  Relatório : {caminho}")
    print(f"  Gráficos  : {args.output_dir}")


if __name__ == "__main__":
    main()
