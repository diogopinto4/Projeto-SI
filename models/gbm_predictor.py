"""
Modelo de previsão de preços por *gradient boosting* — alternativa ao LSTM.

Motivação
---------
A comparação experimental (``scripts/comparar_modelos.py`` →
``data/relatorio/comparacao_modelos.md``) mostrou que o LSTM fica **abaixo** de
baselines triviais (Naive, ARIMA) com o volume de dados atual: as séries de
preços têm poucas dezenas de dias e mudam pouco dia-a-dia, regime em que uma
rede recorrente fica sub-treinada. Modelos baseados em **árvores com boosting**
são o estado-da-arte para *forecasting* tabular de curto prazo com poucos dados:
treinam em segundos, não precisam de normalização e exploram diretamente as
features já construídas (lags, médias móveis, dia-da-semana, cadeia, promoção).

Decisões
--------
* **Implementação:** ``sklearn.ensemble.HistGradientBoostingRegressor`` —
  *gradient boosting* por histograma, equivalente em espírito ao LightGBM, mas
  já incluído no scikit-learn (sem dependência nova nem ``libomp`` no macOS).
* **Modelo global:** um único modelo treinado sobre todos os produtos (como o
  LSTM), partilhando padrões entre séries. O ``id``/``cadeia_id`` e as features
  temporais permitem-lhe distinguir produtos e cadeias.
* **Mesmas features que o LSTM** (:data:`FEATURE_COLS`) e mesma janela de
  validação (últimos ``janela`` dias por produto), previsão a 1 passo → a
  comparação é *apples-to-apples* (mesmas métricas, em euros).
* **Alvo parametrizado como rácio** ``preço_{t+1} / preço_t`` em vez do preço
  absoluto. Um GBM global a prever euros absolutos colapsa em produtos caros
  (erros enormes, previsões instáveis) porque otimiza o erro quadrático numa
  escala que vai de 0,20 € a 60 €+. O rácio é **invariante à escala** (≈ 1,0
  para todos os produtos) e reconstrói-se com ``pred = preço_t × rácio`` — o
  análogo "sem treino" ao ``MinMaxScaler`` por produto que o LSTM usa.

Uso::

    python models/gbm_predictor.py                      # avalia em todos os produtos
    python models/gbm_predictor.py --max-produtos 200   # subset rápido
    python models/gbm_predictor.py --importancia        # + importância das features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Raiz do projeto no path para imports limpos quando corrido como script.
_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ))

from models.baselines import calcular_metricas_em_euros

# ---------------------------------------------------------------------------
# Features e target
# ---------------------------------------------------------------------------

#: Features de input do modelo. Espelha ``FEATURE_COLS`` de
#: ``models/price_predictor.py`` (e ``scripts/build_forecasting_dataset.py``)
#: para que a comparação com o LSTM use exatamente o mesmo conjunto.
FEATURE_COLS: list[str] = [
    "preco",
    "promo_flag",
    "weekday",
    "weekofyear",
    "month",
    "dayofmonth",
    "cadeia_id",
    "dias_desde_ultima_obs",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "media_movel_3",
    "media_movel_7",
    "desvio_movel_7",
]

#: Coluna alvo — preço do dia seguinte (previsão a 1 passo), igual ao LSTM.
TARGET_COL = "target_t_plus_1"


# ---------------------------------------------------------------------------
# Construção do modelo
# ---------------------------------------------------------------------------

def criar_modelo(random_state: int = 42) -> HistGradientBoostingRegressor:
    """Cria um regressor de *gradient boosting* configurado para o problema.

    Os hiperparâmetros são conservadores (boa generalização com poucos dados):
    learning rate baixo, *early stopping* com validação interna, e regularização
    via número de folhas e mínimo de amostras por folha.

    Args:
        random_state: Semente para reprodutibilidade.

    Returns:
        Instância (não treinada) de ``HistGradientBoostingRegressor``.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Avaliação por produto (mesma metodologia que o LSTM)
# ---------------------------------------------------------------------------

def _preparar(df_dataset: pd.DataFrame, max_produtos: int | None) -> pd.DataFrame:
    """Limpa NaN das features/target e ordena por (produto, dia).

    Args:
        df_dataset: DataFrame do ``forecasting_dataset.csv``.
        max_produtos: Se fornecido, restringe a este número de produtos (útil
            para iterações rápidas — afeta treino e validação).

    Returns:
        DataFrame filtrado e ordenado, pronto para o split temporal.
    """
    df = df_dataset.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    df = df.sort_values(["id_produto_loja", "dia"])
    if max_produtos is not None:
        produtos = df["id_produto_loja"].unique()[:max_produtos]
        df = df[df["id_produto_loja"].isin(produtos)]
    return df


def avaliar_gbm_por_produto(
    df_dataset: pd.DataFrame,
    janela: int = 7,
    max_produtos: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Treina um GBM global e devolve métricas por produto na janela do LSTM.

    Split temporal idêntico ao ``avaliar_modelo`` do LSTM: para cada produto,
    os últimos ``janela`` dias (com target válido) são validação; tudo o resto
    (de todos os produtos) é treino de um único modelo global. O modelo prevê o
    rácio ``preço_{t+1}/preço_t``; reconstrói-se o preço e as métricas são em
    euros, sem inversão de escala.

    Apenas produtos com história suficiente (``>= janela*2 + 1`` observações)
    são avaliados — o mesmo critério dos baselines, para comparar o mesmo
    conjunto de produtos. Produtos mais curtos podem ainda contribuir para o
    treino do modelo global.

    Args:
        df_dataset: DataFrame do ``forecasting_dataset.csv`` (coluna ``dia``).
        janela: Dias de validação por produto. Default 7 — igual ao LSTM.
        max_produtos: Limite de produtos (para testes rápidos).
        random_state: Semente para reprodutibilidade.

    Returns:
        DataFrame "long": uma linha por produto com ``produto_id``, ``modelo``
        (``"gbm"``), ``n_amostras``, ``rmse_euros``, ``mae_euros``, ``mape_pct``.
        Vazio se não houver produtos com história suficiente.
    """
    df = _preparar(df_dataset, max_produtos)
    if df.empty:
        return pd.DataFrame(
            columns=["produto_id", "modelo", "n_amostras",
                     "rmse_euros", "mae_euros", "mape_pct"]
        )

    # Produtos com história suficiente (mesmo critério dos baselines).
    contagem = df.groupby("id_produto_loja").size()
    validos = set(contagem[contagem >= janela * 2 + 1].index)

    # Marcar as últimas `janela` linhas de cada produto como validação.
    # cumcount(ascending=False): 0 = última linha, 1 = penúltima, ...
    rank_desc = df.groupby("id_produto_loja").cumcount(ascending=False)
    is_val = (rank_desc < janela) & df["id_produto_loja"].isin(validos)

    treino = df[~is_val]
    val = df[is_val]
    if treino.empty or val.empty:
        return pd.DataFrame(
            columns=["produto_id", "modelo", "n_amostras",
                     "rmse_euros", "mae_euros", "mape_pct"]
        )

    # Alvo = rácio preço_{t+1} / preço_t (invariante à escala). Reconstrói-se o
    # preço com pred = preço_t × rácio. Ver docstring do módulo para a razão.
    modelo = criar_modelo(random_state=random_state)
    modelo.fit(treino[FEATURE_COLS], treino[TARGET_COL] / treino["preco"])

    val = val.copy()
    val["_pred"] = val["preco"].to_numpy() * modelo.predict(val[FEATURE_COLS])

    rows: list[dict] = []
    for produto_id, grupo in val.groupby("id_produto_loja"):
        metricas = calcular_metricas_em_euros(
            grupo[TARGET_COL].to_numpy(),
            grupo["_pred"].to_numpy(),
        )
        rows.append({
            "produto_id": int(produto_id),
            "modelo":     "gbm",
            "n_amostras": int(len(grupo)),
            **metricas,
        })

    return pd.DataFrame(rows)


def importancia_features(
    df_dataset: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Treina o GBM em todos os dados e devolve a importância por permutação.

    Útil para o relatório: mostra que features mais pesam na previsão (tipicamente
    ``preco`` e ``lag_1``, já que o preço muda pouco dia-a-dia).

    Args:
        df_dataset: DataFrame do ``forecasting_dataset.csv``.
        random_state: Semente para reprodutibilidade.

    Returns:
        DataFrame com colunas ``feature`` e ``importancia``, ordenado desc.
    """
    from sklearn.inspection import permutation_importance

    df = _preparar(df_dataset, max_produtos=None)
    modelo = criar_modelo(random_state=random_state)
    # Mesmo alvo (rácio) que avaliar_gbm_por_produto, para consistência.
    alvo = df[TARGET_COL] / df["preco"]
    modelo.fit(df[FEATURE_COLS], alvo)

    # Amostra para a permutação (a permutação é cara em datasets grandes).
    amostra = df.sample(min(5000, len(df)), random_state=random_state)
    resultado = permutation_importance(
        modelo, amostra[FEATURE_COLS], amostra[TARGET_COL] / amostra["preco"],
        n_repeats=5, random_state=random_state, scoring="neg_root_mean_squared_error",
    )
    return (
        pd.DataFrame({"feature": FEATURE_COLS, "importancia": resultado.importances_mean})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI — avalia o GBM no dataset de forecasting."""
    parser = argparse.ArgumentParser(
        description="Modelo de gradient boosting para previsão de preços (alternativa ao LSTM)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_RAIZ / "data" / "generated" / "forecasting_dataset.csv",
        help="CSV de forecasting (default: data/generated/forecasting_dataset.csv).",
    )
    parser.add_argument(
        "--janela", type=int, default=7,
        help="Dias de validação por produto (default: 7, igual ao LSTM).",
    )
    parser.add_argument(
        "--max-produtos", type=int,
        help="Limitar a este número de produtos (útil para testes rápidos).",
    )
    parser.add_argument(
        "--importancia", action="store_true",
        help="Calcular e mostrar a importância das features (permutação).",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"[ERRO] Dataset não encontrado em {args.dataset}.")
        print("       Corre: python scripts/build_forecasting_dataset.py")
        sys.exit(1)

    print(f"[CARREGAR] {args.dataset}")
    df = pd.read_csv(args.dataset, parse_dates=["dia"])
    print(f"  {len(df):,} linhas, {df['id_produto_loja'].nunique()} produtos.")

    print("\n[AVALIAR] A treinar GBM global e a avaliar por produto...")
    metricas = avaliar_gbm_por_produto(
        df, janela=args.janela, max_produtos=args.max_produtos,
    )
    if metricas.empty:
        print("[ERRO] Sem produtos com história suficiente para avaliar.")
        sys.exit(1)

    print(f"\n[RESULTADO] {len(metricas)} produtos avaliados.")
    print(f"  RMSE médio : {metricas['rmse_euros'].mean():.4f} €")
    print(f"  MAE médio  : {metricas['mae_euros'].mean():.4f} €")
    print(f"  MAPE médio : {metricas['mape_pct'].mean():.2f} %")

    if args.importancia:
        print("\n[IMPORTÂNCIA] A calcular importância das features (permutação)...")
        imp = importancia_features(df)
        print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
