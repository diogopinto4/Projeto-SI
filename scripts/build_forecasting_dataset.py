"""
Pipeline de construção do dataset de forecasting de preços.

Este script extrai o histórico de preços da base de dados PostgreSQL,
agrega-o ao nível diário por produto-loja, expande o painel para cobrir
todos os dias (incluindo dias sem observação via forward-fill), e calcula
as features de tempo e lag necessárias para treinar o LSTM.

Pipeline de transformações::

    BD → load_history()
       → aggregate_daily()      (média diária, flag de promoção)
       → expand_daily_panel()   (forward-fill para dias sem observação)
       → add_time_features()    (weekday, month, weekofyear, dayofmonth)
       → add_cadeia_id()        (codificação inteira da cadeia)
       → add_lag_features()     (lag 1/2/3/7 + médias móveis)
       → add_target()           (preço t+horizonte)
       → CSV

Uso::

    python scripts/build_forecasting_dataset.py --output data/generated/forecasting_dataset.csv
    python scripts/build_forecasting_dataset.py --horizonte 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import psycopg2

# Permite importar db_config sem instalar o projeto como package
sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Mapeamento de nome de cadeia para inteiro usado como feature pelo LSTM.
#: Ao adicionar uma nova cadeia, acrescentar aqui com o próximo inteiro disponível.
CADEIA_MAP: dict[str, int] = {
    "Continente": 0,
    "Pingo Doce": 1,
    "Auchan":     2,
}

#: Colunas estáticas (identidade do produto) — propagadas por ffill/bfill
#: para preencher dias interpolados no painel expandido.
STATIC_COLS: list[str] = [
    "id_produto_loja",
    "id_produto_mestre",
    "id_loja",
    "cadeia",
    "canal",
    "nome_produto",
    "marca",
    "categoria_geral",
    "quantidade_valor",
    "quantidade_unidade",
]

#: Colunas numéricas com forward-fill — o último preço observado é
#: transportado para os dias seguintes sem observação.
CARRY_COLS: list[str] = [
    "preco",
    "preco_original",
    "preco_unitario_valor",
]

#: Colunas de texto (não numéricas) — cast para dtype "string" do pandas
#: para suportar NA nativo sem confundir com float NaN.
TEXT_COLS: list[str] = [
    "cadeia",
    "canal",
    "nome_produto",
    "marca",
    "categoria_geral",
    "quantidade_unidade",
    "preco_unitario_unidade",
]

#: Query principal — junta histórico de preços com produto-loja, produto-mestre e loja.
#: Resultados ordenados para que o agrupamento subsequente seja eficiente.
_HISTORY_QUERY = """
    SELECT
        hp.id_historico,
        hp.data_recolha::date            AS dia,
        hp.id_produto_loja,
        pl.id_produto_mestre,
        l.id_loja,
        l.insignia                       AS cadeia,
        l.canal,
        pm.nome_padronizado              AS nome_produto,
        pm.marca,
        pm.categoria_geral,
        pm.quantidade_valor,
        pm.quantidade_unidade,
        hp.preco_unitario_valor,
        hp.preco_unitario_unidade,
        hp.preco_atual,
        hp.preco_original,
        hp.em_promocao
    FROM historico_precos hp
    JOIN produtos_loja pl  ON pl.id_produto_loja   = hp.id_produto_loja
    JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
    JOIN lojas l            ON l.id_loja             = pl.id_loja
    ORDER BY hp.id_produto_loja, hp.data_recolha::date;
"""


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

def load_history() -> pd.DataFrame:
    """Carrega o histórico completo de preços da base de dados.

    Executa :data:`_HISTORY_QUERY` e devolve o resultado como DataFrame.
    A coluna ``dia`` é convertida para ``datetime64``.

    Returns:
        DataFrame com uma linha por entrada em ``historico_precos``.

    Raises:
        ValueError: Se ``historico_precos`` estiver vazia.
        psycopg2.Error: Se a ligação à BD falhar.
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(_HISTORY_QUERY)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        raise ValueError("Não existem dados em historico_precos.")

    df["dia"] = pd.to_datetime(df["dia"])
    return df


# ---------------------------------------------------------------------------
# Transformações
# ---------------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega múltiplas entradas do mesmo produto-loja no mesmo dia.

    Quando o mesmo produto tem várias recolhas no mesmo dia (ex: scraping
    redundante), esta função colapsa-as numa única linha por (dia, produto-loja).

    Estratégia de agregação:
    - ``preco``: média dos preços no dia.
    - ``preco_original``: máximo (o valor mais alto representa o PVPR).
    - ``preco_unitario_valor``: média dos valores unitários.
    - ``preco_unitario_unidade``: moda (unidade mais frequente no dia).
    - ``promo_flag``: ``True`` se qualquer recolha indicou promoção.
    - ``observacoes_no_dia``: contagem de recolhas originais.

    .. note::
        ``preco_unitario_unidade`` é agregado separadamente do grupo
        principal para evitar duplicados quando a mesma unidade tem grafias
        ligeiramente diferentes (ex: "kg" vs "Kg").

    Args:
        df: DataFrame bruto de :func:`load_history`.

    Returns:
        DataFrame com uma linha por (dia, id_produto_loja).
    """
    group_cols = [
        "dia",
        "id_produto_loja",
        "id_produto_mestre",
        "id_loja",
        "cadeia",
        "canal",
        "nome_produto",
        "marca",
        "categoria_geral",
        "quantidade_valor",
        "quantidade_unidade",
    ]

    def _mode_or_none(series: pd.Series) -> str | None:
        """Devolve a moda da série, ou None se a série for toda NaN."""
        clean = series.dropna()
        if clean.empty:
            return None
        return clean.mode().iloc[0]

    daily = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            preco=("preco_atual", "mean"),
            preco_original=("preco_original", "max"),
            preco_unitario_valor=("preco_unitario_valor", "mean"),
            preco_unitario_unidade=("preco_unitario_unidade", _mode_or_none),
            promo_flag=("em_promocao", "max"),
            observacoes_no_dia=("id_historico", "count"),
        )
        .sort_values(["id_produto_loja", "dia"])
        .reset_index(drop=True)
    )

    daily["promo_flag"] = daily["promo_flag"].fillna(False).astype(bool)
    return daily


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Impõe os dtypes corretos a todas as colunas do DataFrame.

    Converte:
    - IDs inteiros para ``Int64`` (nullable integer).
    - Colunas numéricas para ``float64``.
    - Colunas de texto para ``string`` (pandas StringDtype, suporta NA).

    Args:
        df: DataFrame após :func:`aggregate_daily`.

    Returns:
        Cópia do DataFrame com dtypes corrigidos.
    """
    df = df.copy()

    for col in ["id_produto_loja", "id_produto_mestre", "id_loja"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["quantidade_valor", "preco", "preco_original", "preco_unitario_valor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in TEXT_COLS:
        df[col] = df[col].astype("string")

    return df


def expand_daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Expande o painel para cobrir todos os dias entre a primeira e última observação.

    Para cada produto-loja, preenche os dias em falta com ``date_range`` e
    propaga os valores mais recentes via forward-fill (``ffill``), simulando
    o princípio de que o preço se mantém igual até à próxima recolha.

    Colunas adicionadas:
    - ``foi_observado_no_dia`` (int 0/1): indica dias com recolha real.
    - ``dias_desde_ultima_obs`` (int): distância em dias à última recolha real.

    Args:
        daily: DataFrame de :func:`aggregate_daily`, um produto-loja por linha.

    Returns:
        DataFrame com painel contínuo (sem lacunas de dias) por produto-loja.
    """
    daily = enforce_dtypes(daily)
    expanded_parts = []

    for _, group in daily.groupby("id_produto_loja", sort=False):
        g = group.sort_values("dia").copy()

        # Gerar grelha de dias sem lacunas
        full_days = pd.date_range(g["dia"].min(), g["dia"].max(), freq="D")
        g = (
            g.set_index("dia")
            .reindex(full_days)
            .reset_index()
            .rename(columns={"index": "dia"})
        )

        # Marcar dias com e sem observação
        g["foi_observado_no_dia"] = g["observacoes_no_dia"].notna().astype(int)
        g["observacoes_no_dia"] = (
            pd.to_numeric(g["observacoes_no_dia"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # Propagar metadados estáticos para dias interpolados
        for col in STATIC_COLS:
            g[col] = g[col].ffill().bfill()

        # Propagar preços numéricos (forward-fill apenas — sem look-ahead)
        for col in CARRY_COLS:
            g[col] = pd.to_numeric(g[col], errors="coerce").ffill()

        # Propagar unidade textual separadamente
        g["preco_unitario_unidade"] = g["preco_unitario_unidade"].astype("string").ffill()

        g["promo_flag"] = g["promo_flag"].astype("boolean").fillna(False).astype(bool)

        # Calcular distância à última observação real
        ultima_obs = g["dia"].where(g["foi_observado_no_dia"] == 1).ffill()
        g["dias_desde_ultima_obs"] = (
            (g["dia"] - ultima_obs).dt.days.fillna(0).astype(int)
        )

        expanded_parts.append(g)

    expanded = pd.concat(expanded_parts, ignore_index=True)
    return expanded.sort_values(["id_produto_loja", "dia"]).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features calendáricas derivadas da coluna ``dia``.

    Features adicionadas:
    - ``weekday`` (0=segunda, 6=domingo)
    - ``month`` (1–12)
    - ``weekofyear`` (1–53, ISO 8601)
    - ``dayofmonth`` (1–31)

    Args:
        df: DataFrame com coluna ``dia`` (datetime64).

    Returns:
        Cópia do DataFrame com as quatro novas colunas.
    """
    df = df.copy()
    df["weekday"]    = df["dia"].dt.weekday
    df["month"]      = df["dia"].dt.month
    df["weekofyear"] = df["dia"].dt.isocalendar().week.astype(int)
    df["dayofmonth"] = df["dia"].dt.day
    return df


def add_cadeia_id(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica a cadeia de supermercado como inteiro para uso no LSTM.

    Usa :data:`CADEIA_MAP` para mapear o nome da cadeia para um ID numérico.
    Cadeias não reconhecidas ficam com ``cadeia_id = -1``.

    Args:
        df: DataFrame com coluna ``cadeia`` (string).

    Returns:
        Cópia do DataFrame com a coluna ``cadeia_id`` (int) adicionada.
    """
    df = df.copy()
    df["cadeia_id"] = df["cadeia"].map(CADEIA_MAP).fillna(-1).astype(int)

    desconhecidas = df.loc[df["cadeia_id"] == -1, "cadeia"].dropna().unique()
    if len(desconhecidas) > 0:
        warnings.warn(
            f"add_cadeia_id: cadeias não reconhecidas receberam cadeia_id=-1: "
            f"{sorted(desconhecidas)}. "
            "Adiciona-as a CADEIA_MAP em scripts/build_forecasting_dataset.py "
            "para que o LSTM as distinga corretamente.",
            stacklevel=2,
        )

    return df


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 7)) -> pd.DataFrame:
    """Calcula features de lag e médias móveis sobre o preço diário.

    Todas as features são calculadas com ``shift(1)`` mínimo para garantir
    que não há look-ahead (o modelo nunca vê o preço do próprio dia como input).

    Features adicionadas:
    - ``lag_N`` para cada N em ``lags``: preço de N dias atrás.
    - ``media_movel_3``: média dos 3 dias anteriores (min_periods=1).
    - ``media_movel_7``: média dos 7 dias anteriores (min_periods=1).
    - ``desvio_movel_7``: desvio padrão dos 7 dias anteriores (0.0 se < 2 pontos).

    Args:
        df: DataFrame com painel expandido e colunas ``preco`` e ``id_produto_loja``.
        lags: Tupla com os horizontes de lag a calcular. Por omissão ``(1, 2, 3, 7)``.

    Returns:
        Cópia do DataFrame com as novas colunas de lag e médias móveis.
    """
    df = df.copy()
    grouped = df.groupby("id_produto_loja", sort=False)

    for lag in lags:
        df[f"lag_{lag}"] = grouped["preco"].shift(lag)

    df["media_movel_3"] = grouped["preco"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["media_movel_7"] = grouped["preco"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    # min_periods=2: desvio com um único ponto é zero, não NaN
    df["desvio_movel_7"] = grouped["preco"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).std().fillna(0.0)
    )

    return df


def add_target(df: pd.DataFrame, horizonte: int = 1) -> pd.DataFrame:
    """Adiciona a variável alvo: preço ``horizonte`` dias à frente.

    A coluna resultante chama-se ``target_t_plus_{horizonte}``. As últimas
    ``horizonte`` linhas de cada produto ficam com NaN (sem alvo disponível)
    e devem ser excluídas antes do treino.

    Args:
        df: DataFrame com coluna ``preco`` e ``id_produto_loja``.
        horizonte: Número de dias à frente para o alvo. Por omissão 1.

    Returns:
        Cópia do DataFrame com a coluna ``target_t_plus_{horizonte}`` adicionada.
    """
    df = df.copy()
    df[f"target_t_plus_{horizonte}"] = (
        df.groupby("id_produto_loja", sort=False)["preco"].shift(-horizonte)
    )
    return df


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """Executa o pipeline completo e guarda o dataset em CSV."""
    parser = argparse.ArgumentParser(
        description="Constrói o dataset de forecasting a partir do histórico de preços na BD."
    )
    parser.add_argument(
        "--output",
        default="data/generated/forecasting_dataset.csv",
        help="Caminho do ficheiro CSV de saída (default: data/generated/forecasting_dataset.csv).",
    )
    parser.add_argument(
        "--horizonte",
        type=int,
        default=1,
        help="Horizonte da variável alvo em dias (default: 1).",
    )
    args = parser.parse_args()

    print("A carregar histórico de preços...")
    df = load_history()
    print(f"  {len(df)} entradas em historico_precos.")

    print("A agregar ao nível diário...")
    df = aggregate_daily(df)

    print("A expandir painel (forward-fill)...")
    df = expand_daily_panel(df)

    print("A calcular features de tempo, cadeia e lag...")
    df = add_time_features(df)
    df = add_cadeia_id(df)
    df = add_lag_features(df)

    print(f"A calcular variável alvo (horizonte={args.horizonte})...")
    df = add_target(df, horizonte=args.horizonte)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDataset guardado: {output_path}")
    print(f"  Linhas totais:          {len(df):>8,}")
    print(f"  Produtos distintos:     {df['id_produto_loja'].nunique():>8,}")
    print(f"  Dias distintos:         {df['dia'].nunique():>8,}")
    print(f"  Dias c/ observação:     {int(df['foi_observado_no_dia'].sum()):>8,}")
    print(f"  Cadeias:                {df.groupby('cadeia_id')['cadeia'].first().to_dict()}")


if __name__ == "__main__":
    main()
