"""
Sistema de Recomendação de Compras de Supermercado.

Fornece quatro funcionalidades de alto nível:

1. :func:`pesquisar_produtos` — encontrar produtos na BD por palavra-chave.
2. :func:`recomendar_melhor_loja` — comparar preços atuais entre lojas.
3. :func:`otimizar_lista_compras` — minimizar o custo de uma lista de compras.
4. :func:`recomendar_momento_compra` — prever se vale a pena esperar por um
   preço mais baixo, usando Monte Carlo Dropout sobre o LSTM.

Uso::

    python models/recommender.py --pesquisar "arroz"
    python models/recommender.py --melhor-loja "arroz agulha"
    python models/recommender.py --lista "arroz agulha,azeite virgem extra,atum natural"
    python models/recommender.py --momento --produto-id 1
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
import warnings
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from db_config import DB_CONFIG

sys.path.insert(0, str(Path(__file__).parent))
from price_predictor import prever_preco_com_incerteza
from geolocation import (
    custo_deslocacao_euros,
    distancia_minima_por_insignia,
    haversine_km,
    resolver_custo_km,
)


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def get_connection() -> psycopg2.extensions.connection:
    """Abre uma nova ligação à base de dados.

    Returns:
        Ligação psycopg2 pronta a usar.

    Raises:
        psycopg2.OperationalError: Se a ligação falhar.
    """
    return psycopg2.connect(**DB_CONFIG)


def _query_to_df(
    cur: psycopg2.extensions.cursor,
    query: str,
    params: tuple = (),
) -> pd.DataFrame:
    """Executa uma query e devolve o resultado como DataFrame.

    Converte automaticamente colunas do tipo ``Decimal`` para ``float``
    (psycopg2 mapeia ``NUMERIC`` para ``Decimal`` por omissão).

    Args:
        cur: Cursor psycopg2 aberto.
        query: SQL a executar.
        params: Parâmetros posicionais.

    Returns:
        DataFrame com os resultados da query.
    """
    cur.execute(query, params)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)

    for col in df.columns:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(sample, Decimal):
            df[col] = df[col].apply(lambda x: float(x) if x is not None else None)

    return df


#: Regex que reconhece tokens de quantidade isolados na query (ex: "1.5L",
#: "750ml", "1Kg", "120g", "6un"). Devolve grupo 1 = valor, grupo 2 = unidade.
_QTY_TOKEN_RE = re.compile(
    r"^(\d+(?:[.,]\d+)?)\s*(ml|cl|dl|l|g|gr|kg|un)$",
    re.IGNORECASE,
)

#: Conversão de unidades para a forma canónica usada na BD (ml para volumes,
#: g para massa, un para contagem). Mantém alinhamento com
#: :func:`scripts.ingest.converter_para_unidade_canonica`.
_UNIT_CANONICAL: dict[str, tuple[str, int]] = {
    "ml": ("ml", 1),
    "cl": ("ml", 10),
    "dl": ("ml", 100),
    "l":  ("ml", 1000),
    "g":  ("g",  1),
    "gr": ("g",  1),
    "kg": ("g",  1000),
    "un": ("un", 1),
}


def _extrair_filtros_quantidade(termo: str) -> tuple[str, list[tuple[int, str]]]:
    """Separa tokens de quantidade do termo de pesquisa textual.

    Reconhece padrões como ``"1.5L"``, ``"750ml"``, ``"1Kg"`` e converte-os
    para a unidade canónica (ml/g/un). Isto permite que pesquisas como
    ``"coca-cola 1.5L"`` sejam interpretadas como ``"coca-cola" + quantidade
    1500 ml`` em vez de procurar a string literal ``"1.5L"`` no nome do produto
    (que falha porque a BD guarda quantidades como ``1500ml``).

    Args:
        termo: Texto de pesquisa em formato livre.

    Returns:
        Tupla ``(termo_limpo, filtros_quantidade)`` onde ``filtros_quantidade``
        é uma lista de ``(valor_canonico, unidade_canonica)`` a aplicar como
        filtro adicional na BD.
    """
    palavras: list[str] = []
    qty_filters: list[tuple[int, str]] = []
    for p in termo.split():
        m = _QTY_TOKEN_RE.match(p.strip())
        if m:
            valor = float(m.group(1).replace(",", "."))
            unit_raw = m.group(2).lower()
            unit_canonical, factor = _UNIT_CANONICAL[unit_raw]
            qty_filters.append((int(valor * factor), unit_canonical))
        else:
            palavras.append(p)
    return " ".join(palavras), qty_filters


#: Stop-words portuguesas a ignorar ao contar tokens significativos no nome
#: dos produtos. Reduz a inflação artificial do número de tokens em nomes
#: longos com partículas gramaticais que não acrescentam significado
#: (ex: "Bolachas De Chocolate Com Recheio De Creme" → 4 tokens significativos
#: em vez de 7). Mantida pequena e conservadora.
_STOP_WORDS_PT: frozenset[str] = frozenset({
    "de", "do", "da", "dos", "das", "com", "sem", "para", "em",
    "no", "na", "nos", "nas", "e", "ou", "a", "o", "as", "os",
    "ao", "aos", "à", "às",
})


def _tokens_significativos(texto: str) -> list[str]:
    """Devolve a lista de tokens não-stop-word do texto, em minúsculas."""
    return [
        t for t in str(texto).lower().split()
        if t.strip() and t.strip() not in _STOP_WORDS_PT
    ]


def _filtra_ruido_por_tamanho(
    df: pd.DataFrame,
    termo: str,
    col_nome: str = "nome_padronizado",
    max_ratio: float = 4.0,
) -> pd.DataFrame:
    """Remove resultados onde o nome do produto excede ``max_ratio`` vezes a
    contagem de tokens significativos da query.

    Caso típico: pesquisa ``"uva passa"`` (2 tokens significativos) traz
    ``"Gelado Auchan Rum Com Passas De Uva 1000ml"`` (6 tokens significativos
    depois de ignorar "Com" e "De"). Ratio 6/2 = 3.0 ≤ 4.0 → ainda passa
    (depois é a similaridade pg_trgm que decide), mas
    ``"Massa Esparguete Com Atum Em Azeite De Soja De Marca X Quantidade Y"``
    com 10+ tokens é filtrada.

    Args:
        df: DataFrame com resultados de pesquisa.
        termo: Termo original (para contar tokens significativos).
        col_nome: Nome da coluna que contém o nome do produto.
        max_ratio: Rácio máximo aceitável entre tokens significativos do
            nome e da query (default ``4.0`` — equilibra precisão/recall).

    Returns:
        DataFrame filtrado (índice reset).
    """
    if df.empty:
        return df
    n_query = max(1, len(_tokens_significativos(termo)))

    def _ratio_ok(nome) -> bool:
        if nome is None:
            return False
        n_result = max(1, len(_tokens_significativos(nome)))
        return (n_result / n_query) <= max_ratio

    mask = df[col_nome].apply(_ratio_ok)
    return df[mask].reset_index(drop=True)


def _condicoes_multi_ilike(campo: str, nome: str) -> tuple[str, list]:
    """Constrói condição SQL AND-ILIKE para pesquisa precisa.

    Todas as palavras do ``nome`` têm de aparecer no ``campo``.
    Palavras com menos de 2 caracteres são ignoradas. Tokens reconhecidos
    como quantidade (ex: ``"1.5L"``) são convertidos para filtros em
    ``pm.quantidade_valor`` + ``pm.quantidade_unidade`` (unidade canónica
    ml/g/un) em vez de ILIKE no nome.

    Args:
        campo: Nome da coluna SQL (ex: ``"pm.nome_padronizado"``).
        nome: Texto de pesquisa com uma ou mais palavras.

    Returns:
        Tupla ``(sql_snippet, params_list)`` pronta a interpolar na query.
    """
    nome_limpo, qty_filters = _extrair_filtros_quantidade(nome)
    palavras = [p.strip() for p in nome_limpo.split() if len(p.strip()) >= 2]
    if not palavras and not qty_filters:
        return f"unaccent({campo}) ILIKE unaccent(%s)", [f"%{nome}%"]

    parts: list[str] = []
    params: list = []
    if palavras:
        parts.extend(f"unaccent({campo}) ILIKE unaccent(%s)" for _ in palavras)
        params.extend(f"%{p}%" for p in palavras)
    for valor, unidade in qty_filters:
        parts.append("(pm.quantidade_valor = %s AND pm.quantidade_unidade = %s)")
        params.extend([valor, unidade])
    return " AND ".join(parts), params


def _condicoes_multi_ou(campo: str, nome: str) -> tuple[str, list]:
    """Constrói condição SQL OR-ILIKE para pesquisa ampla (fallback).

    Pelo menos uma palavra do ``nome`` tem de aparecer no ``campo``.
    Usado quando a pesquisa AND não retorna resultados. Filtros de
    quantidade (se presentes) continuam aplicados em AND para não dispersar
    o resultado.

    Args:
        campo: Nome da coluna SQL.
        nome: Texto de pesquisa.

    Returns:
        Tupla ``(sql_snippet, params_list)`` pronta a interpolar na query.
    """
    nome_limpo, qty_filters = _extrair_filtros_quantidade(nome)
    palavras = [p.strip() for p in nome_limpo.split() if len(p.strip()) >= 2]
    if not palavras and not qty_filters:
        return f"unaccent({campo}) ILIKE unaccent(%s)", [f"%{nome}%"]

    params: list = []
    blocks: list[str] = []
    if palavras:
        text_or = " OR ".join(f"unaccent({campo}) ILIKE unaccent(%s)" for _ in palavras)
        blocks.append(f"({text_or})")
        params.extend(f"%{p}%" for p in palavras)
    for valor, unidade in qty_filters:
        blocks.append("(pm.quantidade_valor = %s AND pm.quantidade_unidade = %s)")
        params.extend([valor, unidade])
    return " AND ".join(blocks), params


def _lojas_candidatas_para_lista(
    cur: psycopg2.extensions.cursor,
    lista: list[str],
) -> list[str]:
    """Devolve todas as lojas que têm pelo menos um item da lista.

    Não basta olhar para as lojas vencedoras por item: uma loja pode não ser a
    mais barata em nenhum produto isolado e ainda assim ser a melhor opção para
    comprar a lista completa.
    """
    lojas: set[str] = set()
    for nome in lista:
        cond_sql, cond_params = _condicoes_multi_ilike("pm.nome_padronizado", nome)
        query = f"""
            SELECT DISTINCT l.insignia
            FROM precos_atuais pa
            JOIN produtos_loja   pl ON pl.id_produto_loja   = pa.id_produto_loja
            JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
            JOIN lojas            l ON l.id_loja            = pl.id_loja
            WHERE {cond_sql}
        """
        cur.execute(query, tuple(cond_params))
        lojas.update(row[0] for row in cur.fetchall())
    return sorted(lojas)


def _executar_pesquisa(
    cur: psycopg2.extensions.cursor,
    cond_sql: str,
    cond_params: list,
    limite: int,
    min_similarity: float | None = None,
    termo: str | None = None,
) -> pd.DataFrame:
    """Executa a query de pesquisa de produtos com as condições fornecidas.

    Quando ``termo`` é fornecido, os resultados são ordenados por similaridade
    trigrama descendente (pg_trgm), garantindo que "Azeite Virgem Extra Gallo"
    aparece antes de "Atum em Azeite Virgem Extra" numa pesquisa por
    "azeite virgem extra".

    Args:
        cur: Cursor psycopg2 aberto.
        cond_sql: Fragmento SQL da cláusula WHERE (sem ``WHERE``).
        cond_params: Parâmetros correspondentes a ``cond_sql``.
        limite: Número máximo de resultados.
        min_similarity: Se fornecido, filtra por ``similarity(nome, termo) > min_similarity``
            (requer pg_trgm). Usado no fallback OR para evitar falsos positivos.
        termo: Texto original de pesquisa. Activa ordenação por similaridade
            e, opcionalmente, filtragem por ``min_similarity``.

    Returns:
        DataFrame com colunas: ``nome_padronizado``, ``marca``,
        ``quantidade_valor``, ``quantidade_unidade``, ``num_lojas``,
        ``preco_min``, ``preco_max``.
    """
    sim_filter = ""
    sim_params: list = []
    if min_similarity is not None and termo:
        sim_filter = "AND similarity(pm.nome_padronizado, %s) > %s"
        sim_params = [termo, min_similarity]

    # Ordenar por similaridade quando o termo é conhecido, alfabético caso contrário
    if termo:
        order_clause = "ORDER BY similarity(pm.nome_padronizado, %s) DESC, pm.nome_padronizado"
        order_params: list = [termo]
    else:
        order_clause = "ORDER BY pm.nome_padronizado"
        order_params = []

    query = f"""
        SELECT
            pm.nome_padronizado,
            MAX(pm.marca)              AS marca,
            MAX(pm.quantidade_valor)   AS quantidade_valor,
            MAX(pm.quantidade_unidade) AS quantidade_unidade,
            COUNT(DISTINCT l.insignia) AS num_lojas,
            MIN(pa.preco_atual)::float AS preco_min,
            MAX(pa.preco_atual)::float AS preco_max
        FROM produtos_mestre pm
        JOIN produtos_loja   pl ON pl.id_produto_mestre = pm.id_produto_mestre
        JOIN lojas            l  ON l.id_loja            = pl.id_loja
        LEFT JOIN precos_atuais pa ON pa.id_produto_loja = pl.id_produto_loja
        WHERE {cond_sql} {sim_filter}
        GROUP BY pm.nome_padronizado
        {order_clause}
        LIMIT %s
    """
    return _query_to_df(cur, query, tuple(cond_params + sim_params + order_params + [limite]))


# ---------------------------------------------------------------------------
# 1. Pesquisa de produtos
# ---------------------------------------------------------------------------

def pesquisar_produtos(termo: str, limite: int = 20) -> pd.DataFrame | None:
    """Lista produtos na BD que correspondem ao termo de pesquisa.

    Tenta primeiro uma pesquisa AND (todas as palavras presentes). Se não
    encontrar resultados, tenta uma pesquisa OR (pelo menos uma palavra).
    Aplica filtro de ruído por dimensão do nome para evitar matches em
    produtos onde os tokens da query aparecem mas representam uma fração
    pequena (ex: "uva passa" → "Gelado Rum Com Passas De Uva").

    Args:
        termo: Texto de pesquisa livre (ex: ``"arroz agulha"``).
        limite: Número máximo de resultados a devolver (default: 20).

    Returns:
        DataFrame com os produtos encontrados, ou ``None`` se não houver
        qualquer resultado. Excepções na BD são apanhadas e logadas — devolve
        ``None`` em vez de propagar (o agente apresenta "nenhum produto" em
        vez de 500).
    """
    # Defesa F1 — try/except cobre TODA a função.
    # Pesquisas comuns (ex: "arroz") estavam a propagar excepções não-tratadas
    # como 500 ao dashboard. Causas possíveis: filtragem com NaN, conversão
    # de Decimal, similarity() com inputs degenerados, ou `df.to_string()` em
    # dataframes com tipos mistos. Apanhamos qualquer falha aqui e devolvemos
    # None, deixando o utilizador ver "Nenhum produto encontrado" em vez de
    # um erro técnico.
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cond_sql, cond_params = _condicoes_multi_ilike("pm.nome_padronizado", termo)
                # Pedimos mais do que o limite final para podermos descartar ruído
                # via :func:`_filtra_ruido_por_tamanho` sem ficar abaixo do limite
                # pedido pelo utilizador.
                df = _executar_pesquisa(cur, cond_sql, cond_params, limite * 3, termo=termo)

                if df.empty:
                    print("[PESQUISA] AND sem resultados. A tentar pesquisa mais ampla...")
                    cond_sql, cond_params = _condicoes_multi_ou("pm.nome_padronizado", termo)
                    # Threshold de similaridade 0.20 (alinhado com recomendar_melhor_loja).
                    # 0.08 era demasiado permissivo — pesquisas como "produto inexistente"
                    # ou "bolachas oreo" caíam em produtos sem qualquer relação semântica
                    # apenas porque partilhavam um trigrama solto.
                    df = _executar_pesquisa(
                        cur, cond_sql, cond_params, limite * 3,
                        min_similarity=0.20, termo=termo,
                    )

        df = _filtra_ruido_por_tamanho(df, termo).head(limite)

        if df.empty:
            print(f"[PESQUISA] Nenhum produto encontrado para '{termo}'.")
            return None

        print(f"\n[PESQUISA] Produtos para '{termo}' ({len(df)} resultado(s)):")
        # ``to_string`` pode falhar em dataframes com tipos exóticos (numpy NaN
        # em colunas typed como Decimal, etc.). Protegemos para não derrubar
        # toda a função se for só um problema de display.
        try:
            print(df.to_string(index=False))
        except Exception as exc_print:
            print(f"[PESQUISA] (Aviso: não foi possível imprimir a tabela — {exc_print})")
        return df

    except Exception as exc:
        print(f"[PESQUISA][ERRO] Excepção ao processar '{termo}': {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 2. Recomendar melhor loja para um produto
# ---------------------------------------------------------------------------

def recomendar_melhor_loja(nome_produto: str, top_n: int = 5) -> pd.DataFrame | None:
    """Devolve os ``top_n`` preços mais baratos para um produto, por loja.

    Usa ``DISTINCT ON (nome_padronizado, loja)`` para não repetir o mesmo
    produto quando há múltiplas entradas na BD com nomes ligeiramente diferentes.
    Calcula a poupança percentual face ao preço mais caro encontrado.

    Args:
        nome_produto: Termo de pesquisa para o produto (ex: ``"arroz agulha"``).
        top_n: Número máximo de resultados a mostrar (default: 5).

    Returns:
        DataFrame com os resultados ordenados por preço ascendente, ou
        ``None`` se não houver qualquer produto correspondente. Excepções
        na BD são apanhadas (defesa F1) e devolvem ``None``.
    """
    # Defesa F1 — toda a função coberta por try/except.
    try:
        cond_sql, cond_params = _condicoes_multi_ilike("pm.nome_padronizado", nome_produto)

        query = f"""
            SELECT DISTINCT ON (pm.nome_padronizado, l.insignia)
                pl.id_produto_loja,
                pm.nome_padronizado,
                pm.quantidade_valor,
                pm.quantidade_unidade,
                pm.marca,
                l.insignia                       AS loja,
                pa.preco_atual::float            AS preco_atual,
                pa.em_promocao,
                pa.data_recolha::date            AS ultima_atualizacao,
                pa.preco_unitario_valor::float   AS preco_unitario_valor,
                pa.preco_unitario_unidade
            FROM precos_atuais pa
            JOIN produtos_loja   pl ON pl.id_produto_loja   = pa.id_produto_loja
            JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
            JOIN lojas            l  ON l.id_loja            = pl.id_loja
            WHERE {cond_sql}
              AND similarity(pm.nome_padronizado, %s) > 0.20
            ORDER BY pm.nome_padronizado, l.insignia, pa.preco_atual ASC
        """

        with get_connection() as conn:
            with conn.cursor() as cur:
                df = _query_to_df(cur, query, tuple(cond_params + [nome_produto]))

        if df.empty:
            print(f"[RECOMENDACAO] Nenhum produto encontrado para '{nome_produto}'.")
            print(f"  -> Corre: python models/recommender.py --pesquisar \"{nome_produto.split()[0]}\"")
            return None

        # Filtro F7: remover resultados onde o nome do produto tem muito mais
        # tokens significativos que a query — apanha falsos positivos como
        # "uva passa" → "Gelado Rum Com Passas De Uva" onde a similaridade
        # trigrama é alta mas o produto é semanticamente outro.
        df = _filtra_ruido_por_tamanho(df, nome_produto)
        if df.empty:
            print(f"[RECOMENDACAO] Resultados filtrados por ruído para '{nome_produto}'.")
            return None

        # Decidir critério de ordenação (preço unitário vs absoluto):
        #
        # F5 — Só usar preço unitário se todos os candidatos partilharem a
        # mesma unidade (€/kg ou €/l ou €/un). Caso contrário, comparar €/l
        # com €/kg é dimensionalmente inválido e produz "Desconto vs mais caro"
        # absurdo (ex: gelado a 2.79€/l vs uva passa a 6.45€/kg dá 56.7%
        # "desconto" para o gelado mesmo sendo um produto diferente).
        tem_preco_unitario = df["preco_unitario_valor"].notna().sum() >= max(1, len(df) // 2)
        unidades_unicas = df["preco_unitario_unidade"].dropna().unique()
        unidades_coerentes = len(unidades_unicas) <= 1
        usar_unitario = tem_preco_unitario and unidades_coerentes
        sort_col = "preco_unitario_valor" if usar_unitario else "preco_atual"
        df = df.sort_values([sort_col, "preco_atual"], na_position="last").head(top_n).copy()

        ref_max = df[sort_col].max()
        if ref_max and ref_max > 0:
            df["poupanca_pct"] = ((ref_max - df[sort_col]) / ref_max * 100).round(1)
        else:
            df["poupanca_pct"] = 0.0

        print(f"\n[RECOMENDACAO] Melhores precos para '{nome_produto}':")
        cols = [
            "nome_padronizado", "loja", "preco_atual", "em_promocao",
            "poupanca_pct", "preco_unitario_valor", "preco_unitario_unidade",
            "ultima_atualizacao",
        ]
        try:
            print(df[cols].to_string(index=False))
        except Exception as exc_print:
            print(f"[RECOMENDACAO] (Aviso: não foi possível imprimir tabela — {exc_print})")
        return df

    except Exception as exc:
        print(f"[RECOMENDACAO][ERRO] Excepção ao processar '{nome_produto}': {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 3. Otimizar lista de compras
# ---------------------------------------------------------------------------

def _melhor_preco_para_item(
    cur: psycopg2.extensions.cursor,
    nome: str,
    loja: str | None = None,
) -> pd.DataFrame:
    """Encontra o produto com maior semelhança ao ``nome`` e menor preço.

    Usa uma subquery para colapsar cada nome de produto ao seu menor preço
    (``DISTINCT ON``), e depois ordena pelo score de similaridade calculado
    pelo pg_trgm para garantir que "azeite virgem extra" retorna um azeite
    e não "atum em azeite" (que é mais barato mas irrelevante).

    Args:
        cur: Cursor psycopg2 aberto.
        nome: Termo de pesquisa para o item.
        loja: Se fornecido, filtra por esta cadeia de supermercado.

    Returns:
        DataFrame com 0 ou 1 linha contendo: ``nome_padronizado``, ``loja``,
        ``preco_atual``, ``em_promocao``, ``id_produto_loja``.
    """
    cond_sql, cond_params = _condicoes_multi_ilike("pm.nome_padronizado", nome)
    loja_sql = "AND l.insignia = %s" if loja else ""
    loja_param = [loja] if loja else []

    # Subquery: colapsar cada produto ao menor preço (DISTINCT ON nome)
    # Outer query: filtrar por similaridade mínima (evita falsos positivos do tipo
    # "leite amêndoa" → "Chocolate Leite Regina Amêndoas 20g", onde os tokens
    # aparecem mas o produto é fundamentalmente outro) e ordenar pelo score.
    # 0.20 alinhado com :func:`recomendar_melhor_loja`.
    query = f"""
        SELECT nome_padronizado, loja, preco_atual, em_promocao, id_produto_loja
        FROM (
            SELECT DISTINCT ON (pm.nome_padronizado)
                pm.nome_padronizado,
                l.insignia              AS loja,
                pa.preco_atual::float   AS preco_atual,
                pa.em_promocao,
                pl.id_produto_loja,
                similarity(pm.nome_padronizado, %s) AS sim_score
            FROM precos_atuais pa
            JOIN produtos_loja   pl ON pl.id_produto_loja   = pa.id_produto_loja
            JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
            JOIN lojas            l  ON l.id_loja            = pl.id_loja
            WHERE {cond_sql} {loja_sql}
            ORDER BY pm.nome_padronizado, pa.preco_atual ASC
        ) sub
        WHERE sub.sim_score >= 0.20
        ORDER BY sim_score DESC, preco_atual ASC
        LIMIT 1
    """
    # nome como primeiro param (para similarity()), depois cond_params e loja_param
    return _query_to_df(cur, query, tuple([nome] + cond_params + loja_param))


def otimizar_lista_compras(lista: list[str]) -> dict | None:
    """Minimiza o custo de uma lista de compras distribuída entre lojas.

    Para cada item da lista encontra o preço mais baixo (qualquer loja) e
    calcula também o custo se todos os itens forem comprados na mesma loja.
    Nota: não considera custo de deslocação entre lojas.

    Args:
        lista: Lista de termos de pesquisa (ex: ``["arroz agulha", "azeite"]``).

    Returns:
        Dicionário com:
        - ``"itens"``: lista de registos dos produtos encontrados.
        - ``"nao_encontrados"``: lista de termos da lista para os quais nenhum
          produto foi encontrado na BD (útil para avisar o utilizador na UI).
        - ``"custo_dividido"``: soma do preço mais baixo de cada item em
          **qualquer loja**. Estratégia "compra dividida" (ida a várias
          cadeias). Pode ser superior ao custo na melhor loja única.
        - ``"custo_melhor_loja"``: custo total **na melhor loja única** que
          tenha lista completa. ``None`` se nenhuma loja cobre todos os itens.
        - ``"melhor_loja"``: nome dessa loja única (``None`` se inaplicável).
        - ``"custo_minimo"``: ``min(custo_dividido, custo_melhor_loja)`` — a
          melhor opção entre as duas estratégias.
        - ``"melhor_estrategia"``: ``"loja_unica"`` ou ``"dividir"`` (ou
          ``"dividir_parcial"`` quando nenhuma loja tem lista completa).
        - ``"poupanca_split"``: poupança da divisão face à loja única.
          Positiva quando dividir compensa, 0 quando loja única é igual ou
          melhor, ``None`` quando não há loja única para comparar.
        - ``"detalhe_por_loja"``: custo e itens em falta por loja.
        Ou ``None`` se nenhum item for encontrado.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:

            resultados: list[dict] = []
            nao_encontrados: list[str] = []

            for nome in lista:
                df = _melhor_preco_para_item(cur, nome)
                if df.empty:
                    nao_encontrados.append(nome)
                    print(f"  [AVISO] '{nome}' nao encontrado.")
                    print(f"          -> Corre --pesquisar \"{nome.split()[0]}\" para ver alternativas.")
                else:
                    row = df.iloc[0].to_dict()
                    row["item_pesquisado"] = nome
                    resultados.append(row)

            if not resultados:
                print("[LISTA] Nenhum item encontrado. Usa --pesquisar para explorar a BD.")
                return None

            df_lista = pd.DataFrame(resultados)
            # custo_dividido = soma do menor preço de cada item em qualquer
            # loja. É o custo da estratégia "compra dividida" (ida a várias
            # cadeias). Pode ser MAIS caro que comprar tudo na melhor loja
            # única (caso típico: a "melhor loja única" tem alternativas
            # baratas para todos os itens, mesmo que os globalmente mais
            # baratos estejam noutras lojas).
            custo_dividido = round(float(df_lista["preco_atual"].sum()), 2)

            lojas_unicas = _lojas_candidatas_para_lista(cur, lista)
            if not lojas_unicas:
                lojas_unicas = sorted(df_lista["loja"].unique())
            custo_por_loja: dict[str, dict] = {}

            for loja in lojas_unicas:
                total = 0.0
                em_falta: list[str] = []
                itens_loja: list[dict] = []   # breakdown por loja para UI
                for nome in lista:
                    df_l = _melhor_preco_para_item(cur, nome, loja=loja)
                    if df_l.empty:
                        em_falta.append(nome)
                    else:
                        row = df_l.iloc[0].to_dict()
                        row["item_pesquisado"] = nome
                        total += float(row["preco_atual"])
                        itens_loja.append(row)
                custo_por_loja[loja] = {
                    "total":    round(total, 2),
                    "em_falta": em_falta,
                    "itens":    itens_loja,
                }

    lojas_completas = {l: v for l, v in custo_por_loja.items() if not v["em_falta"]}
    if lojas_completas:
        melhor_loja = min(lojas_completas, key=lambda l: lojas_completas[l]["total"])
        custo_melhor_loja = float(custo_por_loja[melhor_loja]["total"])
        # custo_minimo = melhor entre as duas estratégias (sem clamp escondido).
        custo_minimo = min(custo_dividido, custo_melhor_loja)
        if custo_dividido < custo_melhor_loja:
            melhor_estrategia = "dividir"
            poupanca_split = round(custo_melhor_loja - custo_dividido, 2)
        else:
            melhor_estrategia = "loja_unica"
            poupanca_split = 0.0
    else:
        melhor_loja = None
        custo_melhor_loja = None
        custo_minimo = custo_dividido
        melhor_estrategia = "dividir_parcial"
        poupanca_split = None

    print("\n[LISTA DE COMPRAS] Otimizacao de custo:")
    print(f"  Itens encontrados  : {len(resultados)}/{len(lista)}")
    if nao_encontrados:
        print(f"  Nao encontrados    : {', '.join(nao_encontrados)}")
    print(f"\n  Custo dividido (multi-loja) : {custo_dividido:.2f} EUR")
    if melhor_loja is None:
        print("  Melhor loja unica : indisponivel (nenhuma loja cobre a lista completa)")
    else:
        print(f"  Melhor loja unica ({melhor_loja:12s}) : {custo_melhor_loja:.2f} EUR")

    if melhor_estrategia == "dividir_parcial":
        print("  -> Recomendado: dividir por lojas (nenhuma cobre a lista completa).")
    elif melhor_estrategia == "dividir":
        print(f"  -> Recomendado DIVIDIR — poupas {poupanca_split:.2f} EUR vs loja única.")
    else:
        diff_loja = custo_melhor_loja - custo_dividido if custo_melhor_loja else 0
        if diff_loja < -0.01:
            print(f"  -> Loja única ({melhor_loja}) é melhor por {-diff_loja:.2f} EUR.")
        else:
            print(f"  -> Indiferente — qualquer estratégia dá ~{custo_minimo:.2f} EUR.")

    print("\n  Detalhe por item (preco mais baixo em qualquer loja):")
    print(df_lista[["item_pesquisado", "nome_padronizado", "loja",
                     "preco_atual", "em_promocao"]].to_string(index=False))

    print("\n  Custo total por loja:")
    for loja, info in sorted(custo_por_loja.items(), key=lambda x: x[1]["total"]):
        falta = f"  [sem: {', '.join(info['em_falta'])}]" if info["em_falta"] else ""
        print(f"    {loja:15s}: {info['total']:.2f} EUR{falta}")

    return {
        "itens": df_lista.to_dict(orient="records"),
        "nao_encontrados": nao_encontrados,
        "custo_minimo": round(custo_minimo, 2),
        "custo_dividido": custo_dividido,
        "melhor_loja": melhor_loja,
        "custo_melhor_loja": custo_melhor_loja,
        "melhor_estrategia": melhor_estrategia,
        "poupanca_split": (round(poupanca_split, 2) if poupanca_split is not None else None),
        "detalhe_por_loja": custo_por_loja,
    }


# ---------------------------------------------------------------------------
# 3b. Otimizar lista de compras COM CUSTO DE DESLOCAÇÃO (GPS)
# ---------------------------------------------------------------------------

def otimizar_lista_compras_geo(
    lista: list[str],
    user_lat: float,
    user_lon: float,
    custo_km: float | str | None = None,
    raio_km: float = 30.0,
) -> dict | None:
    """Otimiza a lista de compras considerando o **custo de deslocação ida-volta**.

    Variante geográfica de :func:`otimizar_lista_compras`. Para cada cadeia
    com lista completa disponível, calcula o **custo total = preço dos produtos +
    custo de deslocação à loja física mais próxima do utilizador**. Devolve
    a melhor cadeia segundo este critério, mais o detalhe por cadeia para
    permitir ao utilizador ver os trade-offs.

    A loja física relevante é a **mais próxima da insígnia** (não a mais
    próxima absoluta), porque os preços vêm do site online da cadeia — vai
    pagar o mesmo no Continente de Braga ou no Continente do Porto, mas a
    deslocação muda.

    O custo monetário (custo_km) é resolvido via
    :func:`models.geolocation.resolver_custo_km`, aceitando:
    - ``None`` → default 0.20 €/km (preset "equilibrado")
    - String de preset: ``"so_combustivel"``, ``"equilibrado"``, ``"tarifa_at"``
    - Número direto: ex. ``0.18``

    Args:
        lista: Lista de termos de pesquisa (igual a :func:`otimizar_lista_compras`).
        user_lat: Latitude do utilizador (graus decimais WGS84).
        user_lon: Longitude do utilizador (graus decimais WGS84).
        custo_km: Custo por km (preset ou valor). Default: 0.20 €/km.
        raio_km: Raio máximo para considerar uma loja física como "alcançável"
            (default: 30 km). Cadeias sem loja física no raio são consideradas
            inacessíveis e excluídas da comparação.

    Returns:
        Dicionário com:
        - ``"itens"``: lista de registos como em :func:`otimizar_lista_compras`.
        - ``"localizacao_utilizador"``: ``{lat, lon}`` ecoado.
        - ``"custo_km"``: valor €/km efectivo após resolução.
        - ``"raio_km"``: raio efectivo usado.
        - ``"custo_minimo_split_multi_loja"``: custo total se dividir pela melhor
          loja física por item (com deslocação a cada loja distinta). Pode não
          ser realista se forçar muitas deslocações.
        - ``"melhor_opcao"``: ``{"insignia", "loja_fisica", "distancia_km",
          "custo_produtos", "custo_deslocacao", "custo_total"}`` da cadeia
          recomendada. ``None`` se nenhuma cadeia for alcançável.
        - ``"detalhe_por_cadeia"``: ``{insignia: {custo_produtos, distancia_km,
          custo_deslocacao, custo_total, loja_fisica, em_falta}}`` para todas
          as cadeias com lista completa.
        Ou ``None`` se nenhum item for encontrado.
    """
    # Resolver o custo €/km (aceita preset ou valor)
    custo_km_efetivo = resolver_custo_km(custo_km)

    # Reaproveitar a otimização sem GPS — dá-nos custo_por_loja e itens
    base = otimizar_lista_compras(lista)
    if base is None:
        return None

    # Identificar cadeias com lista completa (sem em_falta)
    detalhe_loja = base["detalhe_por_loja"]
    cadeias_completas = [ins for ins, info in detalhe_loja.items() if not info["em_falta"]]

    # Calcular distância à loja física mais próxima de cada cadeia
    distancias = distancia_minima_por_insignia(
        user_lat, user_lon,
        insignias=list(detalhe_loja.keys()),   # incluir incompletas para diagnóstico
        raio_km=raio_km,
    )

    # Construir detalhe por cadeia com custo total (produtos + deslocação)
    detalhe_por_cadeia: dict[str, dict] = {}
    for ins, info_loja in detalhe_loja.items():
        loja_fisica = distancias.get(ins)
        if loja_fisica is None:
            # Cadeia sem loja física no raio — não comparável
            detalhe_por_cadeia[ins] = {
                "custo_produtos":   float(info_loja["total"]),
                "loja_fisica":      None,
                "distancia_km":     None,
                "custo_deslocacao": None,
                "custo_total":      None,
                "em_falta":         info_loja["em_falta"],
                "alcancavel":       False,
            }
            continue

        custo_desl = custo_deslocacao_euros(loja_fisica["distancia_km"],
                                             custo_km=custo_km_efetivo)
        detalhe_por_cadeia[ins] = {
            "custo_produtos":   float(info_loja["total"]),
            "loja_fisica":      {
                "id_loja_fisica": loja_fisica["id_loja_fisica"],
                "nome_loja":      loja_fisica["nome_loja"],
                "morada":         loja_fisica.get("morada"),
                "cidade":         loja_fisica.get("cidade"),
                "latitude":       loja_fisica["latitude"],
                "longitude":      loja_fisica["longitude"],
            },
            "distancia_km":     loja_fisica["distancia_km"],
            "custo_deslocacao": custo_desl,
            "custo_total":      round(float(info_loja["total"]) + custo_desl, 2),
            "em_falta":         info_loja["em_falta"],
            "alcancavel":       True,
        }

    # Escolher a melhor cadeia (custo_total mínimo, entre cadeias alcançáveis com lista completa)
    candidatas = {
        ins: d for ins, d in detalhe_por_cadeia.items()
        if d["alcancavel"] and ins in cadeias_completas
    }
    if candidatas:
        melhor_ins = min(candidatas, key=lambda ins: candidatas[ins]["custo_total"])
        melhor = {
            "insignia":        melhor_ins,
            "loja_fisica":     candidatas[melhor_ins]["loja_fisica"],
            "distancia_km":    candidatas[melhor_ins]["distancia_km"],
            "custo_produtos":  candidatas[melhor_ins]["custo_produtos"],
            "custo_deslocacao": candidatas[melhor_ins]["custo_deslocacao"],
            "custo_total":     candidatas[melhor_ins]["custo_total"],
        }
    else:
        melhor = None

    # Imprimir relatório no terminal (igual padrão das outras funções)
    print(f"\n[LISTA + GPS] Localização: ({user_lat:.4f}, {user_lon:.4f}) | "
          f"custo: {custo_km_efetivo}€/km | raio: {raio_km} km")

    if melhor is None:
        print("  Nenhuma cadeia tem a lista completa e loja física alcançável.")
    else:
        print(f"\n  RECOMENDAÇÃO: {melhor['insignia']} — total {melhor['custo_total']:.2f}€")
        print(f"    Produtos        : {melhor['custo_produtos']:.2f}€")
        print(f"    Deslocação      : {melhor['custo_deslocacao']:.2f}€ "
              f"({melhor['distancia_km']:.1f} km × 2 ida-volta × {custo_km_efetivo}€/km)")
        print(f"    Loja            : {melhor['loja_fisica']['nome_loja']}")

    print("\n  Comparação por cadeia:")
    print(f"    {'Cadeia':14s} {'Produtos':>10s} {'Distância':>12s} {'Desloc.':>10s} {'Total':>10s}")
    for ins, d in sorted(detalhe_por_cadeia.items(),
                          key=lambda kv: (kv[1]["custo_total"] is None, kv[1]["custo_total"] or 0)):
        if d["alcancavel"] and not d["em_falta"]:
            print(f"    {ins:14s} {d['custo_produtos']:9.2f}€ "
                  f"{d['distancia_km']:11.1f}km {d['custo_deslocacao']:9.2f}€ "
                  f"{d['custo_total']:9.2f}€")
        elif d["em_falta"]:
            print(f"    {ins:14s} {d['custo_produtos']:9.2f}€    [lista incompleta: "
                  f"{len(d['em_falta'])} item(s) em falta]")
        else:
            print(f"    {ins:14s}                  [sem loja física no raio de {raio_km} km]")

    return {
        "itens":                          base["itens"],
        "nao_encontrados":                base.get("nao_encontrados", []),
        "localizacao_utilizador":         {"lat": user_lat, "lon": user_lon},
        "custo_km":                       custo_km_efetivo,
        "raio_km":                        raio_km,
        "custo_minimo_split_multi_loja":  base["custo_minimo"],
        "melhor_opcao":                   melhor,
        "detalhe_por_cadeia":             detalhe_por_cadeia,
    }


# ---------------------------------------------------------------------------
# 3c. Otimização com rota de 2 lojas (multi-loja)
# ---------------------------------------------------------------------------

def _melhor_preco_em_cada_cadeia(
    cur: psycopg2.extensions.cursor,
    lista: list[str],
    insignias: list[str],
) -> dict[str, dict[str, dict | None]]:
    """Constrói uma matriz ``preços[item][cadeia]`` com o melhor preço de cada
    item em cada cadeia.

    Para cada combinação (item, cadeia), reutiliza :func:`_melhor_preco_para_item`
    com o filtro ``loja=cadeia``. Items não disponíveis numa cadeia ficam como
    ``None``. É a base para o algoritmo de split entre 2 lojas em
    :func:`otimizar_lista_compras_geo_multi_loja`.

    Args:
        cur: Cursor psycopg2 aberto.
        lista: Lista de termos de pesquisa.
        insignias: Lista de cadeias a considerar.

    Returns:
        Dict ``{nome_item: {insignia: registo_dict_ou_None}}``.
    """
    matriz: dict[str, dict[str, dict | None]] = {}
    for nome in lista:
        matriz[nome] = {}
        for ins in insignias:
            df = _melhor_preco_para_item(cur, nome, loja=ins)
            matriz[nome][ins] = df.iloc[0].to_dict() if not df.empty else None
    return matriz


def otimizar_lista_compras_geo_multi_loja(
    lista: list[str],
    user_lat: float,
    user_lon: float,
    custo_km: float | str | None = None,
    raio_km: float = 30.0,
) -> dict | None:
    """Otimiza a lista permitindo **dividir entre 2 cadeias** (rota triangular).

    Estende :func:`otimizar_lista_compras_geo` (que avalia "comprar tudo numa
    cadeia") considerando também "comprar parte em A e parte em B na mesma
    viagem". O modelo de deslocação assume uma rota triangular::

        user → loja_A → loja_B → user

    porque numa ida ao supermercado planeada é razoável visitar as duas em
    sequência (em vez de duas idas independentes desde casa).

    Decisão: limitamos a **2 cadeias** (não 3 ou 4). Mais cadeias multiplicariam
    a complexidade combinatorial sem ganho realista — poucos utilizadores fazem
    3+ paragens de supermercado na mesma viagem.

    Algoritmo:
        1. Calcular a melhor opção single-store (delega para
           :func:`otimizar_lista_compras_geo`).
        2. Construir matriz ``preços[item][cadeia]``.
        3. Para cada par ordenado (A, B) de cadeias alcançáveis:
           - Para cada item, escolher a cadeia (entre A e B) com preço mais baixo.
           - Se algum item não estiver em nenhuma das duas, par é inviável.
           - Se todos os items ficarem numa só, equivale a single — saltar.
           - Calcular distância triangular e custo total.
        4. Recomendar o par com menor custo_total, mas **só** se for melhor
           que o single. Caso contrário, mantém a recomendação single.

    Args:
        lista: Lista de termos de pesquisa.
        user_lat: Latitude do utilizador (WGS84).
        user_lon: Longitude do utilizador (WGS84).
        custo_km: Custo por km (preset ou valor). Default: 0.20 €/km.
        raio_km: Raio máximo (km) para considerar uma cadeia alcançável.

    Returns:
        Dicionário com a estrutura completa de :func:`otimizar_lista_compras_geo`
        mais:
        - ``"melhor_par"``: dict descrevendo a melhor combinação de 2 cadeias
          (com itens em cada, custos parciais, rota e total) — ``None`` se
          nenhum par for viável.
        - ``"todos_os_pares"``: lista de todos os pares avaliados, ordenada
          por custo_total. Útil para mostrar trade-offs.
        - ``"recomendacao"``: ``"single"`` ou ``"par"`` consoante o que tem
          menor custo_total.
        - ``"poupanca_par"``: diferença entre custo_total single e par
          (0 se single ganha).
        Ou ``None`` se nem o single nem o par produzirem resultados.
    """
    # 1. Single-store (reutiliza toda a lógica existente)
    single = otimizar_lista_compras_geo(
        lista, user_lat=user_lat, user_lon=user_lon,
        custo_km=custo_km, raio_km=raio_km,
    )
    if single is None:
        return None

    custo_km_efetivo = single["custo_km"]

    # 2. Construir matriz preços[item][cadeia] para cadeias alcançáveis
    insignias_alcancaveis = [
        ins for ins, d in single["detalhe_por_cadeia"].items()
        if d.get("alcancavel")
    ]
    if len(insignias_alcancaveis) < 2:
        # Sem pelo menos 2 cadeias alcançáveis, não há split possível
        return {**single, "melhor_par": None, "todos_os_pares": [],
                "recomendacao": "single", "poupanca_par": 0.0}

    with get_connection() as conn:
        with conn.cursor() as cur:
            matriz = _melhor_preco_em_cada_cadeia(cur, lista, insignias_alcancaveis)

    # 3. Avaliar cada par (ordenado para garantir 1 entrada por par)
    pares: list[dict] = []
    for i, ins_a in enumerate(insignias_alcancaveis):
        loja_a = single["detalhe_por_cadeia"][ins_a]["loja_fisica"]
        for ins_b in insignias_alcancaveis[i + 1:]:
            loja_b = single["detalhe_por_cadeia"][ins_b]["loja_fisica"]

            itens_em_a: list[dict] = []
            itens_em_b: list[dict] = []
            em_falta: list[str] = []
            custo_a = 0.0
            custo_b = 0.0

            for nome in lista:
                p_a = matriz[nome].get(ins_a)
                p_b = matriz[nome].get(ins_b)
                if p_a is None and p_b is None:
                    em_falta.append(nome)
                    continue
                # Escolhe a cadeia mais barata (ties → A para determinismo)
                if p_b is None or (p_a is not None and float(p_a["preco_atual"]) <= float(p_b["preco_atual"])):
                    preco = float(p_a["preco_atual"])
                    custo_a += preco
                    itens_em_a.append({
                        "item_pesquisado": nome,
                        "nome_padronizado": p_a["nome_padronizado"],
                        "preco_atual":     preco,
                        "id_produto_loja": int(p_a["id_produto_loja"]),
                    })
                else:
                    preco = float(p_b["preco_atual"])
                    custo_b += preco
                    itens_em_b.append({
                        "item_pesquisado": nome,
                        "nome_padronizado": p_b["nome_padronizado"],
                        "preco_atual":     preco,
                        "id_produto_loja": int(p_b["id_produto_loja"]),
                    })

            # Par só faz sentido se:
            #   (a) não há items em falta
            #   (b) ambas as cadeias têm pelo menos 1 item
            # Caso contrário, equivale a single-store em A ou em B.
            if em_falta or not itens_em_a or not itens_em_b:
                continue

            # Distância triangular: user → A → B → user
            d_ua = haversine_km(user_lat, user_lon,
                                 loja_a["latitude"], loja_a["longitude"])
            d_ab = haversine_km(loja_a["latitude"], loja_a["longitude"],
                                 loja_b["latitude"], loja_b["longitude"])
            d_bu = haversine_km(loja_b["latitude"], loja_b["longitude"],
                                 user_lat, user_lon)
            dist_total = d_ua + d_ab + d_bu
            custo_desloc = round(dist_total * custo_km_efetivo, 2)
            custo_total = round(custo_a + custo_b + custo_desloc, 2)

            pares.append({
                "cadeias":            [ins_a, ins_b],
                "loja_fisica_a":      loja_a,
                "loja_fisica_b":      loja_b,
                "itens_em_a":         itens_em_a,
                "itens_em_b":         itens_em_b,
                "custo_produtos_a":   round(custo_a, 2),
                "custo_produtos_b":   round(custo_b, 2),
                "custo_produtos":     round(custo_a + custo_b, 2),
                "distancia_user_a_km":  round(d_ua, 2),
                "distancia_a_b_km":     round(d_ab, 2),
                "distancia_b_user_km":  round(d_bu, 2),
                "distancia_total_km":   round(dist_total, 2),
                "custo_deslocacao":   custo_desloc,
                "custo_total":        custo_total,
            })

    pares.sort(key=lambda p: p["custo_total"])
    melhor_par = pares[0] if pares else None

    # 4. Decidir recomendação: single vs par
    custo_single = single.get("melhor_opcao", {}).get("custo_total") if single.get("melhor_opcao") else None

    if melhor_par is None or custo_single is None or melhor_par["custo_total"] >= custo_single:
        recomendacao = "single"
        poupanca_par = 0.0
    else:
        recomendacao = "par"
        poupanca_par = round(custo_single - melhor_par["custo_total"], 2)

    # Imprimir resumo no terminal
    print("\n[LISTA + GPS + MULTI-LOJA]")
    if melhor_par is None:
        print("  Sem pares viáveis — comparação multi-loja não aplicável.")
    else:
        print(f"  Melhor single-store : {custo_single:.2f}€ "
              f"({single['melhor_opcao']['insignia']})" if custo_single else "  Sem single-store recomendado.")
        print(f"  Melhor par          : {melhor_par['custo_total']:.2f}€ "
              f"({' + '.join(melhor_par['cadeias'])})")
        if recomendacao == "par":
            print(f"  -> RECOMENDADO DIVIDIR: poupança {poupanca_par:.2f}€")
            print(f"     Itens em {melhor_par['cadeias'][0]} ({len(melhor_par['itens_em_a'])}): "
                  f"{melhor_par['custo_produtos_a']:.2f}€")
            print(f"     Itens em {melhor_par['cadeias'][1]} ({len(melhor_par['itens_em_b'])}): "
                  f"{melhor_par['custo_produtos_b']:.2f}€")
            print(f"     Rota: user → {melhor_par['cadeias'][0]} ({melhor_par['distancia_user_a_km']:.1f}km) "
                  f"→ {melhor_par['cadeias'][1]} ({melhor_par['distancia_a_b_km']:.1f}km) "
                  f"→ user ({melhor_par['distancia_b_user_km']:.1f}km) = {melhor_par['distancia_total_km']:.1f}km")
        else:
            print(f"  -> SINGLE-STORE CONTINUA MELHOR (dividir custaria mais "
                  f"{melhor_par['custo_total'] - (custo_single or 0):.2f}€).")

    return {
        **single,
        "melhor_par":     melhor_par,
        "todos_os_pares": pares,
        "recomendacao":   recomendacao,
        "poupanca_par":   poupanca_par,
    }


# ---------------------------------------------------------------------------
# 3d. Produto perto de mim — cruzar pesquisa de produto com lojas físicas
# ---------------------------------------------------------------------------

def produto_perto_de_mim(
    termo_pesquisa: str,
    user_lat: float,
    user_lon: float,
    custo_km: float | str | None = None,
    raio_km: float = 30.0,
    top_n: int = 5,
) -> list[dict] | None:
    """Encontra **um produto** nas cadeias e cruza com a loja física mais próxima.

    Caso de uso: "quero comprar arroz agulha — quanto custa em cada cadeia
    e onde fica a loja física mais próxima de mim, considerando o custo de
    deslocação?".

    Diferente de :func:`otimizar_lista_compras_geo` (que tenta minimizar o
    custo de uma **lista** e recomenda uma única cadeia) — esta função devolve
    **todas as cadeias** com o produto, ordenadas por custo total
    (preço + 2 × distância × €/km, ida-volta), para o utilizador comparar.

    Args:
        termo_pesquisa: Termo de pesquisa para o produto (ex: ``"arroz agulha"``).
        user_lat: Latitude do utilizador (WGS84).
        user_lon: Longitude do utilizador (WGS84).
        custo_km: Preset (``"so_combustivel"``/``"equilibrado"``/``"tarifa_at"``)
            ou valor numérico. Default: ``equilibrado`` (0.20 €/km).
        raio_km: Raio máximo (km) para considerar uma cadeia alcançável.
        top_n: Número máximo de cadeias a devolver na resposta.

    Returns:
        Lista de dicts (uma entrada por cadeia alcançável com o produto),
        ordenada por ``custo_total`` ascendente. Cada entrada tem:

        - ``insignia``, ``produto`` (nome padronizado), ``id_produto_loja``
        - ``preco_atual`` (€), ``em_promocao`` (bool)
        - ``preco_unitario_valor``, ``preco_unitario_unidade``
        - ``loja_fisica`` (dict com nome, morada, cidade, lat/lon, distância)
        - ``distancia_km``, ``custo_deslocacao`` (€), ``custo_total`` (€)

        Ou ``None`` se nenhuma cadeia tiver o produto.
    """
    # 1. Encontrar o produto em cada cadeia (reutiliza recomendar_melhor_loja)
    df_precos = recomendar_melhor_loja(termo_pesquisa, top_n=10)
    if df_precos is None or df_precos.empty:
        return None

    custo_km_efetivo = resolver_custo_km(custo_km)

    # 2. Identificar cadeias e ir buscar a loja física mais próxima de cada uma
    insignias = df_precos["loja"].unique().tolist()
    distancias = distancia_minima_por_insignia(
        user_lat, user_lon, insignias=insignias, raio_km=raio_km,
    )

    # 3. Construir resultado linha por linha (1 entrada por cadeia alcançável)
    resultados: list[dict] = []
    for _, row in df_precos.iterrows():
        ins = row["loja"]
        loja_fisica = distancias.get(ins)
        if loja_fisica is None:
            continue   # cadeia sem loja física no raio
        # df_precos pode ter várias linhas da mesma cadeia (DISTINCT ON nome+loja
        # dá mais que 1 se houver produtos com nomes ligeiramente diferentes).
        # Vamos manter só a entrada de menor preço por cadeia.
        ja_incluida = next((r for r in resultados if r["insignia"] == ins), None)
        if ja_incluida is not None:
            continue

        preco = float(row["preco_atual"])
        custo_desl = custo_deslocacao_euros(
            loja_fisica["distancia_km"], custo_km=custo_km_efetivo,
        )
        resultados.append({
            "insignia":               ins,
            "produto":                row["nome_padronizado"],
            "id_produto_loja":        int(row["id_produto_loja"]),
            "preco_atual":            preco,
            "em_promocao":            bool(row["em_promocao"]),
            "preco_unitario_valor":   (float(row["preco_unitario_valor"])
                                       if row.get("preco_unitario_valor") is not None
                                       else None),
            "preco_unitario_unidade": row.get("preco_unitario_unidade"),
            "loja_fisica": {
                "id_loja_fisica":  loja_fisica["id_loja_fisica"],
                "nome_loja":       loja_fisica["nome_loja"],
                "morada":          loja_fisica.get("morada"),
                "cidade":          loja_fisica.get("cidade"),
                "latitude":        loja_fisica["latitude"],
                "longitude":       loja_fisica["longitude"],
            },
            "distancia_km":           loja_fisica["distancia_km"],
            "custo_deslocacao":       custo_desl,
            "custo_total":            round(preco + custo_desl, 2),
        })

    if not resultados:
        return []

    resultados.sort(key=lambda r: r["custo_total"])
    return resultados[:top_n]


# ---------------------------------------------------------------------------
# 4. Recomendar momento de compra (LSTM + Monte Carlo Dropout)
# ---------------------------------------------------------------------------

def recomendar_momento_compra(
    produto_id: int,
    caminho_csv: str,
    horizonte: int = 7,
    limiar_descida_pct: float = 2.0,
    n_amostras_mc: int = 50,
) -> dict | None:
    """Recomenda comprar agora ou aguardar com base em previsão probabilística.

    Usa :func:`~price_predictor.prever_preco_com_incerteza` (Monte Carlo Dropout)
    para gerar um intervalo de confiança sobre a evolução futura do preço.
    A decisão de "aguardar" só é tomada quando a descida prevista supera
    ``limiar_descida_pct``; a confiança da recomendação é "alta" se o
    percentil 5% (cenário pessimista) também superar o limiar.

    Args:
        produto_id: ``id_produto_loja`` a analisar.
        caminho_csv: Caminho para o CSV de forecasting.
        horizonte: Número de dias a analisar (default: 7).
        limiar_descida_pct: Descida mínima prevista (%) para recomendar espera
            (default: 2.0).
        n_amostras_mc: Número de simulações Monte Carlo (default: 50).

    Returns:
        Dicionário com: ``produto_id``, ``nome_produto``, ``preco_atual``,
        ``preco_minimo_previsto``, ``data_minimo``, ``descida_pct``,
        ``descida_pct_ic_inf``, ``recomendacao`` (``"comprar_agora"`` ou
        ``"aguardar"``), ``previsoes``.
        Ou ``None`` se o modelo não estiver disponível.
    """
    nome_produto = f"produto {produto_id}"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pm.nome_padronizado, l.insignia
                    FROM produtos_loja   pl
                    JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
                    JOIN lojas            l  ON l.id_loja            = pl.id_loja
                    WHERE pl.id_produto_loja = %s
                """, (produto_id,))
                row = cur.fetchone()
                if row:
                    nome_produto = f"{row[0]} ({row[1]})"
    except Exception as exc:
        warnings.warn(f"[AVISO] Nao foi possivel obter nome do produto {produto_id}: {exc}")

    resultado_mc = prever_preco_com_incerteza(
        produto_id, caminho_csv, horizonte, n_amostras=n_amostras_mc
    )
    if resultado_mc is None:
        return None

    df_previsoes = resultado_mc["previsoes"]
    preco_agora = resultado_mc["preco_atual"]

    preco_minimo_medio = float(df_previsoes["preco_medio"].min())
    preco_ic_inf_min = float(df_previsoes["ic_5pct"].min())
    dia_minimo = df_previsoes.loc[df_previsoes["preco_medio"].idxmin(), "data"]

    descida_pct_medio = max(0.0, (preco_agora - preco_minimo_medio) / preco_agora * 100) if preco_agora > 0 else 0.0
    descida_pct_ic_inf = max(0.0, (preco_agora - preco_ic_inf_min) / preco_agora * 100) if preco_agora > 0 else 0.0

    print(f"\n[MOMENTO DE COMPRA] {nome_produto}")
    print(f"  Preco atual (ultimo real)        : {preco_agora:.2f} EUR")
    print(f"  Previsao media minima            : {preco_minimo_medio:.2f} EUR (em {dia_minimo})")
    print(f"  Descida esperada (media MC)      : {descida_pct_medio:.2f}%")
    print(f"  Descida optimista (IC 5%)        : {descida_pct_ic_inf:.2f}%")
    print(f"  Limiar para recomendar espera    : {limiar_descida_pct}%")

    print(f"\n  Previsoes dia a dia ({n_amostras_mc} simulacoes MC):")
    print(df_previsoes.to_string(index=False))

    if df_previsoes["preco_std"].mean() < 0.01:
        print("\n  [INFO] Incerteza muito baixa — modelo preve preco estavel.")
        print("         Normal com historico curto (<60 dias).")

    if descida_pct_medio >= limiar_descida_pct:
        dias_espera = (pd.Timestamp(dia_minimo) - pd.Timestamp(df_previsoes["data"].iloc[0])).days + 1
        recomendacao = "aguardar"
        confianca = "alta" if descida_pct_ic_inf >= limiar_descida_pct else "moderada"
        print(f"\n  -> AGUARDAR {dias_espera} dia(s): descida de {descida_pct_medio:.1f}% prevista (confianca: {confianca}).")
    else:
        recomendacao = "comprar_agora"
        print(f"\n  -> COMPRAR AGORA: descida prevista ({descida_pct_medio:.2f}%) abaixo do limiar.")

    return {
        "produto_id": produto_id,
        "nome_produto": nome_produto,
        "preco_atual": round(preco_agora, 2),
        "preco_minimo_previsto": round(preco_minimo_medio, 2),
        "data_minimo": str(dia_minimo),
        "descida_pct": round(descida_pct_medio, 2),
        "descida_pct_ic_inf": round(descida_pct_ic_inf, 2),
        "recomendacao": recomendacao,
        "previsoes": df_previsoes.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI do sistema de recomendação."""
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendacao de Compras de Supermercado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python models/recommender.py --pesquisar "arroz"
  python models/recommender.py --melhor-loja "arroz agulha"
  python models/recommender.py --lista "arroz agulha,azeite virgem extra,atum natural"
  python models/recommender.py --momento --produto-id 1
        """,
    )
    parser.add_argument(
        "--pesquisar", type=str, metavar="TERMO",
        help="Listar produtos na BD que contem o termo.",
    )
    parser.add_argument(
        "--melhor-loja", type=str, metavar="PRODUTO",
        help="Encontrar melhor loja para um produto.",
    )
    parser.add_argument(
        "--lista", type=str, metavar="'ITEM1,ITEM2,...'",
        help="Otimizar lista de compras (itens separados por virgula).",
    )
    parser.add_argument(
        "--momento", action="store_true",
        help="Recomendar momento de compra via previsao LSTM (requer --produto-id).",
    )
    parser.add_argument(
        "--produto-id", type=int,
        help="id_produto_loja a usar com --momento.",
    )
    parser.add_argument(
        "--dataset",
        default="data/generated/forecasting_dataset.csv",
        help="Caminho para o forecasting_dataset.csv (default: data/generated/forecasting_dataset.csv).",
    )
    parser.add_argument(
        "--horizonte", type=int, default=7,
        help="Dias a analisar com --momento (default: 7).",
    )
    parser.add_argument(
        "--limiar", type=float, default=2.0,
        help="Descida minima em %% para recomendar esperar (default: 2.0).",
    )
    # --- Opções GPS / custo de deslocação ---
    parser.add_argument(
        "--lat", type=float, metavar="LAT",
        help="Latitude do utilizador (WGS84). Activa otimização com custo de deslocação.",
    )
    parser.add_argument(
        "--lon", type=float, metavar="LON",
        help="Longitude do utilizador (WGS84). Activa otimização com custo de deslocação.",
    )
    parser.add_argument(
        "--custo-km", type=str, default=None, metavar="PRESET_OR_VALOR",
        help="Custo por km: preset ('so_combustivel', 'equilibrado', 'tarifa_at') ou número (ex: 0.18). Default: equilibrado (0.20).",
    )
    parser.add_argument(
        "--raio-km", type=float, default=30.0, metavar="KM",
        help="Raio máximo (km) para considerar lojas físicas alcançáveis (default: 30).",
    )
    parser.add_argument(
        "--multi-loja", action="store_true",
        help="Avaliar também a divisão da lista entre 2 cadeias (rota triangular). "
             "Requer --lat/--lon.",
    )
    parser.add_argument(
        "--perto-de-mim", type=str, metavar="PRODUTO",
        help="Encontrar um produto e ordenar cadeias por custo total (preço + "
             "deslocação à loja física mais próxima). Requer --lat/--lon.",
    )
    args = parser.parse_args()

    if args.pesquisar:
        pesquisar_produtos(args.pesquisar)

    if args.melhor_loja:
        recomendar_melhor_loja(args.melhor_loja)

    if args.lista:
        itens = [i.strip() for i in args.lista.split(",") if i.strip()]
        # Se o utilizador passou lat+lon, usa a variante GPS; caso contrário a normal.
        if args.lat is not None and args.lon is not None:
            if args.multi_loja:
                otimizar_lista_compras_geo_multi_loja(
                    itens,
                    user_lat=args.lat, user_lon=args.lon,
                    custo_km=args.custo_km, raio_km=args.raio_km,
                )
            else:
                otimizar_lista_compras_geo(
                    itens,
                    user_lat=args.lat, user_lon=args.lon,
                    custo_km=args.custo_km, raio_km=args.raio_km,
                )
        else:
            if args.lat is not None or args.lon is not None:
                parser.error("--lat e --lon têm de ser usados juntos.")
            if args.multi_loja:
                parser.error("--multi-loja requer --lat e --lon.")
            otimizar_lista_compras(itens)

    if args.perto_de_mim:
        if args.lat is None or args.lon is None:
            parser.error("--perto-de-mim requer --lat e --lon.")
        resultado = produto_perto_de_mim(
            args.perto_de_mim,
            user_lat=args.lat, user_lon=args.lon,
            custo_km=args.custo_km, raio_km=args.raio_km,
            top_n=5,
        )
        if not resultado:
            print(f"\n[PERTO DE MIM] Sem cadeias com '{args.perto_de_mim}' "
                  f"no raio de {args.raio_km} km.")
        else:
            print(f"\n[PERTO DE MIM] Cadeias com '{args.perto_de_mim}' "
                  f"ordenadas por custo total:")
            for r in resultado:
                print(f"  {r['insignia']:12s} | {r['produto'][:40]:40s} | "
                      f"{r['preco_atual']:.2f}€ produto + "
                      f"{r['custo_deslocacao']:.2f}€ desloc "
                      f"({r['distancia_km']:.1f}km) "
                      f"= {r['custo_total']:.2f}€ total")
                print(f"  {' ':12s}   loja: {r['loja_fisica']['nome_loja']}")

    if args.momento:
        if not args.produto_id:
            parser.error("--momento requer --produto-id")
        recomendar_momento_compra(
            produto_id=args.produto_id,
            caminho_csv=args.dataset,
            horizonte=args.horizonte,
            limiar_descida_pct=args.limiar,
        )


if __name__ == "__main__":
    main()
