"""
Back-test do recomendador de compras vs baselines.

Mede o ganho real do recomendador comparando-o com 3 estratégias mais simples:

1. **Recomendador (multi-loja)** — escolher o menor preço entre todas as cadeias
   para cada item da lista. É o que :func:`models.recommender.otimizar_lista_compras`
   devolve no campo ``custo_minimo``.
2. **Lealdade a uma cadeia** — comprar **tudo na mesma cadeia** (a única que
   tem a lista completa, ou a mais barata entre as que têm). Mede o ganho de
   comparar cadeias.
3. **Aleatório** — para cada item, escolher uma cadeia ao acaso (entre as que
   têm o produto). Simula um utilizador que não compara preços.

O back-test percorre vários dias do histórico de preços real, simulando o
custo de cada estratégia naqueles dias. A média ao longo do histórico é o
**ganho esperado** que o recomendador entrega.

Uso::

    python scripts/backtest_recomendador.py
    python scripts/backtest_recomendador.py --output data/generated/backtest.json
    python scripts/backtest_recomendador.py --seed 42 --n-amostras-aleatorio 200

Output:
    JSON em ``data/generated/backtest_recomendador.json`` com:
    - ``listas``: definições das listas testadas
    - ``por_dia``: custos por estratégia para cada (lista, dia)
    - ``agregado``: médias por estratégia + poupança do recomendador

    Consumido pelo dashboard ("Tab Validação do Recomendador").
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG


# ---------------------------------------------------------------------------
# Listas de compras realistas para o back-test
# ---------------------------------------------------------------------------
#
# Escolhidas para cobrir diferentes perfis: refeições básicas, pequeno-almoço,
# limpeza, snacks. Cada lista tem 4-7 itens — o tamanho típico de uma compra
# semanal pontual em supermercado.

LISTAS_BACKTEST: dict[str, list[str]] = {
    "Refeição básica": [
        "arroz agulha", "azeite virgem extra", "atum natural",
        "esparguete", "feijão vermelho",
    ],
    "Pequeno-almoço": [
        "leite meio gordo", "cereais chocolate", "iogurte natural",
        "marmelada", "café cápsulas",
    ],
    "Snacks e bebidas": [
        "água sem gás", "sumo laranja", "bolachas maria",
        "chocolate tablete", "batatas fritas pacote",
    ],
    "Mediterrânea": [
        "azeite virgem extra", "massa esparguete", "tomate pelado",
        "queijo flamengo", "atum azeite",
    ],
    "Económica": [
        "arroz", "massa", "atum natural", "feijão", "azeite",
    ],
}


# ---------------------------------------------------------------------------
# Carregamento de dados (histórico real)
# ---------------------------------------------------------------------------

def carregar_historico_para_backtest() -> pd.DataFrame:
    """Carrega histórico de preços + metadados de produto/cadeia.

    Returns:
        DataFrame com colunas ``data, insignia, nome_padronizado,
        id_produto_loja, preco``. Uma linha por (produto, dia).
    """
    sql = """
        SELECT
            h.data_recolha::date  AS data,
            l.insignia            AS insignia,
            pm.nome_padronizado   AS nome_padronizado,
            pl.id_produto_loja    AS id_produto_loja,
            h.preco_atual::float  AS preco
        FROM historico_precos h
        JOIN produtos_loja   pl ON pl.id_produto_loja   = h.id_produto_loja
        JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
        JOIN lojas            l  ON l.id_loja            = pl.id_loja
        WHERE h.preco_atual IS NOT NULL AND h.preco_atual > 0
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(sql, conn)
    df["data"] = pd.to_datetime(df["data"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Núcleo do back-test
# ---------------------------------------------------------------------------

def _melhor_match_por_cadeia(
    df_dia: pd.DataFrame,
    termo: str,
) -> dict[str, float | None]:
    """Para um item de lista e os preços de um dia, devolve o melhor preço por cadeia.

    Usa "todas as palavras com >=2 chars têm de aparecer no nome" — alinhado
    com o pg_trgm `_condicoes_multi_ilike` do recomendador real, sem o trigram
    score (suficiente para o back-test).

    Returns:
        Dict ``{insignia: menor_preco | None}``. None se a cadeia não tiver o
        produto. As 3 cadeias estão sempre presentes na chave.
    """
    palavras = [p.lower() for p in termo.split() if len(p) >= 2]
    if not palavras:
        return {ins: None for ins in ("Auchan", "Continente", "Pingo Doce")}

    nomes = df_dia["nome_padronizado"].str.lower()
    # Conjunto: todas as palavras presentes no nome_padronizado
    mask = nomes.notna()
    for p in palavras:
        mask &= nomes.str.contains(p, na=False, regex=False)

    sub = df_dia.loc[mask]
    resultado: dict[str, float | None] = {}
    for ins in ("Auchan", "Continente", "Pingo Doce"):
        precos = sub.loc[sub["insignia"] == ins, "preco"]
        resultado[ins] = float(precos.min()) if not precos.empty else None
    return resultado


def simular_lista_no_dia(
    lista: list[str],
    df_dia: pd.DataFrame,
    rng: random.Random,
) -> dict | None:
    """Simula o custo de uma lista de compras num único dia, por estratégia.

    Args:
        lista: Lista de termos de pesquisa (ex: ``["arroz", "azeite"]``).
        df_dia: Subset do histórico para o dia em causa.
        rng: Gerador aleatório (para a estratégia aleatória — determinístico
            quando inicializado com seed).

    Returns:
        Dict com ``custo_recomendador``, ``custo_aleatorio``, ``custo_lealdade_*``
        para cada cadeia. None se a lista não tiver matches suficientes (>50%
        em falta), porque uma comparação assim não é informativa.
    """
    melhores_por_item: list[dict[str, float | None]] = []
    itens_encontrados = 0
    for termo in lista:
        match = _melhor_match_por_cadeia(df_dia, termo)
        if any(v is not None for v in match.values()):
            itens_encontrados += 1
        melhores_por_item.append(match)

    # Excluir dias onde menos de metade dos itens existiam
    if itens_encontrados < (len(lista) + 1) // 2:
        return None

    # --- Recomendador (multi-loja: menor preço entre cadeias por item) ---
    custos_rec: list[float] = []
    for match in melhores_por_item:
        precos = [p for p in match.values() if p is not None]
        if precos:
            custos_rec.append(min(precos))
    custo_recomendador = sum(custos_rec) if custos_rec else 0.0

    # --- Lealdade (tudo numa única cadeia — só vale se todos os itens existirem) ---
    custos_lealdade: dict[str, float | None] = {}
    for ins in ("Auchan", "Continente", "Pingo Doce"):
        precos_ins = [m[ins] for m in melhores_por_item if m[ins] is not None]
        if len(precos_ins) == itens_encontrados:
            custos_lealdade[ins] = sum(precos_ins)
        else:
            custos_lealdade[ins] = None   # cadeia não tem a lista completa

    # --- Aleatório (para cada item, escolher cadeia ao acaso entre as que têm) ---
    custos_aleatorio: list[float] = []
    for match in melhores_por_item:
        opcoes = [(ins, p) for ins, p in match.items() if p is not None]
        if opcoes:
            _, p = rng.choice(opcoes)
            custos_aleatorio.append(p)
    custo_aleatorio = sum(custos_aleatorio) if custos_aleatorio else 0.0

    return {
        "custo_recomendador": round(custo_recomendador, 2),
        "custo_aleatorio":    round(custo_aleatorio, 2),
        "custo_lealdade":     {ins: (round(v, 2) if v is not None else None)
                                for ins, v in custos_lealdade.items()},
        "itens_encontrados":  itens_encontrados,
        "itens_totais":       len(lista),
    }


def correr_backtest(
    df_historico: pd.DataFrame,
    listas: dict[str, list[str]],
    n_amostras_aleatorio: int = 100,
    seed: int = 42,
) -> dict:
    """Corre o back-test completo: cada lista × cada dia × cada estratégia.

    Para a estratégia "Aleatório", calcula-se a **média de N amostras** para
    reduzir variância (com 1 amostra a comparação seria ruidosa).

    Args:
        df_historico: DataFrame com colunas ``data, insignia, nome_padronizado,
            id_produto_loja, preco``.
        listas: Dict ``{nome_lista: itens}``.
        n_amostras_aleatorio: Quantas vezes repetir a estratégia aleatória
            para média estável. Default 100.
        seed: Semente do gerador aleatório (para reprodutibilidade).

    Returns:
        Dict com chaves ``listas``, ``por_dia``, ``agregado``, ``metadata``.
    """
    rng_global = random.Random(seed)
    dias = sorted(df_historico["data"].unique())

    por_dia: list[dict] = []
    for nome_lista, itens in listas.items():
        print(f"[BACKTEST] Lista '{nome_lista}' ({len(itens)} itens)...")
        for dia in dias:
            df_dia = df_historico[df_historico["data"] == dia]
            # Para a aleatória, média sobre N amostras (reduz variância)
            amostras_aleatorio: list[float] = []
            resultado_base: dict | None = None
            for k in range(n_amostras_aleatorio):
                rng = random.Random(rng_global.randint(0, 10**9))
                sim = simular_lista_no_dia(itens, df_dia, rng)
                if sim is None:
                    break
                if resultado_base is None:
                    resultado_base = sim
                amostras_aleatorio.append(sim["custo_aleatorio"])
            if resultado_base is None:
                continue

            por_dia.append({
                "lista":             nome_lista,
                "data":              str(dia),
                "custo_recomendador": resultado_base["custo_recomendador"],
                "custo_aleatorio_media": round(
                    sum(amostras_aleatorio) / len(amostras_aleatorio), 2
                ),
                "custo_aleatorio_max":   round(max(amostras_aleatorio), 2),
                "custo_aleatorio_min":   round(min(amostras_aleatorio), 2),
                "custo_lealdade":    resultado_base["custo_lealdade"],
                "itens_encontrados": resultado_base["itens_encontrados"],
                "itens_totais":      resultado_base["itens_totais"],
            })

    # ----- Agregação por lista e global -----
    agregado_por_lista: dict[str, dict] = {}
    for nome_lista in listas:
        registos = [r for r in por_dia if r["lista"] == nome_lista]
        if not registos:
            continue
        n = len(registos)
        rec_medio   = sum(r["custo_recomendador"] for r in registos) / n
        ale_medio   = sum(r["custo_aleatorio_media"] for r in registos) / n
        poupanca_vs_aleatorio_eur = ale_medio - rec_medio
        poupanca_vs_aleatorio_pct = (
            poupanca_vs_aleatorio_eur / ale_medio * 100 if ale_medio > 0 else 0.0
        )

        # Lealdade: comparar o recomendador contra "ser leal a uma única cadeia".
        # Cada cadeia só tem a lista completa num subconjunto de dias. Comparar a
        # média da cadeia (poucos dias, eventualmente baratos) contra a média do
        # recomendador sobre TODOS os dias seria injusto — uma cadeia com 2/24
        # dias pode parecer mais barata quando, nesses mesmos dias, o recomendador
        # a iguala sempre (o recomendador escolhe o mínimo por item). Por isso:
        #   (1) só consideramos cadeias com cobertura >= COBERTURA_MIN dos dias
        #       válidos (ser "leal" a uma cadeia que quase nunca tem a lista não é
        #       uma estratégia realista);
        #   (2) comparamos o recomendador sobre OS MESMOS dias da cadeia escolhida.
        # Com esta correção a poupança vs lealdade é sempre >= 0 e mede o ganho
        # marginal real de dividir a lista entre cadeias.
        COBERTURA_MIN = 0.5
        custo_lealdade_medio: dict[str, float] = {}
        lealdade_cadeias: dict[str, dict] = {}
        for ins in ("Auchan", "Continente", "Pingo Doce"):
            dias_ins = [r for r in registos if r["custo_lealdade"][ins] is not None]
            if not dias_ins:
                continue
            custo_lealdade_medio[ins] = sum(
                r["custo_lealdade"][ins] for r in dias_ins
            ) / len(dias_ins)
            if len(dias_ins) >= COBERTURA_MIN * n:
                lealdade_cadeias[ins] = {
                    "lealdade_medio":  custo_lealdade_medio[ins],
                    "rec_medio_match": sum(r["custo_recomendador"] for r in dias_ins)
                                       / len(dias_ins),
                }

        # Melhor cadeia única = a mais barata ENTRE as de cobertura suficiente.
        if lealdade_cadeias:
            melhor_ins = min(
                lealdade_cadeias, key=lambda i: lealdade_cadeias[i]["lealdade_medio"]
            )
            ld = lealdade_cadeias[melhor_ins]
            poupanca_vs_lealdade_eur = ld["lealdade_medio"] - ld["rec_medio_match"]
            poupanca_vs_lealdade_pct = (
                poupanca_vs_lealdade_eur / ld["lealdade_medio"] * 100
                if ld["lealdade_medio"] > 0 else 0.0
            )
        else:
            melhor_ins = None
            poupanca_vs_lealdade_eur = 0.0
            poupanca_vs_lealdade_pct = 0.0

        agregado_por_lista[nome_lista] = {
            "n_dias":                    n,
            "custo_recomendador_medio":  round(rec_medio, 2),
            "custo_aleatorio_medio":     round(ale_medio, 2),
            "custo_lealdade_medio":      {k: round(v, 2) for k, v in custo_lealdade_medio.items()},
            "melhor_cadeia_lealdade":    melhor_ins,
            "poupanca_vs_aleatorio_eur": round(poupanca_vs_aleatorio_eur, 2),
            "poupanca_vs_aleatorio_pct": round(poupanca_vs_aleatorio_pct, 2),
            "poupanca_vs_melhor_lealdade_eur": round(poupanca_vs_lealdade_eur, 2),
            "poupanca_vs_melhor_lealdade_pct": round(poupanca_vs_lealdade_pct, 2),
        }

    # ----- Agregação global (média simples por lista) -----
    if agregado_por_lista:
        n_listas = len(agregado_por_lista)
        global_poupanca_aleatorio_pct = sum(
            v["poupanca_vs_aleatorio_pct"] for v in agregado_por_lista.values()
        ) / n_listas
        global_poupanca_aleatorio_eur = sum(
            v["poupanca_vs_aleatorio_eur"] for v in agregado_por_lista.values()
        ) / n_listas
        # vs lealdade: só conta listas onde existe uma cadeia de cobertura
        # suficiente — caso contrário não há comparação de lealdade realista.
        listas_lealdade = [
            v for v in agregado_por_lista.values()
            if v["melhor_cadeia_lealdade"] is not None
        ]
        global_poupanca_lealdade_pct = (
            sum(v["poupanca_vs_melhor_lealdade_pct"] for v in listas_lealdade)
            / len(listas_lealdade)
        ) if listas_lealdade else 0.0
    else:
        global_poupanca_aleatorio_pct = 0.0
        global_poupanca_lealdade_pct = 0.0
        global_poupanca_aleatorio_eur = 0.0

    return {
        "metadata": {
            "n_listas": len(listas),
            "n_dias":   len({r["data"] for r in por_dia}),
            "n_amostras_aleatorio": n_amostras_aleatorio,
            "seed":     seed,
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
        },
        "listas":   listas,
        "por_dia":  por_dia,
        "agregado_por_lista": agregado_por_lista,
        "global": {
            "poupanca_vs_aleatorio_pct": round(global_poupanca_aleatorio_pct, 2),
            "poupanca_vs_aleatorio_eur": round(global_poupanca_aleatorio_eur, 2),
            "poupanca_vs_melhor_lealdade_pct": round(global_poupanca_lealdade_pct, 2),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI do back-test."""
    parser = argparse.ArgumentParser(
        description="Back-test do recomendador de compras vs baselines.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent.parent / "data/generated/backtest_recomendador.json"),
        help="Ficheiro JSON de output (default: data/generated/backtest_recomendador.json).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente do RNG para reprodutibilidade (default: 42).",
    )
    parser.add_argument(
        "--n-amostras-aleatorio", type=int, default=100,
        help="Amostras por dia para a estratégia aleatória (default: 100).",
    )
    args = parser.parse_args()

    print("[BACKTEST] A carregar histórico de preços da BD...")
    df = carregar_historico_para_backtest()
    print(f"[BACKTEST] {len(df):,} observações em {df['data'].nunique()} dias.")

    resultado = correr_backtest(
        df, LISTAS_BACKTEST,
        n_amostras_aleatorio=args.n_amostras_aleatorio,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # ----- Resumo no terminal -----
    print(f"\n[BACKTEST] Resultados guardados em: {out_path}")
    print("\n[BACKTEST] Resumo por lista:")
    print(f"  {'Lista':25s} {'Rec(€)':>8s} {'Aleat(€)':>9s} {'Ganho(€)':>9s} {'Ganho(%)':>9s} "
          f"{'vs lealdade(%)':>15s}")
    for nome, v in resultado["agregado_por_lista"].items():
        print(f"  {nome[:25]:25s} "
              f"{v['custo_recomendador_medio']:>8.2f} "
              f"{v['custo_aleatorio_medio']:>9.2f} "
              f"{v['poupanca_vs_aleatorio_eur']:>+9.2f} "
              f"{v['poupanca_vs_aleatorio_pct']:>+8.1f}% "
              f"{v['poupanca_vs_melhor_lealdade_pct']:>+14.1f}%")

    print("\n[BACKTEST] Global:")
    print(f"  Poupança média vs comprar aleatório       : "
          f"{resultado['global']['poupanca_vs_aleatorio_pct']:+.2f}% "
          f"({resultado['global']['poupanca_vs_aleatorio_eur']:+.2f}€/lista)")
    print(f"  Poupança média vs melhor cadeia única     : "
          f"{resultado['global']['poupanca_vs_melhor_lealdade_pct']:+.2f}%")


if __name__ == "__main__":
    main()
