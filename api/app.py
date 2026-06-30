"""
API REST do sistema multiagente de análise de preços de supermercado.

Serve de interface HTTP para o sistema SPADE, delegando todas as consultas
aos agentes especializados via XMPP (através do UserInterfaceAgent).

Todos os endpoints são assíncronos e aguardam a resposta do agente antes de
devolver o resultado ao cliente. Erros do agente (performative "failure")
são mapeados para respostas HTTP 500; timeouts para 504.

Arranque:
    Iniciada automaticamente por ``main.py`` em conjunto com os agentes SPADE.
    A instância do UserInterfaceAgent é injetada via ``init_app()`` antes de
    o servidor uvicorn começar a aceitar pedidos.

Documentação interativa:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NoReturn

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agents.UserInterfaceAgent import UserInterfaceAgent


# ---------------------------------------------------------------------------
# Estado global — injetado por main.py antes de arrancar o servidor
# ---------------------------------------------------------------------------

_agent: UserInterfaceAgent | None = None


def init_app(agent: UserInterfaceAgent) -> None:
    """Injeta o UserInterfaceAgent na aplicação FastAPI.

    Deve ser chamado por ``main.py`` depois de o agente estar inicializado
    e antes de iniciar o servidor uvicorn.

    Args:
        agent: Instância já arrancada do UserInterfaceAgent.
    """
    global _agent
    _agent = agent


def _ag() -> UserInterfaceAgent:
    """Devolve o agente ou levanta 503 se ainda não inicializado."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sistema multiagente ainda não inicializado.",
        )
    return _agent


def _tratar_erro(exc: Exception, agente: str) -> NoReturn:
    """Converte excepções de agente em respostas HTTP adequadas e levanta sempre."""
    if isinstance(exc, asyncio.TimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"{agente} não respondeu dentro do tempo limite.",
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(exc),
    )


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SuperMarket Price Analysis",
    description=(
        "API REST do sistema multiagente de análise de preços de supermercado.\n\n"
        "Todos os pedidos são delegados via XMPP aos agentes SPADE especializados:\n"
        "- **RecommendationAgent** — pesquisa, comparação e otimização de compras\n"
        "- **PredictionAgent** — previsão LSTM + Monte Carlo Dropout\n"
        "- **DatabaseAgent** — histórico de preços e preços atuais por loja"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS permissivo para desenvolvimento — restringir em produção
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class ListaCompras(BaseModel):
    """Corpo do pedido de otimização de lista de compras."""

    lista: list[str] = Field(
        ...,
        min_length=1,
        examples=[["arroz agulha", "azeite virgem extra", "atum natural"]],
        description="Produtos a otimizar (nomes em linguagem natural).",
    )


class ListaComprasGeo(BaseModel):
    """Corpo do pedido de otimização de lista com custo de deslocação (GPS).

    A localização do utilizador (lat/lon WGS84) é obrigatória. ``custo_km``
    aceita um preset (``"so_combustivel"``, ``"equilibrado"``, ``"tarifa_at"``)
    ou um valor numérico em €/km.

    ``multi_loja=True`` avalia também a divisão entre 2 cadeias (rota triangular
    ``user → A → B → user``), recomendando split só se for estritamente melhor
    que a melhor opção single-store.
    """

    lista: list[str] = Field(
        ..., min_length=1,
        examples=[["arroz agulha", "azeite virgem extra"]],
        description="Produtos a otimizar.",
    )
    lat: float = Field(..., ge=-90, le=90, description="Latitude WGS84 (graus).")
    lon: float = Field(..., ge=-180, le=180, description="Longitude WGS84 (graus).")
    custo_km: str | float | None = Field(
        default=None,
        examples=["equilibrado", 0.20],
        description="Preset ('so_combustivel'/'equilibrado'/'tarifa_at') ou €/km. Default: equilibrado.",
    )
    raio_km: float = Field(
        default=30.0, gt=0, le=500,
        description="Raio máximo (km) para considerar uma loja física alcançável.",
    )
    multi_loja: bool = Field(
        default=False,
        description="Se True, avalia também a divisão da lista entre 2 cadeias "
                    "(rota triangular). Default: False (só single-store).",
    )


# ---------------------------------------------------------------------------
# Sistema — estado e saúde
# ---------------------------------------------------------------------------

@app.get(
    "/saude",
    tags=["Sistema"],
    summary="Estado da API e do sistema multiagente",
)
async def saude():
    """Devolve o estado da API e os JIDs dos agentes ligados (versão leve)."""
    ag = _ag()
    return {
        "status": "ok",
        "agentes": {
            "recommendation": ag.recommendation_jid,
            "prediction":     ag.prediction_jid,
            "database":       ag.database_jid,
            "location":       ag.location_jid,   # None se sistema arrancou --sem-localizacao
        },
    }


@app.get(
    "/saude/completo",
    tags=["Sistema"],
    summary="Diagnóstico completo do sistema (BD, modelo, dataset, lojas)",
)
async def saude_completo():
    """Diagnóstico operacional do sistema.

    Verifica todas as dependências críticas e devolve estado individual de cada
    componente. Pensado para monitorização/demonstração — abrir esta página
    permite confirmar que tudo está em condições de produção.

    Devolve sempre 200, mesmo com falhas individuais (cada componente tem o seu
    ``status``). O campo top-level ``status`` é ``"ok"`` apenas se todos os
    componentes essenciais (BD, agentes XMPP) estiverem operacionais.
    """
    from datetime import datetime
    from pathlib import Path

    import psycopg2

    from scripts.db_config import DB_CONFIG

    ag = _ag()
    componentes: dict = {}

    # 1. Base de dados — ligação + contagem de registos
    try:
        with psycopg2.connect(connect_timeout=3, **DB_CONFIG) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM produtos_loja")
            n_produtos = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM historico_precos")
            n_historico = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM lojas_fisicas WHERE ativa = TRUE")
            n_lojas_fisicas = cur.fetchone()[0]
            cur.execute("SELECT MAX(data_recolha) FROM precos_atuais")
            ultima_recolha = cur.fetchone()[0]
        componentes["base_dados"] = {
            "status":            "ok",
            "produtos_loja":     n_produtos,
            "historico_precos":  n_historico,
            "lojas_fisicas":     n_lojas_fisicas,
            "ultima_recolha":    str(ultima_recolha) if ultima_recolha else None,
        }
    except Exception as exc:
        componentes["base_dados"] = {"status": "erro", "detalhe": str(exc)}

    # 2. Modelo LSTM — existência de artefactos + nº de produtos treinados
    try:
        models_dir = Path(__file__).resolve().parent.parent / "models" / "saved_models"
        ficheiros = {
            "pesos":   models_dir / "lstm_global.pt",
            "scalers": models_dir / "scalers.pkl",
            "meta":    models_dir / "model_meta.pkl",
        }
        em_falta = [nome for nome, p in ficheiros.items() if not p.exists()]
        if em_falta:
            componentes["modelo_lstm"] = {
                "status":   "nao_treinado",
                "em_falta": em_falta,
                "dica":     "Corre: python models/price_predictor.py --treinar",
            }
        else:
            import pickle
            with open(ficheiros["meta"], "rb") as f:
                meta = pickle.load(f)
            mtime = ficheiros["pesos"].stat().st_mtime
            componentes["modelo_lstm"] = {
                "status":              "ok",
                "produtos_treinados":  len(meta.get("produtos_treinados", [])),
                "janela_dias":         meta.get("janela"),
                "treinado_em":         datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            }
    except Exception as exc:
        componentes["modelo_lstm"] = {"status": "erro", "detalhe": str(exc)}

    # 3. Dataset de forecasting — existência + frescura
    try:
        dataset = Path(__file__).resolve().parent.parent / "data/generated/forecasting_dataset.csv"
        if not dataset.exists():
            componentes["dataset"] = {
                "status": "ausente",
                "dica":   "Corre: python scripts/build_forecasting_dataset.py",
            }
        else:
            mtime = dataset.stat().st_mtime
            tamanho_mb = round(dataset.stat().st_size / 1024 / 1024, 2)
            componentes["dataset"] = {
                "status":       "ok",
                "tamanho_mb":   tamanho_mb,
                "atualizado":   datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            }
    except Exception as exc:
        componentes["dataset"] = {"status": "erro", "detalhe": str(exc)}

    # 4. Agentes XMPP
    componentes["agentes"] = {
        "status":         "ok",
        "recommendation": ag.recommendation_jid,
        "prediction":     ag.prediction_jid,
        "database":       ag.database_jid,
        "location":       ag.location_jid,
    }

    # Estado global: "ok" apenas se BD e agentes estão operacionais
    ok = (
        componentes["base_dados"]["status"] == "ok"
        and componentes["agentes"]["status"] == "ok"
    )
    return {
        "status":      "ok" if ok else "degradado",
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "componentes": componentes,
    }


# ---------------------------------------------------------------------------
# Produtos — pesquisa e comparação de preços
# ---------------------------------------------------------------------------

@app.get(
    "/produtos/pesquisar",
    tags=["Produtos"],
    summary="Pesquisar produtos por palavra-chave",
)
async def pesquisar_produtos(
    termo: str,
    limite: int = 20,
):
    """Pesquisa produtos pelo nome usando similaridade de texto (pg_trgm).

    Devolve os produtos mais relevantes com preço atual e loja.

    - **termo**: palavra-chave de pesquisa (ex: `arroz`, `azeite virgem`)
    - **limite**: número máximo de resultados (default: 20)
    """
    try:
        return await _ag().recomendar(
            "pesquisar", {"termo": termo, "limite": limite}
        )
    except Exception as exc:
        _tratar_erro(exc, "RecommendationAgent")


@app.get(
    "/produtos/melhor-loja",
    tags=["Produtos"],
    summary="Melhor loja para um produto",
)
async def melhor_loja(
    nome: str,
    top_n: int = 5,
):
    """Devolve as melhores ofertas para um produto, ordenadas por preço.

    Usa similaridade de texto para identificar o produto mais relevante
    e compara o seu preço entre todas as lojas disponíveis na BD.

    - **nome**: nome do produto (ex: `arroz agulha`, `leite meio-gordo`)
    - **top_n**: número de resultados a devolver (default: 5)
    """
    try:
        return await _ag().recomendar(
            "melhor_loja", {"produto": nome, "top_n": top_n}
        )
    except Exception as exc:
        _tratar_erro(exc, "RecommendationAgent")


# ---------------------------------------------------------------------------
# Compras — otimização e momento de compra
# ---------------------------------------------------------------------------

@app.post(
    "/compras/otimizar",
    tags=["Compras"],
    summary="Otimizar lista de compras entre lojas",
)
async def otimizar_lista(body: ListaCompras):
    """Distribui os itens da lista pelas lojas mais baratas.

    Para cada produto da lista, encontra o melhor preço disponível na BD.
    O resultado indica em que loja comprar cada produto para minimizar
    o custo total da lista.
    """
    try:
        return await _ag().recomendar(
            "otimizar_lista", {"lista": body.lista}
        )
    except Exception as exc:
        _tratar_erro(exc, "RecommendationAgent")


@app.get(
    "/compras/momento/{produto_id}",
    tags=["Compras"],
    summary="Recomendar momento de compra (comprar agora ou esperar?)",
)
async def momento_compra(
    produto_id: int,
    horizonte: int = 7,
    limiar: float = 2.0,
    amostras: int = 50,
):
    """Recomenda comprar agora ou esperar, com base em previsão LSTM.

    Usa Monte Carlo Dropout para quantificar a incerteza da previsão e
    decide se a descida esperada justifica adiar a compra.

    - **produto_id**: ``id_produto_loja`` (obtido via `/produtos/pesquisar`)
    - **horizonte**: dias a prever (default: 7)
    - **limiar**: descida mínima (%) para recomendar espera (default: 2.0)
    - **amostras**: simulações Monte Carlo (default: 50)
    """
    try:
        return await _ag().recomendar(
            "momento_compra",
            {
                "produto_id": produto_id,
                "horizonte": horizonte,
                "limiar": limiar,
                "amostras": amostras,
            },
        )
    except Exception as exc:
        _tratar_erro(exc, "RecommendationAgent")


# ---------------------------------------------------------------------------
# Previsão de preços
# ---------------------------------------------------------------------------

@app.get(
    "/previsao/{produto_id}",
    tags=["Previsão"],
    summary="Previsão determinista de preço (LSTM)",
)
async def prever_preco(
    produto_id: int,
    horizonte: int = 7,
):
    """Previsão determinista de preço para os próximos N dias.

    Usa o modelo LSTM global em modo ``eval()`` (Dropout desativado).
    Para previsões com intervalos de confiança, usa `/previsao/{id}/incerteza`.

    - **produto_id**: ``id_produto_loja``
    - **horizonte**: dias a prever (default: 7)

    Devolve uma lista de ``{data, preco_previsto}`` por dia.
    """
    try:
        return await _ag().prever(
            "prever", {"produto_id": produto_id, "horizonte": horizonte}
        )
    except Exception as exc:
        _tratar_erro(exc, "PredictionAgent")


@app.get(
    "/previsao/{produto_id}/incerteza",
    tags=["Previsão"],
    summary="Previsão com intervalos de confiança (Monte Carlo Dropout)",
)
async def prever_preco_com_incerteza(
    produto_id: int,
    horizonte: int = 7,
    amostras: int = 50,
):
    """Previsão probabilística com intervalos de confiança.

    Mantém o Dropout ativo durante a inferência e corre N passes forward
    independentes. A variação entre simulações captura a incerteza epistémica.

    - **produto_id**: ``id_produto_loja``
    - **horizonte**: dias a prever (default: 7)
    - **amostras**: simulações Monte Carlo (default: 50)

    Devolve previsão média, desvio-padrão e intervalos 5%–95% por dia.
    """
    try:
        return await _ag().prever(
            "prever_mc",
            {"produto_id": produto_id, "horizonte": horizonte, "amostras": amostras},
        )
    except Exception as exc:
        _tratar_erro(exc, "PredictionAgent")


# ---------------------------------------------------------------------------
# Geolocalização e custo de deslocação (via LocationAgent e RecommendationAgent)
# ---------------------------------------------------------------------------

@app.get(
    "/lojas-fisicas/proximas",
    tags=["Geolocalização"],
    summary="Lojas físicas próximas de uma localização GPS",
)
async def lojas_proximas(
    lat: float,
    lon: float,
    raio_km: float = 20.0,
    insignia: str | None = None,
    limite: int = 50,
):
    """Devolve as lojas físicas das cadeias dentro de um raio, ordenadas por distância.

    Usa a fórmula de Haversine sobre as coordenadas GPS guardadas em `lojas_fisicas`.

    - **lat**, **lon**: coordenadas do utilizador em graus WGS84.
    - **raio_km**: raio máximo (default: 20 km).
    - **insignia**: filtrar por uma cadeia específica (``Continente``, ``Pingo Doce``, ``Auchan``).
    - **limite**: número máximo de lojas a devolver.

    Devolve lista de ``{id_loja_fisica, insignia, nome_loja, morada, cidade,
    codigo_postal, latitude, longitude, distancia_km}``.
    """
    try:
        return await _ag().localizar(
            "lojas_proximas",
            {"lat": lat, "lon": lon, "raio_km": raio_km,
             "insignia": insignia, "limite": limite},
        )
    except Exception as exc:
        _tratar_erro(exc, "LocationAgent")


@app.post(
    "/compras/otimizar-geo",
    tags=["Geolocalização"],
    summary="Otimizar lista de compras incluindo custo de deslocação",
)
async def otimizar_lista_geo(body: ListaComprasGeo):
    """Otimiza a lista de compras considerando o custo total = produtos + deslocação.

    Para cada cadeia com a lista completa disponível, identifica a loja física
    mais próxima do utilizador e soma o custo de deslocação ida-volta ao custo
    dos produtos. A recomendação é a cadeia com menor custo total.

    O custo por km pode ser:

    - Um preset: ``"so_combustivel"`` (0.12€/km), ``"equilibrado"`` (0.20€/km, default),
      ``"tarifa_at"`` (0.36€/km — Portaria 1553-D/2008).
    - Um valor numérico em €/km (ex: 0.18).

    Quando ``multi_loja=True``, avalia também a divisão da lista entre 2 cadeias
    (rota triangular ``user → A → B → user``). A resposta inclui:

    - ``melhor_par``: combinação de 2 cadeias com menor custo total (se viável).
    - ``todos_os_pares``: lista de todos os pares avaliados.
    - ``recomendacao``: ``"single"`` ou ``"par"`` consoante o que tem menor custo.
    - ``poupanca_par``: € poupados ao dividir vs. single-store.

    Devolve uma comparação completa entre cadeias permitindo ao utilizador
    ver o trade-off entre preço dos produtos e distância a percorrer.
    """
    try:
        return await _ag().recomendar(
            "otimizar_lista_geo",
            {"lista":      body.lista,
             "lat":        body.lat,
             "lon":        body.lon,
             "custo_km":   body.custo_km,
             "raio_km":    body.raio_km,
             "multi_loja": body.multi_loja},
        )
    except Exception as exc:
        _tratar_erro(exc, "RecommendationAgent")


@app.get(
    "/custo-deslocacao",
    tags=["Geolocalização"],
    summary="Calcular custo monetário de uma deslocação",
)
async def custo_deslocacao(
    distancia_km: float,
    custo_km: str | float | None = None,
    ida_e_volta: bool = True,
):
    """Converte uma distância numa estimativa de custo monetário.

    - **distancia_km**: distância one-way em km.
    - **custo_km**: preset (``"so_combustivel"``/``"equilibrado"``/``"tarifa_at"``)
      ou valor em €/km. Default: equilibrado (0.20 €/km).
    - **ida_e_volta**: se ``true`` (default), multiplica por 2.

    Devolve ``{distancia_km, custo_km, ida_e_volta, custo_euros}``.
    """
    try:
        return await _ag().localizar(
            "custo_deslocacao",
            {"distancia_km": distancia_km, "custo_km": custo_km,
             "ida_e_volta": ida_e_volta},
        )
    except Exception as exc:
        _tratar_erro(exc, "LocationAgent")


@app.get(
    "/produtos/perto-de-mim",
    tags=["Geolocalização"],
    summary="Produto ordenado por custo total (preço + deslocação à loja física mais próxima)",
)
async def produto_perto_de_mim(
    nome: str,
    lat: float,
    lon: float,
    custo_km: str | float | None = None,
    raio_km: float = 30.0,
    top_n: int = 5,
):
    """Encontra um produto nas cadeias, cruza com lojas físicas e ordena por custo total.

    Diferente de ``/compras/otimizar-geo`` (que recomenda uma cadeia para uma
    **lista**) — este endpoint trabalha com **um único produto** e devolve
    **todas** as cadeias alcançáveis, ordenadas por **custo total** = preço +
    deslocação ida-volta. Permite ao utilizador comparar pessoalmente as
    opções e decidir.

    - **nome**: termo de pesquisa (ex: ``"arroz agulha"``, ``"azeite virgem extra"``)
    - **lat**, **lon**: coordenadas GPS do utilizador (WGS84)
    - **custo_km**: preset (``"so_combustivel"``/``"equilibrado"``/``"tarifa_at"``)
      ou valor numérico em €/km. Default: equilibrado (0.20).
    - **raio_km**: raio máximo (km) para considerar uma cadeia alcançável.
    - **top_n**: número máximo de resultados.

    Devolve lista de ``{insignia, produto, preco_atual, em_promocao,
    preco_unitario_valor, loja_fisica, distancia_km, custo_deslocacao,
    custo_total}``, ordenada por ``custo_total`` ascendente.
    """
    try:
        return await _ag().recomendar(
            "produto_perto_de_mim",
            {"termo":    nome,
             "lat":      lat,
             "lon":      lon,
             "custo_km": custo_km,
             "raio_km":  raio_km,
             "top_n":    top_n},
        )
    except Exception as exc:
        _tratar_erro(exc, "RecommendationAgent")


@app.get(
    "/custo-deslocacao/presets",
    tags=["Geolocalização"],
    summary="Listar presets de €/km disponíveis",
)
async def custo_deslocacao_presets():
    """Devolve os presets de €/km com label e descrição.

    Os valores são fixos no código (não vêm da BD). Útil para a UI montar
    um dropdown/radio com as opções disponíveis e a sua explicação.
    """
    try:
        return await _ag().localizar("presets_custo_km", {})
    except Exception as exc:
        _tratar_erro(exc, "LocationAgent")


# ---------------------------------------------------------------------------
# Dados — histórico e preços atuais (via DatabaseAgent)
# ---------------------------------------------------------------------------

@app.get(
    "/historico/{produto_id}",
    tags=["Dados"],
    summary="Histórico de preços de um produto",
)
async def historico_produto(
    produto_id: int,
    dias: int = 30,
):
    """Histórico de preços de um produto-loja nos últimos N dias.

    - **produto_id**: ``id_produto_loja``
    - **dias**: janela temporal em dias (default: 30)

    Devolve uma lista de ``{data, preco, em_promocao}`` por observação.
    """
    try:
        return await _ag().consultar_bd(
            "historico_produto",
            {"id_produto_loja": produto_id, "dias": dias},
        )
    except Exception as exc:
        _tratar_erro(exc, "DatabaseAgent")


@app.get(
    "/lojas/{insignia}/precos",
    tags=["Dados"],
    summary="Preços atuais de todos os produtos de uma loja",
)
async def precos_atuais_loja(insignia: str):
    """Devolve os preços atuais de todos os produtos de uma loja.

    - **insignia**: nome da cadeia — ``Continente``, ``Pingo Doce`` ou ``Auchan``

    Devolve uma lista de ``{nome, preco, em_promocao}``.
    """
    try:
        return await _ag().consultar_bd(
            "precos_atuais_loja", {"insignia": insignia}
        )
    except Exception as exc:
        _tratar_erro(exc, "DatabaseAgent")
