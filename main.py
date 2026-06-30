"""
Ponto de entrada do sistema multiagente de preços de supermercado.

Inicia e gere o ciclo de vida de todos os agentes SPADE:

    Recolha de dados (Sensorização):
        AuchanScraper      — scraping periódico do Auchan Portugal (sitemap + AJAX SFCC)
        ContinenteScraper  — scraping periódico do Continente Online
        PingoDoceScraper   — scraping periódico do Pingo Doce (sitemap)

    Persistência:
        DatabaseAgent      — ingestão na BD e resposta a consultas

    Análise e IA (Aprendizagem Profunda + ASM):
        PredictionAgent    — previsão LSTM + retreino periódico
        RecommendationAgent— pesquisa, melhor loja, lista, momento de compra

    Monitorização (Sensorização e Ambiente):
        MonitorAgent       — monitor de preços + detetor de anomalias

    Coordenação (ASM):
        OrchestratorAgent  — pipeline pós-ingestão + coordenação geral

    Interface HTTP↔XMPP:
        UserInterfaceAgent — ponte entre a API REST e o sistema multiagente

    API REST (FastAPI + uvicorn):
        Exposta em http://localhost:<API_PORT> (default: 8000)
        Documentação: http://localhost:<API_PORT>/docs

Pré-requisitos:
    1. Servidor XMPP Openfire a correr: docker-compose up -d
    2. Contas XMPP criadas previamente via scripts/setup_openfire.py
       (os agentes arrancam com auto_register=False — não tentam criar contas
       ad-hoc, dependeriam do plugin "In-Band Registration" do Openfire)
    3. PostgreSQL a correr e schema inicializado (sql/schema.sql)
    4. Ficheiro .env preenchido (ver .env.example)

Uso:
    python main.py
    python main.py --retreinar
    python main.py --sem-predicao --sem-recomendacao
    python main.py --auchan-limite 200 --pingo-doce-limite 200
    python main.py --continente-categorias mercearia bebidas --continente-periodo 43200
    python main.py --sem-api                          # sem servidor HTTP
    python main.py --api-porta 9000                   # porta personalizada

Defaults aplicados quando não há flags explícitas:
    Auchan      : alimentacao (umbrella: mercearia, lacticínios, congelados, sabores)
    Pingo Doce  : 8 categorias do sitemap (mercearia + bolachas + iogurtes +
                  congelados + queijos + bebidas + padaria + café)
    Continente  : mercearia + bebidas + laticinios + congelados, 10 páginas/categoria
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ----- Silenciamento de warnings ruidosos do aioxmpp ---------------------------
# O Openfire envia pings IQ periódicos para keepalive da ligação XMPP.
# SPADE/aioxmpp não regista um handler dedicado para esses pings, e cada um gera
# uma linha "unhandleable IQ request: ... aioxmpp.ping.xso.Ping ...".
# São avisos cosméticos — a ligação continua viva e os agentes funcionam — mas
# enchem o terminal. Subimos o nível dos loggers relevantes para ERROR para os
# silenciar sem esconder erros reais.
logging.getLogger("aioxmpp").setLevel(logging.ERROR)
logging.getLogger("aioxmpp.dispatcher").setLevel(logging.ERROR)

from agents.AuchanScraper       import AuchanScraper
from agents.ContinenteScraper   import ContinenteScraper
from agents.PingoDoceScraper    import PingoDoceScraper
from agents.DatabaseAgent       import DatabaseAgent
from agents.PredictionAgent     import PredictionAgent
from agents.RecommendationAgent import RecommendationAgent
from agents.LocationAgent       import LocationAgent
from agents.MonitorAgent        import MonitorAgent
from agents.OrchestratorAgent   import OrchestratorAgent
from agents.UserInterfaceAgent  import UserInterfaceAgent
from api.app import app as fastapi_app, init_app

# Carrega variáveis do ficheiro .env na raiz do projeto
load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# JIDs e credenciais XMPP
# ---------------------------------------------------------------------------
# Criar estas contas em http://localhost:9090 → User Management → Create Account
# Configurar no ficheiro .env (ver .env.example para as variáveis disponíveis).

XMPP_SERVER = os.getenv("XMPP_SERVER", "localhost")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "password")
DB_JID = os.getenv("XMPP_DB_JID", f"database_agent@{XMPP_SERVER}")
AUCHAN_JID = os.getenv("XMPP_AUCHAN_JID", f"auchan_scraper@{XMPP_SERVER}")
CONTINENTE_JID = os.getenv("XMPP_CONTINENTE_JID", f"continente_scraper@{XMPP_SERVER}")
PINGO_DOCE_JID = os.getenv("XMPP_PINGO_DOCE_JID", f"pingo_doce_scraper@{XMPP_SERVER}")
PREDICTION_JID = os.getenv("XMPP_PREDICTION_JID", f"prediction_agent@{XMPP_SERVER}")
RECOMMENDATION_JID = os.getenv("XMPP_RECOMMENDATION_JID", f"recommendation_agent@{XMPP_SERVER}")
LOCATION_JID = os.getenv("XMPP_LOCATION_JID", f"location_agent@{XMPP_SERVER}")
MONITOR_JID = os.getenv("XMPP_MONITOR_JID", f"monitor_agent@{XMPP_SERVER}")
ORCHESTRATOR_JID = os.getenv("XMPP_ORCHESTRATOR_JID", f"orchestrator_agent@{XMPP_SERVER}")
UI_JID = os.getenv("XMPP_UI_JID", f"ui_agent@{XMPP_SERVER}")

API_HOST = os.getenv("API_HOST", "0.0.0.0")


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    """Inicia todos os agentes e mantém o sistema em execução."""

    # ------------------------------------------------------------------
    # Validar combinações de flags problemáticas
    # ------------------------------------------------------------------

    if args.retreinar and args.sem_predicao:
        print(
            "[AVISO] --retreinar foi passado mas o PredictionAgent está desativado "
            "(--sem-predicao). O retreino automático não terá efeito."
        )

    if args.sem_api and args.api_porta != 8000:
        print("[AVISO] --api-porta ignorado porque --sem-api foi especificado.")

    # ------------------------------------------------------------------
    # Instanciar agentes
    # ------------------------------------------------------------------

    orchestrator_agent = OrchestratorAgent(
        ORCHESTRATOR_JID,
        XMPP_PASSWORD,
        prediction_agent_jid = PREDICTION_JID if not args.sem_predicao else None,
    )
    orchestrator_agent.config["retreinar"] = args.retreinar

    db_agent = DatabaseAgent(
        DB_JID,
        XMPP_PASSWORD,
        orchestrator_jid = ORCHESTRATOR_JID,
    )

    auchan_agent = AuchanScraper(AUCHAN_JID, XMPP_PASSWORD, DB_JID) \
        if not args.sem_scrapers else None
    continente_agent = ContinenteScraper(CONTINENTE_JID, XMPP_PASSWORD, DB_JID) \
        if not args.sem_scrapers else None
    pingo_doce_agent = PingoDoceScraper(PINGO_DOCE_JID, XMPP_PASSWORD, DB_JID) \
        if not args.sem_scrapers else None

    prediction_agent = PredictionAgent(PREDICTION_JID, XMPP_PASSWORD) \
        if not args.sem_predicao else None

    recommendation_agent = RecommendationAgent(RECOMMENDATION_JID, XMPP_PASSWORD) \
        if not args.sem_recomendacao else None

    location_agent = LocationAgent(LOCATION_JID, XMPP_PASSWORD) \
        if not args.sem_localizacao else None

    monitor_agent = MonitorAgent(
        MONITOR_JID,
        XMPP_PASSWORD,
        orchestrator_jid = ORCHESTRATOR_JID,
    )

    ui_agent = UserInterfaceAgent(
        UI_JID,
        XMPP_PASSWORD,
        recommendation_jid = RECOMMENDATION_JID,
        prediction_jid = PREDICTION_JID,
        database_jid = DB_JID,
        location_jid = LOCATION_JID if not args.sem_localizacao else None,
    ) if not args.sem_api else None

    # ------------------------------------------------------------------
    # Aplicar configurações da linha de comandos antes de arrancar
    # ------------------------------------------------------------------

    # Scrapers — Auchan
    if auchan_agent:
        if args.auchan_categoria:
            auchan_agent.config["categoria"] = args.auchan_categoria
        if args.auchan_subcategoria:
            auchan_agent.config["subcategoria"] = args.auchan_subcategoria
        # Limite default de 500 para equilibrar volume com os outros scrapers.
        # Com shuffle activado, cada ciclo cobre uma amostra diferente do catálogo.
        auchan_agent.config["limite"] = args.auchan_limite if args.auchan_limite else 500
        if args.auchan_periodo:
            auchan_agent.config["periodo"] = args.auchan_periodo

    # Scrapers — Continente
    if continente_agent:
        # Por omissão scrapa as 4 categorias principais (mercearia + bebidas +
        # lacticínios + congelados). Sem flag explícita o agente ficaria
        # inativo porque o DEFAULT_CONFIG do agente tem ``categorias: []``.
        continente_agent.config["categorias"] = (
            args.continente_categorias
            or ["mercearia", "bebidas", "laticinios", "congelados"]
        )
        if args.continente_queries:
            continente_agent.config["queries"] = [
                q.strip() for q in args.continente_queries.split(",")
            ]
        if args.continente_periodo:
            continente_agent.config["periodo"] = args.continente_periodo

    # Scrapers — Pingo Doce
    if pingo_doce_agent:
        if args.pingo_doce_categoria:
            pingo_doce_agent.config["categoria"] = args.pingo_doce_categoria
        if args.pingo_doce_subcategoria:
            pingo_doce_agent.config["subcategoria"] = args.pingo_doce_subcategoria
        # Limite default de 500 para equilibrar volume com os outros scrapers.
        pingo_doce_agent.config["limite"] = args.pingo_doce_limite if args.pingo_doce_limite else 500
        if args.pingo_doce_periodo:
            pingo_doce_agent.config["periodo"] = args.pingo_doce_periodo

    # Monitor
    if args.monitor_periodo:
        monitor_agent.config["periodo"] = args.monitor_periodo

    # ------------------------------------------------------------------
    # Registo central de agentes — ordem de arranque é determinada por esta
    # lista. O OrchestratorAgent é o primeiro (precisa de estar à escuta
    # antes de receber eventos); depois DatabaseAgent; depois scrapers;
    # depois os restantes (Monitor, Prediction, Recommendation, Location, UI).
    # Agentes opcionais aparecem como ``None`` e são filtrados.
    # ------------------------------------------------------------------
    agentes_registo: list[tuple[str, str, object]] = [
        ("OrchestratorAgent",  ORCHESTRATOR_JID,    orchestrator_agent),
        ("DatabaseAgent",      DB_JID,              db_agent),
        ("AuchanScraper",      AUCHAN_JID,          auchan_agent),
        ("ContinenteScraper",  CONTINENTE_JID,      continente_agent),
        ("PingoDoceScraper",   PINGO_DOCE_JID,      pingo_doce_agent),
        ("MonitorAgent",       MONITOR_JID,         monitor_agent),
        ("PredictionAgent",    PREDICTION_JID,      prediction_agent),
        ("RecommendationAgent", RECOMMENDATION_JID, recommendation_agent),
        ("LocationAgent",      LOCATION_JID,        location_agent),
        ("UserInterfaceAgent", UI_JID,              ui_agent),
    ]
    agentes_ativos = [(label, jid, ag) for label, jid, ag in agentes_registo if ag is not None]

    # ------------------------------------------------------------------
    # Arrancar todos os agentes activos pela ordem do registo
    # ------------------------------------------------------------------
    for _label, _jid, ag in agentes_ativos:
        await ag.start(auto_register=False)

    # ------------------------------------------------------------------
    # Resumo de arranque
    # ------------------------------------------------------------------
    porta = args.api_porta if not args.sem_api else None
    print("\n" + "=" * 65)
    print("  Sistema multiagente iniciado.")
    for label, jid, ag in agentes_registo:
        valor = jid if ag is not None else "[desativado]"
        print(f"  {label:<19} : {valor}")
    if ui_agent:
        print(f"  {'API REST':<19} : http://{API_HOST}:{porta}/docs")
    else:
        print(f"  {'API REST':<19} : [desativada]")
    print(f"  {'Retreino automático':<19} : {'sim' if args.retreinar else 'não'}")
    print("  Prima Ctrl+C para parar.")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Loop principal — uvicorn (API ativa) ou loop simples (--sem-api)
    # ------------------------------------------------------------------
    agentes_a_parar = [ag for _label, _jid, ag in agentes_ativos]

    try:
        if ui_agent:
            import uvicorn

            # Injetar o agente na app FastAPI antes de aceitar pedidos
            init_app(ui_agent)

            # loop="none" usa o event loop asyncio já ativo (partilhado com SPADE)
            config = uvicorn.Config(
                fastapi_app,
                host      = API_HOST,
                port      = args.api_porta,
                log_level = "info",
                loop      = "none",
            )
            server = uvicorn.Server(config)
            await server.serve()
            # serve() regressa quando o servidor é parado (Ctrl+C ou sinal)
        else:
            while True:
                await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nA parar o sistema multiagente...")
    finally:
        await asyncio.gather(
            *[a.stop() for a in agentes_a_parar],
            return_exceptions=True,
        )
        await asyncio.sleep(0.5)
        print("Sistema parado.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Constrói o parser CLI e devolve os argumentos da linha de comandos."""
    parser = argparse.ArgumentParser(
        description="Sistema multiagente de análise de preços de supermercado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Arrancar com defaults (todas as funcionalidades, scraping ativo dos 3 supermercados)
  python main.py

  # Retreino automático do LSTM após cada ingestão
  python main.py --retreinar

  # Teste rápido sem modelos de IA (só scrapers + BD + monitor)
  python main.py --sem-predicao --sem-recomendacao \\
                 --auchan-limite 50 --pingo-doce-limite 50

  # Limites personalizados por supermercado
  python main.py --auchan-limite 200 --pingo-doce-limite 200 \\
                 --continente-categorias mercearia

  # Scrapers a cada 12h, monitor a cada 3h
  python main.py --auchan-periodo 43200 --pingo-doce-periodo 43200 \\
                 --monitor-periodo 10800
        """,
    )

    # --- Scrapers: Auchan ---
    grp_a = parser.add_argument_group("Auchan")
    grp_a.add_argument("--auchan-categoria", type=str, metavar="CAT",
        help="Categoria a recolher (ex: alimentacao, bebidas). Default: alimentacao.")
    grp_a.add_argument("--auchan-subcategoria", type=str, metavar="SUBCAT",
        help="Subcategoria opcional (ex: arroz, massas).")
    grp_a.add_argument("--auchan-limite", type=int, metavar="N",
        help="Limite de produtos a recolher (útil para testes).")
    grp_a.add_argument("--auchan-periodo", type=int, metavar="SEGUNDOS",
        help="Período de scraping em segundos (default: 21600 = 6h).")

    # --- Scrapers: Continente ---
    grp_c = parser.add_argument_group("Continente")
    grp_c.add_argument("--continente-categorias", nargs="+",
        choices=["mercearia", "bebidas", "laticinios", "congelados"], metavar="CAT",
        help="Categorias a recolher (default: mercearia, bebidas, laticinios, congelados).")
    grp_c.add_argument("--continente-queries", type=str, metavar="TERMOS",
        help="Termos de pesquisa separados por vírgula (ex: 'arroz,massa').")
    grp_c.add_argument("--continente-periodo", type=int, metavar="SEGUNDOS",
        help="Período de scraping em segundos (default: 21600 = 6h).")

    # --- Scrapers: Pingo Doce ---
    grp_pd = parser.add_argument_group("Pingo Doce")
    grp_pd.add_argument("--pingo-doce-categoria", type=str, metavar="CAT",
        help="Categoria de produto (default: mercearia).")
    grp_pd.add_argument("--pingo-doce-subcategoria", type=str, metavar="SUBCAT",
        help="Subcategoria opcional (ex: arroz, conservas).")
    grp_pd.add_argument("--pingo-doce-limite", type=int, metavar="N",
        help="Limite de produtos a recolher (útil para testes).")
    grp_pd.add_argument("--pingo-doce-periodo", type=int, metavar="SEGUNDOS",
        help="Período de scraping em segundos (default: 21600 = 6h).")

    # --- Monitor ---
    grp_m = parser.add_argument_group("Monitor")
    grp_m.add_argument("--monitor-periodo", type=int, metavar="SEGUNDOS",
        help="Período de monitorização em segundos (default: 21600 = 6h).")

    # --- Pipeline / IA ---
    grp_ia = parser.add_argument_group("Pipeline e IA")
    grp_ia.add_argument("--sem-scrapers", action="store_true",
        help="Não iniciar os scrapers (útil para consultar a API com dados já existentes).")
    grp_ia.add_argument("--retreinar", action="store_true",
        help="Retreinar o LSTM automaticamente após cada ingestão.")
    grp_ia.add_argument("--sem-predicao", action="store_true",
        help="Não iniciar o PredictionAgent (útil para testes ou máquinas sem GPU).")
    grp_ia.add_argument("--sem-recomendacao", action="store_true",
        help="Não iniciar o RecommendationAgent.")
    grp_ia.add_argument("--sem-localizacao", action="store_true",
        help="Não iniciar o LocationAgent (desactiva custo de deslocação por GPS).")

    # --- API REST ---
    grp_api = parser.add_argument_group("API REST")
    grp_api.add_argument("--sem-api", action="store_true",
        help="Não iniciar o servidor HTTP (UserInterfaceAgent + uvicorn).")
    grp_api.add_argument("--api-porta", type=int, default=int(os.getenv("API_PORT", "8000")),
        metavar="PORTA",
        help="Porta do servidor HTTP (default: %(default)s; sobrepõe API_PORT do .env).")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        sys.exit(0)
