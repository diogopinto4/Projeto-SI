"""
Agente SPADE para scraping periódico do Continente Online.

Comportamentos:
    - ReceiveBehaviour (CyclicBehaviour): recebe reconfiguração dinâmica
      via mensagem FIPA-ACL com performative "inform".
    - ScrapeBehaviour (PeriodicBehaviour): executa scraping periodicamente
      e envia o batch de produtos ao DatabaseAgent via "request".

Protocolo de mensagens:
    Recebe (de qualquer agente):
        performative: inform
        body: dict com nova configuração (jsonpickle)

    Envia (para DatabaseAgent):
        performative: request
        body: {"id_loja", "data_extracao", "produtos": [...]} (jsonpickle)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import jsonpickle
import requests
from spade import agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message

# Adiciona a raiz do projecto ao path para importar scrapers/
sys.path.insert(0, str(Path(__file__).parent.parent))
import scrapers.continente_scraper as cs


class ContinenteScraper(agent.Agent):
    """Agente scraper do Continente Online.

    Executa scraping periódico das categorias/queries configuradas e envia
    os dados recolhidos ao DatabaseAgent para ingestão na BD.
    """

    DEFAULT_CONFIG: dict = {
        "id_loja": "Continente",   # deve corresponder ao campo 'loja' do schema
        "queries": [],             # ex: ["arroz", "massa"]
        "categorias": [],          # ex: ["mercearia", "bebidas"]
        "paginas": 10,             # páginas por tarefa de scraping
        "periodo": 21600,          # 6 horas em segundos
        "ativo": True,
        "debug": False,
    }

    def __init__(self, jid: str, password: str, db_agent_jid: str) -> None:
        super().__init__(jid, password)
        self.db_agent_jid = db_agent_jid
        self.config = dict(self.DEFAULT_CONFIG)
        self.scrape_behaviour: PeriodicBehaviour | None = None
        # Sessão HTTP partilhada entre ciclos — criada em setup() e reutilizada
        # para evitar o overhead de handshake TLS em cada ciclo de scraping.
        self.session: requests.Session | None = None

    # ------------------------------------------------------------------
    # Comportamento de receção de reconfiguração dinâmica
    # ------------------------------------------------------------------

    class ReceiveBehaviour(CyclicBehaviour):
        """Aguarda mensagens de reconfiguração do agente.

        Aceita apenas mensagens com ``performative=inform`` e body que descodifique
        para um dicionário. Mensagens malformadas são ignoradas com aviso para
        não derrubar o behaviour.
        """

        async def run(self) -> None:
            msg = await self.receive(timeout=10)
            if not msg:
                return

            if msg.get_metadata("performative") != "inform":
                return

            try:
                new_config = jsonpickle.decode(msg.body)
                if not isinstance(new_config, dict):
                    return  # ignora payloads que não são config (ex: relatórios)
                print(f"[ContinenteScraper] Nova configuração recebida: {new_config}")
                self.agent.config.update(new_config)

                # Reiniciar ScrapeBehaviour com o novo período
                if self.agent.scrape_behaviour:
                    self.agent.remove_behaviour(self.agent.scrape_behaviour)

                periodo = self.agent.config.get("periodo", self.agent.DEFAULT_CONFIG["periodo"])
                new_b = self.agent.ScrapeBehaviour(period=periodo)
                self.agent.scrape_behaviour = new_b
                self.agent.add_behaviour(new_b)
                print("[ContinenteScraper] ScrapeBehaviour reiniciado com novo período.")
            except Exception as exc:
                print(f"[ContinenteScraper] Erro ao processar reconfiguração: {exc}")

    # ------------------------------------------------------------------
    # Comportamento periódico de scraping
    # ------------------------------------------------------------------

    class ScrapeBehaviour(PeriodicBehaviour):
        """Executa o scraping periodicamente e envia o batch ao DatabaseAgent."""

        async def run(self) -> None:
            config = self.agent.config

            if not config.get("ativo", True):
                print("[ContinenteScraper] Scraping desativado. Ciclo ignorado.")
                return

            queries = config.get("queries", [])
            categorias = config.get("categorias", [])
            # Fallback alinhado com DEFAULT_CONFIG (10 páginas por categoria).
            paginas = config.get("paginas", self.agent.DEFAULT_CONFIG["paginas"])
            debug = config.get("debug", False)

            tasks = cs.discover_scrape_tasks(
                queries=",".join(queries) if queries else None,
                categorias=categorias or None,
            )

            if not tasks:
                print("[ContinenteScraper] Nenhuma tarefa configurada. Ciclo ignorado.")
                return

            run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # cs.scrape_tasks usa time.sleep internamente — corre em thread separada
            # para não bloquear o event loop do SPADE durante o HTTP scraping.
            # A sessão é reutilizada entre ciclos (criada em setup) para evitar
            # o overhead de handshake TLS em cada ciclo.
            all_products = await asyncio.to_thread(
                cs.scrape_tasks, self.agent.session, tasks, paginas, run_timestamp, debug
            )

            if not all_products:
                print("[ContinenteScraper] Nenhum produto recolhido.")
                return

            print(f"[ContinenteScraper] {len(all_products)} produtos recolhidos. A enviar ao DatabaseAgent...")

            payload = {
                "id_loja": config.get("id_loja", "Continente"),
                "data_extracao": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "produtos": all_products,
            }

            msg = Message(to=self.agent.db_agent_jid)
            msg.set_metadata("performative", "request")
            msg.body = jsonpickle.encode(payload)
            await self.send(msg)
            print("[ContinenteScraper] Batch enviado ao DatabaseAgent.")

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        print(f"[ContinenteScraper] A iniciar (db_agent={self.db_agent_jid})...")
        # Criar sessão HTTP uma única vez — reutilizada em todos os ciclos
        self.session = cs.make_session()
        rb = self.ReceiveBehaviour()
        self.scrape_behaviour = self.ScrapeBehaviour(period=self.config["periodo"])
        self.add_behaviour(rb)
        self.add_behaviour(self.scrape_behaviour)