"""
Agente SPADE para scraping periódico do Auchan Portugal.

Mesma arquitetura dos outros scrapers (ContinenteScraper, PingoDoceScraper)
mas adaptado ao scraper do Auchan, que usa sitemap XML para descoberta de
produtos e o endpoint AJAX SFCC como fonte primária de dados.

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

sys.path.insert(0, str(Path(__file__).parent.parent))
import scrapers.auchan_scraper as cs


class AuchanScraper(agent.Agent):
    """Agente scraper do Auchan Portugal.

    Percorre o sitemap XML do Auchan periodicamente, recolhe preços via
    endpoint AJAX SFCC (com fallback HTML) e envia o batch ao DatabaseAgent
    para ingestão na base de dados.
    """

    DEFAULT_CONFIG: dict = {
        "id_loja": "Auchan",
        "categoria": "alimentacao",  # filtro de categoria no sitemap
        "subcategoria": None,       # filtro de subcategoria opcional
        "limite": None,             # None = sem limite (produção)
        "periodo": 21600,           # 6 horas em segundos
        "ativo": True,
        "debug": False,
    }

    def __init__(self, jid: str, password: str, db_agent_jid: str) -> None:
        super().__init__(jid, password)
        self.db_agent_jid = db_agent_jid
        self.config = dict(self.DEFAULT_CONFIG)
        self.scrape_behaviour: PeriodicBehaviour | None = None
        # Sessão HTTP partilhada entre ciclos — criada em setup() e reutilizada.
        self.session: requests.Session | None = None

    # ------------------------------------------------------------------
    # Comportamento de receção de reconfiguração dinâmica
    # ------------------------------------------------------------------

    class ReceiveBehaviour(CyclicBehaviour):
        """Aguarda mensagens de reconfiguração dinâmica do agente.

        Aceita apenas mensagens com ``performative=inform`` e body que descodifique
        para um dicionário. Mensagens malformadas são ignoradas com aviso para
        não derrubar o behaviour (a falha não-tratada pararia a cyclic loop).
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
                print(f"[AuchanScraper] Nova configuração recebida: {new_config}")
                self.agent.config.update(new_config)

                if self.agent.scrape_behaviour:
                    self.agent.remove_behaviour(self.agent.scrape_behaviour)

                periodo = self.agent.config.get("periodo", self.agent.DEFAULT_CONFIG["periodo"])
                new_b = self.agent.ScrapeBehaviour(period=periodo)
                self.agent.scrape_behaviour = new_b
                self.agent.add_behaviour(new_b)
                print("[AuchanScraper] ScrapeBehaviour reiniciado com novo período.")
            except Exception as exc:
                print(f"[AuchanScraper] Erro ao processar reconfiguração: {exc}")

    # ------------------------------------------------------------------
    # Comportamento periódico de scraping
    # ------------------------------------------------------------------

    class ScrapeBehaviour(PeriodicBehaviour):
        """Executa o scraping periodicamente e envia o batch ao DatabaseAgent."""

        async def run(self) -> None:
            config = self.agent.config

            if not config.get("ativo", True):
                print("[AuchanScraper] Scraping desativado. Ciclo ignorado.")
                return

            run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # cs.scrape_products usa time.sleep internamente — corre em thread separada
            # para não bloquear o event loop do SPADE durante o HTTP scraping.
            # A sessão é reutilizada entre ciclos (criada em setup) para evitar
            # o overhead de handshake TLS em cada ciclo.
            all_products = await asyncio.to_thread(
                cs.scrape_products,
                self.agent.session,
                categoria=config.get("categoria", "alimentacao"),
                subcategoria=config.get("subcategoria", None),
                limite=config.get("limite", None),
                run_timestamp=run_timestamp,
                debug=config.get("debug", False),
            )

            if not all_products:
                print("[AuchanScraper] Nenhum produto recolhido.")
                return

            print(f"[AuchanScraper] {len(all_products)} produtos recolhidos. A enviar ao DatabaseAgent...")

            payload = {
                "id_loja": config.get("id_loja", "Auchan"),
                "data_extracao": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "produtos": all_products,
            }

            msg = Message(to=self.agent.db_agent_jid)
            msg.set_metadata("performative", "request")
            msg.body = jsonpickle.encode(payload)
            await self.send(msg)
            print("[AuchanScraper] Batch enviado ao DatabaseAgent.")

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        print(f"[AuchanScraper] A iniciar (db_agent={self.db_agent_jid})...")
        # Criar sessão HTTP uma única vez — reutilizada em todos os ciclos
        self.session = cs.make_session()
        rb = self.ReceiveBehaviour()
        self.scrape_behaviour = self.ScrapeBehaviour(period=self.config["periodo"])
        self.add_behaviour(rb)
        self.add_behaviour(self.scrape_behaviour)
