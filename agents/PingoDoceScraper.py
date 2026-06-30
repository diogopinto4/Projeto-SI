"""
Agente SPADE para scraping periódico do Pingo Doce.

Mesma arquitetura do ContinenteScraper mas adaptado ao scraper do Pingo Doce,
que usa sitemap XML em vez de paginação direta.

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
import scrapers.pingo_doce_scraper as cs


class PingoDoceScraper(agent.Agent):
    """Agente scraper do Pingo Doce.

    Percorre o sitemap XML do Pingo Doce periodicamente, recolhe preços
    e envia o batch ao DatabaseAgent para ingestão na BD.
    """

    DEFAULT_CONFIG: dict = {
        "id_loja": "Pingo Doce",
        # Lista de categorias a recolher do sitemap. Equilibra cobertura com
        # tempo de scraping. Inclui as principais secções de alimentação +
        # bebidas + lacticínios + congelados para emparelhar com Auchan/Continente.
        "categorias": [
            "mercearia",
            "bolachas-cereais-e-guloseimas",
            "iogurtes-e-sobremesas",
            # Nota: a PD tem um zero-width space (%E2%80%8B) no fim deste slug
            # no sitemap. O ``filter_urls`` normaliza ambos os lados para
            # tolerar esse carácter invisível — ver ``_url_norm`` em
            # ``scrapers/pingo_doce_scraper.py``.
            "leite-e-bebidas-vegetais",
            "congelados",
            "charcutaria-e-queijos",
            "aguas-sumos-e-refrigerantes",
            "padaria-e-pastelaria",
            "cafe-cha-e-achocolatados",
        ],
        "categoria": None,          # legado: ignorado se 'categorias' definida
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
                print(f"[PingoDoceScraper] Nova configuração recebida: {new_config}")
                self.agent.config.update(new_config)

                if self.agent.scrape_behaviour:
                    self.agent.remove_behaviour(self.agent.scrape_behaviour)

                periodo = self.agent.config.get("periodo", self.agent.DEFAULT_CONFIG["periodo"])
                new_b = self.agent.ScrapeBehaviour(period=periodo)
                self.agent.scrape_behaviour = new_b
                self.agent.add_behaviour(new_b)
                print("[PingoDoceScraper] ScrapeBehaviour reiniciado com novo período.")
            except Exception as exc:
                print(f"[PingoDoceScraper] Erro ao processar reconfiguração: {exc}")

    # ------------------------------------------------------------------
    # Comportamento periódico de scraping
    # ------------------------------------------------------------------

    class ScrapeBehaviour(PeriodicBehaviour):
        """Executa o scraping periodicamente e envia o batch ao DatabaseAgent."""

        async def run(self) -> None:
            config = self.agent.config

            if not config.get("ativo", True):
                print("[PingoDoceScraper] Scraping desativado. Ciclo ignorado.")
                return

            run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # cs.scrape_products usa time.sleep internamente — corre em thread separada
            # para não bloquear o event loop do SPADE durante o HTTP scraping.
            # A sessão é reutilizada entre ciclos (criada em setup) para evitar
            # o overhead de handshake TLS em cada ciclo.
            all_products = await asyncio.to_thread(
                cs.scrape_products,
                self.agent.session,
                categorias=config.get("categorias", None),
                categoria=config.get("categoria", None),
                subcategoria=config.get("subcategoria", None),
                limite=config.get("limite", None),
                run_timestamp=run_timestamp,
                debug=config.get("debug", False),
            )

            if not all_products:
                print("[PingoDoceScraper] Nenhum produto recolhido.")
                return

            print(f"[PingoDoceScraper] {len(all_products)} produtos recolhidos. A enviar ao DatabaseAgent...")

            payload = {
                "id_loja": config.get("id_loja", "Pingo Doce"),
                "data_extracao": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "produtos": all_products,
            }

            msg = Message(to=self.agent.db_agent_jid)
            msg.set_metadata("performative", "request")
            msg.body = jsonpickle.encode(payload)
            await self.send(msg)
            print("[PingoDoceScraper] Batch enviado ao DatabaseAgent.")

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        print(f"[PingoDoceScraper] A iniciar (db_agent={self.db_agent_jid})...")
        # Criar sessão HTTP uma única vez — reutilizada em todos os ciclos
        self.session = cs.make_session()
        rb = self.ReceiveBehaviour()
        self.scrape_behaviour = self.ScrapeBehaviour(period=self.config["periodo"])
        self.add_behaviour(rb)
        self.add_behaviour(self.scrape_behaviour)