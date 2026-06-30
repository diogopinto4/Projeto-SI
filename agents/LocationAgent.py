"""
Agente SPADE de geolocalização e custo de deslocação.

Encapsula o módulo ``models/geolocation.py`` no paradigma de agentes,
expondo as suas funcionalidades via mensagens FIPA-ACL. Funciona como agente
sensor de geolocalização — recebe coordenadas GPS do utilizador e devolve
informação sobre lojas físicas próximas e custos de deslocação.

Justificação académica: a feature de "custo de deslocação" introduz um novo
domínio de conhecimento (geografia + economia de transporte) ortogonal ao
de preços de produtos. Encapsulá-la num agente dedicado mantém a separação
de preocupações característica de um sistema multi-agente bem estruturado.

Comportamentos:
    - ServeBehaviour (CyclicBehaviour): responde a pedidos de geolocalização
      do RecommendationAgent (ou de qualquer outro agente / cliente externo).

Protocolo de mensagens:
    Recebe (lojas próximas a um ponto):
        performative: query
        body: {"type": "lojas_proximas",
               "params": {"lat":      float,
                          "lon":      float,
                          "raio_km":  float (opcional, default 20),
                          "insignia": str   (opcional, filtrar por cadeia),
                          "limite":   int   (opcional, default 50)}}

    Recebe (distância a cada loja física mais próxima por insígnia):
        performative: query
        body: {"type": "distancia_minima",
               "params": {"lat":       float,
                          "lon":       float,
                          "insignias": list[str],
                          "raio_km":   float (opcional, default 30)}}

    Recebe (custo de deslocação para uma distância):
        performative: query
        body: {"type": "custo_deslocacao",
               "params": {"distancia_km": float,
                          "custo_km":     float|str (opcional, default "equilibrado"),
                          "ida_e_volta":  bool      (opcional, default True)}}

    Recebe (listar presets de €/km disponíveis):
        performative: query
        body: {"type": "presets_custo_km"}

    Envia (sucesso):
        performative: inform
        body: resultado serializado via jsonpickle

    Envia (erro):
        performative: failure
        body: {"erro": str}
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import jsonpickle
from spade import agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from agents.messaging import construir_reply

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.geolocation import (
    PRESETS_CUSTO_KM,
    custo_deslocacao_euros,
    distancia_minima_por_insignia,
    lojas_proximas,
    resolver_custo_km,
)


class LocationAgent(agent.Agent):
    """Agente de geolocalização e cálculo de custo de deslocação.

    Recebe coordenadas GPS do utilizador (via RecommendationAgent ou cliente
    externo) e devolve informação sobre lojas físicas próximas, distâncias
    e custos monetários de deslocação.

    O agente não tem comportamento periódico — apenas responde a queries.
    Não mantém estado entre pedidos (cada query é autónoma).
    """

    DEFAULT_CONFIG: dict = {
        "raio_default_proximas": 20.0,    # raio para query lojas_proximas
        "raio_default_minima":   30.0,    # raio para query distancia_minima
        "limite_default":        50,      # limite para query lojas_proximas
        "ativo":                 True,
    }

    def __init__(self, jid: str, password: str) -> None:
        super().__init__(jid, password)
        self.config = dict(self.DEFAULT_CONFIG)

    # ------------------------------------------------------------------
    # Comportamento de serviço
    # ------------------------------------------------------------------

    class ServeBehaviour(CyclicBehaviour):
        """Responde a pedidos de geolocalização via FIPA-ACL."""

        #: Tabela de despacho: tipo de query → método handler. Simples e
        #: extensível sem mexer no método ``run``.
        _HANDLERS: dict[str, str] = {
            "lojas_proximas":   "_handle_lojas_proximas",
            "distancia_minima": "_handle_distancia_minima",
            "custo_deslocacao": "_handle_custo_deslocacao",
            "presets_custo_km": "_handle_presets",
        }

        async def run(self) -> None:
            msg = await self.receive(timeout=10)
            if not msg:
                return

            if msg.get_metadata("performative") != "query":
                print(f"[LocationAgent] Performative inesperado: "
                      f"{msg.get_metadata('performative')!r}")
                return

            try:
                data = jsonpickle.decode(msg.body)
                query_type = data.get("type", "")
                handler = self._HANDLERS.get(query_type)

                if handler is None:
                    raise ValueError(f"Tipo de query desconhecido: {query_type!r}")

                await getattr(self, handler)(msg, data.get("params", {}))

            except Exception as exc:
                print(f"[LocationAgent] Erro ao processar mensagem: {exc}")
                await self._enviar_falha(msg, str(exc))

        # --------------------------------------------------------------
        # Handlers individuais
        # --------------------------------------------------------------

        async def _handle_lojas_proximas(self, msg: Message, params: dict) -> None:
            """Devolve lojas físicas dentro de um raio do ponto GPS dado."""
            lat = params.get("lat")
            lon = params.get("lon")
            if lat is None or lon is None:
                raise ValueError("Parâmetros 'lat' e 'lon' obrigatórios.")

            config = self.agent.config
            raio = params.get("raio_km", config.get("raio_default_proximas"))
            insignia = params.get("insignia")
            limite = params.get("limite", config.get("limite_default"))

            # lojas_proximas é bloqueante (psycopg2) — corre em thread separada
            # para não bloquear o event loop do SPADE.
            resultado = await asyncio.to_thread(
                lojas_proximas,
                float(lat), float(lon),
                raio_km=float(raio),
                insignia=insignia,
                limite=int(limite),
            )

            await self._enviar_resposta(msg, resultado)

        async def _handle_distancia_minima(self, msg: Message, params: dict) -> None:
            """Devolve a loja mais próxima de cada insígnia indicada."""
            lat = params.get("lat")
            lon = params.get("lon")
            if lat is None or lon is None:
                raise ValueError("Parâmetros 'lat' e 'lon' obrigatórios.")

            insignias = params.get("insignias")
            if not insignias:
                raise ValueError("Parâmetro 'insignias' obrigatório (lista não-vazia).")

            config = self.agent.config
            raio = params.get("raio_km", config.get("raio_default_minima"))

            # distancia_minima_por_insignia é bloqueante (BD) — corre em thread.
            resultado = await asyncio.to_thread(
                distancia_minima_por_insignia,
                float(lat), float(lon),
                insignias=list(insignias),
                raio_km=float(raio),
            )

            await self._enviar_resposta(msg, resultado)

        async def _handle_custo_deslocacao(self, msg: Message, params: dict) -> None:
            """Calcula o custo monetário de uma deslocação dada a distância e o €/km.

            Esta query não toca na BD — é puramente aritmética. Devolve um
            dicionário com a desagregação dos cálculos para que o cliente
            possa apresentá-los ao utilizador.
            """
            distancia = params.get("distancia_km")
            if distancia is None:
                raise ValueError("Parâmetro 'distancia_km' obrigatório.")

            custo_km_input = params.get("custo_km")          # preset ou valor
            ida_e_volta = params.get("ida_e_volta", True)

            custo_km_efetivo = resolver_custo_km(custo_km_input)
            custo = custo_deslocacao_euros(
                float(distancia),
                custo_km=custo_km_efetivo,
                ida_e_volta=bool(ida_e_volta),
            )

            resultado = {
                "distancia_km":     float(distancia),
                "custo_km":         custo_km_efetivo,
                "ida_e_volta":      bool(ida_e_volta),
                "custo_euros":      custo,
            }
            await self._enviar_resposta(msg, resultado)

        async def _handle_presets(self, msg: Message, params: dict) -> None:
            """Devolve a lista de presets de €/km disponíveis no sistema."""
            # PRESETS_CUSTO_KM é estático — não precisa de thread.
            await self._enviar_resposta(msg, dict(PRESETS_CUSTO_KM))

        # --------------------------------------------------------------
        # Utilitários de resposta (delegam para agents.messaging)
        # --------------------------------------------------------------

        async def _enviar_resposta(self, original: Message, resultado: object) -> None:
            """Envia reply ``inform`` ao remetente, propagando ``correlation_id``."""
            await self.send(construir_reply(original, "inform", resultado))

        async def _enviar_falha(self, original: Message, erro: str) -> None:
            """Envia reply ``failure`` ao remetente, propagando ``correlation_id``."""
            await self.send(construir_reply(original, "failure", {"erro": erro}))

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        print("[LocationAgent] A iniciar...")
        self.add_behaviour(self.ServeBehaviour())
