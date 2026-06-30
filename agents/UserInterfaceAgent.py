"""
Agente SPADE de interface HTTP↔XMPP.

Serve de ponte entre a API REST (FastAPI) e o sistema multiagente (SPADE/XMPP),
permitindo que clientes HTTP consultem os agentes sem conhecimento do protocolo
XMPP nem das mensagens FIPA-ACL.

Cada pedido HTTP é convertido numa mensagem FIPA-ACL com um ``correlation_id``
único (UUID4). O agente envia a mensagem ao agente-alvo, aguarda a resposta e
resolve o ``asyncio.Future`` associado, que desbloqueia o handler FastAPI e
retorna o resultado ao cliente como JSON.

Comportamentos:
    - SendBehaviour (CyclicBehaviour): drena a fila de saída (``_outbound``) e
      envia mensagens XMPP via SPADE. Necessário porque em SPADE as mensagens
      só podem ser enviadas a partir de comportamentos.
    - ReceiveBehaviour (CyclicBehaviour): recebe respostas e resolve os Futures
      pendentes por ``correlation_id``.

Protocolo de mensagens:
    Envia (para RecommendationAgent / PredictionAgent / DatabaseAgent):
        performative: query
        metadata: correlation_id (UUID4)
        body: {"type": str, "params": dict} (jsonpickle)

    Recebe (de qualquer agente, em resposta):
        performative: inform | failure
        metadata: correlation_id (obrigatório para correlação)
        body: resultado (jsonpickle)
"""

from __future__ import annotations

import asyncio
import uuid

import jsonpickle
from spade import agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message


class UserInterfaceAgent(agent.Agent):
    """Agente ponte entre a API REST (FastAPI) e o sistema multiagente (XMPP).

    Converte pedidos HTTP assíncronos em mensagens FIPA-ACL e aguarda as
    respostas usando ``asyncio.Future`` com timeout configurável.

    Atributos:
        recommendation_jid: JID do RecommendationAgent.
        prediction_jid: JID do PredictionAgent.
        database_jid: JID do DatabaseAgent.
        location_jid: JID do LocationAgent (opcional — None se não usar GPS).
    """

    #: Timeout padrão para queries a agentes, em segundos.
    DEFAULT_TIMEOUT: float = 30.0

    def __init__(
        self,
        jid: str,
        password: str,
        recommendation_jid: str,
        prediction_jid: str,
        database_jid: str,
        location_jid: str | None = None,
    ) -> None:
        super().__init__(jid, password)
        self.recommendation_jid = recommendation_jid
        self.prediction_jid = prediction_jid
        self.database_jid = database_jid
        self.location_jid = location_jid
        # Inicializados em setup() — asyncio.Queue requer event loop ativo.
        self._pending: dict[str, asyncio.Future] = {}
        self._outbound: asyncio.Queue | None = None

    # ------------------------------------------------------------------
    # Comportamento de envio — drena a fila de saída
    # ------------------------------------------------------------------

    class SendBehaviour(CyclicBehaviour):
        """Drena ``_outbound`` e envia mensagens via SPADE.

        Em SPADE, mensagens só podem ser enviadas dentro de comportamentos.
        Esta behaviour serve de proxy entre o método ``query()`` (chamado
        do contexto FastAPI) e o canal XMPP.
        """

        async def run(self) -> None:
            try:
                msg = await asyncio.wait_for(
                    self.agent._outbound.get(),
                    timeout=1.0,
                )
                await self.send(msg)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    # Comportamento de receção — correlaciona respostas com Futures
    # ------------------------------------------------------------------

    class ReceiveBehaviour(CyclicBehaviour):
        """Recebe respostas dos agentes e resolve os Futures pendentes."""

        async def run(self) -> None:
            msg = await self.receive(timeout=5)
            if not msg:
                return

            correlation_id = msg.get_metadata("correlation_id")
            if not correlation_id:
                print(
                    f"[UserInterfaceAgent] Resposta sem correlation_id de "
                    f"{str(msg.sender).split('/')[0]!r} — ignorada."
                )
                return

            future = self.agent._pending.pop(correlation_id, None)
            if future is None or future.done():
                # Resposta tardia (já expirou o timeout ou Future cancelado)
                return

            resultado = jsonpickle.decode(msg.body)
            if msg.get_metadata("performative") == "failure":
                future.set_exception(
                    RuntimeError(resultado.get("erro", "Erro desconhecido no agente."))
                )
            else:
                future.set_result(resultado)

    # ------------------------------------------------------------------
    # API pública — método de consulta
    # ------------------------------------------------------------------

    async def query(
        self,
        target_jid: str,
        performative: str,
        query_type: str,
        params: dict,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> object:
        """Envia uma query a um agente e aguarda a resposta.

        Args:
            target_jid: JID do agente destinatário.
            performative: Performative FIPA-ACL (normalmente ``"query"``).
            query_type: Campo ``"type"`` do body da mensagem.
            params: Parâmetros da query (serializados em jsonpickle).
            timeout: Tempo máximo de espera em segundos.

        Returns:
            Resultado deserializado (dict, lista, etc.).

        Raises:
            RuntimeError: Se o agente devolver performative ``"failure"``.
            asyncio.TimeoutError: Se o agente não responder dentro do timeout.
            RuntimeError: Se o agente ainda não foi inicializado.
        """
        if self._outbound is None:
            raise RuntimeError(
                "UserInterfaceAgent ainda não inicializado (setup() não executado)."
            )

        correlation_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[correlation_id] = future

        msg = Message(to=target_jid)
        msg.set_metadata("performative", performative)
        msg.set_metadata("correlation_id", correlation_id)
        msg.body = jsonpickle.encode({"type": query_type, "params": params})

        await self._outbound.put(msg)

        try:
            # asyncio.shield protege o Future de ser cancelado se o caller
            # for cancelado (ex: timeout no FastAPI), garantindo que o
            # correlation_id seja sempre removido de _pending.
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(correlation_id, None)
            raise

    # ------------------------------------------------------------------
    # Métodos de conveniência por agente-alvo
    # ------------------------------------------------------------------

    async def recomendar(
        self, query_type: str, params: dict, timeout: float = DEFAULT_TIMEOUT
    ) -> object:
        """Atalho para queries ao RecommendationAgent."""
        return await self.query(
            self.recommendation_jid, "query", query_type, params, timeout=timeout
        )

    async def prever(
        self, query_type: str, params: dict, timeout: float = DEFAULT_TIMEOUT
    ) -> object:
        """Atalho para queries ao PredictionAgent."""
        return await self.query(
            self.prediction_jid, "query", query_type, params, timeout=timeout
        )

    async def consultar_bd(
        self, query_type: str, params: dict, timeout: float = DEFAULT_TIMEOUT
    ) -> object:
        """Atalho para queries ao DatabaseAgent."""
        return await self.query(
            self.database_jid, "query", query_type, params, timeout=timeout
        )

    async def localizar(
        self, query_type: str, params: dict, timeout: float = DEFAULT_TIMEOUT
    ) -> object:
        """Atalho para queries ao LocationAgent.

        Raises:
            RuntimeError: Se o LocationAgent não estiver configurado (sistema
                arrancado com --sem-localizacao).
        """
        if self.location_jid is None:
            raise RuntimeError(
                "LocationAgent não configurado. "
                "Arranca o sistema sem --sem-localizacao para activar GPS."
            )
        return await self.query(
            self.location_jid, "query", query_type, params, timeout=timeout
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        print("[UserInterfaceAgent] A iniciar...")
        # Criar Queue com event loop ativo
        self._outbound = asyncio.Queue()
        self.add_behaviour(self.SendBehaviour())
        self.add_behaviour(self.ReceiveBehaviour())
