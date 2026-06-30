"""
Utilitários partilhados de mensagens FIPA-ACL entre agentes SPADE.

Centraliza o padrão de construção de respostas (reply) com propagação do
``correlation_id``, que era duplicado idêntico em :mod:`agents.DatabaseAgent`,
:mod:`agents.RecommendationAgent` e :mod:`agents.LocationAgent`.

O ``correlation_id`` é usado pelo :class:`agents.UserInterfaceAgent` para
emparelhar pedidos HTTP com as respostas assíncronas dos agentes.
"""

from __future__ import annotations

import jsonpickle
from spade.message import Message


def construir_reply(
    original: Message,
    performative: str,
    payload: object,
) -> Message:
    """Constrói um :class:`Message` de resposta com performative e payload.

    Propaga o ``correlation_id`` da mensagem original (necessário para o
    :class:`UserInterfaceAgent` correlacionar respostas com pedidos HTTP).

    Args:
        original: Mensagem original recebida (define ``to`` e ``correlation_id``).
        performative: Performative FIPA-ACL (``"inform"`` ou ``"failure"``).
        payload: Objecto a serializar para o body. Codificado com ``jsonpickle``.

    Returns:
        Mensagem pronta a ser enviada com ``behaviour.send(...)``.
    """
    reply = Message(to=str(original.sender))
    reply.set_metadata("performative", performative)
    correlation_id = original.get_metadata("correlation_id")
    if correlation_id:
        reply.set_metadata("correlation_id", correlation_id)
    reply.body = jsonpickle.encode(payload)
    return reply
