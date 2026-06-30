"""
Agente SPADE responsável pela gestão da base de dados.

Centraliza todas as operações de escrita e leitura na BD PostgreSQL,
delegando a lógica de ingestão para scripts/ingest.py (sem duplicação).

Protocolo de mensagens:
    Recebe (de ScraperAgents):
        performative: request
        body: {"id_loja", "data_extracao", "produtos": [...]} (jsonpickle)
        → Ingere os produtos via scripts/ingest.ingestao()

    Recebe (de PredictionAgent ou outros):
        performative: query
        body: {"type": str, "params": dict} (jsonpickle)
        → Executa query na BD e responde com os resultados

    Envia (em resposta a query):
        performative: inform  (sucesso)
        performative: failure (erro)
        body: resultado (jsonpickle)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import jsonpickle
import psycopg2
from spade import agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from agents.messaging import construir_reply

# Adiciona a raiz do projecto ao path para importar scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ingest import ingestao
from scripts.db_config import DB_CONFIG


class DatabaseAgent(agent.Agent):
    """Agente central de gestão da BD.

    Recebe batches de produtos dos scrapers e consultas de outros agentes.
    Toda a lógica de normalização e ingestão está em scripts/ingest.py.

    Após cada ingestão bem-sucedida notifica o OrchestratorAgent (se configurado)
    com uma mensagem ``inform`` do tipo ``"ingestao_concluida"``, para que o
    pipeline de pós-processamento (build_dataset, retreino) seja desencadeado
    automaticamente.
    """

    class StoreBehaviour(CyclicBehaviour):
        """Ciclo de receção de mensagens com dispatch por performative."""

        async def run(self) -> None:
            msg = await self.receive(timeout=10)
            if not msg:
                return

            performative = msg.get_metadata("performative")

            if performative == "request":
                await self._handle_ingest(msg)

            elif performative == "query":
                await self._handle_query(msg)

            else:
                print(f"[DatabaseAgent] Performative desconhecido: {performative!r}")

        # --------------------------------------------------------------

        async def _handle_ingest(self, msg: Message) -> None:
            """Ingere um batch de produtos recebido de um scraper.

            Após a ingestão, notifica o OrchestratorAgent (se configurado) com
            um resumo da operação para que o pipeline de pós-processamento
            (build_dataset, retreino) seja desencadeado automaticamente.
            """
            try:
                data = jsonpickle.decode(msg.body)
                produtos = data.get("produtos", [])
                id_loja = data.get("id_loja", "?")

                if not produtos:
                    print(f"[DatabaseAgent] Batch vazio de '{id_loja}' — ignorado.")
                    return

                print(f"[DatabaseAgent] Ingestão: {len(produtos)} produtos de '{id_loja}'")

                # ingestao() é bloqueante (psycopg2); corre numa thread separada
                # para não bloquear o event loop do SPADE durante a ingestão.
                total_ok, falhas = await asyncio.to_thread(
                    ingestao, produtos, False
                )
                print(f"[DatabaseAgent] Concluído: {total_ok} ok, {len(falhas)} falhas.")

                # Notificar OrchestratorAgent para desencadear o pipeline
                if self.agent.orchestrator_jid:
                    notif = Message(to=self.agent.orchestrator_jid)
                    notif.set_metadata("performative", "inform")
                    notif.body = jsonpickle.encode({
                        "type": "ingestao_concluida",
                        "total_ok": total_ok,
                        "falhas": len(falhas),
                        "loja": id_loja,
                    })
                    await self.send(notif)

            except Exception as exc:
                print(f"[DatabaseAgent] Erro na ingestão: {exc}")

        # --------------------------------------------------------------

        async def _handle_query(self, msg: Message) -> None:
            """Executa uma consulta à BD e responde ao remetente.

            Propaga o ``correlation_id`` da mensagem original para que o
            UserInterfaceAgent consiga correlacionar respostas com pedidos HTTP.
            """
            try:
                data = jsonpickle.decode(msg.body)
                query_type = data.get("type", "")
                params = data.get("params", {})

                resultado = await asyncio.to_thread(
                    self._executar_query, query_type, params
                )
                await self.send(construir_reply(msg, "inform", resultado))

            except Exception as exc:
                print(f"[DatabaseAgent] Erro na consulta: {exc}")
                await self.send(construir_reply(msg, "failure", {"erro": str(exc)}))

        # --------------------------------------------------------------

        def _executar_query(self, query_type: str, params: dict) -> dict:
            """Executa uma consulta predefinida e devolve os resultados.

            Extensível: adicionar novos tipos conforme necessário (ex: para
            o futuro agente de previsão de preços).
            """
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:

                    if query_type == "historico_produto":
                        # Histórico de preços de um produto (para o LSTM)
                        id_pl = params.get("id_produto_loja")
                        dias = params.get("dias", 30)
                        # Nota: INTERVAL '%s days' não funciona com psycopg2 porque %s
                        # dentro de um literal SQL não é substituído. Usa-se a forma
                        # (%s * INTERVAL '1 day') que é parametrizável correctamente.
                        cur.execute("""
                            SELECT data_recolha, preco_atual, em_promocao
                            FROM historico_precos
                            WHERE id_produto_loja = %s
                              AND data_recolha >= NOW() - (%s * INTERVAL '1 day')
                            ORDER BY data_recolha;
                        """, (id_pl, dias))
                        rows = cur.fetchall()
                        return {
                            "id_produto_loja": id_pl,
                            "historico": [
                                {"data": str(r[0]), "preco": float(r[1]), "em_promocao": r[2]}
                                for r in rows
                            ],
                        }

                    elif query_type == "precos_atuais_loja":
                        # Preços atuais de uma loja (para o recomendador)
                        insignia = params.get("insignia", "")
                        cur.execute("""
                            SELECT pm.nome_padronizado, pa.preco_atual, pa.em_promocao
                            FROM precos_atuais pa
                            JOIN produtos_loja pl  ON pl.id_produto_loja = pa.id_produto_loja
                            JOIN lojas l           ON l.id_loja = pl.id_loja
                            JOIN produtos_mestre pm ON pm.id_produto_mestre = pl.id_produto_mestre
                            WHERE l.insignia = %s
                            ORDER BY pm.nome_padronizado;
                        """, (insignia,))
                        rows = cur.fetchall()
                        return {
                            "insignia": insignia,
                            "produtos": [
                                {"nome": r[0], "preco": float(r[1]), "em_promocao": r[2]}
                                for r in rows
                            ],
                        }

                    else:
                        raise ValueError(f"Tipo de consulta desconhecido: {query_type!r}")

    # ------------------------------------------------------------------

    def __init__(
        self,
        jid: str,
        password: str,
        orchestrator_jid: str | None = None,
    ) -> None:
        super().__init__(jid, password)
        self.orchestrator_jid = orchestrator_jid

    async def setup(self) -> None:
        orquestrador = self.orchestrator_jid or "—"
        print(f"[DatabaseAgent] A iniciar (orquestrador={orquestrador})...")
        self.add_behaviour(self.StoreBehaviour())