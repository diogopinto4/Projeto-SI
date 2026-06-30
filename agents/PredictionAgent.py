"""
Agente SPADE para previsão de preços com LSTM global.

Encapsula o módulo ``models/price_predictor.py`` no paradigma de agentes,
expondo as suas funcionalidades via mensagens FIPA-ACL.

Comportamentos:
    - ServeBehaviour (CyclicBehaviour): responde a pedidos de previsão e de
      treino manual enviados por outros agentes (ex: OrchestratorAgent).
    - TrainBehaviour (PeriodicBehaviour): retreina o modelo periodicamente
      (por omissão a cada 24h) para incorporar dados recentes de scraping.

Protocolo de mensagens:
    Recebe (previsão determinista):
        performative: query
        body: {"type": "prever", "params": {"produto_id": int, "horizonte": int}}

    Recebe (previsão Monte Carlo com intervalos de confiança):
        performative: query
        body: {"type": "prever_mc", "params": {"produto_id": int,
                                                "horizonte": int,
                                                "amostras": int}}

    Recebe (retreino manual):
        performative: request
        body: {"type": "treinar"}

    Envia (sucesso):
        performative: inform
        body: lista de dicts com {"data", "preco_previsto"}  (prever)
              ou {"previsoes": [...], "preco_atual": float}  (prever_mc)
              ou {"status": "treino_concluido", "ts": str}   (treinar)
              ou {"status": "treino_em_curso"}               (treinar, se já a correr)

    Envia (erro):
        performative: failure
        body: {"erro": str}
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import jsonpickle
from spade import agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.price_predictor import (
    prever_preco,
    prever_preco_com_incerteza,
    treinar_modelo_global,
)

#: Caminho absoluto por omissão para o dataset de forecasting gerado pelo pipeline.
DATASET_PATH_DEFAULT = str(Path(__file__).parent.parent / "data/generated/forecasting_dataset.csv")


class PredictionAgent(agent.Agent):
    """Agente de previsão de preços usando LSTM global.

    Disponibiliza previsões deterministas e probabilísticas (Monte Carlo
    Dropout) para qualquer produto com histórico suficiente na BD.
    O modelo é retreinado automaticamente com a periodicidade configurada.
    """

    DEFAULT_CONFIG: dict = {
        "dataset": DATASET_PATH_DEFAULT,
        "janela": 7,          # dias de contexto por sample
        "epocas": 100,        # épocas máximas de treino
        "paciencia": 15,      # early stopping
        "horizonte": 7,       # dias a prever por omissão
        "amostras_mc": 50,    # simulações Monte Carlo por omissão
        "periodo_treino": 86400,  # 24h em segundos
        "ativo": True,
    }

    def __init__(self, jid: str, password: str) -> None:
        super().__init__(jid, password)
        self.config = dict(self.DEFAULT_CONFIG)
        # Guard contra dois treinos simultâneos (TrainBehaviour + pedido manual).
        # Sem este flag, dois asyncio.to_thread(treinar_modelo_global) concorrentes
        # escreveriam sobre o mesmo ficheiro de artefactos e corromperiam o modelo.
        self._treino_em_curso = False

    # ------------------------------------------------------------------
    # Comportamento de serviço (resposta a mensagens)
    # ------------------------------------------------------------------

    class ServeBehaviour(CyclicBehaviour):
        """Responde a pedidos de previsão e de treino via FIPA-ACL."""

        async def run(self) -> None:
            msg = await self.receive(timeout=10)
            if not msg:
                return

            performative = msg.get_metadata("performative")

            if performative == "query":
                await self._handle_query(msg)
            elif performative == "request":
                await self._handle_request(msg)
            else:
                print(f"[PredictionAgent] Performative desconhecido: {performative!r}")

        # --------------------------------------------------------------

        async def _handle_query(self, msg: Message) -> None:
            """Executa uma previsão e responde ao remetente.

            Propaga o ``correlation_id`` da mensagem original para que o
            UserInterfaceAgent consiga correlacionar respostas com pedidos HTTP.
            """
            config = self.agent.config
            correlation_id = msg.get_metadata("correlation_id")
            try:
                data = jsonpickle.decode(msg.body)
                query_type = data.get("type", "")
                params = data.get("params", {})
                resultado = None

                if query_type == "prever":
                    # prever_preco lê CSV e corre inferência PyTorch — bloqueante
                    df = await asyncio.to_thread(
                        prever_preco,
                        params["produto_id"],
                        config.get("dataset", DATASET_PATH_DEFAULT),
                        params.get("horizonte", config.get("horizonte", 7)),
                    )
                    if df is not None:
                        # Converter datas para string para serialização JSON
                        resultado = [
                            {"data": str(r["data"]), "preco_previsto": r["preco_previsto"]}
                            for r in df.to_dict(orient="records")
                        ]

                elif query_type == "prever_mc":
                    # prever_preco_com_incerteza faz 50 passes Monte Carlo — bloqueante
                    res = await asyncio.to_thread(
                        prever_preco_com_incerteza,
                        params["produto_id"],
                        config.get("dataset", DATASET_PATH_DEFAULT),
                        params.get("horizonte", config.get("horizonte", 7)),
                        params.get("amostras", config.get("amostras_mc", 50)),
                    )
                    if res is not None:
                        previsoes = res["previsoes"].copy()
                        previsoes["data"] = previsoes["data"].astype(str)
                        resultado = {
                            "previsoes": previsoes.to_dict(orient="records"),
                            "preco_atual": res["preco_atual"],
                        }

                else:
                    raise ValueError(f"Tipo de query desconhecido: {query_type!r}")

                if resultado is None:
                    raise ValueError(
                        f"Modelo ou produto não disponível para produto_id="
                        f"{params.get('produto_id')!r}. "
                        "Verifica se o modelo está treinado e o produto tem histórico suficiente."
                    )

                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", "inform")
                if correlation_id:
                    reply.set_metadata("correlation_id", correlation_id)
                reply.body = jsonpickle.encode(resultado)
                await self.send(reply)

            except Exception as exc:
                print(f"[PredictionAgent] Erro na query: {exc}")
                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", "failure")
                if correlation_id:
                    reply.set_metadata("correlation_id", correlation_id)
                reply.body = jsonpickle.encode({"erro": str(exc)})
                await self.send(reply)

        # --------------------------------------------------------------

        async def _handle_request(self, msg: Message) -> None:
            """Processa um pedido de treino manual."""
            config = self.agent.config
            try:
                data = jsonpickle.decode(msg.body)
                if data.get("type") != "treinar":
                    return

                if self.agent._treino_em_curso:
                    print("[PredictionAgent] Treino já em curso — pedido manual ignorado.")
                    reply = Message(to=str(msg.sender))
                    reply.set_metadata("performative", "inform")
                    reply.body = jsonpickle.encode({"status": "treino_em_curso"})
                    await self.send(reply)
                    return

                print(f"[PredictionAgent] Treino manual solicitado por {msg.sender}.")
                self.agent._treino_em_curso = True
                try:
                    # treinar_modelo_global é bloqueante (PyTorch + CSV I/O).
                    # Keyword args: a assinatura é (caminho_csv, janela, epochs,
                    # batch_size, lr, min_observacoes, paciencia) — argumentos
                    # posicionais aqui colocariam "paciencia" em "batch_size".
                    await asyncio.to_thread(
                        lambda: treinar_modelo_global(
                            caminho_csv=config.get("dataset", DATASET_PATH_DEFAULT),
                            janela=config.get("janela", 7),
                            epochs=config.get("epocas", 100),
                            paciencia=config.get("paciencia", 15),
                        )
                    )
                finally:
                    self.agent._treino_em_curso = False

                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", "inform")
                reply.body = jsonpickle.encode({
                    "status": "treino_concluido",
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                await self.send(reply)

            except Exception as exc:
                print(f"[PredictionAgent] Erro no treino manual: {exc}")
                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", "failure")
                reply.body = jsonpickle.encode({"erro": str(exc)})
                await self.send(reply)

    # ------------------------------------------------------------------
    # Comportamento periódico de treino
    # ------------------------------------------------------------------

    class TrainBehaviour(PeriodicBehaviour):
        """Retreina o LSTM periodicamente para incorporar novos dados de scraping."""

        async def run(self) -> None:
            config = self.agent.config
            dataset_path = Path(config.get("dataset", DATASET_PATH_DEFAULT))

            if not config.get("ativo", True):
                print("[PredictionAgent] Retreino periódico desativado. Ciclo ignorado.")
                return

            if self.agent._treino_em_curso:
                print("[PredictionAgent] Treino manual em curso — retreino periódico adiado.")
                return

            if not dataset_path.exists():
                print(
                    f"[PredictionAgent] Dataset não encontrado em {dataset_path} — "
                    "retreino periódico adiado até o pipeline gerar o ficheiro."
                )
                return

            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"[PredictionAgent] A iniciar retreino periódico ({ts})...")
            self.agent._treino_em_curso = True
            try:
                # treinar_modelo_global é bloqueante (PyTorch + CSV I/O).
                # Keyword args para evitar trocar "paciencia" com "batch_size"
                # (ver nota equivalente em _handle_request).
                await asyncio.to_thread(
                    lambda: treinar_modelo_global(
                        caminho_csv=str(dataset_path),
                        janela=config.get("janela", 7),
                        epochs=config.get("epocas", 100),
                        paciencia=config.get("paciencia", 15),
                    )
                )
                print("[PredictionAgent] Retreino periódico concluído.")
            except Exception as exc:
                print(f"[PredictionAgent] Erro no retreino periódico: {exc}")
            finally:
                self.agent._treino_em_curso = False

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        periodo = self.config.get("periodo_treino", self.DEFAULT_CONFIG["periodo_treino"])
        print(f"[PredictionAgent] A iniciar (retreino a cada {periodo}s)...")
        self.add_behaviour(self.ServeBehaviour())
        self.add_behaviour(self.TrainBehaviour(period=periodo))
