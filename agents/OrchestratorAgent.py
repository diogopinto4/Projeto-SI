"""
Agente SPADE coordenador do pipeline de análise de preços.

Funciona como maestro do sistema multiagente: reage a eventos dos outros
agentes (ingestão concluída, relatórios de monitorização) e desencadeia
o pipeline de pós-processamento na sequência correta.

Pipeline desencadeado após ingestão:
    1. Construir dataset de forecasting  (build_forecasting_dataset.py)
    2. [Opcional] Solicitar retreino ao PredictionAgent
    3. Registar relatórios de monitorização no log

Comportamentos:
    - ListenBehaviour (CyclicBehaviour): aguarda eventos de outros agentes
      e decide a ação a tomar com base no tipo de mensagem recebida.

    - PipelineBehaviour (OneShotBehaviour): adicionado dinamicamente quando
      é acionado por um evento de ingestão. Constrói o dataset e,
      opcionalmente, pede ao PredictionAgent para retreinar o modelo.

Protocolo de mensagens:
    Recebe (de DatabaseAgent, após ingestão):
        performative: inform
        body: {"type": "ingestao_concluida", "total_ok": int, "falhas": int,
               "loja": str}

    Recebe (de MonitorAgent, após ciclo de monitorização):
        performative: inform
        body: {"type": "relatorio_monitor", "n_alertas": int,
               "n_anomalias": int, "ts": str, "top_alertas": [...]}

    Recebe (trigger manual de qualquer agente):
        performative: request
        body: {"type": "trigger_pipeline"}

    Envia (para PredictionAgent, quando retreinar=True):
        performative: request
        body: {"type": "treinar"}
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import jsonpickle
from spade import agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_forecasting_dataset import (
    add_cadeia_id,
    add_lag_features,
    add_target,
    add_time_features,
    aggregate_daily,
    expand_daily_panel,
    load_history,
)

#: Caminho absoluto por omissão do dataset gerado (relativo à raiz do projeto).
DATASET_PATH_DEFAULT = str(Path(__file__).parent.parent / "data/generated/forecasting_dataset.csv")


class OrchestratorAgent(agent.Agent):
    """Agente coordenador do pipeline de análise de preços.

    Reage a notificações de outros agentes e desencadeia o pipeline de
    construção do dataset e (opcionalmente) retreino do modelo LSTM.
    """

    DEFAULT_CONFIG: dict = {
        "dataset": DATASET_PATH_DEFAULT,
        "horizonte": 1,     # horizonte do alvo no dataset (dias)
        "retreinar": False, # pedir retreino ao PredictionAgent após cada pipeline
        "ativo": True,
    }

    def __init__(
        self,
        jid: str,
        password: str,
        prediction_agent_jid: str | None = None,
    ) -> None:
        super().__init__(jid, password)
        self.prediction_agent_jid = prediction_agent_jid
        self.config = dict(self.DEFAULT_CONFIG)
        # Contador de pipelines executados nesta sessão
        self._pipelines_executados = 0
        # Guarda contra race condition: evita dois PipelineBehaviour simultâneos
        self._pipeline_em_curso = False

    # ------------------------------------------------------------------
    # Comportamento de escuta de eventos
    # ------------------------------------------------------------------

    class ListenBehaviour(CyclicBehaviour):
        """Aguarda eventos de todos os outros agentes e reage em conformidade."""

        async def run(self) -> None:
            msg = await self.receive(timeout=10)
            if not msg:
                return

            performative = msg.get_metadata("performative")

            if performative == "inform":
                await self._handle_inform(msg)
            elif performative == "request":
                await self._handle_request(msg)
            else:
                print(f"[OrchestratorAgent] Performative desconhecido: {performative!r}")

        # --------------------------------------------------------------

        async def _handle_inform(self, msg: Message) -> None:
            """Trata eventos informativos de outros agentes."""
            try:
                data = jsonpickle.decode(msg.body)
                tipo = data.get("type", "")
                remetente = str(msg.sender).split("/")[0]  # só o JID base

                if tipo == "ingestao_concluida":
                    total_ok = data.get("total_ok", "?")
                    falhas = data.get("falhas", 0)
                    loja = data.get("loja", "?")
                    print(
                        f"[OrchestratorAgent] Ingestão concluída — loja={loja!r}, "
                        f"ok={total_ok}, falhas={falhas}."
                    )
                    if self.agent.config.get("ativo", True):
                        if self.agent._pipeline_em_curso:
                            print(
                                "[OrchestratorAgent] Pipeline já em curso — "
                                "notificação de ingestão ignorada."
                            )
                        else:
                            self.agent.add_behaviour(
                                self.agent.PipelineBehaviour()
                            )

                elif tipo == "relatorio_monitor":
                    n_alertas = data.get("n_alertas", 0)
                    n_anomalias = data.get("n_anomalias", 0)
                    ts = data.get("ts", "")
                    print(
                        f"[OrchestratorAgent] Relatório de monitorização ({ts}) — "
                        f"alertas={n_alertas}, anomalias={n_anomalias}."
                    )
                    if n_alertas > 0:
                        top = data.get("top_alertas", [])
                        for alerta in top:
                            print(
                                f"  [{alerta.get('tipo_alerta','?'):25s}] "
                                f"{str(alerta.get('nome_padronizado',''))[:40]} "
                                f"({alerta.get('loja','?')}) "
                                f"{alerta.get('variacao_pct', 0):+.1f}%"
                            )
                    if n_anomalias > 0:
                        print(
                            f"  [AVISO] {n_anomalias} anomalia(s) — "
                            "verifica data/anomalias/ e considera retreinar o modelo."
                        )

                else:
                    print(
                        f"[OrchestratorAgent] Mensagem inform de {remetente!r} "
                        f"com tipo desconhecido: {tipo!r}"
                    )

            except Exception as exc:
                print(f"[OrchestratorAgent] Erro ao processar inform: {exc}")

        # --------------------------------------------------------------

        async def _handle_request(self, msg: Message) -> None:
            """Trata pedidos de trigger manual do pipeline."""
            try:
                data = jsonpickle.decode(msg.body)
                if data.get("type") == "trigger_pipeline":
                    print(
                        f"[OrchestratorAgent] Trigger manual recebido de "
                        f"{str(msg.sender).split('/')[0]!r}."
                    )
                    if self.agent.config.get("ativo", True):
                        if self.agent._pipeline_em_curso:
                            print(
                                "[OrchestratorAgent] Pipeline já em curso — "
                                "trigger manual ignorado."
                            )
                        else:
                            self.agent.add_behaviour(self.agent.PipelineBehaviour())
            except Exception as exc:
                print(f"[OrchestratorAgent] Erro ao processar request: {exc}")

    # ------------------------------------------------------------------
    # Comportamento de pipeline (OneShotBehaviour — disparado por evento)
    # ------------------------------------------------------------------

    class PipelineBehaviour(OneShotBehaviour):
        """Executa o pipeline de pós-ingestão uma única vez.

        Constrói o dataset de forecasting e, se ``retreinar=True``,
        envia pedido de retreino ao PredictionAgent.
        """

        async def run(self) -> None:
            config = self.agent.config
            self.agent._pipeline_em_curso = True
            self.agent._pipelines_executados += 1
            n = self.agent._pipelines_executados
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"\n[OrchestratorAgent] Pipeline #{n} iniciado ({ts}).")

            # ----------------------------------------------------------
            # Passo 1: construir dataset de forecasting
            # ----------------------------------------------------------
            dataset_path = config.get("dataset", DATASET_PATH_DEFAULT)
            horizonte = config.get("horizonte", 1)

            try:
                # Todo o pipeline é bloqueante (psycopg2 + pandas + CSV write).
                # Corre numa thread separada para não bloquear o event loop.
                def _construir_dataset() -> tuple:
                    df = load_history()
                    n_carregadas = len(df)
                    df = aggregate_daily(df)
                    df = expand_daily_panel(df)
                    df = add_time_features(df)
                    df = add_cadeia_id(df)
                    df = add_lag_features(df)
                    df = add_target(df, horizonte=horizonte)
                    out = Path(dataset_path)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out, index=False)
                    return out, n_carregadas, df["id_produto_loja"].nunique(), len(df)

                print("[OrchestratorAgent] A carregar histórico e construir features...")
                output, n_carregadas, n_produtos, n_linhas = await asyncio.to_thread(
                    _construir_dataset
                )
                print(f"  {n_carregadas} entradas carregadas da BD.")
                print(
                    f"[OrchestratorAgent] Dataset guardado em {output} "
                    f"({n_linhas:,} linhas, {n_produtos} produtos)."
                )

            except ValueError as exc:
                # BD vazia — pipeline abortado sem erro fatal
                print(f"[OrchestratorAgent] Pipeline abortado: {exc}")
                self.agent._pipeline_em_curso = False
                return
            except Exception as exc:
                print(f"[OrchestratorAgent] Erro no pipeline: {exc}")
                self.agent._pipeline_em_curso = False
                return

            # ----------------------------------------------------------
            # Passo 2 (opcional): solicitar retreino ao PredictionAgent
            # ----------------------------------------------------------
            try:
                if config.get("retreinar", False) and self.agent.prediction_agent_jid:
                    await self._solicitar_retreino()
            except Exception as exc:
                print(f"[OrchestratorAgent] Erro ao solicitar retreino: {exc}")
            finally:
                self.agent._pipeline_em_curso = False
                print(f"[OrchestratorAgent] Pipeline #{n} concluído.")

        async def _solicitar_retreino(self) -> None:
            """Envia pedido de retreino ao PredictionAgent."""
            msg = Message(to=self.agent.prediction_agent_jid)
            msg.set_metadata("performative", "request")
            msg.body = jsonpickle.encode({"type": "treinar"})
            await self.send(msg)
            print(
                f"[OrchestratorAgent] Pedido de retreino enviado ao "
                f"PredictionAgent ({self.agent.prediction_agent_jid})."
            )

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        pred_jid = self.prediction_agent_jid or "—"
        print(f"[OrchestratorAgent] A iniciar (prediction_agent={pred_jid})...")
        self.add_behaviour(self.ListenBehaviour())
