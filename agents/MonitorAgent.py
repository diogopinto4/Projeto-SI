"""
Agente SPADE de monitorização de preços e deteção de anomalias.

Corresponde à UC "Sensorização e Ambiente" — atua como sensor contínuo do
mercado, detetando mudanças de preço e dados potencialmente corrompidos.

Comportamentos:
    - MonitorBehaviour (PeriodicBehaviour): corre periodicamente e executa
      duas análises em sequência:

      1. **Monitor de preços** (``scripts/price_monitor.py``) — deteta
         promoções iniciadas/terminadas e subidas/descidas de preço dentro
         de uma janela temporal configurável.

      2. **Detetor de anomalias** (``scripts/anomaly_detector.py``) — sinaliza
         preços suspeitos por IQR e por variação abrupta entre dias consecutivos.

      No final de cada ciclo, envia um relatório resumido ao OrchestratorAgent
      (se configurado) para que este possa desencadear ações corretivas.

Protocolo de mensagens:
    Envia (para OrchestratorAgent, após cada ciclo):
        performative: inform
        body: {"type":         "relatorio_monitor",
               "n_alertas":   int,
               "n_anomalias": int,
               "ts":          str (ISO),
               "top_alertas": lista dos 5 maiores alertas (opcional)}

    Recebe (reconfiguração dinâmica):
        performative: inform
        body: {"type": "reconfigurar", ...chaves de configuração...} (jsonpickle)
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
from scripts.anomaly_detector import carregar_historico, detetar_anomalias
from scripts.price_monitor import (
    detetar_mudancas_preco,
    gerar_alertas,
    guardar_alertas,
)


class MonitorAgent(agent.Agent):
    """Agente sensor de monitorização contínua de preços.

    Corre ``price_monitor`` e ``anomaly_detector`` periodicamente e publica
    um relatório resumido ao OrchestratorAgent depois de cada análise.
    """

    DEFAULT_CONFIG: dict = {
        "periodo": 21600,           # 6h em segundos (alinhado com os scrapers)
        "janela_horas": 24,         # janela do monitor de preços
        "threshold_pct": 2.0,       # variação mínima (%) para gerar alerta
        "desvios_iqr": 3.0,         # multiplicador IQR para anomalias
        "variacao_max_pct": 0.5,    # variação abrupta máxima (50%)
        "max_gap_dias": 3,          # gap máximo (dias) para critério variação abrupta
        "guardar_alertas": True,    # guardar alertas em CSV
        "ativo": True,
    }

    def __init__(
        self,
        jid: str,
        password: str,
        orchestrator_jid: str | None = None,
    ) -> None:
        super().__init__(jid, password)
        self.orchestrator_jid = orchestrator_jid
        self.config = dict(self.DEFAULT_CONFIG)
        self.monitor_behaviour: PeriodicBehaviour | None = None

    # ------------------------------------------------------------------
    # Comportamento de receção de reconfiguração dinâmica
    # ------------------------------------------------------------------

    class ReceiveBehaviour(CyclicBehaviour):
        """Aguarda mensagens de reconfiguração dinâmica do agente."""

        async def run(self) -> None:
            msg = await self.receive(timeout=10)
            if not msg:
                return

            if msg.get_metadata("performative") != "inform":
                return

            try:
                data = jsonpickle.decode(msg.body)
                # Só aceita reconfiguração se o body tiver "type": "reconfigurar".
                # Sem este campo, a mensagem é ignorada (pode ser um relatório de
                # outro agente cujo body contenha acidentalmente chaves homónimas).
                if data.get("type") != "reconfigurar":
                    return

                new_config = {k: v for k, v in data.items() if k != "type"}
                print(f"[MonitorAgent] Nova configuração recebida: {new_config}")
                self.agent.config.update(new_config)

                # Reiniciar MonitorBehaviour se o período foi alterado
                if "periodo" in new_config and self.agent.monitor_behaviour:
                    self.agent.remove_behaviour(self.agent.monitor_behaviour)
                    novo_b = self.agent.MonitorBehaviour(
                        period=self.agent.config["periodo"]
                    )
                    self.agent.monitor_behaviour = novo_b
                    self.agent.add_behaviour(novo_b)
                    print("[MonitorAgent] MonitorBehaviour reiniciado com novo período.")

            except Exception as exc:
                print(f"[MonitorAgent] Erro ao processar reconfiguração: {exc}")

    # ------------------------------------------------------------------
    # Comportamento periódico de monitorização
    # ------------------------------------------------------------------

    class MonitorBehaviour(PeriodicBehaviour):
        """Executa análise de preços e anomalias e reporta ao orquestrador."""

        async def run(self) -> None:
            config = self.agent.config

            if not config.get("ativo", True):
                print("[MonitorAgent] Monitorização desativada. Ciclo ignorado.")
                return

            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"\n[MonitorAgent] Ciclo de monitorização — {ts}")

            n_alertas = 0
            n_anomalias = 0
            top_alertas = []

            # ----------------------------------------------------------
            # 1. Monitor de preços (bloqueante — corre em thread separada)
            # ----------------------------------------------------------
            try:
                janela_horas = config.get("janela_horas", 24)
                threshold_pct = config.get("threshold_pct", 2.0)
                guardar = config.get("guardar_alertas", True)

                def _correr_monitor() -> tuple:
                    df_m = detetar_mudancas_preco(janela_horas=janela_horas)
                    als = gerar_alertas(df_m, threshold_pct=threshold_pct)
                    cam = guardar_alertas(als) if guardar and len(als) > 0 else None
                    return als, cam

                alertas, caminho = await asyncio.to_thread(_correr_monitor)
                n_alertas = len(alertas)

                if n_alertas > 0:
                    print(f"[MonitorAgent] {n_alertas} alerta(s) de preço detetado(s).")
                    if caminho:
                        print(f"[MonitorAgent] Alertas guardados em: {caminho}")

                    # Top 5 alertas para incluir no relatório
                    cols = ["nome_padronizado", "loja", "variacao_pct", "tipo_alerta"]
                    top_alertas = (
                        alertas[cols]
                        .head(5)
                        .to_dict(orient="records")
                    )
                else:
                    print("[MonitorAgent] Sem alterações de preço relevantes.")

            except Exception as exc:
                print(f"[MonitorAgent] Erro no monitor de preços: {exc}")

            # ----------------------------------------------------------
            # 2. Detetor de anomalias (bloqueante — corre em thread separada)
            # ----------------------------------------------------------
            try:
                desvios_iqr = config.get("desvios_iqr", 3.0)
                variacao_max_pct = config.get("variacao_max_pct", 0.5)
                max_gap_dias = config.get("max_gap_dias", 3)

                def _correr_detector() -> object:
                    df_h = carregar_historico()
                    return detetar_anomalias(
                        df_h,
                        desvios_iqr=desvios_iqr,
                        variacao_max_pct=variacao_max_pct,
                        max_gap_dias=max_gap_dias,
                    )

                df_anomalias = await asyncio.to_thread(_correr_detector)
                n_anomalias = 0 if df_anomalias.empty else len(df_anomalias)

                if n_anomalias > 0:
                    print(f"[MonitorAgent] {n_anomalias} anomalia(s) detetada(s).")
                    iqr_n = (df_anomalias["criterio"] == "iqr").sum()
                    var_n = (df_anomalias["criterio"] == "variacao_abrupta").sum()
                    print(f"  IQR: {iqr_n} | Variação abrupta: {var_n}")
                else:
                    print("[MonitorAgent] Sem anomalias de preço detetadas.")

            except Exception as exc:
                print(f"[MonitorAgent] Erro no detetor de anomalias: {exc}")

            # ----------------------------------------------------------
            # 3. Notificar OrchestratorAgent
            # ----------------------------------------------------------
            if self.agent.orchestrator_jid:
                await self._notificar_orquestrador(n_alertas, n_anomalias, top_alertas)

        async def _notificar_orquestrador(
            self,
            n_alertas: int,
            n_anomalias: int,
            top_alertas: list[dict],
        ) -> None:
            """Envia resumo do ciclo de monitorização ao OrchestratorAgent."""
            payload = {
                "type": "relatorio_monitor",
                "n_alertas": n_alertas,
                "n_anomalias": n_anomalias,
                "top_alertas": top_alertas,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            msg = Message(to=self.agent.orchestrator_jid)
            msg.set_metadata("performative", "inform")
            msg.body = jsonpickle.encode(payload)
            await self.send(msg)
            print("[MonitorAgent] Relatório enviado ao OrchestratorAgent.")

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        periodo = self.config.get("periodo", self.DEFAULT_CONFIG["periodo"])
        orquestrador = self.orchestrator_jid or "—"
        print(f"[MonitorAgent] A iniciar (período={periodo}s, orquestrador={orquestrador})...")
        self.add_behaviour(self.ReceiveBehaviour())
        self.monitor_behaviour = self.MonitorBehaviour(period=periodo)
        self.add_behaviour(self.monitor_behaviour)
