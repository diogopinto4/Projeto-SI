"""
Agendador de pipeline periódico — Sensorização e Ambiente.

Executa automaticamente o ciclo scraping → ingestão → monitorização através
de ``run_pipeline.py``, com opção de re-treino do LSTM a intervalos regulares.

Dois modos de agendamento (APScheduler):
- **Intervalo fixo** (``--intervalo N``): executa de N em N horas.
- **Hora fixa** (``--hora HH:MM``): executa diariamente à hora indicada.

Todo o output é registado em ``data/logs/scheduler.log`` e também no terminal.

Uso::

    python scripts/scheduler.py                      # a cada 24h, sem re-treino
    python scripts/scheduler.py --intervalo 12       # a cada 12 horas
    python scripts/scheduler.py --hora 03:00         # diariamente às 03:00
    python scripts/scheduler.py --retreinar          # inclui re-treino LSTM
    python scripts/scheduler.py --so-uma-vez         # executa uma vez e sai
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJETO_ROOT = Path(__file__).parent.parent

#: Directório de logs. Criado automaticamente se não existir.
LOG_DIR = PROJETO_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def executar_pipeline(com_treino: bool = False) -> bool:
    """Lança ``run_pipeline.py`` como subprocesso e aguarda o resultado.

    Args:
        com_treino: Se ``True``, passa ``--retreinar`` ao pipeline para
            incluir re-treino do LSTM neste ciclo.

    Returns:
        ``True`` se o processo terminar com exit code 0, ``False`` caso
        contrário (erro ou timeout).
    """
    log.info("=== Inicio do ciclo de pipeline agendado ===")
    inicio = datetime.now()

    cmd = [sys.executable, str(PROJETO_ROOT / "scripts" / "run_pipeline.py")]
    if com_treino:
        cmd.append("--retreinar")   # flag correta em run_pipeline.py

    try:
        resultado = subprocess.run(
            cmd,
            cwd=str(PROJETO_ROOT),
            timeout=3600,   # máximo 1 hora por execução
        )
        duracao = (datetime.now() - inicio).total_seconds()
        if resultado.returncode == 0:
            log.info(f"Ciclo concluido com sucesso em {duracao:.0f}s")
            return True
        else:
            log.error(f"Ciclo terminou com erro (codigo {resultado.returncode}) em {duracao:.0f}s")
            return False
    except subprocess.TimeoutExpired:
        log.error("Pipeline excedeu o limite de 1 hora. A terminar.")
        return False
    except Exception as exc:
        log.error(f"Erro inesperado ao executar pipeline: {exc}")
        return False


def main() -> None:
    """Ponto de entrada CLI do agendador."""
    parser = argparse.ArgumentParser(
        description="Agendador de pipeline de precos de supermercado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--intervalo",
        type=int,
        default=24,
        help="Intervalo entre execucoes em horas (default: 24). Ignorado se --hora for usado.",
    )
    parser.add_argument(
        "--hora",
        type=str,
        default=None,
        metavar="HH:MM",
        help="Hora diaria de execucao, ex: '03:00'. Usa fuso Europe/Lisbon.",
    )
    parser.add_argument(
        "--retreinar",
        action="store_true",
        help="Incluir re-treino do LSTM em cada ciclo (passa --retreinar ao run_pipeline.py).",
    )
    parser.add_argument(
        "--so-uma-vez",
        action="store_true",
        help="Executar uma unica vez e sair (util para testes sem agendamento).",
    )
    args = parser.parse_args()

    # Modo de execução única — sem APScheduler
    if args.so_uma_vez:
        log.info("Modo de execucao unica — sem agendamento.")
        sucesso = executar_pipeline(com_treino=args.retreinar)
        sys.exit(0 if sucesso else 1)

    # Importar APScheduler apenas quando necessário para evitar ImportError
    # em ambientes sem a dependência instalada
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        log.error("APScheduler nao instalado. Corre: pip install apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="Europe/Lisbon")

    if args.hora:
        hora_str, min_str = args.hora.split(":")
        trigger = CronTrigger(
            hour=int(hora_str), minute=int(min_str), timezone="Europe/Lisbon"
        )
        log.info(f"Agendado para executar diariamente as {args.hora} (hora de Lisboa).")
    else:
        trigger = IntervalTrigger(hours=args.intervalo)
        log.info(f"Agendado para executar a cada {args.intervalo} hora(s).")

    scheduler.add_job(
        executar_pipeline,
        trigger=trigger,
        kwargs={"com_treino": args.retreinar},
        id="pipeline_precos",
        name="Pipeline de Precos de Supermercado",
        misfire_grace_time=600,  # executa até 10 min depois se falhar o horário exato
    )

    log.info("Agendador iniciado. Prima Ctrl+C para parar.")
    log.info("A executar o primeiro ciclo imediatamente...")
    executar_pipeline(com_treino=args.retreinar)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Agendador parado pelo utilizador.")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
