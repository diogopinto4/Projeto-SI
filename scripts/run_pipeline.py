"""
Pipeline completo de atualização de dados e reavaliação do modelo.

Encadeia todos os scripts do projeto numa única execução usando subprocess,
de modo a poder ser chamado manualmente ou pelo scheduler automático.

Ordem de execução:
  1. Ingestão dos ficheiros JSON/CSV mais recentes dos scrapers (se --input)
  2. Reconstrução do forecasting dataset a partir da BD
  3. Monitorização de mudanças de preço e geração de alertas
  4. Re-treino do modelo LSTM (apenas se --retreinar)
  5. Avaliação do modelo LSTM (se o modelo existir)

Uso::

    # Execução normal após cada ciclo de scraping
    python scripts/run_pipeline.py --input "scrapers/output/*.json"

    # Com re-treino do modelo
    python scripts/run_pipeline.py --input "scrapers/output/*.json" --retreinar

    # Só rebuild do dataset e avaliação (sem nova ingestão)
    python scripts/run_pipeline.py --so-dataset
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def correr_passo(descricao: str, comando: list[str]) -> bool:
    """Executa um passo do pipeline e reporta sucesso ou falha no terminal.

    O stdout e stderr do subprocesso são herdados (não capturados), pelo que
    o output de cada passo aparece directamente no terminal.

    Args:
        descricao: Texto a mostrar antes de executar o comando.
        comando: Lista de strings que forma o comando (passada a subprocess.run).

    Returns:
        ``True`` se o processo terminar com exit code 0, ``False`` caso contrário.
    """
    print(f"\n{'─'*60}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {descricao}")
    print(f"{'─'*60}")

    inicio = time.time()
    resultado = subprocess.run(comando, capture_output=False)
    duracao = time.time() - inicio

    if resultado.returncode != 0:
        print(f"\n  [FALHOU] em {duracao:.1f}s (exit code {resultado.returncode})")
        return False

    print(f"\n  [OK] Concluido em {duracao:.1f}s")
    return True


def main() -> None:
    """Ponto de entrada CLI do pipeline completo."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo: ingesta -> dataset -> monitorizacao -> avaliacao.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Padrao glob para ficheiros a ingerir (ex: 'scrapers/output/*.json').",
    )
    parser.add_argument(
        "--retreinar",
        action="store_true",
        help="Re-treinar o modelo LSTM apos reconstrucao do dataset.",
    )
    parser.add_argument(
        "--so-dataset",
        action="store_true",
        help="Saltar ingestao — so reconstruir dataset e avaliar.",
    )
    parser.add_argument(
        "--janela",
        type=int,
        default=7,
        help="Janela temporal (dias) para split treino/validacao no LSTM (default: 7).",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=100,
        help="Numero maximo de epocas de treino do LSTM (default: 100).",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=5,
        help="Minimo de observacoes por produto para incluir no treino (default: 5).",
    )
    parser.add_argument(
        "--paciencia",
        type=int,
        default=15,
        help="Paciencia do early stopping em epocas (default: 15).",
    )
    args = parser.parse_args()

    python = sys.executable
    raiz = Path(__file__).parent.parent
    dataset_path = str(raiz / "data" / "generated" / "forecasting_dataset.csv")

    inicio_total = time.time()
    print(f"\n{'='*60}")
    print(f"  PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    passos_ok: list[tuple[str, bool]] = []

    # 1. Ingestão
    if args.input and not args.so_dataset:
        ok = correr_passo(
            "Ingestao de dados dos scrapers",
            [python, str(raiz / "scripts" / "ingest.py"), "--input", args.input],
        )
        passos_ok.append(("Ingestao", ok))
        if not ok:
            print("\n[PIPELINE] Ingestao falhou — a continuar com dados existentes na BD...")

    # 2. Reconstrução do dataset
    ok = correr_passo(
        "Reconstrucao do forecasting dataset",
        [python, str(raiz / "scripts" / "build_forecasting_dataset.py"),
         "--output", dataset_path],
    )
    passos_ok.append(("Dataset", ok))

    # 3. Monitorização de preços
    ok = correr_passo(
        "Monitorizacao de mudancas de preco (ultimas 24h)",
        [python, str(raiz / "scripts" / "price_monitor.py"),
         "--horas", "24", "--guardar"],
    )
    passos_ok.append(("Monitor", ok))

    # 4. Re-treino (opcional)
    if args.retreinar:
        ok = correr_passo(
            "Re-treino do modelo LSTM",
            [
                python, str(raiz / "models" / "price_predictor.py"),
                "--treinar",
                "--dataset",   dataset_path,
                "--janela",    str(args.janela),
                "--epocas",    str(args.epocas),
                "--min-obs",   str(args.min_obs),
                "--paciencia", str(args.paciencia),
            ],
        )
        passos_ok.append(("Treino LSTM", ok))

    # 5. Avaliação do modelo (só se o modelo já existir)
    model_path = raiz / "models" / "saved_models" / "lstm_global.pt"
    if model_path.exists():
        ok = correr_passo(
            "Avaliacao do modelo LSTM",
            [python, str(raiz / "models" / "price_predictor.py"),
             "--avaliar", "--dataset", dataset_path],
        )
        passos_ok.append(("Avaliacao LSTM", ok))
    else:
        print("\n[PIPELINE] Modelo LSTM nao encontrado — usa --retreinar para treinar.")
        passos_ok.append(("Avaliacao LSTM", False))

    # Resumo final
    duracao_total = time.time() - inicio_total
    print(f"\n{'='*60}")
    print(f"  RESUMO DO PIPELINE ({duracao_total:.1f}s total)")
    print(f"{'='*60}")
    for nome, ok in passos_ok:
        estado = "[OK]" if ok else "[FALHOU]"
        print(f"  {estado} {nome}")
    print()


if __name__ == "__main__":
    main()
