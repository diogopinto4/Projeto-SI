"""
Estudo de ablação de features do LSTM global.

Mede o **impacto de cada feature de input** na qualidade da previsão. Para
cada uma das 15 features do modelo, "zeramos" essa feature no conjunto de
validação (sem retreinar) e re-medimos o RMSE médio. A degradação face ao
baseline (modelo sem perturbação) é uma proxy da importância da feature.

Esta abordagem é mais barata que feature permutation (que requer permutações
aleatórias por feature × por amostra) e é justa para um modelo já treinado —
não temos de retreinar 15 vezes.

**Limitação assumida**: zerar não é equivalente a remover a feature em
treino; o modelo pode ter aprendido a confiar em valores não-zero. Mas a
ordenação relativa entre features é informativa, e é o que nos interessa para
o relatório.

Uso::

    python scripts/feature_ablation.py
    python scripts/feature_ablation.py --output data/generated/feature_ablation.json

Output:
    JSON em ``data/generated/feature_ablation.json`` com:
    - ``baseline``: RMSE médio do modelo sem ablação
    - ``ablacao``: lista de ``{feature, rmse_medio, delta_rmse, delta_pct}``
      ordenada por delta_rmse descendente (mais importante no topo)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.price_predictor import (
    FEATURE_COLS,
    _carregar_artefactos,
    calcular_metricas,
    carregar_dataset,
    preparar_serie,
    DEVICE,
)


def _avaliar_com_feature_zerada(
    model: torch.nn.Module,
    scalers: dict,
    df,
    janela: int,
    feature_idx: int | None,
) -> float:
    """Avalia o modelo no conjunto de validação, opcionalmente zerando uma feature.

    Args:
        model: Modelo LSTM treinado, em modo eval.
        scalers: Dicionário ``{produto_id: MinMaxScaler}``.
        df: Dataset completo (saída de :func:`carregar_dataset`).
        janela: Tamanho da janela de contexto.
        feature_idx: Índice da feature a zerar (em coords. **normalizadas**).
            ``None`` para baseline (sem ablação).

    Returns:
        RMSE médio (em euros) sobre todos os produtos elegíveis. NaN se
        nenhum produto produzir validação.
    """
    rmses: list[float] = []
    for produto_id, scaler in scalers.items():
        df_p = df[df["id_produto_loja"] == produto_id].sort_values("dia").copy()
        df_p_clean = df_p.dropna(subset=FEATURE_COLS)
        if len(df_p_clean) < janela * 2 + 1:
            continue

        n_val = janela
        df_val = df_p.iloc[-n_val - janela:]
        X_val, y_val = preparar_serie(df_val, scaler, janela, fit_scaler=False, apenas_observados=True)
        if len(X_val) == 0:
            continue

        if feature_idx is not None:
            # Zerar a feature alvo (em todas as posições temporais)
            X_val = X_val.copy()
            X_val[:, :, feature_idx] = 0.0

        with torch.no_grad():
            preds = model(torch.tensor(X_val).to(DEVICE)).cpu().numpy()

        m = calcular_metricas(y_val, preds, scaler, produto_id)
        if m["rmse_euros"] == m["rmse_euros"]:   # not NaN
            rmses.append(m["rmse_euros"])

    return float(np.mean(rmses)) if rmses else float("nan")


def correr_ablacao(caminho_csv: str) -> dict:
    """Corre o estudo de ablação completo.

    Para cada feature em :data:`FEATURE_COLS`, mede o RMSE médio com essa
    feature zerada e compara com o baseline. Devolve a lista ordenada por
    impacto (delta_rmse descendente).

    Args:
        caminho_csv: Caminho para o dataset de forecasting.

    Returns:
        Dict com ``baseline``, ``ablacao`` (lista ordenada) e ``metadata``.
    """
    artefactos = _carregar_artefactos()
    if artefactos is None:
        raise RuntimeError(
            "Modelo não treinado. Corre `python models/price_predictor.py --treinar` primeiro."
        )

    model = artefactos["model"]
    scalers = artefactos["scalers"]
    janela = artefactos["janela"]
    model.eval()

    print(f"[ABLAÇÃO] A carregar dataset: {caminho_csv}")
    df = carregar_dataset(caminho_csv)
    print(f"[ABLAÇÃO] {len(scalers)} produtos treinados | janela={janela}d | features={len(FEATURE_COLS)}")

    # ----- Baseline -----
    print("[ABLAÇÃO] A avaliar baseline (sem ablação)...")
    rmse_base = _avaliar_com_feature_zerada(model, scalers, df, janela, None)
    print(f"[ABLAÇÃO] RMSE baseline: {rmse_base:.4f} EUR")

    # ----- Ablação por feature -----
    resultados: list[dict] = []
    for idx, nome in enumerate(FEATURE_COLS):
        rmse_abl = _avaliar_com_feature_zerada(model, scalers, df, janela, idx)
        delta = rmse_abl - rmse_base
        delta_pct = (delta / rmse_base * 100) if rmse_base > 0 else 0.0
        print(f"  [{idx:2d}] {nome:25s}  RMSE={rmse_abl:.4f}  Δ={delta:+.4f} ({delta_pct:+5.1f}%)")
        resultados.append({
            "feature":      nome,
            "rmse_euros":   round(rmse_abl, 4),
            "delta_rmse":   round(delta, 4),
            "delta_pct":    round(delta_pct, 2),
        })

    resultados.sort(key=lambda x: x["delta_rmse"], reverse=True)

    return {
        "metadata": {
            "n_produtos":  len(scalers),
            "janela":      janela,
            "n_features":  len(FEATURE_COLS),
            "gerado_em":   datetime.now().isoformat(timespec="seconds"),
        },
        "baseline":  {"rmse_euros": round(rmse_base, 4)},
        "ablacao":   resultados,
    }


def main() -> None:
    """Ponto de entrada CLI da ablação de features."""
    parser = argparse.ArgumentParser(
        description="Estudo de ablação de features do LSTM global.",
    )
    parser.add_argument(
        "--dataset", type=str,
        default=str(Path(__file__).parent.parent / "data/generated/forecasting_dataset.csv"),
        help="Caminho para o forecasting_dataset.csv.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent.parent / "data/generated/feature_ablation.json"),
        help="Ficheiro JSON de output.",
    )
    args = parser.parse_args()

    resultado = correr_ablacao(args.dataset)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print("\n[ABLAÇÃO] Top 5 features mais importantes (maior aumento de RMSE quando zeradas):")
    for r in resultado["ablacao"][:5]:
        print(f"  {r['feature']:25s}  Δ RMSE={r['delta_rmse']:+.4f} ({r['delta_pct']:+5.1f}%)")

    print(f"\n[ABLAÇÃO] Resultados guardados em: {out_path}")


if __name__ == "__main__":
    main()
