"""
Modelos de previsão baseline para comparação com o LSTM.

Implementa 4 baselines clássicos para validar quantitativamente o valor
acrescentado do LSTM. Estes baselines servem de **piso de qualidade** —
se o LSTM não os superar, a sua complexidade adicional não se justifica.

Modelos disponíveis:

1. :func:`prever_naive` — "amanhã = preço de hoje" (zero-knowledge baseline).
2. :func:`prever_media_movel` — média dos últimos N dias.
3. :func:`prever_arima` — ARIMA(p,d,q) clássico via `statsmodels`.
4. (LSTM em ``price_predictor.py`` — referência principal.)

Todos os modelos partilham a mesma interface::

    def prever_X(serie: np.ndarray, horizonte: int) -> np.ndarray

onde ``serie`` é um array 1D com a história de preços (em euros reais, não
normalizados), e o retorno é um array 1D de comprimento ``horizonte``.

Decisões de design (justificadas no relatório):

* **Métricas iguais às do LSTM** (RMSE/MAE/MAPE em euros reais). Comparação
  apples-to-apples.

* **Split temporal idêntico ao LSTM** (últimos ``janela=7`` dias = validação).
  Sem isto, comparações entre modelos não seriam justas.

* **ARIMA(1,1,1) "all-purpose"** em vez de ``auto_arima``: pmdarima é 10×
  mais lento, e (1,1,1) — uma diferenciação + AR(1) + MA(1) — é o ponto de
  partida canónico da literatura. Quando ``statsmodels`` não converge para
  um produto específico, retornamos NaN (excluímos esse produto das médias).

* **Médias móveis de 3 e 7 dias** porque captam ciclos de curto prazo
  (3 dias) e potenciais ciclos semanais comerciais (7 dias). Mais dias
  amaciaria demais para séries de validação de 7 pontos.
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def prever_naive(serie: np.ndarray, horizonte: int) -> np.ndarray:
    """Baseline naive: prediz o último valor observado repetido ``horizonte`` vezes.

    Args:
        serie: História de preços (1D, comprimento >= 1).
        horizonte: Número de dias a prever (>= 1).

    Returns:
        Array de shape ``(horizonte,)`` com o último preço repetido.
    """
    ultimo = serie[-1]
    return np.full(horizonte, ultimo, dtype=float)


def prever_media_movel(
    serie: np.ndarray,
    horizonte: int,
    janela: int = 7,
) -> np.ndarray:
    """Baseline de média móvel: prediz a média dos últimos ``janela`` valores.

    A previsão é **constante** ao longo do horizonte (não actualiza a média
    com previsões anteriores — isso seria uma cadeia de Markov, mais complexa
    e tipicamente pior).

    Args:
        serie: História de preços (1D).
        horizonte: Número de dias a prever.
        janela: Quantos dias incluir na média. Se a série for mais curta que
            ``janela``, usa-se toda a série.

    Returns:
        Array de shape ``(horizonte,)`` com a média repetida.
    """
    janela_efetiva = min(janela, len(serie))
    if janela_efetiva == 0:
        return np.zeros(horizonte, dtype=float)
    media = float(np.mean(serie[-janela_efetiva:]))
    return np.full(horizonte, media, dtype=float)


def prever_arima(
    serie: np.ndarray,
    horizonte: int,
    ordem: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """Baseline ARIMA(p,d,q) clássico via ``statsmodels``.

    Args:
        serie: História de preços (1D, recomendado >= 10 pontos).
        horizonte: Número de dias a prever.
        ordem: Tupla ``(p, d, q)``. Default ``(1, 1, 1)`` — 1 lag AR, 1
            diferenciação, 1 lag MA. É o ponto de partida canónico da
            literatura e funciona bem para séries com tendência ligeira.

    Returns:
        Array de shape ``(horizonte,)`` com as previsões. Se o modelo não
        convergir, devolve um array de NaN (para sinalizar falha sem
        rebentar o pipeline).
    """
    from statsmodels.tsa.arima.model import ARIMA   # import tardio (statsmodels é pesado)

    if len(serie) < sum(ordem) + 1:
        # Sem pontos suficientes para os parâmetros — devolve NaN
        return np.full(horizonte, np.nan, dtype=float)

    try:
        # statsmodels emite avisos verbosos para séries curtas/com pouca
        # variação — silenciamos porque não são erros, só hints estatísticos.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = ARIMA(serie, order=ordem).fit()
        previsao = modelo.forecast(steps=horizonte)
        return np.asarray(previsao, dtype=float)
    except Exception:
        # Convergência falhada, série constante, etc. — degradar silenciosamente
        return np.full(horizonte, np.nan, dtype=float)


# ---------------------------------------------------------------------------
# Métricas — partilhadas entre modelos
# ---------------------------------------------------------------------------

def calcular_metricas_em_euros(
    y_real: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula RMSE, MAE e MAPE em euros reais (sem normalização inversa).

    Os baselines trabalham diretamente em euros (não normalizam), portanto
    esta versão é mais simples que :func:`models.price_predictor.calcular_metricas`
    (que precisa de inverter o scaler).

    NaN em ``y_pred`` (ex: ARIMA que não convergiu) são tratados como falha
    do modelo nessa observação: as métricas retornam NaN para esse produto.

    Args:
        y_real: Array 1D com valores observados (euros).
        y_pred: Array 1D com previsões (euros). Pode conter NaN.

    Returns:
        Dict com ``rmse_euros``, ``mae_euros``, ``mape_pct``.
        Devolve NaN se algum y_pred for NaN ou se y_real estiver vazio.
    """
    if len(y_real) == 0 or np.any(np.isnan(y_pred)):
        return {"rmse_euros": float("nan"),
                "mae_euros":  float("nan"),
                "mape_pct":   float("nan")}

    erro = y_real - y_pred
    rmse = float(np.sqrt(np.mean(erro ** 2)))
    mae  = float(np.mean(np.abs(erro)))
    mask = y_real > 0.01
    if mask.any():
        mape = float(np.mean(np.abs(erro[mask] / y_real[mask])) * 100)
    else:
        mape = float("nan")

    return {
        "rmse_euros": round(rmse, 4),
        "mae_euros":  round(mae, 4),
        "mape_pct":   round(mape, 2),
    }


# ---------------------------------------------------------------------------
# Registo de modelos disponíveis — para iteração no script comparativo
# ---------------------------------------------------------------------------

#: Mapeamento ``nome → (função, kwargs_fixos)``. Cada função tem assinatura
#: ``(serie, horizonte) → np.ndarray``. Permite ao :mod:`scripts.comparar_modelos`
#: iterar sobre todos os baselines uniformemente.
BASELINES: dict[str, tuple] = {
    "naive":     (prever_naive,       {}),
    "media_3":   (prever_media_movel, {"janela": 3}),
    "media_7":   (prever_media_movel, {"janela": 7}),
    "arima_111": (prever_arima,       {"ordem": (1, 1, 1)}),
}


def aplicar_baseline(nome: str, serie: np.ndarray, horizonte: int) -> np.ndarray:
    """Despacha para o baseline indicado.

    Args:
        nome: Chave em :data:`BASELINES`.
        serie: História de preços.
        horizonte: Número de dias a prever.

    Returns:
        Previsão como array 1D.

    Raises:
        KeyError: Se ``nome`` não estiver registado em :data:`BASELINES`.
    """
    fn, kwargs = BASELINES[nome]
    return fn(serie, horizonte, **kwargs)
