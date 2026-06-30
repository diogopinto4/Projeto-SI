"""
Módulo de previsão de preços com LSTM global.

Treina um único LSTM partilhado por todos os produtos (abordagem "global"),
com split temporal por produto para evitar data leakage.

Pipeline::

    CSV (forecasting_dataset.csv)
      → carregar_dataset()
      → split treino/validação por produto (últimos ``janela`` dias = validação)
      → MinMaxScaler por produto (fit apenas no treino)
      → preparar_serie()  →  SerieTemporalDataset
      → LSTMPrecoPredictor (LSTM 2 camadas + Dropout + Linear)
      → treino com early stopping (ReduceLROnPlateau + paciência)
      → avaliar_modelo()  →  RMSE/MAE/MAPE vs baseline naive
      → prever_preco() / prever_preco_com_incerteza()

Uso::

    # Gerar dataset primeiro:
    python scripts/build_forecasting_dataset.py

    # Treinar:
    python models/price_predictor.py --treinar

    # Avaliar:
    python models/price_predictor.py --avaliar

    # Prever (previsao pontual):
    python models/price_predictor.py --prever --produto-id 1 --horizonte 7

    # Prever com intervalos de confiança (Monte Carlo Dropout):
    python models/price_predictor.py --prever --mc --produto-id 1
"""

from __future__ import annotations

import argparse
import pickle
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------

#: Diretório de persistência de modelos, scalers e metadados.
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Cache de artefactos (model + scalers + meta)
# ---------------------------------------------------------------------------
# Invalidado automaticamente quando qualquer ficheiro de artefacto é alterado
# no disco (após retreino). Evita recarregar pesos em cada chamada de inferência.

_artefactos_cache: dict | None = None   # {"model", "scalers", "meta", "janela"}
_artefactos_mtime: float = 0.0          # mtime combinado dos 3 ficheiros


def _mtime_combinado() -> float:
    """Devolve a soma dos mtimes de lstm_global.pt, scalers.pkl e model_meta.pkl.

    Retorna 0.0 se qualquer ficheiro não existir (cache inválida).
    """
    paths = [
        MODELS_DIR / "lstm_global.pt",
        MODELS_DIR / "scalers.pkl",
        MODELS_DIR / "model_meta.pkl",
    ]
    try:
        return sum(p.stat().st_mtime for p in paths)
    except FileNotFoundError:
        return 0.0


def _carregar_artefactos() -> dict | None:
    """Carrega model + scalers + meta do disco, com cache por mtime.

    Retorna None se qualquer ficheiro não existir.  Em caso de sucesso,
    actualiza o cache global e devolve o dicionário de artefactos.
    """
    global _artefactos_cache, _artefactos_mtime

    mtime_atual = _mtime_combinado()
    if mtime_atual == 0.0:
        return None  # ficheiros ainda não existem

    if _artefactos_cache is not None and mtime_atual == _artefactos_mtime:
        return _artefactos_cache  # cache válida

    # Carrega do disco
    model_path = MODELS_DIR / "lstm_global.pt"
    scalers_path = MODELS_DIR / "scalers.pkl"
    meta_path = MODELS_DIR / "model_meta.pkl"

    for p in [model_path, scalers_path, meta_path]:
        if not p.exists():
            print(f"[ERRO] Ficheiro não encontrado: {p}\nCorre --treinar primeiro.")
            return None

    with open(scalers_path, "rb") as f:
        scalers: dict = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta: dict = pickle.load(f)

    model = LSTMPrecoPredictor(input_size=N_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    _artefactos_cache = {
        "model": model,
        "scalers": scalers,
        "meta": meta,
        "janela": meta["janela"],
    }
    _artefactos_mtime = mtime_atual
    return _artefactos_cache


# ---------------------------------------------------------------------------
# Constantes de features
# ---------------------------------------------------------------------------

#: Ordem canónica das features de input do LSTM.
#: Deve estar sincronizada com FEATURE_COLS em scripts/build_forecasting_dataset.py.
FEATURE_COLS: list[str] = [
    "preco",
    "promo_flag",
    "weekday",
    "weekofyear",
    "month",
    "dayofmonth",
    "cadeia_id",
    "dias_desde_ultima_obs",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "media_movel_3",
    "media_movel_7",
    "desvio_movel_7",
]

TARGET_COL = "target_t_plus_1"
N_FEATURES = len(FEATURE_COLS)  # 15

# Índices das features individuais — usados para preencher arrays raw sem
# depender de dicionários (mais rápido em loops de inferência).
PRECO_IDX      = FEATURE_COLS.index("preco")
PROMO_IDX      = FEATURE_COLS.index("promo_flag")
WEEKDAY_IDX    = FEATURE_COLS.index("weekday")
WEEKOFYEAR_IDX = FEATURE_COLS.index("weekofyear")
MONTH_IDX      = FEATURE_COLS.index("month")
DAY_IDX        = FEATURE_COLS.index("dayofmonth")
CADEIA_IDX     = FEATURE_COLS.index("cadeia_id")
DIAS_OBS_IDX   = FEATURE_COLS.index("dias_desde_ultima_obs")
LAG1_IDX       = FEATURE_COLS.index("lag_1")
LAG2_IDX       = FEATURE_COLS.index("lag_2")
LAG3_IDX       = FEATURE_COLS.index("lag_3")
LAG7_IDX       = FEATURE_COLS.index("lag_7")
MM3_IDX        = FEATURE_COLS.index("media_movel_3")
MM7_IDX        = FEATURE_COLS.index("media_movel_7")
DM7_IDX        = FEATURE_COLS.index("desvio_movel_7")


# ---------------------------------------------------------------------------
# Carregamento e preparação do dataset
# ---------------------------------------------------------------------------

def carregar_dataset(caminho_csv: str) -> pd.DataFrame:
    """Carrega o CSV gerado por ``build_forecasting_dataset.py`` e valida as colunas.

    Args:
        caminho_csv: Caminho para o ficheiro CSV de forecasting.

    Returns:
        DataFrame ordenado por (id_produto_loja, dia).

    Raises:
        ValueError: Se alguma coluna obrigatória estiver ausente do CSV.
        FileNotFoundError: Se o ficheiro não existir.
    """
    df = pd.read_csv(caminho_csv, parse_dates=["dia"])

    colunas_necessarias = FEATURE_COLS + [TARGET_COL, "id_produto_loja"]
    ausentes = [c for c in colunas_necessarias if c not in df.columns]
    if ausentes:
        raise ValueError(
            f"Colunas em falta no dataset: {ausentes}\n"
            "Regenera o dataset com: python scripts/build_forecasting_dataset.py"
        )

    return df.sort_values(["id_produto_loja", "dia"]).reset_index(drop=True)


def preparar_serie(
    df_produto: pd.DataFrame,
    scaler: MinMaxScaler,
    janela: int,
    fit_scaler: bool = True,
    apenas_observados: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrói janelas deslizantes normalizadas para um único produto.

    Cada janela de ``janela`` dias produz um sample de treino/validação.
    O target de cada sample é o preço normalizado do dia ``janela+1``
    (i.e., o dia imediatamente a seguir à janela).

    Args:
        df_produto: DataFrame filtrado para um único ``id_produto_loja``.
        scaler: Scaler pré-instanciado. Ajustado nos dados se ``fit_scaler=True``,
            ou apenas aplicado se ``fit_scaler=False`` (validação/teste).
        janela: Número de dias de contexto por sample.
        fit_scaler: Se ``True``, chama ``scaler.fit_transform``; caso contrário
            ``scaler.transform``. Deve ser ``False`` para dados de validação.
        apenas_observados: Se ``True``, descarta janelas cujo dia-alvo foi
            forward-filled (``foi_observado_no_dia == 0``). Evita que o modelo
            aprenda a prever cópias do último preço observado em dias sem
            scraping. Os dias sem observação continuam a contribuir para o
            **contexto** das janelas — só não são usados como target.

    Returns:
        Tupla ``(X, y)`` onde:
        - ``X`` tem shape ``(n_samples, janela, N_FEATURES)``, dtype float32.
        - ``y`` tem shape ``(n_samples,)``, dtype float32 (preço normalizado).
        Ambos são arrays vazios se a série for demasiado curta.
    """
    df = df_produto.dropna(subset=FEATURE_COLS).copy()
    if len(df) < janela + 1:
        return (
            np.empty((0, janela, N_FEATURES), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    features = df[FEATURE_COLS].values.astype(np.float32)
    features_norm = scaler.fit_transform(features) if fit_scaler else scaler.transform(features)

    observados = (
        df["foi_observado_no_dia"].values.astype(int)
        if apenas_observados and "foi_observado_no_dia" in df.columns
        else None
    )

    X, y = [], []
    for i in range(len(features_norm) - janela):
        if observados is not None and observados[i + janela] == 0:
            continue
        X.append(features_norm[i: i + janela])
        y.append(features_norm[i + janela, PRECO_IDX])

    if not X:
        return (
            np.empty((0, janela, N_FEATURES), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class SerieTemporalDataset(Dataset):
    """Dataset PyTorch que encapsula matrizes X e y numpy como tensores.

    Args:
        X: Array de shape ``(n, janela, n_features)``.
        y: Array de shape ``(n,)`` com os targets.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Modelo LSTM
# ---------------------------------------------------------------------------

class LSTMPrecoPredictor(nn.Module):
    """LSTM global para previsão de preços de supermercado.

    Arquitectura::

        Input  → LSTM (2 camadas, hidden=64) → Dropout(0.3) → Linear(1)

    O Dropout em modo ``train()`` é aproveitado para Monte Carlo Dropout
    durante a inferência probabilística.

    Args:
        input_size: Número de features de input por timestep (default: N_FEATURES).
        hidden_size: Dimensão do estado oculto LSTM (default: 64).
        num_layers: Número de camadas LSTM empilhadas (default: 2).
        dropout: Taxa de dropout entre camadas LSTM e antes da camada linear
            (default: 0.3). Ignorado entre camadas se ``num_layers == 1``.
    """

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Usa apenas o último timestep do output LSTM.

        Args:
            x: Tensor de shape ``(batch, janela, input_size)``.

        Returns:
            Tensor de shape ``(batch,)`` com o preço normalizado previsto.
        """
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Métricas de avaliação
# ---------------------------------------------------------------------------

def calcular_metricas(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    scaler: MinMaxScaler,
    produto_id: int,
) -> dict:
    """Inverte a normalização e calcula RMSE, MAE e MAPE em euros reais.

    A inversão usa um array dummy de shape ``(n, N_FEATURES)`` com zeros
    em todas as features exceto a posição ``PRECO_IDX``, para que o scaler
    possa inverter correctamente apenas a dimensão de preço.

    Args:
        y_real: Targets normalizados, shape ``(n,)``.
        y_pred: Previsões normalizadas, shape ``(n,)``.
        scaler: Scaler ajustado ao produto.
        produto_id: ID do produto (incluído no dict de resultado).

    Returns:
        Dicionário com: ``produto_id``, ``n_amostras``, ``rmse_euros``,
        ``mae_euros``, ``mape_pct``.
    """
    def _inverter(valores_norm: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(valores_norm), N_FEATURES), dtype=np.float32)
        dummy[:, PRECO_IDX] = valores_norm
        return scaler.inverse_transform(dummy)[:, PRECO_IDX]

    real = _inverter(y_real)
    pred = _inverter(y_pred)

    rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
    mae = float(np.mean(np.abs(real - pred)))
    mask = real > 0.01
    mape = float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100) if mask.any() else float("nan")

    return {
        "produto_id": produto_id,
        "n_amostras": len(real),
        "rmse_euros": round(rmse, 4),
        "mae_euros": round(mae, 4),
        "mape_pct":   round(mape, 2),
    }


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar_modelo_global(
    caminho_csv: str,
    janela: int = 7,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    min_observacoes: int = 5,
    paciencia: int = 15,
) -> None:
    """Treina o LSTM global com dados de todos os produtos e guarda os artefactos.

    Estratégia de split temporal por produto:
    - Validação: últimos ``janela`` dias da série de cada produto.
    - Treino: todos os dias anteriores à janela de validação.

    Este split garante que sempre existe validação independentemente do tamanho
    do histórico, e evita data leakage ao nunca usar dados futuros no treino.

    Artefactos guardados em :data:`MODELS_DIR`:
    - ``lstm_global.pt`` — pesos do melhor modelo (por loss de validação).
    - ``scalers.pkl``    — dicionário ``{produto_id: MinMaxScaler}``.
    - ``model_meta.pkl`` — metadados (janela, features, IDs treinados).

    Args:
        caminho_csv: Caminho para o CSV de forecasting.
        janela: Dias de contexto por sample e tamanho da janela de validação.
        epochs: Número máximo de épocas.
        batch_size: Tamanho do batch para o DataLoader.
        lr: Learning rate inicial do Adam.
        min_observacoes: Mínimo de observações sem NaN para incluir um produto.
        paciencia: Número de épocas sem melhoria na loss de validação antes
            do early stopping.
    """
    print(f"[TREINO] A carregar dataset de {caminho_csv}...")
    df = carregar_dataset(caminho_csv)

    todos_X_train, todos_y_train = [], []
    todos_X_val, todos_y_val = [], []
    scalers: dict[int, MinMaxScaler] = {}

    produtos = df["id_produto_loja"].unique()
    ignorados = 0
    print(f"[TREINO] Produtos encontrados: {len(produtos)}")

    for produto_id in produtos:
        df_p = df[df["id_produto_loja"] == produto_id].sort_values("dia").copy()
        df_p_clean = df_p.dropna(subset=FEATURE_COLS)

        # Necessário: mín. observações + contexto de treino + contexto de validação
        min_necessario = min_observacoes + janela * 2 + 1
        if len(df_p_clean) < min_necessario:
            ignorados += 1
            continue

        n_val = janela
        df_train = df_p.iloc[:-n_val]
        df_val = df_p.iloc[-n_val - janela:]  # inclui contexto para a primeira janela de val

        scaler = MinMaxScaler()
        X_train, y_train = preparar_serie(df_train, scaler, janela, fit_scaler=True, apenas_observados=True)
        X_val, y_val = preparar_serie(df_val, scaler, janela, fit_scaler=False, apenas_observados=True)

        if len(X_train) == 0 or len(X_val) == 0:
            ignorados += 1
            continue

        todos_X_train.append(X_train)
        todos_y_train.append(y_train)
        todos_X_val.append(X_val)
        todos_y_val.append(y_val)
        scalers[int(produto_id)] = scaler

    print(f"[TREINO] Produtos com dados suficientes: {len(scalers)} | Ignorados: {ignorados}")

    if not todos_X_train:
        print("[ERRO] Nenhuma serie com dados suficientes. Aguarda mais ciclos de scraping.")
        return

    X_train_all = np.concatenate(todos_X_train, axis=0)
    y_train_all = np.concatenate(todos_y_train, axis=0)
    X_val_all = np.concatenate(todos_X_val, axis=0)
    y_val_all = np.concatenate(todos_y_val, axis=0)

    train_loader = DataLoader(
        SerieTemporalDataset(X_train_all, y_train_all),
        batch_size=batch_size,
        shuffle=True,
    )

    model = LSTMPrecoPredictor(input_size=N_FEATURES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    print(f"[TREINO] Amostras treino: {len(X_train_all)} | Validacao: {len(X_val_all)} | Device: {DEVICE}")
    print(f"[TREINO] Features: {N_FEATURES} | Janela: {janela} dias | Paciencia: {paciencia} epocas")
    print(f"[TREINO] A treinar por ate {epochs} epocas...")

    best_val_loss = float("inf")
    epocas_sem_melhoria = 0

    X_v = torch.tensor(X_val_all).to(DEVICE)
    y_v = torch.tensor(y_val_all).to(DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train_all)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss       = val_loss
            epocas_sem_melhoria = 0
            torch.save(model.state_dict(), MODELS_DIR / "lstm_global.pt")
        else:
            epocas_sem_melhoria += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoca {epoch:3d}/{epochs} | "
                f"Loss treino: {train_loss:.6f} | "
                f"Loss val: {val_loss:.6f} | "
                f"Melhor val: {best_val_loss:.6f}"
            )

        if epocas_sem_melhoria >= paciencia:
            print(f"\n[TREINO] Early stopping na epoca {epoch} (sem melhoria ha {paciencia} epocas).")
            break

    # Guardar artefactos
    with open(MODELS_DIR / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    meta = {
        "janela":             janela,
        "n_features":         N_FEATURES,
        "feature_cols":       FEATURE_COLS,
        "produtos_treinados": list(scalers.keys()),
    }
    with open(MODELS_DIR / "model_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Invalidar cache — os ficheiros acabaram de ser actualizados
    global _artefactos_cache, _artefactos_mtime
    _artefactos_cache = None
    _artefactos_mtime = 0.0

    print(f"\n[TREINO] Concluido. Melhor loss val: {best_val_loss:.6f}")
    print(f"  Modelo:  {MODELS_DIR / 'lstm_global.pt'}")
    print(f"  Scalers: {MODELS_DIR / 'scalers.pkl'} ({len(scalers)} produtos)")


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def avaliar_modelo(caminho_csv: str) -> pd.DataFrame | None:
    """Avalia o modelo guardado no conjunto de validação de cada produto.

    Compara RMSE/MAE/MAPE do LSTM contra um baseline naive ("preço de amanhã
    = preço de hoje") para quantificar o ganho real do modelo.

    Args:
        caminho_csv: Caminho para o CSV de forecasting.

    Returns:
        DataFrame com métricas por produto, ou ``None`` se o modelo não existir
        ou não houver resultados disponíveis.
    """
    artefactos = _carregar_artefactos()
    if artefactos is None:
        print("[ERRO] Modelo nao treinado. Corre --treinar primeiro.")
        return None

    model = artefactos["model"]
    scalers = artefactos["scalers"]
    janela = artefactos["janela"]

    model.eval()

    df = carregar_dataset(caminho_csv)
    resultados = []
    baseline_resultados = []

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

        with torch.no_grad():
            preds = model(torch.tensor(X_val).to(DEVICE)).cpu().numpy()

        resultados.append(calcular_metricas(y_val, preds, scaler, produto_id))

        # Baseline: último preço da janela como previsão
        y_naive = X_val[:, -1, PRECO_IDX]
        baseline_resultados.append(calcular_metricas(y_val, y_naive, scaler, produto_id))

    if not resultados:
        print("[AVALIACAO] Sem resultados disponíveis.")
        return None

    df_res = pd.DataFrame(resultados)
    df_base = pd.DataFrame(baseline_resultados)

    print("\n[AVALIACAO] Metricas por produto — top 20 por RMSE:")
    print(df_res.sort_values("rmse_euros").head(20).to_string(index=False))

    print(f"\n[AVALIACAO] Resumo global ({len(df_res)} produtos):")
    print(f"  RMSE medio : {df_res['rmse_euros'].mean():.4f} EUR")
    print(f"  MAE medio  : {df_res['mae_euros'].mean():.4f} EUR")
    print(f"  MAPE medio : {df_res['mape_pct'].mean():.2f} %")

    print("\n[BASELINE NAIVE] (prever sempre o preco atual):")
    print(f"  RMSE medio : {df_base['rmse_euros'].mean():.4f} EUR")
    print(f"  MAE medio  : {df_base['mae_euros'].mean():.4f} EUR")
    print(f"  MAPE medio : {df_base['mape_pct'].mean():.2f} %")

    rmse_ganho = df_base["rmse_euros"].mean() - df_res["rmse_euros"].mean()
    print(f"\n  Ganho do LSTM vs Naive — RMSE {rmse_ganho:+.4f} EUR")
    if rmse_ganho > 0:
        print("  O LSTM esta a fazer melhor do que o naive.")
    else:
        print("  ATENCAO: O LSTM nao esta a superar o baseline naive.")
        print("  (Normal com poucos dias de historico — aguarda mais scraping.)")

    df_res.to_csv(MODELS_DIR / "avaliacao_val.csv", index=False)
    df_base.to_csv(MODELS_DIR / "avaliacao_baseline.csv", index=False)
    print("\n  Resultados guardados em:")
    print(f"    {MODELS_DIR / 'avaliacao_val.csv'}")
    print(f"    {MODELS_DIR / 'avaliacao_baseline.csv'}")
    return df_res


# ---------------------------------------------------------------------------
# Previsão — construção de features raw
# ---------------------------------------------------------------------------

def _raw_features_para_dia(
    data: date,
    preco_raw: float,
    historico_precos: deque,
    dias_desde_obs: int,
    promo: float = 0.0,
    cadeia_id: int = 0,
) -> np.ndarray:
    """Constrói um vector de features raw para um único dia futuro.

    Usado durante a inferência iterativa para preencher features de lag e
    médias móveis a partir do histórico acumulado de previsões.

    Args:
        data: Data do dia a prever.
        preco_raw: Preço bruto (não normalizado) previsto para este dia.
        historico_precos: Deque com os últimos preços brutos (max 7 elementos).
        dias_desde_obs: Número de dias desde a última observação real.
        promo: Flag de promoção (default: 0.0 — sem promoção).
        cadeia_id: ID inteiro da cadeia (default: 0).

    Returns:
        Array float32 de shape ``(N_FEATURES,)`` com todos os valores preenchidos.
    """
    hist = list(historico_precos)
    lag1 = hist[-1] if len(hist) >= 1 else preco_raw
    lag2 = hist[-2] if len(hist) >= 2 else lag1
    lag3 = hist[-3] if len(hist) >= 3 else lag2
    lag7 = hist[-7] if len(hist) >= 7 else (hist[0] if hist else preco_raw)
    mm3 = float(np.mean(hist[-3:])) if hist else preco_raw
    mm7 = float(np.mean(hist[-7:])) if hist else preco_raw
    dm7 = float(np.std(hist[-7:], ddof=1)) if len(hist) >= 2 else 0.0

    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[PRECO_IDX]      = preco_raw
    row[PROMO_IDX]      = promo
    row[WEEKDAY_IDX]    = data.weekday()
    row[WEEKOFYEAR_IDX] = data.isocalendar().week
    row[MONTH_IDX]      = data.month
    row[DAY_IDX]        = data.day
    row[CADEIA_IDX]     = cadeia_id
    row[DIAS_OBS_IDX]   = dias_desde_obs
    row[LAG1_IDX]       = lag1
    row[LAG2_IDX]       = lag2
    row[LAG3_IDX]       = lag3
    row[LAG7_IDX]       = lag7
    row[MM3_IDX]        = mm3
    row[MM7_IDX]        = mm7
    row[DM7_IDX]        = dm7
    return row


# ---------------------------------------------------------------------------
# Setup de inferência — partilhado por prever_preco e prever_preco_com_incerteza
# ---------------------------------------------------------------------------

def _preparar_inferencia(
    produto_id: int,
    caminho_csv: str,
) -> dict | None:
    """Carrega artefactos + dataset e devolve o contexto necessário à inferência.

    Encapsula validações e setup que são idênticos em :func:`prever_preco` e
    :func:`prever_preco_com_incerteza`: artefactos do modelo, scaler do
    produto, dataset, série limpa, features normalizadas, janela inicial e
    histórico de preços recentes.

    Returns:
        Dicionário com ``model``, ``scaler``, ``janela``, ``features_norm``,
        ``cadeia_id``, ``hist_precos`` (deque maxlen=7), ``ultima_data``,
        ``preco_atual``. ``None`` se o modelo, o produto ou a série não
        estiverem disponíveis (já imprime mensagem de erro).
    """
    artefactos = _carregar_artefactos()
    if artefactos is None:
        print("[ERRO] Modelo nao treinado. Corre --treinar primeiro.")
        return None

    scalers = artefactos["scalers"]
    janela = artefactos["janela"]

    if produto_id not in scalers:
        disponiveis = sorted(scalers.keys())
        print(f"[ERRO] Produto {produto_id} sem scaler guardado.")
        print(f"  Produtos disponiveis ({len(disponiveis)}): {disponiveis[:20]}...")
        return None

    scaler = scalers[produto_id]
    df = carregar_dataset(caminho_csv)
    df_p = df[df["id_produto_loja"] == produto_id].sort_values("dia").copy()

    if df_p.empty:
        print(f"[ERRO] Produto {produto_id} nao encontrado no dataset.")
        return None

    df_p_clean = df_p.dropna(subset=FEATURE_COLS)
    if len(df_p_clean) < janela:
        print(f"[ERRO] Serie demasiado curta (minimo {janela} dias sem NaN).")
        return None

    features_norm = scaler.transform(df_p_clean[FEATURE_COLS].values.astype(np.float32))
    cadeia_id_produto = (
        int(df_p_clean["cadeia_id"].iloc[-1])
        if "cadeia_id" in df_p_clean.columns else 0
    )

    return {
        "model":         artefactos["model"],
        "scaler":        scaler,
        "janela":        janela,
        "features_norm": features_norm,
        "cadeia_id":     cadeia_id_produto,
        "hist_precos":   df_p_clean["preco"].values[-7:].tolist(),
        "ultima_data":   df_p_clean["dia"].max().date(),
        "preco_atual":   float(df_p_clean["preco"].iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Previsão pontual
# ---------------------------------------------------------------------------

def prever_preco(
    produto_id: int,
    caminho_csv: str,
    horizonte: int = 7,
) -> pd.DataFrame | None:
    """Gera previsão determinista de preço para os próximos ``horizonte`` dias.

    Usa o modelo treinado em modo ``eval()`` (dropout desativado).
    Para previsão com intervalos de confiança, usa :func:`prever_preco_com_incerteza`.

    Args:
        produto_id: ``id_produto_loja`` a prever.
        caminho_csv: Caminho para o CSV de forecasting.
        horizonte: Número de dias a prever.

    Returns:
        DataFrame com colunas ``data`` e ``preco_previsto``, ou ``None``
        se o modelo ou o produto não estiverem disponíveis.
    """
    ctx = _preparar_inferencia(produto_id, caminho_csv)
    if ctx is None:
        return None

    model        = ctx["model"]
    scaler       = ctx["scaler"]
    janela       = ctx["janela"]
    cadeia_id_produto = ctx["cadeia_id"]
    ultima_data  = ctx["ultima_data"]
    preco_atual  = ctx["preco_atual"]

    janela_atual = ctx["features_norm"][-janela:].copy()
    hist_precos = deque(ctx["hist_precos"], maxlen=7)

    model.eval()

    previsoes = []
    with torch.no_grad():
        for i in range(horizonte):
            pred_norm = model(torch.tensor(janela_atual).unsqueeze(0).to(DEVICE)).item()

            dummy = np.zeros((1, N_FEATURES), dtype=np.float32)
            dummy[0, PRECO_IDX] = pred_norm
            pred_raw = max(0.01, round(float(scaler.inverse_transform(dummy)[0, PRECO_IDX]), 2))

            proxima_data = ultima_data + timedelta(days=i + 1)
            previsoes.append({"data": proxima_data, "preco_previsto": pred_raw})

            hist_precos.append(pred_raw)
            raw_next = _raw_features_para_dia(
                data=ultima_data + timedelta(days=i + 2),
                preco_raw=pred_raw,
                historico_precos=hist_precos,
                dias_desde_obs=i + 2,
                cadeia_id=cadeia_id_produto,
            )
            janela_atual = np.vstack([janela_atual[1:], scaler.transform(raw_next.reshape(1, -1))[0]])

    df_previsoes = pd.DataFrame(previsoes)
    print(f"\n[PREVISAO] Produto {produto_id} | Preco atual: {preco_atual:.2f}EUR")
    print(f"  Proximos {horizonte} dias:")
    print(df_previsoes.to_string(index=False))
    return df_previsoes


# ---------------------------------------------------------------------------
# Previsão com incerteza (Monte Carlo Dropout)
# ---------------------------------------------------------------------------

def prever_preco_com_incerteza(
    produto_id: int,
    caminho_csv: str,
    horizonte: int = 7,
    n_amostras: int = 50,
) -> dict | None:
    """Previsão com intervalos de confiança via Monte Carlo Dropout.

    Mantém o modelo em modo ``train()`` durante a inferência para que o
    Dropout permaneça ativo, e corre ``n_amostras`` passagens forward
    independentes. A variação entre simulações captura a incerteza epistémica.

    Args:
        produto_id: ``id_produto_loja`` a prever.
        caminho_csv: Caminho para o CSV de forecasting.
        horizonte: Número de dias a prever.
        n_amostras: Número de simulações Monte Carlo (default: 50).

    Returns:
        Dicionário com:
        - ``"previsoes"``: DataFrame com colunas ``data``, ``preco_medio``,
          ``preco_std``, ``ic_5pct``, ``ic_95pct``.
        - ``"preco_atual"``: Último preço real observado.
        Ou ``None`` se o modelo não estiver disponível.
    """
    ctx = _preparar_inferencia(produto_id, caminho_csv)
    if ctx is None:
        return None

    model         = ctx["model"]
    scaler        = ctx["scaler"]
    janela        = ctx["janela"]
    features_norm = ctx["features_norm"]
    cadeia_id_produto = ctx["cadeia_id"]
    hist_base     = ctx["hist_precos"]
    ultima_data   = ctx["ultima_data"]
    preco_atual   = ctx["preco_atual"]

    model.train()  # mantém dropout activo para Monte Carlo

    todas_simulacoes: list[list[float]] = []

    with torch.no_grad():
        for _ in range(n_amostras):
            janela_atual = features_norm[-janela:].copy()
            hist_sim = deque(hist_base, maxlen=7)
            simulacao = []

            for i in range(horizonte):
                pred_norm = model(torch.tensor(janela_atual).unsqueeze(0).to(DEVICE)).item()

                dummy = np.zeros((1, N_FEATURES), dtype=np.float32)
                dummy[0, PRECO_IDX] = pred_norm
                pred_raw = max(0.01, round(float(scaler.inverse_transform(dummy)[0, PRECO_IDX]), 2))

                simulacao.append(pred_raw)
                hist_sim.append(pred_raw)

                raw_next = _raw_features_para_dia(
                    data=ultima_data + timedelta(days=i + 2),
                    preco_raw=pred_raw,
                    historico_precos=hist_sim,
                    dias_desde_obs=i + 2,
                    cadeia_id=cadeia_id_produto,
                )
                nova_entrada = scaler.transform(raw_next.reshape(1, -1))[0]
                janela_atual = np.vstack([janela_atual[1:], nova_entrada])

            todas_simulacoes.append(simulacao)

    arr = np.array(todas_simulacoes)  # shape: (n_amostras, horizonte)
    datas = [ultima_data + timedelta(days=i + 1) for i in range(horizonte)]

    df_resultado = pd.DataFrame({
        "data": datas,
        "preco_medio": np.mean(arr, axis=0).round(2),
        "preco_std": np.std(arr, axis=0).round(4),
        "ic_5pct": np.percentile(arr, 5, axis=0).round(2),
        "ic_95pct": np.percentile(arr, 95, axis=0).round(2),
    })

    print(f"\n[MC PREVISAO] Produto {produto_id} | Preco atual: {preco_atual:.2f}EUR | {n_amostras} simulacoes")
    print(df_resultado.to_string(index=False))

    return {"previsoes": df_resultado, "preco_atual": preco_atual}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI do módulo de previsão de preços."""
    parser = argparse.ArgumentParser(
        description="Previsao de precos com LSTM global."
    )
    parser.add_argument("--treinar", action="store_true", help="Treinar modelo global.")
    parser.add_argument("--prever", action="store_true", help="Gerar previsoes para um produto.")
    parser.add_argument("--avaliar", action="store_true", help="Avaliar modelo no conjunto de validacao.")
    parser.add_argument("--mc", action="store_true", help="Previsao com intervalos de confianca (Monte Carlo Dropout).")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent.parent / "data/generated/forecasting_dataset.csv"),
        help="Caminho para o CSV de forecasting.",
    )
    parser.add_argument("--produto-id", type=int, help="id_produto_loja para previsao.")
    parser.add_argument("--janela", type=int, default=7, help="Janela de contexto em dias (default: 7).")
    parser.add_argument("--horizonte", type=int, default=7, help="Dias a prever (default: 7).")
    parser.add_argument("--epocas", type=int, default=100, help="Epocas maximas de treino (default: 100).")
    parser.add_argument("--min-obs", type=int, default=5, help="Minimo de observacoes por produto (default: 5).")
    parser.add_argument("--paciencia", type=int, default=15, help="Early stopping: epocas sem melhoria (default: 15).")
    parser.add_argument("--amostras", type=int, default=50, help="Simulacoes Monte Carlo (default: 50).")
    args = parser.parse_args()

    if args.treinar:
        treinar_modelo_global(
            caminho_csv=args.dataset,
            janela=args.janela,
            epochs=args.epocas,
            min_observacoes=args.min_obs,
            paciencia=args.paciencia,
        )

    if args.avaliar:
        avaliar_modelo(caminho_csv=args.dataset)

    if args.prever:
        if not args.produto_id:
            parser.error("--prever requer --produto-id")
        if args.mc:
            prever_preco_com_incerteza(
                produto_id=args.produto_id,
                caminho_csv=args.dataset,
                horizonte=args.horizonte,
                n_amostras=args.amostras,
            )
        else:
            prever_preco(
                produto_id=args.produto_id,
                caminho_csv=args.dataset,
                horizonte=args.horizonte,
            )


if __name__ == "__main__":
    main()
