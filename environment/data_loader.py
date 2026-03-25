"""
data_loader.py
--------------
Charge et valide des donnees OHLCV (Open, High, Low, Close, Volume)
depuis un fichier CSV ou genere des donnees synthetiques pour les tests.

Colonnes attendues : timestamp, open, high, low, close, volume
"""

import pandas as pd
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """
    Charge un CSV OHLCV, valide les colonnes et nettoie les donnees.

    Args:
        path: Chemin vers le fichier CSV.

    Returns:
        DataFrame avec index datetime, colonnes : open, high, low, close, volume.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si des colonnes obligatoires sont absentes ou si les donnees sont invalides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Validation des colonnes
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    # Index datetime
    timestamp_col = next((c for c in df.columns if "time" in c or "date" in c), None)
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()

    df = df[list(REQUIRED_COLUMNS)].astype(float)

    # Suppression des NaN et coherence OHLC
    df = df.dropna()
    invalid = (df["low"] > df["high"]) | (df["close"] <= 0) | (df["volume"] < 0)
    n_invalid = invalid.sum()
    if n_invalid > 0:
        print(f"[data_loader] Avertissement : {n_invalid} lignes OHLC invalides supprimees.")
        df = df[~invalid]

    if len(df) < 50:
        raise ValueError(f"Pas assez de donnees apres nettoyage : {len(df)} lignes (minimum 50).")

    return df


def load_multi_ohlcv(paths: list, normalize_prices: bool = True) -> pd.DataFrame:
    """
    Charge et concatene plusieurs CSV OHLCV en un seul dataset d'entrainement.

    Chaque actif est normalise pour commencer a 100 (si normalize_prices=True)
    afin que le modele ne soit pas perturbe par les differences d'echelle de prix
    (ex: BTC a 30 000$ vs EUR/USD a 1.05).

    Args:
        paths:            Liste de chemins vers des CSV OHLCV.
        normalize_prices: Si True, normalise chaque actif a un prix de depart de 100.

    Returns:
        DataFrame OHLCV concatene, index reset.
    """
    frames = []
    for path in paths:
        df = load_ohlcv(path)
        if normalize_prices:
            # Normalisation : tous les prix indexes sur 100 au debut
            scale = df["close"].iloc[0] / 100.0
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] / scale
            # Volume normalise aussi (log-scale)
            df["volume"] = np.log1p(df["volume"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def generate_synthetic_ohlcv(
    n_steps: int = 1000,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genere des donnees OHLCV synthetiques via un mouvement brownien geometrique.
    Utile pour les tests sans donnees reelles.

    Args:
        n_steps:     Nombre de periodes (bougies).
        start_price: Prix de depart.
        volatility:  Ecart-type des rendements journaliers (ex: 0.02 = 2%).
        trend:       Derive journaliere moyenne (ex: 0.0001 = faible hausse).
        seed:        Graine aleatoire pour la reproductibilite.

    Returns:
        DataFrame OHLCV avec index DatetimeIndex (frequence horaire).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_steps, freq="h")

    # Prix de cloture via GBM discret
    returns = rng.normal(loc=trend, scale=volatility, size=n_steps)
    close = start_price * np.exp(np.cumsum(returns))

    # Construction OHLC realiste autour du close
    noise = rng.uniform(0.001, volatility, size=n_steps)
    open_ = close * (1 + rng.normal(0, noise / 2))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, noise))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, noise))
    volume = rng.uniform(1_000, 100_000, size=n_steps)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
