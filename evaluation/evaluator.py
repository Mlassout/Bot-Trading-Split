"""
evaluator.py
------------
Evalue un modele PPO sur chaque actif du jeu de test individuellement.
Calcule les metriques cles : rendement, Sharpe, drawdown, win rate, etc.

Usage :
    python main.py evaluate --checkpoints models/checkpoints/ --test-data data/test/
    python main.py evaluate --model models/best/best_model.zip --test-data data/test/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.data_loader import load_ohlcv
from environment.trading_env import TradingEnv
from trader.risk_aware_env import RiskAwareTradingEnv
from risk_manager.risk_manager import RiskConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metriques
# ------------------------------------------------------------------

def compute_metrics(portfolio_values: list, initial_capital: float, n_trades: int, n_wins: int) -> dict:
    """Calcule les metriques de performance a partir de la serie de valeurs du portefeuille."""
    pv = np.array(portfolio_values)
    returns = np.diff(pv) / pv[:-1]

    total_return = (pv[-1] - initial_capital) / initial_capital * 100

    # Sharpe annualise (trading journalier = 252 jours)
    if returns.std() > 1e-10:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(pv)
    drawdowns = (pv - peak) / peak
    max_drawdown = drawdowns.min() * 100

    # Calmar ratio
    calmar = (total_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0.0

    # Win rate
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0.0

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2),
        "calmar":           round(calmar, 3),
        "win_rate_pct":     round(win_rate, 1),
        "n_trades":         n_trades,
        "final_capital":    round(pv[-1], 2),
        "n_steps":          len(pv),
    }


# ------------------------------------------------------------------
# Evaluation d'un modele sur un actif
# ------------------------------------------------------------------

def evaluate_on_asset(
    model: PPO,
    df: pd.DataFrame,
    risk_config: Optional[RiskConfig] = None,
    initial_capital: float = 10_000.0,
    window_size: int = 20,
    vecnorm_path: Optional[str] = None,
) -> dict:
    """
    Fait tourner le modele sur un dataset complet et retourne les metriques.
    """
    # Silencer le Risk Manager
    logging.getLogger("risk_manager.risk_manager").setLevel(logging.ERROR)

    base_env = TradingEnv(
        df=df,
        initial_capital=initial_capital,
        window_size=window_size,
        transaction_cost=0.001,
        holding_penalty=0.0001,
    )
    env = RiskAwareTradingEnv(
        env=base_env,
        risk_config=risk_config or RiskConfig(),
    )

    # VecNormalize pour que les observations soient dans le meme espace que l'entrainement
    vec_env = DummyVecEnv([lambda: env])
    if vecnorm_path and Path(vecnorm_path).exists():
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    obs = vec_env.reset()
    portfolio_values = [initial_capital]
    n_trades = 0
    n_wins = 0
    last_buy_price = None
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        done = dones[0]
        info = infos[0]

        portfolio_values.append(info.get("portfolio_value", portfolio_values[-1]))

        # Comptage trades
        executed = info.get("action_executed", -1)
        if executed == 1:  # BUY
            last_buy_price = info.get("price", None)
        elif executed == 2 and last_buy_price is not None:  # SELL
            n_trades += 1
            if info.get("price", 0) > last_buy_price:
                n_wins += 1
            last_buy_price = None

    vec_env.close()
    return compute_metrics(portfolio_values, initial_capital, n_trades, n_wins)


# ------------------------------------------------------------------
# Evaluation sur tous les actifs d'un dossier
# ------------------------------------------------------------------

def evaluate_all_assets(
    model_path: str,
    test_data_dir: str,
    initial_capital: float = 10_000.0,
    window_size: int = 20,
    checkpoint_steps: Optional[int] = None,
    vecnorm_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Charge un modele et l'evalue sur tous les CSV du dossier test.

    Returns:
        DataFrame avec une ligne par actif et toutes les metriques.
    """
    model = PPO.load(model_path)
    csv_files = sorted(glob.glob(os.path.join(test_data_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV dans : {test_data_dir}")

    risk_cfg = RiskConfig(max_drawdown_pct=0.10, max_daily_loss_pct=0.02, stop_loss_pct=0.02)
    results = []

    for csv_path in csv_files:
        asset_name = Path(csv_path).stem
        try:
            df = load_ohlcv(csv_path)
            metrics = evaluate_on_asset(
                model=model,
                df=df,
                risk_config=risk_cfg,
                initial_capital=initial_capital,
                window_size=window_size,
                vecnorm_path=vecnorm_path,
            )
            metrics["asset"] = asset_name
            metrics["model"] = Path(model_path).stem
            if checkpoint_steps is not None:
                metrics["checkpoint_steps"] = checkpoint_steps
            results.append(metrics)
            status = "+" if metrics["total_return_pct"] > 0 else "-"
            logger.info(
                f"  [{status}] {asset_name:<15} "
                f"Return={metrics['total_return_pct']:>+7.2f}%  "
                f"Sharpe={metrics['sharpe']:>+5.3f}  "
                f"DD={metrics['max_drawdown_pct']:>6.2f}%  "
                f"WinRate={metrics['win_rate_pct']:>5.1f}%  "
                f"Trades={metrics['n_trades']}"
            )
        except Exception as e:
            logger.warning(f"  ERR {asset_name}: {e}")

    df_results = pd.DataFrame(results)
    cols = ["asset", "model", "checkpoint_steps", "total_return_pct", "sharpe",
            "max_drawdown_pct", "calmar", "win_rate_pct", "n_trades", "final_capital", "n_steps"]
    cols = [c for c in cols if c in df_results.columns]
    return df_results[cols]


# ------------------------------------------------------------------
# Evaluation de tous les checkpoints (comparaison par etape)
# ------------------------------------------------------------------

def evaluate_all_checkpoints(
    checkpoints_dir: str,
    test_data_dir: str,
    output_csv: str = "evaluation/results_by_checkpoint.csv",
    initial_capital: float = 10_000.0,
    vecnorm_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evalue chaque checkpoint sauvegarde et compare les performances.
    Permet de voir la progression du modele au fil de l'entrainement.
    """
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoints_dir, "*.zip")))
    if not checkpoint_files:
        raise FileNotFoundError(f"Aucun checkpoint dans : {checkpoints_dir}")

    all_results = []
    for ckpt_path in checkpoint_files:
        # Extraction du nombre de steps depuis le nom du fichier
        # ex: trader_ppo_100000_steps.zip -> 100000
        stem = Path(ckpt_path).stem
        try:
            steps = int([p for p in stem.split("_") if p.isdigit()][0])
        except (IndexError, ValueError):
            steps = None

        logger.info(f"\n{'='*60}")
        logger.info(f"Checkpoint : {stem} ({steps:,} steps)" if steps else f"Checkpoint : {stem}")
        logger.info(f"{'='*60}")

        df_result = evaluate_all_assets(
            model_path=ckpt_path,
            test_data_dir=test_data_dir,
            initial_capital=initial_capital,
            checkpoint_steps=steps,
            vecnorm_path=vecnorm_path,
        )
        all_results.append(df_result)

    combined = pd.concat(all_results, ignore_index=True)
    Path(output_csv).parent.mkdir(exist_ok=True)
    combined.to_csv(output_csv, index=False)
    logger.info(f"\nResultats sauvegardes : {output_csv}")
    return combined
