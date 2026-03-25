"""
trainer.py
----------
Entraine le Trader avec l'algorithme PPO de Stable Baselines3.

Le Trader apprend a maximiser le PnL a l'interieur des contraintes du Risk Manager.
L'environnement d'entrainement est RiskAwareTradingEnv.

Usage :
    python trader/trainer.py
    # ou depuis main.py :
    python main.py train
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from environment.data_loader import generate_synthetic_ohlcv, load_ohlcv, load_multi_ohlcv
from environment.trading_env import TradingEnv
from trader.risk_aware_env import RiskAwareTradingEnv
from risk_manager.risk_manager import RiskConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Callback de logging custom
# ------------------------------------------------------------------

class TradingMetricsCallback(BaseCallback):
    """Affiche des metriques de trading specifiques pendant l'entrainement."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_pnls = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "realized_pnl" in info and "episode" in info:
                self._episode_pnls.append(info["realized_pnl"])
                if len(self._episode_pnls) % 100 == 0 and self.verbose > 0:
                    mean_pnl = np.mean(self._episode_pnls[-100:])
                    logger.info(
                        f"[Trainer] Episodes: {len(self._episode_pnls)} | "
                        f"PnL moyen (100 ep): {mean_pnl:.2f}"
                    )
        return True


# ------------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------------

def make_env(
    df,
    risk_config: Optional[RiskConfig] = None,
    seed: int = 0,
    window_size: int = 20,
    initial_capital: float = 10_000.0,
):
    """Factory pour creer un environnement entraine compatible VecEnv."""
    def _init():
        base_env = TradingEnv(
            df=df,
            initial_capital=initial_capital,
            window_size=window_size,
            transaction_cost=0.001,
            holding_penalty=0.001,
        )
        env = RiskAwareTradingEnv(
            env=base_env,
            risk_config=risk_config or RiskConfig(),
            max_holding_steps=500,   # laisse les tendances longues se developper
        )
        env = Monitor(env)
        return env
    return _init


# ------------------------------------------------------------------
# Entrainement principal
# ------------------------------------------------------------------

def train(
    total_timesteps: int = 500_000,
    data_path: Optional[str] = None,
    data_paths: Optional[list] = None,
    model_name: str = "trader_ppo",
    n_envs: int = 4,
    window_size: int = 20,
    initial_capital: float = 10_000.0,
    risk_config: Optional[RiskConfig] = None,
    seed: int = 42,
) -> PPO:
    """
    Lance l'entrainement du Trader PPO.

    Args:
        total_timesteps: Nombre total de steps d'entrainement.
        data_path:       Chemin vers un CSV OHLCV. Si None, donnees synthetiques.
        model_name:      Nom du fichier de sauvegarde du modele.
        n_envs:          Nombre d'environnements paralleles pour SB3.
        window_size:     Taille de la fenetre d'observation.
        initial_capital: Capital de depart.
        risk_config:     Configuration du Risk Manager.
        seed:            Graine aleatoire.

    Returns:
        Le modele PPO entraine.
    """
    # Silencer le Risk Manager pendant l'entrainement (trop verbeux sur 500k steps)
    logging.getLogger("risk_manager.risk_manager").setLevel(logging.ERROR)

    logger.info("=" * 60)
    logger.info("ENTRAINEMENT DU TRADER (PPO)")
    logger.info("=" * 60)

    # Chargement des donnees
    if data_paths and len(data_paths) > 1:
        logger.info(f"Mode MULTI-ACTIFS : {len(data_paths)} fichiers")
        for p in data_paths:
            logger.info(f"  -> {p}")
        df = load_multi_ohlcv(data_paths, normalize_prices=True)
        logger.info(f"Dataset fusionne : {len(df)} bougies totales")
    elif data_path:
        logger.info(f"Chargement des donnees depuis : {data_path}")
        df = load_ohlcv(data_path)
    else:
        logger.info("Donnees synthetiques (GBM, 5000 steps)")
        df = generate_synthetic_ohlcv(n_steps=5_000, seed=seed)

    logger.info(f"Dataset : {len(df)} bougies | Capital : {initial_capital} | Fenetre : {window_size}")

    # Split train/eval (80/20)
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_eval = df.iloc[split:].reset_index(drop=True)

    # Environnements d'entrainement (vectorises)
    train_env = DummyVecEnv([
        make_env(df_train, risk_config, seed=seed + i, window_size=window_size, initial_capital=initial_capital)
        for i in range(n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Environnement d'evaluation
    eval_env = DummyVecEnv([
        make_env(df_eval, risk_config, seed=seed + 100, window_size=window_size, initial_capital=initial_capital)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(MODELS_DIR / "logs"),
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=model_name,
    )
    metrics_callback = TradingMetricsCallback(verbose=1)

    # Learning rate schedule : decroit lineairement de 3e-4 a 1e-4
    def lr_schedule(progress: float) -> float:
        """progress : 1.0 au debut, 0.0 a la fin."""
        return 1e-4 + progress * (3e-4 - 1e-4)

    # Modele PPO
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr_schedule,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.985,        # horizon plus court, plus adapte aux marches financiers
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            # Reseaux actor (pi) et critic (vf) separes : evite que l'un perturbe l'autre
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "activation_fn": __import__("torch").nn.Tanh,  # Tanh prefere a ReLU en RL
        },
        verbose=1,
        seed=seed,
    )

    logger.info(f"Debut de l'entrainement : {total_timesteps:,} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, metrics_callback],
    )

    # Sauvegarde
    model_path = MODELS_DIR / f"{model_name}.zip"
    vecnorm_path = MODELS_DIR / f"{model_name}_vecnorm.pkl"
    model.save(str(model_path))
    train_env.save(str(vecnorm_path))
    logger.info(f"Modele sauvegarde : {model_path}")
    logger.info(f"VecNormalize sauvegarde : {vecnorm_path}")

    return model


if __name__ == "__main__":
    train(total_timesteps=200_000)
