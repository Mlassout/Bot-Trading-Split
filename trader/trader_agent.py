"""
trader_agent.py
---------------
Wrapper d'inference pour le Trader RL entraine.

Charge le modele PPO sauvegarde et expose une interface simple :
    agent.predict(observation) → action (int)

Gere egalement la normalisation des observations (VecNormalize).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import base64
import pickle
import zipfile
import numpy as np
import logging
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _load_ppo_safe(model_path: Path) -> PPO:
    """
    Charge un modele PPO depuis un .zip en contournant PPO.load() qui segfault
    sur Python 3.13 (optimizer pickle cross-platform).
    Charge directement les poids policy.pth avec torch.load().
    """
    class _Env(gym.Env):
        def __init__(self, obs_dim, n_actions):
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(n_actions)
        def reset(self, **kw): return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        def step(self, a): return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

    with zipfile.ZipFile(str(model_path)) as z:
        data = json.loads(z.read("data"))
        policy_weights = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)

    obs_space  = pickle.loads(base64.b64decode(data["observation_space"][":serialized:"]))
    act_space  = pickle.loads(base64.b64decode(data["action_space"][":serialized:"]))
    policy_kwargs = pickle.loads(base64.b64decode(data["policy_kwargs"][":serialized:"]))

    obs_dim   = obs_space.shape[0]
    n_actions = act_space.n

    env = DummyVecEnv([lambda: _Env(obs_dim, n_actions)])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
    model.policy.load_state_dict(policy_weights)
    model.policy.set_training_mode(False)
    logger.info(f"[TraderAgent] Poids charges via torch.load (Python 3.13 compat)")
    return model


class TraderAgent:
    """
    Interface d'inference pour le Trader RL.

    Usage :
        agent = TraderAgent.from_file("models/trader_ppo.zip")
        action = agent.predict(observation)
        print(f"Action : {ACTION_NAMES[action]}")
    """

    def __init__(self, model: PPO, vec_normalize: Optional[VecNormalize] = None):
        self.model = model
        self.vec_normalize = vec_normalize
        self._n_predictions = 0

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        model_path: str | Path,
        vecnorm_path: Optional[str | Path] = None,
        env_factory=None,
    ) -> "TraderAgent":
        """
        Charge un modele sauvegarde depuis le disque.

        Args:
            model_path:   Chemin vers le fichier .zip du modele PPO.
            vecnorm_path: Chemin vers le fichier .pkl de VecNormalize (optionnel).
            env_factory:  Callable retournant un environnement (pour reconstruire VecNormalize).

        Returns:
            Instance de TraderAgent prete a l'inference.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modele introuvable : {model_path}")

        logger.info(f"[TraderAgent] Chargement du modele : {model_path}")
        model = _load_ppo_safe(model_path)

        vec_normalize = None
        if vecnorm_path and Path(vecnorm_path).exists():
            if env_factory is None:
                raise ValueError("env_factory requis pour charger VecNormalize.")
            dummy_env = DummyVecEnv([env_factory])
            vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
            logger.info(f"[TraderAgent] VecNormalize charge : {vecnorm_path}")

        return cls(model=model, vec_normalize=vec_normalize)

    @classmethod
    def from_best(cls, env_factory=None) -> "TraderAgent":
        """Charge le meilleur modele sauvegarde par EvalCallback."""
        best_path = MODELS_DIR / "best" / "best_model.zip"
        vecnorm_path = MODELS_DIR / "trader_ppo_vecnorm.pkl"
        return cls.from_file(
            model_path=best_path,
            vecnorm_path=vecnorm_path if vecnorm_path.exists() else None,
            env_factory=env_factory,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predit l'action optimale pour une observation donnee.

        Args:
            observation:  Vecteur d'observation (shape correspondant a l'espace d'obs).
            deterministic: True = action greedy, False = action stochastique.

        Returns:
            Action entiere : 0=HOLD, 1=BUY, 2=SELL
        """
        obs = np.array(observation, dtype=np.float32)

        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

        action, _ = self.model.predict(obs, deterministic=deterministic)
        self._n_predictions += 1

        action_int = int(action)
        logger.debug(f"[TraderAgent] Prediction #{self._n_predictions}: {ACTION_NAMES[action_int]}")
        return action_int

    def get_action_probabilities(self, observation: np.ndarray) -> dict:
        """
        Retourne les probabilites de chaque action (utile pour le debug).

        Returns:
            {'HOLD': p0, 'BUY': p1, 'SELL': p2}
        """
        import torch
        obs = np.array(observation, dtype=np.float32)
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy().flatten()

        return {ACTION_NAMES[i]: float(probs[i]) for i in range(3)}

    @property
    def n_predictions(self) -> int:
        return self._n_predictions
