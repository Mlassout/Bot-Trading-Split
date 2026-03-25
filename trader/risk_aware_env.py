"""
risk_aware_env.py
-----------------
Wrapper Gymnasium autour de TradingEnv qui integre le Risk Manager.

Toute action proposee par le Trader passe d'abord par le Risk Manager.
Ce dernier peut :
  - ALLOW    → action executee telle quelle
  - BLOCK_SELL → action forcee a SELL (2), penalite de reward
  - BLOCK_HALT → action forcee a HOLD (0), penalite de reward

Supporte egalement les parametres du Structureur :
  - position_size_factor : reduit la taille de position (ex: 0.5 = 50% du capital)
  - allowed_actions      : sous-ensemble d'actions autorisees
  - max_holding_steps    : nombre max de steps pour garder une position
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from environment.trading_env import TradingEnv
from risk_manager.risk_manager import RiskManager, RiskDecision, RiskConfig
import pandas as pd


class RiskAwareTradingEnv(gym.Wrapper):
    """
    Wrapper sur TradingEnv integrant le Risk Manager et les parametres du Structureur.

    Args:
        env:                  Instance de TradingEnv sous-jacente.
        risk_config:          Configuration des seuils du Risk Manager.
        position_size_factor: Fraction du capital a utiliser par trade (Structureur).
        allowed_actions:      Liste des actions autorisees [0,1,2] (Structureur).
        max_holding_steps:    Steps max avant cloture forcee (Structureur). None = infini.
    """

    HOLD = 0
    BUY  = 1
    SELL = 2

    RISK_OVERRIDE_PENALTY = -0.05  # Penalite quand le Risk Manager override une action

    def __init__(
        self,
        env: TradingEnv,
        risk_config: Optional[RiskConfig] = None,
        position_size_factor: float = 1.0,
        allowed_actions: Optional[list] = None,
        max_holding_steps: Optional[int] = None,
    ):
        super().__init__(env)
        self.risk_manager = RiskManager(
            initial_capital=env.initial_capital,
            config=risk_config or RiskConfig(),
        )
        self.position_size_factor = max(0.0, min(1.0, position_size_factor))
        self.allowed_actions = allowed_actions or [0, 1, 2]
        self.max_holding_steps = max_holding_steps

        self._holding_steps: int = 0
        self._last_portfolio_value: float = env.initial_capital
        self._risk_overrides: int = 0
        self._last_action: int = 0

        # Ajouter 3 dimensions a l'observation : [risk_halted, holding_steps_norm, last_action]
        base_shape = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_shape + 3,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Interface Gymnasium
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.risk_manager.reset(self.env.initial_capital)
        self._holding_steps = 0
        self._last_portfolio_value = self.env.initial_capital
        self._risk_overrides = 0
        self._last_action = 0
        return self._augment_obs(obs, info), info

    def step(self, action: int):
        info_before = self.env._get_info()
        portfolio_value = info_before["portfolio_value"]

        # --- Filtrage par allowed_actions (Structureur) ---
        if action not in self.allowed_actions:
            action = self.HOLD

        # --- Cloture forcee si max_holding_steps atteint ---
        if (
            self.max_holding_steps is not None
            and info_before["position"] == 1
            and self._holding_steps >= self.max_holding_steps
        ):
            action = self.SELL

        # --- Evaluation par le Risk Manager ---
        risk_state = self.risk_manager.evaluate(
            action=action,
            portfolio_value=portfolio_value,
            capital=info_before["capital"],
            position=info_before["position"],
            entry_price=self.env._entry_price,
            current_price=info_before["price"],
            shares=info_before["shares"],
        )

        risk_penalty = 0.0
        if risk_state.decision == RiskDecision.BLOCK_SELL:
            action = self.SELL
            risk_penalty = self.RISK_OVERRIDE_PENALTY
            self._risk_overrides += 1
        elif risk_state.decision == RiskDecision.BLOCK_HALT:
            action = self.HOLD
            risk_penalty = self.RISK_OVERRIDE_PENALTY
            self._risk_overrides += 1

        # --- Application du position_size_factor sur BUY ---
        # On patche temporairement le capital de l'env pour limiter la taille
        original_capital = None
        if action == self.BUY and self.position_size_factor < 1.0 and info_before["position"] == 0:
            original_capital = self.env._capital
            self.env._capital = original_capital * self.position_size_factor

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Restaurer le capital non-utilise
        if original_capital is not None and action == self.BUY:
            unused = original_capital * (1.0 - self.position_size_factor)
            self.env._capital += unused

        # --- Tracking post-step ---
        if info["position"] == 1:
            self._holding_steps += 1
        else:
            if info_before["position"] == 1:
                # Trade vient d'etre cloture : notifier le Risk Manager
                pnl = info["realized_pnl"] - info_before["realized_pnl"]
                self.risk_manager.record_trade_result(pnl)
            self._holding_steps = 0

        reward += risk_penalty
        info["action_executed"] = action
        self._last_action = action
        info["risk_decision"] = risk_state.decision.value
        info["risk_reason"] = risk_state.reason
        info["risk_overrides"] = self._risk_overrides
        info["drawdown_pct"] = risk_state.drawdown_pct
        info["risk_halted"] = self.risk_manager.is_halted
        self._last_portfolio_value = info["portfolio_value"]

        return self._augment_obs(obs, info), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Parametres Structureur (modifiables a chaud)
    # ------------------------------------------------------------------

    def update_structurer_params(
        self,
        position_size_factor: Optional[float] = None,
        allowed_actions: Optional[list] = None,
        max_holding_steps: Optional[int] = None,
    ) -> None:
        """Met a jour les parametres du Structureur sans reinitialiser l'episode."""
        if position_size_factor is not None:
            self.position_size_factor = max(0.0, min(1.0, position_size_factor))
        if allowed_actions is not None:
            self.allowed_actions = allowed_actions
        if max_holding_steps is not None:
            self.max_holding_steps = max_holding_steps

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _augment_obs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Ajoute les dimensions de risque a l'observation de base."""
        risk_halted = float(self.risk_manager.is_halted)
        holding_norm = (self._holding_steps / (self.max_holding_steps or 500)) if self.max_holding_steps else 0.0
        last_action_norm = self._last_action / 2.0  # normalise entre 0 et 1
        return np.append(obs, [risk_halted, holding_norm, last_action_norm]).astype(np.float32)
