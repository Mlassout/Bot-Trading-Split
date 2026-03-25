"""
trading_env.py
--------------
Environnement de trading compatible Gymnasium (API Gym standard).

Actions discretes :
    0 = HOLD   (ne rien faire)
    1 = BUY    (ouvrir/renforcer une position longue)
    2 = SELL   (fermer la position longue)

Observation (vecteur numerique normalise) :
    [open, high, low, close, volume]  sur une fenetre glissante de `window_size` bougies
    + [position_actuelle, pnl_non_realise_pct, capital_pct, drawdown_pct]

Reward (dense mark-to-market) :
    A chaque step, la reward = variation de la valeur du portefeuille normalisee
    par le capital initial. Cela donne un signal dense a chaque bougie,
    elimine le comportement "hold forever" du to des rewards sparses.

    Composantes :
        + step_return    : variation mark-to-market du portefeuille
        - drawdown_penalty : penalite proportionnelle au drawdown depuis le pic
        - inaction_penalty : penalite si HOLD en dehors d'une position (encourage a trader)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from .data_loader import load_ohlcv, generate_synthetic_ohlcv


class TradingEnv(gym.Env):
    """
    Environnement de trading sur donnees OHLCV historiques.

    Args:
        df:              DataFrame OHLCV (index datetime, colonnes open/high/low/close/volume).
                         Si None, des donnees synthetiques sont generees automatiquement.
        initial_capital: Capital de depart en USD (ou unites monetaires).
        window_size:     Nombre de bougies passees visibles par l'agent.
        transaction_cost:Frais de transaction en pourcentage (ex: 0.001 = 0.1%).
        holding_penalty: Penalite par step pour une position ouverte (encourage les clotures).
        max_steps:       Nombre maximum de steps par episode (None = tout le dataset).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        initial_capital: float = 10_000.0,
        window_size: int = 20,
        transaction_cost: float = 0.001,
        holding_penalty: float = 0.0001,   # garde pour compatibilite (non utilise)
        max_steps: Optional[int] = None,
        drawdown_penalty_factor: float = 2.0,   # multiplicateur penalite drawdown
        inaction_penalty: float = 0.0002,        # penalite par step si HOLD sans position
    ):
        super().__init__()

        self.df = df if df is not None else generate_synthetic_ohlcv()
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.holding_penalty = holding_penalty
        self.drawdown_penalty_factor = drawdown_penalty_factor
        self.inaction_penalty = inaction_penalty
        self.max_steps = max_steps if max_steps else len(self.df) - window_size - 1

        self._prices = self.df["close"].values.astype(np.float32)
        self._ohlcv = self.df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        self._n_features = 5  # open, high, low, close, volume

        # --- Espaces Gymnasium ---
        # Action : 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation : fenetre OHLCV + 4 variables de compte (position, unrealized, capital, volatility)
        obs_size = self.window_size * self._n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Etat interne (initialise par reset())
        self._current_step: int = 0
        self._capital: float = 0.0
        self._position: int = 0        # 0 = flat, 1 = long
        self._entry_price: float = 0.0
        self._shares: float = 0.0
        self._realized_pnl: float = 0.0
        self._total_trades: int = 0
        self._peak_value: float = 0.0       # pour calcul drawdown
        self._prev_portfolio_value: float = 0.0  # pour reward mark-to-market

    # ------------------------------------------------------------------
    # Interface Gymnasium
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._current_step = self.window_size
        self._capital = self.initial_capital
        self._position = 0
        self._entry_price = 0.0
        self._shares = 0.0
        self._realized_pnl = 0.0
        self._total_trades = 0
        self._peak_value = self.initial_capital
        self._prev_portfolio_value = self.initial_capital

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Action invalide : {action}"

        current_price = self._prices[self._current_step]

        # --- Execution de l'action ---
        if action == 1 and self._position == 0:
            # BUY
            cost = current_price * (1 + self.transaction_cost)
            self._shares = self._capital / cost
            self._entry_price = current_price
            self._capital = 0.0
            self._position = 1
            self._total_trades += 1

        elif action == 2 and self._position == 1:
            # SELL
            proceeds = self._shares * current_price * (1 - self.transaction_cost)
            pnl_trade = proceeds - (self._shares * self._entry_price)
            self._realized_pnl += pnl_trade
            self._capital = proceeds
            self._shares = 0.0
            self._position = 0
            self._entry_price = 0.0
            self._total_trades += 1

        self._current_step += 1

        # --- Fin d'episode ---
        terminated = self._is_bankrupt()
        truncated = self._current_step >= self.window_size + self.max_steps

        # Cloture forcee en fin d'episode si position ouverte
        if (terminated or truncated) and self._position == 1:
            final_price = self._prices[min(self._current_step, len(self._prices) - 1)]
            proceeds = self._shares * final_price * (1 - self.transaction_cost)
            pnl_trade = proceeds - (self._shares * self._entry_price)
            self._realized_pnl += pnl_trade
            self._capital = proceeds
            self._position = 0

        # ------------------------------------------------------------------
        # REWARD DENSE MARK-TO-MARKET
        # ------------------------------------------------------------------
        curr_price = self._prices[min(self._current_step, len(self._prices) - 1)]
        curr_value = self._get_portfolio_value(curr_price)

        # 1. Step return : variation de valeur normalisee par le capital initial
        step_return = (curr_value - self._prev_portfolio_value) / self.initial_capital

        # 2. Mise a jour du pic (pour calcul drawdown)
        self._peak_value = max(self._peak_value, curr_value)

        # 3. Penalite drawdown : proportionnelle a la perte depuis le pic
        #    Ex : drawdown -15% -> penalite = 0.15 * factor / initial_capital (par step)
        drawdown = (curr_value - self._peak_value) / self._peak_value  # negatif ou 0
        drawdown_penalty = abs(min(drawdown, 0)) * self.drawdown_penalty_factor / len(self._prices)

        # 4. Penalites d'inaction :
        #    - HOLD sans position : pousse a entrer en trade
        #    - HOLD avec position ouverte : pousse a sortir (evite le buy & hold indefini)
        inaction_penalty = self.inaction_penalty if (action == 0 and self._position == 0) else 0.0
        holding_penalty = self.holding_penalty if (action == 0 and self._position == 1) else 0.0

        reward = step_return - drawdown_penalty - inaction_penalty - holding_penalty

        self._prev_portfolio_value = curr_value

        obs = self._get_observation()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        price = self._prices[self._current_step - 1]
        unrealized = self._get_unrealized_pnl(price)
        portfolio_value = self._get_portfolio_value(price)
        msg = (
            f"Step {self._current_step:5d} | "
            f"Prix: {price:8.2f} | "
            f"Position: {'LONG' if self._position else 'FLAT':4s} | "
            f"PnL realise: {self._realized_pnl:+8.2f} | "
            f"PnL non-realise: {unrealized:+8.2f} | "
            f"Portefeuille: {portfolio_value:8.2f}"
        )
        if mode == "human":
            print(msg)
        return msg

    # ------------------------------------------------------------------
    # Methodes internes
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Construit le vecteur d'observation normalise."""
        window = self._ohlcv[self._current_step - self.window_size : self._current_step]

        ref_price = window[-1, 3] + 1e-8   # close de la derniere bougie
        ref_volume = window[:, 4].mean() + 1e-8

        normalized = window.copy()
        # Log-normalisation des prix (meilleure stationnarite que la division simple)
        normalized[:, :4] = np.log(normalized[:, :4] / ref_price + 1e-8)
        normalized[:, 4] = normalized[:, 4] / ref_volume

        ohlcv_flat = normalized.flatten()

        # Volatilite rolling (proxy turbulence) : std des log-rendements sur la fenetre
        log_returns = np.diff(np.log(window[:, 3] + 1e-8))
        rolling_vol = float(np.clip(log_returns.std(), 0.0, 0.5)) if len(log_returns) > 1 else 0.0

        # Variables de compte
        current_price = self._prices[self._current_step - 1]
        portfolio_value = self._get_portfolio_value(current_price)
        unrealized_pct = self._get_unrealized_pnl(current_price) / self.initial_capital
        capital_pct = self._capital / self.initial_capital

        account_state = np.array(
            [float(self._position), unrealized_pct, capital_pct, rolling_vol],
            dtype=np.float32,
        )

        obs = np.concatenate([ohlcv_flat, account_state])
        # Securite : eliminer tout NaN/Inf residuel avant d'envoyer au reseau
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

    def _get_unrealized_pnl(self, current_price: float) -> float:
        if self._position == 0:
            return 0.0
        return self._shares * (current_price - self._entry_price)

    def _get_portfolio_value(self, current_price: float) -> float:
        return self._capital + self._shares * current_price

    def _is_bankrupt(self) -> bool:
        price = self._prices[self._current_step - 1]
        return self._get_portfolio_value(price) < self.initial_capital * 0.05

    def _get_info(self) -> dict:
        price = self._prices[self._current_step - 1]
        return {
            "step": self._current_step,
            "price": float(price),
            "position": self._position,
            "capital": float(self._capital),
            "shares": float(self._shares),
            "realized_pnl": float(self._realized_pnl),
            "unrealized_pnl": float(self._get_unrealized_pnl(price)),
            "portfolio_value": float(self._get_portfolio_value(price)),
            "total_trades": self._total_trades,
        }
