"""
orchestrator.py
---------------
L'Orchestrateur — le chef d'orchestre de l'architecture multi-agents.

Coordonne tous les modules dans une boucle de paper trading :

    Chaque N steps :
        Analyste  → score sentiment macro (titres financiers)
        Structureur → parametres de trading
        → update RiskAwareTradingEnv

    A chaque step :
        Trader (RL) → action proposee
        Risk Manager → validation / override
        TradingEnv  → execution + PnL

Modes de paper trading :
  - BACKTEST    : rejoue des donnees historiques a vitesse maximale
  - SIMULATION  : mode "temps reel simule" avec pauses entre chaque step
  - LIVE        : branchement sur un courtier reel en mode paper (futur)

Toutes les donnees de session sont loggees dans un fichier CSV.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable
from datetime import datetime

from environment.data_loader import generate_synthetic_ohlcv, load_ohlcv
from environment.trading_env import TradingEnv
from trader.risk_aware_env import RiskAwareTradingEnv
from trader.trader_agent import TraderAgent
from analyst.analyst import Analyst, AnalystMode, SentimentResult
from structurer.structurer import Structurer
from risk_manager.risk_manager import RiskConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Fournisseur de titres (a remplacer par un vrai flux en production)
# ------------------------------------------------------------------

_SAMPLE_HEADLINES = [
    # Haussiers
    [
        "Fed signals pause in rate hikes, markets rally strongly",
        "Tech sector posts record earnings, beats all estimates",
        "Strong jobs report boosts investor confidence",
    ],
    # Neutres
    [
        "Markets await Fed decision, trading volumes low",
        "Mixed economic data leaves analysts divided",
        "Geopolitical tensions offset positive earnings",
    ],
    # Baissiers
    [
        "Inflation higher than expected, rate hike fears return",
        "Major bank warns of recession risk in 2025",
        "Oil price surge threatens global growth outlook",
        "Tech layoffs accelerate, demand outlook darkens",
    ],
]


def _default_headlines_provider(step: int) -> list[str]:
    """Fournit des titres rotatifs pour la simulation (pas de vraie API news)."""
    return _SAMPLE_HEADLINES[step % len(_SAMPLE_HEADLINES)]


# ------------------------------------------------------------------
# Configuration de session
# ------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Configuration complete d'une session de paper trading."""
    initial_capital: float = 10_000.0
    window_size: int = 20
    analyst_update_freq: int = 50        # Mettre a jour le sentiment tous les N steps
    max_steps: Optional[int] = None      # None = tout le dataset
    step_delay_sec: float = 0.0          # 0 = backtest, >0 = simulation temps reel
    analyst_mode: str = "mock"           # "mock" ou "api"
    model_path: Optional[str] = None     # Chemin vers modele PPO entraine
    data_path: Optional[str] = None      # Chemin vers CSV OHLCV
    risk_max_drawdown: float = 0.10
    risk_daily_loss: float = 0.02
    risk_stop_loss: float = 0.02
    log_to_csv: bool = True
    verbose: bool = True


@dataclass
class StepLog:
    """Donnees loggees a chaque step."""
    step: int
    timestamp: str
    price: float
    action_trader: int
    action_executed: int
    risk_decision: str
    position: int
    capital: float
    shares: float
    portfolio_value: float
    realized_pnl: float
    unrealized_pnl: float
    drawdown_pct: float
    sentiment_score: float
    sentiment_bias: str
    regime: str
    position_size_factor: float


# ------------------------------------------------------------------
# Orchestrateur
# ------------------------------------------------------------------

class Orchestrator:
    """
    Orchestre la session de paper trading.

    Args:
        config:              Configuration de la session.
        headlines_provider:  Callable(step) → list[str]. Fournit les titres a l'Analyste.
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        headlines_provider: Optional[Callable] = None,
    ):
        self.config = config or SessionConfig()
        self.headlines_provider = headlines_provider or _default_headlines_provider
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._step_logs: list[StepLog] = []

        # Initialisation des composants
        self._init_components()

    def _init_components(self) -> None:
        cfg = self.config

        # Donnees
        if cfg.data_path:
            df = load_ohlcv(cfg.data_path)
        else:
            df = generate_synthetic_ohlcv(n_steps=2_000, seed=42)

        # Environnement
        risk_config = RiskConfig(
            max_drawdown_pct=cfg.risk_max_drawdown,
            max_daily_loss_pct=cfg.risk_daily_loss,
            stop_loss_pct=cfg.risk_stop_loss,
        )
        base_env = TradingEnv(
            df=df,
            initial_capital=cfg.initial_capital,
            window_size=cfg.window_size,
            max_steps=cfg.max_steps,
        )
        self.env = RiskAwareTradingEnv(env=base_env, risk_config=risk_config)

        # Trader
        if cfg.model_path and Path(cfg.model_path).exists():
            self.trader = TraderAgent.from_file(cfg.model_path)
            logger.info(f"[Orchestrateur] Modele charge : {cfg.model_path}")
        else:
            logger.warning("[Orchestrateur] Aucun modele trouve. Trader en mode aleatoire.")
            self.trader = None  # politique aleatoire

        # Analyste + Structureur
        mode = AnalystMode.API if cfg.analyst_mode == "api" else AnalystMode.MOCK
        self.analyst = Analyst(mode=mode)
        self.structurer = Structurer()

        # Etat de session
        self._current_sentiment = SentimentResult(
            bias="neutral", score=0.0, confidence=0.5,
            reasoning="Initialisation", headlines_used=0
        )
        self._current_params = self.structurer.translate(self._current_sentiment)

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Lance la session de paper trading.

        Returns:
            Dictionnaire de metriques de session.
        """
        cfg = self.config
        logger.info("=" * 70)
        logger.info(f"SESSION PAPER TRADING — ID: {self._session_id}")
        logger.info(f"Capital: {cfg.initial_capital} | Analyst: {cfg.analyst_mode.upper()}")
        logger.info("=" * 70)

        obs, _ = self.env.reset(seed=42)
        self.env.update_structurer_params(
            position_size_factor=self._current_params.position_size_factor,
            allowed_actions=self._current_params.allowed_actions,
            max_holding_steps=self._current_params.max_holding_steps,
        )

        done = False
        step = 0
        peak_value = cfg.initial_capital

        while not done:
            step += 1

            # --- Mise a jour sentiment (tous les N steps) ---
            if step % cfg.analyst_update_freq == 1:
                headlines = self.headlines_provider(step)
                self._current_sentiment = self.analyst.analyze(headlines)
                self._current_params = self.structurer.translate(self._current_sentiment)
                self.env.update_structurer_params(
                    position_size_factor=self._current_params.position_size_factor,
                    allowed_actions=self._current_params.allowed_actions,
                    max_holding_steps=self._current_params.max_holding_steps,
                )

            # --- Decision du Trader ---
            if self.trader is not None:
                action_trader = self.trader.predict(obs)
            else:
                action_trader = self.env.action_space.sample()

            # --- Execution (Risk Manager integre dans l'env) ---
            obs, reward, terminated, truncated, info = self.env.step(action_trader)
            done = terminated or truncated

            # --- Tracking ---
            pv = info["portfolio_value"]
            peak_value = max(peak_value, pv)

            log = StepLog(
                step=step,
                timestamp=datetime.now().isoformat(),
                price=info["price"],
                action_trader=action_trader,
                action_executed=info.get("action_executed", action_trader),
                risk_decision=info.get("risk_decision", "ALLOW"),
                position=info["position"],
                capital=info["capital"],
                shares=info["shares"],
                portfolio_value=pv,
                realized_pnl=info["realized_pnl"],
                unrealized_pnl=info["unrealized_pnl"],
                drawdown_pct=info.get("drawdown_pct", 0.0),
                sentiment_score=self._current_sentiment.score,
                sentiment_bias=self._current_sentiment.bias,
                regime=self._current_params.regime,
                position_size_factor=self._current_params.position_size_factor,
            )
            self._step_logs.append(log)

            if cfg.verbose and step % 100 == 0:
                self.env.env.render(mode="human")
                logger.info(f"  Sentiment: {self._current_sentiment.bias} ({self._current_sentiment.score:+.2f}) | Regime: {self._current_params.regime}")

            if cfg.step_delay_sec > 0:
                time.sleep(cfg.step_delay_sec)

        # --- Sauvegarde et rapport ---
        metrics = self._compute_metrics(peak_value)
        if cfg.log_to_csv:
            self._save_logs()
        self._print_report(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Metriques et rapport
    # ------------------------------------------------------------------

    def _compute_metrics(self, peak_value: float) -> dict:
        if not self._step_logs:
            return {}
        final = self._step_logs[-1]
        returns = [
            (self._step_logs[i].portfolio_value - self._step_logs[i-1].portfolio_value)
            / self._step_logs[i-1].portfolio_value
            for i in range(1, len(self._step_logs))
        ]
        returns_arr = np.array(returns)
        sharpe = (
            (returns_arr.mean() / (returns_arr.std() + 1e-8)) * np.sqrt(252)
            if len(returns_arr) > 1 else 0.0
        )
        return {
            "session_id": self._session_id,
            "total_steps": len(self._step_logs),
            "initial_capital": self.config.initial_capital,
            "final_portfolio_value": final.portfolio_value,
            "total_return_pct": (final.portfolio_value / self.config.initial_capital - 1) * 100,
            "realized_pnl": final.realized_pnl,
            "max_drawdown_pct": max(l.drawdown_pct for l in self._step_logs) * 100,
            "sharpe_ratio": sharpe,
            "risk_overrides": self.env._risk_overrides,
            "total_trades": self.env.env._total_trades,
            "analyst_calls": self.analyst.call_count,
        }

    def _save_logs(self) -> None:
        path = LOGS_DIR / f"session_{self._session_id}.csv"
        with open(path, "w", newline="") as f:
            if self._step_logs:
                writer = csv.DictWriter(f, fieldnames=asdict(self._step_logs[0]).keys())
                writer.writeheader()
                writer.writerows(asdict(l) for l in self._step_logs)
        logger.info(f"[Orchestrateur] Logs sauvegardes : {path}")

    def _print_report(self, metrics: dict) -> None:
        logger.info("=" * 70)
        logger.info("RAPPORT DE SESSION")
        logger.info("=" * 70)
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k:<30s}: {v:.4f}")
            else:
                logger.info(f"  {k:<30s}: {v}")
        logger.info("=" * 70)
