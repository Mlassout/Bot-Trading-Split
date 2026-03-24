"""
stress_test.py
--------------
Module de Gouvernance — Stress Tests.

Injecte des scenarios de marche catastrophiques dans TradingEnv pour verifier
que le Risk Manager et le Trader survivent aux conditions extremes.

Scenarios implementes :
  1. FLASH_CRASH      : chute de -30% instantanee
  2. VOLATILITY_SPIKE : volatilite x10 pendant N bougies
  3. TREND_REVERSAL   : inversion de tendance brutale
  4. LIQUIDITY_CRISIS : transaction costs x20 (spread enorme)
  5. SLOW_BLEED       : baisse lente et constante (-0.5%/bougie)
  6. DEAD_CAT_BOUNCE  : crash suivi d'un rebond partiel (piege haussier)

Usage :
    python governance/stress_test.py
    # ou :
    python main.py stress
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from environment.data_loader import generate_synthetic_ohlcv
from environment.trading_env import TradingEnv
from trader.risk_aware_env import RiskAwareTradingEnv
from risk_manager.risk_manager import RiskManager, RiskConfig, RiskDecision

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class StressScenario(Enum):
    FLASH_CRASH      = "flash_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    TREND_REVERSAL   = "trend_reversal"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SLOW_BLEED       = "slow_bleed"
    DEAD_CAT_BOUNCE  = "dead_cat_bounce"


@dataclass
class StressResult:
    """Resultat d'un scenario de stress test."""
    scenario: str
    passed: bool
    initial_capital: float
    final_portfolio_value: float
    max_drawdown_pct: float
    risk_overrides: int
    total_steps: int
    halt_triggered: bool
    stop_loss_triggered: bool
    notes: str = ""

    @property
    def capital_preservation_pct(self) -> float:
        return (self.final_portfolio_value / self.initial_capital) - 1.0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.scenario:20s} | "
            f"Capital: {self.capital_preservation_pct:+.1%} | "
            f"MaxDD: -{self.max_drawdown_pct:.1%} | "
            f"Overrides Risk: {self.risk_overrides:3d} | "
            f"Halt: {'OUI' if self.halt_triggered else 'non':3s} | "
            f"{self.notes}"
        )


# ------------------------------------------------------------------
# Generateurs de donnees de stress
# ------------------------------------------------------------------

def _make_flash_crash(base_price: float = 100.0, n_steps: int = 500) -> pd.DataFrame:
    """Crash de -30% au step 200, puis stabilisation."""
    df = generate_synthetic_ohlcv(n_steps=n_steps, start_price=base_price, volatility=0.005, seed=1)
    crash_idx = 200
    crash_factor = 0.70
    df.iloc[crash_idx:, :4] *= crash_factor  # open, high, low, close
    # Normaliser pour que high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"]  = df[["open", "low",  "close"]].min(axis=1)
    return df


def _make_volatility_spike(base_price: float = 100.0, n_steps: int = 500) -> pd.DataFrame:
    """Volatilite normale, puis x10 pendant 100 bougies."""
    df_normal = generate_synthetic_ohlcv(n_steps=200, start_price=base_price, volatility=0.005, seed=2)
    last_price = float(df_normal["close"].iloc[-1])
    df_spike   = generate_synthetic_ohlcv(n_steps=100, start_price=last_price, volatility=0.05, seed=3)
    df_after   = generate_synthetic_ohlcv(n_steps=200, start_price=float(df_spike["close"].iloc[-1]),
                                          volatility=0.005, seed=4)
    df = pd.concat([df_normal, df_spike, df_after], ignore_index=True)
    return df


def _make_trend_reversal(base_price: float = 100.0, n_steps: int = 500) -> pd.DataFrame:
    """Tendance haussiere forte, puis inversion brutale baissiere."""
    half = n_steps // 2
    df_up   = generate_synthetic_ohlcv(n_steps=half, start_price=base_price, trend=+0.003, volatility=0.008, seed=5)
    last_price = float(df_up["close"].iloc[-1])
    df_down = generate_synthetic_ohlcv(n_steps=half, start_price=last_price, trend=-0.004, volatility=0.010, seed=6)
    return pd.concat([df_up, df_down], ignore_index=True)


def _make_liquidity_crisis(base_price: float = 100.0, n_steps: int = 300) -> pd.DataFrame:
    """Donnees normales, mais transaction_cost sera gonfle au niveau de l'env."""
    return generate_synthetic_ohlcv(n_steps=n_steps, start_price=base_price, volatility=0.005, seed=7)


def _make_slow_bleed(base_price: float = 100.0, n_steps: int = 500) -> pd.DataFrame:
    """Baisse lente et constante sans rebond."""
    return generate_synthetic_ohlcv(n_steps=n_steps, start_price=base_price, trend=-0.005, volatility=0.003, seed=8)


def _make_dead_cat_bounce(base_price: float = 100.0, n_steps: int = 600) -> pd.DataFrame:
    """Crash -40%, rebond +20%, puis reprise de la baisse."""
    df1 = generate_synthetic_ohlcv(n_steps=150, start_price=base_price, trend=-0.008, volatility=0.015, seed=9)
    p2 = float(df1["close"].iloc[-1])
    df2 = generate_synthetic_ohlcv(n_steps=100, start_price=p2, trend=+0.005, volatility=0.012, seed=10)
    p3 = float(df2["close"].iloc[-1])
    df3 = generate_synthetic_ohlcv(n_steps=350, start_price=p3, trend=-0.006, volatility=0.010, seed=11)
    return pd.concat([df1, df2, df3], ignore_index=True)


_SCENARIO_FACTORIES = {
    StressScenario.FLASH_CRASH:      _make_flash_crash,
    StressScenario.VOLATILITY_SPIKE: _make_volatility_spike,
    StressScenario.TREND_REVERSAL:   _make_trend_reversal,
    StressScenario.LIQUIDITY_CRISIS: _make_liquidity_crisis,
    StressScenario.SLOW_BLEED:       _make_slow_bleed,
    StressScenario.DEAD_CAT_BOUNCE:  _make_dead_cat_bounce,
}


# ------------------------------------------------------------------
# Classe principale
# ------------------------------------------------------------------

class StressTest:
    """
    Executeur de stress tests sur le systeme Trader + Risk Manager.

    Args:
        initial_capital:         Capital de depart.
        risk_config:             Configuration du Risk Manager.
        max_dd_pass_threshold:   Drawdown max acceptable pour "PASS" (ex: 0.12 = 12%).
        agent_policy:            Callable(obs) → action. None = politique aleatoire.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        risk_config: Optional[RiskConfig] = None,
        max_dd_pass_threshold: float = 0.12,
        agent_policy: Optional[Callable] = None,
    ):
        self.initial_capital = initial_capital
        self.risk_config = risk_config or RiskConfig(
            max_drawdown_pct=0.10,
            max_daily_loss_pct=0.02,
            stop_loss_pct=0.02,
        )
        self.max_dd_pass_threshold = max_dd_pass_threshold
        self.agent_policy = agent_policy or (lambda obs: np.random.randint(0, 3))
        self._results: list[StressResult] = []

    def run_all(self, verbose: bool = True) -> list[StressResult]:
        """Lance tous les scenarios de stress test."""
        logger.info("=" * 70)
        logger.info("STRESS TESTS — MODULE GOUVERNANCE")
        logger.info("=" * 70)

        self._results = []
        for scenario in StressScenario:
            result = self.run_scenario(scenario, verbose=False)
            self._results.append(result)
            if verbose:
                logger.info(str(result))

        self._print_summary()
        return self._results

    def run_scenario(
        self,
        scenario: StressScenario,
        verbose: bool = True,
        transaction_cost_override: Optional[float] = None,
    ) -> StressResult:
        """
        Execute un scenario specifique.

        Args:
            scenario:                  Le scenario a tester.
            verbose:                   Afficher les details step-by-step.
            transaction_cost_override: Remplace le cout de transaction (ex: crise liquidite).

        Returns:
            StressResult avec les metriques de survie.
        """
        factory = _SCENARIO_FACTORIES[scenario]
        df = factory(base_price=100.0)

        # Transaction cost eleve pour la crise de liquidite
        txcost = transaction_cost_override
        if scenario == StressScenario.LIQUIDITY_CRISIS:
            txcost = 0.02  # 2% au lieu de 0.1%

        base_env = TradingEnv(
            df=df,
            initial_capital=self.initial_capital,
            transaction_cost=txcost or 0.001,
            holding_penalty=0.0001,
        )
        env = RiskAwareTradingEnv(
            env=base_env,
            risk_config=self.risk_config,
        )

        obs, _ = env.reset(seed=42)
        done = False
        step = 0
        peak_value = self.initial_capital
        max_drawdown = 0.0
        halt_triggered = False
        stop_loss_triggered = False

        while not done:
            action = self.agent_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # Tracking drawdown
            pv = info["portfolio_value"]
            peak_value = max(peak_value, pv)
            dd = (peak_value - pv) / peak_value
            max_drawdown = max(max_drawdown, dd)

            # Tracking evenements Risk Manager
            if info.get("risk_halted"):
                halt_triggered = True
            if info.get("risk_decision") == RiskDecision.BLOCK_SELL.value:
                stop_loss_triggered = True

            if verbose and step % 50 == 0:
                env.env.render(mode="human")

        final_pv = info["portfolio_value"]
        passed = max_drawdown <= self.max_dd_pass_threshold

        notes = []
        if halt_triggered:
            notes.append("Halt declenche")
        if stop_loss_triggered:
            notes.append("Stop-loss declenche")
        if final_pv < self.initial_capital * 0.5:
            notes.append("ALERTE : perte > 50%")
            passed = False

        return StressResult(
            scenario=scenario.value,
            passed=passed,
            initial_capital=self.initial_capital,
            final_portfolio_value=final_pv,
            max_drawdown_pct=max_drawdown,
            risk_overrides=info.get("risk_overrides", 0),
            total_steps=step,
            halt_triggered=halt_triggered,
            stop_loss_triggered=stop_loss_triggered,
            notes=" | ".join(notes) if notes else "RAS",
        )

    def _print_summary(self) -> None:
        if not self._results:
            return
        passed = sum(1 for r in self._results if r.passed)
        total = len(self._results)
        logger.info("=" * 70)
        logger.info(f"RESULTATS STRESS TESTS : {passed}/{total} scenarios passes")
        if passed < total:
            logger.warning("CERTAINS SCENARIOS ONT ECHOUE — Revoir la config du Risk Manager.")
        else:
            logger.info("Tous les scenarios sont passes. Architecture resiliente.")
        logger.info("=" * 70)

    @property
    def results(self) -> list[StressResult]:
        return self._results

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self._results)


if __name__ == "__main__":
    st = StressTest()
    st.run_all(verbose=True)
