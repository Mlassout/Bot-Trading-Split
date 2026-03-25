"""
risk_manager.py
---------------
Risk Manager a regles strictes — le gardien du capital.
Ne prend JAMAIS de decisions de trading, il BLOQUE ou AUTORISE.

Regles implementees (par ordre de priorite) :
  1. HALT  — Drawdown max depuis le pic       (ex: -10% du capital de pic)
  2. HALT  — Perte journaliere max             (ex: -2% du capital initial)
  3. CLOSE — Stop-loss sur trade en cours     (ex: -2% sur la position ouverte)
  4. ALLOW — Tout le reste passe

Decisions possibles :
  RiskDecision.ALLOW      → executer l'action du Trader telle quelle
  RiskDecision.BLOCK_SELL → forcer SELL (cloturer la position immediatement)
  RiskDecision.BLOCK_HALT → forcer HOLD + interdire tout nouveau BUY aujourd'hui
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    ALLOW = "ALLOW"
    BLOCK_SELL = "BLOCK_SELL"   # Force la cloture de la position
    BLOCK_HALT = "BLOCK_HALT"   # Coupe tout, aucun nouveau trade


@dataclass
class RiskState:
    """Snapshot de l'etat de risque courant."""
    decision: RiskDecision
    reason: str
    portfolio_value: float
    peak_value: float
    drawdown_pct: float
    daily_pnl_pct: float
    trade_pnl_pct: float


@dataclass
class RiskConfig:
    """Parametres de configuration du Risk Manager."""
    max_drawdown_pct: float = 0.10        # -10% depuis le pic → HALT
    max_daily_loss_pct: float = 0.02      # -2% de perte journaliere → HALT
    stop_loss_pct: float = 0.02           # -2% sur trade ouvert → CLOSE
    max_position_size_pct: float = 1.0    # 100% du capital (pas de levier)
    max_consecutive_losses: int = 5       # 5 trades perdants d'affilee → HALT


class RiskManager:
    """
    Risk Manager a regles strictes.

    Usage :
        rm = RiskManager(initial_capital=10_000)
        decision_state = rm.evaluate(
            action=1,                   # action proposee par le Trader (0/1/2)
            portfolio_value=9_500,
            capital=9_500,
            position=0,
            entry_price=0.0,
            current_price=100.0,
            shares=0.0,
        )
        if decision_state.decision == RiskDecision.ALLOW:
            # passer l'action a l'environnement
        elif decision_state.decision == RiskDecision.BLOCK_SELL:
            # forcer action=2 (SELL)
        elif decision_state.decision == RiskDecision.BLOCK_HALT:
            # forcer action=0 (HOLD), interdire BUY
    """

    def __init__(
        self,
        initial_capital: float,
        config: Optional[RiskConfig] = None,
    ):
        self.initial_capital = initial_capital
        self.config = config or RiskConfig()

        self._peak_value: float = initial_capital
        self._daily_start_value: float = initial_capital
        self._halted: bool = False
        self._consecutive_losses: int = 0
        self._last_trade_pnl: float = 0.0
        self._last_logged_reason: str = ""  # evite de spammer le meme message

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def evaluate(
        self,
        action: int,
        portfolio_value: float,
        capital: float,
        position: int,
        entry_price: float,
        current_price: float,
        shares: float,
    ) -> RiskState:
        """
        Evalue si l'action proposee est acceptable.

        Args:
            action:          0=HOLD, 1=BUY, 2=SELL
            portfolio_value: valeur totale du portefeuille (capital + shares * price)
            capital:         liquidites disponibles
            position:        0=flat, 1=long
            entry_price:     prix d'entree de la position courante (0 si flat)
            current_price:   prix courant de l'actif
            shares:          nombre d'unites detenues

        Returns:
            RiskState avec la decision finale.
        """
        # Mise a jour du pic
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        drawdown_pct = (self._peak_value - portfolio_value) / self._peak_value
        daily_pnl_pct = (portfolio_value - self._daily_start_value) / self._daily_start_value

        # PnL non-realise sur le trade courant
        if position == 1 and entry_price > 0:
            trade_pnl_pct = (current_price - entry_price) / entry_price
        else:
            trade_pnl_pct = 0.0

        # --- Regle 1 : Drawdown max → HALT ---
        if drawdown_pct >= self.config.max_drawdown_pct:
            reason = (
                f"DRAWDOWN MAX atteint : -{drawdown_pct:.1%} depuis pic "
                f"(limite : -{self.config.max_drawdown_pct:.1%})"
            )
            self._log_once(reason, logger.warning)
            self._halted = True
            decision = RiskDecision.BLOCK_SELL if position == 1 else RiskDecision.BLOCK_HALT
            return RiskState(decision, reason, portfolio_value, self._peak_value,
                             drawdown_pct, daily_pnl_pct, trade_pnl_pct)

        # --- Regle 2 : Perte journaliere max → HALT ---
        if daily_pnl_pct <= -self.config.max_daily_loss_pct:
            reason = (
                f"PERTE JOURNALIERE MAX atteinte : {daily_pnl_pct:.1%} "
                f"(limite : -{self.config.max_daily_loss_pct:.1%})"
            )
            self._log_once(reason, logger.warning)
            self._halted = True
            decision = RiskDecision.BLOCK_SELL if position == 1 else RiskDecision.BLOCK_HALT
            return RiskState(decision, reason, portfolio_value, self._peak_value,
                             drawdown_pct, daily_pnl_pct, trade_pnl_pct)

        # --- Regle 3 : Halt actif → bloquer tout nouveau BUY ---
        if self._halted and action == 1:
            reason = "HALT actif : aucun nouveau BUY autorise jusqu'a reset journalier"
            self._log_once(reason, logger.debug)
            return RiskState(RiskDecision.BLOCK_HALT, reason, portfolio_value,
                             self._peak_value, drawdown_pct, daily_pnl_pct, trade_pnl_pct)

        # --- Regle 4 : Stop-loss sur position ouverte → CLOSE ---
        if position == 1 and trade_pnl_pct <= -self.config.stop_loss_pct:
            reason = (
                f"STOP-LOSS declenche : trade a {trade_pnl_pct:.1%} "
                f"(limite : -{self.config.stop_loss_pct:.1%})"
            )
            self._log_once(reason, logger.warning)
            return RiskState(RiskDecision.BLOCK_SELL, reason, portfolio_value,
                             self._peak_value, drawdown_pct, daily_pnl_pct, trade_pnl_pct)

        # --- Regle 5 : Pertes consecutives → HALT ---
        if self._consecutive_losses >= self.config.max_consecutive_losses and action == 1:
            reason = (
                f"TROP DE PERTES CONSECUTIVES : {self._consecutive_losses} "
                f"(limite : {self.config.max_consecutive_losses})"
            )
            self._log_once(reason, logger.warning)
            return RiskState(RiskDecision.BLOCK_HALT, reason, portfolio_value,
                             self._peak_value, drawdown_pct, daily_pnl_pct, trade_pnl_pct)

        self._last_logged_reason = ""  # reset dedup sur ALLOW

        return RiskState(
            RiskDecision.ALLOW, "OK", portfolio_value, self._peak_value,
            drawdown_pct, daily_pnl_pct, trade_pnl_pct
        )

    def _log_once(self, reason: str, log_fn) -> None:
        """Loggue uniquement si le message est different du precedent (evite le spam)."""
        key = reason[:40]  # cle de deduplication tronquee
        if key != self._last_logged_reason:
            log_fn(f"[RiskManager] {reason}")
            self._last_logged_reason = key

    def record_trade_result(self, pnl: float) -> None:
        """
        A appeler apres chaque cloture de trade pour tracker les pertes consecutives.

        Args:
            pnl: PnL realise du trade (positif = gain, negatif = perte)
        """
        self._last_trade_pnl = pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset_daily(self, portfolio_value: float) -> None:
        """
        A appeler au debut de chaque nouvelle journee de trading.
        Reinitialise les compteurs journaliers et leve le halt si applicable.
        Le peak est remis au niveau actuel : max_drawdown_pct est une limite journaliere.
        """
        self._daily_start_value = portfolio_value
        self._peak_value = portfolio_value  # drawdown repart de zero chaque jour
        self._halted = False
        self._consecutive_losses = 0
        logger.info(f"[RiskManager] Reset journalier. Portefeuille: {portfolio_value:.2f}")

    def reset(self, initial_capital: Optional[float] = None) -> None:
        """Reset complet (nouvel episode)."""
        cap = initial_capital or self.initial_capital
        self._peak_value = cap
        self._daily_start_value = cap
        self._halted = False
        self._consecutive_losses = 0
        self._last_trade_pnl = 0.0

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses
