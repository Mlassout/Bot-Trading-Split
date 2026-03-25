"""
discord_notifier.py
-------------------
Envoie des notifications Discord lors des evenements de trading importants.

Configure via variable d'environnement DISCORD_WEBHOOK_URL ou directement
en passant l'URL au constructeur.

Evenements notifies :
  - Session demarree / terminee
  - BUY execute
  - SELL execute (avec PnL du trade)
  - Risk Manager : HALT ou STOP-LOSS declenche
"""

import os
import logging
import requests
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """
    Envoie des messages enrichis sur un webhook Discord.

    Args:
        webhook_url: URL du webhook Discord. Si None, utilise DISCORD_WEBHOOK_URL de l'env.
        asset:       Nom de l'actif trade (ex: "BTC_USD").
        enabled:     Desactiver proprement sans retirer le code (utile en backtest).
    """

    COLORS = {
        "buy":      0x2ECC71,   # vert
        "sell_win": 0x2ECC71,   # vert
        "sell_loss":0xE74C3C,   # rouge
        "halt":     0xE74C3C,   # rouge
        "info":     0x3498DB,   # bleu
        "warning":  0xF39C12,   # orange
    }

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        asset: str = "UNKNOWN",
        enabled: bool = True,
    ):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self.asset = asset
        self.enabled = enabled and bool(self.webhook_url)

        if enabled and not self.webhook_url:
            logger.warning("[Discord] DISCORD_WEBHOOK_URL non definie — notifications desactivees.")

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def session_start(self, initial_capital: float, model_path: str = "") -> None:
        self._send(
            title="🚀 Session démarrée",
            description=f"Bot Max est en ligne sur **{self.asset}**",
            color=self.COLORS["info"],
            fields=[
                ("Capital initial", f"${initial_capital:,.2f}", True),
                ("Modèle", model_path or "N/A", True),
                ("Heure", datetime.now().strftime("%H:%M:%S"), True),
            ],
        )

    def session_end(self, final_capital: float, initial_capital: float,
                    total_trades: int, realized_pnl: float) -> None:
        total_return = (final_capital / initial_capital - 1) * 100
        color = self.COLORS["sell_win"] if total_return >= 0 else self.COLORS["sell_loss"]
        sign = "+" if total_return >= 0 else ""
        self._send(
            title="🏁 Session terminée",
            description=f"Session clôturée sur **{self.asset}**",
            color=color,
            fields=[
                ("Capital final",   f"${final_capital:,.2f}", True),
                ("Rendement total", f"{sign}{total_return:.2f}%", True),
                ("PnL réalisé",     f"${realized_pnl:+,.2f}", True),
                ("Trades exécutés", str(total_trades), True),
            ],
        )

    def trade_buy(self, price: float, capital_used: float, step: int) -> None:
        self._send(
            title="📈 BUY",
            description=f"Position ouverte sur **{self.asset}**",
            color=self.COLORS["buy"],
            fields=[
                ("Prix d'entrée", f"${price:,.4f}", True),
                ("Capital engagé", f"${capital_used:,.2f}", True),
                ("Step", str(step), True),
            ],
        )

    def trade_sell(self, price: float, pnl: float, pnl_pct: float,
                   entry_price: float, step: int) -> None:
        color = self.COLORS["sell_win"] if pnl >= 0 else self.COLORS["sell_loss"]
        emoji = "✅" if pnl >= 0 else "❌"
        sign = "+" if pnl >= 0 else ""
        self._send(
            title=f"{emoji} SELL",
            description=f"Position fermée sur **{self.asset}**",
            color=color,
            fields=[
                ("Prix d'entrée",  f"${entry_price:,.4f}", True),
                ("Prix de sortie", f"${price:,.4f}", True),
                ("PnL",            f"{sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)", True),
                ("Step", str(step), True),
            ],
        )

    def risk_halt(self, reason: str, drawdown_pct: float, portfolio_value: float) -> None:
        self._send(
            title="🛑 RISK MANAGER — HALT",
            description=f"Trading suspendu sur **{self.asset}**",
            color=self.COLORS["halt"],
            fields=[
                ("Raison",     reason, False),
                ("Drawdown",   f"-{drawdown_pct:.2f}%", True),
                ("Portefeuille", f"${portfolio_value:,.2f}", True),
            ],
        )

    def risk_stop_loss(self, price: float, entry_price: float, loss_pct: float) -> None:
        self._send(
            title="⛔ STOP-LOSS déclenché",
            description=f"Position liquidée sur **{self.asset}**",
            color=self.COLORS["halt"],
            fields=[
                ("Prix d'entrée",  f"${entry_price:,.4f}", True),
                ("Prix actuel",    f"${price:,.4f}", True),
                ("Perte sur trade", f"-{abs(loss_pct):.2f}%", True),
            ],
        )

    # ------------------------------------------------------------------
    # Envoi HTTP
    # ------------------------------------------------------------------

    def _send(self, title: str, description: str, color: int,
              fields: list[tuple] = None) -> None:
        if not self.enabled:
            return

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Bot Max • Trading RL"},
        }

        if fields:
            embed["fields"] = [
                {"name": name, "value": value, "inline": inline}
                for name, value, inline in fields
            ]

        payload = {"embeds": [embed]}

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            if resp.status_code not in (200, 204):
                logger.warning(f"[Discord] Erreur HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[Discord] Impossible d'envoyer la notification : {e}")
