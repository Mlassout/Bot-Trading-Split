"""
structurer.py
-------------
Le Structureur : traduit le sentiment macro de l'Analyste en parametres
operationnels pour le Trader et le Risk Manager.

C'est la "courroie de transmission" entre la vision macro et l'execution.

Regles de traduction (exemple institutionnel) :

  Score sentiment | Biais       | Position size | Actions autorisees | Max holding
  ────────────────┼─────────────┼───────────────┼────────────────────┼────────────
  [+0.5, +1.0]   | Fort haussier  | 100%          | HOLD, BUY, SELL    | None
  [+0.2, +0.5[   | Haussier       | 75%           | HOLD, BUY, SELL    | 100 steps
  [-0.2, +0.2[   | Neutre         | 50%           | HOLD, SELL         | 50 steps
  [-0.5, -0.2[   | Baissier       | 25%           | HOLD, SELL         | 20 steps
  [-1.0, -0.5[   | Fort baissier  | 0%            | HOLD               | 0 (fermer)

  La confiance de l'Analyste module encore ces parametres (confidence < 0.4 → plus conservateur).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from analyst.analyst import SentimentResult

logger = logging.getLogger(__name__)


@dataclass
class TradingParams:
    """
    Parametres operationnels generes par le Structureur.
    Injectes dans RiskAwareTradingEnv via update_structurer_params().
    """
    position_size_factor: float         # [0.0, 1.0] — fraction du capital par trade
    allowed_actions: list               # sous-ensemble de [0, 1, 2]
    max_holding_steps: Optional[int]    # None = pas de limite
    regime: str                         # "strong_bull" | "bull" | "neutral" | "bear" | "strong_bear"
    sentiment_score: float              # score brut de l'Analyste
    confidence: float                   # confiance de l'Analyste

    def is_trading_allowed(self) -> bool:
        """Retourne True si au moins un trade actif est possible."""
        return 1 in self.allowed_actions or 2 in self.allowed_actions

    def __str__(self) -> str:
        actions_str = {0: "HOLD", 1: "BUY", 2: "SELL"}
        acts = [actions_str[a] for a in self.allowed_actions]
        hold_str = str(self.max_holding_steps) if self.max_holding_steps else "illimite"
        return (
            f"[Structureur] Regime: {self.regime.upper():12s} | "
            f"Size: {self.position_size_factor:.0%} | "
            f"Actions: {acts} | "
            f"MaxHolding: {hold_str} steps"
        )


# Seuils de sentiment
_THRESHOLDS = [
    (+0.50,  "strong_bull", 1.00, [0, 1, 2], None),
    (+0.20,  "bull",        0.75, [0, 1, 2], 100),
    (-0.20,  "neutral",     0.50, [0, 2],    50),
    (-0.50,  "bear",        0.25, [0, 2],    20),
    (-1.01,  "strong_bear", 0.00, [0],       0),
]


class Structurer:
    """
    Traduit le sentiment macro en parametres de trading.

    Args:
        confidence_threshold: En-dessous de ce seuil de confiance, reduit la
                               position_size_factor de 50% supplementaires.
        min_position_size:    Plancher de position_size_factor (evite 0% complet
                               sauf en strong_bear explicite).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        min_position_size: float = 0.10,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_position_size = min_position_size
        self._last_params: Optional[TradingParams] = None

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def translate(self, sentiment: SentimentResult) -> TradingParams:
        """
        Traduit un SentimentResult en TradingParams.

        Args:
            sentiment: Resultat de l'Analyste.

        Returns:
            TradingParams a injecter dans RiskAwareTradingEnv.
        """
        score = sentiment.score
        confidence = sentiment.confidence

        # Determiner le regime et les parametres de base
        regime, base_size, allowed, max_hold = self._lookup_regime(score)

        # Modulation par la confiance
        final_size = base_size
        if confidence < self.confidence_threshold and regime != "strong_bear":
            # Faible confiance → reduire de moitie, garder un plancher
            final_size = max(self.min_position_size, base_size * 0.5)
            logger.info(
                f"[Structureur] Confiance faible ({confidence:.0%}) → "
                f"position_size reduite : {base_size:.0%} → {final_size:.0%}"
            )

        # En strong_bear, forcer la fermeture immediate (max_holding_steps=0)
        if regime == "strong_bear":
            final_size = 0.0
            allowed = [0]     # HOLD uniquement (force la cloture des positions existantes)
            max_hold = 0

        params = TradingParams(
            position_size_factor=final_size,
            allowed_actions=allowed,
            max_holding_steps=max_hold if max_hold != 0 else None,
            regime=regime,
            sentiment_score=score,
            confidence=confidence,
        )

        self._last_params = params
        logger.info(str(params))
        return params

    def translate_score(self, score: float, confidence: float = 1.0) -> TradingParams:
        """
        Raccourci pour creer un SentimentResult minimal et traduire.

        Args:
            score:      Score de sentiment [-1.0, +1.0]
            confidence: Confiance [0.0, 1.0]
        """
        bias = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
        synthetic = SentimentResult(
            bias=bias, score=score, confidence=confidence,
            reasoning="Score synthetique", headlines_used=0
        )
        return self.translate(synthetic)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _lookup_regime(self, score: float) -> tuple:
        """Retourne (regime, base_size, allowed_actions, max_holding) selon le score."""
        for threshold, regime, size, allowed, max_hold in _THRESHOLDS:
            if score >= threshold:
                return regime, size, allowed[:], max_hold
        return "strong_bear", 0.0, [0], 0

    @property
    def last_params(self) -> Optional[TradingParams]:
        return self._last_params
