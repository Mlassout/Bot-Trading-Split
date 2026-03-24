"""
analyst.py
----------
L'Analyste : interprete le sentiment macro-economique a partir de titres financiers.

Deux modes de fonctionnement :
  - MOCK    : analyse par mots-cles, aucune API requise. Ideal pour les tests.
  - API     : appel a un LLM via l'API OpenAI-compatible (Mistral, OpenAI, Ollama, etc.)
              Configure via variable d'environnement ou parametres.

Sortie standardisee :
    SentimentResult(
        bias="bullish",       # bullish | bearish | neutral
        score=0.75,           # [-1.0 a +1.0], positif = haussier
        confidence=0.8,       # [0.0 a 1.0]
        reasoning="...",      # explication en texte libre
        headlines_used=3,     # nombre de titres analyses
    )
"""

import os
import re
import logging
import json
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AnalystMode(Enum):
    MOCK = "mock"
    API  = "api"


@dataclass
class SentimentResult:
    """Resultat standardise de l'analyse de sentiment."""
    bias: str              # "bullish" | "bearish" | "neutral"
    score: float           # [-1.0, +1.0]
    confidence: float      # [0.0, 1.0]
    reasoning: str
    headlines_used: int = 0
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "bias": self.bias,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "headlines_used": self.headlines_used,
        }

    def __str__(self) -> str:
        sign = "+" if self.score >= 0 else ""
        return (
            f"[Analyste] Biais: {self.bias.upper():8s} | "
            f"Score: {sign}{self.score:.2f} | "
            f"Confiance: {self.confidence:.0%} | "
            f"{self.reasoning[:80]}"
        )


# ------------------------------------------------------------------
# Mots-cles pour le mode MOCK
# ------------------------------------------------------------------

_BULLISH_KEYWORDS = [
    "rally", "surge", "gain", "rise", "bull", "record high", "growth",
    "strong", "beat", "profit", "revenue", "positive", "upgrade", "buy",
    "hausse", "progression", "croissance", "benefice", "rebond", "achat",
]

_BEARISH_KEYWORDS = [
    "crash", "fall", "drop", "loss", "bear", "recession", "inflation",
    "rate hike", "layoff", "warning", "miss", "downgrade", "sell", "risk",
    "baisse", "chute", "perte", "recession", "crise", "vente", "deficit",
]

# ------------------------------------------------------------------
# Prompt LLM
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """Tu es un analyste financier senior specialise dans l'analyse de sentiment macro-economique.
On te donnera des titres de presse financiere. Tu dois retourner UNIQUEMENT un objet JSON valide avec ces champs :
{
  "bias": "bullish" | "bearish" | "neutral",
  "score": float entre -1.0 (tres bearish) et +1.0 (tres bullish),
  "confidence": float entre 0.0 et 1.0,
  "reasoning": "explication concise en 1-2 phrases"
}
Ne retourne rien d'autre que ce JSON."""


class Analyst:
    """
    Analyste de sentiment macro-economique.

    Args:
        mode:        AnalystMode.MOCK ou AnalystMode.API
        api_key:     Cle API (ou None, utilise LLM_API_KEY de l'env)
        api_base:    URL de base de l'API (defaut: OpenAI, compatible Mistral/Ollama)
        model:       Nom du modele LLM (ex: "mistral-small", "gpt-4o-mini")
        max_tokens:  Tokens max pour la reponse du LLM
    """

    DEFAULT_API_BASE  = "https://api.openai.com/v1"
    DEFAULT_MODEL     = "gpt-4o-mini"

    def __init__(
        self,
        mode: AnalystMode = AnalystMode.MOCK,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 256,
    ):
        self.mode = mode
        self.max_tokens = max_tokens
        self._api_key  = api_key or os.getenv("LLM_API_KEY", "")
        self._api_base = api_base or os.getenv("LLM_API_BASE", self.DEFAULT_API_BASE)
        self._model    = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self._call_count = 0

        if mode == AnalystMode.API and not self._api_key:
            logger.warning(
                "[Analyst] Mode API active mais LLM_API_KEY non definie. "
                "Fallback automatique sur le mode MOCK."
            )
            self.mode = AnalystMode.MOCK

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def analyze(self, headlines: list[str]) -> SentimentResult:
        """
        Analyse une liste de titres financiers et retourne un score de sentiment.

        Args:
            headlines: Liste de titres de presse (strings).

        Returns:
            SentimentResult avec biais, score et explication.
        """
        if not headlines:
            return SentimentResult(
                bias="neutral", score=0.0, confidence=0.0,
                reasoning="Aucun titre fourni.", headlines_used=0
            )

        self._call_count += 1
        if self.mode == AnalystMode.MOCK:
            result = self._analyze_mock(headlines)
        else:
            result = self._analyze_api(headlines)

        logger.info(str(result))
        return result

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyse un texte libre (le decoupe en lignes comme titres)."""
        headlines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return self.analyze(headlines)

    # ------------------------------------------------------------------
    # Mode MOCK (analyse par mots-cles)
    # ------------------------------------------------------------------

    def _analyze_mock(self, headlines: list[str]) -> SentimentResult:
        combined = " ".join(headlines).lower()

        bullish_hits = sum(1 for kw in _BULLISH_KEYWORDS if kw in combined)
        bearish_hits = sum(1 for kw in _BEARISH_KEYWORDS if kw in combined)
        total_hits = bullish_hits + bearish_hits

        if total_hits == 0:
            score = 0.0
            bias = "neutral"
            confidence = 0.3
            reasoning = "Aucun signal clair dans les titres analyses."
        else:
            raw_score = (bullish_hits - bearish_hits) / total_hits
            score = max(-1.0, min(1.0, raw_score))
            confidence = min(0.9, 0.3 + (total_hits / len(headlines)) * 0.6)

            if score > 0.2:
                bias = "bullish"
                reasoning = f"{bullish_hits} signaux haussiers vs {bearish_hits} baissiers."
            elif score < -0.2:
                bias = "bearish"
                reasoning = f"{bearish_hits} signaux baissiers vs {bullish_hits} haussiers."
            else:
                bias = "neutral"
                reasoning = f"Signaux equilibres : {bullish_hits} haussiers, {bearish_hits} baissiers."

        return SentimentResult(
            bias=bias, score=score, confidence=confidence,
            reasoning=reasoning, headlines_used=len(headlines),
            raw_response="[MOCK]"
        )

    # ------------------------------------------------------------------
    # Mode API (appel LLM reel)
    # ------------------------------------------------------------------

    def _analyze_api(self, headlines: list[str]) -> SentimentResult:
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("[Analyst] Package 'openai' non installe. Fallback MOCK.")
            return self._analyze_mock(headlines)

        headlines_text = "\n".join(f"- {h}" for h in headlines)
        user_message = f"Analyse le sentiment de ces titres financiers :\n{headlines_text}"

        try:
            client = OpenAI(api_key=self._api_key, base_url=self._api_base)
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=0.1,
            )
            raw_text = response.choices[0].message.content.strip()
            return self._parse_llm_response(raw_text, len(headlines))

        except Exception as e:
            logger.error(f"[Analyst] Erreur API LLM : {e}. Fallback MOCK.")
            return self._analyze_mock(headlines)

    def _parse_llm_response(self, raw_text: str, n_headlines: int) -> SentimentResult:
        """Parse la reponse JSON du LLM."""
        try:
            # Extraire le JSON meme si le LLM a ajoute du texte autour
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not json_match:
                raise ValueError("Aucun JSON trouve dans la reponse.")
            data = json.loads(json_match.group())

            bias  = str(data.get("bias", "neutral")).lower()
            score = float(data.get("score", 0.0))
            confidence = float(data.get("confidence", 0.5))
            reasoning  = str(data.get("reasoning", ""))

            # Validation
            bias = bias if bias in ("bullish", "bearish", "neutral") else "neutral"
            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            return SentimentResult(
                bias=bias, score=score, confidence=confidence,
                reasoning=reasoning, headlines_used=n_headlines,
                raw_response=raw_text
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"[Analyst] Impossible de parser la reponse LLM : {e}. Fallback MOCK.")
            # Fallback sur mock si le LLM renvoie un format invalide
            return SentimentResult(
                bias="neutral", score=0.0, confidence=0.2,
                reasoning=f"Erreur de parsing LLM : {e}",
                headlines_used=n_headlines, raw_response=raw_text
            )

    @property
    def call_count(self) -> int:
        return self._call_count
