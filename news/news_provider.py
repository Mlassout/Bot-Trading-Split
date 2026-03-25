"""
news_provider.py
----------------
Recupere de vraies headlines financieres via yfinance pour alimenter l'Analyste.

Mapping des noms d'actifs internes vers les tickers yfinance :
    BTC_USD  → BTC-USD
    ETH_USD  → ETH-USD
    AAPL     → AAPL
    GSPC     → ^GSPC
    ...

Usage :
    provider = NewsProvider(asset="BTC_USD")
    headlines = provider.get_headlines(max_headlines=5)
    # → ["Bitcoin surges past $90k...", "Fed holds rates steady...", ...]
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping noms internes → tickers yfinance
ASSET_TO_TICKER = {
    # Crypto
    "BTC_USD":  "BTC-USD",
    "ETH_USD":  "ETH-USD",
    "ADA_USD":  "ADA-USD",
    "BNB_USD":  "BNB-USD",
    "SOL_USD":  "SOL-USD",
    "XRP_USD":  "XRP-USD",
    # Actions US
    "AAPL":     "AAPL",
    "AMZN":     "AMZN",
    "MSFT":     "MSFT",
    "NVDA":     "NVDA",
    "TSLA":     "TSLA",
    "CLF":      "CLF",
    # Indices US
    "GSPC":     "^GSPC",
    "NDX":      "^NDX",
    "DJI":      "^DJI",
    "RUT":      "^RUT",
    # Indices mondiaux
    "FCHI":     "^FCHI",
    "GDAXI":    "^GDAXI",
    "FTSE":     "^FTSE",
    "N225":     "^N225",
    "HSI":      "^HSI",
    # Forex
    "EURUSDX":  "EURUSD=X",
    "GBPUSDX":  "GBPUSD=X",
    "USDJPYX":  "USDJPY=X",
    "USDCHFX":  "USDCHF=X",
    "AUDUSDX":  "AUDUSD=X",
    # Matieres premieres
    "GCF":      "GC=F",   # Or
    "HGF":      "HG=F",   # Cuivre
    "NGF":      "NG=F",   # Gaz naturel
    "SIF":      "SI=F",   # Argent
}

# Tickers generiques de contexte macro (toujours inclus pour la vision macro)
_MACRO_TICKERS = ["^GSPC", "BTC-USD", "GC=F"]


class NewsProvider:
    """
    Fournit des headlines financieres reelles via yfinance.

    Args:
        asset:        Nom de l'actif interne (ex: "BTC_USD", "NVDA").
        max_headlines: Nombre max de titres retournes par appel.
        include_macro: Inclure des news macro generales (S&P500, BTC, Or)
                       pour enrichir le contexte de l'Analyste.
    """

    def __init__(
        self,
        asset: str = "GSPC",
        max_headlines: int = 8,
        include_macro: bool = True,
    ):
        self.asset = asset
        self.max_headlines = max_headlines
        self.include_macro = include_macro
        self._ticker = ASSET_TO_TICKER.get(asset, asset)
        self._cache: list[str] = []
        self._cache_calls: int = 0

    def get_headlines(self) -> list[str]:
        """
        Recupere les dernieres headlines pour cet actif.

        Returns:
            Liste de titres (strings). Liste vide si yfinance indisponible.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("[NewsProvider] yfinance non installe. pip install yfinance")
            return []

        headlines = []

        # News specifiques a l'actif
        try:
            ticker_obj = yf.Ticker(self._ticker)
            news_items = ticker_obj.news or []
            for item in news_items[:self.max_headlines]:
                title = self._extract_title(item)
                if title:
                    headlines.append(title)
        except Exception as e:
            logger.warning(f"[NewsProvider] Erreur yfinance ({self._ticker}): {e}")

        # News macro complementaires (si actif n'est pas deja macro)
        if self.include_macro and self._ticker not in _MACRO_TICKERS:
            for macro_ticker in _MACRO_TICKERS:
                try:
                    macro_news = yf.Ticker(macro_ticker).news or []
                    for item in macro_news[:2]:
                        title = self._extract_title(item)
                        if title and title not in headlines:
                            headlines.append(title)
                except Exception:
                    pass

        if headlines:
            logger.info(f"[NewsProvider] {len(headlines)} headlines recuperees pour {self.asset}")
            self._cache = headlines
        else:
            # Fallback sur le cache si yfinance retourne rien
            logger.warning(f"[NewsProvider] Aucune news pour {self.asset}, utilisation du cache.")
            headlines = self._cache

        self._cache_calls += 1
        return headlines[:self.max_headlines]

    def _extract_title(self, item: dict) -> str:
        """Extrait le titre d'un item yfinance (format variable selon la version)."""
        # yfinance >= 0.2.x
        content = item.get("content", {})
        if isinstance(content, dict):
            title = content.get("title", "")
            if title:
                return title.strip()
        # yfinance < 0.2.x
        return item.get("title", "").strip()
