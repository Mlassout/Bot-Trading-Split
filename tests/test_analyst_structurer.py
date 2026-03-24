"""
test_analyst_structurer.py
---------------------------
Tests de validation de l'Analyste et du Structureur (Etape 3).

Usage :
    python -m pytest tests/test_analyst_structurer.py -v
    python tests/test_analyst_structurer.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst.analyst import Analyst, AnalystMode, SentimentResult
from structurer.structurer import Structurer


# --- Tests Analyste ---

def test_analyst_mock_bullish_headlines():
    analyst = Analyst(mode=AnalystMode.MOCK)
    headlines = [
        "Markets surge on strong earnings",
        "Tech stocks rally to record highs",
        "Fed signals growth optimism",
    ]
    result = analyst.analyze(headlines)
    assert result.score > 0, f"Score attendu positif, obtenu {result.score}"
    assert result.bias == "bullish"
    assert 0.0 <= result.confidence <= 1.0
    print(f"[PASS] test_analyst_mock_bullish — score={result.score:.2f}, bias={result.bias}")


def test_analyst_mock_bearish_headlines():
    analyst = Analyst(mode=AnalystMode.MOCK)
    headlines = [
        "Stock market crash fears intensify",
        "Recession warning issued by major banks",
        "Inflation surge triggers sell-off",
        "Bear market deepens, losses accelerate",
    ]
    result = analyst.analyze(headlines)
    assert result.score < 0, f"Score attendu negatif, obtenu {result.score}"
    assert result.bias == "bearish"
    print(f"[PASS] test_analyst_mock_bearish — score={result.score:.2f}, bias={result.bias}")


def test_analyst_empty_headlines():
    analyst = Analyst(mode=AnalystMode.MOCK)
    result = analyst.analyze([])
    assert result.bias == "neutral"
    assert result.score == 0.0
    print("[PASS] test_analyst_empty_headlines")


def test_analyst_score_bounds():
    analyst = Analyst(mode=AnalystMode.MOCK)
    for headlines in [
        ["crash crash crash crash crash crash crash crash"],
        ["rally rally rally rally rally rally rally rally"],
        ["the weather is nice today"],
    ]:
        result = analyst.analyze(headlines)
        assert -1.0 <= result.score <= 1.0, f"Score hors bornes : {result.score}"
        assert 0.0 <= result.confidence <= 1.0
    print("[PASS] test_analyst_score_bounds")


def test_analyst_api_fallback_to_mock_without_key():
    """Sans cle API, le mode API doit basculer sur MOCK automatiquement."""
    analyst = Analyst(mode=AnalystMode.API, api_key="")  # cle vide
    assert analyst.mode == AnalystMode.MOCK, "Fallback MOCK attendu sans cle API"
    result = analyst.analyze(["markets rally"])
    assert result is not None
    print("[PASS] test_analyst_api_fallback_to_mock_without_key")


# --- Tests Structureur ---

def test_structurer_strong_bull():
    s = Structurer()
    params = s.translate_score(score=+0.8, confidence=0.9)
    assert params.regime == "strong_bull"
    assert params.position_size_factor == 1.0
    assert 1 in params.allowed_actions
    print(f"[PASS] test_structurer_strong_bull — {params}")


def test_structurer_strong_bear():
    s = Structurer()
    params = s.translate_score(score=-0.8, confidence=0.9)
    assert params.regime == "strong_bear"
    assert params.position_size_factor == 0.0
    assert 1 not in params.allowed_actions, "BUY ne doit pas etre autorise en strong_bear"
    print(f"[PASS] test_structurer_strong_bear — {params}")


def test_structurer_neutral_disables_buy():
    s = Structurer()
    params = s.translate_score(score=0.0, confidence=0.9)
    assert params.regime == "neutral"
    assert 1 not in params.allowed_actions, "BUY ne doit pas etre autorise en neutre"
    print(f"[PASS] test_structurer_neutral_disables_buy — {params}")


def test_structurer_low_confidence_reduces_size():
    s = Structurer(confidence_threshold=0.5)
    params_high_conf = s.translate_score(score=+0.5, confidence=0.9)
    params_low_conf  = s.translate_score(score=+0.5, confidence=0.3)
    assert params_low_conf.position_size_factor < params_high_conf.position_size_factor, (
        f"Taille attendue plus petite avec faible confiance : "
        f"{params_low_conf.position_size_factor:.2f} vs {params_high_conf.position_size_factor:.2f}"
    )
    print(f"[PASS] test_structurer_low_confidence_reduces_size — "
          f"high={params_high_conf.position_size_factor:.0%}, low={params_low_conf.position_size_factor:.0%}")


def test_structurer_from_real_sentiment():
    """Integration Analyste → Structureur."""
    analyst = Analyst(mode=AnalystMode.MOCK)
    structurer = Structurer()
    headlines = ["Tech rally continues", "Strong GDP growth beat expectations"]
    sentiment = analyst.analyze(headlines)
    params = structurer.translate(sentiment)
    assert params.regime in ("strong_bull", "bull", "neutral", "bear", "strong_bear")
    assert 0.0 <= params.position_size_factor <= 1.0
    print(f"[PASS] test_structurer_from_real_sentiment — regime={params.regime}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_analyst_mock_bullish_headlines,
        test_analyst_mock_bearish_headlines,
        test_analyst_empty_headlines,
        test_analyst_score_bounds,
        test_analyst_api_fallback_to_mock_without_key,
        test_structurer_strong_bull,
        test_structurer_strong_bear,
        test_structurer_neutral_disables_buy,
        test_structurer_low_confidence_reduces_size,
        test_structurer_from_real_sentiment,
    ]
    print("=" * 60)
    print("VALIDATION - Analyste + Structureur (Etape 3)")
    print("=" * 60)
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__} — {e}")
            failed += 1
    print("=" * 60)
    print(f"Resultat : {len(tests)-failed}/{len(tests)} tests passes. "
          f"{'Etape 3 VALIDEE.' if not failed else 'Corriger avant de continuer.'}")
    print("=" * 60)
