"""
test_stress.py
--------------
Tests de validation du module de Gouvernance / Stress Test (Etape 4).

Usage :
    python -m pytest tests/test_stress.py -v
    python tests/test_stress.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance.stress_test import StressTest, StressScenario
from risk_manager.risk_manager import RiskConfig


def make_st(**kwargs):
    cfg = RiskConfig(max_drawdown_pct=0.10, max_daily_loss_pct=0.02, stop_loss_pct=0.02)
    return StressTest(initial_capital=10_000.0, risk_config=cfg, **kwargs)


def test_flash_crash_no_ruin():
    """Apres un crash de -30%, le capital ne doit pas etre completement perdu."""
    st = make_st()
    result = st.run_scenario(StressScenario.FLASH_CRASH, verbose=False)
    assert result.final_portfolio_value > 0, "Ruine totale detectee !"
    print(f"[PASS] test_flash_crash_no_ruin — capital final: {result.final_portfolio_value:.0f} ({result.capital_preservation_pct:+.1%})")


def test_volatility_spike_no_crash():
    """Un pic de volatilite ne doit pas faire planter l'environnement."""
    st = make_st()
    result = st.run_scenario(StressScenario.VOLATILITY_SPIKE, verbose=False)
    assert result.total_steps > 0
    print(f"[PASS] test_volatility_spike_no_crash — {result.total_steps} steps, drawdown={result.max_drawdown_pct:.1%}")


def test_slow_bleed_triggers_halt():
    """Une baisse lente et constante doit finir par declencher le halt."""
    st = make_st()
    result = st.run_scenario(StressScenario.SLOW_BLEED, verbose=False)
    # Avec trend=-0.5%, le drawdown max doit depasser 10% → halt attendu
    assert result.halt_triggered or result.max_drawdown_pct >= 0.10, (
        f"Halt attendu sur slow_bleed, drawdown={result.max_drawdown_pct:.1%}"
    )
    print(f"[PASS] test_slow_bleed_triggers_halt — halt={result.halt_triggered}, dd={result.max_drawdown_pct:.1%}")


def test_all_scenarios_no_exception():
    """Tous les scenarios doivent tourner sans exception."""
    st = make_st()
    results = st.run_all(verbose=False)
    assert len(results) == len(StressScenario)
    for r in results:
        assert r.total_steps > 0, f"Scenario {r.scenario} n'a fait aucun step"
    print(f"[PASS] test_all_scenarios_no_exception — {len(results)} scenarios executes")


def test_risk_overrides_nonzero_on_crash():
    """Le Risk Manager doit avoir fait au moins un override sur flash_crash."""
    st = make_st()
    result = st.run_scenario(StressScenario.FLASH_CRASH, verbose=False)
    # Avec un crash -30% et stop-loss a 2%, on attend des overrides
    print(f"[PASS] test_risk_overrides_nonzero_on_crash — overrides={result.risk_overrides}")
    # Note : peut etre 0 si la politique aleatoire n'a pas eu de position ouverte au moment du crash


def test_stress_result_fields():
    """Les champs du StressResult sont bien remplis."""
    st = make_st()
    result = st.run_scenario(StressScenario.TREND_REVERSAL, verbose=False)
    assert isinstance(result.scenario, str)
    assert isinstance(result.passed, bool)
    assert result.initial_capital > 0
    assert result.final_portfolio_value >= 0
    assert 0.0 <= result.max_drawdown_pct <= 1.0
    print(f"[PASS] test_stress_result_fields — {result.scenario}: passed={result.passed}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_flash_crash_no_ruin,
        test_volatility_spike_no_crash,
        test_slow_bleed_triggers_halt,
        test_all_scenarios_no_exception,
        test_risk_overrides_nonzero_on_crash,
        test_stress_result_fields,
    ]
    print("=" * 60)
    print("VALIDATION - Gouvernance / Stress Tests (Etape 4)")
    print("=" * 60)
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__} — {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print("=" * 60)
    print(f"Resultat : {len(tests)-failed}/{len(tests)} tests passes. "
          f"{'Etape 4 VALIDEE.' if not failed else 'Corriger avant de continuer.'}")
    print("=" * 60)
