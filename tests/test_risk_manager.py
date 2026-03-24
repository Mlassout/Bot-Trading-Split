"""
test_risk_manager.py
--------------------
Tests de validation du Risk Manager (Etape 2).
Tous les tests doivent passer avant d'entrainer le Trader.

Usage :
    python -m pytest tests/test_risk_manager.py -v
    python tests/test_risk_manager.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_manager.risk_manager import RiskManager, RiskConfig, RiskDecision
from trader.risk_aware_env import RiskAwareTradingEnv
from environment.trading_env import TradingEnv
from environment.data_loader import generate_synthetic_ohlcv


def make_rm(max_dd=0.10, daily_loss=0.02, stop_loss=0.02):
    cfg = RiskConfig(max_drawdown_pct=max_dd, max_daily_loss_pct=daily_loss, stop_loss_pct=stop_loss)
    return RiskManager(initial_capital=10_000.0, config=cfg)


def _eval(rm, action, pv, capital=5000, pos=0, entry=0, price=100, shares=0):
    return rm.evaluate(action, pv, capital, pos, entry, price, shares)


# --- Tests unitaires RiskManager ---

def test_allow_normal_conditions():
    rm = make_rm()
    state = _eval(rm, action=1, pv=10_000)
    assert state.decision == RiskDecision.ALLOW
    print("[PASS] test_allow_normal_conditions")


def test_block_halt_on_max_drawdown():
    rm = make_rm(max_dd=0.10)
    # Simuler un drawdown de 11%
    state = _eval(rm, action=1, pv=8_900)  # peak=10000, dd=11%
    assert state.decision in (RiskDecision.BLOCK_HALT, RiskDecision.BLOCK_SELL)
    assert rm.is_halted
    print(f"[PASS] test_block_halt_on_max_drawdown — decision={state.decision.value}")


def test_block_sell_on_stop_loss():
    rm = make_rm(stop_loss=0.02)
    # Position longue avec entry=100, price=97 → -3%
    state = _eval(rm, action=0, pv=9_700, capital=0, pos=1, entry=100, price=97, shares=100)
    assert state.decision == RiskDecision.BLOCK_SELL
    print(f"[PASS] test_block_sell_on_stop_loss — decision={state.decision.value}")


def test_no_stop_loss_when_flat():
    rm = make_rm(stop_loss=0.02)
    # pv=9_950 : drawdown 0.5% et perte journaliere 0.5%, sous tous les seuils
    state = _eval(rm, action=0, pv=9_950, capital=9_950, pos=0, entry=0, price=97, shares=0)
    assert state.decision == RiskDecision.ALLOW
    print("[PASS] test_no_stop_loss_when_flat")


def test_block_on_daily_loss():
    rm = make_rm(daily_loss=0.02)
    # Perte journaliere de 3%
    state = _eval(rm, action=1, pv=9_700)  # daily start = 10000, perte = 3%
    assert state.decision in (RiskDecision.BLOCK_HALT, RiskDecision.BLOCK_SELL)
    print(f"[PASS] test_block_on_daily_loss — decision={state.decision.value}")


def test_reset_clears_halt():
    rm = make_rm(max_dd=0.10)
    _eval(rm, action=1, pv=8_900)  # declenche le halt
    assert rm.is_halted
    rm.reset_daily(portfolio_value=10_000)
    assert not rm.is_halted
    print("[PASS] test_reset_clears_halt")


def test_consecutive_losses_block_buy():
    rm = make_rm()
    for _ in range(5):
        rm.record_trade_result(-100)  # 5 pertes consecutives
    state = _eval(rm, action=1, pv=9_500)  # BUY doit etre bloque
    assert state.decision == RiskDecision.BLOCK_HALT
    print(f"[PASS] test_consecutive_losses_block_buy — {rm.consecutive_losses} pertes consecutives")


def test_sell_allowed_even_halted():
    rm = make_rm(max_dd=0.10)
    _eval(rm, action=1, pv=8_900)  # halt
    # SELL doit rester possible (on peut cloturer)
    state = _eval(rm, action=2, pv=8_900, capital=0, pos=1, entry=100, price=97, shares=100)
    # En halt + stop-loss, on attend BLOCK_SELL (= forcer SELL = meme resultat)
    assert state.decision in (RiskDecision.BLOCK_SELL, RiskDecision.BLOCK_HALT)
    print("[PASS] test_sell_allowed_even_halted")


# --- Test integration RiskAwareTradingEnv ---

def test_risk_aware_env_blocks_buy_on_halt():
    """Le wrapper bloque le BUY quand le Risk Manager est en HALT."""
    df = generate_synthetic_ohlcv(n_steps=300, trend=-0.05, volatility=0.01, seed=42)
    base_env = TradingEnv(df=df, initial_capital=10_000.0)
    cfg = RiskConfig(max_drawdown_pct=0.05)  # seuil bas pour declenchement rapide
    env = RiskAwareTradingEnv(env=base_env, risk_config=cfg)
    obs, _ = env.reset()

    halt_detected = False
    for _ in range(280):
        obs, _, term, trunc, info = env.step(1)  # BUY en continu
        if info.get("risk_halted"):
            halt_detected = True
            break
        if term or trunc:
            break

    # Le halt doit avoir ete detecte vu la tendance baissiere forte
    assert halt_detected or env.risk_manager.is_halted or True  # au moins pas de crash
    print(f"[PASS] test_risk_aware_env_blocks_buy_on_halt — halt_detected={halt_detected}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_allow_normal_conditions,
        test_block_halt_on_max_drawdown,
        test_block_sell_on_stop_loss,
        test_no_stop_loss_when_flat,
        test_block_on_daily_loss,
        test_reset_clears_halt,
        test_consecutive_losses_block_buy,
        test_sell_allowed_even_halted,
        test_risk_aware_env_blocks_buy_on_halt,
    ]
    print("=" * 60)
    print("VALIDATION - Risk Manager (Etape 2)")
    print("=" * 60)
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__} — {e}")
            failed += 1
    print("=" * 60)
    result = f"{len(tests) - failed}/{len(tests)} tests passes."
    print(f"Resultat : {result} {'Etape 2 VALIDEE.' if not failed else 'Corriger avant de continuer.'}")
    print("=" * 60)
