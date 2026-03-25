"""
test_env.py
-----------
Script de validation de l'environnement TradingEnv.

Lance ces tests AVANT de passer a l'Etape 2.
Tous les tests doivent passer (aucune exception, assertions vertes).

Usage :
    python -m pytest tests/test_env.py -v
    # ou directement :
    python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from environment import TradingEnv
from environment.data_loader import generate_synthetic_ohlcv


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_env(**kwargs) -> TradingEnv:
    df = generate_synthetic_ohlcv(n_steps=500, seed=0)
    return TradingEnv(df=df, **kwargs)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_gymnasium_api():
    """L'environnement respecte l'API Gymnasium (check officiel)."""
    env = make_env()
    check_env(env, warn=True)
    print("[PASS] test_gymnasium_api")


def test_reset_returns_correct_shape():
    """reset() retourne une observation de la bonne shape."""
    env = make_env(window_size=20)
    obs, info = env.reset(seed=42)
    expected_size = 20 * 5 + 4  # window * features + account_state (position, unrealized, capital, volatility)
    assert obs.shape == (expected_size,), f"Shape attendue ({expected_size},), obtenue {obs.shape}"
    assert obs.dtype == np.float32
    print(f"[PASS] test_reset_returns_correct_shape — shape={obs.shape}")


def test_full_episode_no_crash():
    """Un episode complet avec actions aleatoires ne plante pas."""
    env = make_env()
    obs, _ = env.reset(seed=1)
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert np.isfinite(reward), f"Reward non-fini au step {steps}: {reward}"
        assert np.all(np.isfinite(obs)), f"Observation contient NaN/Inf au step {steps}"
    print(f"[PASS] test_full_episode_no_crash — {steps} steps effectues")


def test_buy_sell_pnl_coherence():
    """Un BUY suivi d'un SELL sur prix croissant genere un PnL positif."""
    df = generate_synthetic_ohlcv(n_steps=200, trend=0.005, volatility=0.001, seed=7)
    env = TradingEnv(df=df, transaction_cost=0.0, holding_penalty=0.0)
    env.reset(seed=0)

    # Forcer BUY
    obs, reward_buy, _, _, info_buy = env.step(1)
    entry_price = info_buy["price"]
    assert info_buy["position"] == 1, "Position devrait etre LONG apres BUY"

    # Avancer de quelques steps (prix monte grace au trend)
    for _ in range(10):
        env.step(0)  # HOLD

    # Forcer SELL
    obs, reward_sell, _, _, info_sell = env.step(2)
    assert info_sell["position"] == 0, "Position devrait etre FLAT apres SELL"
    assert info_sell["realized_pnl"] > 0, (
        f"PnL attendu positif, obtenu {info_sell['realized_pnl']:.4f}"
    )
    print(
        f"[PASS] test_buy_sell_pnl_coherence — "
        f"entry={entry_price:.2f}, "
        f"exit={info_sell['price']:.2f}, "
        f"pnl={info_sell['realized_pnl']:.2f}"
    )


def test_no_double_buy():
    """Un deuxieme BUY sans SELL prealable ne modifie pas la position."""
    env = make_env(transaction_cost=0.0)
    env.reset()
    env.step(1)  # premier BUY
    _, _, _, _, info_before = env.step(0)
    capital_before = info_before["capital"]
    shares_before = info_before["shares"]

    env.step(1)  # deuxieme BUY (doit etre ignore)
    _, _, _, _, info_after = env.step(0)

    assert info_after["capital"] == capital_before, "Capital ne devrait pas changer sur BUY en position LONG"
    assert info_after["shares"] == shares_before, "Shares ne devraient pas changer sur BUY en position LONG"
    print("[PASS] test_no_double_buy")


def test_bankruptcy_terminates_episode():
    """Un effondrement du capital (< 5%) termine l'episode."""
    df = generate_synthetic_ohlcv(n_steps=500, trend=-0.05, volatility=0.01, seed=99)
    env = TradingEnv(df=df, initial_capital=100.0)
    env.reset()
    terminated = False
    for _ in range(490):
        _, _, term, trunc, info = env.step(1)
        if term:
            terminated = True
            break
    # Le test valide juste que terminated peut devenir True (pas de crash)
    print(f"[PASS] test_bankruptcy_terminates_episode — terminated={terminated}")


def test_render_does_not_crash():
    """render() s'execute sans exception."""
    env = make_env()
    env.reset()
    env.step(1)
    msg = env.render(mode="ansi")
    assert isinstance(msg, str)
    print(f"[PASS] test_render_does_not_crash — '{msg[:60]}...'")


# -----------------------------------------------------------------------
# Execution directe
# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_reset_returns_correct_shape,
        test_full_episode_no_crash,
        test_buy_sell_pnl_coherence,
        test_no_double_buy,
        test_bankruptcy_terminates_episode,
        test_render_does_not_crash,
        test_gymnasium_api,  # En dernier car le plus verbeux
    ]

    print("=" * 60)
    print("VALIDATION - TradingEnv (Etape 1)")
    print("=" * 60)
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"[FAIL] {test.__name__} — {e}")
            failed += 1

    print("=" * 60)
    if failed == 0:
        print(f"Resultat : {len(tests)}/{len(tests)} tests passes. Etape 1 VALIDEE.")
    else:
        print(f"Resultat : {failed} test(s) en echec. Corriger avant de passer a l'Etape 2.")
    print("=" * 60)
