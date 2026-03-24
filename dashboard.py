"""
dashboard.py
------------
Dashboard de visualisation des sessions de paper trading.

Usage :
    python dashboard.py                        # derniere session
    python dashboard.py logs/session_xxx.csv   # session specifique
    python dashboard.py --paper                # relance une session puis affiche
"""

import sys
import os
import argparse
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------
# Palette et style
# -----------------------------------------------------------------------

COLORS = {
    "price":        "#4A90D9",
    "portfolio":    "#27AE60",
    "buy":          "#2ECC71",
    "sell":         "#E74C3C",
    "halt":         "#E67E22",
    "stop_loss":    "#C0392B",
    "pnl_pos":      "#27AE60",
    "pnl_neg":      "#E74C3C",
    "drawdown":     "#E74C3C",
    "sentiment":    "#9B59B6",
    "bg":           "#1A1A2E",
    "grid":         "#2C2C4E",
    "text":         "#ECF0F1",
    "axes_bg":      "#16213E",
}

REGIME_COLORS = {
    "strong_bull": "#27AE60",
    "bull":        "#82E0AA",
    "neutral":     "#F0B27A",
    "bear":        "#E59866",
    "strong_bear": "#E74C3C",
}


def _apply_dark_style(fig, axes):
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in axes:
        ax.set_facecolor(COLORS["axes_bg"])
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["grid"])
        ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.6)


# -----------------------------------------------------------------------
# Chargement
# -----------------------------------------------------------------------

def load_latest_session(logs_dir: str = "logs") -> pd.DataFrame:
    files = sorted(glob.glob(f"{logs_dir}/session_*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier de session dans '{logs_dir}/'")
    path = files[-1]
    print(f"Session chargee : {path}")
    return pd.read_csv(path), path


def load_session(path: str) -> pd.DataFrame:
    return pd.read_csv(path), path


# -----------------------------------------------------------------------
# Dashboard principal
# -----------------------------------------------------------------------

def plot_dashboard(df: pd.DataFrame, session_path: str = "") -> None:
    n = len(df)
    steps = df["step"].values

    # ----------------------------------------------------------------
    # Pretraitement
    # ----------------------------------------------------------------
    buy_steps  = df[df["action_executed"] == 1].index
    sell_steps = df[df["action_executed"] == 2].index
    halt_steps = df[df["risk_decision"] == "BLOCK_HALT"].index
    sl_steps   = df[df["risk_decision"] == "BLOCK_SELL"].index

    total_return = (df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1) * 100
    max_dd = df["drawdown_pct"].max() * 100
    n_trades = (df["action_executed"] == 2).sum()
    n_overrides = (df["risk_decision"] != "ALLOW").sum()

    # ----------------------------------------------------------------
    # Layout : 3 colonnes x 4 lignes
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Bot Max — Dashboard Trading  |  Session : {Path(session_path).stem}  |  "
        f"Rendement : {total_return:+.2f}%  |  MaxDD : {max_dd:.1f}%  |  "
        f"Trades : {n_trades}  |  Overrides Risk : {n_overrides}",
        color=COLORS["text"], fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    ax_price    = fig.add_subplot(gs[0:2, 0:2])   # grand : prix + trades
    ax_portfolio= fig.add_subplot(gs[2, 0:2])      # portfolio value
    ax_drawdown = fig.add_subplot(gs[3, 0:2])      # drawdown
    ax_pnl      = fig.add_subplot(gs[0, 2])        # PnL realise
    ax_sentiment= fig.add_subplot(gs[1, 2])        # sentiment score
    ax_regime   = fig.add_subplot(gs[2, 2])        # regime timeline
    ax_stats    = fig.add_subplot(gs[3, 2])        # stats textuelles

    all_axes = [ax_price, ax_portfolio, ax_drawdown, ax_pnl, ax_sentiment, ax_regime, ax_stats]
    _apply_dark_style(fig, all_axes)

    # ----------------------------------------------------------------
    # 1. Cours du prix + BUY/SELL/HALT markers
    # ----------------------------------------------------------------
    ax_price.plot(steps, df["price"], color=COLORS["price"], linewidth=1.2, label="Prix")

    if len(buy_steps) > 0:
        ax_price.scatter(
            df.loc[buy_steps, "step"], df.loc[buy_steps, "price"],
            marker="^", color=COLORS["buy"], s=80, zorder=5, label="BUY"
        )
    if len(sell_steps) > 0:
        ax_price.scatter(
            df.loc[sell_steps, "step"], df.loc[sell_steps, "price"],
            marker="v", color=COLORS["sell"], s=80, zorder=5, label="SELL"
        )
    if len(halt_steps) > 0:
        ax_price.axvspan(
            df.loc[halt_steps[0], "step"] if len(halt_steps) > 0 else 0,
            steps[-1],
            alpha=0.08, color=COLORS["halt"], label="Zone HALT"
        )
    if len(sl_steps) > 0:
        ax_price.scatter(
            df.loc[sl_steps, "step"], df.loc[sl_steps, "price"],
            marker="x", color=COLORS["stop_loss"], s=60, zorder=5, label="Stop-Loss"
        )

    ax_price.set_title("Cours du Prix — Signaux de Trading", fontsize=10)
    ax_price.set_ylabel("Prix", fontsize=9)
    ax_price.legend(loc="upper right", fontsize=7, facecolor=COLORS["axes_bg"],
                    labelcolor=COLORS["text"])

    # ----------------------------------------------------------------
    # 2. Valeur du portefeuille
    # ----------------------------------------------------------------
    ax_portfolio.plot(steps, df["portfolio_value"], color=COLORS["portfolio"], linewidth=1.4)
    ax_portfolio.axhline(
        df["portfolio_value"].iloc[0], color=COLORS["text"], linestyle="--",
        linewidth=0.8, alpha=0.5, label="Capital initial"
    )
    ax_portfolio.fill_between(
        steps, df["portfolio_value"].iloc[0], df["portfolio_value"],
        where=df["portfolio_value"] >= df["portfolio_value"].iloc[0],
        alpha=0.2, color=COLORS["pnl_pos"]
    )
    ax_portfolio.fill_between(
        steps, df["portfolio_value"].iloc[0], df["portfolio_value"],
        where=df["portfolio_value"] < df["portfolio_value"].iloc[0],
        alpha=0.2, color=COLORS["pnl_neg"]
    )
    ax_portfolio.set_title("Valeur du Portefeuille", fontsize=10)
    ax_portfolio.set_ylabel("USD", fontsize=9)
    ax_portfolio.legend(loc="lower left", fontsize=7, facecolor=COLORS["axes_bg"],
                        labelcolor=COLORS["text"])

    # ----------------------------------------------------------------
    # 3. Drawdown
    # ----------------------------------------------------------------
    dd = df["drawdown_pct"] * 100
    ax_drawdown.fill_between(steps, 0, -dd, color=COLORS["drawdown"], alpha=0.6)
    ax_drawdown.plot(steps, -dd, color=COLORS["drawdown"], linewidth=1.0)
    ax_drawdown.set_title("Drawdown (%)", fontsize=10)
    ax_drawdown.set_ylabel("%", fontsize=9)
    ax_drawdown.set_xlabel("Step", fontsize=9)

    # ----------------------------------------------------------------
    # 4. PnL realise
    # ----------------------------------------------------------------
    pnl = df["realized_pnl"]
    colors_pnl = [COLORS["pnl_pos"] if v >= 0 else COLORS["pnl_neg"] for v in pnl]
    ax_pnl.plot(steps, pnl, color=COLORS["pnl_pos"], linewidth=1.2)
    ax_pnl.fill_between(steps, 0, pnl,
                         where=pnl >= 0, alpha=0.3, color=COLORS["pnl_pos"])
    ax_pnl.fill_between(steps, 0, pnl,
                         where=pnl < 0, alpha=0.3, color=COLORS["pnl_neg"])
    ax_pnl.axhline(0, color=COLORS["text"], linewidth=0.7, alpha=0.5)
    ax_pnl.set_title("PnL Realise (USD)", fontsize=10)
    ax_pnl.set_ylabel("USD", fontsize=9)

    # ----------------------------------------------------------------
    # 5. Score de sentiment
    # ----------------------------------------------------------------
    ax_sentiment.plot(steps, df["sentiment_score"], color=COLORS["sentiment"],
                      linewidth=1.2, drawstyle="steps-post")
    ax_sentiment.fill_between(steps, 0, df["sentiment_score"],
                               where=df["sentiment_score"] >= 0,
                               alpha=0.3, color=COLORS["pnl_pos"], step="post")
    ax_sentiment.fill_between(steps, 0, df["sentiment_score"],
                               where=df["sentiment_score"] < 0,
                               alpha=0.3, color=COLORS["pnl_neg"], step="post")
    ax_sentiment.axhline(0, color=COLORS["text"], linewidth=0.7, alpha=0.5)
    ax_sentiment.set_ylim(-1.1, 1.1)
    ax_sentiment.set_title("Score Sentiment Analyste", fontsize=10)
    ax_sentiment.set_ylabel("Score", fontsize=9)

    # ----------------------------------------------------------------
    # 6. Timeline des regimes (Structureur)
    # ----------------------------------------------------------------
    regime_map = {"strong_bull": 2, "bull": 1, "neutral": 0, "bear": -1, "strong_bear": -2}
    regime_vals = df["regime"].map(regime_map).fillna(0).values
    for i in range(len(steps) - 1):
        r = df["regime"].iloc[i]
        color = REGIME_COLORS.get(r, COLORS["text"])
        ax_regime.fill_between(
            [steps[i], steps[i+1]], regime_vals[i], 0,
            color=color, alpha=0.7, linewidth=0
        )
    ax_regime.set_yticks([-2, -1, 0, 1, 2])
    ax_regime.set_yticklabels(["StrongBear", "Bear", "Neutral", "Bull", "StrongBull"],
                               fontsize=7)
    ax_regime.set_title("Regime (Structureur)", fontsize=10)
    ax_regime.set_xlabel("Step", fontsize=9)

    # ----------------------------------------------------------------
    # 7. Stats textuelles
    # ----------------------------------------------------------------
    ax_stats.axis("off")
    initial = df["portfolio_value"].iloc[0]
    final   = df["portfolio_value"].iloc[-1]
    pnl_total = df["realized_pnl"].iloc[-1]
    win_trades = (df[df["action_executed"] == 2]["realized_pnl"].diff() > 0).sum()

    stats_lines = [
        ("Capital initial",     f"{initial:,.2f} $"),
        ("Capital final",       f"{final:,.2f} $"),
        ("Rendement total",     f"{total_return:+.2f}%"),
        ("PnL realise",         f"{pnl_total:+.2f} $"),
        ("Drawdown max",        f"-{max_dd:.2f}%"),
        ("Nb trades",           str(int(n_trades))),
        ("Risk overrides",      str(int(n_overrides))),
        ("Total steps",         str(n)),
    ]
    y_pos = 0.95
    ax_stats.text(0.0, 1.02, "Statistiques de Session", transform=ax_stats.transAxes,
                  color=COLORS["text"], fontsize=10, fontweight="bold")
    for label, value in stats_lines:
        color = COLORS["text"]
        if "Rendement" in label or "PnL" in label:
            color = COLORS["pnl_pos"] if pnl_total >= 0 else COLORS["pnl_neg"]
        ax_stats.text(0.02, y_pos, f"{label}:", transform=ax_stats.transAxes,
                      color=COLORS["text"], fontsize=8.5)
        ax_stats.text(0.98, y_pos, value, transform=ax_stats.transAxes,
                      color=color, fontsize=8.5, ha="right", fontweight="bold")
        y_pos -= 0.115

    plt.show()


# -----------------------------------------------------------------------
# Relancer une session paper trading + afficher
# -----------------------------------------------------------------------

def run_and_plot(max_steps: int = 500, model_path: str = None) -> None:
    print("Lancement d'une session paper trading...")
    from orchestrator.orchestrator import Orchestrator, SessionConfig
    from pathlib import Path as _Path

    # Auto-detection du meilleur modele disponible
    if model_path is None:
        candidates = [
            "models/best/best_model.zip",
            "models/trader_ppo.zip",
        ]
        for c in candidates:
            if _Path(c).exists():
                model_path = c
                print(f"Modele detecte : {model_path}")
                break

    cfg = SessionConfig(
        initial_capital=10_000.0,
        analyst_mode="mock",
        verbose=False,
        max_steps=max_steps,
        model_path=model_path,
        log_to_csv=True,
    )
    orch = Orchestrator(config=cfg)
    orch.run()

    # Charger et afficher la session qui vient d'etre generee
    df, path = load_latest_session()
    plot_dashboard(df, path)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser(description="Dashboard Bot Max")
    parser.add_argument("session", nargs="?", default=None,
                        help="Chemin vers un CSV de session (defaut: derniere session)")
    parser.add_argument("--paper", action="store_true",
                        help="Relancer une session paper trading avant d'afficher")
    parser.add_argument("--steps", type=int, default=500,
                        help="Nombre de steps pour --paper (defaut: 500)")
    args = parser.parse_args()

    if args.paper:
        run_and_plot(max_steps=args.steps)
    elif args.session:
        df, path = load_session(args.session)
        plot_dashboard(df, path)
    else:
        df, path = load_latest_session()
        plot_dashboard(df, path)
