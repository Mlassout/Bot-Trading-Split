"""
compare.py
----------
Charge les resultats d'evaluation et genere les graphiques de comparaison.

Usage :
    python main.py compare --results evaluation/results_by_checkpoint.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

COLORS = plt.cm.tab20.colors


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "checkpoint_steps" in df.columns:
        df["checkpoint_steps"] = pd.to_numeric(df["checkpoint_steps"], errors="coerce")
    return df


def plot_returns_by_asset(df: pd.DataFrame, output_dir: str = "evaluation/plots") -> None:
    """
    Graphique 1 : Rendement final par actif pour chaque checkpoint.
    Permet de voir quels actifs le modele maitrise en premier.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "checkpoint_steps" not in df.columns:
        # Modele unique — bar chart simple
        fig, ax = plt.subplots(figsize=(14, 6))
        df_sorted = df.sort_values("total_return_pct", ascending=True)
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_sorted["total_return_pct"]]
        bars = ax.barh(df_sorted["asset"], df_sorted["total_return_pct"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Rendement (%)")
        ax.set_title("Rendement par actif — Test out-of-sample 2024-2026")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        plt.tight_layout()
        path = os.path.join(output_dir, "returns_by_asset.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Sauvegarde : {path}")
        return

    # Multi-checkpoints — heatmap
    pivot = df.pivot_table(index="asset", columns="checkpoint_steps", values="total_return_pct")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), max(8, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-30, vmax=30)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c/1000)}k" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7,
                        color="black" if abs(val) < 20 else "white")
    plt.colorbar(im, ax=ax, label="Rendement (%)")
    ax.set_xlabel("Checkpoint (steps)")
    ax.set_title("Rendement par actif selon le checkpoint d'entrainement")
    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap_returns.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Sauvegarde : {path}")


def plot_metrics_progression(df: pd.DataFrame, output_dir: str = "evaluation/plots") -> None:
    """
    Graphique 2 : Evolution des metriques agregees au fil des checkpoints.
    Montre si le modele s'ameliore globalement avec plus d'entrainement.
    """
    if "checkpoint_steps" not in df.columns or df["checkpoint_steps"].isna().all():
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    agg = df.groupby("checkpoint_steps").agg(
        mean_return=("total_return_pct", "mean"),
        median_return=("total_return_pct", "median"),
        mean_sharpe=("sharpe", "mean"),
        mean_drawdown=("max_drawdown_pct", "mean"),
        pct_profitable=("total_return_pct", lambda x: (x > 0).mean() * 100),
    ).reset_index()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    def steps_labels(x):
        return [f"{int(v/1000)}k" for v in x]

    # 1. Rendement moyen
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(agg["checkpoint_steps"], agg["mean_return"], "o-", color="#3498db", label="Moyenne")
    ax1.plot(agg["checkpoint_steps"], agg["median_return"], "s--", color="#9b59b6", label="Mediane")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax1.fill_between(agg["checkpoint_steps"], agg["mean_return"], 0,
                     alpha=0.15, color="#3498db")
    ax1.set_title("Rendement moyen / median (%)")
    ax1.set_xticklabels(steps_labels(agg["checkpoint_steps"]))
    ax1.legend(fontsize=8)

    # 2. Sharpe moyen
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(agg["checkpoint_steps"], agg["mean_sharpe"], "o-", color="#2ecc71")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.axhline(1, color="#f39c12", linewidth=1, linestyle="--", label="Sharpe = 1 (bon)")
    ax2.set_title("Sharpe ratio moyen")
    ax2.set_xticklabels(steps_labels(agg["checkpoint_steps"]))
    ax2.legend(fontsize=8)

    # 3. Drawdown max moyen
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(agg["checkpoint_steps"], agg["mean_drawdown"], "o-", color="#e74c3c")
    ax3.axhline(-10, color="#f39c12", linewidth=1, linestyle="--", label="Limite -10%")
    ax3.set_title("Drawdown max moyen (%)")
    ax3.set_xticklabels(steps_labels(agg["checkpoint_steps"]))
    ax3.legend(fontsize=8)

    # 4. % actifs profitables
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(range(len(agg)), agg["pct_profitable"],
            color=["#2ecc71" if v >= 50 else "#e74c3c" for v in agg["pct_profitable"]])
    ax4.axhline(50, color="black", linewidth=0.8, linestyle=":")
    ax4.set_xticks(range(len(agg)))
    ax4.set_xticklabels(steps_labels(agg["checkpoint_steps"]))
    ax4.set_title("% actifs profitables")
    ax4.set_ylim(0, 100)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Checkpoint (steps)")

    fig.suptitle("Progression du modele au fil de l'entrainement", fontsize=13, fontweight="bold")
    path = os.path.join(output_dir, "metrics_progression.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Sauvegarde : {path}")


def plot_asset_ranking(df: pd.DataFrame, checkpoint_steps: int = None,
                       output_dir: str = "evaluation/plots") -> None:
    """
    Graphique 3 : Classement des actifs (Return, Sharpe, Drawdown) pour un checkpoint donne.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if checkpoint_steps and "checkpoint_steps" in df.columns:
        sub = df[df["checkpoint_steps"] == checkpoint_steps].copy()
    else:
        sub = df.copy()

    if sub.empty:
        return

    sub = sub.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(sub) * 0.35)))

    # Return
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub["total_return_pct"]]
    axes[0].barh(sub["asset"], sub["total_return_pct"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Rendement (%)")
    axes[0].invert_yaxis()

    # Sharpe
    colors2 = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub["sharpe"]]
    axes[1].barh(sub["asset"], sub["sharpe"], color=colors2)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(1, color="#f39c12", linewidth=1, linestyle="--")
    axes[1].set_title("Sharpe ratio")
    axes[1].invert_yaxis()

    # Drawdown
    axes[2].barh(sub["asset"], sub["max_drawdown_pct"], color="#e74c3c")
    axes[2].axvline(-10, color="#f39c12", linewidth=1, linestyle="--")
    axes[2].set_title("Max Drawdown (%)")
    axes[2].invert_yaxis()

    label = f" — {checkpoint_steps:,} steps" if checkpoint_steps else ""
    fig.suptitle(f"Classement des 30 actifs{label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"ranking{'_' + str(checkpoint_steps) if checkpoint_steps else ''}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Sauvegarde : {path}")


def print_summary_table(df: pd.DataFrame, checkpoint_steps: int = None) -> None:
    """Affiche un tableau recapitulatif dans le terminal."""
    if checkpoint_steps and "checkpoint_steps" in df.columns:
        sub = df[df["checkpoint_steps"] == checkpoint_steps].copy()
    else:
        sub = df.copy()

    sub = sub.sort_values("total_return_pct", ascending=False)

    print(f"\n{'='*80}")
    print(f"{'ACTIF':<18} {'RETURN':>8} {'SHARPE':>8} {'DRAWDOWN':>10} {'WIN%':>7} {'TRADES':>7}")
    print(f"{'-'*80}")
    for _, row in sub.iterrows():
        sign = "+" if row["total_return_pct"] >= 0 else ""
        print(
            f"{row['asset']:<18} "
            f"{sign}{row['total_return_pct']:>7.2f}%  "
            f"{row['sharpe']:>8.3f}  "
            f"{row['max_drawdown_pct']:>9.2f}%  "
            f"{row['win_rate_pct']:>6.1f}%  "
            f"{int(row['n_trades']):>6}"
        )

    profitable = (sub["total_return_pct"] > 0).sum()
    print(f"{'-'*80}")
    print(f"  Actifs profitables : {profitable}/{len(sub)}  |  "
          f"Return moyen : {sub['total_return_pct'].mean():+.2f}%  |  "
          f"Sharpe moyen : {sub['sharpe'].mean():.3f}  |  "
          f"DD moyen : {sub['max_drawdown_pct'].mean():.2f}%")
    print(f"{'='*80}\n")


def run_comparison(results_csv: str, output_dir: str = "evaluation/plots") -> None:
    """Point d'entree principal : charge les resultats et genere tous les graphiques."""
    df = load_results(results_csv)
    print(f"Resultats charges : {len(df)} lignes, {df['asset'].nunique()} actifs")

    checkpoints = sorted(df["checkpoint_steps"].dropna().unique()) if "checkpoint_steps" in df.columns else []

    plot_returns_by_asset(df, output_dir)
    plot_metrics_progression(df, output_dir)

    # Classement pour chaque checkpoint
    for ckpt in checkpoints:
        plot_asset_ranking(df, checkpoint_steps=int(ckpt), output_dir=output_dir)
        print_summary_table(df, checkpoint_steps=int(ckpt))

    if not checkpoints:
        plot_asset_ranking(df, output_dir=output_dir)
        print_summary_table(df)

    print(f"\nTous les graphiques sauvegardes dans : {output_dir}/")
