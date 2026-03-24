"""
main.py
-------
Point d'entree CLI du systeme Bot Max.

Commandes disponibles :

    python main.py validate     — Etape 1 : valider l'environnement de marche
    python main.py train        — Etape 2 : entrainer le Trader RL
    python main.py stress       — Etape 4 : lancer les stress tests de gouvernance
    python main.py paper        — Etape 5 : lancer une session de paper trading

Options communes :
    --data PATH       : chemin vers un CSV OHLCV
    --capital FLOAT   : capital de depart (defaut: 10000)
    --steps INT       : timesteps d'entrainement (defaut: 500000)
    --model PATH      : chemin vers un modele PPO sauvegarde
    --analyst api     : utiliser le vrai LLM (defaut: mock)
    --verbose         : affichage detaille

Exemple :
    python main.py train --steps 300000 --data data/btc_hourly.csv
    python main.py paper --model models/trader_ppo.zip --analyst mock
"""

import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def cmd_validate(args):
    """Lance les tests de validation de l'environnement (Etape 1)."""
    logger.info("=== VALIDATION ENVIRONNEMENT (Etape 1) ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, "tests/test_env.py"],
        capture_output=False,
    )
    sys.exit(result.returncode)


def cmd_train(args):
    """Entraine le Trader RL (Etape 2)."""
    logger.info("=== ENTRAINEMENT TRADER RL (Etape 2) ===")
    from trader.trainer import train
    from risk_manager.risk_manager import RiskConfig

    risk_cfg = RiskConfig(
        max_drawdown_pct=0.10,
        max_daily_loss_pct=0.02,
        stop_loss_pct=0.02,
    )
    # Multi-actifs : si --data pointe sur un dossier, on charge tous les CSV
    import glob, os
    data_paths = None
    data_path = args.data
    if data_path and os.path.isdir(data_path):
        data_paths = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        logger.info(f"Dossier detecte : {len(data_paths)} fichiers CSV")
        data_path = None

    model = train(
        total_timesteps=args.steps,
        data_path=data_path,
        data_paths=data_paths,
        risk_config=risk_cfg,
        initial_capital=args.capital,
    )
    logger.info("Entrainement termine.")


def cmd_stress(args):
    """Lance les stress tests (Etape 4)."""
    logger.info("=== STRESS TESTS GOUVERNANCE (Etape 4) ===")
    from governance.stress_test import StressTest
    from risk_manager.risk_manager import RiskConfig

    risk_cfg = RiskConfig(
        max_drawdown_pct=0.10,
        max_daily_loss_pct=0.02,
        stop_loss_pct=0.02,
    )

    # Si un modele est disponible, utiliser le vrai Trader
    agent_policy = None
    if args.model:
        try:
            from trader.trader_agent import TraderAgent
            agent = TraderAgent.from_file(args.model)
            agent_policy = lambda obs: agent.predict(obs)
            logger.info(f"Modele charge pour le stress test : {args.model}")
        except Exception as e:
            logger.warning(f"Impossible de charger le modele : {e}. Politique aleatoire utilisee.")

    st = StressTest(
        initial_capital=args.capital,
        risk_config=risk_cfg,
        agent_policy=agent_policy,
    )
    results = st.run_all(verbose=args.verbose)

    passed = sum(1 for r in results if r.passed)
    logger.info(f"\nBilan : {passed}/{len(results)} scenarios passes.")
    sys.exit(0 if st.all_passed else 1)


def cmd_paper(args):
    """Lance une session de paper trading (Etape 5)."""
    logger.info("=== PAPER TRADING (Etape 5) ===")
    from orchestrator.orchestrator import Orchestrator, SessionConfig

    cfg = SessionConfig(
        initial_capital=args.capital,
        data_path=args.data,
        model_path=args.model,
        analyst_mode=args.analyst,
        verbose=args.verbose,
        max_steps=args.max_steps,
        analyst_update_freq=50,
        log_to_csv=True,
    )
    orch = Orchestrator(config=cfg)
    metrics = orch.run()

    total_return = metrics.get("total_return_pct", 0.0)
    logger.info(f"\nSession terminee. Rendement total : {total_return:+.2f}%")


# ------------------------------------------------------------------
# Parser CLI
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bot Max — Systeme de trading algorithmique multi-agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Commun
    def add_common(p):
        p.add_argument("--data",     type=str,   default=None,     help="Chemin CSV OHLCV")
        p.add_argument("--capital",  type=float, default=10_000.0, help="Capital de depart")
        p.add_argument("--model",    type=str,   default=None,     help="Chemin modele PPO (.zip)")
        p.add_argument("--verbose",  action="store_true",          help="Affichage detaille")

    # validate
    p_val = sub.add_parser("validate", help="Valider l'environnement de marche (Etape 1)")
    add_common(p_val)

    # train
    p_train = sub.add_parser("train", help="Entrainer le Trader RL (Etape 2)")
    add_common(p_train)
    p_train.add_argument("--steps", type=int, default=500_000, help="Timesteps d'entrainement")

    # stress
    p_stress = sub.add_parser("stress", help="Stress tests de gouvernance (Etape 4)")
    add_common(p_stress)

    # paper
    p_paper = sub.add_parser("paper", help="Session de paper trading (Etape 5)")
    add_common(p_paper)
    p_paper.add_argument("--analyst",   type=str, default="mock",  choices=["mock", "api"],
                         help="Mode de l'Analyste (mock ou api)")
    p_paper.add_argument("--max-steps", type=int, default=None, dest="max_steps",
                         help="Nombre max de steps de la session")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluer le modele sur tous les actifs de test")
    p_eval.add_argument("--test-data",    type=str, default="data/test",           dest="test_data",
                        help="Dossier contenant les CSV de test (defaut: data/test)")
    p_eval.add_argument("--checkpoints",  type=str, default=None,
                        help="Dossier checkpoints pour comparaison par etape (ex: models/checkpoints)")
    p_eval.add_argument("--model",        type=str, default="models/best/best_model.zip",
                        help="Modele a evaluer si pas de --checkpoints")
    p_eval.add_argument("--capital",      type=float, default=10_000.0)
    p_eval.add_argument("--output",       type=str, default="evaluation/results_by_checkpoint.csv",
                        help="Chemin du CSV de resultats")
    p_eval.add_argument("--vecnorm",      type=str, default=None,
                        help="Chemin du VecNormalize pkl (optionnel)")

    # compare
    p_cmp = sub.add_parser("compare", help="Generer les graphiques de comparaison depuis les resultats")
    p_cmp.add_argument("--results", type=str, default="evaluation/results_by_checkpoint.csv",
                       help="CSV produit par la commande evaluate")
    p_cmp.add_argument("--output-dir", type=str, default="evaluation/plots", dest="output_dir")

    return parser


def cmd_evaluate(args):
    """Evalue le modele sur les 30 actifs de test."""
    logger.info("=== EVALUATION OUT-OF-SAMPLE (2024-2026) ===")
    from evaluation.evaluator import evaluate_all_assets, evaluate_all_checkpoints
    from pathlib import Path

    if args.checkpoints:
        logger.info(f"Mode multi-checkpoints : {args.checkpoints}")
        df = evaluate_all_checkpoints(
            checkpoints_dir=args.checkpoints,
            test_data_dir=args.test_data,
            output_csv=args.output,
            initial_capital=args.capital,
            vecnorm_path=args.vecnorm,
        )
    else:
        logger.info(f"Modele : {args.model}")
        df = evaluate_all_assets(
            model_path=args.model,
            test_data_dir=args.test_data,
            initial_capital=args.capital,
            vecnorm_path=args.vecnorm,
        )
        Path(args.output).parent.mkdir(exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"Resultats : {args.output}")

    from evaluation.compare import print_summary_table
    print_summary_table(df)


def cmd_compare(args):
    """Genere les graphiques de comparaison."""
    logger.info("=== COMPARAISON DES PERFORMANCES ===")
    from evaluation.compare import run_comparison
    run_comparison(results_csv=args.results, output_dir=args.output_dir)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "validate": cmd_validate,
        "train":    cmd_train,
        "stress":   cmd_stress,
        "paper":    cmd_paper,
        "evaluate": cmd_evaluate,
        "compare":  cmd_compare,
    }
    commands[args.command](args)
