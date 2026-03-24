# CLAUDE.md — Bot Max : Système de Trading Algorithmique Multi-Agents

## Contexte du projet

Architecture de trading algorithmique inspirée des grandes banques d'investissement.
**Règle d'or : ne jamais passer à l'étape suivante tant que la précédente n'est pas 100% fonctionnelle.**

## Architecture (5 micro-services)

```
Analyste (LLM) → Structureur → Trader (RL) → Risk Manager → Environnement
                                                ↑
                                          Gouvernance (Stress Test)
```

| Module | Dossier | Rôle |
|---|---|---|
| Environnement | `environment/` | Gym custom OHLCV, reward dense mark-to-market |
| Risk Manager | `risk_manager/` | Règles strictes : stop-loss, drawdown, perte journalière |
| Trader | `trader/` | Agent PPO (Stable Baselines3) |
| Analyste | `analyst/` | LLM sentiment macro (mock + API OpenAI-compatible) |
| Structureur | `structureur/` | Traduit sentiment → paramètres de trading |
| Gouvernance | `governance/` | Stress tests : flash crash, volatilité extrême |
| Orchestrateur | `orchestrator/` | Boucle principale, paper trading |
| Évaluation | `evaluation/` | Comparaison multi-checkpoints, graphiques |

## Commandes principales

```bash
# Entraînement sur 30 actifs réels
python main.py train --data data/train --steps 1000000

# Évaluation out-of-sample (2024-2026) sur tous les checkpoints
python main.py evaluate --checkpoints models/checkpoints/ --test-data data/test/ --vecnorm models/trader_ppo_vecnorm.pkl

# Génération des graphiques comparatifs
python main.py compare

# Paper trading avec le meilleur modèle
python main.py paper --data data/test --model models/best/best_model.zip

# Dashboard interactif
python dashboard.py
```

## Données

### Train (2010/2018 → 2023-12-31) — `data/train/`
30 actifs réels Yahoo Finance (daily) :
- **Crypto (6)** : BTC, ETH, SOL, BNB, XRP, ADA
- **Indices US (4)** : S&P500, Nasdaq, Dow, Russell 2000
- **Indices monde (5)** : DAX, Nikkei, FTSE, Hang Seng, CAC 40
- **Matières premières (5)** : Gold, Oil, Silver, Gas, Copper
- **Forex (5)** : EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF
- **Actions US (5)** : AAPL, MSFT, TSLA, NVDA, AMZN

### Test out-of-sample (2024-01-01 → 2026-03-24) — `data/test/`
Mêmes 30 actifs, **jamais vus pendant l'entraînement**.

## Reward Function (dense mark-to-market)

Fichier : `environment/trading_env.py`

```
reward = step_return - drawdown_penalty - inaction_penalty
```

- `step_return` : variation mark-to-market du portefeuille / capital initial (signal dense à chaque bougie)
- `drawdown_penalty` : pénalité proportionnelle au drawdown depuis le pic (× `drawdown_penalty_factor=2.0`)
- `inaction_penalty` : pénalité si HOLD sans position ouverte (`0.0002` par step)

**Pourquoi cette reward ?** L'ancienne reward sparse (seulement au SELL) causait un comportement "hold forever" — le modèle divergeait après 180k steps.

## Résultats connus

### Run #1 — Données synthétiques (GBM)
- Reward très volatile, peu représentatif du marché réel

### Run #2 — 6 actifs réels (BTC, ETH, S&P500, Nasdaq, Gold, EUR/USD)
- Meilleure reward à 30k steps (+841), puis divergence

### Run #3 — 30 actifs réels, reward sparse (1M steps)
- **Best model : 50k–100k steps** (BTC +62%, NVDA +38%, Gold +76%)
- **Divergence après 180k steps** : reward chute à -856, épisodes 4× plus longs
- Cause : reward sparse → comportement "hold forever" sur 96k bougies

### Run #4 — 30 actifs, reward dense mark-to-market (EN COURS)
- Objectif : éliminer la divergence, stabiliser l'apprentissage

## Modèles sauvegardés

```
models/
├── best/best_model.zip         ← meilleur checkpoint évalué (Run #3 : ~50k steps)
├── checkpoints/                ← checkpoints tous les 50k steps
│   ├── trader_ppo_50000_steps.zip
│   ├── trader_ppo_100000_steps.zip
│   └── ...
├── trader_ppo.zip              ← modèle final (souvent sous-optimal vs best)
└── trader_ppo_vecnorm.pkl      ← normalisation VecNormalize (obligatoire pour l'inférence)
```

## Risk Manager — Règles actuelles

| Règle | Seuil |
|---|---|
| Stop-loss trade | -2% |
| Perte journalière max | -2% |
| Drawdown max depuis pic | -10% |
| Taille position max | 95% du capital |

## État d'avancement des étapes

- [x] **Étape 1** : Environnement Gym (TradingEnv, reward dense)
- [x] **Étape 2** : Trader PPO + Risk Manager (opérationnel)
- [x] **Étape 3** : Analyste (mode mock) + Structureur (opérationnel)
- [x] **Étape 4** : Gouvernance / Stress Test (opérationnel)
- [x] **Étape 5** : Orchestrateur + Paper Trading (opérationnel)
- [ ] **Optimisation** : reward dense → stabiliser la convergence après 180k steps
- [ ] **Prochaine étape** : connecter un vrai courtier en mode paper (ex: Alpaca, IBKR)

## Stack technique

```
Python 3.11
gymnasium>=0.29         # environnement RL
stable-baselines3>=2.3  # algorithme PPO
torch>=2.2              # backend deep learning
yfinance>=1.2           # données réelles
openai>=1.0             # API LLM Analyste (compatible Mistral)
pandas, numpy, matplotlib
python-dotenv           # gestion clés API (.env)
```

## Variables d'environnement (.env)

```env
OPENAI_API_KEY=...      # ou clé Mistral pour l'Analyste
OPENAI_BASE_URL=...     # ex: https://api.mistral.ai/v1 pour Mistral
```

## Points d'attention

- **Ne jamais utiliser `models/trader_ppo.zip` directement** : c'est le modèle final souvent dégradé. Toujours préférer `models/best/best_model.zip`.
- **VecNormalize obligatoire** : charger `trader_ppo_vecnorm.pkl` avec le modèle, sinon l'observation est mal normalisée.
- **Obs space** : `window_size=20` bougies × 5 features + 4 variables de compte = **104 features**. Tout changement de `window_size` invalide les modèles existants.
- **Logs de session** : chaque paper trading génère un CSV dans `logs/session_YYYYMMDD_HHMMSS.csv`.
- **Graphiques d'évaluation** : générés dans `evaluation/plots/`.
