# CLAUDE.md — Bot Max : Système de Trading Algorithmique Multi-Agents

## Contexte du projet

Architecture de trading algorithmique inspirée des grandes banques d'investissement.
**Règle d'or : ne jamais passer à l'étape suivante tant que la précédente n'est pas 100% fonctionnelle.**

Idées intégrées depuis les meilleurs dépôts GitHub (FinRL, gym-anytrading, ElegantRL).

## Architecture (multi-agents)

```
NewsProvider (yfinance) → Analyste (LLM) → Structureur → paramètres de trading
                                                               ↓
                                              Orchestrateur (boucle principale)
                                                    ↓              ↓
                                            Trader (PPO RL)    DiscordNotifier
                                                    ↓
                                           RiskAwareTradingEnv
                                            ↓               ↓
                                       Risk Manager     TradingEnv
```

| Module | Dossier | Rôle |
|---|---|---|
| Environnement | `environment/` | Gym custom OHLCV, reward dense mark-to-market, log-norm |
| Risk Manager | `risk_manager/` | Stop-loss, drawdown, perte journalière |
| Trader | `trader/` | Agent PPO (Stable Baselines3), obs=107 dims |
| Analyste | `analyst/` | LLM sentiment macro (Llama/Ollama ou mock) |
| Structureur | `structurer/` | Traduit sentiment → position_size, allowed_actions |
| Orchestrateur | `orchestrator/` | Boucle principale paper trading |
| NewsProvider | `news/` | Vraies headlines financières via yfinance |
| DiscordNotifier | `notifications/` | Notifications webhook Discord (trades, sessions) |
| Gouvernance | `governance/` | Stress tests : flash crash, volatilité extrême |
| Évaluation | `evaluation/` | Comparaison multi-checkpoints, métriques |

## Commandes principales

```bash
# Entraînement sur 30 actifs réels (obs=107, ~25 min)
python main.py train --data data/train --steps 500000

# Évaluation out-of-sample sur tous les checkpoints
python main.py evaluate --checkpoints models/checkpoints/ --test-data data/test/ --vecnorm models/trader_ppo_vecnorm.pkl

# Paper trading — mode mock (rapide, sans LLM)
python main.py paper --model models/best/best_model.zip --analyst mock

# Paper trading — mode API (Llama via Ollama, vraies news yfinance)
python main.py paper --model models/best/best_model.zip --analyst api

# Génération des graphiques comparatifs
python main.py compare
```

## Espace d'observation (107 dimensions)

```
TradingEnv (104 dims) :
  - 20 bougies × 5 features OHLCV (log-normalisés par rapport au close actuel)
  - position actuelle (0=flat, 1=long)
  - PnL non-réalisé en % du capital initial
  - capital disponible en % du capital initial
  - volatilité rolling (std des log-rendements sur la fenêtre)

RiskAwareTradingEnv ajoute +3 dims :
  - risk_halted (0/1) : Risk Manager a bloqué le trading
  - holding_steps normalisé (0→1 sur max_holding_steps=500)
  - dernière action normalisée (0/1/2 → 0.0/0.5/1.0)
```

**CRITIQUE** : tout modèle entraîné avec obs=107 est incompatible avec un autre obs size.
Les anciens modèles (obs=105) sont archivés dans `models/archive_obs105/` — ne pas utiliser.

## Hyperparamètres PPO actuels

| Param | Valeur | Pourquoi |
|---|---|---|
| gamma | 0.985 | Horizon court, adapté aux marchés financiers |
| ent_coef | 0.03 | Exploration active (évite convergence prématurée) |
| net_arch | dict(pi=[256,256], vf=[256,256]) | Réseaux actor/critic séparés |
| activation_fn | Tanh | Plus stable que ReLU en RL |
| learning_rate | linéaire 3e-4 → 1e-4 | Décroissance progressive |
| n_steps | 4096 | Rollout long pour variance réduite |
| batch_size | 128 | |
| holding_penalty | 0.001 | Casse le comportement buy & hold |
| max_holding_steps | 500 | Force la clôture des positions longues |
| n_envs | 4 | Parallélisation |

## Reward Function (dense mark-to-market)

```
reward = step_return - drawdown_penalty - inaction_penalty - holding_penalty
```

- `step_return` : variation mark-to-market / capital initial (signal dense à chaque bougie)
- `drawdown_penalty` : pénalité proportionnelle au drawdown depuis le pic (× 2.0)
- `inaction_penalty=0.0002` : si HOLD sans position (pousse à trader)
- `holding_penalty=0.001` : si HOLD avec position ouverte (pousse à clôturer)

## Données

### Train (`data/train/`) — 30 actifs, ~96k bougies fusionnées
- **Crypto (6)** : BTC, ETH, SOL, BNB, XRP, ADA
- **Indices US (4)** : S&P500, Nasdaq, Dow, Russell 2000
- **Indices monde (5)** : DAX, Nikkei, FTSE, Hang Seng, CAC 40
- **Matières premières (4)** : Gold, Silver, Gas, Copper
- **Forex (5)** : EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF
- **Actions US (6)** : AAPL, MSFT, TSLA, NVDA, AMZN, CLF

Prix normalisés à 100 au départ de chaque actif (`load_multi_ohlcv`).

### Test out-of-sample (`data/test/`) — mêmes 30 actifs, jamais vus en entraînement

## Modèles

```
models/
├── best/best_model.zip         ← meilleur checkpoint selon EvalCallback
├── checkpoints/                ← sauvegarde tous les 50k steps
├── trader_ppo.zip              ← modèle final (run courant)
├── trader_ppo_vecnorm.pkl      ← stats VecNormalize (obligatoire pour l'inférence)
└── archive_obs105/             ← anciens modèles obs=105 (INCOMPATIBLES, ne pas utiliser)
```

**CRITIQUE** : toujours charger `trader_ppo_vecnorm.pkl` avec le modèle correspondant.
Un modèle sans son VecNormalize associé aura des observations mal normalisées → 0 trades.

## Configuration (.env)

```env
# Analyste LLM (Ollama + Llama local)
LLM_API_KEY=ollama
LLM_API_BASE=http://localhost:11434/v1
LLM_MODEL=llama3.2

# Notifications Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Pour utiliser le mode API : `ollama serve` doit tourner, et `ollama pull llama3.2` doit être fait.

## Risk Manager — Règles actuelles

| Règle | Seuil |
|---|---|
| Stop-loss trade | -2% |
| Perte journalière max | -2% |
| Drawdown max depuis pic | -10% |

## Historique des runs d'entraînement

| Run | Obs | Steps | Résultat |
|---|---|---|---|
| #1 | 103 | 200k | Données synthétiques, peu représentatif |
| #2 | 103 | ~200k | 6 actifs réels, divergence rapide |
| #3 | 103→105 | 1M | Reward sparse → divergence à 180k steps, best à 50-100k |
| #4 | 105 | 950k | Reward dense, mais code changé en cours → archivé |
| #5 | **107** | 500k | Best eval à 200k steps (return +3.03%, Sharpe -0.169, 10/30 actifs profitables) |

## Meilleur modèle actuel (Run #5)

- **Checkpoint** : `models/checkpoints/trader_ppo_200000_steps.zip` → copié dans `models/best/best_model.zip`
- **VecNormalize** : `models/trader_ppo_vecnorm.pkl`
- **Évaluation out-of-sample** : return moyen +3.03%, Sharpe -0.169, 10/30 actifs profitables
- **Graphiques** : `evaluation/plots/` (heatmap_returns, metrics_progression, ranking_Xk)
- Pattern confirmé : le modèle surapprentit après 300k steps — best toujours tôt (200-250k)

## Bugs corrigés (session 2026-03-25)

| Fichier | Bug | Fix |
|---|---|---|
| `structurer/structurer.py` | NEUTRAL n'autorisait pas BUY → 0 trades | `[0, 2]` → `[0, 1, 2]` pour neutral |
| `risk_manager/risk_manager.py` | `reset_daily()` ne remettait pas `_peak_value` → halt permanent | Reset aussi `_peak_value` et `_consecutive_losses` |
| `orchestrator/orchestrator.py` | `reset_daily()` jamais appelé → halt définitif dès 1er drawdown | Appel toutes les 78 steps |
| `orchestrator/orchestrator.py` | Discord 429 : notif HALT envoyée à chaque step | Déduplication avec `_last_halt_notified` |
| `orchestrator/orchestrator.py` | Seuils trop serrés pour données synthétiques | stop_loss/daily_loss 2%→5%, drawdown 10%→20% |

## État d'avancement

- [x] Environnement Gym (TradingEnv, reward dense mark-to-market)
- [x] Trader PPO + Risk Manager
- [x] Analyste LLM (Llama/Ollama) + Structureur
- [x] Gouvernance / Stress Test
- [x] Orchestrateur + Paper Trading
- [x] Notifications Discord
- [x] Vraies news financières (yfinance)
- [x] Améliorations FinRL/ElegantRL (log-norm, volatilité, last_action, hypers)
- [x] Entraînement Run #5 (obs=107, 500k steps)
- [x] Évaluation Run #5 — best checkpoint 200k, graphiques générés
- [x] Bugs paper trading corrigés (structurer, risk manager, orchestrateur)
- [ ] Valider paper trading end-to-end sans risk overrides excessifs
- [ ] Connexion courtier réel paper trading (Alpaca, IBKR)

## Points d'attention

- **Exécution** : utiliser le Python Windows `C:\Users\malas\AppData\Local\Programs\Python\Python312\python.exe` (les packages SB3/torch sont installés là, pas dans WSL)
- **Unicode WSL** : préfixer avec `PYTHONIOENCODING=utf-8` si caractères spéciaux dans le terminal
- **Git push** : faire depuis PowerShell Windows (WSL ne peut pas s'authentifier sur GitHub)
- **obs size** : changer `window_size` ou les features invalide TOUS les modèles existants
- **VecNormalize** : modèle + vecnorm doivent toujours être du même run d'entraînement
- **NaN guard** : `_get_observation()` applique `nan_to_num` en sortie — ne pas retirer
- **reset_daily** : le `_peak_value` est remis au niveau du portefeuille courant → drawdown = limite journalière, pas de session
