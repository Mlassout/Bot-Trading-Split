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

# Dashboard sur la dernière session
python dashboard.py

# Dashboard sur une session spécifique
python dashboard.py logs/session_xxx.csv
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
| Pertes consécutives | 5 trades → HALT |

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
| `trader/trader_agent.py` | `PPO.load()` segfault sur Python 3.13 macOS (pickle cross-platform) | `_load_ppo_safe()` : charge `policy.pth` via `torch.load(weights_only=True)` |

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
- [x] Fix compatibilité macOS Python 3.13 (PPO.load → _load_ppo_safe)
- [ ] Valider paper trading end-to-end sans risk overrides excessifs
- [ ] Connexion courtier réel paper trading (Alpaca, IBKR)
- [ ] Indicateurs techniques (RSI, ATR, EMA, MACD) dans l'observation
- [ ] Walk-forward analysis
- [ ] Trailing stop dans le Risk Manager
- [ ] Corriger facteur d'annualisation Sharpe (horaire → sqrt(252*24))

## Points d'attention

- **Exécution macOS** : utiliser `source venv/bin/activate` (venv créé le 2026-03-25)
- **Exécution Windows** : `C:\Users\malas\AppData\Local\Programs\Python\Python312\python.exe`
- **Unicode WSL** : préfixer avec `PYTHONIOENCODING=utf-8` si caractères spéciaux dans le terminal
- **Git push** : faire depuis PowerShell Windows (WSL ne peut pas s'authentifier sur GitHub)
- **obs size** : changer `window_size` ou les features invalide TOUS les modèles existants
- **VecNormalize** : modèle + vecnorm doivent toujours être du même run d'entraînement
- **NaN guard** : `_get_observation()` applique `nan_to_num` en sortie — ne pas retirer
- **reset_daily** : le `_peak_value` est remis au niveau du portefeuille courant → drawdown = limite journalière, pas de session
- **use_real_news=True par défaut** : en paper trading avec yfinance, les news actuelles bearish bloquent tous les trades (régime STRONG_BEAR). Passer `use_real_news=False` pour tester avec headlines synthétiques rotatives.
- **Seuils risque incohérents** : training (stop_loss/daily_loss=2%, DD=10%) ≠ orchestrateur (5%/5%/20%) — à aligner avant le prochain run

---

# AUDIT TECHNIQUE — 2026-03-25

> Audit complet réalisé par analyse statique de tout le code source.
> Niveau de maturité estimé : **3/10 production, 7/10 prototype de recherche**.

## Diagnostic global

Bot Max est un prototype de recherche bien structuré avec une vision architecturale claire (Alpha Generator → Portfolio Manager → Risk Desk). Les performances réelles sont faibles (Sharpe -0.169, 10/30 actifs profitables) en raison d'un signal trop faible : absence d'indicateurs techniques, analyse de sentiment naïve (comptage de mots-clés), pas de walk-forward. L'architecture est bonne, les composants sont là — c'est le signal qui manque.

## Points forts

| Point fort | Localisation | Détail |
|---|---|---|
| Architecture multi-agents | Tous modules | Séparation signal macro / décision tactique / risk guardrails |
| Environnement Gym correct | `trading_env.py` | API Gymnasium complète, log-norm OHLCV, clôture forcée fin épisode |
| Log-normalisation OHLCV | `trading_env.py:224` | `log(price/ref_price)` — meilleure stationnarité que division linéaire |
| Volatilité rolling | `trading_env.py:230` | std des log-returns sur fenêtre — proxy turbulence pertinent |
| Risk Manager à règles dures | `risk_manager.py` | 5 règles claires, séparé du RL, bien testé |
| VecNormalize correct | `trainer.py:165` | norm_obs + norm_reward + clip_obs=10 + pkl sauvegardé |
| Hyperparamètres PPO | `trainer.py:196-216` | gamma=0.985, ent_coef=0.03, Tanh, LR schedule — bien fondés |
| Évaluation out-of-sample | `evaluator.py` | 30 actifs, Sharpe/Calmar/MaxDD/WinRate, multi-checkpoints |
| Stress tests | `stress_test.py` | 6 scénarios (flash crash, volatility spike, dead cat bounce...) |
| Tests unitaires | `tests/` | check_env officiel, cas limites PnL, stop-loss, double-buy |
| Dashboard | `dashboard.py` | 7 panneaux, dark theme, markers BUY/SELL/HALT |
| Dense reward | `trading_env.py:171-187` | Mark-to-market à chaque bougie — évite reward sparse |

## Bugs et fragilités critiques

### BUG CRITIQUE — Capital patching pour le position sizing
**Fichier** : `risk_aware_env.py:127-136`
```python
original_capital = self.env._capital
self.env._capital = original_capital * self.position_size_factor
# ... step() ...
self.env._capital += unused
```
Accès direct à l'attribut privé `_capital`. Si le risk manager change l'action entre la modification et le step, le capital est corrompu. **À refactoriser** : passer `position_size_factor` comme paramètre de `TradingEnv.step()`.

### BUG CRITIQUE — Facteur Sharpe incorrect pour données horaires
**Fichier** : `evaluator.py:48`
```python
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # FAUX pour horaire
# Correct pour données horaires :
sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # ≈ 77.8
```
Le Sharpe actuel est **surestimé d'un facteur ~5.5** pour des données horaires.

### BUG FORT — VecNormalize eval_env sans stats train
**Fichier** : `trainer.py:168-171`
L'`eval_env` crée une nouvelle `VecNormalize` avec `training=False` mais sans charger les stats de `train_env`. Ses stats sont initialisées à `mean=0, std=1` → l'EvalCallback sélectionne le meilleur modèle avec des observations mal normalisées.

### BUG FORT — `evaluate_all_assets` utilise `PPO.load()` (segfault Python 3.13)
**Fichier** : `evaluator.py:159`
Même bug que `trader_agent.py` — non corrigé dans l'évaluateur.

### FRAGILITÉ — Reward poisoning par risk overrides
**Fichier** : `risk_aware_env.py:116-122`
Le PPO reçoit `RISK_OVERRIDE_PENALTY=-0.05` quand le Risk Manager intervient, mais l'agent ne peut pas apprendre à éviter ces overrides (le risque n'est pas observable). Cela perturbe l'apprentissage sans signal utile.

### FRAGILITÉ — Seuils risque incohérents training/paper
**Fichier** : `orchestrator.py:106-109`
Training : stop_loss=2%, daily_loss=2%, DD=10%
Paper trading : stop_loss=5%, daily_loss=5%, DD=20%
Le modèle apprend avec des règles strictes mais est déployé avec des règles laxistes.

### FRAGILITÉ — Split train/eval sur données multi-actifs concaténées
**Fichier** : `trainer.py:155-158`
Le split 80/20 coupe dans la concaténation de 30 actifs. Pas de séparation temporelle stricte par actif → risque de leakage si les actifs de test et de train couvrent la même période.

### FRAGILITÉ — Drawdown multi-jours non protégé
`reset_daily()` remet `_peak_value` au capital courant chaque matin. Un bot qui perd 5% par jour pendant 10 jours n'est jamais halted (chaque jour repart du capital de la veille).

## Techniques absentes ou sous-exploitées

### CRITIQUE — Indicateurs techniques
**Absent** : RSI, MACD, ATR, Bollinger Bands, EMA/SMA, volume relatif.
**Impact** : Le modèle ne voit que des prix bruts sur 20 bougies. Sans indicateurs, le signal est quasi-inexistant → principale cause du Sharpe négatif.
**Fix** : Ajouter `pandas-ta` dans `trading_env.py:_get_observation()`.

### CRITIQUE — Walk-forward analysis
**Absent** : Un seul split fixe. Impossible de savoir si les résultats sont robustes dans le temps.
**Fix** : Fenêtre glissante 6 mois entraînement / 1 mois test dans `evaluation/`.

### FORT — Slippage et market impact réalistes
**Absent** : Seul `transaction_cost=0.001` fixe. En réalité : spread bid-ask variable (0.05-0.5% sur crypto), impact de marché, latence.

### FORT — Take-profit / Trailing stop
**Absent** dans le Risk Manager. Les gains non-sécurisés s'évaporent.

### FORT — Identifiant d'actif dans l'observation
**Absent** : En multi-actifs, l'agent ne sait pas quel actif il trade. Un one-hot encoding des 30 actifs améliorerait la généralisation.

### FORT — Validation statistique du Sharpe
**Absent** : Pas de bootstrap, pas de permutation test. Le "Probabilistic Sharpe Ratio" (López de Prado) permettrait de tester la significativité.

### MOYEN — MLflow pour tracking des expériences
**Absent** : Les runs sont journalisés manuellement dans ce fichier. MLflow donnerait reproductibilité et comparaison automatique.

### MOYEN — HMM pour détection de régime
**Présent de manière naïve** : Le structureur utilise des keywords LLM. Un HMM sur returns + volatilité (hmmlearn, 2-3 états) serait bien plus fiable et basé sur les données.

## Bibliothèques recommandées

| Bibliothèque | Usage | Priorité | Coût intégration |
|---|---|---|---|
| `pandas-ta` | Indicateurs techniques (RSI, ATR, EMA, MACD) | **HAUTE** | Faible |
| `vectorbt` | Backtesting vectorisé, métriques avancées, benchmark buy-and-hold | **HAUTE** | Moyen |
| `optuna` | Optimisation hyperparamètres PPO + reward weights | **HAUTE** | Faible |
| `lightgbm` | Classificateur direction T+1 comme signal complémentaire | MOYENNE | Moyen |
| `hmmlearn` | Détection de régime basée données (remplace sentiment keywords) | MOYENNE | Faible |
| `mlflow` | Tracking expériences, reproductibilité des runs | MOYENNE | Faible |
| `pyfolio` / `empyrical` | Métriques correctes (Sharpe annualisé, Sortino, Omega) | MOYENNE | Faible |
| `ccxt` | Connexion exchanges crypto pour live trading | MOYENNE | Élevé |
| `shap` | Explainability — quelles features le modèle utilise | BASSE | Moyen |
| `polars` | Remplacement pandas pour datasets > 1M lignes | BASSE | Moyen |

## Roadmap priorisée

### Étape 1 — Corrections critiques (1-2 jours)
1. Corriger facteur Sharpe : `sqrt(252*24)` pour horaire — `evaluator.py:48`
2. Corriger VecNormalize eval_env : charger stats depuis train_env — `trainer.py:168-171`
3. Fixer `evaluator.py:159` avec `_load_ppo_safe` (même fix que trader_agent.py)
4. Aligner seuils risque training/orchestrateur — `orchestrator.py:106-109`
5. Refactoriser capital patching → paramètre `position_size` dans `TradingEnv.step()` — `risk_aware_env.py:127-136`

### Étape 2 — Features engineering (1 semaine)
6. Intégrer `pandas-ta` : RSI(14), ATR(14), EMA(20/50 ratio), MACD signal, volume_ratio dans `_get_observation()`
7. Augmenter `window_size` de 20 à 50 bougies
8. Ajouter one-hot encoding de l'actif dans l'observation (30 dims)
9. **Attention** : tout changement d'obs_size invalide les modèles existants → nouveaux modèles nécessaires

### Étape 3 — Pipeline de validation rigoureux (1 semaine)
10. Walk-forward analysis (rolling 6M train / 1M test) dans `evaluation/`
11. Benchmark buy-and-hold dans l'évaluation
12. Corriger split multi-actifs (séparation temporelle par actif)
13. Intégrer mlflow dans `trainer.py`

### Étape 4 — Signal complémentaire (1 semaine)
14. Entraîner un LightGBM classificateur de direction (T+1) sur features OHLCV + indicateurs
15. Si accuracy > 52% : intégrer comme feature dans l'observation RL
16. Tester HMM (hmmlearn, 3 états) comme détecteur de régime en remplacement du structureur sentiment

### Étape 5 — Risk management avancé (1 semaine)
17. Trailing stop dans le Risk Manager (ex: -3% depuis le pic du trade)
18. Slippage réaliste dans `TradingEnv.step()` (fonction de vol et volume)
19. Frais de financement overnight

### Étape 6 — Live paper trading réel (2+ semaines)
20. Intégrer `ccxt` (Binance testnet ou Alpaca sandbox)
21. Flux de données websocket temps réel
22. Mécanisme de reprise après panne (sauvegarde état session)
23. Circuit breakers réseau (retry, timeout)
