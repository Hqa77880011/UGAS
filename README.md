# UGAS

**Safe Offline-to-Online Reinforcement Learning with Uncertainty-Gated Action Shielding** for hourly microgrid scheduling. Conservative ensemble value learning supplies a calibrated uncertainty band; a physical action projection enforces the current battery, backlog, recoverability and grid constraints.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Obtain [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) records and export building, PV, pricing, carbon and weather series into a chronological CSV with one row per hour:

```text
load,pv,buy,sell,carbon,temperature
```

Use actual data rows beneath this header. Units are kW for base load/PV, USD/kWh for prices, kgCO₂/kWh for carbon and °C for temperature. The default partition requires 365 complete days: first 200 offline, next 45 calibration, next 60 online development and final 60 held out. Alternate declared lengths are available through `--offline-days`, `--calibration-days`, `--online-days` and `--test-days`; the CSV must match their sum exactly. Assign partitions before generating trajectories.

Optional columns `forecast_load_0..23`, `forecast_pv_0..23`, `forecast_buy_0..23` and `forecast_temperature_0..23` provide forecasts available at each decision. If absent, forecasts use a time-of-day profile fitted only from the offline partition. Horizon zero uses the current observation. Test outcomes are never substituted for day-ahead predictions.

```bash
ugas prepare --csv data/hourly.csv --output data/prepared
```

## Training and inference

```bash
ugas collect --data data/prepared --output data/replay --seed 42
ugas offline --data data/prepared --replay data/replay --output runs/offline --seed 42
ugas calibrate --data data/prepared --checkpoint runs/offline/model.pt --replay data/replay --output runs/calibrate --seed 42
ugas online --data data/prepared --replay data/replay --checkpoint runs/calibrate/model.pt --output runs/online --interactions 1440 --seed 42
```

## Evaluation

```bash
ugas evaluate --data data/prepared --checkpoint runs/online/step_1440.pt --output runs/final --seed 42
```

## Outputs

Prepared role/replay archives stay under `data/`. Offline/calibration folders contain models and losses; online saves interaction checkpoints including `step_1440.pt`. Evaluation writes `metrics.json`, `days.json`, `steps.json` and `daily_cost.png`.

## Modules and metrics

| Component / metric | Entry | Command / setting |
|---|---|---|
| Chronological roles and offline-only forecast profiles | `ugas.cli.prepare` | `python -m ugas.cli prepare ...` |
| Conservative critic ensemble and actor updates | `ugas/learning.py:Learner.update` | `python -m ugas.cli offline ... --critics 5 --conservative 1` |
| Fixed calibration Q95, band gate and physical QP | `ugas.cli.calibrate,uncertainty_gate,execute`; `ugas.environment.project_action` | calibration and online commands above |
| Daily mean / exact empirical CVaR95 / worst-day cost | `ugas.cli.empirical_cvar,evaluate` | evaluation command above |
| Raw infeasibility, executed violations, magnitudes, inference/ensemble/QP timing, target attainment | `ugas.cli.evaluate` | `metrics.json`, `steps.json` |
| Paired-seed aggregate/first target checkpoint | `ugas.cli.summarize` | `python -m ugas.cli summarize --reports runs/seed42/metrics.json runs/seed43/metrics.json --output summary.json` |

CVaR integrates the worst 5% of the empirical distribution, with fractional probability mass at its boundary. This avoids changing the effective tail level when the day count is not a multiple of 20. The gate is zero at/below 0.15 and at/above 0.55 and jumps immediately above 0.15, as specified. No direct CVaR training objective is added.

## Comparisons and ablations

Use identical prepared data, seeds and interaction budgets. `online`/`evaluate` support `--exploration fixed|none`, `--no-projection` and `--policy rule|mpc`; retrain with `offline --conservative 0`, `--critics 1` or `--constraint-penalty 1` for those controls. `evaluate --ood` applies a compound shift; `--exploratory` evaluates the noisy controller. Online learning uses a 50:50 offline/online minibatch and saves executed-action transitions. Optimizers start fresh at each offline/online entry.

## Checks

```bash
python -m pip install pytest
python -m pytest tests -q -p no:cacheprovider
```

## Independent learner controls

`ugas/baselines.py` implements distinct updates, sharing only the microgrid state/action interface and network widths: twin-critic SAC with no conservative term; IQL with expectile value regression (0.8), logged-action Q backup and advantage-weighted log-likelihood (inverse temperature 3, weight cap 100); Cal-QL with 10 uniform/current/next-policy samples, importance-corrected log-sum-exp conservative loss and complete-episode behavior return floors. The incomplete online suffix is assigned no MC floor, rather than a fabricated zero return. `Learner.load` restores the algorithm and IQL value state. Cal-QL now defaults to automatic entropy tuning and a ten-proposal max-target backup, with the explicit settings below. SAC retains its own fixed entropy coefficient0.1.

```bash
ugas offline --data data/prepared --replay data/replay --algorithm sac --updates 0 --output runs/sac_init --seed 42
ugas online --data data/prepared --replay data/replay --checkpoint runs/sac_init/model.pt --output runs/sac_online --offline-fraction 0 --exploration fixed --interactions 1440 --seed 42
ugas offline --data data/prepared --replay data/replay --algorithm iql --output runs/iql --seed 42
ugas offline --data data/prepared --replay data/replay --algorithm calql --output runs/calql --seed 42
ugas calibrate --data data/prepared --replay data/replay --checkpoint runs/calql/model.pt --output runs/calql_calibrated --seed 42
ugas online --data data/prepared --replay data/replay --checkpoint runs/calql_calibrated/model.pt --output runs/calql_online --exploration fixed --seed 42
```

Cal-QL with fixed noise and the same physical projection is the matched shielded control. Use identical prepared roles, action projection, budgets and exogenous seeds. Select hyperparameters using the online-development role; test data never fit MC floors or normalization.

Algorithm sources: [IQL value/critic update](https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py), [IQL actor update](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py), [Cal-QL conservative SAC](https://github.com/nakamotoo/Cal-QL/blob/main/JaxCQL/conservative_sac.py).

SAC samples its squashed-Gaussian actor during online interaction; ordinary evaluation uses its mean. IQL/Cal-QL support the configured microgrid fixed/band/no-noise interface.

## Cal-QL entropy and target-backup settings

Cal-QL follows the pinned [conservative SAC implementation](https://github.com/nakamotoo/Cal-QL/blob/ac6eafec22e8d60836573e1f488c7f626ce8a77e/JaxCQL/conservative_sac.py) and [launcher](https://github.com/nakamotoo/Cal-QL/blob/ac6eafec22e8d60836573e1f488c7f626ce8a77e/JaxCQL/conservative_sac_main.py): automatic log-alpha tuning, initial alpha 1, target entropy -2, policy/alpha Adam learning rate 1e-4, ten-action maximum backup and `backup_entropy=False`. The target maximizes the twin-critic minimum, then optionally subtracts the selected proposal entropy.

```bash
# Two-update check.
ugas offline --replay data/replay --algorithm calql --updates 2 --batch-size 4 --output runs/calql-check
# Configurable comparison: fixed alpha1 / single proposal / soft target.
ugas offline --replay data/replay --algorithm calql --updates 2 --batch-size 4 --calql-fixed-entropy --calql-single-target --calql-backup-entropy --output runs/calql-variant
python -m pytest tests/test_algorithms.py tests/test_implementation.py -q
```

Checkpoints store algorithm flags, target entropy, log-alpha and alpha optimizer state. Critic widths, reward scaling and conservative coefficient (default 1) are configurable.
