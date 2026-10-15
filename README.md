# UGAS

Implementation of **Safe Offline-to-Online Reinforcement Learning with Uncertainty-Gated Action Shielding** for hourly microgrid scheduling. Conservative ensemble value learning supplies a calibrated uncertainty band; a physical action projection enforces the current battery, backlog, recoverability and grid constraints.

## Install and prepare hourly data

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
ugas prepare --csv data/hourly.csv --output data/prepared
ugas collect --data data/prepared --output data/replay
ugas offline --replay data/replay --output runs/offline
ugas calibrate --checkpoint runs/offline/model.pt --replay data/replay --output runs/calibrate
ugas online --checkpoint runs/calibrate/model.pt --output runs/online
ugas evaluate --checkpoint runs/online/step_1440.pt --output runs/final
```

Export CityLearn building, PV, pricing, carbon and weather series into a chronological CSV with one row per hour:

```text
load,pv,buy,sell,carbon,temperature
```

Use actual data rows beneath this header. Units are kW for base load/PV, USD/kWh for prices, kgCO₂/kWh for carbon and °C for temperature. The default partition requires 365 complete days: first 200 offline, next 45 calibration, next 60 online development and final 60 held out. Alternate declared lengths are available through `--offline-days`, `--calibration-days`, `--online-days` and `--test-days`; the CSV must match their sum exactly. Assign partitions before generating trajectories.

Optional columns `forecast_load_0..23`, `forecast_pv_0..23`, `forecast_buy_0..23` and `forecast_temperature_0..23` provide forecasts available at each decision. If absent, forecasts use a time-of-day profile fitted only from the offline partition. Horizon zero uses the current observation. Test outcomes are never substituted for day-ahead predictions.

## Physical model and data collection

The battery has 100 kWh capacity, SOC limits 0.1–0.9, efficiency 0.95 and ±40 kW power. Flexible load has ±15 kW power and 0–45 kWh backlog, with zero terminal backlog. Grid power is limited to −60..160 kW. Positive battery power charges; positive flexible power serves deferred load. Every day starts at SOC 0.5 and zero backlog.

The cost contains import/export energy, battery throughput at 0.02 USD/kWh, carbon at 0.025 USD/kgCO₂, current backlog at 0.05 USD/kWh, and a terminal daily demand charge of 0.12 USD/kW. A demand-peak state is included so the terminal demand charge remains observable to the policy.

`collect` runs rule, forecast MPC and noisy MPC on the offline days. All logged actions pass the physical projection; noisy MPC adds Gaussian noise with 8/3 kW standard deviations before projection. The calibration-state collection uses the rule controller. Forecast MPC uses the same physical equations and forecasts with a constrained 24-step SLSQP solve; unsuccessful solves fall back to the rule action. The environment is a standalone single-bus implementation of the manuscript equations using exported CityLearn data, not a standards-complete electrical simulator.

## Offline learning, calibration and adaptation

The actor and critics use two 256-unit ReLU layers. The default ensemble has five independently initialized critics, bootstrap minibatches, minimum target value, entropy regularization and the manuscript's policy-sampled conservative difference. `--updates` defaults to 500000, `--batch-size` to 256, `--critics` to 5 and `--conservative` to 1. Adam uses 0.0003, discount is 0.99 and target averaging is 0.005. Actions are normalized by `[40,15]` inside the learner.

Calibration computes the 95th percentile of ensemble disagreement on held-out calibration states. Online actions use the deterministic actor mean plus Gaussian 5/2 kW noise scaled by the uncertainty band. The default band is 0.15–0.55; exploration is zero outside it. The projection then finds the closest feasible action with weights `[1,2]`.

The two-dimensional QP is solved by enumerating feasible active-boundary projections and vertices. If its feasible polygon is empty, an explicitly flagged least-violation emergency action is used. This is recorded as an infeasible-set event and system violation. The projection includes terminal backlog recoverability; it cannot hide infeasible demand by carrying backlog beyond the day.

Online learning performs one update per interaction from 128 offline and 128 online samples by default, with replacement when the early online buffer is small. `--interactions` accepts 0, 240, 720 or 1440 and other budgets inside the online partition. Checkpoints are saved at 0/240/720/1440 and the requested final step. For an independent budget comparison, start each run from the same calibrated checkpoint and seed.

## Baselines and ablations

```bash
ugas evaluate --policy rule --output runs/rule
ugas evaluate --policy mpc --output runs/mpc
ugas online --exploration fixed --output runs/fixed_noise
ugas online --no-projection --output runs/no_projection
ugas offline --conservative 0 --output runs/no_conservative
ugas offline --critics 1 --output runs/single_critic
ugas offline --constraint-penalty 10 --output runs/penalty
ugas evaluate --checkpoint runs/online/step_0.pt --output runs/budget0
ugas evaluate --checkpoint runs/online/step_240.pt --output runs/budget240
ugas evaluate --checkpoint runs/online/step_720.pt --output runs/budget720
ugas evaluate --checkpoint runs/online/step_1440.pt --ood --output runs/extreme
```

Calibrate newly fitted offline ablations before online adaptation, then evaluate their resulting checkpoints. Fixed-noise is the common-learner gate control. `--constraint-penalty` adds differentiable normalized physical violations to the actor objective; combining it with no projection provides a soft-constraint comparison. The supplied learner is described by its actual objective rather than relabeled as Cal-QL, IQL or RLPD.

Evaluation normally uses deterministic means. Pass `--exploratory --exploration band` or `fixed` to evaluate the corresponding noisy execution policy. The OOD condition shifts temperature by +5°C, reduces PV to 55% during hours 10–15, multiplies import prices by 2.5 during hours 17–20, and adds load/PV forecast errors at 10%/15%. Exogenous and action random streams are separate so paired policies receive identical disturbances.

## Metrics and outputs

Evaluation writes `metrics.json`, `days.json`, complete `steps.json` traces and `daily_cost.png`. Metrics include mean daily cost, CVaR95, worst-day cost, raw-action infeasibility, executed violation rate, infeasible-set rate, SOC/grid/backlog/recoverability/power violations, terminal unrecovered backlog, per-type violation magnitudes, normalized total magnitude, policy/uncertainty/QP timings and checked projection failure rate. Infeasible sets and numerical projection failures are separate fields. Per-type magnitudes retain their physical units; the combined magnitude is explicitly normalized.

Summarize matched seed and budget reports:

```bash
ugas summarize --reports seed1/budget0/metrics.json seed1/budget240/metrics.json seed2/budget0/metrics.json seed2/budget240/metrics.json --output summary.json
```

The summary reports seed-bootstrap cost intervals, worst-seed mean cost, worst day, target-attainment rate and each seed's first evaluated budget meeting both 42.5 USD mean cost and 0.5% executed violations. A target never reached is `null`. The first-attainment result is resolved at the supplied budget grid. Use independent seeds and keep standard/OOD and execution-policy conditions separate; duplicate seed/condition/budget reports are rejected.

This implementation optimizes expected discounted operating cost. CVaR is an evaluation endpoint. No field voltage/frequency or thermal battery dynamics are inferred from the single-bus model.
