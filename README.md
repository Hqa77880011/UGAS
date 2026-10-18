# UGAS

Implementation of **Safe Offline-to-Online Reinforcement Learning with Uncertainty-Gated Action Shielding** for hourly microgrid scheduling. Conservative ensemble value learning supplies a calibrated uncertainty band; a physical action projection enforces the current battery, backlog, recoverability and grid constraints.

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

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
