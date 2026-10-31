import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .environment import ACTION_SCALE, Microgrid, inequalities, mpc_action, project_action, rule_action
from .learning import Learner, draw_batch
from .baselines import make_learner, returns_to_go


def empirical_cvar(costs, level=0.95):
    """Upper-tail empirical CVaR, including fractional mass at the tail boundary."""
    values = np.asarray(costs, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("CVaR requires a nonempty finite vector of daily costs")
    if not 0 <= level < 1:
        raise ValueError("CVaR level must lie in [0, 1)")
    descending = np.sort(values)[::-1]
    mass = len(values) * (1 - level)
    whole = int(np.floor(mass))
    tail = descending[:whole].sum()
    if whole < len(values):
        tail += (mass - whole) * descending[whole]
    return float(tail / mass)


def uncertainty_gate(q, low=0.15, high=0.55):
    """The registered discontinuous intermediate-disagreement band rule."""
    if not 0 <= low < high <= 1:
        raise ValueError("Gate thresholds must satisfy 0 <= low < high <= 1")
    return (high - q) / (high - low) if low < q <= high else 0.0


def prepare(args):
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = np.array([[float(r[k]) for k in ["load", "pv", "buy", "sell", "carbon", "temperature"]] for r in rows], np.float32)
    splits = [args.offline_days, args.calibration_days, args.online_days, args.test_days]
    if len(values) != sum(splits) * 24 or not np.isfinite(values).all():
        raise ValueError("CSV must contain the exact declared chronological daily partitions")
    profile = values[:args.offline_days * 24].reshape(-1, 24, 6).mean(0)
    forecast = np.stack([profile[(np.arange(24) + t) % 24][:, [0, 1, 2, 5]] for t in range(len(rows))])
    names = ["load", "pv", "buy", "temperature"]
    for j, name in enumerate(names):
        for h in range(24):
            column = f"forecast_{name}_{h}"
            if column in rows[0]:
                forecast[:, h, j] = [float(r[column]) for r in rows]
    forecast[:, 0] = values[:, [0, 1, 2, 5]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    offset = 0
    for split, days in zip(["offline", "calibration", "online", "test"], splits):
        end = offset + days * 24
        np.savez_compressed(out / f"{split}.npz", days=values[offset:end].reshape(days, 24, 6),
                            forecasts=forecast[offset:end].reshape(days, 24, 24, 4))
        offset = end


def collect(args):
    rng = np.random.default_rng(args.seed)
    packed = []
    for split, controllers in [("offline", ["rule", "mpc", "noisy"]), ("calibration", ["rule"])]:
        source = np.load(Path(args.data) / f"{split}.npz", allow_pickle=False)
        samples = []
        for controller in controllers:
            for day, forecast in zip(source["days"], source["forecasts"]):
                env = Microgrid(day, forecast)
                for _ in range(24):
                    state = env.state()
                    raw = rule_action(env) if controller == "rule" else mpc_action(env)[0]
                    if controller == "noisy":
                        raw = raw + rng.normal(0, [8, 3])
                    action, feasible = project_action(raw, env.soc, env.backlog, day[env.hour, 0], day[env.hour, 1], env.hour)
                    next_state, reward, done, _ = env.step(action, raw, feasible)
                    samples.append((state, action / ACTION_SCALE, reward, next_state, done))
        arrays = list(zip(*samples))
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{split}_transitions.npz", state=arrays[0], action=arrays[1], reward=arrays[2], next_state=arrays[3], done=arrays[4])
        packed.append(len(samples))


def replay(path, with_returns=False):
    source = np.load(path, allow_pickle=False)
    arrays = tuple(source[k] for k in ["state", "action", "reward", "next_state", "done"])
    return arrays + (returns_to_go(arrays[2], arrays[4]),) if with_returns else arrays


def offline(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    algorithm = getattr(args, "algorithm", "ugas")
    data = replay(Path(args.replay) / "offline_transitions.npz", algorithm == "calql")
    learner = make_learner(algorithm, args.device, args.critics, args.conservative, args.constraint_penalty)
    history = []
    for step in range(args.updates):
        critic, actor = learner.update(draw_batch(data, args.batch_size, rng))
        if (step + 1) % 1000 == 0:
            history.append({"step": step + 1, "critic_loss": critic, "actor_loss": actor})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    learner.save(out / "model.pt")
    (out / "loss.json").write_text(json.dumps(history, indent=2))


def calibrate(args):
    learner, saved = Learner.load(args.checkpoint, args.device)
    states = replay(Path(args.replay) / "calibration_transitions.npz")[0]
    disagreements = [learner.action(state)[1] for state in states]
    q95 = float(np.quantile(disagreements, 0.95))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    learner.save(out / "model.pt", q95, saved["interactions"])


def execute(learner, saved, env, args, rng, exploratory):
    begin = perf_counter()
    inference_ms, uncertainty_ms, mpc_success = None, None, None
    if args.policy in {"rule", "mpc"}:
        if args.policy == "rule":
            raw = rule_action(env)
        else:
            raw, mpc_success = mpc_action(env)
        uncertainty = 0.0
    else:
        action, uncertainty, inference_ms, uncertainty_ms = learner.action(env.state(), deterministic=not (learner.algorithm=="sac" and exploratory))
        q = min(1, uncertainty / (saved["q95"] + 1e-8))
        gate = uncertainty_gate(q, args.low, args.high)
        if args.exploration == "fixed":
            gate = 1
        if args.exploration == "none" or not exploratory:
            gate = 0
        if learner.algorithm=="sac":
            gate=0  # SAC explores through its learned squashed-Gaussian policy.
        raw = action * ACTION_SCALE if learner.algorithm=="sac" else action * ACTION_SCALE + gate * rng.normal(0, [5, 2])
    policy_seconds = perf_counter() - begin
    begin = perf_counter()
    physical, feasible = project_action(raw, env.soc, env.backlog, env.day[env.hour, 0], env.day[env.hour, 1], env.hour)
    matrix, bound = inequalities(env.soc, env.backlog, env.day[env.hour, 0], env.day[env.hour, 1], env.hour)
    solver_failure = bool(feasible and np.any(matrix @ physical > bound + 1e-6))
    projected = physical if not args.no_projection else np.asarray(raw)
    projection_seconds = perf_counter() - begin
    next_state, reward, done, info = env.step(projected, raw, feasible)
    info.update({"raw_action": np.asarray(raw).tolist(), "executed_action": projected.tolist(),
                 "uncertainty": uncertainty, "projection_correction": float(np.linalg.norm(projected - raw)),
                 "policy_and_uncertainty_ms": policy_seconds * 1000, "projection_ms": projection_seconds * 1000,
                 "policy_inference_ms": inference_ms, "uncertainty_ms": uncertainty_ms,
                 "projection_solver_failure": solver_failure, "mpc_success": mpc_success})
    return next_state, reward, done, info, projected


def online(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    learner, saved = Learner.load(args.checkpoint, args.device)
    offline_data = replay(Path(args.replay) / "offline_transitions.npz", learner.algorithm == "calql")
    source = np.load(Path(args.data) / "online.npz", allow_pickle=False)
    if args.interactions > len(source["days"]) * 24:
        raise ValueError("Online budget exceeds its locked day partition")
    samples, logs = [], []
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    learner.save(out / "step_0.pt", saved["q95"], 0)
    for step in range(args.interactions):
        if step % 24 == 0:
            day = step // 24
            env = Microgrid(source["days"][day], source["forecasts"][day])
        state = env.state()
        next_state, reward, done, info, action = execute(learner, saved, env, args, rng, True)
        samples.append((state, action / ACTION_SCALE, reward, next_state, done))
        online_data = tuple(np.array(v) for v in zip(*samples))
        if learner.algorithm == "calql":
            online_data = online_data + (returns_to_go(online_data[2], online_data[4]),)
        offline_count = int(args.batch_size * getattr(args, "offline_fraction", .5))
        off_batch = draw_batch(offline_data, offline_count, rng)
        on_batch = draw_batch(online_data, args.batch_size - offline_count, rng)
        learner.update(tuple(np.concatenate([a, b]) for a, b in zip(off_batch, on_batch)))
        logs.append({"step": step + 1, **info})
        if step + 1 in {240, 720, 1440, args.interactions}:
            learner.save(out / f"step_{step + 1}.pt", saved["q95"], step + 1)
    (out / "transitions.json").write_text(json.dumps(logs, indent=2, allow_nan=False))


def evaluate(args):
    learner, saved = Learner.load(args.checkpoint, args.device)
    source = np.load(Path(args.data) / "test.npz", allow_pickle=False)
    rng = np.random.default_rng(args.seed)
    exogenous_rng = np.random.default_rng(args.seed + 1000)
    days, all_steps = [], []
    for day_id, (original, forecasts) in enumerate(zip(source["days"], source["forecasts"])):
        day, forecast = original.copy(), forecasts.copy()
        if args.ood:
            day[:, 5] += 5
            day[10:16, 1] *= 0.55
            day[17:21, 2] *= 2.5
            forecast[:, :, 0] = np.maximum(0, forecast[:, :, 0] + exogenous_rng.normal(size=(24, 24)) * 0.1 * forecast[:, :, 0])
            forecast[:, :, 1] = np.maximum(0, forecast[:, :, 1] + exogenous_rng.normal(size=(24, 24)) * 0.15 * forecast[:, :, 1])
        env, rows = Microgrid(day, forecast), []
        for _ in range(24):
            _, _, _, info, _ = execute(learner, saved, env, args, rng, args.exploratory)
            rows.append(info)
        total_cost = sum(r["cost"] for r in rows)
        days.append({"day": day_id, "cost_usd": total_cost, "system_violation_rate": float(np.mean([r["system_violation"] for r in rows]))})
        all_steps.extend({"day": day_id, "hour": h, **r} for h, r in enumerate(rows))
    costs = np.array([d["cost_usd"] for d in days])
    violations = np.array([r["system_violation"] for r in all_steps])
    metrics = {"seed": args.seed, "interactions": saved["interactions"], "policy": args.policy,
               "exploration": args.exploration if args.exploratory else "none", "projection": not args.no_projection,
               "ood": args.ood, "mean_daily_cost_usd": float(costs.mean()),
               "cvar95_usd": empirical_cvar(costs), "worst_day_cost_usd": float(costs.max()),
               "raw_action_infeasibility": float(np.mean([r["raw_infeasible"] for r in all_steps])),
               "system_violation_rate": float(violations.mean()), "infeasible_set_rate": float(np.mean([r["infeasible_set"] for r in all_steps])),
               "policy_and_uncertainty_ms": float(np.mean([r["policy_and_uncertainty_ms"] for r in all_steps])),
               "qp_ms": float(np.mean([r["projection_ms"] for r in all_steps])),
               "qp_solver_failure_rate": float(np.mean([r["projection_solver_failure"] for r in all_steps])),
               "target_attained": bool(costs.mean() <= 42.5 and violations.mean() <= 0.005)}
    for key in ["policy_inference_ms", "uncertainty_ms"]:
        times = [r[key] for r in all_steps if r[key] is not None]
        metrics[key] = float(np.mean(times)) if times else None
    metrics["normalized_total_violation_magnitude"] = 0.0
    scales = {"soc": 0.8, "grid": 220, "backlog": 45, "terminal_backlog": 45, "recoverability": 345, "battery_power": 80, "flexible_power": 30}
    for key, scale in scales.items():
        values = np.array([r["magnitude"][key] for r in all_steps])
        metrics[key + "_violation_fraction"] = float(np.mean(values > 1e-5))
        metrics[key + "_violation_magnitude"] = float(values.sum())
        metrics["normalized_total_violation_magnitude"] += float(values.sum()) / scale
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out / "days.json").write_text(json.dumps(days, indent=2, allow_nan=False))
    (out / "steps.json").write_text(json.dumps(all_steps, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([d["day"] for d in days], costs)
    ax.set(xlabel="Held-out day", ylabel="Operating cost (USD)")
    fig.tight_layout()
    fig.savefig(out / "daily_cost.png", dpi=180)
    plt.close(fig)


def summarize(args):
    rows = [json.loads(Path(path).read_text()) for path in args.reports]
    grouped, attained = {}, {}
    rng = np.random.default_rng(args.seed)
    for row in rows:
        method = f"{row['policy']}_{row['exploration']}_projection{row['projection']}_ood{row['ood']}"
        key = method + f"_steps{row['interactions']}"
        group = grouped.setdefault(key, [])
        if any(previous["seed"] == row["seed"] for previous in group):
            raise ValueError("Duplicate seed/condition/interaction report")
        group.append(row)
        attained.setdefault(method, {}).setdefault(str(row["seed"]), [])
        if row["target_attained"]:
            attained[method][str(row["seed"])].append(row["interactions"])
    result = {}
    for key, group in grouped.items():
        values = np.array([r["mean_daily_cost_usd"] for r in group])
        resamples = rng.choice(values, (args.bootstrap, len(values)), replace=True).mean(1)
        result[key] = {"seeds": len(group), "mean_daily_cost_usd": float(values.mean()),
                       "mean_cost_ci95": np.quantile(resamples, [0.025, 0.975]).tolist(),
                       "worst_seed_mean_cost_usd": float(values.max()),
                       "worst_day_cost_usd": max(r["worst_day_cost_usd"] for r in group),
                       "target_attainment_rate": float(np.mean([r["target_attained"] for r in group]))}
    result["sample_efficiency"] = {method: {seed: min(budgets) if budgets else None for seed, budgets in by_seed.items()} for method, by_seed in attained.items()}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--output", default="summary.json")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(function=summarize)
    p = sub.add_parser("prepare")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="data/prepared")
    for key, value in [("offline-days", 200), ("calibration-days", 45), ("online-days", 60), ("test-days", 60)]:
        p.add_argument("--" + key, type=int, default=value)
    p.set_defaults(function=prepare)
    p = sub.add_parser("collect")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", default="data/replay")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(function=collect)
    for command in ["offline", "calibrate", "online", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--replay", default="data/replay")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--checkpoint", default="runs/calibrate/model.pt" if command in {"online", "evaluate"} else "runs/offline/model.pt")
        p.add_argument("--device", default="cpu")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--batch-size", type=int, default=256)
        if command == "offline":
            p.add_argument("--updates", type=int, default=500000)
            p.add_argument("--algorithm", choices=["ugas", "sac", "iql", "calql"], default="ugas")
            p.add_argument("--critics", type=int, default=5)
            p.add_argument("--conservative", type=float, default=1)
            p.add_argument("--constraint-penalty", type=float, default=0)
            p.set_defaults(function=offline)
        elif command == "calibrate":
            p.set_defaults(function=calibrate)
        else:
            p.add_argument("--policy", choices=["learned", "rule", "mpc"], default="learned")
            p.add_argument("--no-projection", action="store_true")
            p.add_argument("--exploration", choices=["band", "fixed", "none"], default="band")
            p.add_argument("--low", type=float, default=0.15)
            p.add_argument("--high", type=float, default=0.55)
            if command == "online":
                p.add_argument("--interactions", type=int, default=1440)
                p.add_argument("--offline-fraction", type=float, default=.5)
                p.set_defaults(function=online)
            else:
                p.add_argument("--ood", action="store_true")
                p.add_argument("--exploratory", action="store_true")
                p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
