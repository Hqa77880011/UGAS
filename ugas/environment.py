import numpy as np
from scipy.optimize import minimize

ACTION_SCALE = np.array([40, 15], np.float32)


def inequalities(soc, backlog, load, pv, hour):
    lower = max(-40, -(soc - 0.1) * 100 * 0.95)
    upper = min(40, (0.9 - soc) * 100 / 0.95)
    flex_lower = max(-15, backlog - 45, backlog - 15 * (23 - hour))
    flex_upper = min(15, backlog)
    matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]], float)
    bound = np.array([upper, -lower, flex_upper, -flex_lower, 160 - load + pv, 60 + load - pv], float)
    return matrix, bound


def project_action(raw, soc, backlog, load, pv, hour):
    matrix, bound = inequalities(soc, backlog, load, pv, hour)
    candidates = []
    raw = np.asarray(raw, float)
    if np.all(matrix @ raw <= bound + 1e-8):
        return raw, True
    inverse_weight = np.array([1, 0.5])
    for normal, limit in zip(matrix, bound):
        point = raw - inverse_weight * normal * ((normal @ raw - limit) / np.sum(normal ** 2 * inverse_weight))
        if np.all(matrix @ point <= bound + 1e-7):
            candidates.append(point)
    for i in range(len(bound)):
        for j in range(i):
            pair = matrix[[i, j]]
            if abs(np.linalg.det(pair)) < 1e-10:
                continue
            point = np.linalg.solve(pair, bound[[i, j]])
            if np.all(matrix @ point <= bound + 1e-7):
                candidates.append(point)
    if candidates:
        costs = [np.sum((point - raw) ** 2 * [1, 2]) for point in candidates]
        return candidates[int(np.argmin(costs))], True
    result = minimize(lambda action: float(np.sum(np.maximum(matrix @ action - bound, 0) ** 2) + 1e-6 * np.sum((action - raw) ** 2)), np.clip(raw, -ACTION_SCALE, ACTION_SCALE), bounds=list(zip(-ACTION_SCALE, ACTION_SCALE)), method="L-BFGS-B")
    return result.x, False


class Microgrid:
    def __init__(self, day, forecasts):
        self.day, self.forecasts = np.asarray(day), np.asarray(forecasts)
        self.hour, self.soc, self.backlog, self.peak = 0, 0.5, 0.0, 0.0

    def state(self):
        current = self.day[self.hour]
        forecast_scale = np.array([160, 100, 1, 40])
        state = np.r_[self.soc, self.backlog / 45, current[0] / 160, current[1] / 100,
                      current[2], current[4], current[5] / 40, self.peak / 160,
                      (self.forecasts[self.hour] / forecast_scale).ravel(),
                      np.sin(2 * np.pi * self.hour / 24), np.cos(2 * np.pi * self.hour / 24)]
        return state.astype(np.float32)

    def step(self, action, raw=None, feasible=True):
        hour = self.hour
        load, pv, buy, sell, carbon, _ = self.day[hour]
        battery, flexible = np.asarray(action)
        prior_backlog = self.backlog
        raw = action if raw is None else raw
        matrix, bound = inequalities(self.soc, self.backlog, load, pv, hour)
        raw_infeasible = bool(np.any(matrix @ raw > bound + 1e-6))
        grid = load + flexible - pv + battery
        self.soc += (0.95 * battery if battery >= 0 else battery / 0.95) / 100
        self.backlog -= flexible
        self.peak = max(self.peak, grid)
        cost = buy * max(grid, 0) - sell * max(-grid, 0) + 0.02 * abs(battery)
        cost += 0.025 * carbon * max(grid, 0) + 0.05 * max(prior_backlog, 0)
        if hour == 23:
            cost += 0.12 * self.peak
        magnitudes = {"soc": max(0.1 - self.soc, self.soc - 0.9, 0),
                      "grid": max(-60 - grid, grid - 160, 0),
                      "backlog": max(-self.backlog, self.backlog - 45, 0),
                      "terminal_backlog": abs(self.backlog) if hour == 23 else 0,
                      "recoverability": max(self.backlog - 15 * (23 - hour), 0),
                      "battery_power": max(abs(battery) - 40, 0), "flexible_power": max(abs(flexible) - 15, 0)}
        violation = any(value > 1e-5 for value in magnitudes.values()) or not feasible
        self.hour += 1
        done = self.hour == 24
        info = {"cost": float(cost), "raw_infeasible": raw_infeasible, "system_violation": violation,
                "infeasible_set": not feasible, "soc": float(self.soc), "grid": float(grid),
                "backlog": float(self.backlog), "magnitude": magnitudes}
        return np.zeros(106, np.float32) if done else self.state(), -float(cost), done, info


def rule_action(env):
    load, pv, price, _, _, _ = env.day[env.hour]
    battery = -20 if price > 0.2 else 15 if pv > load else 0
    flexible = min(env.backlog, 15) if price < 0.15 or env.hour == 23 else -5
    return project_action([battery, flexible], env.soc, env.backlog, load, pv, env.hour)[0]


def mpc_action(env):
    horizon = 24 - env.hour
    forecast = env.forecasts[env.hour, :horizon]
    sell, carbon = env.day[env.hour, 3:5]

    def rollout(vector):
        actions = vector.reshape(horizon, 2)
        soc, backlog, peak = env.soc, env.backlog, env.peak
        cost, slack = 0.0, []
        for i, (battery, flexible) in enumerate(actions):
            load, pv, buy, _ = forecast[i]
            matrix, bound = inequalities(soc, backlog, load, pv, env.hour + i)
            slack.extend(bound - matrix @ actions[i])
            grid = load + flexible - pv + battery
            soc += (0.95 * battery if battery >= 0 else battery / 0.95) / 100
            peak = max(peak, grid)
            cost += buy * max(grid, 0) - sell * max(-grid, 0) + 0.02 * abs(battery) + 0.025 * carbon * max(grid, 0) + 0.05 * backlog
            backlog -= flexible
        return cost + 0.12 * peak, np.array(slack)

    initial = np.zeros((horizon, 2))
    initial[:, 1] = env.backlog / horizon
    fitted = minimize(lambda v: rollout(v)[0], initial.ravel(), method="SLSQP",
                      bounds=[(-40, 40), (-15, 15)] * horizon,
                      constraints=[{"type": "ineq", "fun": lambda v: rollout(v)[1]}], options={"maxiter": 150, "ftol": 1e-7})
    raw = fitted.x[:2] if fitted.success else rule_action(env)
    return raw, bool(fitted.success)
