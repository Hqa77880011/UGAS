"""Distinct SAC, IQL and Cal-QL updates for the microgrid interface."""

import math
import numpy as np
import torch
from torch.nn import functional as F
from .learning import Learner, mlp


def returns_to_go(reward, done, discount=0.99):
    """Discount complete episodes; unfinished suffixes have no MC lower bound."""
    result = np.full(len(reward), -np.inf, dtype=np.float32)
    value, complete = 0.0, False
    for i in range(len(reward) - 1, -1, -1):
        if done[i]:
            value, complete = 0.0, True
        if complete:
            value = float(reward[i]) + discount * value
            result[i] = value
    return result


def expectile_loss(residual, expectile=0.8):
    """Asymmetric value regression, with more weight on positive Q-V residuals."""
    return (
        torch.where(residual > 0, expectile, 1 - expectile) * residual.square()
    ).mean()


def logged_log_prob(actor, state, action):
    mean, log_std = actor.network(state).chunk(2, -1)
    bounded = action.clamp(-1 + 1e-6, 1 - 1e-6)
    raw = torch.atanh(bounded)
    dist = torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())
    return (dist.log_prob(raw) - torch.log1p(-bounded.square() + 1e-6)).sum(-1)


class SAC(Learner):
    """Twin-critic entropy-regularized SAC, without conservative regularization."""

    algorithm = "sac"

    def __init__(self, device="cpu"):
        super().__init__(2, device, conservative=0)


class IQL(Learner):
    """In-sample expectile critic with advantage-weighted behavior regression."""

    algorithm = "iql"

    def __init__(self, device="cpu", expectile=0.8, temperature=3.0):
        super().__init__(2, device, conservative=0)
        self.expectile, self.temperature = expectile, temperature
        self.value = mlp(106, 1).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

    def update(self, batch):
        s, a, r, ns, done = [
            torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for v in batch[:5]
        ]
        with torch.no_grad():
            logged_q = torch.stack([q(s, a) for q in self.targets]).min(0).values
        vloss = expectile_loss(logged_q - self.value(s)[:, 0], self.expectile)
        self.value_optimizer.zero_grad()
        vloss.backward()
        self.value_optimizer.step()
        with torch.no_grad():
            target = r + 0.99 * (1 - done) * self.value(ns)[:, 0]
            weight = (
                ((logged_q - self.value(s)[:, 0]) * self.temperature)
                .exp()
                .clamp_max(100)
            )
        qloss = sum(F.mse_loss(q(s, a), target) for q in self.critics)
        self.critic_optimizer.zero_grad()
        qloss.backward()
        self.critic_optimizer.step()
        aloss = -(weight * logged_log_prob(self.actor, s, a)).mean()
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()
        with torch.no_grad():
            for target_q, q in zip(self.targets, self.critics):
                for tp, p in zip(target_q.parameters(), q.parameters()):
                    tp.lerp_(p, 0.005)
        self.last_value_loss = float(vloss.detach())
        return float(qloss.detach()), float(aloss.detach())


class CalQL(Learner):
    """CQL importance-sampled log partition, calibrated by behavior MC returns."""

    algorithm = "calql"

    def __init__(self, device="cpu", samples=10):
        super().__init__(2, device, conservative=1)
        self.samples = samples

    @staticmethod
    def conservative_gap(
        random_q, policy_q, next_q, policy_logp, next_logp, logged_q, mc
    ):
        # Calibrate policy proposals before importance-density subtraction.
        floor = mc[:, None]
        calibrated = torch.maximum(policy_q, floor)
        calibrated_next = torch.maximum(next_q, floor)
        candidates = torch.cat(
            [
                random_q + 2 * math.log(2),
                calibrated - policy_logp,
                calibrated_next - next_logp,
            ],
            1,
        )
        return (torch.logsumexp(candidates, 1) - logged_q).mean()

    def update(self, batch):
        if len(batch) != 6:
            raise ValueError(
                "Cal-QL requires per-transition complete-episode returns-to-go"
            )
        s, a, r, ns, done, mc = [
            torch.as_tensor(v, dtype=torch.float32, device=self.device) for v in batch
        ]
        n = len(s)
        repeated = s[:, None].expand(-1, self.samples, -1).reshape(-1, 106)
        next_repeated = ns[:, None].expand(-1, self.samples, -1).reshape(-1, 106)
        with torch.no_grad():
            na, nlp = self.actor(ns)
            target = r + 0.99 * (1 - done) * (
                torch.stack([q(ns, na) for q in self.targets]).min(0).values
                - self.entropy * nlp
            )
            pa, plp = self.actor(repeated)
            npa, nplp = self.actor(next_repeated)
            random = torch.empty(n * self.samples, 2, device=self.device).uniform_(
                -1, 1
            )
        qloss = s.sum() * 0
        gaps = []
        for q in self.critics:
            logged = q(s, a)
            gap = self.conservative_gap(
                q(repeated, random).reshape(n, -1),
                q(repeated, pa).reshape(n, -1),
                q(repeated, npa).reshape(n, -1),
                plp.reshape(n, -1),
                nplp.reshape(n, -1),
                logged,
                mc,
            )
            qloss = qloss + 0.5 * F.mse_loss(logged, target) + self.conservative * gap
            gaps.append(float(gap.detach()))
        self.critic_optimizer.zero_grad()
        qloss.backward()
        self.critic_optimizer.step()
        for p in self.critics.parameters():
            p.requires_grad_(False)
        proposal, lp = self.actor(s)
        aloss = (
            self.entropy * lp
            - torch.stack([q(s, proposal) for q in self.critics]).min(0).values
        ).mean()
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()
        for p in self.critics.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            for tq, q in zip(self.targets, self.critics):
                for tp, p in zip(tq.parameters(), q.parameters()):
                    tp.lerp_(p, 0.005)
        self.last_conservative_gap = float(np.mean(gaps))
        return float(qloss.detach()), float(aloss.detach())


def make_learner(
    algorithm, device="cpu", critics=5, conservative=1.0, constraint_penalty=0
):
    """Construct the selected mathematical update, retaining the UGAS learner."""
    if algorithm == "ugas":
        return Learner(
            critics, device, conservative, constraint_penalty=constraint_penalty
        )
    if constraint_penalty:
        raise ValueError("Soft constraint penalty is a UGAS/common-learner control")
    learner = {"sac": SAC, "iql": IQL, "calql": CalQL}[algorithm](device)
    if algorithm == "calql":
        learner.conservative = conservative
    return learner
