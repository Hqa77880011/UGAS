import copy
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, outputs))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = mlp(106, 4)

    def forward(self, state, deterministic=False):
        mean, log_std = self.network(state).chunk(2, -1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        action = raw.tanh()
        log_probability = (distribution.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1)
        return action, log_probability


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = mlp(108, 1)

    def forward(self, state, action):
        return self.network(torch.cat([state, action], -1)).squeeze(-1)


class Learner:
    def __init__(self, critics, device, conservative=1, entropy=0.1, constraint_penalty=0):
        self.device, self.conservative, self.entropy = device, conservative, entropy
        self.constraint_penalty = constraint_penalty
        self.actor = Actor().to(device)
        self.critics = nn.ModuleList([Critic() for _ in range(critics)]).to(device)
        self.targets = copy.deepcopy(self.critics).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)

    def update(self, batch):
        state, action, reward, next_state, done = [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch]
        with torch.no_grad():
            next_action, logp = self.actor(next_state)
            target = reward + 0.99 * (1 - done) * (torch.stack([q(next_state, next_action) for q in self.targets]).min(0).values - self.entropy * logp)
            proposed, _ = self.actor(state)
        loss = state.sum() * 0
        for critic in self.critics:
            index = torch.randint(len(state), (len(state),), device=self.device)
            prediction = critic(state[index], action[index])
            loss = loss + 0.5 * F.mse_loss(prediction, target[index])
            loss = loss + self.conservative * (critic(state[index], proposed[index]).mean() - prediction.mean())
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        for p in self.critics.parameters():
            p.requires_grad_(False)
        proposal, logp = self.actor(state)
        q = torch.stack([critic(state, proposal) for critic in self.critics]).min(0).values
        actor_loss = (self.entropy * logp - q).mean()
        if self.constraint_penalty:
            battery, flexible = proposal[:, 0] * 40, proposal[:, 1] * 15
            soc = state[:, 0] + torch.where(battery >= 0, 0.95 * battery, battery / 0.95) / 100
            backlog = state[:, 1] * 45 - flexible
            grid = state[:, 2] * 160 + flexible - state[:, 3] * 100 + battery
            hour = torch.remainder(torch.round(torch.atan2(state[:, -2], state[:, -1]) * 24 / (2 * np.pi)), 24)
            penalty = F.relu(0.1 - soc) + F.relu(soc - 0.9)
            penalty = penalty + (F.relu(-backlog) + F.relu(backlog - 45) + F.relu(backlog - 15 * (23 - hour))) / 45
            penalty = penalty + (F.relu(-60 - grid) + F.relu(grid - 160)) / 220
            actor_loss = actor_loss + self.constraint_penalty * penalty.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.critics.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            for target_network, critic in zip(self.targets, self.critics):
                for target_parameter, parameter in zip(target_network.parameters(), critic.parameters()):
                    target_parameter.lerp_(parameter, 0.005)
        return float(loss.detach()), float(actor_loss.detach())

    def save(self, path, q95=1, interactions=0):
        torch.save({"actor": self.actor.state_dict(), "critics": self.critics.state_dict(), "targets": self.targets.state_dict(),
                    "count": len(self.critics), "conservative": self.conservative, "entropy": self.entropy,
                    "constraint_penalty": self.constraint_penalty, "q95": q95, "interactions": interactions}, path)

    @classmethod
    def load(cls, path, device):
        saved = torch.load(path, map_location=device, weights_only=True)
        learner = cls(saved["count"], device, saved["conservative"], saved["entropy"], saved["constraint_penalty"])
        learner.actor.load_state_dict(saved["actor"])
        learner.critics.load_state_dict(saved["critics"])
        learner.targets.load_state_dict(saved["targets"])
        return learner, saved

    def action(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, device=self.device)[None]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            begin = perf_counter()
            mean = self.actor(state, deterministic=True)[0]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_ms = 1000 * (perf_counter() - begin)
            begin = perf_counter()
            q = torch.stack([critic(state, mean) for critic in self.critics])
            uncertainty = float(q.std(0, unbiased=True)) if len(self.critics) > 1 else 0.0
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            uncertainty_ms = 1000 * (perf_counter() - begin)
            return mean.cpu().numpy()[0], uncertainty, inference_ms, uncertainty_ms


def draw_batch(arrays, count, rng):
    index = rng.integers(len(arrays[0]), size=count)
    return tuple(np.asarray(array)[index] for array in arrays)
