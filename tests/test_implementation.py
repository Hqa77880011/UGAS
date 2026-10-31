"""Small offline/online dispatch check and analytic tail-risk/projection cases."""

import json
from types import SimpleNamespace as NS
import numpy as np
import torch
from ugas.cli import (
    prepare,
    offline,
    calibrate,
    online,
    evaluate,
    empirical_cvar,
    uncertainty_gate,
)
from ugas.environment import Microgrid, project_action, inequalities, ACTION_SCALE
from ugas.learning import Learner


def test_cvar_gate_and_projection():
    # Worst 1.5 out of 30 equiprobable costs: (29 + .5*28)/1.5.
    assert np.isclose(empirical_cvar(np.arange(30)), (29 + 0.5 * 28) / 1.5)
    assert empirical_cvar([1, 2, 3, 4], 0.5) == 3.5
    assert uncertainty_gate(0.15) == 0 and uncertainty_gate(0.55) == 0
    assert np.isclose(uncertainty_gate(0.35), 0.5)
    action, feasible = project_action([40, 15], 0.9, 0, 80, 0, 23)
    assert feasible
    np.testing.assert_allclose(action, [0, 0], atol=1e-7)
    assert np.all(
        inequalities(0.9, 0, 80, 0, 23)[0] @ action
        <= inequalities(0.9, 0, 80, 0, 23)[1] + 1e-7
    )


def test_csv_learning_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    import pandas as pd

    hourly = pd.DataFrame(
        {
            "load": np.full(96, 40.0),
            "pv": np.zeros(96),
            "buy": np.full(96, 0.15),
            "sell": np.full(96, 0.05),
            "carbon": np.full(96, 0.3),
            "temperature": np.full(96, 25.0),
        }
    )
    source = tmp_path / "hourly.csv"
    hourly.to_csv(source, index=False)
    data = tmp_path / "data"
    replay = tmp_path / "replay"
    replay.mkdir()
    prepare(
        NS(
            csv=str(source),
            output=str(data),
            offline_days=1,
            calibration_days=1,
            online_days=1,
            test_days=1,
        )
    )
    samples = []
    for split in ["offline", "calibration"]:
        d = np.load(data / f"{split}.npz")
        env = Microgrid(d["days"][0], d["forecasts"][0])
        rows = []
        for _ in range(24):
            s = env.state()
            action, ok = project_action([0, 0], env.soc, env.backlog, 40, 0, env.hour)
            ns, r, done, _ = env.step(action, action, ok)
            rows.append((s, action / ACTION_SCALE, r, ns, done))
        arrays = list(zip(*rows))
        np.savez(
            replay / f"{split}_transitions.npz",
            **dict(zip(["state", "action", "reward", "next_state", "done"], arrays)),
        )
        if split == "offline":
            samples = tuple(np.array(x)[:4] for x in arrays)
    learner = Learner(2, "cpu")
    old_actor = learner.actor.network[0].weight.detach().clone()
    old_q = learner.critics[0].network[0].weight.detach().clone()
    losses = learner.update(samples)
    assert np.isfinite(losses).all()
    assert not torch.equal(
        old_actor, learner.actor.network[0].weight
    ) and not torch.equal(old_q, learner.critics[0].network[0].weight)
    for model in [learner.actor, learner.critics[0]]:
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
    learner.save(tmp_path / "roundtrip.pt")
    restored, _ = Learner.load(tmp_path / "roundtrip.pt", "cpu")
    np.testing.assert_allclose(
        restored.action(samples[0][0])[0],
        learner.action(samples[0][0])[0],
        rtol=1e-6,
        atol=1e-7,
    )
    offline(
        NS(
            seed=42,
            replay=str(replay),
            critics=2,
            device="cpu",
            conservative=1.0,
            constraint_penalty=0.0,
            updates=2,
            batch_size=4,
            output=str(tmp_path / "offline"),
        )
    )
    calibrate(
        NS(
            checkpoint=str(tmp_path / "offline/model.pt"),
            device="cpu",
            replay=str(replay),
            output=str(tmp_path / "calibrate"),
        )
    )
    args = NS(
        seed=42,
        device="cpu",
        checkpoint=str(tmp_path / "calibrate/model.pt"),
        replay=str(replay),
        data=str(data),
        output=str(tmp_path / "online"),
        interactions=2,
        batch_size=4,
        policy="learned",
        low=0.15,
        high=0.55,
        exploration="band",
        no_projection=False,
    )
    online(args)
    args.checkpoint = str(tmp_path / "online/step_2.pt")
    args.output = str(tmp_path / "eval")
    args.ood = False
    args.exploratory = False
    evaluate(args)
    metrics = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert metrics["system_violation_rate"] == 0
    assert len(json.loads((tmp_path / "eval/steps.json").read_text())) == 24
