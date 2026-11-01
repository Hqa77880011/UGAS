"""Algorithm identities, MC boundaries, gradients and serialized baseline dispatch."""

import math
import numpy as np
import torch
from ugas.baselines import CalQL, IQL, SAC, expectile_loss, returns_to_go
from ugas.learning import Learner


def test_mc_expectile_and_calibration_direction():
    np.testing.assert_allclose(
        returns_to_go([1, 2, 3, 4], [0, 1, 0, 0], 0.5)[:2], [2, 2]
    )
    assert np.isneginf(returns_to_go([1, 2, 3, 4], [0, 1, 0, 0], 0.5)[2:]).all()
    assert torch.isclose(
        expectile_loss(torch.tensor([-1.0, 1.0]), 0.8), torch.tensor(0.5)
    )
    q = torch.full((1, 1), -3.0, requires_grad=True)
    lp = torch.zeros_like(q)
    gap = CalQL.conservative_gap(
        q, q, q, lp, lp, torch.tensor([-2.0]), torch.tensor([-1.0])
    )
    expected = math.log(4 * math.exp(-3) + 2 * math.exp(-1)) + 2
    assert math.isclose(float(gap.detach()), expected, rel_tol=1e-6)
    gap.backward()
    # The policy terms below the MC floor have zero gradient; uniform term remains.
    assert 0 < float(q.grad) < 1


def test_distinct_updates_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    batch = (
        rng.normal(size=(6, 106)).astype("float32"),
        np.tanh(rng.normal(size=(6, 2))).astype("float32"),
        -np.arange(1, 7, dtype="float32"),
        rng.normal(size=(6, 106)).astype("float32"),
        np.array([0, 1, 0, 1, 0, 1], dtype="float32"),
    )
    updates = []
    for cls in [SAC, IQL, CalQL]:
        torch.manual_seed(17)
        model = cls(expectile=0.75, temperature=1.2) if cls is IQL else cls()
        old = model.actor.network[0].weight.detach().clone()
        extra = (returns_to_go(batch[2], batch[4]),) if cls is CalQL else ()
        torch.manual_seed(9)
        losses = model.update(batch + extra)
        assert np.isfinite(losses).all() and not torch.equal(
            old, model.actor.network[0].weight
        )
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.actor.parameters()
        )
        if cls is IQL:
            assert all(p.grad is not None for p in model.value.parameters())
        updates.append(model.actor.network[0].weight.detach().clone())
        path = tmp_path / f"{model.algorithm}.pt"
        model.save(path)
        restored, _ = Learner.load(path, "cpu")
        assert type(restored) is cls
        if cls is IQL:
            assert restored.expectile == 0.75 and restored.temperature == 1.2
        np.testing.assert_allclose(
            model.action(batch[0][0])[0], restored.action(batch[0][0])[0], atol=1e-7
        )
    assert not torch.equal(updates[0], updates[1]) and not torch.equal(
        updates[1], updates[2]
    )


def test_sac_stochastic_exploration_uses_actor_not_external_gate():
    from types import SimpleNamespace as NS
    from ugas.cli import execute
    from ugas.environment import Microgrid

    torch.set_num_threads(1)
    torch.manual_seed(17)
    learner = SAC()
    day = np.tile(
        np.array([40.0, 0.0, 0.15, 0.05, 0.3, 25.0], dtype="float32"), (24, 1)
    )
    forecasts = np.zeros((24, 24, 4), dtype="float32")

    class NoExternalNoise:
        def normal(self, *args, **kwargs):
            raise AssertionError("SAC must sample its actor, not UGAS gate noise")

    args = NS(
        policy="learned", low=0.15, high=0.55, exploration="fixed", no_projection=False
    )
    first = execute(
        learner, {"q95": 1.0}, Microgrid(day, forecasts), args, NoExternalNoise(), True
    )[3]
    second = execute(
        learner, {"q95": 1.0}, Microgrid(day, forecasts), args, NoExternalNoise(), True
    )[3]
    assert not np.allclose(first["raw_action"], second["raw_action"])


def test_calql_max_twin_minimum_and_selected_entropy():
    first = torch.tensor([[10., 4.], [3., 0.]])
    second = torch.tensor([[0., 5.], [2., 8.]])
    logp = torch.tensor([[-1., -3.], [-4., -2.]])
    torch.testing.assert_close(CalQL.backup_values(first, second, logp, .5, True, False), torch.tensor([4., 2.]))
    torch.testing.assert_close(CalQL.backup_values(first, second, logp, .5, True, True), torch.tensor([5.5, 4.]))
    torch.testing.assert_close(CalQL.backup_values(first, second, logp, .5, False, True), torch.tensor([.5, 4.]))


def test_calql_entropy_checkpoint_and_legacy_identity(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(2)
    batch = (rng.normal(size=(4, 106)).astype('float32'), np.zeros((4, 2), 'float32'),
             -np.ones(4, 'float32'), rng.normal(size=(4, 106)).astype('float32'),
             np.ones(4, 'float32'), -np.ones(4, 'float32'))
    learner = CalQL(samples=3)
    before = learner.log_alpha.detach().clone()
    learner.update(batch)
    assert not torch.equal(before, learner.log_alpha) and np.isfinite(learner.entropy)
    checkpoint = tmp_path / 'calql.pt'
    learner.save(checkpoint)
    restored, saved = Learner.load(checkpoint, 'cpu')
    assert restored.automatic_entropy and restored.max_target_backup and not restored.backup_entropy
    torch.testing.assert_close(restored.log_alpha, learner.log_alpha)
    assert restored.entropy == learner.entropy and restored.samples == 3
    assert restored.alpha_optimizer.state_dict()['state'][0]['step'] == 1
    fixed = CalQL(automatic_entropy=False, max_target_backup=False, backup_entropy=True, entropy=.1)
    fixed.update(batch)
    assert fixed.entropy == .1
    saved['algorithm_settings'] = dict(samples=3)
    saved['log_alpha'] = saved['alpha_optimizer'] = None
    saved['entropy'] = .1
    torch.save(saved, checkpoint)
    legacy, _ = Learner.load(checkpoint, 'cpu')
    assert not legacy.automatic_entropy and not legacy.max_target_backup and legacy.backup_entropy
