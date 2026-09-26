from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.distributions import Categorical, DiagGaussian, SquashedGaussian, StateIndependentStd
from core.torch_utils import explained_variance, hard_update, soft_update
from nets import make_net
from nets.mlp import NoisyLinear


def test_categorical_logprob_and_mask():
    logits = torch.randn(8, 4)
    dist = Categorical(logits)
    actions = dist.sample()
    assert actions.shape == (8,)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(8), atol=1e-5)
    assert dist.log_prob(actions).shape == (8,)

    mask = torch.zeros(8, 4, dtype=torch.bool)
    mask[:, 1] = True
    masked = Categorical(logits, mask)
    assert torch.all(masked.sample() == 1)
    assert float(masked.entropy().max()) == pytest.approx(0.0, abs=1e-4)


def test_diag_gaussian_matches_torch():
    mean = torch.randn(16, 3)
    log_std = torch.full((16, 3), -0.5)
    ours = DiagGaussian(mean, log_std)
    ref = torch.distributions.Normal(mean, log_std.exp())
    x = ours.sample()
    assert torch.allclose(ours.log_prob(x), ref.log_prob(x).sum(-1), atol=1e-5)
    assert torch.allclose(ours.entropy(), ref.entropy().sum(-1), atol=1e-5)


def test_squashed_gaussian_bounds_and_logprob_consistency():
    mean = torch.randn(32, 2)
    log_std = torch.full((32, 2), -1.0)
    dist = SquashedGaussian(mean, log_std)
    action, log_prob = dist.sample_with_logprob()
    assert float(action.abs().max()) < 1.0
    assert torch.allclose(log_prob, dist.log_prob(action), atol=1e-3)


def test_state_independent_std_is_learnable():
    head = StateIndependentStd(3)
    dist = head(torch.zeros(5, 3))
    dist.log_prob(dist.sample()).sum().backward()
    assert head.log_std.grad is not None


def test_mlp_and_target_updates():
    net = make_net("mlp", 4, 2, hidden=(16, 16), layer_norm=True)
    target = make_net("mlp", 4, 2, hidden=(16, 16), layer_norm=True)
    hard_update(target, net)
    for a, b in zip(net.parameters(), target.parameters()):
        assert torch.allclose(a, b)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    soft_update(target, net, 0.5)
    assert net(torch.zeros(3, 4)).shape == (3, 2)


def test_dueling_and_noisy_layers():
    head = make_net("dueling", 8, 5)
    assert head(torch.randn(4, 8)).shape == (4, 5)
    noisy = NoisyLinear(8, 5)
    noisy.reset_noise()
    noisy.train()
    a = noisy(torch.ones(1, 8))
    noisy.eval()
    b = noisy(torch.ones(1, 8))
    assert not torch.allclose(a, b)


def test_cnn_infers_flat_dim():
    net = make_net("nature_cnn", (4, 84, 84), out_dim=64)
    assert net(torch.zeros(2, 4, 84, 84)).shape == (2, 64)


def test_explained_variance():
    target = torch.randn(100)
    assert explained_variance(target.clone(), target) == pytest.approx(1.0, abs=1e-5)
    assert explained_variance(torch.zeros(100), target) < 0.5


def test_categorical_sampling_follows_probs():
    logits = torch.log(torch.tensor([[0.1, 0.6, 0.3]])).repeat(20000, 1)
    counts = torch.bincount(Categorical(logits).sample(), minlength=3).float() / 20000
    assert torch.allclose(counts, torch.tensor([0.1, 0.6, 0.3]), atol=0.02)


def test_categorical_is_shift_invariant():
    logits = torch.randn(64, 5)
    a = Categorical(logits)
    b = Categorical(logits + 7.0)
    actions = torch.arange(5).repeat(64, 1)[:, :1].squeeze(-1)
    assert torch.allclose(a.log_prob(actions), b.log_prob(actions), atol=1e-5)
    assert torch.allclose(a.entropy(), b.entropy(), atol=1e-5)
