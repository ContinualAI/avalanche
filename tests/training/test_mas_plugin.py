import unittest

import torch
from torch import nn

from avalanche.training.plugins.mas import MASPlugin


class _Clock:
    train_exp_counter = 1  # not the first experience -> penalty path runs


class _Strategy:
    def __init__(self, loss, model):
        self.clock = _Clock()
        self.loss = loss
        self.model = model


def _make_plugin_and_model():
    plugin = MASPlugin(lambda_reg=1.0, alpha=0.5)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    plugin.importance = {"weight": torch.tensor([1.0])}
    plugin.params = {"weight": torch.tensor([0.0])}
    return plugin, model


class TestMASPluginZeroLoss(unittest.TestCase):
    def test_before_backward_does_not_raise_on_zero_loss(self):
        """Regression test for #1676.

        When the training loss is numerically 0.0 (e.g. a near-perfect batch),
        ``strategy.loss`` is an all-zero tensor. The previous check
        ``if not strategy.loss:`` evaluated that tensor as falsy and raised
        ``ValueError("Loss is not available")`` even though the SGD update had
        correctly been performed. The check now relies on ``requires_grad``, so a
        real (grad-bearing) zero loss is accepted and the penalty is applied.
        """
        plugin, model = _make_plugin_and_model()

        # Mirror the real training-loop loss: ``self.loss = torch.zeros(1) +
        # criterion()`` -> a non-leaf 0.0 tensor that requires grad.
        base = torch.zeros(1, requires_grad=True)
        loss = base + torch.tensor(0.0)
        strategy = _Strategy(loss, model)

        # Must not raise (previously raised ValueError("Loss is not available")).
        plugin.before_backward(strategy)

        # Penalty applied: importance * (weight - params)^2 = 1 * (1 - 0)^2 = 1.
        self.assertAlmostEqual(float(strategy.loss.item()), 1.0, places=5)

    def test_before_backward_raises_when_loss_has_no_graph(self):
        """When no autograd graph was built, the loss is genuinely unavailable,
        so ValueError must still be raised.

        This uses assertRaises(ValueError) on purpose: it both verifies the
        availability check still fires (a removed/inverted/weakened check would
        not raise) and locks that the surfaced error is a ValueError, rather than
        being swallowed or replaced by a different exception.
        """
        plugin, model = _make_plugin_and_model()

        # Real BaseSGDTemplate initial loss: torch.zeros(1), requires_grad=False.
        loss = torch.zeros(1)
        strategy = _Strategy(loss, model)

        with self.assertRaises(ValueError):
            plugin.before_backward(strategy)


if __name__ == "__main__":
    unittest.main()
