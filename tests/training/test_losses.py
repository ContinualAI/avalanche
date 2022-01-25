import unittest

import torch
from avalanche.training.losses import ICaRLLossPlugin


class TestICaRLLossPlugin(unittest.TestCase):
    def test_loss(self):
        mb_y = torch.tensor([0, 1, 2, 3], dtype=torch.float)

        old_pred = torch.tensor(
            [[2, 1, 0, 0], [1, 2, 0, 0], [2, 1, 0, 0], [1, 2, 0, 0]],
            dtype=torch.float,
        )

        new_pred = torch.tensor(
            [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]],
            dtype=torch.float,
        )

        criterion = ICaRLLossPlugin()
        loss1 = criterion(new_pred, mb_y)

        criterion.old_logits = old_pred
        criterion.old_classes = [0, 1]

        loss2 = criterion(new_pred, mb_y)

        assert loss2 < loss1

        criterion = ICaRLLossPlugin()
        loss3 = criterion(new_pred, mb_y)

        assert loss3 == loss1
