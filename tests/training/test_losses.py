
import unittest

import torch

from avalanche.training.losses import ICaRLDistillationLoss


class TestICaRLLossPlugin(unittest.TestCase):

    def test_loss(self):
        mb_y = torch.tensor([0, 1, 2, 3], dtype=torch.float)

        old_pred = torch.tensor([
            [2, 1, 0, 0],
            [1, 2, 0, 0],
            [2, 1, 0, 0],
            [1, 2, 0, 0]
        ], dtype=torch.float)

        new_pred = torch.tensor([
            [2, 1, 0, 0],
            [1, 2, 0, 0],
            [0, 0, 2, 1],
            [0, 0, 1, 2]
        ], dtype=torch.float)

        criterion = ICaRLLossPlugin()
        loss1 = criterion(new_pred, mb_y)

        criterion.set_old([0, 1], old_pred)
        loss2 = criterion(new_pred, mb_y)

        assert loss2 < loss1

        criterion = ICaRLDistillationLoss()
        loss3 = criterion(new_pred, mb_y)

        assert loss3 == loss1
