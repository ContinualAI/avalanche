import unittest

import torch
import torch.nn as nn

from avalanche.training.losses import ICaRLLossPlugin, MaskedCrossEntropy


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


class TestMaskedCrossEntropy(unittest.TestCase):
    def test_loss(self):
        cross_entropy = nn.CrossEntropyLoss()

        criterion = MaskedCrossEntropy(mask="new")
        criterion._adaptation([1, 2, 3, 4])
        criterion._adaptation([5, 6, 7])

        mb_y = torch.tensor([5, 5, 6, 7, 6])

        new_pred = torch.rand(5, 8)
        new_pred_new = new_pred[:, criterion.current_mask(new_pred.shape[1])]

        loss1 = criterion(new_pred, mb_y)
        loss2 = cross_entropy(new_pred_new, mb_y - 5)

        criterion.mask = "seen"
        loss3 = criterion(new_pred, mb_y)

        self.assertAlmostEqual(float(loss1), float(loss2), places=5)
        self.assertNotAlmostEqual(float(loss1), float(loss3), places=5)


if __name__ == "__main__":
    unittest.main()
