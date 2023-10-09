import unittest

import torch
import torch.nn as nn

from avalanche.training.losses import ICaRLLossPlugin, NewClassesCrossEntropy


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


class TestNewClassesCrossEntropy(unittest.TestCase):
    def test_loss(self):
        cross_entropy = nn.CrossEntropyLoss()

        criterion = NewClassesCrossEntropy()
        criterion.current_classes = [5, 6, 7]

        mb_y = torch.tensor([5, 5, 6, 7, 6])

        new_pred = torch.rand(5, 8)
        new_pred_new = new_pred[:, criterion.current_classes]

        loss1 = criterion(new_pred, mb_y)
        loss2 = cross_entropy(new_pred_new, mb_y - 5)

        assert float(loss1) == float(loss2)


if __name__ == "__main__":
    unittest.main()
