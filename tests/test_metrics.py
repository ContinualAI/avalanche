""" Metrics Tests"""

import unittest
import torch
from torch import nn

from avalanche.evaluation_deprecated.metrics import MAC


class MultiTaskTests(unittest.TestCase):
    def test_ff_model(self):
        xn, hn, yn = 50, 100, 10

        model = nn.Sequential(
            nn.Linear(xn, hn),  # 5'000 mul
            nn.ReLU(),
            nn.Linear(hn, hn),  # 10'000 mul
            nn.ReLU(),
            nn.Linear(hn, yn)  # 1'000 mul
        )  # 16'000 mul
        dummy = torch.randn(32, xn)
        met = MAC()
        # self.assertEqual(16000, met.compute(model, dummy))

    def test_cnn_model(self):
        xn, hn, yn = 50, 100, 10
        model = nn.Sequential(
            nn.Conv2d(1, 10, 4),  # 353'440 mul
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22090, hn),  # 2'209'000 mul
            nn.Linear(hn, yn)  # 1'000 mul
        )  # 2'563'440 mul
        dummy = torch.randn(32, 1, xn, xn)
        met = MAC()
        # print(2563440, met.compute(model, dummy))


if __name__ == '__main__':
    unittest.main()
