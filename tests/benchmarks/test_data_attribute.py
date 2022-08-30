import unittest

import torch

from avalanche.benchmarks.utils.data_attribute import DataAttribute


class DataAttributeTests(unittest.TestCase):
    def test_tensor_uniques(self):
        """Test that uniques are correctly computed for tensors."""
        t = torch.zeros(10)
        da = DataAttribute("task_labels", t)
        self.assertEqual(da.uniques, {0.0})
