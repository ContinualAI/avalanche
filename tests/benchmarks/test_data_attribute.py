import unittest

import torch

from avalanche.benchmarks.utils.data_attribute import DataAttribute


class DataAttributeTests(unittest.TestCase):
    def test_tensor_uniques(self):
        """Test that uniques are correctly computed for tensors."""
        t = torch.zeros(10)
        da = DataAttribute(t, "task_labels")
        self.assertEqual(da.uniques, {0.0})

    def test_count(self):
        """Test that count is correctly computed."""
        t0 = torch.zeros(10, dtype=torch.int)
        t1 = torch.ones(10, dtype=torch.int)
        da = DataAttribute(torch.cat([t0, t1]), "task_labels")
        self.assertEqual(da.count, {0: 10, 1: 10})

    def test_val_to_idx(self):
        """Test that val_to_idx is correctly computed."""
        t0 = torch.zeros(10, dtype=torch.int)
        t1 = torch.ones(10, dtype=torch.int)
        da = DataAttribute(torch.cat([t0, t1]), "task_labels")
        self.assertEqual(
            da.val_to_idx, {0: list(range(10)), 1: list(range(10, 20))}
        )

    def test_subset(self):
        """Test that subset is correctly computed."""
        t0 = torch.zeros(10, dtype=torch.int)
        t1 = torch.ones(10, dtype=torch.int)
        da = DataAttribute(torch.cat([t0, t1]), "task_labels")
        self.assertEqual(list(da.subset(range(10)).data), list(t0))
        self.assertEqual(list(da.subset(range(10, 20)).data), list(t1))

    def test_concat(self):
        """Test that concat is correctly computed."""
        t0 = torch.zeros(10, dtype=torch.int)
        t1 = torch.ones(10, dtype=torch.int)
        da = DataAttribute(torch.cat([t0, t1]), "task_labels")
        self.assertEqual(
            list(da.concat(da).data), list(torch.cat([t0, t1, t0, t1]))
        )
