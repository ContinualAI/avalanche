import unittest

import numpy as np
import torch

from avalanche.benchmarks.utils import (
    _taskaware_classification_subset,
    make_avalanche_dataset,
)
from avalanche.benchmarks.utils.data_attribute import DataAttribute, TensorDataAttribute


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
        self.assertEqual(da.val_to_idx, {0: list(range(10)), 1: list(range(10, 20))})

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
        self.assertEqual(list(da.concat(da).data), list(torch.cat([t0, t1, t0, t1])))


class TensorDataAttributeTests(unittest.TestCase):
    def test_subset(self):
        """Test that subset is correctly computed."""
        t0 = torch.zeros(10)
        t1 = torch.ones(10)
        da = TensorDataAttribute(torch.cat([t0, t1]), "logit")
        self.assertEqual(list(da.subset(range(10)).data), list(t0))
        self.assertEqual(list(da.subset(range(10, 20)).data), list(t1))

    def test_concat(self):
        """Test that concat is correctly computed."""
        t0 = torch.zeros(10)
        t1 = torch.ones(10)
        da = DataAttribute(torch.cat([t0, t1]), "logits")
        self.assertEqual(list(da.concat(da).data), list(torch.cat([t0, t1, t0, t1])))

    def test_swap(self):
        """Test that data attributes are
        always returned in the same order"""
        # Fake x, y
        t1 = list(zip(np.arange(10), np.arange(10)))
        t2 = torch.ones(10).tolist()
        t3 = (torch.ones(10) * 2).tolist()
        t4 = (torch.ones(10) * 3).tolist()

        dataset = make_avalanche_dataset(
            t1,
            data_attributes=[
                TensorDataAttribute(t2, name="logits", use_in_getitem=True),
                TensorDataAttribute(t3, name="logits2", use_in_getitem=True),
            ],
        )

        # Now add another attribute
        dataset = make_avalanche_dataset(
            dataset,
            data_attributes=[
                TensorDataAttribute(t4, name="logits0", use_in_getitem=True),
            ],
        )

        self.assertSequenceEqual([0.0, 0.0, 1.0, 2.0, 3.0], dataset[0])


if __name__ == "__main__":
    unittest.main()
