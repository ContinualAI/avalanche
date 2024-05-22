import unittest

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch

from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.scenarios.task_aware import task_incremental_benchmark
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from torch.utils.data import TensorDataset
import numpy as np


class TestsTaskAware(unittest.TestCase):
    def test_taskaware(self):
        """Common use case: add task labels to class-incremental benchmark."""
        n_classes, n_samples_per_class, n_features = 10, 3, 7

        for _ in range(10000):
            dataset = make_classification(
                n_samples=n_classes * n_samples_per_class,
                n_classes=n_classes,
                n_features=n_features,
                n_informative=6,
                n_redundant=0,
            )

            # The following check is required to ensure that at least 2 exemplars
            # per class are generated. Otherwise, the train_test_split function will
            # fail.
            _, unique_count = np.unique(dataset[1], return_counts=True)
            if np.min(unique_count) > 1:
                break

        X = torch.from_numpy(dataset[0]).float()
        y = torch.from_numpy(dataset[1]).long()
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, train_size=0.6, shuffle=True, stratify=y
        )

        d1 = TensorDataset(train_X, train_y)
        da = DataAttribute(train_y.tolist(), "targets")
        d1 = ClassificationDataset(d1, data_attributes=[da])

        d2 = TensorDataset(test_X, test_y)
        da = DataAttribute(test_y.tolist(), "targets")
        d2 = ClassificationDataset(d2, data_attributes=[da])

        bm_ci = class_incremental_benchmark(
            {"train": d1, "test": d2}, num_experiences=n_classes
        )
        bm_ti = task_incremental_benchmark(bm_ci)

        assert len(list(bm_ti.train_stream)) == len(list(bm_ci.train_stream))
        assert len(list(bm_ti.test_stream)) == len(list(bm_ci.test_stream))

        ci_train = bm_ci.train_stream
        for eid, exp in enumerate(bm_ti.train_stream):
            assert exp.task_label == eid
            assert isinstance(exp.task_labels, list)
            assert len(ci_train[eid].dataset) == len(exp.dataset)


if __name__ == "__main__":
    unittest.main()
