#!/usr/bin/env python3
import unittest

import numpy as np

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader


class TestReplayDataLoader(unittest.TestCase):
    def setUp(self):
        scenario = SplitMNIST(2)
        dataset_for_current = scenario.train_stream[1].dataset
        dataset_for_memory = scenario.train_stream[0].dataset

        indices_big_set = np.random.choice(
            np.arange(len(dataset_for_current)), size=10000, replace=False
        )

        indices_small_set = np.random.choice(
            np.arange(len(dataset_for_current)), size=1000, replace=False
        )

        indices_tiny_set = np.random.choice(
            np.arange(len(dataset_for_current)), size=100, replace=False
        )

        self.big_task_set = dataset_for_current.subset(indices_big_set)
        self.small_task_set = dataset_for_current.subset(indices_small_set)
        self.tiny_task_set = dataset_for_current.subset(indices_tiny_set)

        indices_memory = np.random.choice(
            np.arange(len(dataset_for_memory)), size=2000, replace=False
        )

        indices_memory_small = np.random.choice(
            np.arange(len(dataset_for_memory)), size=100, replace=False
        )

        self.memory_set = dataset_for_memory.subset(indices_memory)
        self.small_memory_set = dataset_for_memory.subset(indices_memory_small)

        self._batch_size = None
        self._task_dataset = None

    def _make_loader(self, memory_set=None, **kwargs):
        if memory_set is None:
            memory_set = self.memory_set

        loader = ReplayDataLoader(
            self._task_dataset,
            memory_set,
            batch_size=self._batch_size,
            batch_size_mem=self._batch_size,
            oversample_small_tasks=True,
            drop_last=True,
            **kwargs
        )
        return loader

    def _test_batch_size(self, loader, expected_size=None):
        if expected_size is None:
            expected_size = self._batch_size * 2

        for batch in loader:
            self.assertEqual(len(batch[0]), expected_size)

    def _test_length(self, loader):
        self.assertEqual(len(loader), self._length)

    def _test_actual_length(self, loader):
        counter = 0
        for batch in loader:
            counter += 1
        self.assertEqual(counter, self._length)

    def _launch_test_suite(self, loader):
        self._test_batch_size(loader)
        self._test_length(loader)
        self._test_actual_length(loader)

    def _launch_test_suite_dropped_memory(self, loader):
        self._test_batch_size(loader, expected_size=self._batch_size)
        self._test_length(loader)
        self._test_actual_length(loader)

    def test_bigger_memory(self):
        self._batch_size = 64
        self._task_dataset = self.small_task_set
        loader = self._make_loader()
        self._launch_test_suite(loader)

    def test_smaller_memory(self):
        self._batch_size = 64
        self._task_dataset = self.big_task_set
        loader = self._make_loader()
        self._launch_test_suite(loader)

    def test_big_batch_size(self):
        self._batch_size = 256
        self._task_dataset = self.big_task_set
        loader = self._make_loader()
        self._launch_test_suite(loader)

    def test_zero_iterations_memory(self):
        self._batch_size = 256
        self._task_dataset = self.big_task_set
        loader = self._make_loader(memory_set=self.small_memory_set)
        self._launch_test_suite_dropped_memory(loader)

    def test_zero_iterations_current(self):
        self._batch_size = 256
        self._task_dataset = self.tiny_task_set
        loader = self._make_loader(memory_set=self.memory_set)
        self.assertEqual(0, self._length)
        self._launch_test_suite(loader)

    def test_small_batch_size(self):
        self._batch_size = 5
        self._task_dataset = self.big_task_set
        loader = self._make_loader()
        self._launch_test_suite(loader)

    @property
    def _length(self):
        return len(self._task_dataset) // self._batch_size


if __name__ == "__main__":
    unittest.main()
