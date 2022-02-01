import unittest
import weakref
import gc

import torch

from avalanche.benchmarks import (
    dataset_benchmark,
    GenericExperience,
    GenericCLScenario,
)
from avalanche.benchmarks.utils import (
    AvalancheTensorDataset,
    AvalancheDatasetType,
)


class GenericCLScenarioTests(unittest.TestCase):
    def test_classes_in_exp(self):
        train_exps = []

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 70, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
        )

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
        )

        test_exps = []
        test_x = torch.rand(200, 3, 28, 28)
        test_y = torch.randint(100, 200, (200,))
        test_t = torch.randint(0, 5, (200,))
        test_exps.append(
            AvalancheTensorDataset(test_x, test_y, task_labels=test_t)
        )

        other_stream_exps = []
        other_x = torch.rand(200, 3, 28, 28)
        other_y = torch.randint(400, 600, (200,))
        other_t = torch.randint(0, 5, (200,))
        other_stream_exps.append(
            AvalancheTensorDataset(other_x, other_y, task_labels=other_t)
        )

        benchmark_instance = dataset_benchmark(
            train_datasets=train_exps,
            test_datasets=test_exps,
            other_streams_datasets={"other": other_stream_exps},
        )

        train_0_classes = benchmark_instance.classes_in_experience["train"][0]
        train_1_classes = benchmark_instance.classes_in_experience["train"][1]
        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)

        # Test deprecated behavior
        train_0_classes = benchmark_instance.classes_in_experience[0]
        train_1_classes = benchmark_instance.classes_in_experience[1]
        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)
        # End test deprecated behavior

        test_0_classes = benchmark_instance.classes_in_experience["test"][0]
        test_0_classes_min = min(test_0_classes)
        test_0_classes_max = max(test_0_classes)
        self.assertGreaterEqual(test_0_classes_min, 100)
        self.assertLess(test_0_classes_max, 200)

        other_0_classes = benchmark_instance.classes_in_experience["other"][0]
        other_0_classes_min = min(other_0_classes)
        other_0_classes_max = max(other_0_classes)
        self.assertGreaterEqual(other_0_classes_min, 400)
        self.assertLess(other_0_classes_max, 600)

    def test_classes_in_this_experience(self):
        train_exps = []

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 70, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
        )

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
        )

        test_exps = []
        test_x = torch.rand(200, 3, 28, 28)
        test_y = torch.randint(100, 200, (200,))
        test_t = torch.randint(0, 5, (200,))
        test_exps.append(
            AvalancheTensorDataset(test_x, test_y, task_labels=test_t)
        )

        other_stream_exps = []
        other_x = torch.rand(200, 3, 28, 28)
        other_y = torch.randint(400, 600, (200,))
        other_t = torch.randint(0, 5, (200,))
        other_stream_exps.append(
            AvalancheTensorDataset(other_x, other_y, task_labels=other_t)
        )

        benchmark_instance = dataset_benchmark(
            train_datasets=train_exps,
            test_datasets=test_exps,
            other_streams_datasets={"other": other_stream_exps},
        )

        train_exp_0: GenericExperience = benchmark_instance.train_stream[0]
        train_exp_1: GenericExperience = benchmark_instance.train_stream[1]
        train_0_classes = train_exp_0.classes_in_this_experience
        train_1_classes = train_exp_1.classes_in_this_experience
        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)

        test_exp_0: GenericExperience = benchmark_instance.test_stream[0]
        test_0_classes = test_exp_0.classes_in_this_experience
        test_0_classes_min = min(test_0_classes)
        test_0_classes_max = max(test_0_classes)
        self.assertGreaterEqual(test_0_classes_min, 100)
        self.assertLess(test_0_classes_max, 200)

        other_exp_0: GenericExperience = benchmark_instance.other_stream[0]
        other_0_classes = other_exp_0.classes_in_this_experience
        other_0_classes_min = min(other_0_classes)
        other_0_classes_max = max(other_0_classes)
        self.assertGreaterEqual(other_0_classes_min, 400)
        self.assertLess(other_0_classes_max, 600)

    def test_lazy_benchmark(self):
        train_exps, test_exps, other_stream_exps = self._make_tensor_datasets()

        def train_gen():
            # Lazy generator of the training stream
            for dataset in train_exps:
                yield dataset

        def test_gen():
            # Lazy generator of the test stream
            for dataset in test_exps:
                yield dataset

        def other_gen():
            # Lazy generator of the "other" stream
            for dataset in other_stream_exps:
                yield dataset

        benchmark_instance = GenericCLScenario(
            stream_definitions=dict(
                train=(
                    (train_gen(), len(train_exps)),
                    [
                        train_exps[0].targets_task_labels,
                        train_exps[1].targets_task_labels,
                    ],
                ),
                test=(
                    (test_gen(), len(test_exps)),
                    [test_exps[0].targets_task_labels],
                ),
                other=(
                    (other_gen(), len(other_stream_exps)),
                    [other_stream_exps[0].targets_task_labels],
                ),
            )
        )

        # --- START: Test classes timeline before first experience ---
        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(0)
        self.assertIsNone(current_classes)
        self.assertSetEqual(set(), set(prev_classes))
        self.assertIsNone(cumulative_classes)
        self.assertIsNone(future_classes)
        # --- END: Test classes timeline before first experience ---

        train_exp_0: GenericExperience = benchmark_instance.train_stream[0]
        # --- START: Test classes timeline at first experience ---
        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(0)

        self.assertSetEqual(set(train_exps[0].targets), set(current_classes))
        self.assertSetEqual(set(), set(prev_classes))
        self.assertSetEqual(set(train_exps[0].targets), set(cumulative_classes))
        self.assertIsNone(future_classes)

        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(1)

        self.assertIsNone(current_classes)
        self.assertSetEqual(set(train_exps[0].targets), set(prev_classes))
        # None because we didn't load exp 0 yet
        self.assertIsNone(cumulative_classes)
        self.assertSetEqual(set(), set(future_classes))
        # --- END: Test classes timeline at first experience ---

        train_exp_1: GenericExperience = benchmark_instance.train_stream[1]
        # --- START: Test classes timeline at second experience ---
        # Check if get_classes_timeline(0) is consistent
        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(0)

        self.assertSetEqual(set(train_exps[0].targets), set(current_classes))
        self.assertSetEqual(set(), set(prev_classes))
        self.assertSetEqual(set(train_exps[0].targets), set(cumulative_classes))
        # We now have access to future classes!
        self.assertSetEqual(set(train_exps[1].targets), set(future_classes))

        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(1)

        self.assertSetEqual(set(train_exps[1].targets), set(current_classes))
        self.assertSetEqual(set(train_exps[0].targets), set(prev_classes))
        self.assertSetEqual(
            set(train_exps[0].targets).union(set(train_exps[1].targets)),
            set(cumulative_classes),
        )
        self.assertSetEqual(set(), set(future_classes))
        # --- END: Test classes timeline at second experience ---

        train_0_classes = train_exp_0.classes_in_this_experience
        train_1_classes = train_exp_1.classes_in_this_experience
        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)

        with self.assertRaises(IndexError):
            train_exp_2: GenericExperience = benchmark_instance.train_stream[2]

        test_exp_0: GenericExperience = benchmark_instance.test_stream[0]
        test_0_classes = test_exp_0.classes_in_this_experience
        test_0_classes_min = min(test_0_classes)
        test_0_classes_max = max(test_0_classes)
        self.assertGreaterEqual(test_0_classes_min, 100)
        self.assertLess(test_0_classes_max, 200)

        with self.assertRaises(IndexError):
            test_exp_1: GenericExperience = benchmark_instance.test_stream[1]

        other_exp_0: GenericExperience = benchmark_instance.other_stream[0]
        other_0_classes = other_exp_0.classes_in_this_experience
        other_0_classes_min = min(other_0_classes)
        other_0_classes_max = max(other_0_classes)
        self.assertGreaterEqual(other_0_classes_min, 400)
        self.assertLess(other_0_classes_max, 600)

        with self.assertRaises(IndexError):
            other_exp_1: GenericExperience = benchmark_instance.other_stream[1]

    def test_lazy_benchmark_drop_old_ones(self):
        train_exps, test_exps, other_stream_exps = self._make_tensor_datasets()

        train_dataset_exp_0_weak_ref = weakref.ref(train_exps[0])
        train_dataset_exp_1_weak_ref = weakref.ref(train_exps[1])

        train_gen = GenericCLScenarioTests._generate_stream(train_exps)
        test_gen = GenericCLScenarioTests._generate_stream(test_exps)
        other_gen = GenericCLScenarioTests._generate_stream(other_stream_exps)

        benchmark_instance = GenericCLScenario(
            stream_definitions=dict(
                train=(
                    (train_gen, len(train_exps)),
                    [
                        train_exps[0].targets_task_labels,
                        train_exps[1].targets_task_labels,
                    ],
                ),
                test=(
                    (test_gen, len(test_exps)),
                    [test_exps[0].targets_task_labels],
                ),
                other=(
                    (other_gen, len(other_stream_exps)),
                    [other_stream_exps[0].targets_task_labels],
                ),
            )
        )

        # --- START: Test classes timeline before first experience ---
        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(0)
        self.assertIsNone(current_classes)
        self.assertSetEqual(set(), set(prev_classes))
        self.assertIsNone(cumulative_classes)
        self.assertIsNone(future_classes)
        # --- END: Test classes timeline before first experience ---

        train_exp_0: GenericExperience = benchmark_instance.train_stream[0]
        # --- START: Test classes timeline at first experience ---
        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(0)

        self.assertSetEqual(set(train_exps[0].targets), set(current_classes))
        self.assertSetEqual(set(), set(prev_classes))
        self.assertSetEqual(set(train_exps[0].targets), set(cumulative_classes))
        self.assertIsNone(future_classes)

        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(1)

        self.assertIsNone(current_classes)
        self.assertSetEqual(set(train_exps[0].targets), set(prev_classes))
        # None because we didn't load exp 0 yet
        self.assertIsNone(cumulative_classes)
        self.assertSetEqual(set(), set(future_classes))
        # --- END: Test classes timeline at first experience ---

        # Check if it works when the previous experience is dropped
        benchmark_instance.train_stream.drop_previous_experiences(0)
        train_exp_1: GenericExperience = benchmark_instance.train_stream[1]
        # --- START: Test classes timeline at second experience ---
        # Check if get_classes_timeline(0) is consistent
        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(0)

        self.assertSetEqual(set(train_exps[0].targets), set(current_classes))
        self.assertSetEqual(set(), set(prev_classes))
        self.assertSetEqual(set(train_exps[0].targets), set(cumulative_classes))
        # We now have access to future classes!
        self.assertSetEqual(set(train_exps[1].targets), set(future_classes))

        (
            current_classes,
            prev_classes,
            cumulative_classes,
            future_classes,
        ) = benchmark_instance.get_classes_timeline(1)

        self.assertSetEqual(set(train_exps[1].targets), set(current_classes))
        self.assertSetEqual(set(train_exps[0].targets), set(prev_classes))
        self.assertSetEqual(
            set(train_exps[0].targets).union(set(train_exps[1].targets)),
            set(cumulative_classes),
        )
        self.assertSetEqual(set(), set(future_classes))
        # --- END: Test classes timeline at second experience ---

        train_0_classes = train_exp_0.classes_in_this_experience
        train_1_classes = train_exp_1.classes_in_this_experience
        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)

        with self.assertRaises(IndexError):
            train_exp_2: GenericExperience = benchmark_instance.train_stream[2]

        test_exp_0: GenericExperience = benchmark_instance.test_stream[0]
        test_0_classes = test_exp_0.classes_in_this_experience
        test_0_classes_min = min(test_0_classes)
        test_0_classes_max = max(test_0_classes)
        self.assertGreaterEqual(test_0_classes_min, 100)
        self.assertLess(test_0_classes_max, 200)

        with self.assertRaises(IndexError):
            test_exp_1: GenericExperience = benchmark_instance.test_stream[1]

        other_exp_0: GenericExperience = benchmark_instance.other_stream[0]
        other_0_classes = other_exp_0.classes_in_this_experience
        other_0_classes_min = min(other_0_classes)
        other_0_classes_max = max(other_0_classes)
        self.assertGreaterEqual(other_0_classes_min, 400)
        self.assertLess(other_0_classes_max, 600)

        with self.assertRaises(IndexError):
            other_exp_1: GenericExperience = benchmark_instance.other_stream[1]

        train_exps = None
        train_exp_0 = None
        train_exp_1 = None
        train_0_classes = None
        train_1_classes = None
        train_gen = None

        # The generational GC is needed, ref-count is not enough here
        gc.collect()

        # This will check that the train dataset of exp0 has been garbage
        # collected correctly
        self.assertIsNone(train_dataset_exp_0_weak_ref())
        self.assertIsNotNone(train_dataset_exp_1_weak_ref())

        benchmark_instance.train_stream.drop_previous_experiences(1)
        gc.collect()

        # This will check that exp1 has been garbage collected correctly
        self.assertIsNone(train_dataset_exp_0_weak_ref())
        self.assertIsNone(train_dataset_exp_1_weak_ref())

        with self.assertRaises(Exception):
            exp_0 = benchmark_instance.train_stream[0]

        with self.assertRaises(Exception):
            exp_1 = benchmark_instance.train_stream[1]

    def _make_tensor_datasets(self):
        train_exps = []

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 70, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            AvalancheTensorDataset(
                tensor_x,
                tensor_y,
                task_labels=tensor_t,
                dataset_type=AvalancheDatasetType.CLASSIFICATION,
            )
        )

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            AvalancheTensorDataset(
                tensor_x,
                tensor_y,
                task_labels=tensor_t,
                dataset_type=AvalancheDatasetType.CLASSIFICATION,
            )
        )

        test_exps = []
        test_x = torch.rand(200, 3, 28, 28)
        test_y = torch.randint(100, 200, (200,))
        test_t = torch.randint(0, 5, (200,))
        test_exps.append(
            AvalancheTensorDataset(
                test_x,
                test_y,
                task_labels=test_t,
                dataset_type=AvalancheDatasetType.CLASSIFICATION,
            )
        )

        other_stream_exps = []
        other_x = torch.rand(200, 3, 28, 28)
        other_y = torch.randint(400, 600, (200,))
        other_t = torch.randint(0, 5, (200,))
        other_stream_exps.append(
            AvalancheTensorDataset(
                other_x,
                other_y,
                task_labels=other_t,
                dataset_type=AvalancheDatasetType.CLASSIFICATION,
            )
        )

        return train_exps, test_exps, other_stream_exps

    @staticmethod
    def _generate_stream(dataset_list):
        # Lazy generator of a stream
        for dataset in dataset_list:
            yield dataset


if __name__ == "__main__":
    unittest.main()
