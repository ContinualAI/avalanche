import unittest

import torch

from avalanche.benchmarks import benchmark_from_datasets, EagerCLStream, CLScenario
from avalanche.benchmarks import DatasetExperience
from avalanche.benchmarks.scenarios.task_aware import with_task_labels
from avalanche.benchmarks.scenarios.supervised import with_classes_timeline
from avalanche.benchmarks.utils import _make_taskaware_tensor_classification_dataset


class ExperienceDecoratorTests(unittest.TestCase):
    def _assert_is_unique_and_int(self, iterable):
        uniques = set()
        n_elements = len(iterable)
        for x in iterable:
            self.assertIsInstance(x, int)
            uniques.add(x)
        self.assertEqual(n_elements, len(uniques))

    def test_with_task_labels(self):
        exps = []

        tx = torch.rand(10, 17)
        ty = torch.randint(0, 2, (10,))
        t_tl = torch.tensor([0 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty, task_labels=t_tl)
        exps.append(DatasetExperience(dataset=data))

        t_tl = torch.tensor([7 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty, task_labels=t_tl)
        exps.append(DatasetExperience(dataset=data))

        stream = EagerCLStream(name="train", exps=exps)
        bm = CLScenario([stream])
        bm = with_task_labels(bm)

        assert len(bm.streams) == 1
        stream = bm.train_stream

        # exp 0 has tl==0
        tls = stream[0].task_labels
        self._assert_is_unique_and_int(tls)
        assert stream[0].task_label == 0

        # exp 1 has tl==7
        tls = stream[1].task_labels
        self._assert_is_unique_and_int(tls)
        assert stream[1].task_label == 7

    def test_with_classes_timeline_all_attributes(self):
        exps = []

        # class order is [11, 3, 4, 4]
        # exp 0
        tx = torch.rand(10, 17)
        ty = torch.tensor([11 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty)
        exps.append(DatasetExperience(dataset=data))

        # exp 1
        tx = torch.rand(10, 17)
        ty = torch.tensor([3 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty)
        exps.append(DatasetExperience(dataset=data))

        # exp 2
        tx = torch.rand(10, 17)
        ty = torch.tensor([4 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty)
        exps.append(DatasetExperience(dataset=data))

        # exp 3
        tx = torch.rand(10, 17)
        ty = torch.tensor([4 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty)
        exps.append(DatasetExperience(dataset=data))

        stream = EagerCLStream(name="train", exps=exps)
        bm = CLScenario([stream])
        bm = with_classes_timeline(bm)
        assert len(bm.streams) == 1
        stream = bm.train_stream

        # class order is [11, 3, 4, 4]
        # exp 0
        exp = stream[0]
        assert exp.classes_in_this_experience == {11}
        assert len(exp.previous_classes) == 0
        assert exp.classes_seen_so_far == {11}
        assert exp.future_classes == {4, 3}

        # exp 1
        exp = stream[1]
        assert exp.classes_in_this_experience == {3}
        assert exp.previous_classes == {11}
        assert exp.classes_seen_so_far == {11, 3}
        assert exp.future_classes == {4}

        # exp 2
        exp = stream[2]
        assert exp.classes_in_this_experience == {4}
        assert exp.previous_classes == {11, 3}
        assert exp.classes_seen_so_far == {11, 3, 4}
        assert len(exp.future_classes) == 0

        # exp 3
        exp = stream[3]
        assert exp.classes_in_this_experience == {4}
        assert exp.previous_classes == {11, 3, 4}
        assert exp.classes_seen_so_far == {11, 3, 4}
        assert len(exp.future_classes) == 0

    def test_with_classes_timeline(self):
        train_exps = []

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 70, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            _make_taskaware_tensor_classification_dataset(
                tensor_x, tensor_y, task_labels=tensor_t
            )
        )

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(
            _make_taskaware_tensor_classification_dataset(
                tensor_x, tensor_y, task_labels=tensor_t
            )
        )

        test_exps = []
        test_x = torch.rand(200, 3, 28, 28)
        test_y = torch.randint(100, 200, (200,))
        test_t = torch.randint(0, 5, (200,))
        test_exps.append(
            _make_taskaware_tensor_classification_dataset(
                test_x, test_y, task_labels=test_t
            )
        )

        other_stream_exps = []
        other_x = torch.rand(200, 3, 28, 28)
        other_y = torch.randint(400, 600, (200,))
        other_t = torch.randint(0, 5, (200,))
        other_stream_exps.append(
            _make_taskaware_tensor_classification_dataset(
                other_x, other_y, task_labels=other_t
            )
        )

        bm = benchmark_from_datasets(
            train=train_exps, test=test_exps, other=other_stream_exps
        )
        bm = with_classes_timeline(bm)

        train_0_classes = bm.train_stream[0].classes_in_this_experience
        train_1_classes = bm.train_stream[1].classes_in_this_experience
        self._assert_is_unique_and_int(bm.train_stream[0].classes_in_this_experience)
        self._assert_is_unique_and_int(bm.train_stream[1].classes_in_this_experience)

        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)

        test_0_classes = bm.test_stream[0].classes_in_this_experience
        self._assert_is_unique_and_int(bm.test_stream[0].classes_in_this_experience)

        test_0_classes_min = min(test_0_classes)
        test_0_classes_max = max(test_0_classes)
        self.assertGreaterEqual(test_0_classes_min, 100)
        self.assertLess(test_0_classes_max, 200)

        other_0_classes = bm.other_stream[0].classes_in_this_experience
        self._assert_is_unique_and_int(bm.other_stream[0].classes_in_this_experience)

        other_0_classes_min = min(other_0_classes)
        other_0_classes_max = max(other_0_classes)
        self.assertGreaterEqual(other_0_classes_min, 400)
        self.assertLess(other_0_classes_max, 600)
