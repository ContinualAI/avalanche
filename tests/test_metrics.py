""" Metrics Tests"""

import unittest

import torch

from avalanche.evaluation.metrics import (
    TaskAwareAccuracy,
    AverageMeanClassAccuracy,
    MultiStreamAMCA,
    ClassAccuracy,
    TaskAwareLoss,
    ConfusionMatrix,
    DiskUsage,
    MAC,
    CPUUsage,
    MaxGPU,
    MaxRAM,
    Mean,
    Sum,
    ElapsedTime,
    Forgetting,
    ForwardTransfer,
)


#################################
#################################
#    STANDALONE METRIC TEST     #
#################################
#################################


class GeneralMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3
        self.input_size = 10
        self.n_classes = 3
        self.n_tasks = 2
        self.out = torch.randn(self.batch_size, self.input_size)
        self.y = torch.randint(0, self.n_classes, (self.batch_size,))
        self.task_labels = torch.randint(0, self.n_tasks, (self.batch_size,))

    def test_accuracy(self):
        metric = TaskAwareAccuracy()
        self.assertEqual(metric.result(), {})
        metric.update(self.out, self.y, 0)
        self.assertLessEqual(metric.result(0)[0], 1)
        self.assertGreaterEqual(metric.result(0)[0], 0)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_accuracy_task_per_pattern(self):
        metric = TaskAwareAccuracy()
        self.assertEqual(metric.result(), {})
        metric.update(self.out, self.y, self.task_labels)
        out = metric.result()
        for k, v in out.items():
            self.assertIn(k, self.task_labels.tolist())
            self.assertLessEqual(v, 1)
            self.assertGreaterEqual(v, 0)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_class_accuracy(self):
        metric = ClassAccuracy()
        self.assertDictEqual(metric.result(), {})

        metric.update(self.out, self.y, 0)
        result = metric.result()

        for task_id, task_classes in result.items():
            self.assertIsInstance(task_id, int)
            self.assertIsInstance(task_classes, dict)

            self.assertEqual(task_id, 0)
            expected_n_classes = len(torch.unique(self.y))
            self.assertEqual(len(task_classes), expected_n_classes)
            for class_id, class_accuracy in task_classes.items():
                self.assertLess(class_id, self.n_classes)
                self.assertGreaterEqual(class_id, 0)
                self.assertLessEqual(class_accuracy, 1)
                self.assertGreaterEqual(class_accuracy, 0)

        metric.reset()
        self.assertDictEqual(
            metric.result(), {0: {int(c): 0.0 for c in torch.unique(self.y)}}
        )

    def test_class_accuracy_extended(self):
        metric = ClassAccuracy()

        my_y = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out = torch.as_tensor([0, 0, 1, 2, 1, 1, 1])
        # 0: 50%, 1: 100%, 2: 0%
        metric.update(my_out, my_y, 0)

        self.assertDictEqual(metric.result(), {0: {0: 0.5, 1: 1.00, 2: 0.0}})

        metric.reset()
        self.assertDictEqual(metric.result(), {0: {0: 0.0, 1: 0.0, 2: 0.0}})

        # Add a task
        metric.update(my_out, my_y, 1)
        self.assertDictEqual(
            metric.result(),
            {0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {0: 0.5, 1: 1.00, 2: 0.0}},
        )

        metric.reset()
        self.assertDictEqual(
            metric.result(),
            {0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {0: 0.0, 1: 0.0, 2: 0.0}},
        )

    def test_class_accuracy_static(self):
        with self.assertRaises(Exception):
            metric = ClassAccuracy(classes={0: (0, 1, 2, "aaa"), 1: (0, 1, 2)})

        metric = ClassAccuracy(classes={0: (0, 1, 2), 1: (0, 1, 2)})
        self.assertDictEqual(
            metric.result(),
            {0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {0: 0.0, 1: 0.0, 2: 0.0}},
        )

        metric.reset()
        self.assertDictEqual(
            metric.result(),
            {0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {0: 0.0, 1: 0.0, 2: 0.0}},
        )

        my_y = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out = torch.as_tensor([0, 0, 1, 2, 1, 1, 1])
        # 0: 50%, 1: 100%, 2: 0%
        metric.update(my_out, my_y, 0)

        self.assertDictEqual(
            metric.result(),
            {0: {0: 0.5, 1: 1.00, 2: 0.0}, 1: {0: 0.0, 1: 0.0, 2: 0.0}},
        )

        metric.reset()
        self.assertDictEqual(
            metric.result(),
            {0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {0: 0.0, 1: 0.0, 2: 0.0}},
        )

        # Add a task
        metric.update(my_out, my_y, 1)
        self.assertDictEqual(
            metric.result(),
            {0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {0: 0.5, 1: 1.00, 2: 0.0}},
        )

        metric.reset()
        self.assertDictEqual(
            metric.result(),
            {
                0: {0: 0.0, 1: 0.0, 2: 0.0},
                1: {0: 0.0, 1: 0.0, 2: 0.0},
            },
        )

    def test_amca_first_update(self):
        metric = AverageMeanClassAccuracy()
        self.assertDictEqual(metric.result(), {})

        my_y = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out = torch.as_tensor([0, 0, 1, 2, 1, 1, 1])
        my_amca = (0.5 + 1.0 + 0.0) / 3
        # 0: 50%, 1: 100%, 2: 0%

        my_y2 = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out2 = torch.as_tensor([2, 2, 2, 2, 1, 2, 1])
        my_amca2 = (0.0 + 0.0 + 0.5) / 3
        # 0: 0%, 1: 0%, 2: 50%

        my_amca_1_and_2 = (my_amca + my_amca2) / 2

        metric.next_experience()
        self.assertDictEqual(metric.result(), {})

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca})

        metric.reset()
        metric.next_experience()
        # After reset, previously dynamically added tasks are remembered
        self.assertDictEqual(metric.result(), {0: 0.0})

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca})

        metric.update(my_out2, my_y2, 0)
        self.assertDictEqual(metric.result(), {0: my_amca_1_and_2})
        metric.next_experience()
        self.assertDictEqual(metric.result(), {0: my_amca_1_and_2 / 2})

    def test_amca_single_task_dynamic(self):
        metric = AverageMeanClassAccuracy()
        self.assertDictEqual(metric.result(), {})

        my_y = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out = torch.as_tensor([0, 0, 1, 2, 1, 1, 1])
        my_amca = (0.5 + 1.0 + 0.0) / 3
        # 0: 50%, 1: 100%, 2: 0%

        my_y2 = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out2 = torch.as_tensor([2, 2, 2, 2, 1, 2, 1])
        my_amca2 = (0.0 + 0.0 + 0.5) / 3
        # 0: 0%, 1: 0%, 2: 50%

        my_amca_1_and_2 = (my_amca + my_amca2) / 2

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca})

        metric.reset()
        # After reset, previously dynamically added tasks are remembered
        self.assertDictEqual(metric.result(), {0: 0.0})

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca})

        metric.update(my_out2, my_y2, 0)
        self.assertDictEqual(metric.result(), {0: my_amca_1_and_2})
        metric.next_experience()
        self.assertDictEqual(metric.result(), {0: my_amca_1_and_2 / 2})

    def test_amca_two_task_dynamic(self):
        metric = AverageMeanClassAccuracy()
        self.assertEqual(metric.result(), {})

        my_y = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out = torch.as_tensor([0, 0, 1, 2, 1, 1, 1])
        my_amca = (0.5 + 1.0 + 0.0) / 3
        # 0: 50%, 1: 100%, 2: 0%

        my_y2 = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out2 = torch.as_tensor([2, 2, 2, 2, 1, 2, 1])
        my_amca2 = (0.0 + 0.0 + 0.5) / 3
        # 0: 0%, 1: 0%, 2: 50%

        my_amca_1_and_2 = (my_amca + my_amca2) / 2

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca})

        metric.reset()
        # After reset, previously dynamically added tasks are remembered
        self.assertDictEqual(metric.result(), {0: 0.0})

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca})

        metric.update(my_out2, my_y2, 1)
        self.assertDictEqual(metric.result(), {0: my_amca, 1: my_amca2})
        metric.next_experience()
        self.assertDictEqual(metric.result(), {0: my_amca / 2, 1: my_amca2 / 2})

        metric.reset()
        self.assertDictEqual(metric.result(), {0: 0.0, 1: 0.0})

    def test_amca_two_task_static(self):
        # Test AMCA by passing the classes parameter (non-dynamic)
        with self.assertRaises(Exception):
            metric = AverageMeanClassAccuracy(
                classes={0: (0, 1, 2, "aaa"), 1: (0, 1, 2)}
            )

        metric = AverageMeanClassAccuracy(classes={0: (0, 1, 2), 1: (0, 1, 2)})
        self.assertDictEqual(metric.result(), {0: 0.0, 1: 0.0})
        metric.reset()
        self.assertDictEqual(metric.result(), {0: 0.0, 1: 0.0})

        metric.next_experience()
        self.assertDictEqual(metric.result(), {0: 0.0, 1: 0.0})
        metric.reset()

        my_y = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out = torch.as_tensor([0, 0, 1, 2, 1, 1, 1])
        my_amca = (0.5 + 1.0 + 0.0) / 3
        # 0: 50%, 1: 100%, 2: 0%

        my_y2 = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out2 = torch.as_tensor([2, 2, 2, 2, 1, 2, 1])
        my_amca2 = (0.0 + 0.0 + 0.5) / 3
        # 0: 0%, 1: 0%, 2: 50%

        my_amca_1_and_2 = (my_amca + my_amca2) / 2

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca, 1: 0.0})

        metric.reset()
        # After reset, previously dynamically added tasks are remembered
        self.assertDictEqual(metric.result(), {0: 0.0, 1: 0.0})

        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {0: my_amca, 1: 0.0})

        metric.update(my_out2, my_y2, 1)
        self.assertDictEqual(metric.result(), {0: my_amca, 1: my_amca2})
        metric.next_experience()
        self.assertDictEqual(metric.result(), {0: my_amca / 2, 1: my_amca2 / 2})

        many_ts = [0] * len(my_out)
        many_ts += [1] * len(my_out2)
        metric.update(
            torch.cat((my_out, my_out2)),
            torch.cat((my_y, my_y2)),
            torch.as_tensor(many_ts),
        )

        self.assertDictEqual(metric.result(), {0: my_amca, 1: my_amca2})

        metric.next_experience()
        self.assertDictEqual(
            metric.result(), {0: my_amca * 2 / 3, 1: my_amca2 * 2 / 3}
        )

        metric.reset()
        self.assertDictEqual(metric.result(), {0: 0.0, 1: 0.0})

    def test_multistream_amca_two_task_dynamic(self):
        metric = MultiStreamAMCA()
        self.assertEqual(metric.result(), {})

        my_y = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out = torch.as_tensor([0, 0, 1, 2, 1, 1, 1])
        my_amca = (0.5 + 1.0 + 0.0) / 3
        # 0: 50%, 1: 100%, 2: 0%

        my_y2 = torch.as_tensor([0, 0, 1, 0, 2, 2, 0])
        my_out2 = torch.as_tensor([2, 2, 2, 2, 1, 2, 1])
        my_amca2 = (0.0 + 0.0 + 0.5) / 3
        # 0: 0%, 1: 0%, 2: 50%

        with self.assertRaises(RuntimeError):
            # Should call next_experience first
            metric.update(my_out, my_y, 0)

        metric.set_stream("test")
        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {"test": {0: my_amca}})

        metric.reset()
        # After reset, previously dynamically added tasks are remembered
        self.assertDictEqual(metric.result(), {"test": {0: 0.0}})

        with self.assertRaises(RuntimeError):
            # Again, should call next_experience first, even after reset
            metric.update(my_out, my_y, 0)

        metric.set_stream("test")
        metric.update(my_out, my_y, 0)
        self.assertDictEqual(metric.result(), {"test": {0: my_amca}})

        metric.set_stream("train")
        self.assertDictEqual(
            metric.result(), {"test": {0: my_amca}, "train": {}}
        )

        metric.update(my_out2, my_y2, 1)
        self.assertDictEqual(
            metric.result(), {"test": {0: my_amca}, "train": {1: my_amca2}}
        )

        metric.set_stream("test")
        self.assertDictEqual(
            metric.result(), {"test": {0: my_amca}, "train": {1: my_amca2}}
        )

        metric.finish_phase()
        metric.set_stream("train")
        self.assertDictEqual(
            metric.result(),
            {"test": {0: my_amca / 2}, "train": {1: my_amca2 / 2}},
        )

        metric.reset()
        self.assertDictEqual(
            metric.result(), {"test": {0: 0.0}, "train": {1: 0.0}}
        )

    def test_loss(self):
        metric = TaskAwareLoss()
        self.assertEqual(metric.result(0)[0], 0)
        metric.update(torch.tensor(1.0), self.batch_size, 0)
        self.assertGreaterEqual(metric.result(0)[0], 0)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_loss_multi_task(self):
        metric = TaskAwareLoss()
        self.assertEqual(metric.result(), {})
        metric.update(torch.tensor(1.0), 1, 0)
        metric.update(torch.tensor(2.0), 1, 1)
        out = metric.result()
        for k, v in out.items():
            self.assertIn(k, [0, 1])
            if k == 0:
                self.assertEqual(v, 1)
            else:
                self.assertEqual(v, 2)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_cm(self):
        metric = ConfusionMatrix()
        cm = metric.result()
        self.assertTrue((cm == 0).all().item())
        metric.update(self.y, self.out)
        cm = metric.result()
        self.assertTrue((cm >= 0).all().item())
        metric.reset()
        cm = metric.result()
        self.assertTrue((cm == 0).all().item())

    def test_ram(self):
        metric = MaxRAM()
        self.assertEqual(metric.result(), 0)
        metric.start_thread()  # start thread
        self.assertGreaterEqual(metric.result(), 0)
        metric.stop_thread()  # stop thread
        metric.reset()  # stop thread
        self.assertEqual(metric.result(), 0)

    def test_gpu(self):
        if torch.cuda.is_available():
            metric = MaxGPU(0)
            self.assertEqual(metric.result(), 0)
            metric.start_thread()  # start thread
            self.assertGreaterEqual(metric.result(), 0)
            metric.stop_thread()  # stop thread
            metric.reset()  # stop thread
            self.assertEqual(metric.result(), 0)

    def test_cpu(self):
        metric = CPUUsage()
        self.assertEqual(metric.result(), 0)
        metric.update()
        self.assertGreaterEqual(metric.result(), 0)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_disk(self):
        metric = DiskUsage()
        self.assertEqual(metric.result(), 0)
        metric.update()
        self.assertGreaterEqual(metric.result(), 0)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_timing(self):
        metric = ElapsedTime()
        self.assertEqual(metric.result(), 0)
        metric.update()  # need two update calls
        self.assertEqual(metric.result(), 0)
        metric.update()
        self.assertGreaterEqual(metric.result(), 0)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_mac(self):
        model = torch.nn.Linear(self.input_size, 2)
        metric = MAC()
        self.assertEqual(metric.result(), 0)
        metric.update(model, self.out)
        self.assertGreaterEqual(metric.result(), 0)

    def test_mean(self):
        metric = Mean()
        self.assertEqual(metric.result(), 0)
        metric.update(0.1, 1)
        self.assertEqual(metric.result(), 0.1)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_sum(self):
        metric = Sum()
        self.assertEqual(metric.result(), 0)
        metric.update(5)
        self.assertEqual(metric.result(), 5)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_forgetting(self):
        metric = Forgetting()
        f = metric.result()
        self.assertEqual(f, {})
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 1, initial=True)
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 0.4)
        f = metric.result(k=0)
        self.assertEqual(f, 0.6)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_forward_transfer(self):
        metric = ForwardTransfer()
        f = metric.result()
        self.assertEqual(f, {})
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 1, initial=True)
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 0.4)
        f = metric.result(k=0)
        self.assertEqual(f, -0.6)
        metric.reset()
        self.assertEqual(metric.result(), {})


if __name__ == "__main__":
    unittest.main()
