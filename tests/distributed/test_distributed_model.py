import unittest

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from avalanche.distributed import DistributedHelper, DistributedModel
from avalanche.models import SimpleMLP
from avalanche.models.helper_method import as_multitask
from avalanche.models.utils import avalanche_forward, avalanche_model_adaptation
from tests.distributed.distributed_test_utils import \
    check_skip_distributed_test, suppress_dst_tests_output, \
    common_dst_tests_setup
from tests.unit_tests_utils import get_fast_benchmark


class DistributedModelTests(unittest.TestCase):

    def setUp(self) -> None:
        self.use_gpu_in_tests = common_dst_tests_setup()

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_distributed_model(self):
        dt: DistributedModel = DistributedModel()
        model = SimpleMLP()
        self.assertIsNone(dt.local_value)
        self.assertIsNone(dt.value)
        self.assertIsNone(dt.distributed_value)

        device = DistributedHelper.make_device()

        dt.model = model

        self.assertEqual(model, dt.local_value)
        self.assertEqual(model, dt.value)
        self.assertEqual(model, dt.distributed_value)

        if device.type == 'cuda':
            # Additional test: must raise an error if the model 
            # is not already in the correct device
            with self.assertRaises(Exception):
                wrapped = DistributedDataParallel(
                    model, 
                    device_ids=[device])
        
        model = model.to(device)
        wrapped = DistributedDataParallel(
                    model, 
                    device_ids=[device])

        dt.model = wrapped

        self.assertEqual(model, dt.local_value)
        self.assertNotIsInstance(dt.local_value, DistributedDataParallel)

        self.assertIsInstance(dt.value, DistributedDataParallel)
        self.assertEqual(wrapped, dt.value)
        self.assertEqual(wrapped, dt.distributed_value)

        dt.reset_distributed_value()

        self.assertEqual(model, dt.local_value)
        self.assertEqual(model, dt.value)
        self.assertEqual(model, dt.distributed_value)

        self.assertNotIsInstance(dt.value, DistributedDataParallel)

        dt.reset_distributed_value()
        self.assertIsNotNone(dt.local_value)

        dt.value = wrapped
        dt.distributed_model = None

        self.assertIsNotNone(dt.local_value)

        dt.value = None

        self.assertIsNone(dt.local_value)
        self.assertIsNone(dt.distributed_value)
        self.assertIsNone(dt.value)

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_distributed_model_multitask(self):
        dt: DistributedModel = DistributedModel()
        model = SimpleMLP()
        model = as_multitask(model, 'classifier')
        self.assertIsNone(dt.local_value)
        self.assertIsNone(dt.value)
        self.assertIsNone(dt.distributed_value)

        device = DistributedHelper.make_device()

        dt.model = model

        self.assertEqual(model, dt.local_value)
        self.assertEqual(model, dt.value)
        self.assertEqual(model, dt.distributed_value)

        if device.type == 'cuda':
            # Additional test: must raise an error if the model 
            # is not already in the correct device
            with self.assertRaises(Exception):
                wrapped = DistributedDataParallel(
                    model, 
                    device_ids=[device])
        
        model = model.to(device)
        wrapped = DistributedDataParallel(
                    model, 
                    device_ids=[device])

        dt.model = wrapped

        self.assertEqual(model, dt.local_value)
        self.assertNotIsInstance(dt.local_value, DistributedDataParallel)

        self.assertIsInstance(dt.value, DistributedDataParallel)
        self.assertEqual(wrapped, dt.value)
        self.assertEqual(wrapped, dt.distributed_value)

        dt.reset_distributed_value()

        self.assertEqual(model, dt.local_value)
        self.assertEqual(model, dt.value)
        self.assertEqual(model, dt.distributed_value)

        self.assertNotIsInstance(dt.value, DistributedDataParallel)

        dt.reset_distributed_value()
        self.assertIsNotNone(dt.local_value)

        dt.value = wrapped
        dt.distributed_model = None

        self.assertIsNotNone(dt.local_value)

        dt.value = None

        self.assertIsNone(dt.local_value)
        self.assertIsNone(dt.distributed_value)
        self.assertIsNone(dt.value)

        # test model adaptation
        input_size = 28 * 28 * 1
        scenario = get_fast_benchmark(
            use_task_labels=True,
            n_features=input_size,
            n_samples_per_class=256,
            seed=1337
        )
        avalanche_model_adaptation(model, scenario.train_stream[1])
        model.eval()
        dt.value = model
        
        wrapped = DistributedDataParallel(model, device_ids=[device])
        dt.model = wrapped

        self.assertEqual(model, dt.local_value)
        loader = DataLoader(scenario.train_stream[1].dataset, batch_size=32)
        with torch.no_grad():
            for x, y, t in loader:
                x = x.to(device)
                y = y.to(device)
                t = t.to(device)
                self.assertEqual([1] * len(t), t.tolist())
                out_mb = avalanche_forward(dt.model, x, t)
                DistributedHelper.check_equal_tensors(out_mb)
                out_mb_local = avalanche_forward(dt.local_value, x, t)
                DistributedHelper.check_equal_tensors(out_mb_local)
                self.assertTrue(torch.equal(out_mb, out_mb_local))


if __name__ == "__main__":
    with suppress_dst_tests_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
