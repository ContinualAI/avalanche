import random
import unittest

import torch
import torch.distributed as dst
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_helper import \
    RollingSeedContext, BroadcastSeedContext
from avalanche.models import SimpleMLP, as_multitask

from avalanche.training.determinism.rng_manager import RNGManager
from tests.distributed.distributed_test_utils import \
    check_skip_distributed_test, suppress_dst_tests_output, \
    common_dst_tests_setup


class DistributedHelperTests(unittest.TestCase):

    def setUp(self) -> None:
        self.use_gpu_in_tests = common_dst_tests_setup()

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_device_id(self):
        if self.use_gpu_in_tests:
            self.assertEqual(dst.get_rank(), DistributedHelper.get_device_id())
            self.assertEqual(torch.device(f'cuda:{dst.get_rank()}'),
                             DistributedHelper.make_device())
        else:
            self.assertEqual(-1, DistributedHelper.get_device_id())
            self.assertEqual(torch.device('cpu'),
                             DistributedHelper.make_device())

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_wrap_model(self):
        mb_size = 1*2*2*3*5
        num_classes = 11
        torch.manual_seed(1234 + DistributedHelper.rank)
        mb_x = torch.randn((mb_size, 32))
        model = SimpleMLP(num_classes=num_classes, input_size=32)
        self.assertIsInstance(model, Module)

        device = DistributedHelper.make_device()

        if device.type == 'cuda':
            # Additional test: must raise an error if the model 
            # is not already in the correct device
            with self.assertRaises(Exception):
                model_wrapped = DistributedHelper.wrap_model(model)

        model = model.to(device)

        model_wrapped = DistributedHelper.wrap_model(model)
        self.assertIsInstance(model_wrapped, DistributedDataParallel)
        self.assertNotIsInstance(model, DistributedDataParallel)

        device = DistributedHelper.make_device()
        mb_x = mb_x.to(device)
        model = model.to(device)

        model.eval()
        model_wrapped.eval()

        with torch.no_grad():
            mb_out1 = model(mb_x).detach()
            self.assertEqual(mb_out1.device, device)
            self.assertSequenceEqual([mb_size, num_classes], mb_out1.shape)

            mb_out2 = model_wrapped(mb_x).detach()
            self.assertEqual(mb_out2.device, device)
            self.assertSequenceEqual([mb_size, num_classes], mb_out2.shape)

            self.assertTrue(torch.equal(mb_out1, mb_out2))

            mb_out_all = DistributedHelper.cat_all(mb_out2)

            start_idx = mb_size * DistributedHelper.rank
            end_idx = start_idx + mb_size

            self.assertTrue(torch.equal(mb_out1, 
                                        mb_out_all[start_idx: end_idx]))

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_broadcast(self):
        ts = torch.full((10,), DistributedHelper.rank, dtype=torch.long)
        DistributedHelper.broadcast(ts)
        self.assertTrue(torch.equal(ts, torch.zeros((10,), dtype=torch.long)))

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_check_equal_tensors(self):
        torch.manual_seed(1234)
        ts = torch.randn((100,))
        DistributedHelper.check_equal_tensors(ts)

        torch.manual_seed(1234 + DistributedHelper.rank)
        ts = torch.randn((100,))
        with self.assertRaises(Exception):
            DistributedHelper.check_equal_tensors(ts)

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_fields(self):
        self.assertEqual(dst.get_rank(), DistributedHelper.rank)
        self.assertEqual(dst.get_world_size(), DistributedHelper.world_size)
        self.assertEqual(True, DistributedHelper.is_distributed)
        self.assertEqual(dst.get_rank() == 0, DistributedHelper.is_main_process)

        if self.use_gpu_in_tests:
            self.assertEqual('nccl', DistributedHelper.backend)
            self.assertTrue(DistributedHelper.forced_cuda_comm)
        else:
            self.assertEqual('gloo', DistributedHelper.backend)
            self.assertFalse(DistributedHelper.forced_cuda_comm)

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_rolling_seed_aligner(self):
        RNGManager.set_random_seeds(4321)

        with RollingSeedContext():
            RNGManager.set_random_seeds(1234 + DistributedHelper.rank)
            random.randint(0, 2 ** 64 - 1)

        final_value = random.randint(0, 2 ** 64 - 1)
        self.assertEqual(14732185405572191734, final_value)

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_broadcast_seed_aligner(self):
        RNGManager.set_random_seeds(4321)

        with BroadcastSeedContext():
            RNGManager.set_random_seeds(1234 + DistributedHelper.rank)
            random.randint(0, 2 ** 64 - 1)

        final_value = random.randint(0, 2 ** 64 - 1)
        self.assertEqual(15306775005444441373, final_value)


if __name__ == "__main__":
    with suppress_dst_tests_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
