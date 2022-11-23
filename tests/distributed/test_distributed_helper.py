import random
import unittest

import torch
import torch.distributed as dst

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_helper import \
    RollingSeedContext, BroadcastSeedContext

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
