import unittest
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import default_collate

from avalanche.distributed import DistributedHelper, \
    make_classification_distributed_batch, CollateDistributedBatch
from tests.distributed.distributed_test_utils import \
    check_skip_distributed_test, suppress_dst_tests_output, \
    common_dst_tests_setup


class DistributedBatchesTests(unittest.TestCase):

    def setUp(self) -> None:
        self.use_gpu_in_tests = common_dst_tests_setup()

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_classification_batch(self):
        dt = make_classification_distributed_batch('mb')

        self.assertEqual(None, dt.local_value)
        self.assertEqual(None, dt.value)

        batch = (torch.ones((8, 1, 28, 28)),
                 torch.full(
                     (8,), fill_value=DistributedHelper.rank, dtype=torch.long))

        dt.value = batch

        distrib_val = dt.value

        self.assertEqual(2, len(distrib_val))
        self.assertIsInstance(distrib_val, tuple)
        self.assertSequenceEqual((8*DistributedHelper.world_size, 1, 28, 28),
                                 distrib_val[0].shape)
        self.assertIsInstance(distrib_val[0], Tensor)
        self.assertIsInstance(distrib_val[1], Tensor)
        for rank in range(DistributedHelper.world_size):
            expect = torch.full((8,),
                                rank,
                                dtype=torch.long)
            self.assertTrue(torch.equal(expect,
                                        distrib_val[1][8*rank:8*(rank+1)]))

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_unsupervised_classification_batch(self):
        dt = make_classification_distributed_batch('mb')

        self.assertEqual(None, dt.local_value)
        self.assertEqual(None, dt.value)

        batch = torch.ones((8, 1, 28, 28))

        dt.value = batch

        distrib_val = dt.value

        self.assertIsInstance(distrib_val, Tensor)
        self.assertSequenceEqual((8*DistributedHelper.world_size, 1, 28, 28),
                                 distrib_val.shape)

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_tuple_merge_batch_vanilla_collate(self):
        dt: CollateDistributedBatch[Tuple[Tensor, Tensor]] = \
            CollateDistributedBatch(
                'mb',
                None,
                default_collate,
                None)

        self.assertEqual(None, dt.local_value)
        self.assertEqual(None, dt.value)

        batch = (torch.ones((8, 1, 28, 28)),
                 torch.full(
                     (8,), fill_value=DistributedHelper.rank, dtype=torch.long))

        dt.value = batch

        distrib_val = dt.value

        self.assertEqual(2, len(distrib_val))
        self.assertSequenceEqual((8 * DistributedHelper.world_size, 1, 28, 28),
                                 distrib_val[0].shape)
        for rank in range(DistributedHelper.world_size):
            expect = torch.full((8,),
                                rank,
                                dtype=torch.long)
            self.assertTrue(
                torch.equal(
                    expect,
                    distrib_val[1][8 * rank:8 * (rank + 1)]))


if __name__ == "__main__":
    with suppress_dst_tests_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
