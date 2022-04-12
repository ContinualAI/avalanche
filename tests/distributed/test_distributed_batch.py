import contextlib
import os
import unittest
from typing import Tuple, Optional

import torch
from torch import Tensor

from avalanche.distributed import DistributedHelper, ClassificationBatch


@contextlib.contextmanager
def manage_output():
    if os.environ['LOCAL_RANK'] != 0:
        with contextlib.redirect_stderr(None):
            with contextlib.redirect_stdout(None):
                yield
    else:
        yield


class DistributedBatchesTests(unittest.TestCase):

    def setUp(self) -> None:
        DistributedHelper.init_distributed(1234, use_cuda=False)

    @unittest.skipIf(int(os.environ.get('DISTRIBUTED_TESTS', 0)) != 1,
                     'Distributed tests ignored')
    def test_classification_batch(self):
        dt: ClassificationBatch[Optional[Tuple[Tensor, Tensor]]] = \
            ClassificationBatch('mb', None)

        self.assertEqual(None, dt.local_value)
        self.assertEqual(None, dt.value)

        batch = (torch.ones((8, 1, 28, 28)),
                 torch.full(
                     (8,), fill_value=DistributedHelper.rank, dtype=torch.long))

        dt.value = batch

        distrib_val = dt.value

        self.assertEqual(2, len(distrib_val))
        self.assertSequenceEqual((8*DistributedHelper.world_size, 1, 28, 28),
                                 distrib_val[0].shape)
        for rank in range(DistributedHelper.world_size):
            expect = torch.full((8,),
                                rank,
                                dtype=torch.long)
            self.assertTrue(torch.equal(expect,
                                        distrib_val[1][8*rank:8*(rank+1)]))

    @unittest.skipIf(int(os.environ.get('DISTRIBUTED_TESTS', 0)) != 1,
                     'Distributed tests ignored')
    def test_unsupervised_classification_batch(self):
        dt: ClassificationBatch[Optional[Tuple[Tensor, Tensor]]] = \
            ClassificationBatch('mb', None)

        self.assertEqual(None, dt.local_value)
        self.assertEqual(None, dt.value)

        batch = torch.ones((8, 1, 28, 28))

        dt.value = batch

        distrib_val = dt.value

        self.assertIsInstance(distrib_val, Tensor)
        self.assertSequenceEqual((8*DistributedHelper.world_size, 1, 28, 28),
                                 distrib_val.shape)


if __name__ == "__main__":
    with manage_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
