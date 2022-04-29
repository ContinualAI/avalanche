import contextlib
import os
import time
import unittest

import torch

from avalanche.distributed import DistributedHelper
from avalanche.distributed.strategies import DistributedMiniBatchStrategySupport


@contextlib.contextmanager
def manage_output():
    if os.environ['LOCAL_RANK'] != 0:
        with contextlib.redirect_stderr(None):
            with contextlib.redirect_stdout(None):
                yield
    else:
        yield


class DistributedStrategySupportTests(unittest.TestCase):

    def setUp(self) -> None:
        DistributedHelper.init_distributed(1234, use_cuda=False)

    @unittest.skipIf(int(os.environ.get('DISTRIBUTED_TESTS', 0)) != 1,
                     'Distributed tests ignored')
    def test_use_local_works(self):
        uut = DistributedMiniBatchStrategySupport()
        uut.mbatch = torch.full((5, 10), DistributedHelper.rank,
                                dtype=torch.float32)
        uut.mb_output = torch.full((5, 10), DistributedHelper.rank,
                                   dtype=torch.float32)

        # Test without use_local
        got_mbatch = uut.mbatch
        got_mb_output = uut.mb_output

        expected_shape = (DistributedHelper.world_size * 5, 10)

        self.assertSequenceEqual(expected_shape, got_mbatch.shape)
        self.assertSequenceEqual(expected_shape, got_mb_output.shape)

        for row_idx in range(expected_shape[0]):
            from_rank = row_idx // 5
            self.assertTrue(torch.equal(
                torch.full((10,), from_rank, dtype=torch.float32),
                got_mbatch[row_idx]))
            self.assertTrue(torch.equal(
                torch.full((10,), from_rank, dtype=torch.float32),
                got_mb_output[row_idx]))

        # Test with use_local
        uut.mbatch = torch.full((5, 10), DistributedHelper.rank,
                                dtype=torch.float32)
        uut.mb_output = torch.full((5, 10), DistributedHelper.rank,
                                   dtype=torch.float32)

        with uut.use_local():
            got_mbatch = uut.mbatch
            got_mb_output = uut.mb_output

        expected_shape = (5, 10)

        self.assertSequenceEqual(expected_shape, got_mbatch.shape)
        self.assertSequenceEqual(expected_shape, got_mb_output.shape)

        for row_idx in range(expected_shape[0]):
            from_rank = DistributedHelper.rank
            self.assertTrue(torch.equal(
                torch.full((10,), from_rank, dtype=torch.float32),
                got_mbatch[row_idx]))
            self.assertTrue(torch.equal(
                torch.full((10,), from_rank, dtype=torch.float32),
                got_mb_output[row_idx]))


if __name__ == "__main__":
    with manage_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
