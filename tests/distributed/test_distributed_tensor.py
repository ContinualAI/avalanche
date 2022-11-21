import contextlib
import os
import unittest

import torch

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_tensor import \
    DistributedMeanTensor


@contextlib.contextmanager
def manage_output():
    if os.environ['LOCAL_RANK'] != 0:
        with contextlib.redirect_stderr(None):
            with contextlib.redirect_stdout(None):
                yield
    else:
        yield


class DistributedTensorTests(unittest.TestCase):

    def setUp(self) -> None:
        DistributedHelper.init_distributed(1234, use_cuda=False)

    @unittest.skipIf(int(os.environ.get('DISTRIBUTED_TESTS', 0)) != 1,
                     'Distributed tests ignored')
    def test_one_element_tensor(self):
        dt = DistributedMeanTensor('dt', torch.zeros((1,), dtype=torch.float32))

        self.assertEqual(0.0, dt.local_value.float())
        self.assertEqual(0.0, dt.value.float())

        i = DistributedHelper.rank + 1

        dt.value = torch.full((1,), fill_value=i,
                              dtype=torch.float32)

        n = DistributedHelper.world_size
        expected = n * (n + 1) / 2

        self.assertEqual(i, float(dt.local_value))
        self.assertEqual(expected / n, float(dt.value))

    @unittest.skipIf(int(os.environ.get('DISTRIBUTED_TESTS', 0)) != 1,
                     'Distributed tests ignored')
    def test_one_element_tensor_random(self):
        dt = DistributedMeanTensor('dt', torch.zeros((1,), dtype=torch.float32))

        rnd_value = torch.randint(0, 100000, (10,), dtype=torch.float32)
        dt.value = rnd_value

        expected = torch.mean(rnd_value)

        self.assertTrue(torch.allclose(expected, torch.mean(dt.local_value)))
        self.assertTrue(torch.allclose(expected, dt.value))

    @unittest.skipIf(int(os.environ.get('DISTRIBUTED_TESTS', 0)) != 1,
                     'Distributed tests ignored')
    def test_unshaped_tensor(self):
        dt = DistributedMeanTensor('dt',
                                   torch.as_tensor(5, dtype=torch.float32))

        self.assertEqual(5.0, dt.local_value.float())
        self.assertEqual(5.0, dt.value.float())
        self.assertEqual(0, len(dt.local_value.shape))
        self.assertEqual(0, len(dt.value.shape))

        i = DistributedHelper.rank + 1

        dt.value = torch.as_tensor(i, dtype=torch.float32)

        n = DistributedHelper.world_size
        expected = n * (n + 1) / 2

        self.assertEqual(i, float(dt.local_value))
        self.assertEqual(expected / n, float(dt.value))
        self.assertEqual(0, len(dt.local_value.shape))
        self.assertEqual(0, len(dt.value.shape))


if __name__ == "__main__":
    with manage_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
