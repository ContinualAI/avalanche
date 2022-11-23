import unittest

from torch.nn.parallel import DistributedDataParallel

from avalanche.distributed import DistributedHelper, DistributedModel
from avalanche.models import SimpleMLP
from tests.distributed.distributed_test_utils import check_skip_distributed_test, suppress_dst_tests_output, \
    common_dst_tests_setup


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

        dt.model = model

        self.assertEqual(model, dt.local_value)
        self.assertEqual(model, dt.value)
        self.assertEqual(model, dt.distributed_value)

        wrapped = DistributedDataParallel(model)

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


if __name__ == "__main__":
    with suppress_dst_tests_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
