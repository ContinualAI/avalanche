import hashlib
import unittest

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DistributedSampler, DataLoader

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_consistency_verification import hash_dataset
from avalanche.distributed.strategies import DistributedMiniBatchStrategySupport
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from tests.distributed.distributed_test_utils import check_skip_distributed_test, suppress_dst_tests_output, \
    common_dst_tests_setup
from tests.unit_tests_utils import get_fast_benchmark


class DistributedStrategySupportTests(unittest.TestCase):

    def setUp(self) -> None:
        self.use_gpu_in_tests = common_dst_tests_setup()

    @unittest.skipIf(check_skip_distributed_test(),
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

    def _check_loss_equal(self, uut):
        local_loss = uut.local_loss
        global_loss = uut.loss

        self.assertIsInstance(local_loss, Tensor)
        self.assertIsInstance(global_loss, Tensor)
        self.assertEqual(uut.device, local_loss.device)
        self.assertEqual(uut.device, global_loss.device)

        all_losses = DistributedHelper.gather_all_objects(float(local_loss))
        # Note: the results of torch.mean are different from the ones
        # of statistics.mean
        self.assertAlmostEqual(
            float(torch.mean(torch.as_tensor(all_losses))),
            float(global_loss))

    def _check_batches_equal(self, uut: Naive, rank: int, mb_size: int, mb_dist_size: int, input_size: int):
        local_input_mb = uut.local_mbatch
        global_input_mb = uut.mbatch

        self.assertEqual(3, len(local_input_mb))
        self.assertEqual(3, len(global_input_mb))

        for mb_i, mb_elem in enumerate(local_input_mb):
            self.assertIsInstance(mb_elem, Tensor)
            self.assertEqual(uut.device, mb_elem.device)

        for mb_i, mb_elem in enumerate(global_input_mb):
            self.assertIsInstance(mb_elem, Tensor)
            self.assertEqual(uut.device, mb_elem.device)

        self.assertTrue(torch.equal(global_input_mb[0], uut.mb_x))
        self.assertTrue(torch.equal(global_input_mb[1], uut.mb_y))
        self.assertTrue(torch.equal(global_input_mb[2], uut.mb_task_id))

        self.assertSequenceEqual(local_input_mb[0].shape, [mb_dist_size, input_size])
        self.assertSequenceEqual(local_input_mb[1].shape, [mb_dist_size])
        self.assertSequenceEqual(local_input_mb[2].shape, [mb_dist_size])

        self.assertSequenceEqual(global_input_mb[0].shape, [mb_size, input_size])
        self.assertSequenceEqual(global_input_mb[1].shape, [mb_size])
        self.assertSequenceEqual(global_input_mb[2].shape, [mb_size])

        global_index_start = mb_dist_size * rank
        global_index_end = global_index_start + mb_dist_size

        for i in range(3):
            self.assertTrue(torch.equal(local_input_mb[i], global_input_mb[i][global_index_start:global_index_end]))

    def _check_adapted_datasets_equal(self, uut: Naive):
        local_adapted_dataset = uut.adapted_dataset

        DistributedHelper.check_equal_objects(
            hash_dataset(local_adapted_dataset, num_workers=4, hash_engine=hashlib.sha1())
        )

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_naive_classification_dst(self):
        self.assertTrue(DistributedHelper.is_distributed)

        input_size = 28 * 28
        # mb_size == 60, so that it can be tested using [1, 6] parallel processes
        mb_size = 1*2*2*3*4*5
        model = SimpleMLP(input_size=input_size)
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = CrossEntropyLoss()
        device = DistributedHelper.make_device()

        # DST parameters adaptation
        mb_size_dst = mb_size // DistributedHelper.world_size

        uut = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=mb_size_dst,
            eval_mb_size=mb_size_dst,
            train_epochs=2,
            device=device
        )

        self.assertEqual(device, uut.device)

        if not DistributedHelper.is_main_process:
            self.assertEqual(0, len(uut.evaluator.loggers))

        benchmark = get_fast_benchmark(
            n_samples_per_class=400,
            n_features=input_size)

        for exp_idx, train_experience in enumerate(benchmark.train_stream):
            # TODO: insert checks between iterations
            metrics = uut.train(train_experience, drop_last=True)
            self._check_batches_equal(uut, DistributedHelper.rank, mb_size, mb_size_dst, input_size)
            self._check_loss_equal(uut)
            if exp_idx < 2:
                # Do it only for the first 2 experiences to speed up tests
                self._check_adapted_datasets_equal(uut)
            DistributedHelper.check_equal_objects(metrics)

            metrics = uut.eval(benchmark.test_stream, drop_last=True)
            self._check_batches_equal(uut, DistributedHelper.rank, mb_size, mb_size_dst, input_size)
            self._check_loss_equal(uut)
            if exp_idx < 2:
                # Do it only for the first 2 experiences to speed up tests
                self._check_adapted_datasets_equal(uut)
            DistributedHelper.check_equal_objects(metrics)

    @unittest.skipIf(check_skip_distributed_test(),
                     'Distributed tests ignored')
    def test_pytorch_distributed_sampler(self):
        """
        Only used to test the DistributedSampler class from PyTorch.
        """
        self.assertTrue(DistributedHelper.is_distributed)

        input_size = 28 * 28
        mb_size = 210  # Can be tested using [1, 10] parallel processes

        # DST parameters adaptation
        mb_size_dst = mb_size // DistributedHelper.world_size

        benchmark = get_fast_benchmark(
            n_samples_per_class=175 * 4,
            n_features=input_size)

        for train_experience in benchmark.train_stream:
            dataset = train_experience.dataset
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=True
            )
            dataloader = DataLoader(
                dataset,
                batch_size=mb_size_dst,
                sampler=sampler,
                drop_last=True
            )

            for mb_x, mb_y, mb_t in dataloader:
                self.assertSequenceEqual(mb_x.shape, [mb_size_dst, input_size])
                self.assertEqual(len(mb_y), mb_size_dst)


if __name__ == "__main__":
    with suppress_dst_tests_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
