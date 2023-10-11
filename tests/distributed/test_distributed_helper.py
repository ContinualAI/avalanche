import os
import random
import shutil
import tempfile
import time
import unittest
import numpy as np

import torch
import torch.distributed as dst
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from avalanche.benchmarks.scenarios.deprecated.generators import dataset_benchmark
from avalanche.benchmarks.utils.classification_dataset import (
    _make_taskaware_tensor_classification_dataset,
)

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_helper import (
    RollingSeedContext,
    BroadcastSeedContext,
)
from avalanche.models import SimpleMLP, as_multitask
from avalanche.models.utils import avalanche_model_adaptation

from avalanche.training.determinism.rng_manager import RNGManager
from tests.distributed.distributed_test_utils import (
    check_skip_distributed_slow_test,
    check_skip_distributed_test,
    suppress_dst_tests_output,
    common_dst_tests_setup,
)


class DistributedHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.use_gpu_in_tests = common_dst_tests_setup()

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_device_id(self):
        if self.use_gpu_in_tests:
            self.assertEqual(dst.get_rank(), DistributedHelper.get_device_id())
            self.assertEqual(
                torch.device(f"cuda:{dst.get_rank()}"), DistributedHelper.make_device()
            )
        else:
            self.assertEqual(-1, DistributedHelper.get_device_id())
            self.assertEqual(torch.device("cpu"), DistributedHelper.make_device())

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_wrap_model(self):
        mb_size = 1 * 2 * 2 * 3 * 5
        num_classes = 11
        torch.manual_seed(1234 + DistributedHelper.rank)
        mb_x = torch.randn((mb_size, 32))
        mb_y = torch.randint(0, num_classes, (mb_size,))
        mb_t = torch.full((mb_size,), 1)
        model = SimpleMLP(num_classes=num_classes, input_size=32)
        model = as_multitask(model, "classifier")
        self.assertIsInstance(model, Module)

        device = DistributedHelper.make_device()

        if device.type == "cuda":
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
        mb_y = mb_y.to(device)
        mb_t = mb_t.to(device)
        model = model.to(device)

        model.eval()
        model_wrapped.eval()

        benchmark = dataset_benchmark(
            [
                _make_taskaware_tensor_classification_dataset(
                    mb_x, mb_y, mb_t, task_labels=mb_t.tolist()
                )
            ],
            [
                _make_taskaware_tensor_classification_dataset(
                    mb_x, mb_y, mb_t, task_labels=mb_t.tolist()
                )
            ],
        )

        avalanche_model_adaptation(model, benchmark.train_stream[0])

        with torch.no_grad():
            mb_out1 = model(mb_x, mb_t).detach()
            self.assertEqual(mb_out1.device, device)
            self.assertSequenceEqual([mb_size, num_classes], mb_out1.shape)

            mb_out2 = model_wrapped(mb_x, mb_t).detach()
            self.assertEqual(mb_out2.device, device)
            self.assertSequenceEqual([mb_size, num_classes], mb_out2.shape)

            self.assertTrue(torch.equal(mb_out1, mb_out2))

            mb_out_all = DistributedHelper.cat_all(mb_out2)

            start_idx = mb_size * DistributedHelper.rank
            end_idx = start_idx + mb_size

            self.assertTrue(torch.equal(mb_out1, mb_out_all[start_idx:end_idx]))

        self.assertTrue(model is DistributedHelper.unwrap_model(model_wrapped))

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_broadcast_tensor_or_objects(self):
        ts = torch.full((10,), DistributedHelper.rank, dtype=torch.long)
        DistributedHelper.broadcast(ts)
        self.assertTrue(torch.equal(ts, torch.zeros((10,), dtype=torch.long)))

        device = DistributedHelper.make_device()
        ts = ts.to(device)

        my_object = {"a": DistributedHelper.rank, "b": ts}
        my_object_from_main = DistributedHelper.broadcast_object(my_object)

        expect = {"a": 0, "b": torch.full((10,), 0, dtype=torch.long).tolist()}

        self.assertEqual(device, my_object_from_main["b"].device)
        my_object_from_main["b"] = my_object_from_main["b"].tolist()
        self.assertEqual(expect, my_object_from_main)

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_gather_all_objects(self):
        ts = torch.full((10,), DistributedHelper.rank, dtype=torch.long)

        device = DistributedHelper.make_device()
        ts = ts.to(device)

        my_object = {"a": DistributedHelper.rank, "b": ts}
        all_objects = DistributedHelper.gather_all_objects(my_object)
        self.assertIsInstance(all_objects, list)
        self.assertEqual(DistributedHelper.world_size, len(all_objects))

        for rank in range(DistributedHelper.world_size):
            expect = {
                "a": rank,
                "b": torch.full((10,), rank, dtype=torch.long).tolist(),
            }

            self.assertEqual(device, all_objects[rank]["b"].device)
            all_objects[rank]["b"] = all_objects[rank]["b"].tolist()
            self.assertEqual(expect, all_objects[rank])

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_cat_all(self):
        if DistributedHelper.rank == 0:
            ts = torch.full((10 + 1, 5), DistributedHelper.rank, dtype=torch.long)
        else:
            ts = torch.full((10, 5), DistributedHelper.rank, dtype=torch.long)
        device = DistributedHelper.make_device()

        if device.type == "cuda":
            # Additional test: tensors do not need to be on the default device
            DistributedHelper.cat_all(ts)

        ts = ts.to(device)

        concatenated_tensor = DistributedHelper.cat_all(ts)

        self.assertEqual(device, concatenated_tensor.device)

        expect = torch.empty(
            (DistributedHelper.world_size * 10 + 1, 5), dtype=torch.long
        ).to(device)
        for rank in range(DistributedHelper.world_size):
            if rank == 0:
                expect[rank * 10 : (rank + 1) * 10 + 1] = rank
            else:
                expect[1 + rank * 10 : 1 + (rank + 1) * 10] = rank

        self.assertTrue(torch.equal(concatenated_tensor, expect))

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_gather_all_same_size(self):
        ts = torch.full((10, 5), DistributedHelper.rank, dtype=torch.long)
        device = DistributedHelper.make_device()

        if device.type == "cuda":
            # Additional test: tensors do not need to be on the default device
            DistributedHelper.gather_all(ts)

            # On the other hand, PyTorch all_gather requires tensors to be on
            # the default device
            with self.assertRaises(Exception):
                out_t = [
                    torch.empty_like(ts) for _ in range(DistributedHelper.world_size)
                ]
                torch.distributed.all_gather(out_t, ts)

            # ... while this should work
            out_t = [
                torch.empty_like(ts).to(device)
                for _ in range(DistributedHelper.world_size)
            ]
            torch.distributed.all_gather(out_t, ts.to(device))

        ts = ts.to(device)

        for same_shape in [False, True]:
            print(f"same_shape={same_shape}")
            # with self.subTest(same_shape=same_shape):
            tensor_list = DistributedHelper.gather_all(ts, same_shape=same_shape)

            self.assertEqual(DistributedHelper.world_size, len(tensor_list))

            for t in tensor_list:
                self.assertEqual(device, t.device)

            for rank in range(DistributedHelper.world_size):
                expect = torch.full((10, 5), rank, dtype=torch.long).to(device)
                self.assertTrue(torch.equal(tensor_list[rank], expect))

    @unittest.skipIf(check_skip_distributed_slow_test(), "Distributed tests ignored")
    def test_gather_all_performance_known_same_shape(self):
        ts = torch.full((128, 224, 224, 3), DistributedHelper.rank, dtype=torch.float32)
        device = DistributedHelper.make_device()
        ts = ts.to(device)

        resulting_tensors = [
            torch.empty_like(ts).to(device) for _ in range(DistributedHelper.world_size)
        ]

        from tqdm import tqdm

        n_times = 30
        torch.distributed.all_gather(resulting_tensors, ts)
        start_time = time.time()
        for _ in tqdm(range(n_times)):
            torch.distributed.all_gather(resulting_tensors, ts)
        end_time = time.time()
        print(
            "Time taken by PyTorch all_gather",
            end_time - start_time,
            "avg",
            (end_time - start_time) / n_times,
        )

        start_time = time.time()
        out_list = [None for _ in range(DistributedHelper.world_size)]
        torch.distributed.all_gather_object(out_list, ts)

        for _ in tqdm(range(n_times)):
            torch.distributed.all_gather_object(out_list, ts)
        end_time = time.time()
        print(
            "Time taken by PyTorch all_gather_object",
            end_time - start_time,
            "avg",
            (end_time - start_time) / n_times,
        )

    @unittest.skipIf(check_skip_distributed_slow_test(), "Distributed tests ignored")
    def test_gather_all_performance_sync_shape(self):
        max_shape_size = 10
        shape = [128, 6, DistributedHelper.rank + 1] + ([3] * DistributedHelper.rank)

        device = DistributedHelper.make_device()

        def shape_all_gather():
            ts = torch.zeros((max_shape_size,), dtype=torch.int64)
            for i in range(len(shape)):
                ts[i] = shape[i]

            ts = ts.to(device)
            all_tensors_shape = [
                torch.empty_like(ts) for _ in range(DistributedHelper.world_size)
            ]
            torch.distributed.all_gather(all_tensors_shape, ts)
            all_tensors_shape = [t.cpu() for t in all_tensors_shape]

            for i, t in enumerate(all_tensors_shape):
                for x in range(len(t)):
                    if t[x] == 0:
                        if x == 0:
                            # Tensor with 0-length shape
                            all_tensors_shape[i] = t[: x + 1]
                        else:
                            all_tensors_shape[i] = t[:x]
                        break

        def shape_all_gather_objects():
            out_list = [None for _ in range(DistributedHelper.world_size)]
            torch.distributed.all_gather_object(out_list, shape)

        from tqdm import tqdm

        n_times = 1000
        shape_all_gather()
        start_time = time.time()
        for _ in tqdm(range(n_times)):
            shape_all_gather()
        end_time = time.time()
        print(
            "Time taken by PyTorch all_gather",
            end_time - start_time,
            "avg",
            (end_time - start_time) / n_times,
        )

        start_time = time.time()
        shape_all_gather_objects()

        for _ in tqdm(range(n_times)):
            shape_all_gather_objects()
        end_time = time.time()
        print(
            "Time taken by PyTorch all_gather_object",
            end_time - start_time,
            "avg",
            (end_time - start_time) / n_times,
        )

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_gather_all_same_dim0(self):
        ts = torch.full(
            (10, DistributedHelper.rank + 1), DistributedHelper.rank, dtype=torch.long
        )
        device = DistributedHelper.make_device()

        ts = ts.to(device)

        tensor_list = DistributedHelper.gather_all(ts)
        self.assertEqual(DistributedHelper.world_size, len(tensor_list))

        for t in tensor_list:
            self.assertEqual(device, t.device)

        for rank in range(DistributedHelper.world_size):
            expect = torch.full((10, rank + 1), rank, dtype=torch.long).to(device)
            self.assertTrue(torch.equal(tensor_list[rank], expect))

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_gather_all_same_dim1_n(self):
        ts = torch.full(
            (10 + DistributedHelper.rank, 5), DistributedHelper.rank, dtype=torch.long
        )
        device = DistributedHelper.make_device()

        ts = ts.to(device)

        tensor_list = DistributedHelper.gather_all(ts)
        self.assertEqual(DistributedHelper.world_size, len(tensor_list))

        for t in tensor_list:
            self.assertEqual(device, t.device)

        for rank in range(DistributedHelper.world_size):
            expect = torch.full((10 + rank, 5), rank, dtype=torch.long).to(device)
            self.assertTrue(torch.equal(tensor_list[rank], expect))

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_gather_all_zero_shaped(self):
        ts = torch.full(tuple(), DistributedHelper.rank, dtype=torch.long)
        device = DistributedHelper.make_device()

        ts = ts.to(device)

        for same_shape in [False, True]:
            print(f"same_shape={same_shape}")
            # with self.subTest(same_shape=same_shape):
            tensor_list = DistributedHelper.gather_all(ts, same_shape=same_shape)
            self.assertEqual(DistributedHelper.world_size, len(tensor_list))

            for t in tensor_list:
                self.assertEqual(device, t.device)

            for rank in range(DistributedHelper.world_size):
                expect = torch.full(tuple(), rank, dtype=torch.long).to(device)
                self.assertTrue(torch.equal(tensor_list[rank], expect))

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_check_equal_tensors(self):
        if DistributedHelper.world_size == 1 and DistributedHelper.get_device_id() >= 0:
            self.skipTest(
                "When using CUDA, there must be at "
                "least two processes to run this test"
            )
        torch.manual_seed(1234)
        ts = torch.randn((100,))
        DistributedHelper.check_equal_tensors(ts)

        torch.manual_seed(1234 + DistributedHelper.rank)
        ts = torch.randn((100,))
        with self.assertRaises(Exception):
            DistributedHelper.check_equal_tensors(ts)

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_fields(self):
        self.assertEqual(dst.get_rank(), DistributedHelper.rank)
        self.assertEqual(dst.get_world_size(), DistributedHelper.world_size)
        self.assertEqual(True, DistributedHelper.is_distributed)
        self.assertEqual(dst.get_rank() == 0, DistributedHelper.is_main_process)

        if self.use_gpu_in_tests:
            self.assertEqual("nccl", DistributedHelper.backend)
            self.assertTrue(DistributedHelper.forced_cuda_comm)
        else:
            self.assertEqual("gloo", DistributedHelper.backend)
            self.assertFalse(DistributedHelper.forced_cuda_comm)

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_set_random_seeds_and_align(self):
        DistributedHelper.set_random_seeds(5678)

        self.assertEqual(297076, np.random.randint(0, 1000000))
        self.assertEqual(643380, torch.randint(0, 1000000, (1,)).item())
        self.assertEqual(683410, random.randint(0, 1000000))

        if DistributedHelper.is_main_process:
            np.random.randint(0, 1000000)
            torch.randint(0, 1000000, (1,))
            random.randint(0, 1000000)

        DistributedHelper.align_seeds()

        ref_values = (
            int(np.random.randint(0, 1000000)),
            int(torch.randint(0, 1000000, (1,))),
            int(random.randint(0, 1000000)),
        )

        DistributedHelper.check_equal_objects(ref_values)

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_rolling_seed_aligner(self):
        RNGManager.set_random_seeds(4321)

        with RollingSeedContext():
            RNGManager.set_random_seeds(1234 + DistributedHelper.rank)
            random.randint(0, 2**64 - 1)

        final_value = random.randint(0, 2**64 - 1)
        self.assertEqual(14732185405572191734, final_value)

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_broadcast_seed_aligner(self):
        RNGManager.set_random_seeds(4321)

        with BroadcastSeedContext():
            RNGManager.set_random_seeds(1234 + DistributedHelper.rank)
            random.randint(0, 2**64 - 1)

        final_value = random.randint(0, 2**64 - 1)
        self.assertEqual(15306775005444441373, final_value)

    @unittest.skipIf(check_skip_distributed_test(), "Distributed tests ignored")
    def test_main_process_first(self):
        tmpdirname = ""
        try:
            my_rank = DistributedHelper.rank
            if DistributedHelper.is_main_process:
                tmpdirname = tempfile.mkdtemp()

            tmpdirname = DistributedHelper.broadcast_object(tmpdirname)

            with DistributedHelper.main_process_first():
                for _ in range(2):
                    time.sleep(0.1 + my_rank * 0.05)
                    files = list(os.listdir(tmpdirname))
                    if DistributedHelper.is_main_process:
                        self.assertEqual(0, len(files))
                    else:
                        self.assertIn(f"rank0", files)
                        self.assertNotIn(f"rank{my_rank}", files)

                with open(os.path.join(tmpdirname, f"rank{my_rank}"), "w") as f:
                    f.write("ok")

                for _ in range(2):
                    time.sleep(0.1 + my_rank * 0.05)
                    files = list(os.listdir(tmpdirname))
                    if DistributedHelper.is_main_process:
                        self.assertEqual(1, len(files))
                        self.assertIn(f"rank0", files)
                    else:
                        self.assertIn(f"rank0", files)
                        self.assertIn(f"rank{my_rank}", files)

            DistributedHelper.barrier()
            files = set(os.listdir(tmpdirname))
            expect = set([f"rank{rnk}" for rnk in range(DistributedHelper.world_size)])
            self.assertSetEqual(expect, files)
            DistributedHelper.barrier()
        finally:
            if tmpdirname is not None and DistributedHelper.is_main_process:
                shutil.rmtree(tmpdirname)


if __name__ == "__main__":
    with suppress_dst_tests_output():
        verbosity = 1
        if DistributedHelper.rank > 0:
            verbosity = 0
        unittest.main(verbosity=verbosity)
