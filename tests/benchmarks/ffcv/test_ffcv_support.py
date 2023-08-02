import os
import random
import tempfile
import unittest
import torch
from torch.utils.data.sampler import (
    BatchSampler,
    SubsetRandomSampler,
    SequentialSampler,
)
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.classic.cmnist import SplitMNIST
from avalanche.benchmarks.utils.data_loader import MultiDatasetSampler
from avalanche.benchmarks.utils import AvalancheDataset, DataAttribute
from torchvision.transforms import Normalize

try:
    import ffcv

    skip = False
except ImportError:
    skip = True


class FFCVSupportTests(unittest.TestCase):
    @unittest.skipIf(skip, reason="Need ffcv to run these tests")
    def test_simple_scenario(self):
        from avalanche.benchmarks.utils.ffcv_support.ffcv_components import (
            enable_ffcv,
            HybridFfcvLoader,
        )

        train_transform = Normalize((0.1307,), (0.3081,))

        eval_transform = Normalize((0.1307,), (0.3081,))

        use_gpu = str(os.environ["USE_GPU"]).lower() in ["true", "1"]

        if use_gpu:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        benchmark = SplitMNIST(
            5,
            seed=4321,
            shuffle=True,
            return_task_id=True,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

        with tempfile.TemporaryDirectory() as write_dir:
            num_workers = 4

            enable_ffcv(
                benchmark=benchmark,
                write_dir=write_dir,
                device=device,
                ffcv_parameters=dict(num_workers=num_workers),
                print_summary=False,
            )

            dataset_0 = benchmark.train_stream[0].dataset
            dataset_1 = benchmark.train_stream[1].dataset

            subset_indices = list(range(0, len(dataset_0), 5))
            random.shuffle(subset_indices)

            generator_0_a = torch.Generator()
            generator_0_a.manual_seed(2147483647)

            generator_0_b = torch.Generator()
            generator_0_b.manual_seed(2147483647)

            sampler_0_a = BatchSampler(
                SubsetRandomSampler(subset_indices, generator_0_a),
                batch_size=12,
                drop_last=True,
            )

            sampler_0_b = BatchSampler(
                SubsetRandomSampler(subset_indices, generator_0_b),
                batch_size=12,
                drop_last=True,
            )

            sampler_0_a_lst = list(sampler_0_a)
            sampler_0_b_lst = list(sampler_0_b)
            self.assertEqual(sampler_0_a_lst, sampler_0_b_lst)

            sampler_1 = BatchSampler(
                SequentialSampler(dataset_1), batch_size=123, drop_last=False
            )

            batch_sampler_a = MultiDatasetSampler(
                [dataset_0, dataset_1],
                [sampler_0_a, sampler_1],
                oversample_small_datasets=True,
            )

            batch_sampler_b = MultiDatasetSampler(
                [dataset_0, dataset_1],
                [sampler_0_b, sampler_1],
                oversample_small_datasets=True,
            )

            batch_sampler_a_lst = list(batch_sampler_a)
            batch_sampler_b_lst = list(batch_sampler_b)
            self.assertEqual(batch_sampler_a_lst, batch_sampler_b_lst)

            sum_len = len(dataset_0) + len(dataset_1)
            concat_dataset = AvalancheDataset(
                [dataset_0, dataset_1],
                data_attributes=[
                    DataAttribute(
                        list(range(sum_len)), "custom_attr", use_in_getitem=True
                    )
                ],
            )

            ffcv_data_loader = HybridFfcvLoader(
                concat_dataset,
                batch_sampler=batch_sampler_a,
                ffcv_loader_parameters=dict(num_workers=num_workers, drop_last=False),
                device=device,
                persistent_workers=False,
                print_ffcv_summary=False,
                start_immediately=False,
            )

            pytorch_loader = DataLoader(concat_dataset, batch_sampler=batch_sampler_b)

            self.assertEqual(len(ffcv_data_loader), len(pytorch_loader))

            for i, (ffcv_batch, torch_batch) in enumerate(
                zip(ffcv_data_loader, pytorch_loader)
            ):
                for f, t in zip(ffcv_batch, torch_batch):
                    self.assertEqual(f.device, device)
                    f = f.cpu()
                    t = t.cpu()

                    if f.dtype.is_floating_point:
                        self.assertTrue(torch.sum(torch.abs(f - t) > 1e-6).item() == 0)
                    else:
                        self.assertTrue(torch.equal(f, t))


if __name__ == "__main__":
    unittest.main()
