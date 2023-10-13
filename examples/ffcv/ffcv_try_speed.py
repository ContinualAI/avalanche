"""
This scripts can be used to measure the speed of the FFCV dataloader.

Note: this is not the correct way to use FFCV in Avalanche. For a proper
example, please refer to `ffcv_enable.py`. This script should be used
to measure speed only.
"""

import argparse
import time
from typing import Tuple

import torch
import torch.optim.lr_scheduler
from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.classic.ccifar100 import SplitCIFAR100
from avalanche.benchmarks.classic.core50 import CORe50
from avalanche.benchmarks.classic.ctiny_imagenet import SplitTinyImageNet
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.ffcv_support import (
    HybridFfcvLoader,
    enable_ffcv,
)
from avalanche.training.determinism.rng_manager import RNGManager

from ffcv.transforms import ToTensor

from torchvision.transforms import Compose, ToTensor, Normalize

from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    BatchSampler,
    SequentialSampler,
)
from tqdm import tqdm


def main(cuda: int):
    # --- CONFIG
    device = torch.device(
        f"cuda:{cuda}" if cuda >= 0 and torch.cuda.is_available() else "cpu"
    )
    RNGManager.set_random_seeds(1234)

    benchmark_type = "cifar100"

    # --- BENCHMARK CREATION
    if benchmark_type == "mnist":
        benchmark = SplitMNIST(
            n_experiences=5, seed=42, class_ids_from_zero_from_first_exp=True
        )
    elif benchmark_type == "core50":
        benchmark = CORe50()
        benchmark.n_classes = 50
    elif benchmark_type == "cifar100":
        cifar100_train_transform = Compose(
            [
                ToTensor(),
                Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
        )

        cifar100_eval_transform = Compose(
            [
                ToTensor(),
                Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
        )
        benchmark = SplitCIFAR100(
            5,
            seed=1234,
            shuffle=True,
            train_transform=cifar100_train_transform,
            eval_transform=cifar100_eval_transform,
        )
    elif benchmark_type == "tinyimagenet":
        benchmark = SplitTinyImageNet()
    else:
        raise RuntimeError("Unknown benchmark")

    # Note: when Numba uses TBB, then 20 is the limit number of workers
    # However, this limit does not apply when using OpenMP
    # (which may be faster...). If you want to test using OpenMP, then
    # run this script with the following command:
    # NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=32 python benchmark_ffcv.py
    for num_workers in [8, 16, 32]:
        print("num_workers =", num_workers)
        print("device =", device)
        benchmark_pytorch_speed(
            benchmark, device=device, num_workers=num_workers, epochs=4
        )
        benchmark_ffcv_speed(
            benchmark,
            f"./ffcv_test_{benchmark_type}",
            device=device,
            num_workers=num_workers,
            epochs=4,
        )


def benchmark_ffcv_speed(
    benchmark, path, device, batch_size=128, num_workers=1, epochs=1
):
    print("Testing FFCV Loader speed")

    all_train_dataset = [x.dataset for x in benchmark.train_stream]
    avl_set = AvalancheDataset(all_train_dataset)
    avl_set = avl_set.train()

    start_time = time.time()
    enable_ffcv(
        benchmark,
        path,
        device,
        dict(num_workers=num_workers),
        print_summary=False,  # Better keep this true on non-benchmarking code
    )
    end_time = time.time()
    print("FFCV preparation time:", end_time - start_time, "seconds")

    start_time = time.time()
    ffcv_loader = HybridFfcvLoader(
        dataset=avl_set,
        batch_sampler=BatchSampler(
            SequentialSampler(avl_set),
            batch_size=batch_size,
            drop_last=True,
        ),
        ffcv_loader_parameters=dict(num_workers=num_workers),
        device=device,
        print_ffcv_summary=False,
    )

    for _ in tqdm(range(epochs)):
        for batch in tqdm(ffcv_loader):
            # "Touch" tensors to make sure they already moved to GPU
            batch[0][0]
            batch[-1][0]

    end_time = time.time()
    print("FFCV time:", end_time - start_time, "seconds")


def benchmark_pytorch_speed(benchmark, device, batch_size=128, num_workers=1, epochs=1):
    print("Testing PyTorch Loader speed")

    all_train_dataset = [x.dataset for x in benchmark.train_stream]
    avl_set = AvalancheDataset(all_train_dataset)
    avl_set = avl_set.train()

    start_time = time.time()
    torch_loader = DataLoader(
        avl_set,
        batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        persistent_workers=True,
    )

    batch: Tuple[torch.Tensor]
    for _ in tqdm(range(epochs)):
        for batch in tqdm(torch_loader):
            batch = tuple(x.to(device, non_blocking=True) for x in batch)

            # "Touch" tensors to make sure they already moved to GPU
            batch[0][0]
            batch[-1][0]

    end_time = time.time()
    print("PyTorch time:", end_time - start_time, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args.cuda)
