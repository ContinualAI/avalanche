"""
This example shows how to use FFCV data loading system in Avalanche
when compressing RGB images is required.

FFCV allows for various tweaks to be used when manipulating images.
In particular, FFCV allows storing images as JPGs with custom
quality. In addition, the max side of the image and other custom
elements can be set.

This tutorial will show how to set these parameters.
"""

import argparse
from datetime import datetime
import time

import torch
import torch.optim.lr_scheduler
from torch.optim import Adam
from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.classic.ccifar100 import SplitCIFAR100
from avalanche.benchmarks.classic.ctiny_imagenet import SplitTinyImageNet
from avalanche.benchmarks.utils.ffcv_support import enable_ffcv
from avalanche.models import SimpleMLP
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import TensorboardLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(cuda: int):
    device = torch.device(
        f"cuda:{cuda}" if cuda >= 0 and torch.cuda.is_available() else "cpu"
    )
    RNGManager.set_random_seeds(1234)

    benchmark_type = "tinyimagenet"

    # --- BENCHMARK CREATION
    num_workers = 8
    if benchmark_type == "mnist":
        input_size = 28 * 28
        num_workers = 4
        benchmark = SplitMNIST(
            n_experiences=5, seed=42, class_ids_from_zero_from_first_exp=True
        )
    elif benchmark_type == "cifar100":
        benchmark = SplitCIFAR100(5, seed=1234, shuffle=True)
        input_size = 32 * 32 * 3
    elif benchmark_type == "tinyimagenet":
        benchmark = SplitTinyImageNet()
        input_size = 64 * 64 * 3
    else:
        raise RuntimeError("Unknown benchmark")

    # Enabling FFCV in Avalanche is as simple as calling `enable_ffcv`.
    # For additional info regarding on how this works, please refer
    # to the `ffcv_enable.py` example.
    # In this example, the focus is on the RGB encoder customization.
    #
    # `ffcv_parameters` is where we pass custom parameters for the RGB encoder
    # These parameters are listed in the FFCV website:
    # https://docs.ffcv.io/working_with_images.html
    # As an example, here we set parameters like
    # write_mode, compress_probability, and jpeg_quality
    #
    # Note: an alternative way to achieve this is to specify the encoder
    # dictionary directly by passing the `encoder_def` parameter.
    print("Enabling FFCV support...")
    print("The may include writing the datasets in FFCV format. May take some time...")
    enable_ffcv(
        benchmark=benchmark,
        write_dir=f"./ffcv_test_compress_{benchmark_type}",
        device=device,
        ffcv_parameters=dict(
            num_workers=8,
            write_mode="proportion",
            compress_probability=0.25,
            jpeg_quality=90,
        ),
    )
    print("FFCV enabled!")

    # -------------------- THAT'S IT!! ------------------------------
    # The rest of the script is an usual Avalanche code.
    #
    # In certain situations, you may want to pass some custom
    # parameters to the FFCV Loader. This can be achieved
    # when calling `train()` and `eval()` (see the main loop).
    # ---------------------------------------------------------------

    # MODEL CREATION
    model = SimpleMLP(input_size=input_size, num_classes=benchmark.n_classes)

    # METRICS
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True),
        loggers=[TensorboardLogger(f"tb_data/{datetime.now()}"), InteractiveLogger()],
    )

    # CREATE THE STRATEGY INSTANCE
    replay_plugin = ReplayPlugin(mem_size=100, batch_size=125, batch_size_mem=25)
    cl_strategy = Naive(
        model,
        Adam(model.parameters()),
        train_mb_size=128,
        train_epochs=4,
        eval_mb_size=128,
        device=device,
        plugins=[replay_plugin],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    # For FFCV, you can pass the Loader parameters using ffcv_args
    # Notice that some parameters like shuffle, num_workers, ...,
    # which are also found in the PyTorch DataLoader, can be passed
    # to train() and eval() as usual: they will be passed to the FFCV
    # Loader as they would be passed to the PyTorch Dataloader.
    #
    # In addition to the FFCV Loader parameters, you can pass the
    # print_ffcv_summary flag (which is managed by Avalanche),
    # which allows for printing the pipeline and the status of
    # internal checks made by Avalanche. That flag is very useful
    # when setting up an FFCV+Avalanche experiment. Once you are sure
    # that the code works as intended, it is better to remove it as
    # the logging may be quite verbose...
    start_time = time.time()
    for i, experience in enumerate(benchmark.train_stream):
        cl_strategy.train(
            experience,
            shuffle=False,
            persistent_workers=True,
            num_workers=num_workers,
            ffcv_args={"print_ffcv_summary": True, "batches_ahead": 2},
        )

        cl_strategy.eval(
            benchmark.test_stream[: i + 1],
            shuffle=False,
            num_workers=num_workers,
            ffcv_args={"print_ffcv_summary": True, "batches_ahead": 2},
        )
    end_time = time.time()
    print("Overall time:", end_time - start_time, "seconds")


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
