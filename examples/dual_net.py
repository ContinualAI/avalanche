import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location

from avalanche.training.supervised.dual_net import DualNet
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.models.dualnet_model import MaskNet18
from avalanche.benchmarks.datasets.external_datasets.cifar import \
    get_cifar100_dataset


def main(args):
    # Compute device
    device = torch.device(args.device)
    print("Computing device: ", device)
    # Hyperparaeameters
    lr_fast = 0.1
    lr_slow = 3e-4
    mem_size = 1000
    img_size = 32

    cifar_train, cifar_test = get_cifar100_dataset(
        default_dataset_location("cifar100"))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ]
    )
    benchmark = nc_benchmark(
        train_dataset=cifar_train,
        test_dataset=cifar_test,
        n_experiences=20, 
        task_labels=False,
        seed=1234,
        train_transform=transform,
        eval_transform=transform,
    )

    # Model
    nf = 64  # Depends on the dataset
    model = MaskNet18(benchmark.n_classes, nf=nf)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # Strategy
    # Optimizer for the fast updates
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_fast)
    # Optimizer for the slow updates
    optimizer_bt = torch.optim.SGD(model.parameters(), lr=lr_slow)

    cl_strategy = DualNet(
        model,
        optimizer,
        optimizer_bt,
        CrossEntropyLoss(),
        inner_steps=2,
        n_outer=3,
        batch_size_mem=10,
        mem_size=mem_size,
        memory_strength=10.0,
        temperature=2.0,
        beta=0.05,
        task_agnostic_fast_updates=True,
        img_size=img_size,
        train_epochs=1,
        train_mb_size=10,
        eval_mb_size=32,
        device=device,
        evaluator=eval_plugin,
    )

    # Strat training on stream experiecnes
    print("Starting experiment...")
    results = []

    # Benchmark streams
    batch_streams = benchmark.streams.values()

    for i, exp in enumerate(benchmark.train_stream):
        # Create online stream from each experience exp
        ocl_benchmark = OnlineCLScenario(original_streams=batch_streams,
                                         experiences=exp,
                                         experience_size=10,
                                         access_task_boundaries=False)

        # Train on the online train stream of the scenario
        cl_strategy.train(ocl_benchmark.train_stream)

        results.append(cl_strategy.eval(ocl_benchmark.original_test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
    )
    args = parser.parse_args()
    main(args)
