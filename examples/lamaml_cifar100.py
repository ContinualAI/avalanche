import argparse
import torch
import wandb
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
from avalanche.benchmarks.classic import SplitTinyImageNet
from avalanche.models import MTSimpleCNN
from avalanche.training.supervised.lamaml import LaMAML
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- BENCHMARK CREATION
    benchmark = SplitTinyImageNet(
        n_experiences=20,
        return_task_id=True,
        class_ids_from_zero_in_each_exp=True,
    )
    config = {"scenario": "SplitTinyImageNet"}

    # MODEL CREATION
    model = MTSimpleCNN()

    # choose some metrics and evaluation method
    loggers = [InteractiveLogger()]
    if args.wandb_project != "":
        wandb_logger = WandBLogger(
            project_name=args.wandb_project,
            run_name="LaMAML_" + config["scenario"],
            config=config,
        )
        loggers.append(wandb_logger)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=loggers,
    )

    # LAMAML STRATEGY
    rs_buffer = ReservoirSamplingBuffer(max_size=200)
    replay_plugin = ReplayPlugin(
        mem_size=200,
        batch_size=10,
        batch_size_mem=10,
        task_balanced_dataloader=False,
        storage_policy=rs_buffer,
    )

    cl_strategy = LaMAML(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        CrossEntropyLoss(),
        n_inner_updates=5,
        second_order=True,
        grad_clip_norm=1.0,
        learn_lr=True,
        lr_alpha=0.25,
        sync_update=False,
        train_mb_size=10,
        train_epochs=10,
        eval_mb_size=100,
        device=device,
        plugins=[replay_plugin],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))

    if args.wandb_project != "":
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Define the name of the WandB project",
    )
    args = parser.parse_args()
    main(args)
