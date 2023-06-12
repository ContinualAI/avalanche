"""
This is a simple example on how to use the Replay strategy in an online
benchmark created using OnlineCLScenario.
"""

import random
from types import SimpleNamespace
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import numpy as np

from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
from avalanche.models import SlimResNet18
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins import RARPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args

def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

def main(override_args=None):
    args = create_default_args(
        {
            "cuda": 1,
            "mem_size": 200,
            "lr": 0.1,
            "train_mb_size": 10,
            "seed": 14,
            "batch_size_mem": 10,
        },
        override_args
    )
    set_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    use_cifar10 = True

    if use_cifar10:
        scenario = SplitCIFAR10(
            5,
            return_task_id=False,
            seed=args.seed,
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
            fixed_class_order=np.arange(10),
        )
        num_clss = 10
        beta_coef = 0.5
        epsilon_fgsm = 0.03
    else:
        scenario = SplitCIFAR100(
            20,
            return_task_id=False,
            seed=args.seed,
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
            fixed_class_order=np.arange(100),
        )
        num_clss = 100
        beta_coef = 0.4
        epsilon_fgsm = 0.0314
    # ---------

    # MODEL CREATION
    model = SlimResNet18(num_clss, 5)
    optimizer = SGD(model.parameters(), lr=args.lr)

    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]

    training_metrics = []

    evaluation_metrics = [
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True, stream=True),
        # forgetting_metrics(stream=True),
    ]

    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )

    # CREATE THE STRATEGY INSTANCE (ONLINE-REPLAY)
    storage_policy = ReservoirSamplingBuffer(max_size=args.mem_size)
    # replay_plugin = ReplayPlugin(
    #     mem_size=args.mem_size, 
    #     # batch_size=args.batch_size_mem, 
    #     storage_policy=storage_policy
    # )
    replay_plugin = RARPlugin(
        batch_size_mem=args.batch_size_mem,
        mem_size=args.mem_size,
        storage_policy=storage_policy,
        use_adversarial_replay=True,
        use_mixup=True,
        beta_coef=beta_coef,
        epsilon_fgsm=epsilon_fgsm  
    )

    cl_strategy = Naive(
        model,
        optimizer,
        CrossEntropyLoss(),
        train_mb_size=args.train_mb_size,
        eval_mb_size=64,
        device=device,
        evaluator=evaluator,
        plugins=[replay_plugin],
    )

    # TRAINING LOOP
    print("Starting experiment...")
    batch_streams = scenario.streams.values()
    for i, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=args.train_mb_size,
            access_task_boundaries=False,
        )

        cl_strategy.classes_in_this_experience = experience.classes_in_this_experience

        cl_strategy.train(
            ocl_scenario.train_stream,
            # experience,
            eval_streams=[],
            shuffle=True,
        )
        cl_strategy.eval(scenario.test_stream[: i + 1])

if __name__ == "__main__":
    main()
