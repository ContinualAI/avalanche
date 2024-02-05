"""
Use this script to evaluate serialization cost.
Times on my cheap desktop (April 2023):
SAVING TIME:  0.019003629684448242
LOADING TIME:  0.016000032424926758
"""

import os
import time
from torch.optim import SGD

from avalanche.benchmarks import SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.determinism import RNGManager
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.serialization import maybe_load_checkpoint, save_checkpoint

if __name__ == "__main__":
    device = "cpu"
    RNGManager.set_random_seeds(42)
    fname = "./checkpoint.pkl"

    benchmark = SplitCIFAR100(50)
    evaluator = EvaluationPlugin(
        accuracy_metrics(experience=True), loggers=[InteractiveLogger()]
    )

    model = SimpleMLP(input_size=32 * 32 * 3, num_classes=benchmark.n_classes)
    opt = SGD(model.parameters(), lr=0.1)
    strat = Naive(model, opt, train_mb_size=128)

    if os.path.exists(fname):
        os.remove(fname)

    for exp in benchmark.train_stream:
        start_time = time.time()
        cl_strategy, initial_exp = maybe_load_checkpoint(strat, fname)
        print("LOADING TIME: ", time.time() - start_time)

        strat.train(exp)
        strat.eval(exp)

        start_time = time.time()
        save_checkpoint(
            strat,
            fname,
            exclude=[
                # 'optimizer',
                # These attributes do not have state. Do not save.
                # They are automatically set to None by the strategy templates
                # If not, there is a bug...
                # 'experience',
                # 'adapted_dataset',
                # 'dataloader',
                # 'mbatch',
                # 'mb_output',
                # 'current_eval_stream',
                # '_eval_streams'
            ],
        )
        print("SAVING TIME: ", time.time() - start_time)
