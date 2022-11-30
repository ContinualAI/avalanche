import os
import sys
import unittest

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.logging import TextLogger
from avalanche.models import SimpleMLP
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.training import OnlineNaive
from tests.unit_tests_utils import get_fast_benchmark
from avalanche.training.plugins.evaluation import default_evaluator


class StrategyTest(unittest.TestCase):
    if "FAST_TEST" in os.environ:
        fast_test = os.environ["FAST_TEST"].lower() in ["true"]
    else:
        fast_test = False
    if "USE_GPU" in os.environ:
        use_gpu = os.environ["USE_GPU"].lower() in ["true"]
    else:
        use_gpu = False

    print("Fast Test:", fast_test)
    print("Test on GPU:", use_gpu)

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    def init_sit(self):
        model = self.get_model(fast_test=True)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = self.load_benchmark(use_task_labels=False)
        benchmark_streams = benchmark.streams.values()
        ocl_benchmark = OnlineCLScenario(benchmark_streams)
        return model, optimizer, criterion, ocl_benchmark

    def test_naive(self):
        benchmark = self.load_benchmark(use_task_labels=False)
        benchmark_streams = benchmark.streams.values()

        # With task boundaries
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = OnlineNaive(
            model,
            optimizer,
            criterion,
            train_mb_size=1,
            device=self.device,
            eval_mb_size=50,
            evaluator=default_evaluator(),
        )
        ocl_benchmark = OnlineCLScenario(benchmark_streams,
                                         access_task_boundaries=True)
        self.run_strategy_boundaries(ocl_benchmark, strategy)

        # Without task boundaries
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = OnlineNaive(
            model,
            optimizer,
            criterion,
            train_mb_size=1,
            device=self.device,
            eval_mb_size=50,
            evaluator=default_evaluator(),
        )
        ocl_benchmark = OnlineCLScenario(benchmark_streams,
                                         access_task_boundaries=False)
        self.run_strategy_no_boundaries(ocl_benchmark, strategy)

    def load_benchmark(self, use_task_labels=False):
        """
        Returns a NC benchmark from a fake dataset of 10 classes, 5 experiences,
        2 classes per experience.

        :param fast_test: if True loads fake data, MNIST otherwise.
        """
        return get_fast_benchmark(use_task_labels=use_task_labels)

    def get_model(self, fast_test=False):
        if fast_test:
            model = SimpleMLP(input_size=6, hidden_size=10)
            # model.classifier = IncrementalClassifier(
            #     model.classifier.in_features)
            return model
        else:
            model = SimpleMLP()
            # model.classifier = IncrementalClassifier(
            #     model.classifier.in_features)
            return model

    def run_strategy_boundaries(self, benchmark, cl_strategy):
        print("Starting experiment (with boundaries) ...")
        cl_strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        results = []
        for train_batch_info in benchmark.train_stream:
            print("Start of experience ", train_batch_info.current_experience)

            cl_strategy.train(train_batch_info, num_workers=0)
            print("Training completed")

            print("Computing accuracy on the current test set")
            results.append(cl_strategy.eval(benchmark.original_test_stream[:]))

    def run_strategy_no_boundaries(self, benchmark, cl_strategy):
        print("Starting experiment (without boundaries) ...")
        cl_strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        results = []

        cl_strategy.train(benchmark.train_stream, num_workers=0)
        print("Training completed")

        assert cl_strategy.clock.train_exp_counter > 0

        print("Computing accuracy on the current test set")
        results.append(cl_strategy.eval(benchmark.original_test_stream[:]))


if __name__ == "__main__":
    unittest.main()
