import sys
from avalanche.logging.text_logging import TextLogger


def run_strategy(benchmark, cl_strategy):
    print("Starting experiment...")
    cl_strategy.evaluator.loggers = [TextLogger(sys.stdout)]
    results = []
    for exp_idx, train_batch_info in enumerate(benchmark.train_stream):
        print("Start of experience ", exp_idx)

        cl_strategy.train(train_batch_info, num_workers=0)
        print("Training completed")

        print("Computing accuracy on the current test set")
        results.append(cl_strategy.eval(benchmark.test_stream[:]))


__all__ = ["run_strategy"]
