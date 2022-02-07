import torch
from os.path import expanduser
import argparse
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from avalanche.benchmarks import PermutedMNIST, nc_benchmark
from avalanche.training.supervised import EWC
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    bwt_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

"""
This example tests EWC on Split MNIST and Permuted MNIST.
It is possible to choose, among other options, between EWC with separate
penalties and online EWC with a single penalty.

On Permuted MNIST EWC maintains a very good performance on previous tasks
with a wide range of configurations. The average accuracy on previous tasks
at the end of training on all task is around 85%,
with a comparable training accuracy.

On Split MNIST, on the contrary, EWC is not able to remember previous tasks and
is subjected to complete forgetting in all configurations. The training accuracy
is above 90% but the average accuracy on previou tasks is around 20%.
"""


def main(args):
    model = SimpleMLP(hidden_size=args.hs)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print(f"Using device: {device}")

    # create scenario
    if args.scenario == "pmnist":
        scenario = PermutedMNIST(n_experiences=args.permutations)
    elif args.scenario == "smnist":
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        scenario = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False, seed=1234
        )
    else:
        raise ValueError("Wrong scenario name. Allowed pmnist, smnist.")

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    tensorboard_logger = TensorboardLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[interactive_logger, tensorboard_logger],
    )

    # create strategy
    strategy = EWC(
        model,
        optimizer,
        criterion,
        args.ewc_lambda,
        args.ewc_mode,
        decay_factor=args.decay_factor,
        train_epochs=args.epochs,
        device=device,
        train_mb_size=args.minibatch_size,
        evaluator=eval_plugin,
    )

    # train on the selected scenario with the chosen strategy
    print("Starting experiment...")
    results = []
    for experience in scenario.train_stream:
        print("Start training on experience ", experience.current_experience)

        strategy.train(experience)
        print("End training on experience", experience.current_experience)
        print("Computing accuracy on the test set")
        results.append(strategy.eval(scenario.test_stream[:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["pmnist", "smnist"],
        default="smnist",
        help="Choose between Permuted MNIST, Split MNIST.",
    )
    parser.add_argument(
        "--ewc_mode",
        type=str,
        choices=["separate", "online"],
        default="separate",
        help="Choose between EWC and online.",
    )
    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=0.4,
        help="Penalty hyperparameter for EWC",
    )
    parser.add_argument(
        "--decay_factor",
        type=float,
        default=0.1,
        help="Decay factor for importance " "when ewc_mode is online.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hs", type=int, default=256, help="MLP hidden size.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=128, help="Minibatch size."
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5,
        help="Number of experiences in Permuted MNIST.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Specify GPU id to use. Use CPU if -1.",
    )
    args = parser.parse_args()

    main(args)
