import torch
import argparse
from avalanche.benchmarks import SplitMNIST
from avalanche.training.supervised import BiC
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


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

    # create split scenario
    scenario = SplitMNIST(n_experiences=5, return_task_id=False)

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    strategy = BiC(
        model,
        optimizer,
        criterion,
        val_percentage=args.val_exemplar_percentage, 
        T=args.T, 
        mem_size=args.mem_size, 
        stage_2_epochs=args.num_bias_epochs, 
        lamb=args.lamb, 
        verbose=True,
        train_epochs=args.epochs,
        device=device,
        train_mb_size=args.minibatch_size,
        evaluator=eval_plugin,
    )

    # train on the selected scenario with the chosen strategy
    print("Starting experiment...")
    results = []
    for train_batch_info in scenario.train_stream:
        print(
            "Start training on experience ", train_batch_info.current_experience
        )

        strategy.train(train_batch_info, num_workers=0)
        print(
            "End training on experience ", train_batch_info.current_experience
        )
        print("Computing accuracy on the test set")
        results.append(strategy.eval(scenario.test_stream[:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lamb', default=-1, type=float, required=False,
                        help='Forgetting-intransigence trade-off \
                                (default=%(default)s)')
    parser.add_argument('--T', default=2, type=int, required=False,
                        help='Temperature scaling (default=%(default)s)')
    parser.add_argument('--val-exemplar-percentage', default=0.1,
                        type=float, required=False,
                        help='Percentage of exemplars that will be used \
                                    for validation (default=%(default)s)')
    parser.add_argument('--mem-size', default=2000, type=int, required=False,
                        help='Memory size')
    parser.add_argument('--num-bias-epochs', default=200, type=int, 
                        required=False,
                        help='Number of epochs for training bias \
                                    (default=%(default)s)')
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hs", type=int, default=256, help="MLP hidden size.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=128, help="Minibatch size."
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Specify GPU id to use. Use CPU if -1.",
    )
    args = parser.parse_args()

    main(args)
