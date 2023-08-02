import torch
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
from torch.optim import SGD

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from avalanche.models import ExpertGate
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training.supervised import ExpertGateStrategy
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.benchmarks import nc_benchmark

"""
This example tests ExpertGate on Split MNIST or the fast generated 
benchmark. Given all the operations and internal evaluation this algorithm
requires, it runs a little slower than other examples.
"""


def main(args):
    # Set device
    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize model and specify shape
    model = ExpertGate(shape=(3, 227, 227), device=device)

    # Vanilla optimization
    optimizer = SGD(
        model.expert.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005
    )

    # Set up strategy
    strategy = ExpertGateStrategy(
        model,
        optimizer,
        device=device,
        train_mb_size=args.minibatch_size,
        train_epochs=args.epochs,
        eval_every=-1,
        ae_train_mb_size=args.minibatch_size,
        ae_train_epochs=int(args.epochs * 2),
        ae_lr=1e-3,
    )

    # Build scenario (fast or MNIST)
    scenario = build_scenario(args.mnist)

    # Train loop
    for experience in scenario.train_stream:
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        print()
        print(f"Task {t} batch {exp_id}")
        print(f"This batch contains {len(training_dataset)} patterns")
        print(f"Current Classes: {experience.classes_in_this_experience}")

        strategy.train(experience)

    # Evaluation loop
    print("\nEVALUATION")
    for experience in scenario.test_stream:
        strategy.eval(experience)


def build_scenario(mnist=False):
    if not mnist:
        # Fake benchmark is (1,1,6)
        # Data needs to be transformed for AlexNet
        # Repeat the "channel" as AlexNet expects 3 channel input
        # Resize to 227 because AlexNet convolution will reduce the data shape
        CustomDataAlexTransform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize((227, 227)),
            ]
        )

        scenario = get_custom_benchmark(
            use_task_labels=True,
            train_transform=CustomDataAlexTransform,
            eval_transform=CustomDataAlexTransform,
            shuffle=False,
        )
    else:
        # More resource intensive example
        MNISTAlexTransform = transforms.Compose(
            [
                transforms.Resize((227, 227)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )

        # Note: Must provide task ID for training
        scenario = SplitMNIST(
            n_experiences=5,
            return_task_id=True,
            train_transform=MNISTAlexTransform,
            eval_transform=MNISTAlexTransform,
        )

    return scenario


def get_custom_benchmark(
    use_task_labels=False,
    shuffle=False,
    n_samples_per_class=100,
    train_transform=None,
    eval_transform=None,
):
    dataset = make_classification(
        n_samples=10 * n_samples_per_class,
        n_classes=10,
        n_features=6,
        n_informative=6,
        n_redundant=0,
    )

    X = torch.from_numpy(dataset[0]).float()
    y = torch.from_numpy(dataset[1]).long()

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.8, shuffle=True, stratify=y
    )

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    my_nc_benchmark = nc_benchmark(
        train_dataset,
        test_dataset,
        5,
        task_labels=use_task_labels,
        shuffle=shuffle,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    return my_nc_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=256, help="Minibatch size."
    )
    parser.add_argument(
        "--mnist", action="store_true", help="Use the MNIST dataset for the example"
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Specify GPU id to use. Use CPU if -1.",
    )
    args = parser.parse_args()

    main(args)
