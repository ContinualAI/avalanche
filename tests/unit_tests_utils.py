from os.path import expanduser

import os
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from avalanche.benchmarks import nc_benchmark


# Environment variable used to skip some expensive tests that are very unlikely
# to break unless you touch their code directly (e.g. datasets).
FAST_TEST = False
if "FAST_TEST" in os.environ:
    FAST_TEST = os.environ["FAST_TEST"].lower() == "true"

# Environment variable used to update the metric pickles providing the ground
# truth for metric tests. If you change the metrics (names, x values, y
# values, ...) you may need to update them.
UPDATE_METRICS = False
if "UPDATE_METRICS" in os.environ:
    UPDATE_METRICS = os.environ["UPDATE_METRICS"].lower() == "true"

print(f"UPDATE_METRICS: {UPDATE_METRICS}")


def is_github_action():
    """Check whether we are running in a Github action.

    We want to avoid some expensive operations (such as downloading data)
    inside the CI pipeline.
    """
    return "GITHUB_ACTION" in os.environ


def common_setups():
    # adapt_dataset_urls()
    pass


def load_benchmark(use_task_labels=False, fast_test=True):
    """
    Returns a NC Benchmark from a fake dataset of 10 classes, 5 experiences,
    2 classes per experience.
    """
    if fast_test:
        my_nc_benchmark = get_fast_benchmark(use_task_labels)
    else:
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
            transform=Compose([ToTensor()]),
        )

        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
            transform=Compose([ToTensor()]),
        )
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=use_task_labels, seed=1234
        )

    return my_nc_benchmark


def get_fast_benchmark(
    use_task_labels=False, shuffle=True, n_samples_per_class=100
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
        X, y, train_size=0.6, shuffle=True, stratify=y
    )

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    my_nc_benchmark = nc_benchmark(
        train_dataset,
        test_dataset,
        5,
        task_labels=use_task_labels,
        shuffle=shuffle,
    )
    return my_nc_benchmark


def load_experience_train_eval(experience, batch_size=32, num_workers=0):
    for x, y, t in DataLoader(
        experience.dataset.train(),
        batch_size=batch_size,
        num_workers=num_workers,
    ):
        break

    for x, y, t in DataLoader(
        experience.dataset.eval(),
        batch_size=batch_size,
        num_workers=num_workers,
    ):
        break


def get_device():
    if "USE_GPU" in os.environ:
        use_gpu = os.environ["USE_GPU"].lower() in ["true"]
    else:
        use_gpu = False
    print("Test on GPU:", use_gpu)
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return device


__all__ = [
    "common_setups",
    "load_benchmark",
    "get_fast_benchmark",
    "load_experience_train_eval",
    "get_device",
]
