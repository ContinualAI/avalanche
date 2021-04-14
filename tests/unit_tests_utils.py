from os.path import expanduser

import os
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from avalanche.benchmarks import nc_scenario


def adapt_dataset_urls():
    # prev_mnist_urls = MNIST.resources
    # new_resources = [
    #     ('https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz', prev_mnist_urls[0][1]),
    #     ('https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz', prev_mnist_urls[1][1]),
    #     ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz', prev_mnist_urls[2][1]),
    #     ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz', prev_mnist_urls[3][1])
    # ]
    # MNIST.resources = new_resources
    pass


def common_setups():
    # adapt_dataset_urls()
    pass


def load_scenario(use_task_labels=False, fast_test=True):
    """
    Returns a NC Scenario from a fake dataset of 10 classes, 5 experiences,
    2 classes per experience.
    """
    if fast_test:
        my_nc_scenario = get_fast_scenario(use_task_labels)
    else:
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True, download=True,
            transform=Compose([ToTensor()]))

        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False, download=True,
            transform=Compose([ToTensor()]))
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5,
            task_labels=use_task_labels, seed=1234)

    return my_nc_scenario


def get_fast_scenario(use_task_labels=False):
    n_samples_per_class = 100
    dataset = make_classification(
        n_samples=10 * n_samples_per_class,
        n_classes=10,
        n_features=6, n_informative=6, n_redundant=0)

    X = torch.from_numpy(dataset[0]).float()
    y = torch.from_numpy(dataset[1]).long()

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.6, shuffle=True, stratify=y)

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    my_nc_scenario = nc_scenario(train_dataset, test_dataset, 5,
                                 task_labels=use_task_labels)
    return my_nc_scenario


__all__ = [
    'adapt_dataset_urls',
    'common_setups'
]