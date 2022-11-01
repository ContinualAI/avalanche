import dill
from torchvision.datasets import MNIST

from avalanche.benchmarks.datasets import default_dataset_location


def get_mnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("mnist")

    train_set = MNIST(root=dataset_root, train=True, download=True)

    test_set = MNIST(root=dataset_root, train=False, download=True)

    return train_set, test_set


def load_MNIST(root, train, transform, target_transform):
    return MNIST(root=root, train=train, transform=transform,
                 target_transform=target_transform)


@dill.register(MNIST)
def save_MNIST(pickler, obj: MNIST):
    pickler.save_reduce(load_MNIST,
                        (obj.root, obj.train, obj.transform,
                         obj.target_transform), obj=obj)


__all__ = [
    'get_mnist_dataset'
]
