import dill
from torchvision.datasets import CIFAR100, CIFAR10

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.checkpointing import constructor_based_serialization


def get_cifar10_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("cifar10")

    train_set = CIFAR10(str(dataset_root), train=True, download=True)
    test_set = CIFAR10(str(dataset_root), train=False, download=True)

    return train_set, test_set


def get_cifar100_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("cifar100")

    train_set = CIFAR100(str(dataset_root), train=True, download=True)
    test_set = CIFAR100(str(dataset_root), train=False, download=True)

    return train_set, test_set


def load_CIFAR100(root, train, transform, target_transform):
    return CIFAR100(
        root=root, train=train, transform=transform, target_transform=target_transform
    )


@dill.register(CIFAR100)
def checkpoint_CIFAR100(pickler, obj: CIFAR100):
    constructor_based_serialization(
        pickler,
        obj,
        CIFAR100,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            train=obj.train,
            transform=obj.transform,
            target_transform=obj.target_transform,
        ),
    )


@dill.register(CIFAR10)
def checkpoint_CIFAR10(pickler, obj: CIFAR10):
    constructor_based_serialization(
        pickler,
        obj,
        CIFAR10,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            train=obj.train,
            transform=obj.transform,
            target_transform=obj.target_transform,
        ),
    )


__all__ = ["get_cifar10_dataset", "get_cifar100_dataset"]
