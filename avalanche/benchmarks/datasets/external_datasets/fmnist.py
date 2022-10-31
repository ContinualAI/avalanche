import dill
from torchvision.datasets import FashionMNIST

from avalanche.benchmarks.datasets import default_dataset_location


def get_fmnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("fashionmnist")

    train_set = FashionMNIST(dataset_root, train=True, download=True)
    test_set = FashionMNIST(dataset_root, train=False, download=True)
    return train_set, test_set


def load_FashionMNIST(root, train, transform, target_transform):
    return FashionMNIST(root=root, train=train, transform=transform,
                        target_transform=target_transform)


@dill.register(FashionMNIST)
def save_FashionMNIST(pickler, obj: FashionMNIST):
    pickler.save_reduce(load_FashionMNIST,
                        (obj.root, obj.train, obj.transform,
                         obj.target_transform), obj=obj)


__all__ = [
    'get_fmnist_dataset'
]
