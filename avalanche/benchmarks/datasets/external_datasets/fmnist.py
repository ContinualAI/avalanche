import dill
from torchvision.datasets import FashionMNIST

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.checkpointing import constructor_based_serialization


def get_fmnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("fashionmnist")

    train_set = FashionMNIST(dataset_root, train=True, download=True)
    test_set = FashionMNIST(dataset_root, train=False, download=True)
    return train_set, test_set


@dill.register(FashionMNIST)
def checkpoint_FashionMNIST(pickler, obj: FashionMNIST):
    constructor_based_serialization(
        pickler,
        obj,
        FashionMNIST,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            train=obj.train,
            transform=obj.transform,
            target_transform=obj.target_transform,
        ),
    )


__all__ = ["get_fmnist_dataset"]
