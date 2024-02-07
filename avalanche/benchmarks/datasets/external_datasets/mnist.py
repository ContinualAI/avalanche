import dill
import os
from torchvision.datasets import MNIST
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.checkpointing import constructor_based_serialization


class TensorMNIST(MNIST):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index].float().unsqueeze(0) / 255.0
        target = int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    # Override property raw_folder
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


def get_mnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("mnist")

    train_set = TensorMNIST(root=str(dataset_root), train=True, download=True)

    test_set = TensorMNIST(root=str(dataset_root), train=False, download=True)

    return train_set, test_set


@dill.register(TensorMNIST)
def checkpoint_TensorMNIST(pickler, obj: TensorMNIST):
    constructor_based_serialization(
        pickler,
        obj,
        TensorMNIST,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            train=obj.train,
            transform=obj.transform,
            target_transform=obj.target_transform,
        ),
    )


__all__ = ["get_mnist_dataset"]
