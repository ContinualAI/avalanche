import dill
from torchvision.datasets import MNIST
from avalanche.benchmarks.datasets import default_dataset_location


class TensorMNIST(MNIST):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index].float().unsqueeze(0) / 255.
        target = int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_mnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("mnist")

    train_set = TensorMNIST(root=dataset_root, train=True, download=True)

    test_set = TensorMNIST(root=dataset_root, train=False, download=True)

    return train_set, test_set


def load_MNIST(root, train, transform, target_transform):
    return TensorMNIST(root=root, train=train, transform=transform,
                       target_transform=target_transform)


@dill.register(TensorMNIST)
def save_MNIST(pickler, obj: TensorMNIST):
    pickler.save_reduce(load_MNIST,
                        (obj.root, obj.train, obj.transform,
                         obj.target_transform), obj=obj)


__all__ = [
    'get_mnist_dataset'
]
