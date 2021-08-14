import os
__version__ = "0.0.9"

_dataset_add = None


def _avdataset_radd(self, other, *args, **kwargs):
    from avalanche.benchmarks.utils import AvalancheDataset
    global _dataset_add
    if isinstance(other, AvalancheDataset):
        return NotImplemented

    return _dataset_add(self, other, *args, **kwargs)


def _avalanche_monkey_patches():
    from torch.utils.data.dataset import Dataset
    global _dataset_add
    _dataset_add = Dataset.__add__
    Dataset.__add__ = _avdataset_radd


_avalanche_monkey_patches()


# Hotfix for MNIST download issue
# def _adapt_dataset_urls():
#     from torchvision.datasets import MNIST
#     prev_mnist_urls = MNIST.resources
#     new_resources = [
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          'train-images-idx3-ubyte.gz', prev_mnist_urls[0][1]),
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          'train-labels-idx1-ubyte.gz', prev_mnist_urls[1][1]),
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          't10k-images-idx3-ubyte.gz', prev_mnist_urls[2][1]),
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          't10k-labels-idx1-ubyte.gz', prev_mnist_urls[3][1])
#     ]
#     MNIST.resources = new_resources
#
#
# _adapt_dataset_urls()
