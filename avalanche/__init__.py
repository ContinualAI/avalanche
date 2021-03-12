import os
__version__ = "0.0.1"


# Hotfix for MNIST download issue
def _adapt_dataset_urls():
    from torchvision.datasets import MNIST
    prev_mnist_urls = MNIST.resources
    new_resources = [
        ('https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz', prev_mnist_urls[0][1]),
        ('https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz', prev_mnist_urls[1][1]),
        ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz', prev_mnist_urls[2][1]),
        ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz', prev_mnist_urls[3][1])
    ]
    MNIST.resources = new_resources


_adapt_dataset_urls()
