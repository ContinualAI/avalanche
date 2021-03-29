from torchvision.datasets import MNIST


def adapt_dataset_urls():
    # prev_mnist_urls = MNIST.resources
    # # new_resources = [
    # #     ('https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz', prev_mnist_urls[0][1]),
    # #     ('https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz', prev_mnist_urls[1][1]),
    # #     ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz', prev_mnist_urls[2][1]),
    # #     ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz', prev_mnist_urls[3][1])
    # # ]
    #
    # new_resources = [
    #     ('https://github.com/mkolod/MNIST/raw/master/'
    #      'train-images-idx3-ubyte.gz', prev_mnist_urls[0][1]),
    #     ('https://github.com/mkolod/MNIST/raw/master/'
    #      'train-labels-idx1-ubyte.gz', prev_mnist_urls[1][1]),
    #     ('https://github.com/mkolod/MNIST/raw/master/'
    #      't10k-images-idx3-ubyte.gz', prev_mnist_urls[2][1]),
    #     ('https://github.com/mkolod/MNIST/raw/master/'
    #      't10k-labels-idx1-ubyte.gz', prev_mnist_urls[3][1])
    # ]
    #
    # MNIST.resources = new_resources
    pass


def common_setups():
    # adapt_dataset_urls()
    pass


__all__ = [
    'adapt_dataset_urls',
    'common_setups'
]
