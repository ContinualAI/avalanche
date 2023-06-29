################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module conveniently wraps Pytorch Datasets for using a clean and
comprehensive Avalanche API."""

from torchvision.datasets import MNIST as torchMNIST
from torchvision.datasets import FashionMNIST as torchFashionMNIST
from torchvision.datasets import KMNIST as torchKMNIST
from torchvision.datasets import EMNIST as torchEMNIST
from torchvision.datasets import QMNIST as torchQMNIST
from torchvision.datasets import FakeData as torchFakeData
from torchvision.datasets import CocoCaptions as torchCocoCaptions
from torchvision.datasets import CocoDetection as torchCocoDetection
from torchvision.datasets import LSUN as torchLSUN
from torchvision.datasets import ImageFolder as torchImageFolder
from torchvision.datasets import DatasetFolder as torchDatasetFolder
from torchvision.datasets import ImageNet as torchImageNet
from torchvision.datasets import CIFAR10 as torchCIFAR10
from torchvision.datasets import CIFAR100 as torchCIFAR100
from torchvision.datasets import STL10 as torchSTL10
from torchvision.datasets import SVHN as torchSVHN
from torchvision.datasets import PhotoTour as torchPhotoTour
from torchvision.datasets import SBU as torchSBU
from torchvision.datasets import Flickr8k as torchFlickr8k
from torchvision.datasets import Flickr30k as torchFlickr30k
from torchvision.datasets import VOCDetection as torchVOCDetection
from torchvision.datasets import VOCSegmentation as torchVOCSegmentation
from torchvision.datasets import Cityscapes as torchCityscapes
from torchvision.datasets import SBDataset as torchSBDataset
from torchvision.datasets import USPS as torchUSPS
from torchvision.datasets import HMDB51 as torchKHMDB51
from torchvision.datasets import UCF101 as torchUCF101
from torchvision.datasets import CelebA as torchCelebA


def MNIST(*args, **kwargs):
    return torchMNIST(*args, **kwargs)


def FashionMNIST(*args, **kwargs):
    return torchFashionMNIST(*args, **kwargs)


def KMNIST(*args, **kwargs):
    return torchKMNIST(*args, **kwargs)


def EMNIST(*args, **kwargs):
    return torchEMNIST(*args, **kwargs)


def QMNIST(*args, **kwargs):
    return torchQMNIST(*args, **kwargs)


def FakeData(*args, **kwargs):
    return torchFakeData(*args, **kwargs)


def CocoCaptions(*args, **kwargs):
    return torchCocoCaptions(*args, **kwargs)


def CocoDetection(*args, **kwargs):
    return torchCocoDetection(*args, **kwargs)


def LSUN(*args, **kwargs):
    return torchLSUN(*args, **kwargs)


def ImageFolder(*args, **kwargs):
    return torchImageFolder(*args, **kwargs)


def DatasetFolder(*args, **kwargs):
    return torchDatasetFolder(*args, **kwargs)


def ImageNet(*args, **kwargs):
    return torchImageNet(*args, **kwargs)


def CIFAR10(*args, **kwargs):
    return torchCIFAR10(*args, **kwargs)


def CIFAR100(*args, **kwargs):
    return torchCIFAR100(*args, **kwargs)


def STL10(*args, **kwargs):
    return torchSTL10(*args, **kwargs)


def SVHN(*args, **kwargs):
    return torchSVHN(*args, **kwargs)


def PhotoTour(*args, **kwargs):
    return torchPhotoTour(*args, **kwargs)


def SBU(*args, **kwargs):
    return torchSBU(*args, **kwargs)


def Flickr8k(*args, **kwargs):
    return torchFlickr8k(*args, **kwargs)


def Flickr30k(*args, **kwargs):
    return torchFlickr30k(*args, **kwargs)


def VOCDetection(*args, **kwargs):
    return torchVOCDetection(*args, **kwargs)


def VOCSegmentation(*args, **kwargs):
    return torchVOCSegmentation(*args, **kwargs)


def Cityscapes(*args, **kwargs):
    return torchCityscapes(*args, **kwargs)


def SBDataset(*args, **kwargs):
    return torchSBDataset(*args, **kwargs)


def USPS(*args, **kwargs):
    return torchUSPS(*args, **kwargs)


def HMDB51(*args, **kwargs):
    return torchKHMDB51(*args, **kwargs)


def UCF101(*args, **kwargs):
    return torchUCF101(*args, **kwargs)


def CelebA(*args, **kwargs):
    return torchCelebA(*args, **kwargs)


if __name__ == "__main__":
    mnist = MNIST(".", download=True)


__all__ = [
    "MNIST",
    "FashionMNIST",
    "KMNIST",
    "EMNIST",
    "QMNIST",
    "FakeData",
    "CocoCaptions",
    "CocoDetection",
    "LSUN",
    "LSUN",
    "ImageFolder",
    "DatasetFolder",
    "ImageNet",
    "CIFAR10",
    "CIFAR100",
    "STL10",
    "SVHN",
    "PhotoTour",
    "SBU",
    "Flickr8k",
    "Flickr30k",
    "VOCDetection",
    "VOCSegmentation",
    "Cityscapes",
    "SBDataset",
    "USPS",
    "HMDB51",
    "UCF101",
    "CelebA",
]
