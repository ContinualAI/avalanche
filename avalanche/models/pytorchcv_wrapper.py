################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Eli Verwimp                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

"""
This module wraps pytorchcv models, with some convient wrappers.
"""

from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn import Module


def vgg(depth: int, batch_normalization=False, pretrained=False) -> Module:
    """
    Wrapper for VGG net of verious depths availble in pytorchcv. Only availabe for imagenet.
    :param depth: Depth of the model, one of (11, 13, 16, 19)
    :param batch_normalization: include batch normalizaion layers
    :param pretrained: loads model pretrained on imagnet
    """
    available_depths = [11, 13, 16, 19]
    if depth not in available_depths:
        raise ValueError(f"Depth {depth} not available, availble depths are {available_depths}")

    name = f"vgg_{depth}"
    if batch_normalization:
        name = f"bn_{name}"

    return ptcv_get_model(name, pretrained=pretrained)


def resnet(dataset: str, depth: int, pretrained=False) -> Module:
    """
    Loader for (basic) renset available in the pytorchcv package. More variants are availble through
    the general wrapper.
    :param dataset: One of cifar10, cifar100, svhn, imagenet.
    :param depth: depth of the architecture, one of (10, 12, 14, 16, 18, 26, 34, 50, 101, 152, 200) for
                  imagenet, (20, 56, 110, 1001, 1202) for the other datasets.
    :param pretrained: loads model pretrained on `dataset`.
    """

    if dataset in ["cifar10", "cifar100", "svhn"]:
        available_depths = [20, 56, 110, 1001, 1202]
        model_name = f"resnet{depth}_{dataset}"
    elif dataset == "imagenet":
        available_depths = [10, 12, 14, 16, 18, 26, 34, 50, 101, 152, 200]
        model_name = f"resnet{depth}"
    else:
        raise ValueError(f"Unrecognized dataset {dataset}")

    if depth not in available_depths:
        raise ValueError(f"Depth {depth} not available for dataset {dataset}, "
                         f"availble depths are {available_depths}")

    model = ptcv_get_model(model_name, pretrained=pretrained)
    return model


def densenet(dataset: str, depth: int,  pretrained=False) -> Module:
    """
    Loader for densenets available in the pytorchcv package.
    :param dataset: One of cifar10, cifar100, svhn, imagenet.
    :param depth: The depth of the densnet. For imagenet depths (121, 161, 169, 201) are supported.
                  The other datasets support dephts (40, 100, 190, 250).
    :param pretrained: load model pretrained on `dataset`..
    """
    if dataset in ["cifar10", "cifar100", "svhn"]:
        available_depths = [40, 100, 190, 250]
        # other growth rates are available through the general method.
        growth_rate = 40 if depth == 190 else 24
        model_name = f"resnet{depth}_k{growth_rate}_{dataset}"
    elif dataset == "imagenet":
        available_depths = [121, 161, 169, 201]
        model_name = f"resnet{depth}"
    else:
        raise ValueError(f"Unrecognized dataset {dataset}")

    if depth not in available_depths:
        raise ValueError(f"Depth {depth} not available for dataset {dataset}, "
                         f"availble depths are {available_depths}")

    model = ptcv_get_model(model_name, pretrained=pretrained)
    return model


def get_model(name: str, pretrained=False):
    """
    This a direct wrapper to the model getter of `pytorchcv`. For available models see:
     https://github.com/osmr/imgclsmob
    """
    return ptcv_get_model(name, pretrained=pretrained)


