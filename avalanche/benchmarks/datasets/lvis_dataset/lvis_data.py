################################################################################
# Copyright (c) 2022 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 18-02-2022                                                             #
# Author: Lorenzo Pellegrini                                                   #
#                                                                              #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

lvis_archives = [
    (
        "lvis_v1_val.json.zip",  # Validation set annotations
        "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS"
        "/lvis_v1_val.json.zip",
        "87734a7f895990b9552075d7ce723e27",
    ),
    (
        "lvis_v1_train.json.zip",  # Training set annotations
        "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS"
        "/lvis_v1_train.json.zip",
        "4410d837f71203af226950447ef9d422",
    ),
    (
        "val2017.zip",  # Validation set images
        "http://images.cocodataset.org/zips/val2017.zip",
        "442b8da7639aecaf257c1dceb8ba8c80",
    ),
    (
        "train2017.zip",  # Training set images
        "http://images.cocodataset.org/zips/train2017.zip",
        "cced6f7f71b7629ddf16f17bbcfab6b2",
    ),
]


__all__ = ["lvis_archives"]
