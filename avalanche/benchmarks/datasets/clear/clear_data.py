################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-17-2022                                                             #
# Author: Zhiqiu Lin, Jia Shi                                                  #
# E-mail: zl279@cornell.edu, jiashi@andrew.cmu.edu                             #
# Website: https://clear-benchmark.github.io                                   #
################################################################################

""" CLEAR data """
clear10 = [
    (
        "clear10-public.zip",  # name
        "https://clear-challenge.s3.us-east-2.amazonaws.com",
    )
]
clear100_cvpr2022 = [
    (
        "clear100-workshop-avalanche.zip",  # name
        "https://clear-challenge.s3.us-east-2.amazonaws.com",
    )
]

__all__ = ["clear10", "clear100_cvpr2022"]
