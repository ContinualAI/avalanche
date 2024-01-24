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
        "clear10-train.zip",  # name
        "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear10-train.zip",
    ),
    (
        "clear10-test.zip",  # name
        "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear10-test.zip",
    ),
]
clear100 = [
    (
        "clear100-train.zip",  # name
        "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-train.zip",
    ),
    (
        "clear100-test.zip",  # name
        "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-test.zip",
    ),
]
clear10_neurips2021 = [
    (
        "clear10-public.zip",  # name
        "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/",
    )
]
clear100_cvpr2022 = [
    (
        "clear100-workshop-avalanche.zip",  # name
        "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/",
    )
]


__all__ = ["clear10", "clear100", "clear10_neurips2021", "clear100_cvpr2022"]
