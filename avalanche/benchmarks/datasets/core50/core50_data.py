################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-05-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" CORe50 Metadata """

data = [
    (
        "core50_128x128.zip",
        "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip",
        "745f3373fed08d69343f1058ee559e13",
    ),
    (
        "batches_filelists.zip",
        "https://vlomonaco.github.io/core50/data/batches_filelists.zip",
        "e3297508a8998ba0c99a83d6b36bde62",
    ),
    (
        "batches_filelists_NICv2.zip",
        "https://vlomonaco.github.io/core50/data/batches_filelists_NICv2.zip",
        "460f980a6c85b86b1ec8e7c6067bb7a3",
    ),
    (
        "paths.pkl",
        "https://vlomonaco.github.io/core50/data/paths.pkl",
        "b568f86998849184df3ec3465290f1b0",
    ),
    (
        "LUP.pkl",
        "https://vlomonaco.github.io/core50/data/LUP.pkl",
        "33afc26faa460aca98739137fdfa606e",
    ),
    (
        "labels.pkl",
        "https://vlomonaco.github.io/core50/data/labels.pkl",
        "281c95774306a2196f4505f22fd60ab1",
    ),
    (
        "labels2names.pkl",
        "https://vlomonaco.github.io/core50/data/labels2names.pkl",
        "557d0b9f0ec32765ccea65624aa51b3b",
    ),
]

extra_data = [
    (
        "core50_imgs.npz",
        "http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz",
        "3689d65d0a1c760b87821b114c8c4c6c",
    ),
    (
        "core50_32x32.zip",
        "http://vps.continualai.org/data/core50_32x32.zip",
        "d89d34cdc0281fa84074430e9a22b728",
    ),
]

scen2dirs = {
    "ni": "batches_filelists/NI_inc/",
    "nc": "batches_filelists/NC_inc/",
    "nic": "batches_filelists/NIC_inc/",
    "nicv2_79": "NIC_v2_79/",
    "nicv2_196": "NIC_v2_196/",
    "nicv2_391": "NIC_v2_391/",
}

name2cat = {
    "plug_adapter": 0,
    "mobile_phone": 1,
    "scissor": 2,
    "light_bulb": 3,
    "can": 4,
    "glass": 5,
    "ball": 6,
    "marker": 7,
    "cup": 8,
    "remote_control": 9,
}


__all__ = ["data", "extra_data", "scen2dirs", "name2cat"]
