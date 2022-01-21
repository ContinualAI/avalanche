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

""" Endless-CL_Simulator Metadata """


data = [
    (
        "IncrementalClasses_Classification.zip",
        "https://zenodo.org/record/4899267/files/"
        "IncrementalClasses_Classification.zip",
        "8f53c18e46de35ba375d6ed8fced5d47",
    ),
    (
        "IncrementalLighting_Classification.zip",
        "https://zenodo.org/record/4899267/files/"
        "IncrementalLighting_Classification.zip",
        "61a36070d6aae926ef3d121fd17ec501",
    ),
    (
        "IncrementalWeather_Classification.zip",
        "https://zenodo.org/record/4899267/files/"
        "IncrementalWeather_Classification.zip",
        "60e1a0d50b0091e16424d8e88ae2a2a2",
    ),
    (
        "IncrementalClasses_Video.zip",
        "https://zenodo.org/record/4899267/files/IncrementalClasses_Video.zip",
        "d354832d0004e16a429f2231f79a4c40",
    ),
    (
        "IncrementalLighting_Video.zip",
        "https://zenodo.org/record/4899267/files/IncrementalLighting_Video.zip",
        "a7cf1914fd6548e57ab268820045f19b",
    ),
    (
        "IncrementalWeather_Video.zip",
        "https://zenodo.org/record/4899267/files/IncrementalWeather_Video.zip",
        "5fc0718e06faa36da94adefb7e29ac0f",
    ),
]


default_classification_labelmap = {
    "BG": 0,
    "Tree": 1,
    "Car": 2,
    "People": 3,
    "Streetlamp": 4,
}

default_semseg_classmap_obj = {
    "": 0,
    "Tree": 4,
    "Car": 5,
    "People": 6,
    "Streetlamp": 7,
    "Street": 1,
    "SideWalk": 2,
    "Terrain": 0,
    "Building": 3,
}

default_semseg_classmap_obj_reduced = {
    "": 0,
    "Tree": 1,
    "Car": 2,
    "People": 3,
    "Streetlamp": 4,
    "Street": 0,
    "SideWalk": 0,
    "Terrain": 0,
    "Building": 0,
}
