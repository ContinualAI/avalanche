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
    ("IncrementalClasses_Classification.zip",
    "https://zenodo.org/record/4899267/files/IncrementalClasses_Classification.zip",
    "8f53c18e46de35ba375d6ed8fced5d47"),
    ("IncrementalLighting_Classification.zip",
    "https://zenodo.org/record/4899267/files/IncrementalLighting_Classification.zip",
    "61a36070d6aae926ef3d121fd17ec501"),
    ("IncrementalWeather_Classification.zip"
    "https://zenodo.org/record/4899267/files/IncrementalWeather_Classification.zip",
    "60e1a0d50b0091e16424d8e88ae2a2a2")
]


default_classification_labelmap = {
    "BG": 0,
    "Tree": 1,
    "Car": 2,
    "People": 3,
    "Streetlamp": 4
}