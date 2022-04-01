################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-31-2022                                                             #
# Author: Zhiqiu Lin
# E-mail: zl279@cornell.edu
# Website: https://clear-benchmark.github.io                                                #
################################################################################

""" CLEAR10 data. """


base_gdrive_url = "https://drive.google.com/u/0/uc?id="
name_gdriveid_md5 = [
    ("CLEAR-10-PUBLIC-AVALANCHE.zip", # name
     "1m9dAJtMynq1ayjx-R5vRNvN6tSuFgmll", # Google file ID
     "755799fbc7404c714c57d2c1c218ac0b" # MD5
    )
]

__all__ = ["base_gdrive_url", "name_gdriveid_md5"]
