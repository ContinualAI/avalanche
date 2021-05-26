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
    ('core50_128x128.zip',
     'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'),
    ('batches_filelists.zip',
     'https://vlomonaco.github.io/core50/data/batches_filelists.zip'),
    ('batches_filelists_NICv2.zip',  
     'https://vlomonaco.github.io/core50/data/batches_filelists_NICv2.zip'),
    ('paths.pkl', 'https://vlomonaco.github.io/core50/data/paths.pkl'),
    ('LUP.pkl', 'https://vlomonaco.github.io/core50/data/LUP.pkl'),
    ('labels.pkl', 'https://vlomonaco.github.io/core50/data/labels.pkl'),
    ('labels2names.pkl',
     'https://vlomonaco.github.io/core50/data/labels2names.pkl')
]

extra_data = [
    ('core50_imgs.npz',
     'http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz')
]

scen2dirs = {
    'ni': "batches_filelists/NI_inc/",
    'nc': "batches_filelists/NC_inc/",
    'nic': "batches_filelists/NIC_inc/",
    'nicv2_79': "NIC_v2_79/",
    'nicv2_196': "NIC_v2_196/",
    'nicv2_391': "NIC_v2_391/"
}

name2cat = {
    'plug_adapter': 0,
    'mobile_phone': 1,
    'scissor': 2,
    'light_bulb': 3,
    'can': 4,
    'glass': 5,
    'ball': 6,
    'marker': 7,
    'cup': 8,
    'remote_control': 9
}

__all__ = [
    'data',
    'extra_data',
    'scen2dirs',
    'name2cat'
]
