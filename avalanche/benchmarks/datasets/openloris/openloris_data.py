################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-11-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" OpenLoris data. """


base_gdrive_url = "https://drive.google.com/u/0/uc?id="
gdrive_data = [
    ('train.zip',
     '11jgiPB2Z9WRI3bW6VSN8fJZgwFl5mLsF'),
    ('valid.zip',
     '1ChoBAGcQ_wkclPXsel8CjJHC0tD7b4ga'),
    ('test.zip',
     '1J7_ljcwSZNXo6KwlhRZoG0kiEcRK7U6x'),
    ('LUP.pkl',
     '1Os8T30NZ3ZU8liHQPeVbo2nlOoPZuDSV'),
    ('Paths.pkl',
     '1KnuYLdlG3VQrhgbtIANLki81ah8Thezj'),
    ('Labels.pkl',
     '1GkmOxIAvmjSwo22UzmZTSlw8NSmU5Q9H'),
    ('batches_filelists.zip',
     '1r0gbo5_Qlzrdet1GPIrJpVSGRgFU7NEp')
]
avl_vps_data = [
    ('train.zip',
     'http://vps.continualai.org/data/openloris/train.zip'),
    ('valid.zip',
     'http://vps.continualai.org/data/openloris/valid.zip'),
    ('test.zip',
     'http://vps.continualai.org/data/openloris/test.zip'),
    ('LUP.pkl',
     'http://vps.continualai.org/data/openloris/LUP.pkl'),
    ('Paths.pkl',
     'http://vps.continualai.org/data/openloris/Paths.pkl'),
    ('Labels.pkl',
     'http://vps.continualai.org/data/openloris/Labels.pkl'),
    ('batches_filelists.zip',
     'http://vps.continualai.org/data/openloris/batches_filelists.zip')
]

__all__ = [
    'base_gdrive_url',
    'gdrive_data',
    'avl_vps_data'
]
