#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. ContinualAI. All rights reserved.                        #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-07-2019                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" Tensorboard test """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import keyword
import torch

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Tensorboard setup
exp_name = "test"
log_dir = '/home/vincenzo/avalanche-dev/logs/' + exp_name
writer = SummaryWriter(log_dir)

writer.add_hparams(
    {'lr': 0.1, 'bsize': 1},
    {'hparam/accuracy': 10, 'hparam/loss': 10}
)
hyper = json.dumps({
        "mb_size": 12, "inc_train_ep": 10})
for c in ["{", "}", '"']:
    hyper = hyper.replace(c, "")
hyper = hyper.replace(",","<br>")
writer.add_text('hyper', hyper, 0)

# We only need to specify the layout once (instead of per step).
for n_iter in range(100):

    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)

    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    writer.add_scalar('Efficiency/ram', np.random.random(), n_iter)
    writer.add_scalar('Efficiency/disk', np.random.random(), n_iter)


img_batch = np.zeros((16, 3, 100, 100))
for i in range(16):
    img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
    img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

writer.add_images('confusion matrices', img_batch, 0)

for i in range(10):
    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i

    writer.add_image('evolving cm', img, i)

for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)

meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i]*=i/100.0

writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
writer.add_embedding(torch.randn(100, 5), label_img=label_img)
writer.add_embedding(torch.randn(100, 5), metadata=meta)

layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
             'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                  'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

writer.add_custom_scalars(layout)

for n_iter in range(1000):

    writer.add_scalar('nasdaq/aaa', np.random.random(), n_iter)
    writer.add_scalar('nasdaq/bbb', np.random.random(), n_iter)
    writer.add_scalar('nasdaq/ccc', np.random.random(), n_iter)
    writer.add_scalar('dow/aaa', np.random.random(), n_iter)
    writer.add_scalar('dow/bbb', np.random.random(), n_iter)
    writer.add_scalar('dow/ccc', np.random.random(), n_iter)
    writer.add_scalar('twse/0050', np.random.random(), n_iter)
    writer.add_scalar('twse/2330', np.random.random(), n_iter)

vertices_tensor = torch.as_tensor([
    [1, 1, 1],
    [-1, -1, 1],
    [1, -1, -1],
    [-1, 1, -1],
], dtype=torch.float).unsqueeze(0)
colors_tensor = torch.as_tensor([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 255],
], dtype=torch.int).unsqueeze(0)
faces_tensor = torch.as_tensor([
    [0, 2, 3],
    [0, 3, 1],
    [0, 1, 2],
    [1, 3, 2],
], dtype=torch.int).unsqueeze(0)

writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)

r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)

writer.close()

writer.flush()
writer.close()