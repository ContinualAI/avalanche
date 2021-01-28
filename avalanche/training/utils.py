#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 7-12-2017                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

General useful functions for pytorch.

"""
# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import NamedTuple, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import logging


def get_accuracy(model, criterion, batch_size, test_x, test_y, test_it,
                 device=None, mask=None):
    """ Test accuracy given net and data. """

    correct_cnt, ave_loss = 0, 0
    model = model.to(device)

    num_class = torch.max(test_y) + 1
    hits_per_class = [0] * num_class
    pattern_per_class = [0] * num_class

    for i in range(test_it):
        # indexing
        start = i * batch_size
        end = (i + 1) * batch_size

        x = test_x[start:end].to(device)

        y = test_y[start:end].to(device)

        logits = model(x)

        if mask is not None:
            # we put an high negative number so that after softmax that prob
            # will be zero and not contribute to the loss
            idx = (torch.tensor(mask, dtype=torch.float,
                                device=device) == 0).nonzero()
            idx = idx.view(idx.size(0))
            logits[:, idx] = -10e10

        loss = criterion(logits, y)
        _, pred_label = torch.max(logits.data, 1)
        correct_cnt += (pred_label == y.data).sum()
        ave_loss += loss.data[0]

        for label in y.data:
            pattern_per_class[int(label)] += 1

        for i, pred in enumerate(pred_label):
            if pred == y.data[i]:
                hits_per_class[int(pred)] += 1

    accs = np.asarray(hits_per_class) / np.asarray(pattern_per_class)\
        .astype(float)

    acc = correct_cnt * 1.0 / test_y.size(0)

    ave_loss /= test_y.size(0)

    return ave_loss, acc, accs


def train_net(optimizer, model, criterion, batch_size, train_x, train_y,
              train_it, device=None, mask=None):
    """ Train net from memory using pytorch """

    log = logging.getLogger("avalanche")

    correct_cnt, ave_loss = 0, 0
    model = model.to(device)

    for it in range(train_it):

        start = it * batch_size
        end = (it + 1) * batch_size

        optimizer.zero_grad()
        x = train_x[start:end].to(device)
        y = train_y[start:end]
        logits = model(x)

        if mask is not None:
            # we put an high negative number so that after softmax that prob
            # will be zero and not contribute to the loss
            idx = (torch.tensor(mask, dtype=torch.float,
                                device=device) == 0).nonzero()
            idx = idx.view(idx.size(0))
            logits[:, idx] = -10e10

        _, pred_label = torch.max(logits.data, 1)
        correct_cnt += (pred_label == y.data).sum()

        loss = criterion(logits, y)
        ave_loss += loss.data[0]

        loss.backward()
        optimizer.step()

        acc = correct_cnt / ((it + 1) * y.size(0))
        ave_loss /= ((it + 1) * y.size(0))

        if it % 10 == 0:
            log.info(
                '==>>> it: {}, avg. loss: {:.6f}, running train acc: {:.3f}'
                .format(it, ave_loss, acc)
            )

    return ave_loss, acc


def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):
    """ Here we get a batch of PIL imgs and we return them normalized as for
        the pytorch pre-trained models. """

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def maybe_cuda(what, use_cuda=True, **kw):
    """ Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
    """
    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


def change_lr(optimizer, lr):
    """Change the learning rate of the optimizer"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_classifier(model, weigth, bias, clas=None):
    """ Change weights and biases of the last layer in the network. """

    if clas is None:
        # model.classifier.weight.data = torch.from_numpy(weigth).float()
        # model.classifier.bias.data = torch.from_numpy(bias).float()
        model.classifier = torch.nn.Linear(512, 10)

    else:
        raise NotImplementedError()


def reset_classifier(model, val=0, std=None):
    """ Set weights and biases of the last layer in the network to zero. """

    weights = np.zeros_like(model.classifier.weight.data.numpy())
    biases = np.zeros_like(model.classifier.bias.data.numpy())

    if std:
        weights = np.random.normal(
            val, std, model.classifier.weight.data.numpy().shape
        )
    else:
        weights.fill(val)

    biases.fill(0)
    # self.net.classifier[-1].weight.data.normal_(0.0, 0.02)
    # self.net.classifier[-1].bias.data.fill_(0)

    set_classifier(model, weights, biases)


def shuffle_in_unison(dataset, seed=None, in_place=False):
    """
    Shuffle two (or more) list in unison. It's important to shuffle the images
    and the labels maintaining their correspondence.

    :args dataset: list of shuffle with the same order.
    :args seed: set of fixed Cifar parameters.
    :args in_place: if we want to shuffle the same data or we want
                     to return a new shuffled dataset.

    :return: train and test sets composed of images and labels, if in_place
        is set to False.
    """

    if seed:
        np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        if in_place:
            np.random.shuffle(x)
        else:
            new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)

    if not in_place:
        return new_dataset


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """

    f = x - np.max(x)
    return np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)
    # If you do not care about stability use line above:
    # return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def count_lines(fpath):
    """ Count line in file. """

    num_imgs = 0
    with open(fpath, 'r') as f:
        for line in f:
            if '/' in line:
                num_imgs += 1
    return num_imgs


def pad_data(dataset, mb_size):
    """ Padding all the matrices contained in dataset to suit the mini-batch
        size. We assume they have the same shape. """

    num_set = len(dataset)
    x = dataset[0]
    # computing test_iters
    n_missing = x.shape[0] % mb_size
    if n_missing > 0:
        surplus = 1
    else:
        surplus = 0
    it = x.shape[0] // mb_size + surplus

    # padding data to fix batch dimentions
    if n_missing > 0:
        n_to_add = mb_size - n_missing
        for i, data in enumerate(dataset):
            if isinstance(data, torch.Tensor):
                dataset[i] = torch.cat((data[:n_to_add], data), dim=0)
            else:
                dataset[i] = np.concatenate((data[:n_to_add], data))
    if num_set == 1:
        dataset = dataset[0]

    return dataset, it


def compute_one_hot(train_y, class_count):
    """ Compute one-hot from labels. """

    target_y = np.zeros((train_y.shape[0], class_count), dtype=np.float32)
    target_y[np.arange(train_y.shape[0]), train_y.astype(np.int8)] = 1

    return target_y


def imagenet_batch_preproc(img_batch, rgb_swap=True, channel_first=True,
                           avg_sub=True):
    """ Pre-process batch of PIL img for Imagenet pre-trained models with caffe.
        It may be need adjustements depending on the pre-trained model
        since it is training dependent. """

    # we assume img is a 3-channel image loaded with PIL
    # so img has dim (w, h, c)

    if rgb_swap:
        # Swap RGB to BRG
        img_batch = img_batch[:, :, :, ::-1]

    if avg_sub:
        # Subtract channel average
        img_batch[:, :, :, 0] -= 104
        img_batch[:, :, :, 1] -= 117
        img_batch[:, :, :, 2] -= 123

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def load_all_dataset(dataset: Dataset, num_workers: int = 0):
    """
    Retrieves the contents of a whole dataset by using a DataLoader

    :param dataset: The dataset
    :param num_workers: The number of workers the DataLoader should use.
        Defaults to 0.
    :return: The content of the whole Dataset
    """
    # DataLoader parallelism is batch-based. By using "len(dataset)/num_workers"
    # as the batch size, num_workers [+1] batches will be loaded thus
    # using the required number of workers.
    if num_workers > 0:
        batch_size = max(1, len(dataset) // num_workers)
    else:
        batch_size = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                        num_workers=num_workers)
    batches_x = []
    batches_y = []
    for batch_x, batch_y in loader:
        batches_x.append(batch_x)
        batches_y.append(batch_y)

    x, y = torch.cat(batches_x), torch.cat(batches_y)
    return x, y


def zerolike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    """

    return [(k, torch.zeros_like(p).to(p.device))
            for k, p in model.named_parameters()]


def copy_params_dict(model, copy_grad=False):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.clone()) for k, p in model.named_parameters()]


class LayerParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor


def get_layers_and_params(model: Module, prefix='') -> List[LayerParameter]:
    result: List[LayerParameter] = []
    for param_name, param in model.named_parameters(recurse=False):
        result.append(LayerParameter(
            prefix[:-1], model, prefix + param_name, param))

    layer_name: str
    layer: Module
    for layer_name, layer in model.named_modules():
        if layer == model:
            continue

        layer_complete_name = prefix + layer_name + '.'

        result += get_layers_and_params(layer, prefix=layer_complete_name)

    return result


__all__ = ['get_accuracy',
           'train_net',
           'preprocess_imgs',
           'maybe_cuda',
           'change_lr',
           'set_classifier',
           'reset_classifier',
           'shuffle_in_unison',
           'softmax',
           'count_lines',
           'pad_data',
           'compute_one_hot',
           'imagenet_batch_preproc',
           'load_all_dataset',
           'zerolike_params_dict',
           'copy_params_dict',
           'get_layers_and_params']
