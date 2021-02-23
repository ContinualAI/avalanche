################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the MIT License.                                   #
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
from collections import defaultdict
from typing import NamedTuple, List, Optional, Tuple, Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Linear
from torch.utils.data import Dataset, DataLoader
import logging

from avalanche.models.batch_renorm import BatchRenorm2D


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
    has_task_labels = False
    batches_x = []
    batches_y = []
    batches_t = []
    for loaded_element in loader:
        batches_x.append(loaded_element[0])
        batches_y.append(loaded_element[1])
        if len(loaded_element) > 2:
            has_task_labels = True
            batches_t.append(loaded_element[2])

    x, y = torch.cat(batches_x), torch.cat(batches_y)

    if has_task_labels:
        t = torch.cat(batches_t)
        return x, y, t
    else:
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


class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor


def get_layers_and_params(model: Module, prefix='') -> List[LayerAndParameter]:
    result: List[LayerAndParameter] = []
    for param_name, param in model.named_parameters(recurse=False):
        result.append(LayerAndParameter(
            prefix[:-1], model, prefix + param_name, param))

    layer_name: str
    layer: Module
    for layer_name, layer in model.named_modules():
        if layer == model:
            continue

        layer_complete_name = prefix + layer_name + '.'

        result += get_layers_and_params(layer, prefix=layer_complete_name)

    return result


def get_layer_by_name(model: Module, layer_name: str) -> Optional[Module]:
    for layer_param in get_layers_and_params(model):
        if layer_param.layer_name == layer_name:
            return layer_param.layer
    return None


def get_last_fc_layer(model: Module) -> Optional[Tuple[str, Linear]]:
    last_fc = None
    for layer_name, layer in model.named_modules():
        if isinstance(layer, Linear):
            last_fc = (layer_name, layer)

    return last_fc


def swap_last_fc_layer(model: Module, new_layer: Module) -> None:
    last_fc_name, last_fc_layer = get_last_fc_layer(model)
    setattr(model, last_fc_name, new_layer)


def adapt_classification_layer(model: Module, num_classes: int,
                               bias: bool = None) -> Tuple[str, Linear]:
    last_fc_layer: Linear
    last_fc_name, last_fc_layer = get_last_fc_layer(model)

    if bias is not None:
        use_bias = bias
    else:
        use_bias = last_fc_layer.bias is not None

    new_fc = Linear(last_fc_layer.in_features, num_classes, bias=use_bias)
    swap_last_fc_layer(model, new_fc)
    return last_fc_name, new_fc


def replace_bn_with_brn(m: Module, momentum=0.1, r_d_max_inc_step=0.0001,
                        r_max=1.0, d_max=0.0, max_r_max=3.0, max_d_max=5.0):

    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            # print('replaced: ', name, attr_str)
            setattr(m, attr_str,
                    BatchRenorm2D(
                        target_attr.num_features,
                        gamma=target_attr.weight,
                        beta=target_attr.bias,
                        running_mean=target_attr.running_mean,
                        running_var=target_attr.running_var,
                        eps=target_attr.eps,
                        momentum=momentum,
                        r_d_max_inc_step=r_d_max_inc_step,
                        r_max=r_max,
                        d_max=d_max,
                        max_r_max=max_r_max,
                        max_d_max=max_d_max
                        )
                    )
    for n, ch in m.named_children():
        replace_bn_with_brn(ch, momentum, r_d_max_inc_step, r_max, d_max,
                            max_r_max, max_d_max)


def change_brn_pars(
        m: Module, momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0,
        d_max=0.0):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == BatchRenorm2D:
            target_attr.momentum = torch.tensor((momentum), requires_grad=False)
            target_attr.r_max = torch.tensor(r_max, requires_grad=False)
            target_attr.d_max = torch.tensor(d_max, requires_grad=False)
            target_attr.r_d_max_inc_step = r_d_max_inc_step

    for n, ch in m.named_children():
        change_brn_pars(ch, momentum, r_d_max_inc_step, r_max, d_max)


def freeze_everything(model: Module, set_eval_mode: bool = True):
    if set_eval_mode:
        model.eval()

    for layer_param in get_layers_and_params(model):
        layer_param.parameter.requires_grad = False


def unfreeze_everything(model: Module, set_train_mode: bool = True):
    if set_train_mode:
        model.train()

    for layer_param in get_layers_and_params(model):
        layer_param.parameter.requires_grad = True


def freeze_up_to(model: Module,
                 freeze_until_layer: str = None,
                 set_eval_mode: bool = True,
                 set_requires_grad_false: bool = True,
                 layer_filter: Callable[[LayerAndParameter], bool] = None,
                 module_prefix: str = ''):
    """
    A simple utility that can be used to freeze a model.

    :param model: The model.
    :param freeze_until_layer: If not None, the freezing algorithm will continue
        (proceeding from the input towards the output) until the specified layer
        is encountered. The given layer is excluded from the freezing procedure.
    :param set_eval_mode: If True, the frozen layers will be set in eval mode.
        Defaults to True.
    :param set_requires_grad_false: If True, the autograd engine will be
        disabled for frozen parameters. Defaults to True.
    :param layer_filter: A function that, given a :class:`LayerParameter`,
        returns `True` if the parameter must be frozen. If all parameters of
        a layer are frozen, then the layer will be set in eval mode (according
        to the `set_eval_mode` parameter. Defaults to None, which means that all
        parameters will be frozen.
    :param module_prefix: The model prefix. Do not use if non strictly
        necessary.
    :return:
    """

    frozen_layers = set()
    frozen_parameters = set()

    to_freeze_layers = dict()
    for param_def in get_layers_and_params(model, prefix=module_prefix):
        if freeze_until_layer is not None and \
                freeze_until_layer == param_def.layer_name:
            break

        freeze_param = layer_filter is None or layer_filter(param_def)
        if freeze_param:
            if set_requires_grad_false:
                param_def.parameter.requires_grad = False
                frozen_parameters.add(param_def.parameter_name)

            if param_def.layer_name not in to_freeze_layers:
                to_freeze_layers[param_def.layer_name] = (True, param_def.layer)
        else:
            # Don't freeze this parameter -> do not set eval on the layer
            to_freeze_layers[param_def.layer_name] = (False, None)

    if set_eval_mode:
        for layer_name, layer_result in to_freeze_layers.items():
            if layer_result[0]:
                layer_result[1].eval()
                frozen_layers.add(layer_name)

    return frozen_layers, frozen_parameters


def examples_per_class(targets):
    result = defaultdict(int)

    unique_classes, examples_count = torch.unique(
        torch.as_tensor(targets), return_counts=True)
    for unique_idx in range(len(unique_classes)):
        result[int(unique_classes[unique_idx])] = \
            int(examples_count[unique_idx])

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
           'LayerAndParameter',
           'get_layers_and_params',
           'get_layer_by_name',
           'get_last_fc_layer',
           'swap_last_fc_layer',
           'adapt_classification_layer',
           'replace_bn_with_brn',
           'change_brn_pars',
           'freeze_everything',
           'unfreeze_everything',
           'freeze_up_to',
           'examples_per_class']
