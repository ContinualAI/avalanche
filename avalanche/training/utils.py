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

General utility functions for pytorch.

"""
from collections import defaultdict
from typing import NamedTuple, List, Optional, Tuple, Callable

import torch
from torch import Tensor
from torch.nn import Module, Linear
from torch.utils.data import Dataset, DataLoader

from avalanche.models.batch_renorm import BatchRenorm2D


def trigger_plugins(strategy, event, **kwargs):
    """Call plugins on a specific callback

    :return:
    """
    for p in strategy.plugins:
        if hasattr(p, event):
            getattr(p, event)(strategy, **kwargs)


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
    loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers
    )
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

    return [
        (k, torch.zeros_like(p).to(p.device))
        for k, p in model.named_parameters()
    ]


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


def get_layers_and_params(model: Module, prefix="") -> List[LayerAndParameter]:
    result: List[LayerAndParameter] = []
    for param_name, param in model.named_parameters(recurse=False):
        result.append(
            LayerAndParameter(prefix[:-1], model, prefix + param_name, param)
        )

    layer_name: str
    layer: Module
    for layer_name, layer in model.named_modules():
        if layer == model:
            continue

        layer_complete_name = prefix + layer_name + "."

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


def adapt_classification_layer(
    model: Module, num_classes: int, bias: bool = None
) -> Tuple[str, Linear]:
    last_fc_layer: Linear
    last_fc_name, last_fc_layer = get_last_fc_layer(model)

    if bias is not None:
        use_bias = bias
    else:
        use_bias = last_fc_layer.bias is not None

    new_fc = Linear(last_fc_layer.in_features, num_classes, bias=use_bias)
    swap_last_fc_layer(model, new_fc)
    return last_fc_name, new_fc


def replace_bn_with_brn(
    m: Module,
    momentum=0.1,
    r_d_max_inc_step=0.0001,
    r_max=1.0,
    d_max=0.0,
    max_r_max=3.0,
    max_d_max=5.0,
):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            # print('replaced: ', name, attr_str)
            setattr(
                m,
                attr_str,
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
                    max_d_max=max_d_max,
                ),
            )
    for n, ch in m.named_children():
        replace_bn_with_brn(
            ch, momentum, r_d_max_inc_step, r_max, d_max, max_r_max, max_d_max
        )


def change_brn_pars(
    m: Module, momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0, d_max=0.0
):
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


def freeze_up_to(
    model: Module,
    freeze_until_layer: str = None,
    set_eval_mode: bool = True,
    set_requires_grad_false: bool = True,
    layer_filter: Callable[[LayerAndParameter], bool] = None,
    module_prefix: str = "",
):
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
        if (
            freeze_until_layer is not None
            and freeze_until_layer == param_def.layer_name
        ):
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
        torch.as_tensor(targets), return_counts=True
    )
    for unique_idx in range(len(unique_classes)):
        result[int(unique_classes[unique_idx])] = int(
            examples_count[unique_idx]
        )

    return result


__all__ = [
    "load_all_dataset",
    "zerolike_params_dict",
    "copy_params_dict",
    "LayerAndParameter",
    "get_layers_and_params",
    "get_layer_by_name",
    "get_last_fc_layer",
    "swap_last_fc_layer",
    "adapt_classification_layer",
    "replace_bn_with_brn",
    "change_brn_pars",
    "freeze_everything",
    "unfreeze_everything",
    "freeze_up_to",
    "examples_per_class",
]
