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
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.utils.data import DataLoader, Dataset

from avalanche.benchmarks import OnlineCLExperience
from avalanche.models.batch_renorm import BatchRenorm2D


def _at_task_boundary(training_experience, before=True) -> bool:
    """
    Given a training experience,
    returns true if the experience is at the task boundary

    More specifically:

    - If task boundary is not available, returns True

    - If task boundary is available,
      returns True if the experience
      is the first subexp

    - If the experience is not an online experience, returns True

    :param before: If used in before_training_exp,
                   set to True, otherwise set
                   to False

    """

    if isinstance(training_experience, OnlineCLExperience):
        if training_experience.access_task_boundaries:
            if before and training_experience.is_first_subexp:
                return True
            elif (not before) and training_experience.is_last_subexp:
                return True
        else:
            return True
    else:
        return True


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


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
    collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
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


def zerolike_params_dict(model: Module) -> Dict[str, "ParamData"]:
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    """

    return dict(
        [
            (k, ParamData(k, p.shape, device=p.device))
            for k, p in model.named_parameters()
        ]
    )


def copy_params_dict(model, copy_grad=False) -> Dict[str, "ParamData"]:
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """
    out: Dict[str, ParamData] = {}
    for k, p in model.named_parameters():
        if copy_grad and p.grad is None:
            continue
        init = p.grad.data.clone() if copy_grad else p.data.clone()
        out[k] = ParamData(k, p.shape, device=p.device, init_tensor=init)
    return out


class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor


def get_layers_and_params(model: Module, prefix="") -> List[LayerAndParameter]:
    result: List[LayerAndParameter] = []
    for param_name, param in model.named_parameters(recurse=False):
        result.append(LayerAndParameter(prefix[:-1], model, prefix + param_name, param))

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


def get_last_fc_layer(model: Module) -> Tuple[str, Linear]:
    last_fc = None
    for layer_name, layer in model.named_modules():
        if isinstance(layer, Linear):
            last_fc = (layer_name, layer)

    if last_fc is None:
        raise ValueError("No fc layer found.")

    return last_fc


def swap_last_fc_layer(model: Module, new_layer: Module) -> None:
    last_fc_name, last_fc_layer = get_last_fc_layer(model)
    setattr(model, last_fc_name, new_layer)


def adapt_classification_layer(
    model: Module, num_classes: int, bias: Optional[bool] = None
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
        if isinstance(target_attr, torch.nn.BatchNorm2d):
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
        if isinstance(target_attr, BatchRenorm2D):
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
    freeze_until_layer: Optional[str] = None,
    set_eval_mode: bool = True,
    set_requires_grad_false: bool = True,
    layer_filter: Optional[Callable[[LayerAndParameter], bool]] = None,
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
        result[int(unique_classes[unique_idx])] = int(examples_count[unique_idx])

    return result


class ParamData(object):
    def __init__(
        self,
        name: str,
        shape: Optional[tuple] = None,
        init_function: Callable[[torch.Size], torch.Tensor] = torch.zeros,
        init_tensor: Union[torch.Tensor, None] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        An object that contains a tensor with methods to expand it along
        a single dimension.

        :param name: data tensor name as a string
        :param shape: data tensor shape. Will be set to the `init_tensor`
            shape, if provided.
        :param init_function: function used to initialize the data tensor.
            If `init_tensor` is provided, `init_function` will only be used
            on subsequent calls of `reset_like` method.
        :param init_tensor: value to be used when creating the object. If None,
            `init_function` will be used.
        :param device: pytorch like device specification as a string or
            `torch.device`.
        """
        assert isinstance(name, str)
        assert (init_tensor is not None) or (shape is not None)
        if init_tensor is not None and shape is not None:
            assert init_tensor.shape == shape

        self.init_function = init_function
        self.name = name
        if shape is not None:
            self.shape = torch.Size(shape)
        else:
            assert init_tensor is not None
            self.shape = init_tensor.size()

        self.device = torch.device(device)
        if init_tensor is not None:
            self._data: torch.Tensor = init_tensor
        else:
            self.reset_like()

    def reset_like(self, shape=None, init_function=None):
        """
        Reset the tensor with the shape provided or, otherwise, by
        using the one most recently provided. The `init_function`,
        if provided, does not override the default one.

        :param shape: the new shape or None to use the current one
        :param init_function: init function to use or None to use
            the default one.
        """
        if shape is not None:
            self.shape = torch.Size(shape)
        if init_function is None:
            init_function = self.init_function
        self._data = init_function(self.shape).to(self.device)

    def expand(self, new_shape, padding_fn=torch.zeros):
        """
        Expand the data tensor along one dimension.
        The shape cannot shrink. It cannot add new dimensions, either.
        If the shape does not change, this method does nothing.

        :param new_shape: expanded shape
        :param padding_fn: function used to create the padding
            around the expanded tensor.

        :return the expanded tensor or the previous tensor
        """
        assert len(new_shape) == len(self.shape), "Expansion cannot add new dimensions"
        expanded = False
        for i, (snew, sold) in enumerate(zip(new_shape, self.shape)):
            assert snew >= sold, "Shape cannot decrease."
            if snew > sold:
                assert (
                    not expanded
                ), "Expansion cannot occur in more than one dimension."
                expanded = True
                exp_idx = i

        if expanded:
            old_data = self._data.clone()
            old_shape_len = self._data.shape[exp_idx]
            self.reset_like(new_shape, init_function=padding_fn)
            idx = [
                slice(el) if i != exp_idx else slice(old_shape_len)
                for i, el in enumerate(new_shape)
            ]
            self._data[idx] = old_data
        return self.data

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value):
        assert value.shape == self._data.shape, (
            "Shape of new value should be the same of old value. "
            "Use `expand` method to expand one dimension. "
            "Use `reset_like` to reset with a different shape."
        )
        self._data = value

    def __str__(self):
        return f"ParamData_{self.name}:{self.shape}:{self._data}"


__all__ = [
    "trigger_plugins",
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
    "ParamData",
    "cycle",
]
