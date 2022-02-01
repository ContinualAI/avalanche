################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-04-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Utilities to handle optimizer's update when using dynamic architectures.
    Dynamic Modules (e.g. multi-head) can change their parameters dynamically
    during training, which usually requires to update the optimizer to learn
    the new parameters or freeze the old ones.
"""
from collections import defaultdict


def reset_optimizer(optimizer, model):
    """Reset the optimizer to update the list of learnable parameters.

    .. warning::
        This function fails if the optimizer uses multiple parameter groups.

    :param optimizer:
    :param model:
    :return:
    """
    assert len(optimizer.param_groups) == 1
    optimizer.state = defaultdict(dict)
    optimizer.param_groups[0]["params"] = list(model.parameters())


def update_optimizer(optimizer, old_params, new_params, reset_state=True):
    """Update the optimizer by substituting old_params with new_params.

    :param old_params: List of old trainable parameters.
    :param new_params: List of new trainable parameters.
    :param reset_state: Wheter to reset the optimizer's state.
        Defaults to True.
    :return:
    """
    for old_p, new_p in zip(old_params, new_params):
        found = False
        # iterate over group and params for each group.
        for group in optimizer.param_groups:
            for i, curr_p in enumerate(group["params"]):
                if hash(curr_p) == hash(old_p):
                    # update parameter reference
                    group["params"][i] = new_p
                    found = True
                    break
            if found:
                break
        if not found:
            raise Exception(
                f"Parameter {old_params} not found in the "
                f"current optimizer."
            )
    if reset_state:
        # State contains parameter-specific information.
        # We reset it because the model is (probably) changed.
        optimizer.state = defaultdict(dict)


def add_new_params_to_optimizer(optimizer, new_params):
    """Add new parameters to the trainable parameters.

    :param new_params: list of trainable parameters
    """
    optimizer.add_param_group({"params": new_params})
