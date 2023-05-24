import warnings
from collections import defaultdict
from typing import List, TypeVar

from torch import Tensor

from avalanche.benchmarks import OnlineCLExperience
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.templates.strategy_mixin_protocol import \
    SGDStrategyProtocol


def compare_keys(old_dict, new_dict):
    not_in_new = list(set(old_dict.keys()) - set(new_dict.keys()))
    in_both = list(set(old_dict.keys()) & set(new_dict.keys()))
    not_in_old = list(set(new_dict.keys()) - set(old_dict.keys()))
    return not_in_new, in_both, not_in_old


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

    parameters = []
    optimized_param_id = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            optimized_param_id[n] = p
            parameters.append(p)

    optimizer.param_groups[0]["params"] = parameters

    return optimized_param_id


def update_optimizer(optimizer, new_params, old_param_hash, reset_state=False):
    """
    Detects param change and updates only the modified parameter
    """

    not_in_new, in_both, not_in_old = compare_keys(old_param_hash, new_params)

    keys_to_remove = []

    # Change reference to already existing parameters
    # i.e growing IncrementalClassifier
    for key in in_both:
        old_p_hash = old_param_hash[key]
        new_p = new_params[key]

        # Look for old parameter id in current optimizer
        found = False
        for group in optimizer.param_groups:
            for i, curr_p in enumerate(group["params"]):
                if id(curr_p) == id(old_p_hash):
                    found = True

                    # Parameter id has changed, 
                    # most probably parameter 
                    # dimension has changed, we need
                    # to update reference and reset state

                    # Here two things could happen, 
                    # dimension change in which case we
                    # reset state, or id change due to 
                    # checkpointing in which case we match
                    # the state (change the key of the state)

                    if id(curr_p) != id(new_p):
                        group["params"][i] = new_p
                        old_param_hash[key] = new_p

                        if not new_p.requires_grad:
                            keys_to_remove.append(key)

                        if curr_p in optimizer.state:
                            if curr_p.numel() != new_p.numel():
                                optimizer.state.pop(curr_p)
                                optimizer.state[new_p] = {}
                            else:
                                optimizer.state[new_p] = \
                                    optimizer.state.pop(curr_p)

                    break

        if not found:
            raise Exception(
                f"Parameter {key} not found "
                f"in the current optimizer"
            )

    # Remove parameters that are not here anymore
    # This should not happend in most use case
    for key in not_in_new:
        old_p_hash = old_param_hash[key] 
        found = False
        for i, group in enumerate(optimizer.param_groups):
            keys_to_remove.append([])
            for j, curr_p in enumerate(group["params"]):
                if id(curr_p) == id(old_p_hash):
                    found = True
                    keys_to_remove[i].append((j, curr_p))
                    old_param_hash.pop(key)
                    break
        if not found:
            raise Exception(
                f"Parameter {key} not found "
                f"in the current optimizer"
            )
    for i, idx_list in enumerate(keys_to_remove):
        for (j, p) in sorted(idx_list, key=lambda x: x[0], reverse=True):
            del optimizer.param_groups[i]["params"][j]
            if p in optimizer.state:
                optimizer.state.pop(p)

    # Add newly added parameters (i.e Multitask, PNN)
    # by default, add to param groups 0
    for key in not_in_old:
        new_p = new_params[key]
        if new_p.requires_grad:
            optimizer.param_groups[0]["params"].append(new_p)
            old_param_hash[key] = new_p
            optimizer.state[new_p] = {}

    if reset_state:
        optimizer.state = defaultdict(dict)

    return old_param_hash


class BatchObservation(SGDStrategyProtocol):
    def __init__(self):
        super().__init__()
        self.optimized_param_id = None

    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        assert self.experience is not None

        # For training:
        if isinstance(self.experience, OnlineCLExperience) and self.is_training:
            # If the strategy has access to task boundaries, adapt the model
            # for the whole origin experience to add the
            if self.experience.access_task_boundaries:
                avalanche_model_adaptation(model, 
                                           self.experience.origin_experience)
            else:
                avalanche_model_adaptation(model, self.experience)
        else:
            avalanche_model_adaptation(model, self.experience)

        return model.to(self.device)

    def make_optimizer(self, reset_optimizer_state=False):
        """Optimizer initialization.

        Called before each training experience to configure the optimizer.
        """
        if self.optimized_param_id is None:
            optimized_param_id = reset_optimizer(self.optimizer, self.model) 
        else:
            optimized_param_id = \
                update_optimizer(
                    self.optimizer, 
                    dict(self.model.named_parameters()),
                    self.optimized_param_id, 
                    reset_state=reset_optimizer_state
                    )
        return optimized_param_id

    def check_model_and_optimizer(self):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        assert self.experience is not None

        if self.optimized_param_id is None:
            self.optimized_param_id = \
                self.make_optimizer(reset_optimizer_state=True)

        if isinstance(self.experience, OnlineCLExperience):
            if self.experience.access_task_boundaries:
                if self.experience.is_first_subexp:
                    self.model = self.model_adaptation()
                    self.optimized_param_id = \
                        self.make_optimizer(reset_optimizer_state=False)
        else:
            self.model = self.model_adaptation()
            self.optimized_param_id = \
                self.make_optimizer(reset_optimizer_state=False)


__all__ = ["BatchObservation"]
