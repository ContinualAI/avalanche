import warnings
from collections import defaultdict
from typing import List, TypeVar

from torch import Tensor

from avalanche.benchmarks import OnlineCLExperience
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.templates.strategy_mixin_protocol import \
    SGDStrategyProtocol


def compare_keys(old_dict, new_dict):
    not_in_new = list(set(old_dict.keys()) - set(new_dict.keys()))
    in_both = list(set(old_dict.keys()) & set(new_dict.keys()))
    not_in_old = list(set(new_dict.keys()) - set(old_dict.keys()))
    return not_in_new, in_both, not_in_old


def update_optimizer(optimizer, new_params, old_param_hash, reset_state=False):
    """
    Detects param change and updates only the modified parameter
    """

    not_in_new, in_both, not_in_old = compare_keys(old_param_hash, new_params)

    # Change reference to already existing parameters
    # i.e growing IncrementalClassifier
    for key in in_both:
        old_p_hash = old_param_hash[key]
        new_p = new_params[key]

        # Look for old parameter id in current optimizer
        found = False
        for group in optimizer.param_groups:
            for i, curr_p in enumerate(group["params"]):
                if id(curr_p) == old_p_hash:
                    found = True
                    # Parameter id has changed, 
                    # most probably parameter 
                    # dimension has changed, we need
                    # to update reference and reset state
                    if id(curr_p) != id(new_p):
                        group["params"][i] = new_p
                        optimizer.state.pop(curr_p)
                        optimizer.state[new_p] = {}
                    break

        if not found:
            raise Exception(
                f"Parameter {key} not found "
                f"in the current optimizer"
            )

    # Remove parameters that are not here anymore
    # This should not happend in most use case
    keys_to_remove = []
    for key in not_in_new:
        old_p_hash = old_param_hash[key] 
        found = False
        for i, group in enumerate(optimizer.param_groups):
            keys_to_remove.append([])
            for j, curr_p in enumerate(group["params"]):
                if id(curr_p) == old_p_hash:
                    found = True
                    keys_to_remove[i].append((j, curr_p))
                    break
        if not found:
            raise Exception(
                f"Parameter {key} not found "
                f"in the current optimizer"
            )
    for i, idx_list in enumerate(keys_to_remove):
        for (j, p) in sorted(idx_list, key=lambda x: x[0], reverse=True):
            del optimizer.param_groups[i]["params"][j]
            optimizer.state.pop(p)

    # Add newly added parameters (i.e Multitask, PNN)
    # by default, add to param groups 0
    for key in not_in_old:
        new_p = new_params[key]
        optimizer.param_groups[0]["params"].append(new_p)
        optimizer.state[new_p] = {}

    if reset_state:
        optimizer.state = defaultdict(dict)


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
            reset_optimizer(self.optimizer, self.model)
            self.optimized_param_id = {n: id(p) for (n, p) in 
                                       self.model.named_parameters()}
        else:
            update_optimizer(self.optimizer, 
                             dict(self.model.named_parameters()),
                             self.optimized_param_id, 
                             reset_state=reset_optimizer_state)
            self.optimized_param_id = {n: id(p) for (n, p) in 
                                       self.model.named_parameters()}

    def check_model_and_optimizer(self):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        assert self.experience is not None
        if isinstance(self.experience, OnlineCLExperience):
            if self.experience.access_task_boundaries:
                if self.experience.is_first_subexp:
                    self.model = self.model_adaptation()
                    self.make_optimizer(reset_optimizer_state=False)
        else:
            self.model = self.model_adaptation()
            self.make_optimizer(reset_optimizer_state=False)


__all__ = ["BatchObservation"]
