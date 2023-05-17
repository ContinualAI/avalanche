from collections import defaultdict
from typing import List, TypeVar
from torch import Tensor

from avalanche.benchmarks import OnlineCLExperience
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.templates.strategy_mixin_protocol import \
    SGDStrategyProtocol


def detect_param_change(optimizer, new_params) -> bool:
    number_found = 0
    indexes_found = set()
    for group in optimizer.param_groups:
        for old_param in group["params"]:
            found = False
            for i, new_param in enumerate(new_params):
                if id(new_param) == id(old_param):
                    found = True
                    number_found += 1
                    break
            if not found:
                return False
    if number_found != len(new_params):
        return True
    return False


class BatchObservation(SGDStrategyProtocol):
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

    def make_optimizer(self, reset_opt=True, reset_state=False):
        """Optimizer initialization.

        Called before each training experiene to configure the optimizer.
        """
        reset_opt = detect_param_change(
            self.optimizer, list(self.model.named_parameters()),
        ) or reset_opt

        if reset_state:
            self.optimizer.state = defaultdict(dict)

        if reset_opt:
            # Here, optimizer state is also reset
            reset_optimizer(self.optimizer, self.model)

    def check_model_and_optimizer(self):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        assert self.experience is not None
        if isinstance(self.experience, OnlineCLExperience):
            if self.experience.access_task_boundaries:
                if self.experience.is_first_subexp:
                    self.model = self.model_adaptation()
                    self.make_optimizer(reset_opt=False)
        else:
            self.model = self.model_adaptation()
            self.make_optimizer(reset_opt=False)


__all__ = ["BatchObservation"]
