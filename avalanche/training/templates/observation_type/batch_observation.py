import warnings
from collections import defaultdict
from typing import List, TypeVar

from torch import Tensor

from avalanche.benchmarks import OnlineCLExperience
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.templates.strategy_mixin_protocol import \
    SGDStrategyProtocol
from avalanche.models.dynamic_optimizers import (reset_optimizer, 
                                                 update_optimizer)


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

    def make_optimizer(self, reset_optimizer_state=False, **kwargs):
        """Optimizer initialization.

        Called before each training experience to configure the optimizer.

        :param reset_optimizer_state: bool, whether to reset the 
            state of the optimizer, defaults to False

        Warnings: 
            - The first time this function is called 
              for a given strategy it will reset the
              optimizer to gather the (name, param) 
              correspondance of the optimized parameters
              all the model parameters will be put in the
              optimizer, regardless of what parameters are 
              initially put in the optimizer.
        """
        if self.optimized_param_id is None:
            self.optimized_param_id = \
                    reset_optimizer(self.optimizer, self.model) 
        else:
            self.optimized_param_id = \
                update_optimizer(
                    self.optimizer, 
                    dict(self.model.named_parameters()),
                    self.optimized_param_id, 
                    reset_state=reset_optimizer_state
                    )

    def check_model_and_optimizer(self, reset_optimizer_state=False, **kwargs):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        assert self.experience is not None

        if self.optimized_param_id is None:
            self.make_optimizer(reset_optimizer_state=True)

        if isinstance(self.experience, OnlineCLExperience):
            if self.experience.access_task_boundaries:
                if self.experience.is_first_subexp:
                    self.model = self.model_adaptation()
                    self.make_optimizer(
                        reset_optimizer_state=reset_optimizer_state
                    )
            else:
                self.model = self.model_adaptation()
                self.make_optimizer(reset_optimizer_state=reset_optimizer_state)
        else:
            self.model = self.model_adaptation()
            self.make_optimizer(reset_optimizer_state=reset_optimizer_state)


__all__ = ["BatchObservation"]
