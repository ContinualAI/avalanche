import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.nn import Linear

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import (
    examples_per_class,
    get_last_fc_layer,
    get_layer_by_name,
    freeze_everything,
    unfreeze_everything,
)


class CWRStarPlugin(SupervisedPlugin):
    """CWR* Strategy.

    This plugin does not use task identities.
    """

    def __init__(self, model, cwr_layer_name=None, freeze_remaining_model=True):
        """
        :param model: the model.
        :param cwr_layer_name: name of the last fully connected layer. Defaults
            to None, which means that the plugin will attempt an automatic
            detection.
        :param freeze_remaining_model: If True, the plugin will freeze (set
            layers in eval mode and disable autograd for parameters) all the
            model except the cwr layer. Defaults to True.
        """
        super().__init__()
        self.log = logging.getLogger("avalanche")
        self.model = model
        self.cwr_layer_name = cwr_layer_name
        self.freeze_remaining_model = freeze_remaining_model

        # Model setup
        self.model.saved_weights = {}
        self.model.past_j = defaultdict(int)
        self.model.cur_j = defaultdict(int)

        # to be updated
        self.cur_class = []

    def after_training_exp(self, strategy, **kwargs):
        self.consolidate_weights()
        self.set_consolidate_weights()

    def before_training_exp(self, strategy, **kwargs):
        if self.freeze_remaining_model and strategy.clock.train_exp_counter > 0:
            self.freeze_other_layers()

        # Count current classes and number of samples for each of them.
        data = strategy.experience.dataset
        self.model.cur_j = examples_per_class(data.targets)
        self.cur_class = [
            cls for cls in set(self.model.cur_j.keys()) if self.model.cur_j[cls] > 0
        ]

        self.reset_weights(self.cur_class)

    def consolidate_weights(self):
        """Mean-shift for the target layer weights"""

        with torch.no_grad():
            cwr_layer = self.get_cwr_layer()
            assert cwr_layer is not None, "Could not find the CWR layer."
            # calculate the average of the current classes
            globavg = np.average(
                cwr_layer.weight.detach().cpu().numpy()[self.cur_class]
            )
            for c in self.cur_class:
                w = cwr_layer.weight.detach().cpu().numpy()[c]
                # subtract the weight average to the weights
                # to obtain zero mean
                new_w = w - globavg
                # if the class has been already seen
                if c in self.model.saved_weights.keys():
                    wpast_j = np.sqrt(self.model.past_j[c] / self.model.cur_j[c])
                    # consolidation
                    self.model.saved_weights[c] = (
                        self.model.saved_weights[c] * wpast_j + new_w
                    ) / (wpast_j + 1)
                    self.model.past_j[c] += self.model.cur_j[c]
                else:
                    # new class
                    self.model.saved_weights[c] = new_w
                    self.model.past_j[c] = self.model.cur_j[c]

    def set_consolidate_weights(self):
        """set trained weights"""

        with torch.no_grad():
            cwr_layer = self.get_cwr_layer()
            assert cwr_layer is not None, "Could not find the CWR layer."

            for c, w in self.model.saved_weights.items():
                cwr_layer.weight[c].copy_(torch.from_numpy(self.model.saved_weights[c]))

    def reset_weights(self, cur_clas):
        """reset weights"""
        with torch.no_grad():
            cwr_layer = self.get_cwr_layer()
            assert cwr_layer is not None, "Could not find the CWR layer."

            cwr_layer.weight.fill_(0.0)
            for c, w in self.model.saved_weights.items():
                if c in cur_clas:
                    cwr_layer.weight[c].copy_(
                        torch.from_numpy(self.model.saved_weights[c])
                    )

    def get_cwr_layer(self) -> Optional[Linear]:
        result = None
        if self.cwr_layer_name is None:
            last_fc = get_last_fc_layer(self.model)
            if last_fc is not None:
                result = last_fc[1]
        else:
            result = get_layer_by_name(self.model, self.cwr_layer_name)

        return result

    def freeze_other_layers(self):
        cwr_layer = self.get_cwr_layer()
        if cwr_layer is None:
            raise RuntimeError("Can't find a the Linear layer")
        freeze_everything(self.model)
        unfreeze_everything(cwr_layer)
