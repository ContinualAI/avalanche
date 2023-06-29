import copy
from typing import TYPE_CHECKING, Optional

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class GDumbPlugin(SupervisedPlugin, supports_distributed=True):
    """GDumb plugin.

    At each experience the model is trained  from scratch using a buffer of
    samples collected from all the previous learning experiences.
    The buffer is updated at the start of each experience to add new classes or
    new examples of already encountered classes.
    In multitask scenarios, mem_size is the memory size for each task.
    This plugin can be combined with a Naive strategy to obtain the
    standard GDumb strategy.
    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """

    def __init__(self, mem_size: int = 200):
        super().__init__()
        self.mem_size = mem_size

        # model initialization
        # self.buffer = {}  # TODO: remove buffer
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.init_model = None

    def before_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        """Reset model."""
        if self.init_model is None:
            self.init_model = copy.deepcopy(strategy.model)
        else:
            strategy.model = copy.deepcopy(self.init_model)
        strategy.model_adaptation(self.init_model)

    def before_eval_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model_adaptation(self.init_model)

    def after_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        strategy.adapted_dataset = self.storage_policy.buffer


__all__ = ["GDumbPlugin"]
