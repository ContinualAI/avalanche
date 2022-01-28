from avalanche.core import SupervisedStrategyCallbacks
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.skeletons.supervised import SupervisedStrategy


class StrategyPlugin(SupervisedStrategyCallbacks[Any]):
    """
    Base class for strategy plugins. Implements all the callbacks required
    by the BaseStrategy with an empty function. Subclasses should override
    the callbacks.
    """

    def __init__(self):
        super().__init__()
        pass

    def before_training(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_training_exp(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_train_dataset_adaptation(
        self, strategy: "SupervisedStrategy", **kwargs
    ):
        pass

    def after_train_dataset_adaptation(
        self, strategy: "SupervisedStrategy", **kwargs
    ):
        pass

    def before_training_epoch(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_training_iteration(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_forward(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_forward(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_backward(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_backward(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_training_iteration(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_update(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_update(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_training_epoch(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_training_exp(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_training(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_eval(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_eval_dataset_adaptation(
        self, strategy: "SupervisedStrategy", **kwargs
    ):
        pass

    def after_eval_dataset_adaptation(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_eval_exp(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_eval_exp(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_eval(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_eval_iteration(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def before_eval_forward(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_eval_forward(self, strategy: "SupervisedStrategy", **kwargs):
        pass

    def after_eval_iteration(self, strategy: "SupervisedStrategy", **kwargs):
        pass
