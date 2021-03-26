from abc import ABC
from typing import Generic, TypeVar

CallbackResult = TypeVar('CallbackResult')


class StrategyCallbacks(Generic[CallbackResult], ABC):
    """
    Base class for all classes dealing with strategy callbacks. Implements all
    the callbacks of the BaseStrategy with an empty function.
    Subclasses must override the desired callbacks.

    The main two direct subclasses are :class:`StrategyPlugin`, which are used
    to implement continual strategies, and :class:`StrategyLogger`, which are
    used for logging.

    **Training loop**
    The training loop and its callbacks are organized as follows::
        train
            for exp in stream:
                data_adaptation
                for epoch in range(n_epochs):
                    for mbatch in dataloader:
                        forward
                        backward
                        update

            before_training
            before_training_exp
            adapt_train_dataset
            make_train_dataloader
            before_training_epoch
                before_training_iteration
                    before_forward
                    after_forward
                    before_backward
                    after_backward
                after_training_iteration
                before_update
                after_update
            after_training_epoch
            after_training_exp
            after_training

    **Evaluation loop**
    The evaluation loop and its callbacks are organized as follows::
        eval
            before_eval
            adapt_eval_dataset
            make_eval_dataloader
            before_eval_exp
                eval_epoch
                    before_eval_iteration
                    before_eval_forward
                    after_eval_forward
                    after_eval_iteration
            after_eval_exp
            after_eval
    """

    def __init__(self):
        pass

    def before_training(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_training_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_train_dataset_adaptation(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_train_dataset_adaptation(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_training_epoch(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_training_iteration(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_backward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_backward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training_iteration(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_update(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_update(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training_epoch(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval_dataset_adaptation(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval_dataset_adaptation(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval_iteration(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval_iteration(self, *args, **kwargs) -> CallbackResult:
        pass
