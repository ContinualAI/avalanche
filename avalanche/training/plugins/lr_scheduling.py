import warnings
from typing import TYPE_CHECKING

from avalanche.evaluation.metrics import Mean
from avalanche.training.plugins import StrategyPlugin
import inspect

if TYPE_CHECKING:
    from avalanche.training.skeletons.supervised import BaseStrategy


class LRSchedulerPlugin(StrategyPlugin):
    """Learning Rate Scheduler Plugin.

    This plugin manages learning rate scheduling inside of a strategy using the
    PyTorch scheduler passed to the constructor. The step() method of the
    scheduler is called after each training epoch.

    Metric-based schedulers (like ReduceLROnPlateau) are supported as well.
    """

    def __init__(
        self, scheduler, reset_scheduler=True, reset_lr=True, metric=None
    ):
        """
        Creates a ``LRSchedulerPlugin`` instance.

        :param scheduler: a learning rate scheduler that can be updated through
            a step() method and can be reset by setting last_epoch=0.
        :param reset_scheduler: If True, the scheduler is reset at the end of
            the experience. Defaults to True.
        :param reset_lr: If True, the optimizer learning rate is reset to its
            original value. Default to True.
        :param metric: the metric to use. Must be set when using
            metric-based scheduling (like ReduceLROnPlateau). Only "train_loss"
            and "val_loss" are supported at the moment. Beware that,
            when using "val_loss", the periodic evaluation flow must be enabled
            in the strategy. By default, the `eval_every` parameter of the
            base strategy is -1, which means that the validation set is never
            evaluated. Set that value to 1 to obtain the correct results.
            Also, when using `metric="val_loss"`, remember to pass a proper
            validation stream to the strategy train method, otherwise the
            periodic evaluation stream will use the training set to compute
            the validation loss.
        """

        super().__init__()
        self.scheduler = scheduler
        self.reset_scheduler = reset_scheduler
        self.reset_lr = reset_lr
        self.metric = metric
        self.rolling_metric = Mean()

        # Used to detect and manage the periodic eval phase
        self._was_training = False
        self._eval_train_epoch = 0

        arg_names = inspect.getfullargspec(self.scheduler.step)[0]
        needs_metrics = "metrics" in arg_names

        if needs_metrics and self.metric is None:
            raise ValueError(
                "The step method of this scheduler requires a metric "
                "(usually the loss) to be passed. Please set a proper "
                "metric parameter when creating this plugin."
            )
        elif (not needs_metrics) and self.metric is not None:
            warnings.warn(
                "You are passing a metric value but the scheduler"
                "doesn't seem to support metrics..."
            )

        if self.metric not in [None, "train_loss", "val_loss"]:
            raise ValueError(
                'Only scheduling based on "train_loss" and '
                "val_loss"
                ""
                f"is supported at the moment (got {metric}."
            )

        LRSchedulerPlugin._patch_lr_on_plateau(self.scheduler)

    def after_training_epoch(self, strategy: "BaseStrategy", **kwargs):
        if self.metric == "train_loss":
            self.scheduler.step(metrics=self.rolling_metric.result())
            self.rolling_metric.reset()
        elif self.metric != "val_loss":
            self.scheduler.step()
            self.rolling_metric.reset()

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        param_groups = strategy.optimizer.param_groups
        base_lrs = self.scheduler.base_lrs

        if self.reset_lr:
            for group, lr in zip(param_groups, base_lrs):
                group["lr"] = lr

        if self.reset_scheduler:
            self.scheduler.last_epoch = 0

            # Manage the reset of the scheduler
            # Mainly used to call _reset on ReduceLROnPlateau, but may come
            # in handy for other schedulers in the future
            reset_method = getattr(self.scheduler, "reset", None)
            if not callable(reset_method):
                reset_method = getattr(self.scheduler, "_reset", None)

            if callable(reset_method):
                # print('Calling reset method of scheduler')
                reset_method()

    # Methods used to manage ReduceLROnPlateau (keep track of the periodic eval)
    def before_training(self, strategy: "BaseStrategy", **kwargs):
        self._was_training = True

    def after_training(self, strategy: "BaseStrategy", **kwargs):
        self._was_training = False

    def after_eval(self, strategy: "BaseStrategy", **kwargs):

        if self.metric == "val_loss" and self._was_training:

            if strategy.clock.train_exp_epochs == 0:
                # The base strategy may run an evaluation pass on the
                # validation set before running the training loop. In that
                # case, we should just discard the result.
                # print('Ignoring pre-training validation')
                pass
            elif self._eval_train_epoch == strategy.clock.train_exp_epochs:
                # The base strategy may run an evaluation pass on the
                # validation set after the training loop. In that
                # case, we should discard the result only if the validation pass
                # has been duplicated.

                # In fact, the previous branch of the "if" could be omitted
                # because this one can cover both the pre-training and
                # duplicate post-training cases...
                # print('Ignoring post-training duplicate validation '
                #      f'{self._eval_train_epoch}')
                pass
            else:
                # print('Stepping after validation',
                #       self.rolling_metric.result())
                self.scheduler.step(metrics=self.rolling_metric.result())
            self.rolling_metric.reset()
        self._eval_train_epoch = strategy.clock.train_exp_epochs

    def after_training_iteration(self, strategy: "BaseStrategy", **kwargs):
        if self.metric != "train_loss":
            return
        self.rolling_metric.update(strategy.loss, weight=len(strategy.mb_x))

    def after_eval_iteration(self, strategy: "BaseStrategy", **kwargs):
        if self.metric != "val_loss":
            return

        # Check if switched to eval mid-training
        # This only happens when running periodic validation
        if self._was_training:
            self.rolling_metric.update(strategy.loss, weight=len(strategy.mb_x))

    @staticmethod
    def _patch_lr_on_plateau(scheduler):
        # All PyTorch schedulers have the base_lrs field (needed to reset the
        # initial LRs before each experience) with the only exception being
        # ReduceLROnPlateau. This method will add that field to
        # ReduceLROnPlateau.

        if hasattr(scheduler, "base_lrs"):
            return

        # Initialize epoch and base learning rates
        for group in scheduler.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        scheduler.base_lrs = list(
            map(
                lambda group_param: group_param["initial_lr"],
                scheduler.optimizer.param_groups,
            )
        )


__all__ = ["LRSchedulerPlugin"]
