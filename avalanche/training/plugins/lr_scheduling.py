import warnings
from typing import TYPE_CHECKING, Literal

from avalanche.evaluation.metrics import Mean
from avalanche.training.plugins import SupervisedPlugin
import inspect

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class LRSchedulerPlugin(SupervisedPlugin):
    """Learning Rate Scheduler Plugin.

    This plugin manages learning rate scheduling inside of a strategy using the
    PyTorch scheduler passed to the constructor. The step() method of the
    scheduler is called after each training epoch or iteration.

    Metric-based schedulers (like ReduceLROnPlateau) are supported as well.
    """

    def __init__(
        self,
        scheduler,
        reset_scheduler=True,
        reset_lr=True,
        metric=None,
        step_granularity: Literal["epoch", "iteration"] = "epoch",
        first_epoch_only=False,
        first_exp_only=False,
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
        :param step_granularity: defines how often the scheduler's `step()`
            method will be called. Defaults to 'epoch'. Valid values are
            'epoch' and 'iteration'.
        :param first_epoch_only: if True, the scheduler will only be stepped
            in the first epoch of each training experience. This is not mutually
            exclusive with `first_exp_only`: by setting both values to True,
            the scheduler will be stepped only in the very first epoch of the
            whole training stream.
        :param first_exp_only: if True, the scheduler will only be considered
            in the first training experience.
        """

        super().__init__()
        self.scheduler = scheduler
        self.reset_scheduler = reset_scheduler
        self.reset_lr = reset_lr
        self.metric = metric
        self.rolling_metric = Mean()
        self.step_granularity = step_granularity
        self.first_epoch_only = first_epoch_only
        self.first_exp_only = first_exp_only

        # Used to detect and manage the periodic eval phase
        self._was_training = False
        self._just_validated = False
        self._executed_train_iteration = False

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

        if self.step_granularity not in ["iteration", "epoch"]:
            raise ValueError(
                "Wrong value of step_granularity: valid values are "
                '"iteration" and "epoch"'
            )

        LRSchedulerPlugin._patch_lr_on_plateau(self.scheduler)

    def after_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):
        if self.step_granularity == "epoch" and self.metric in [
            None,
            "train_loss",
        ]:
            self._step_scheduler(strategy, **kwargs)

    def before_training_iteration(self, strategy, **kwargs):
        self._just_validated = False

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
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
    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        self._was_training = True

    def after_training(self, strategy: "SupervisedTemplate", **kwargs):
        self._was_training = False

    def before_training_exp(self, strategy, *args, **kwargs):
        self._executed_train_iteration = False

    def after_eval(self, strategy: "SupervisedTemplate", **kwargs):
        if self.metric == "val_loss" and self._was_training:
            if not self._executed_train_iteration:
                # The base strategy may run an evaluation pass on the
                # validation set before running the training loop. In that
                # case, we should just discard the result.
                # print('Ignoring pre-training validation')
                pass
            elif self._just_validated:
                # The base strategy may run an evaluation pass on the
                # validation set after the training loop. In that
                # case, we should discard the result only if the validation pass
                # has been duplicated.
                # print('Ignoring, as just validated')
                pass
            else:
                # print('Stepping after validation',
                #       self.rolling_metric.result())
                self._step_scheduler(strategy, **kwargs)
            self.rolling_metric.reset()

        self._just_validated = True

    def after_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        self._executed_train_iteration = True

        if self.metric == "train_loss":
            self.rolling_metric.update(strategy.loss, weight=len(strategy.mb_x))

        if self.step_granularity == "iteration" and self.metric in [
            None,
            "train_loss",
        ]:
            self._step_scheduler(strategy, **kwargs)

    def after_eval_iteration(self, strategy: "SupervisedTemplate", **kwargs):
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

    def _check_first_epoch_or_experience(self, strategy):
        if strategy.clock.train_exp_counter > 0 and self.first_exp_only:
            return False

        if strategy.clock.train_exp_epochs > 0 and self.first_epoch_only:
            return False

        return True

    def _step_scheduler(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.is_training:
            if self._check_first_epoch_or_experience(strategy):
                if self.metric == "train_loss":
                    self.scheduler.step(metrics=self.rolling_metric.result())
                elif self.metric != "val_loss":
                    self.scheduler.step()

            if self.metric == "train_loss" or self.metric != "val_loss":
                self.rolling_metric.reset()
        else:
            # Validating
            if self._check_first_epoch_or_experience(strategy):
                self.scheduler.step(metrics=self.rolling_metric.result())


__all__ = ["LRSchedulerPlugin"]
