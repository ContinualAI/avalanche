import copy

import torch

from avalanche.models import avalanche_forward, MultiTaskModule
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class LwFPlugin(SupervisedPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None

        self.prev_classes = {"0": set()}
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding to old classes. 
        """

    def _distillation_loss(self, out, prev_out, active_units):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)
        log_p = torch.log_softmax(out / self.temperature, dim=1)[:, au]
        q = torch.softmax(prev_out / self.temperature, dim=1)[:, au]
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res

    def penalty(self, out, x, alpha, curr_model):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
            with torch.no_grad():
                if isinstance(self.prev_model, MultiTaskModule):
                    # output from previous output heads.
                    y_prev = avalanche_forward(self.prev_model, x, None)
                    # in a multitask scenario we need to compute the output
                    # from all the heads, so we need to call forward again.
                    # TODO: can we avoid this?
                    y_curr = avalanche_forward(curr_model, x, None)
                else:  # no task labels
                    y_prev = {"0": self.prev_model(x)}
                    y_curr = {"0": out}

            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads.
                if task_id in self.prev_classes:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
                    au = self.prev_classes[task_id]
                    dist_loss += self._distillation_loss(yc, yp, au)
            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        alpha = (
            self.alpha[strategy.clock.train_exp_counter]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )
        penalty = self.penalty(
            strategy.mb_output, strategy.mb_x, alpha, strategy.model
        )
        strategy.loss += penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        self.prev_model = copy.deepcopy(strategy.model)
        task_ids = strategy.experience.dataset.task_set
        for task_id in task_ids:
            task_data = strategy.experience.dataset.task_set[task_id]
            pc = set(task_data.targets)

            if task_id not in self.prev_classes:
                self.prev_classes[str(task_id)] = pc
            else:
                self.prev_classes[str(task_id)] = self.prev_classes[
                    task_id
                ].union(pc)
