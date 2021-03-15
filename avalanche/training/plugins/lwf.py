import copy

import torch

from avalanche.training.plugins.strategy_plugin import StrategyPlugin


class LwFPlugin(StrategyPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
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

    def _distillation_loss(self, out, prev_out):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """

        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

    def penalty(self, out, x, alpha):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
            y_prev = self.prev_model(x).detach()
            dist_loss = self._distillation_loss(out, y_prev)
            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        alpha = self.alpha[strategy.training_exp_counter] \
            if isinstance(self.alpha, (list, tuple)) else self.alpha
        penalty = self.penalty(strategy.logits, strategy.mb_x, alpha)
        strategy.loss += penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """

        self.prev_model = copy.deepcopy(strategy.model)