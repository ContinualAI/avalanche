import warnings

import torch
import torch.nn.functional as F

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.models.utils import avalanche_forward


class RWalkPlugin(SupervisedPlugin):
    """
    Riemannian Walk (RWalk) plugin.
    RWalk computes the importance of each weight at the end of every training
    iteration, and updates each parameter's importance online using moving
    average. During training on each minibatch, the loss is augmented
    with a penalty which keeps the value of a parameter close to the
    value it had on previous experiences, proportional to a score that has
    high values if small changes cause large improvements in the loss.
    This plugin does not use task identities.

    .. note::
        To reproduce the results of the paper in class-incremental scenarios,
        this plug-in should be used in conjunction with a replay strategy
        (e.g., :class:`ReplayPlugin`).
    """

    def __init__(
        self, ewc_lambda: float = 0.1, ewc_alpha: float = 0.9, delta_t: int = 10
    ):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param ewc_alpha: Specify the moving average factor for the importance
                matrix, as defined RWalk paper (a.k.a. EWC++). Higher values
                lead to higher weight to newly computed importances. Must be
                in [0, 1]. Defaults to 0.9.
        :param delta_t: Specify the iterations interval in which the parameter
                scores are updated. Defaults to 10.
        """

        super().__init__()

        assert 0 <= ewc_alpha <= 1, "`ewc_alpha` must be in [0, 1]."
        assert delta_t >= 1, "`delta_t` must be at least 1."

        self.ewc_alpha = ewc_alpha
        self.ewc_lambda = ewc_lambda
        self.delta_t = delta_t

        # Information computed every delta_t
        self.checkpoint_params = None
        self.checkpoint_loss = None

        # Partial scores (s_t1^t2) computed incrementally every delta_t
        self.checkpoint_scores = None

        # Information stored at the beginning of every iteration
        self.iter_grad = None
        self.iter_importance = None
        self.iter_params = None

        # Information stored at the end of every experience (t_k in the paper)
        self.exp_scores = None
        self.exp_params = None
        self.exp_importance = None
        self.exp_penalties = None

    def _is_checkpoint_iter(self, strategy):
        return (strategy.clock.train_iterations + 1) % self.delta_t == 0

    # Compute the criterion gradient (without penalties)
    def _update_grad(self, strategy):
        model = strategy.model
        batch = strategy.mbatch

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if strategy.device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        x, y, task_labels = batch[0], batch[1], batch[-1]

        strategy.optimizer.zero_grad()
        out = avalanche_forward(model, x, task_labels)
        loss = strategy._criterion(out, y)  # noqa
        loss.backward()

        self.iter_grad = copy_params_dict(model, copy_grad=True)

    # Update the first-order Taylor approximation of the loss variation
    # for a single iteration (Eq. 7 of the RWalk paper for a single t)
    @torch.no_grad()
    def _update_loss(self, strategy):
        for (k1, old_p), (k2, new_p), (k3, p_grad), (k4, p_loss) in zip(
            self.iter_params,
            strategy.model.named_parameters(),
            self.iter_grad,
            self.checkpoint_loss,
        ):
            assert k1 == k2 == k3 == k4, "Error in delta-loss approximation."
            p_loss -= p_grad * (new_p - old_p)

    # Update parameter importance (EWC++, Eq. 6 of the RWalk paper)
    def _update_importance(self, strategy):
        importance = [
            (k, p.grad.data.clone().pow(2))
            for k, p in strategy.model.named_parameters()
        ]

        if self.iter_importance is None:
            self.iter_importance = importance
        else:
            old_importance = self.iter_importance
            self.iter_importance = []

            for (k1, old_imp), (k2, new_imp) in zip(old_importance, importance):
                assert k1 == k2, "Error in importance computation."
                self.iter_importance.append(
                    (
                        k1,
                        (
                            self.ewc_alpha * new_imp
                            + (1 - self.ewc_alpha) * new_imp
                        ),
                    )
                )

    # Add scores for a single delta_t (referred to as s_t1^t2 in the paper)
    @torch.no_grad()
    def _update_score(self, strategy):
        for (k1, score), (k2, loss), (k3, imp), (k4, old_p), (k5, new_p) in zip(
            self.checkpoint_scores,
            self.checkpoint_loss,
            self.iter_importance,
            self.checkpoint_params,
            strategy.model.named_parameters(),
        ):
            assert (
                k1 == k2 == k3 == k4 == k5
            ), "Error in RWalk score computation."

            eps = torch.finfo(loss.dtype).eps
            score += loss / (0.5 * imp * (new_p - old_p).pow(2) + eps)

    # Initialize t_0 checkpoint information
    def before_training(self, strategy, *args, **kwargs):
        self.checkpoint_loss = zerolike_params_dict(strategy.model)
        self.checkpoint_scores = zerolike_params_dict(strategy.model)
        self.checkpoint_params = copy_params_dict(strategy.model)

    # Compute variables at t step, at the end of the iteration
    # it will be used to compute delta variations (t+1 - t)
    def before_training_iteration(self, strategy, *args, **kwargs):
        self._update_grad(strategy)
        self._update_importance(strategy)
        self.iter_params = copy_params_dict(strategy.model)

    # Add loss penalties (Eq. 8 in the RWalk paper)
    def before_backward(self, strategy, *args, **kwargs):
        if strategy.clock.train_exp_counter > 0:
            ewc_loss = 0

            for (k1, penalty), (k2, param_exp), (k3, param) in zip(
                self.exp_penalties,
                self.exp_params,
                strategy.model.named_parameters(),
            ):
                assert k1 == k2 == k3, "Error in RWalk loss computation."

                ewc_loss += (penalty * (param - param_exp).pow(2)).sum()

            strategy.loss += self.ewc_lambda * ewc_loss

    # Compute data at (t+1) time step. If delta_t steps has passed, update
    # also checkpoint data
    def after_training_iteration(self, strategy, *args, **kwargs):
        self._update_loss(strategy)

        if self._is_checkpoint_iter(strategy):
            self._update_score(strategy)

            self.checkpoint_loss = zerolike_params_dict(strategy.model)
            self.checkpoint_params = copy_params_dict(strategy.model)

    # Update experience information
    def after_training_exp(self, strategy, *args, **kwargs):
        self.exp_importance = self.iter_importance
        self.exp_params = copy_params_dict(strategy.model)

        if self.exp_scores is None:
            self.exp_scores = self.checkpoint_scores
        else:
            exp_scores = []

            for (k1, p_score), (k2, p_cp_score) in zip(
                self.exp_scores, self.checkpoint_scores
            ):
                assert k1 == k2, "Error in RWalk score computation."
                exp_scores.append((k1, 0.5 * (p_score + p_cp_score)))

            self.exp_scores = exp_scores

        # Compute weight penalties once for all successive iterations
        # (t_k+1 variables remain constant in Eq. 8 in the paper)
        self.exp_penalties = []

        # Normalize terms in [0,1] interval, as suggested in the paper
        # (the importance is already > 0, while negative scores are relu-ed
        # out, hence we scale only the max-values of both terms)
        max_score = max(map(lambda x: x[1].max(), self.exp_scores))
        max_imp = max(map(lambda x: x[1].max(), self.exp_importance))

        for (k1, imp), (k2, score) in zip(self.exp_importance, self.exp_scores):
            assert k1 == k2, "Error in RWalk penalties computation."

            self.exp_penalties.append(
                (k1, imp / max_imp + F.relu(score) / max_score)
            )

        self.checkpoint_scores = zerolike_params_dict(strategy.model)
