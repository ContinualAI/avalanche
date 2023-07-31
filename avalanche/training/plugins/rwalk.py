from typing import Dict, Optional
import warnings

import torch
import torch.nn.functional as F

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData
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
        self.checkpoint_params: Dict[str, ParamData] = dict()
        self.checkpoint_loss: Dict[str, ParamData] = dict()

        # Partial scores (s_t1^t2) computed incrementally every delta_t
        self.checkpoint_scores: Dict[str, ParamData] = dict()

        # Information stored at the beginning of every iteration
        self.iter_grad: Dict[str, ParamData] = dict()
        self.iter_importance: Optional[Dict[str, ParamData]] = None
        self.iter_params: Dict[str, ParamData] = dict()

        # Information stored at the end of every experience (t_k in the paper)
        self.exp_scores: Optional[Dict[str, ParamData]] = None
        self.exp_params: Optional[Dict[str, ParamData]] = None
        self.exp_importance: Optional[Dict[str, ParamData]] = None
        self.exp_penalties: Optional[Dict[str, ParamData]] = None

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
        for k, new_p in strategy.model.named_parameters():
            if k not in self.iter_grad:
                continue
            old_p = self.iter_params[k]
            p_grad = self.iter_grad[k]
            shape = new_p.shape
            self.checkpoint_loss[k].expand(shape)
            self.checkpoint_loss[k].data -= p_grad.expand(shape) * (
                new_p - old_p.expand(shape)
            )

    # Update parameter importance (EWC++, Eq. 6 of the RWalk paper)
    def _update_importance(self, strategy):
        importance = copy_params_dict(strategy.model, copy_grad=True)
        for k in importance.keys():
            importance[k].data = importance[k].data ** 2

        if self.iter_importance is None:
            self.iter_importance = importance
        else:
            old_importance = self.iter_importance
            self.iter_importance = {}

            for k, new_imp in importance.items():
                if k not in old_importance:
                    self.iter_importance[k] = ParamData(
                        k, device=new_imp.device, init_tensor=new_imp.data
                    )
                else:
                    old_imp = old_importance[k]
                    self.iter_importance[k] = ParamData(
                        k,
                        device=new_imp.device,
                        init_tensor=self.ewc_alpha * new_imp.data
                        + (1 - self.ewc_alpha) * old_imp.expand(new_imp.shape),
                    )

    # Add scores for a single delta_t (referred to as s_t1^t2 in the paper)
    @torch.no_grad()
    def _update_score(self, strategy):
        assert self.iter_importance is not None
        for k, new_p in strategy.model.named_parameters():
            if k not in self.iter_importance:
                # new params do not count
                continue
            loss = self.checkpoint_loss[k]
            imp = self.iter_importance[k]
            old_p = self.checkpoint_params[k]

            shape = new_p.shape
            eps = torch.finfo(loss.data.dtype).eps
            self.checkpoint_scores[k].expand(shape)
            self.checkpoint_scores[k].data += loss.data / (
                0.5 * imp.expand(shape) * (new_p - old_p.expand(shape)).pow(2) + eps
            )

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
            assert self.exp_penalties is not None
            assert self.exp_params is not None
            ewc_loss = 0

            for k, param in strategy.model.named_parameters():
                if k not in self.exp_penalties:
                    # new params do not count
                    continue
                penalty = self.exp_penalties[k]
                param_exp = self.exp_params[k]
                ewc_loss += (
                    penalty.expand(param.shape)
                    * (param - param_exp.expand(param.shape)).pow(2)
                ).sum()

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
        assert self.iter_importance is not None
        self.exp_importance = self.iter_importance
        self.exp_params = copy_params_dict(strategy.model)

        if self.exp_scores is None:
            self.exp_scores = self.checkpoint_scores
        else:
            exp_scores = {}

            for k, p_cp_score in self.checkpoint_scores.items():
                if k not in self.exp_scores:
                    continue
                p_score = self.exp_scores[k]
                shape = p_cp_score.data.shape
                exp_scores[k] = ParamData(
                    k,
                    device=p_score.device,
                    init_tensor=0.5 * (p_score.expand(shape) + p_cp_score.data),
                )
            self.exp_scores = exp_scores

        # Compute weight penalties once for all successive iterations
        # (t_k+1 variables remain constant in Eq. 8 in the paper)
        self.exp_penalties = {}

        # Normalize terms in [0,1] interval, as suggested in the paper
        # (the importance is already > 0, while negative scores are relu-ed
        # out, hence we scale only the max-values of both terms)
        max_score = max(map(lambda x: x.data.max(), self.exp_scores.values()))
        max_imp = max(map(lambda x: x.data.max(), self.exp_importance.values()))

        for k, score in self.exp_scores.items():
            if k not in self.exp_importance:
                continue  # some params may not have gradients
            imp = self.exp_importance[k]
            shape = imp.data.shape
            self.exp_penalties[k] = ParamData(
                k,
                device=imp.device,
                init_tensor=imp.data / max_imp
                + F.relu(score.expand(shape)) / max_score,
            )

        self.checkpoint_scores = zerolike_params_dict(strategy.model)
