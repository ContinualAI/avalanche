"""Regularization methods."""

import copy
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F

from avalanche.models import MultiTaskModule, avalanche_forward
from avalanche._annotations import deprecated


def stable_softmax(x):
    z = x - torch.max(x, dim=1, keepdim=True)[0]
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    softmax = numerator / denominator
    return softmax


def cross_entropy_with_oh_targets(outputs, targets, reduction="mean"):
    """Calculates cross-entropy with temperature scaling,
    targets can also be soft targets but they must sum to 1"""
    outputs = stable_softmax(outputs)
    ce = -(targets * outputs.log()).sum(1)
    if reduction == "mean":
        ce = ce.mean()
    elif reduction == "none":
        return ce
    else:
        raise NotImplementedError("reduction must be mean or none")
    return ce


class RegularizationMethod:
    """RegularizationMethod implement regularization strategies.
    RegularizationMethod is a callable.
    The method `update` is called to update the loss, typically at the end
    of an experience.
    """

    @deprecated(0.7, "please switch to pre_update and post_update methods.")
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def pre_adapt(self, agent, exp):
        pass  # implementation may be empty if adapt is not needed

    def post_adapt(self, agent, exp):
        pass  # implementation may be empty if adapt is not needed

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class LearningWithoutForgetting(RegularizationMethod):
    """Learning Without Forgetting.

    The method applies knowledge distilllation to mitigate forgetting.
    The teacher is the model checkpoint after the last experience.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """
        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None
        self.expcount = 0
        # count number of experiences (used to increase alpha)
        self.prev_classes_by_task = defaultdict(set)
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding
        to old classes. 
        """

    def _distillation_loss(self, out, prev_out, active_units):
        """Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)

        # some people use the crossentropy instead of the KL
        # They are equivalent. We compute
        # kl_div(log_p_curr, p_prev) = p_prev * (log (p_prev / p_curr)) =
        #   p_prev * log(p_prev) - p_prev * log(p_curr).
        # Now, the first term is constant (we don't optimize the teacher),
        # so optimizing the crossentropy and kl-div are equivalent.
        log_p = torch.log_softmax(out[:, au] / self.temperature, dim=1)
        q = torch.softmax(prev_out[:, au] / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res

    def _lwf_penalty(self, out, x, curr_model):
        """
        Compute weighted distillation loss.
        """
        if self.prev_model is None:
            return 0
        else:
            if isinstance(self.prev_model, MultiTaskModule):
                # output from previous output heads.
                with torch.no_grad():
                    y_prev = avalanche_forward(self.prev_model, x, None)
                y_prev = {k: v for k, v in y_prev.items()}
                # in a multitask scenario we need to compute the output
                # from all the heads, so we need to call forward again.
                # TODO: can we avoid this?
                y_curr = avalanche_forward(curr_model, x, None)
                y_curr = {k: v for k, v in y_curr.items()}
            else:  # no task labels. Single task LwF
                with torch.no_grad():
                    y_prev = {0: self.prev_model(x)}
                y_curr = {0: out}

            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads and only for seen units.
                if task_id in self.prev_classes_by_task:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
                    au = self.prev_classes_by_task[task_id]
                    dist_loss += self._distillation_loss(yc, yp, au)
            return dist_loss

    def __call__(self, mb_x, mb_pred, model):
        """
        Add distillation loss
        """
        alpha = (
            self.alpha[self.expcount]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )
        return alpha * self._lwf_penalty(mb_pred, mb_x, model)

    def post_adapt(self, agent, exp):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.

        :param agent: agent state
        :param exp: current experience
        """
        self.expcount += 1
        self.prev_model = copy.deepcopy(agent.model)
        task_ids = exp.dataset.targets_task_labels.uniques

        for task_id in task_ids:
            task_data = exp.dataset.task_set[task_id]
            pc = set(task_data.targets.uniques)

            if task_id not in self.prev_classes_by_task:
                self.prev_classes_by_task[task_id] = pc
            else:
                self.prev_classes_by_task[task_id] = self.prev_classes_by_task[
                    task_id
                ].union(pc)

    @deprecated(0.7, "switch to post_udpate method")
    def update(self, experience, model):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.

        :param experience: current experience
        :param model: current model
        """
        self.expcount += 1
        self.prev_model = copy.deepcopy(model)
        task_ids = experience.dataset.targets_task_labels.uniques

        for task_id in task_ids:
            task_data = experience.dataset.task_set[task_id]
            pc = set(task_data.targets.uniques)

            if task_id not in self.prev_classes_by_task:
                self.prev_classes_by_task[task_id] = pc
            else:
                self.prev_classes_by_task[task_id] = self.prev_classes_by_task[
                    task_id
                ].union(pc)


class ACECriterion(RegularizationMethod):
    """
    Asymetric cross-entropy (ACE) Criterion used in
    "New Insights on Reducing Abrupt Representation
    Change in Online Continual Learning"
    by Lucas Caccia et. al.
    https://openreview.net/forum?id=N8MaByOzUfb
    """

    def __init__(self):
        pass

    def __call__(self, out_in, target_in, out_buffer, target_buffer):
        current_classes = torch.unique(target_in)
        loss_buffer = F.cross_entropy(out_buffer, target_buffer)
        oh_target_in = F.one_hot(target_in, num_classes=out_in.shape[1])
        oh_target_in = oh_target_in[:, current_classes]
        loss_current = cross_entropy_with_oh_targets(
            out_in[:, current_classes], oh_target_in
        )
        return (loss_buffer + loss_current) / 2


class AMLCriterion(RegularizationMethod):
    """
    Asymmetric metric learning (AML) Criterion used in
    "New Insights on Reducing Abrupt Representation
    Change in Online Continual Learning"
    by Lucas Caccia et. al.
    https://openreview.net/forum?id=N8MaByOzUfb
    """

    def __init__(
        self,
        feature_extractor,
        temp: float = 0.1,
        base_temp: float = 0.07,
        same_task_neg: bool = True,
        device: str = "cpu",
    ):
        """
        ER_AML criterion constructor.
        :param feature_extractor: Model able to map an input in a latent space.
        :param temp: Supervised contrastive temperature.
        :param base_temp: Supervised contrastive base temperature.
        :param same_task_neg: Option to remove negative samples of different tasks.
        :param device: Accelerator used to speed up the computation.
        """
        self.device = device
        self.feature_extractor = feature_extractor
        self.temp = temp
        self.base_temp = base_temp
        self.same_task_neg = same_task_neg

    def __sample_pos_neg(
        self,
        y_in: torch.Tensor,
        t_in: torch.Tensor,
        x_memory: torch.Tensor,
        y_memory: torch.Tensor,
        t_memory: torch.Tensor,
    ) -> tuple:
        """
        Method able to sample positive and negative examples with respect the input minibatch from input and buffer minibatches.
        :param x_in: Input of new minibatch.
        :param y_in: Output of new minibatch.
        :param t_in: Task ids of new minibatch.
        :param x_memory: Input of memory.
        :param y_memory: Output of minibatch.
        :param t_memory: Task ids of minibatch.
        :return: Tuple of positive and negative input and output examples and a mask for identify invalid values.
        """
        valid_pos = y_in.reshape(1, -1) == y_memory.reshape(-1, 1)
        if self.same_task_neg:
            same_task = t_in.view(1, -1) == t_memory.view(-1, 1)
            valid_neg = ~valid_pos & same_task
        else:
            valid_neg = ~valid_pos

        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)

        pos_x = x_memory[pos_idx]
        pos_y = y_memory[pos_idx]
        neg_x = x_memory[neg_idx]
        neg_y = y_memory[neg_idx]

        return pos_x, pos_y, neg_x, neg_y

    def __sup_con_loss(
        self,
        anchor_features: torch.Tensor,
        features: torch.Tensor,
        anchor_targets: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Method able to compute the supervised contrastive loss of new minibatch.
        :param anchor_features: Anchor features related to new minibatch duplicated mapped in latent space.
        :param features: Features related to half positive and half negative examples mapped in latent space.
        :param anchor_targets: Labels related to anchor features.
        :param targets: Labels related to features.
        :return: Supervised contrastive loss.
        """
        pos_mask = (
            (anchor_targets.reshape(-1, 1) == targets.reshape(1, -1))
            .float()
            .to(self.device)
        )
        similarity = anchor_features @ features.T / self.temp
        similarity -= similarity.max(dim=1)[0].detach()
        log_prob = similarity - torch.log(torch.exp(similarity).sum(1))
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        loss = -(self.temp / self.base_temp) * mean_log_prob_pos.mean()
        return loss

    def __scale_by_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function able to scale by its norm a certain tensor.
        :param x: Tensor to normalize.
        :return: Normalized tensor.
        """
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        return x / (x_norm + 1e-05)

    def __call__(
        self,
        input_in: torch.Tensor,
        target_in: torch.Tensor,
        task_in: torch.Tensor,
        output_buffer: torch.Tensor,
        target_buffer: torch.Tensor,
        pos_neg_replay: tuple,
    ) -> torch.Tensor:
        """
        Method able to compute the ER_AML loss.
        :param input_in: New inputs examples.
        :param target_in: Labels of new examples.
        :param task_in: Task identifiers of new examples.
        :param output_buffer: Predictions of samples from buffer.
        :param target_buffer: Labels of samples from buffer.
        :param pos_neg_replay: Replay data to compute positive and negative samples.
        :return: ER_AML computed loss.
        """
        pos_x, pos_y, neg_x, neg_y = self.__sample_pos_neg(
            target_in, task_in, *pos_neg_replay
        )
        loss_buffer = F.cross_entropy(output_buffer, target_buffer)
        hidden_in = self.__scale_by_norm(self.feature_extractor(input_in))
        hidden_pos_neg = self.__scale_by_norm(
            self.feature_extractor(torch.cat((pos_x, neg_x)))
        )
        loss_in = self.__sup_con_loss(
            anchor_features=hidden_in.repeat(2, 1),
            features=hidden_pos_neg,
            anchor_targets=target_in.repeat(2),
            targets=torch.cat((pos_y, neg_y)),
        )
        return loss_in + loss_buffer


__all__ = [
    "RegularizationMethod",
    "LearningWithoutForgetting",
    "ACECriterion",
    "AMLCriterion",
]
