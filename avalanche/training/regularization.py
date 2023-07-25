"""Regularization methods."""
import copy
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F

from avalanche.models import MultiTaskModule, avalanche_forward


def cross_entropy_with_oh_targets(outputs, targets, eps=1e-5):
    """Calculates cross-entropy with temperature scaling,
    targets can also be soft targets but they must sum to 1"""
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    ce = -(targets * outputs.log()).sum(1)
    ce = ce.mean()
    return ce


class RegularizationMethod:
    """RegularizationMethod implement regularization strategies.
    RegularizationMethod is a callable.
    The method `update` is called to update the loss, typically at the end
    of an experience.
    """

    def update(self, *args, **kwargs):
        raise NotImplementedError()

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
        device: str = "cpu",
    ):
        """
        ER_AML criterion constructor.
        @param feature_extractor: Model able to map an input in a latent space.
        @param temp: Supervised contrastive temperature.
        @param base_temp: Supervised contrastive base temperature.
        @param device: Accelerator used to speed up the computation.
        """
        self.device = device
        self.feature_extractor = feature_extractor
        self.temp = temp
        self.base_temp = base_temp

    def __sample_pos_neg(
        self,
        x_in: torch.Tensor,
        y_in: torch.Tensor,
        x_buffer: torch.Tensor,
        y_buffer: torch.Tensor,
    ) -> tuple:
        """
        Method able to sample positive and negative examples with respect the input minibatch from input and buffer minibatches.
        @param x_in: Input of new minibatch.
        @param y_in: Output of new minibatch.
        @param x_buffer: Input of buffer minibatch.
        @param y_buffer: Output of buffer minibatch.
        @return: Tuple of positive and negative input and output examples and a mask for identify invalid values.
        """
        x_all = torch.cat((x_buffer, x_in))
        y_all = torch.cat((y_buffer, y_in))
        indexes = torch.arange(y_all.shape[0]).to(self.device)

        same_x = indexes[-x_in.shape[0] :].reshape(1, -1) == indexes.reshape(-1, 1)
        same_y = y_in.reshape(1, -1) == y_all.reshape(-1, 1)

        valid_pos = same_y & ~same_x
        valid_neg = ~same_y

        has_valid_pos = valid_pos.sum(0) > 0
        has_valid_neg = valid_neg.sum(0) > 0
        invalid_idx = ~has_valid_pos | ~has_valid_neg
        is_invalid = torch.zeros_like(y_in).bool()
        is_invalid[invalid_idx] = 1
        if invalid_idx.sum() > 0:
            # avoid operand fail
            valid_pos[:, invalid_idx] = 1
            valid_neg[:, invalid_idx] = 1

        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)

        pos_x = x_all[pos_idx]
        pos_y = y_all[pos_idx]
        neg_x = x_all[neg_idx]
        neg_y = y_all[neg_idx]

        return pos_x, pos_y, neg_x, neg_y, is_invalid

    def __sup_con_loss(
        self,
        anchor_features: torch.Tensor,
        features: torch.Tensor,
        anchor_targets: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Method able to compute the supervised contrastive loss of new minibatch.
        @param anchor_features: Anchor features related to new minibatch duplicated mapped in latent space.
        @param features: Features related to half positive and half negative examples mapped in latent space.
        @param anchor_targets: Labels related to anchor features.
        @param targets: Labels related to features.
        @return: Supervised contrastive loss.
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
        @param x: Tensor to normalize.
        @return: Normalized tensor.
        """
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        return x / (x_norm + 1e-05)

    def __call__(
        self,
        input_in: torch.Tensor,
        target_in: torch.Tensor,
        output_buffer: torch.Tensor,
        target_buffer: torch.Tensor,
        buffer_replay_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Method able to compute the ER_AML loss.
        @param input_in: New inputs examples.
        @param target_in: Labels of new examples.
        @param output_buffer: Predictions of samples from buffer.
        @param target_buffer: Labels of samples from buffer.
        @param buffer_replay_data: Buffer replay data to compute positive and negative samples.
        @return: ER_AML computed loss.
        """
        x_buffer, y_buffer, _ = buffer_replay_data
        pos_x, pos_y, neg_x, neg_y, is_invalid = self.__sample_pos_neg(
            input_in, target_in, x_buffer, y_buffer
        )
        loss_buffer = F.cross_entropy(output_buffer, target_buffer)

        hidden_in = self.__scale_by_norm(self.feature_extractor(input_in)[~is_invalid])

        hidden_pos_neg = self.__scale_by_norm(
            self.feature_extractor(torch.cat((pos_x, neg_x)))
        )
        pos_h, neg_h = hidden_pos_neg.reshape(2, pos_x.shape[0], -1)[:, ~is_invalid]

        loss_in = self.__sup_con_loss(
            anchor_features=hidden_in.repeat(2, 1),
            features=torch.cat((pos_h, neg_h)),
            anchor_targets=target_in[~is_invalid].repeat(2),
            targets=torch.cat((pos_y[~is_invalid], neg_y[~is_invalid])),
        )
        return loss_in + loss_buffer


__all__ = [
    "RegularizationMethod",
    "LearningWithoutForgetting",
    "ACECriterion",
    "AMLCriterion",
]
