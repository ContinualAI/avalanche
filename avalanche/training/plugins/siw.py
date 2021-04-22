from torch.utils.data import random_split, ConcatDataset
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskJoinedBatchDataLoader
import torch
import torch.cuda as tc
from torch.autograd import Variable
import torch.nn as nn
from avalanche.training.utils import get_last_fc_layer, get_layer_by_name
from typing import Optional
from torch.nn import Linear


class SIWPlugin(StrategyPlugin):
    """
    Standardization of Initial Weights (SIW) plugin.
    From https://arxiv.org/pdf/2008.13710.pdf

    Performs past class initial weights replay and state-level score
    calibration. The callbacks `before_training_exp`, `after_backward`,
    `after_training_exp`,`before_eval_exp`, and `after_eval_forward`
    are implemented.

    The `before_training_exp` callback is implemented in order to keep
    track of the classes in each experience

    The `after_backward` callback is implemented in order to freeze past
    class weights in the last fully connected layer

    The `after_training_exp` callback is implemented in order to extract
    new class images' scores and compute the model confidence at
    each incremental state.

    The `before_eval_exp` callback is implemented in order to standardize
    initial weights before inference

    The`after_eval_forward` is implemented in order to apply state-level
    calibration at the inference time

    The :siw_layer_name: parameter concerns the name of the last fully
    connected layer of the network

    The :batch_size: and :num_workers: parameters concern the new class
    scores extraction.
    """

    def __init__(self, model, siw_layer_name='fc', batch_size=32,
                 num_workers=0):
        super().__init__()
        self.confidences = []
        self.classes_per_experience = []
        self.model = model
        self.siw_layer_name = siw_layer_name
        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_siw_layer(self) -> Optional[Linear]:
        result = None
        if self.siw_layer_name is None:
            last_fc = get_last_fc_layer(self.model)
            if last_fc is not None:
                result = last_fc[1]
        else:
            result = get_layer_by_name(self.model, self.siw_layer_name)
        return result

    def before_training_exp(self, strategy, **kwargs):
        """
        Keep track of the classes encountered in each experience
        """
        self.classes_per_experience.append(
            strategy.experience.classes_in_this_experience)

    def after_backward(self, strategy, **kwargs):
        """
        Before executing the optimization step to perform
        back-propagation, we zero the gradients of past class
        weights and bias. This is equivalent to freeze past
        class weights and bias, to let only the feature extractor
        and the new class weights and bias evolve
        """
        previous_classes = len(strategy.experience.previous_classes)
        last_layer = self.get_siw_layer()
        if last_layer is None:
            raise RuntimeError('Can\'t find this Linear layer')

        last_layer.weight.grad[:previous_classes, :] = 0
        last_layer.bias.grad[:previous_classes] = 0

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        """
        Before evaluating the performance of our model,
        we extract new class images' scores and compute the
        model's confidence at each incremental state
        """
        # extract training scores
        strategy.model.eval()

        dataset = strategy.experience.dataset
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size,
            num_workers=self.num_workers)

        # compute model's confidence
        max_top1_scores = []
        for i, data in enumerate(loader):
            inputs, targets, task_labels = data
            if tc.is_available():
                inputs = inputs.to(strategy.device)
            logits = strategy.model(inputs)
            max_score = torch.max(logits, dim=1)[0].tolist()
            max_top1_scores.extend(max_score)
        self.confidences.append(sum(max_top1_scores) /
                                len(max_top1_scores))

    @torch.no_grad()
    def before_eval_exp(self, strategy, **kwargs):
        """
        Standardize all class weights (by subtracting their mean
        and dividing by their standard deviation)
        """

        # standardize last layer weights
        last_layer = self.get_siw_layer()
        if last_layer is None:
            raise RuntimeError('Can\'t find this Linear layer')

        classes_seen_so_far = len(strategy.experience.classes_seen_so_far)

        for i in range(classes_seen_so_far):
            mu = torch.mean(last_layer.weight[i])
            std = torch.std(last_layer.weight[i])

            last_layer.weight.data[i] -= mu
            last_layer.weight.data[i] /= std

    def after_eval_forward(self, strategy, **kwargs):
        """
        Rectify past class scores by multiplying them by the model's
        confidence in the current state and dividing them by the
        model's confidence in the initial state in which a past
        class was encountered for the first time
        """
        for exp in range(len(self.confidences)):
            strategy.logits[:, self.classes_per_experience[exp]] *= \
                self.confidences[strategy.experience.current_experience] \
                / self.confidences[exp]
