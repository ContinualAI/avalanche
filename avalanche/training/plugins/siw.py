from torch.utils.data import random_split, ConcatDataset
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskJoinedBatchDataLoader
import torch
import torch.cuda as tc
from torch.autograd import Variable
import torch.nn as nn


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

    The :batch_size: and :num_workers: parameters concern the new class
    scores extraction.
    """

    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.confidences = []
        self.classes_per_experience = []
        self.num_workers = num_workers
        self.batch_size = batch_size

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
        strategy.model.fc.weight.grad[:previous_classes, :] = 0
        strategy.model.fc.bias.grad[:previous_classes] = 0

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        """
        Extract new class images' scores and compute the model
        confidence at each incremental state
        """
        strategy.model.eval()

        dataset = strategy.experience.dataset
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size,
            num_workers=self.num_workers)

        max_top1_scores = []
        for i, data in enumerate(loader):
            inputs, targets, task_labels = data
            if tc.is_available():
                inputs = inputs.to(strategy.device)
            inputs = Variable(inputs)
            logits = strategy.model(inputs)
            max_score = torch.max(logits, dim=1)[0].tolist()
            max_top1_scores.extend(max_score)
        self.confidences.append(sum(max_top1_scores) /
                                len(max_top1_scores))

    def before_eval_exp(self, strategy, **kwargs):
        """
        Before evaluating the performance of our model, we standardize
        all class weights (by subtracting their mean and dividing by
        their standard deviation)
        """
        previous_classes = len(strategy.experience.previous_classes)
        classes_seen_so_far = len(strategy.experience.classes_seen_so_far)

        for i in range(previous_classes, classes_seen_so_far):
            mu = torch.mean(strategy.model.fc.weight[i])
            std = torch.std(strategy.model.fc.weight[i])

            strategy.model.fc.weight.data[i] -= mu
            strategy.model.fc.weight.data[i] /= std

    def after_eval_forward(self, strategy, **kwargs):
        """
        Rectify past class scores by multiplying them by the model's
        confidence in the current state and dividing them by the
        model's confidence in the initial state in which a past
        class was encountered for the first time
        """
        for exp in range(len(self.confidences)):
            strategy.logits[:, self.classes_per_experience[exp]] *=\
                self.confidences[strategy.experience.current_experience] \
                / self.confidences[exp]
