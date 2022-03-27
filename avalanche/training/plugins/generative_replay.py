################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""

All plugins related to Generative Replay.

"""

from copy import deepcopy
from avalanche.core import SupervisedPlugin
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates.supervised import SupervisedTemplate
import torch


class GenerativeReplayPlugin(SupervisedPlugin):
    """
    Experience generative replay plugin.

    Updates the Dataloader of a strategy before training an experience
    by sampling a generator model and weaving the replay data into
    the original training data. 

    The examples in the created mini-batch contain one part of the original data
    and one part of generative data for each class 
    that has been encountered so far.

    In this version of the plugin the number of replay samples is 
    increased with each new experience. Another way to implempent 
    the algorithm is by weighting the loss function and give more 
    importance to the replayed data as the number of experiences 
    increases. This will be implemented as an option for the user soon.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param untrained_solver: if True we assume this is the beginning of 
        a continual learning task and add replay data only from the second 
        experience onwards, otherwise we sample and add generative replay data
        before training the first experience. Default to True.
    """

    def __init__(self, generator_strategy: "BaseTemplate" = None, 
                 mem_size: int = 200, 
                 batch_size: int = None,
                 batch_size_mem: int = None,
                 task_balanced_dataloader: bool = False,
                 untrained_solver: bool = True):
        '''
        Init.
        '''
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.generator_strategy = generator_strategy
        if self.generator_strategy:
            self.generator = generator_strategy.model
        else: 
            self.generator = None
        self.untrained_solver = untrained_solver
        self.model_is_generator = False

    def before_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
        """Checks whether we are using a user defined external generator 
        or we use the strategy's model as the generator. 
        If the generator is None after initialization 
        we assume that strategy.model is the generator."""
        if not self.generator_strategy:
            self.generator_strategy = strategy
            self.generator = strategy.model
            self.model_is_generator = True

    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        Make deep copies of generator and solver before training new experience.
        """
        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return
        self.old_generator = deepcopy(self.generator)
        self.old_generator.eval()
        if not self.model_is_generator:
            self.old_model = deepcopy(strategy.model)
            self.old_model.eval()

    def after_training_exp(self, strategy: "SupervisedTemplate",
                           num_workers: int = 0, shuffle: bool = True,
                           **kwargs):
        """
        Set untrained_solver boolean to False after (the first) experience,
        in order to start training with replay data from the second experience.
        """
        self.untrained_solver = False

    def before_training_iteration(self, strategy: "SupervisedTemplate",
                                  **kwargs):
        """
        Generating and appending replay data to current minibatch before 
        each training iteration.
        """
        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return
        # extend X with replay data
        replay = self.old_generator.generate(
            len(strategy.mbatch[0]) * (strategy.experience.current_experience)
            ).to(strategy.device)  
        strategy.mbatch[0] = torch.cat([strategy.mbatch[0], replay], dim=0)
        # extend y with predicted labels (or mock labels if model==generator)
        if not self.model_is_generator:
            with torch.no_grad():
                replay_output = self.old_model(replay).argmax(dim=-1)
        else:
            # Mock labels:
            replay_output = torch.zeros(replay.shape[0])
        strategy.mbatch[1] = torch.cat(
            [strategy.mbatch[1], replay_output.to(strategy.device)], dim=0)
        # extend task id batch (we implicitley assume a task-free case)
        strategy.mbatch[-1] = torch.cat([strategy.mbatch[-1], torch.ones(
            replay.shape[0]).to(strategy.device) * strategy.mbatch[-1][0]],
             dim=0)


class TrainGeneratorAfterExpPlugin(SupervisedPlugin):
    """
    TrainGeneratorAfterExpPlugin makes sure that after each experience of 
    training the solver of a scholar model, we also train the generator on the 
    data of the current experience.
    """

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        The training method expects an Experience object 
        with a 'dataset' parameter.
        """
        strategy.plugins[1].generator_strategy.train(strategy.experience) 
