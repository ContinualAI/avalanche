################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-09-2022                                                             #
# Author(s): Nishant Aswani                                                    #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from collections import OrderedDict
import warnings

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD, Adam
from torch.nn.functional import log_softmax

from typing import Optional, List

from avalanche.models.expert_gate import ExpertAutoencoder, ExpertModel, ExpertGate
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.training.supervised import AETraining
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin, LwFPlugin
from avalanche.training.plugins.evaluation import default_evaluator


class ExpertGateStrategy(SupervisedTemplate):
    """Expert Gate strategy. New experts are trained and added
    to the model as tasks are learned sequentially.

    Technique introduced in:
    'Aljundi, Rahaf, Punarjay Chakravarty, and Tinne Tuytelaars.
    "Expert gate: Lifelong learning with a network of experts."
    Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition. 2017.'
    https://arxiv.org/abs/1611.06194

    To use this strategy you need to instantiate an ExpertGate model.
    See the ExpertGate plugin for more details.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        ae_train_mb_size=1,
        ae_train_epochs=2,
        ae_latent_dim=100,
        ae_lr=1e-3,
        temp=2,
        rel_thresh=0.85,
        **base_kwargs
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
        :param ae_train_mb_size: mini-batch size for training of the autoencoder
        :param ae_train_epochs: number of training epochs for the autoencoder
        :param ae_lr: the learning rate for the autoencoder training
        using vanilla SGD
        :param temp: the temperature hyperparameter when selecting the
        expert during the forward method
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        # Check that the model has the correct architecture.
        assert isinstance(
            model, ExpertGate
        ), "ExpertGateStrategy requires an ExpertGate model."

        expertgate = _ExpertGatePlugin()

        if plugins is None:
            plugins = [expertgate]
        else:
            plugins += [expertgate]

        self.ae_train_mb_size = ae_train_mb_size
        self.ae_train_epochs = ae_train_epochs
        self.ae_lr = ae_lr
        self.ae_latent_dim = ae_latent_dim
        self.rel_thresh = rel_thresh
        model.temp = temp

        warnings.warn(
            "This strategy is currently in the alpha stage and we are still "
            "working to reproduce the original paper's results. You can find "
            "the code to reproduce the experiments at "
            "github.com/continualAI/continual-learning-baselines"
        )

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class _ExpertGatePlugin(SupervisedPlugin):
    """The ExpertGate algorithm is a dynamic architecture algorithm.
    For every new task, it trains an autoencoder to reconstruct input data
    and then trains an AlexNet classifier. Prior to AlexNet training,
    the algorithm searches through existing autoencoders, if there are any,
    to find the most related autoencoder and select the expert associated
    to that autoencoder. The new expert is then fine-tuned or trained using
    Learning without Forgetting (LwF) based on the most related previous expert.
    """

    def __init__(self):
        super().__init__()
        # Hyperparameters for LwF plugin
        # Obtained from the ExpertGate paper
        self.alpha = 0.01
        self.temp = 2

        # Initialize instance of the LwF plugin
        self.lwf_plugin = LwFPlugin(self.alpha, self.temp)

    def before_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        super().before_training_exp(strategy, *args, **kwargs)

        # Store task label for easy access
        task_label = strategy.experience.task_label

        # Build an autoencoder for this experience and
        # store it in a dictionary
        autoencoder = self._add_autoencoder(strategy, task_label)

        # Train the autoencoder on current experience
        self._train_autoencoder(strategy, autoencoder)

        # If experts exist, build new expert with feature extraction
        # from the most related existing expert
        new_expert, relatedness = self._select_expert(strategy, task_label)

        # Store the new expert in dictionary
        self._add_expert(strategy, task_label, new_expert)

        # Set the correct expert to be trained
        strategy.model.expert = new_expert

        # Reset the optimizer for the new expert model
        # reset_optimizer(strategy.optimizer, strategy.model.expert)
        # make_optimizer should be called instead of reset_optimizer
        # It puts all parameters from strategy.model in the optimizer
        # To freeze parameters use param.requires_grad = False
        # and param.grad = None
        strategy.make_optimizer(**kwargs)

        # Remove LwF plugin in case it is not needed
        if self.lwf_plugin in strategy.plugins:
            strategy.plugins.remove(self.lwf_plugin)

        print("\nTRAINING EXPERT")
        # If needed, add a new instance of LwF plugin back
        if relatedness > strategy.rel_thresh:
            print("WITH LWF")
            self.lwf_plugin = LwFPlugin(self.alpha, self.temp)
            strategy.plugins.append(self.lwf_plugin)

    # ##############
    # EXPERT METHODS
    # ##############

    def _add_expert(self, strategy: "SupervisedTemplate", task_label, expert):
        """Adds expert to ExpertGate expert dictionary using the task_label
        as a key.
        """
        strategy.model.expert_dict[str(task_label)] = expert

    def _get_expert(self, strategy: "SupervisedTemplate", key):
        """Retrieves expert Alex model from the ExpertGate expert
        dictionary using the task_label as a key.
        """
        return strategy.model.expert_dict[str(key)]

    def _select_expert(self, strategy: "SupervisedTemplate", task_label):
        """Given a task label, calculates the relatedness between
        the autoencoder for this task and all other autoencoders.
        Returns the most related expert and the relatedness value.
        """
        print("\nSELECTING EXPERT")
        # If the expert dictionary is empty,
        # build the first expert

        # Preferably we could use `strategy.experience.benchmark.n_classes`
        # but this attribute is not enforced for all benchmarks
        n_classes = len(strategy.experience.classes_seen_so_far) + len(
            strategy.experience.future_classes
        )

        if len(strategy.model.expert_dict) == 0:
            expert = ExpertModel(
                num_classes=n_classes,
                arch=strategy.model.arch,
                device=strategy.device,
                pretrained_flag=strategy.model.pretrained_flag,
            )
            relatedness = 0

        # If experts exist,
        # select an autoencoder using task relatedness
        else:
            # Build an error dictionary
            error_dict = OrderedDict()

            # Iterate through all autoencoders to get error values
            for autoencoder_id in strategy.model.autoencoder_dict:
                error_dict[
                    str(autoencoder_id)
                ] = self._get_average_reconstruction_error(strategy, autoencoder_id)

            # Send error dictionary to get most relevant autoencoder
            relatedness_dict = self._task_relatedness(strategy, error_dict, task_label)

            # Retrieve best expert
            most_relevant_expert_key = max(relatedness_dict, key=relatedness_dict.get)

            most_relevant_expert = self._get_expert(strategy, most_relevant_expert_key)

            # Build expert with feature template
            expert = ExpertModel(
                num_classes=n_classes,
                arch=strategy.model.arch,
                device=strategy.device,
                pretrained_flag=strategy.model.pretrained_flag,
                provided_template=most_relevant_expert,
            )

            relatedness = relatedness_dict[most_relevant_expert_key]
            print("SELECTED EXPERT FROM TASK ", most_relevant_expert_key)

        print("FINISHED EXPERT SELECTION\n")
        return expert, relatedness

    # ########################
    # EXPERT SELECTION METHODS
    # ########################
    def _task_relatedness(self, strategy: "SupervisedTemplate", error_dict, task_label):
        """Given a task label and error dictionary, returns a dictionary
        of relatedness between the autoencoder of the current task
        and all other tasks.
        """
        # Build a task relatedness dictionary
        relatedness_dict = OrderedDict()

        error_k = error_dict[str(task_label)]

        # Iterate through all reconstruction errros to obtain task_relatedness
        for task, error_a in error_dict.items():
            if task != str(task_label):
                relatedness_dict[str(task)] = 1 - ((error_a - error_k) / error_k)

        return relatedness_dict

    def _get_average_reconstruction_error(
        self, strategy: "SupervisedTemplate", task_label
    ):
        """Given a task label, retrieves an autoencoder and
        evaluates the reconstruction error on the current batch of data.
        """
        autoencoder = self._get_autoencoder(strategy, task_label)

        ae_strategy = AETraining(
            model=autoencoder,
            optimizer=SGD(autoencoder.parameters(), lr=strategy.ae_lr),
            device=strategy.device,
            eval_mb_size=100,
            eval_every=-1,
        )

        # Run evaluation on autoencoder
        ae_strategy.eval(strategy.experience)

        # Build the key for evaluation metrics dictionary
        if strategy.experience.origin_stream.name == "train":
            key = "Loss_Stream/eval_phase/train_stream/Task" + "{:0>3d}".format(
                strategy.experience.task_label
            )

        elif strategy.experience.origin_stream.name == "test":
            key = "Loss_Stream/eval_phase/test_stream/Task" + "{:0>3d}".format(
                strategy.experience.task_label
            )

        # Query for reconstruction loss
        error = ae_strategy.evaluator.get_last_metrics()[key]

        return error

    # ##################
    # AUTENCODER METHODS
    # ##################
    def _add_autoencoder(self, strategy: "SupervisedTemplate", task_label):
        """Builds a new autoencoder and stores it in the ExpertGate
        autoencoder dictionary. Returns the new autoencoder.
        """
        # Build a new autoencoder
        # This shape is equivalent to the output shape of
        # the Alexnet features module
        new_autoencoder = ExpertAutoencoder(
            shape=(256, 6, 6), latent_dim=strategy.ae_latent_dim, device=strategy.device
        )

        # Store autoencoder with task number
        strategy.model.autoencoder_dict[str(task_label)] = new_autoencoder

        return new_autoencoder

    def _get_autoencoder(self, strategy: "SupervisedTemplate", task_label):
        """Retrieves autoencoder from the ExpertGate autoencoder
        dictionary using the task_label as a key.
        """
        return strategy.model.autoencoder_dict[str(task_label)]

    def _train_autoencoder(self, strategy: "SupervisedTemplate", autoencoder):
        """Trains an autoencoder for the ExpertGate plugin."""
        # Setup autoencoder strategy
        ae_strategy = AETraining(
            model=autoencoder,
            optimizer=SGD(
                autoencoder.parameters(),
                lr=strategy.ae_lr,
                momentum=0.9,
                weight_decay=0.0005,
            ),
            device=strategy.device,
            train_mb_size=strategy.ae_train_mb_size,
            train_epochs=strategy.ae_train_epochs,
            eval_every=-1,
        )

        print("\nTRAINING NEW AUTOENCODER")
        # Train with autoencoder strategy
        ae_strategy.train(strategy.experience)
        print("FINISHED TRAINING NEW AUTOENCODER\n")
