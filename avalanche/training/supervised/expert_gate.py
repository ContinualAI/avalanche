from collections import OrderedDict
import torch
from torch.nn import Module, CrossEntropyLoss, MSELoss, Softmax
from torch.optim import Optimizer, SGD

from typing import Optional, List

from avalanche.models.expert_gate import Autoencoder, ExpertModel, ExpertGate

from avalanche.training.supervised import AETraining
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.plugins import (
    SupervisedPlugin, EvaluationPlugin, LwFPlugin)
from avalanche.training.plugins.evaluation import default_evaluator


class ExpertGateStrategy(SupervisedTemplate):
    """Expert Gate strategy.

    To use this strategy you need to instantiate an ExpertGate model.
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
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        # Check that the model has the correct architecture.
        assert isinstance(
            model, ExpertGate), "ExpertGateStrategy requires an ExpertGate model."

        expertgate = _ExpertGatePlugin()

        if plugins is None:
            plugins = [expertgate]
        else:
            plugins += [expertgate]

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

    def __init__(self):
        super().__init__()

    def before_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        super().before_training_exp(strategy, *args, **kwargs)

        task_label = strategy.experience.task_label

        # Build an autoencoder for this experience and store it in a dictionary
        self._add_autoencoder(strategy, task_label, latent_dim=100)

        # Retrieve the autoencoder from the dictionary
        autoencoder = self._retrieve_autoencoder(strategy, task_label)

        # Train the autoencoder on current experience with custom loss
        self._train_autoencoder(strategy, autoencoder)

        # Build new expert with feature extraction from relevant existing expert
        expert, relatedness = self._select_expert(strategy, task_label)

        # Store expert
        self._add_expert(strategy, task_label, expert)

        # Pick training method based on relatedness
        self.alpha = 0.01
        self.temp = 2
        if (relatedness > strategy.model.rel_thresh):
            lwf = LwFPlugin(self.alpha, self.temp)
            if strategy.plugins is None:
                strategy.plugins = [lwf]
            else:
                strategy.plugins.append(lwf)

    def before_eval_iteration(self, strategy: "SupervisedTemplate", *args, **kwargs):
        super().before_eval_iteration(strategy, *args, **kwargs)

        # Build a probability dictionary
        probability_dict = OrderedDict()

        # Softmax layer
        softmax = Softmax(dim=0)

        # Iterate through all autoencoders to get error values
        for autoencoder_id in strategy.model.autoencoder_dict:
            error = self._get_average_reconstruction_error(
                strategy, autoencoder_id)
            x = torch.tensor(-error/self.temp)
            probability_dict[str(autoencoder_id)] = softmax(x)

        # Select an expert for this iteration
        most_relevant_expert_key = max(
                probability_dict, key=probability_dict.get)
        strategy.model.expert = self._retrieve_expert(
                strategy, most_relevant_expert_key)

    # ##############
    # EXPERT METHODS 
    # ##############

    def _add_expert(self, strategy: "SupervisedTemplate", task_label, expert):
        strategy.model.expert_dict[str(task_label)] = expert
        strategy.model.expert = expert

    def _retrieve_expert(self, strategy: "SupervisedTemplate", key):
        return strategy.model.expert_dict[str(key)]

    def _select_expert(self, strategy: "SupervisedTemplate", task_label):

        num_classes = len(strategy.experience.classes_in_this_experience)

        # If the expert dictionary is empty, 
        # build the first expert
        if (len(strategy.model.expert_dict) == 0):
            expert = ExpertModel(num_classes=num_classes, 
                                 arch=strategy.model.arch,
                                 device=strategy.model.device, 
                                 pretrained_flag=strategy.model.pretrained_flag)
            relatedness = 0

        # If experts exist, 
        # select an autoencoder using task relatedness
        else:
            # Build an error dictionary
            error_dict = OrderedDict()

            # Iterate through all autoencoders to get error values
            for autoencoder_id in strategy.model.autoencoder_dict:
                error_dict[str(autoencoder_id)
                           ] = self._get_average_reconstruction_error(strategy, autoencoder_id)

            # Send error dictionary to get most relevant autoencoder
            relatedness_dict = self._task_relatedness(
                strategy, error_dict, task_label)
            # Retrieve best expert
            most_relevant_expert_key = max(
                relatedness_dict, key=relatedness_dict.get)
            most_relevant_expert = self._retrieve_expert(
                strategy, most_relevant_expert_key)

            # Build expert
            expert = ExpertModel(num_classes=num_classes,
                                 arch=strategy.model.arch,
                                 device=strategy.model.device, 
                                 pretrained_flag=strategy.model.pretrained_flag,
                                 feature_template=most_relevant_expert)

            relatedness = relatedness_dict[most_relevant_expert_key]

        return expert, relatedness

    # ########################
    # EXPERT SELECTION METHODS
    # ########################
    def _task_relatedness(self, strategy: "SupervisedTemplate", error_dict, task_label):
        # Build a task relatedness dictionary
        relatedness_dict = OrderedDict()

        error_k = error_dict[str(task_label)]

        # Iterate through all reconstruction errros to obtain task_relatedness
        for task, error_a in error_dict.items():
            if task != str(task_label):
                relatedness_dict[str(task)] = 1 - ((error_a - error_k)/error_k)

        return relatedness_dict

    def _get_average_reconstruction_error(self, strategy: "SupervisedTemplate", task_label):

        autoencoder = self._retrieve_autoencoder(strategy, task_label)

        ae_strategy = AETraining(model=autoencoder, 
                                 optimizer=SGD(
                                     autoencoder.parameters(), lr=0.01),
                                 eval_mb_size=100, 
                                 eval_every=-1)

        # Run evaluation on autoencoder
        ae_strategy.eval(strategy.experience)

        # Build the key for evaluation metrics dictionary
        if (strategy.experience.origin_stream.name == "train"):
            key = 'Loss_Stream/eval_phase/train_stream/Task' + \
                "{:0>3d}".format(strategy.experience.task_label)

        elif (strategy.experience.origin_stream.name == "test"):
            key = 'Loss_Stream/eval_phase/test_stream/Task' + \
                "{:0>3d}".format(strategy.experience.task_label)

        # Query for reconstruction loss
        error = ae_strategy.evaluator.get_last_metrics()[key]

        return error

    # ##################
    # AUTENCODER METHODS 
    # ##################
    def _add_autoencoder(self, strategy: "SupervisedTemplate", task_label, latent_dim=500):

        # Build a new autoencoder
        new_autoencoder = Autoencoder(
            shape=strategy.model.shape, latent_dim=latent_dim)

        # Store autoencoder with task number
        strategy.model.autoencoder_dict[str(task_label)] = new_autoencoder

    def _retrieve_autoencoder(self, strategy: "SupervisedTemplate", task_label):
        return strategy.model.autoencoder_dict[str(task_label)]

    def _train_autoencoder(self, strategy: "SupervisedTemplate", autoencoder):

        # Setup autoencoder strategy
        ae_strategy = AETraining(model=autoencoder, 
                                 optimizer=SGD(
                                     autoencoder.parameters(), lr=0.01),
                                 train_mb_size=100, 
                                 train_epochs=50,
                                 eval_every=-1)

        # Train with autoencoder strategy
        ae_strategy.train(strategy.experience)
