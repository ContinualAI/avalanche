from typing import Sequence, Optional

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator

from .observation_type import *
from .problem_type import *
from .update_type import *
from .base_sgd import BaseSGDTemplate


class SupervisedTemplate(BatchObservation, SupervisedProblem, SGDUpdate,
                         BaseSGDTemplate):
    """Base class for continual learning strategies.

    SupervisedTemplate is the super class of all supervised task-based
    continual learning strategies. It implements a basic training loop and
    callback system that allows to execute code at each experience of the
    training loop. Plugins can be used to implement callbacks to augment the
    training loop with additional behavior (e.g. a memory buffer for replay).

    **Scenarios**
    This strategy supports several continual learning scenarios:

    * class-incremental scenarios (no task labels)
    * multi-task scenarios, where task labels are provided)
    * multi-incremental scenarios, where the same task may be revisited

    The exact scenario depends on the data stream and whether it provides
    the task labels.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience
                adapt_train_dataset
                train_dataset_adaptation
                make_train_dataloader
                train_epoch  # for each epoch
                    # forward
                    # backward
                    # model update

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience
                adapt_eval_dataset
                eval_dataset_adaptation
                make_eval_dataloader
                eval_epoch  # for each epoch
                    # forward

    """

    PLUGIN_CLASS = SupervisedPlugin

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device="cpu",
            plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
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
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
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
            peval_mode=peval_mode,
        )
        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        # self.adapted_dataset = None
        # """ Data used to train. It may be modified by plugins. Plugins can
        # append data to it (e.g. for replay).
        #
        # .. note::
        #
        #    This dataset may contain samples from different experiences. If you
        #    want the original data for the current experience
        #    use :attr:`.BaseTemplate.experience`.


class SupervisedMetaLearningTemplate(BatchObservation, SupervisedProblem,
                                     MetaUpdate, BaseSGDTemplate):
    """Base class for continual learning strategies.

    SupervisedMetaLearningTemplate is the super class of all supervised
    meta-learning task-based continual learning strategies. It implements a
    basic training loop and callback system that allows to execute code at
    each experience of the training loop. Plugins can be used to implement
    callbacks to augment the training loop with additional behavior
    (e.g. a memory buffer for replay).

    **Scenarios**
    This strategy supports several continual learning scenarios:

    * class-incremental scenarios (no task labels)
    * multi-task scenarios, where task labels are provided)
    * multi-incremental scenarios, where the same task may be revisited

    The exact scenario depends on the data stream and whether it provides
    the task labels.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience
                adapt_train_dataset
                train_dataset_adaptation
                make_train_dataloader
                train_epoch  # for each epoch
                    # inner_updates
                    # outer_update

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience
                adapt_eval_dataset
                eval_dataset_adaptation
                make_eval_dataloader
                eval_epoch  # for each epoch
                    # forward

    """

    PLUGIN_CLASS = SupervisedPlugin

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device="cpu",
            plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
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
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
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
            peval_mode=peval_mode,
        )
        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        # self.adapted_dataset = None
        # """ Data used to train. It may be modified by plugins. Plugins can
        # append data to it (e.g. for replay).
        #
        # .. note::
        #
        #    This dataset may contain samples from different experiences. If you
        #    want the original data for the current experience
        #    use :attr:`.BaseTemplate.experience`.


class OnlineSupervisedTemplate(OnlineObservation, SupervisedProblem, SGDUpdate,
                               BaseSGDTemplate):
    """Base class for continual learning strategies.

    OnlineSupervisedTemplate is the super class of all online supervised
    task-based continual learning strategies. It implements a basic training
    loop and callback system that allows to execute code at each experience of
    the training loop. Plugins can be used to implement callbacks to augment the
    training loop with additional behavior (e.g. a memory buffer for replay).

    **Scenarios**
    This strategy supports several continual learning scenarios:

    * class-incremental scenarios (no task labels)
    * multi-task scenarios, where task labels are provided)
    * multi-incremental scenarios, where the same task may be revisited

    The exact scenario depends on the data stream and whether it provides
    the task labels.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience
                adapt_train_dataset
                train_dataset_adaptation
                make_train_dataloader
                train_pass  # for each pass
                    # forward
                    # backward
                    # model update

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience
                adapt_eval_dataset
                eval_dataset_adaptation
                make_eval_dataloader
                eval_epoch  # for each epoch
                    # forward

    """

    PLUGIN_CLASS = SupervisedPlugin

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_passes: int = 1,
            eval_mb_size: Optional[int] = 1,
            device="cpu",
            plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            peval_mode="experience",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_passes,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )

        self.train_passes = train_passes


class OnlineSupervisedMetaLearningTemplate(OnlineObservation, SupervisedProblem,
                                           MetaUpdate, BaseSGDTemplate):
    """Base class for continual learning strategies.

    OnlineSupervisedMetaLearningTemplate is the super class of all online
    supervised meta-learning task-based continual learning strategies.
    It implements a basic training loop and callback system that allows
    to execute code at each experience of the training loop. Plugins can be
    used to implement callbacks to augment the training loop with additional
    behavior (e.g. a memory buffer for replay).

    **Scenarios**
    This strategy supports several continual learning scenarios:

    * class-incremental scenarios (no task labels)
    * multi-task scenarios, where task labels are provided)
    * multi-incremental scenarios, where the same task may be revisited

    The exact scenario depends on the data stream and whether it provides
    the task labels.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience
                adapt_train_dataset
                train_dataset_adaptation
                make_train_dataloader
                train_epoch  # for each epoch
                    # inner_updates
                    # outer_update

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience
                adapt_eval_dataset
                eval_dataset_adaptation
                make_eval_dataloader
                eval_epoch  # for each epoch
                    # forward

    """

    PLUGIN_CLASS = SupervisedPlugin

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_passes: int = 1,
            eval_mb_size: Optional[int] = 1,
            device="cpu",
            plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
        :param train_passes: number of training passes.
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
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_passes,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )
        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        # self.adapted_dataset = None
        # """ Data used to train. It may be modified by plugins. Plugins can
        # append data to it (e.g. for replay).
        #
        # .. note::
        #
        #    This dataset may contain samples from different experiences. If you
        #    want the original data for the current experience
        #    use :attr:`.BaseTemplate.experience`.
