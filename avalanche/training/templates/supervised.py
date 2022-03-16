from typing import Sequence, Optional
from pkg_resources import parse_version

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.models import avalanche_forward
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.base_sgd import BaseSGDTemplate
from avalanche.training.utils import trigger_plugins


class SupervisedTemplate(BaseSGDTemplate):
    """Base class for continual learning strategies.

    BaseTemplate is the super class of all task-based continual learning
    strategies. It implements a basic training loop and callback system
    that allows to execute code at each experience of the training loop.
    Plugins can be used to implement callbacks to augment the training
    loop with additional behavior (e.g. a memory buffer for replay).

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
                    # backward
                    # model update

    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator=default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
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
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )
        self._criterion = criterion

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        self.adapted_dataset = None
        """ Data used to train. It may be modified by plugins. Plugins can 
        append data to it (e.g. for replay). 
         
        .. note::

            This dataset may contain samples from different experiences. If you 
            want the original data for the current experience  
            use :attr:`.BaseTemplate.experience`.
        """

    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        """Loss function."""
        return self._criterion(self.mb_output, self.mb_y)

    def _before_training_exp(self, **kwargs):
        """Setup to train on a single experience."""
        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        super()._before_training_exp(**kwargs)

    def _load_train_state(self, prev_state):
        super()._load_train_state(prev_state)
        self.adapted_dataset = prev_state["adapted_dataset"]
        self.dataloader = prev_state["dataloader"]

    def _save_train_state(self):
        """Save the training state which may be modified by the eval loop.

        This currently includes: experience, adapted_dataset, dataloader,
        is_training, and train/eval modes for each module.

        TODO: we probably need a better way to do this.
        """
        state = super()._save_train_state()
        new_state = {
            "adapted_dataset": self.adapted_dataset,
            "dataloader": self.dataloader,
        }
        return {**state, **new_state}

    def train_dataset_adaptation(self, **kwargs):
        """Initialize `self.adapted_dataset`."""
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.train()

    def _before_eval_exp(self, **kwargs):
        # Data Adaptation
        self._before_eval_dataset_adaptation(**kwargs)
        self.eval_dataset_adaptation(**kwargs)
        self._after_eval_dataset_adaptation(**kwargs)
        super()._before_eval_exp(**kwargs)

    def make_train_dataloader(
        self, num_workers=0, shuffle=True, pin_memory=True,
        persistent_workers=False, **kwargs
    ):
        """Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """

        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version('1.7.0'):
            other_dataloader_args['persistent_workers'] = persistent_workers

        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    def make_eval_dataloader(
            self, num_workers=0, pin_memory=True, persistent_workers=False,
            **kwargs):
        """
        Initializes the eval data loader.
        :param num_workers: How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            (default: 0).
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param kwargs:
        :return:
        """
        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version('1.7.0'):
            other_dataloader_args['persistent_workers'] = persistent_workers

        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model
        avalanche_model_adaptation(model, self.experience.dataset)
        return model.to(self.device)

    def _unpack_minibatch(self):
        """We assume mini-batches have the form <x, y, ..., t>.
        This allows for arbitrary tensors between y and t.
        Keep in mind that in the most general case mb_task_id is a tensor
        which may contain different labels for each sample.
        """
        assert len(self.mbatch) >= 3
        super()._unpack_minibatch()

    def eval_dataset_adaptation(self, **kwargs):
        """Initialize `self.adapted_dataset`."""
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.eval()

    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experiene to configure the optimizer.
        """
        # we reset the optimizer's state after each experience.
        # This allows to add new parameters (new heads) and
        # freezing old units during the model's adaptation phase.
        reset_optimizer(self.optimizer, self.model)

    #########################################################
    # Plugin Triggers                                       #
    #########################################################

    def _before_train_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "before_train_dataset_adaptation", **kwargs)

    def _after_train_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "after_train_dataset_adaptation", **kwargs)

    def _before_eval_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "before_eval_dataset_adaptation", **kwargs)

    def _after_eval_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "after_eval_dataset_adaptation", **kwargs)
