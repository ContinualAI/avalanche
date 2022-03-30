################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-02-2022                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from typing import Optional, Sequence

import torch
from pkg_resources import parse_version
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils.data_loader import detection_collate_fn, \
    TaskBalancedDataLoader, detection_collate_mbatches_fn
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate


class ObjectDetectionTemplate(SupervisedTemplate):
    """
    The object detection naive strategy.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine-tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.

    This strategy can be used as a template for any object detection strategy.
    This template assumes that the provided model follows the same interface
    of torchvision detection models.

    For more info, refer to "TorchVision Object Detection Finetuning Tutorial":
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
            device="cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
            scaler=None):
        """
        Creates a naive detection strategy instance.

        :param model: The PyTorch detection model. This strategy accepts model
            from the torchvision library (as well as all model sharing the same
            interface/behavior)
        :param optimizer: PyTorch optimizer.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
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
        :param scaler: The scaler from PyTorch Automatic Mixed Precision
            package. More info here: https://pytorch.org/docs/stable/amp.html.
            Defaults to None.
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
            peval_mode=peval_mode
        )
        self.scaler = scaler  # torch.cuda.amp.autocast scaler
        """
        The scaler from PyTorch Automatic Mixed Precision package.
        More info here: https://pytorch.org/docs/stable/amp.html
        """

        # Object Detection attributes
        self.detection_loss_dict = None
        """
        A dictionary of detection losses.

        Only valid during the training phase.
        """

        self.detection_predictions = None
        """
        A list of detection predictions.

        This is different from mb_output: mb_output is a list of dictionaries 
        (one dictionary for each image in the input minibatch), 
        while this field, which is populated after calling `criterion()`,
        will be a dictionary {image_id: list_of_predictions}.

        Only valid during the evaluation phase. 
        """

    def make_train_dataloader(
            self, num_workers=0, shuffle=True, pin_memory=True,
            persistent_workers=False, **kwargs):
        """Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param persistent_workers: If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            Used only if `PyTorch >= 1.7.0`.
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
            collate_mbatches=detection_collate_mbatches_fn,
            collate_fn=detection_collate_fn,
            **other_dataloader_args
        )

    def make_eval_dataloader(self, num_workers=0, pin_memory=True, **kwargs):
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
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory,
            collate_fn=detection_collate_fn
        )

    def criterion(self):
        """
        Compute the loss function.

        The initial loss dictionary must be obtained by first running the
        forward pass (the model will return the detection_loss_dict).
        This function will only obtain a single value.

        Beware that the loss can only be obtained for the training phase as no
        loss dictionary is returned when evaluating.
        """
        if self.is_training:
            return sum(
                loss for loss in self.detection_loss_dict.values())
        else:
            # eval does not compute the loss directly.
            # Metrics will use self.mb_output and self.detection_predictions
            # to compute AP, AR, ...
            self.detection_predictions = \
                {target["image_id"].item(): output
                 for target, output in zip(self.mb_y, self.mb_output)}
            return torch.zeros((1,))

    def forward(self):
        """
        Compute the model's output given the current mini-batch.

        For the training phase, a loss dictionary will be returned.
        For the evaluation phase, this will return the model predictions.
        """
        if self.is_training:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                self.detection_loss_dict = self.model(self.mb_x, self.mb_y)
            return self.detection_loss_dict
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outs = self.model(self.mb_x)
            return [{k: v.to('cpu') for k, v in t.items()} for t in outs]

    def _unpack_minibatch(self):
        # Unpack minibatch mainly takes care of moving tensors to devices.
        # In addition, it will prepare the targets in the proper dict format.
        images = list(image.to(self.device) for image in self.mbatch[0])
        targets = [{k: v.to(self.device) for k, v in t.items()}
                   for t in self.mbatch[1]]
        self.mbatch[0] = images
        self.mbatch[1] = targets

    def backward(self):
        if self.scaler is not None:
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()

    def optimizer_step(self, **kwargs):
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()


__all__ = [
    'detection_collate_fn',
    'ObjectDetectionTemplate'
]
