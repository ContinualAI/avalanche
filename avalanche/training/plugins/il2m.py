from typing import Optional

from packaging.version import parse
import torch
import numpy as np

from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer, ExperienceBalancedBuffer
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader


class IL2MPlugin(SupervisedPlugin):
    """
    Class Incremental Learning With Dual Memory (IL2M) plugin.

    Technique introduced in:
    Belouadah, E. and Popescu, A. "IL2M: Class Incremental Learning With Dual
    Memory." Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition. 2019.

    Implementation based on FACIL, as in:
    https://github.com/mmasana/FACIL/blob/master/src/approach/il2m.py
    """

    def __init__(
        self,
        mem_size: int = 2000,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        storage_policy: Optional[ExemplarsBuffer] = None,
    ):
        """
        :param mem_size: replay buffer size.
        :param batch_size: the size of the data batch. If set to `None`, it
            will be set equal to the strategy's batch size.
        :param batch_size_mem: the size of the memory batch. If its value is set
            to `None` (the default value), it will be automatically set equal to
            the data batch size.
        :param storage_policy: The policy that controls how to add new exemplars
            in memory.
        """

        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

        # to store statistics for the classes as learned in the current incremental state
        self.current_classes_means = []
        # to store statistics for past classes as learned in the incremental state in which they were first seen
        self.init_classes_means = []
        # to store statistics for model confidence in different states (i.e. avg top-1 pred scores)
        self.models_confidence = []
        # to store the mapping between classes and the incremental state in which they were first seen
        self.classes2exp = []
        # total number of classes that will be seen
        self.n_classes = 0

    def before_training_exp(
        self,
        strategy: SupervisedTemplate,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):

        if len(self.init_classes_means) == 0:
            self.n_classes = len(strategy.experience.classes_seen_so_far) + len(
                strategy.experience.future_classes
            )
            self.init_classes_means = [0 for _ in range(self.n_classes)]
            self.classes2exp = [-1 for _ in range(self.n_classes)]

        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )

    def after_training_exp(self, strategy: SupervisedTemplate, **kwargs):
        experience = strategy.experience
        self.current_classes_means = [0 for _ in range(self.n_classes)]
        classes_counts = [0 for _ in range(self.n_classes)]
        self.models_confidence.append(0)
        models_counts = 0

        # compute the mean prediction scores that will be used to rectify scores in subsequent incremental states
        with torch.no_grad():
            strategy.model.eval()
            for inputs, targets, _ in strategy.dataloader:
                inputs, targets = inputs.to(strategy.device), targets.to(
                    strategy.device
                )
                outputs = strategy.model(inputs.to(strategy.device))
                scores = outputs.data.cpu().numpy()
                for i in range(len(targets)):
                    target = targets[i].item()
                    classes_counts[target] += 1
                    if target in experience.previous_classes:
                        # compute the mean prediction scores for past classes of the current state
                        self.current_classes_means[target] += scores[i, target]
                    else:
                        # compute the mean prediction scores for the new classes of the current state
                        self.init_classes_means[target] += scores[i, target]
                        # compute the mean top scores for the new classes of the current state
                        self.models_confidence[-1] += np.max(scores[i,])
                        models_counts += 1

        # normalize by corresponding number of samples
        for cls in experience.previous_classes:
            self.current_classes_means[cls] /= classes_counts[cls]
        for cls in experience.classes_in_this_experience:
            self.init_classes_means[cls] /= classes_counts[cls]
        self.models_confidence[-1] /= models_counts
        # store the mapping between classes and the incremental state in which they are first seen
        for cls in experience.classes_in_this_experience:
            self.classes2exp[cls] = experience.current_experience

        # update the buffer of exemplars
        self.storage_policy.post_adapt(strategy, strategy.experience)

    def after_eval_forward(self, strategy: SupervisedTemplate, **kwargs):
        old_classes = strategy.experience.previous_classes
        new_classes = strategy.experience.classes_in_this_experience
        if not old_classes:
            return

        outputs = strategy.mb_output
        targets = strategy.mbatch[1]

        # rectify predicted scores (Eq. 1 in the paper)
        for i in range(len(targets)):
            # if the top-1 class predicted by the network is a new one, rectify the score
            if outputs[i].argmax().item() in new_classes:
                for cls in old_classes:
                    o_exp = self.classes2exp[cls]
                    if (
                        self.current_classes_means[cls] == 0
                    ):  # when evaluation is done before training
                        continue
                    outputs[i, cls] *= (
                        self.init_classes_means[cls] / self.current_classes_means[cls]
                    ) * (self.models_confidence[-1] / self.models_confidence[o_exp])
            # otherwise, rectification is not done because an old class is directly predicted


__all__ = [
    "IL2MPlugin",
]
