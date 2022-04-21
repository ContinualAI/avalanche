from typing import Callable, List, Any

import torch

from avalanche.benchmarks.utils.collate_functions import \
    classification_collate_mbatches_fn, classification_single_values_collate_fn
from avalanche.distributed import CollateDistributedBatch


class DistributedMiniBatchStrategySupport:

    def __init__(self):
        super().__init__()
        self._mbatch = CollateDistributedBatch(
            'mbatch',
            None,
            classification_collate_mbatches_fn,
            classification_single_values_collate_fn
        )

        self._mb_output = CollateDistributedBatch(
            'mb_output',
            None,
            classification_collate_mbatches_fn,
            classification_single_values_collate_fn
        )

    # --- START INPUT MINIBATCH PROPERTY ---
    @property
    def mbatch(self):
        """ Current mini-batch. """
        return self._mbatch.value

    @mbatch.setter
    def mbatch(self, value):
        """ Sets the current mini-batch. """
        self._mbatch.value = value

    @property
    def local_mbatch(self):
        """ The current local mini-batch. """
        return self._mbatch.local_value

    @local_mbatch.setter
    def local_mbatch(self, value):
        """ Sets the current local mini-batch. """
        self._mbatch.local_value = value

    @property
    def distributed_mbatch(self):
        """ The current distributed mini-batch. """
        return self._mbatch.distributed_value

    @distributed_mbatch.setter
    def distributed_mbatch(self, value):
        """ Sets the current distributed mini-batch. """
        self._mbatch.distributed_value = value

    def reset_distributed_mbatch(self):
        """ Resets the distributed value of the mini-batch. """
        self._mbatch.reset_distributed_value()
    # --- END INPUT MINIBATCH PROPERTY ---

    # --- START OUTPUT MINIBATCH PROPERTY ---
    @property
    def mb_output(self):
        """ Model's output computed on the current mini-batch. """
        return self._mb_output.value

    @mb_output.setter
    def mb_output(self, value):
        """ Sets the model's output computed on the current mini-batch. """
        self._mb_output.value = value

    @property
    def local_mb_output(self):
        """ The current local output. """
        return self._mb_output.local_value

    @local_mb_output.setter
    def local_mb_output(self, value):
        """ Sets the current local output. """
        self._mb_output.local_value = value

    @property
    def distributed_mb_output(self):
        """ The current distributed output. """
        return self._mb_output.local_value

    @distributed_mb_output.setter
    def distributed_mb_output(self, value):
        """ Sets the current distributed output. """
        self._mb_output.distributed_value = value

    def reset_distributed_mb_output(self):
        """ Resets the distributed value of the output. """
        self._mb_output.reset_distributed_value()
    # --- END OUTPUT MINIBATCH PROPERTY ---

    # --- START COLLATE FUNCTIONS (INPUT MB) ---
    @property
    def input_batch_collate_fn(self):
        return self._mbatch.tuples_collate_fn

    @input_batch_collate_fn.setter
    def input_batch_collate_fn(self, batch_collate_fn: Callable[[List], Any]):
        self._mbatch.tuples_collate_fn = batch_collate_fn

    @property
    def input_batch_single_values_collate_fn(self):
        return self._mbatch.single_values_collate_fn

    @input_batch_single_values_collate_fn.setter
    def input_batch_single_values_collate_fn(
            self, single_values_collate_fn: Callable[[List], Any]):
        self._mbatch.single_values_collate_fn = single_values_collate_fn

    # --- END COLLATE FUNCTIONS (INPUT MB) ---

    # --- START COLLATE FUNCTIONS (OUTPUT MB) ---
    @property
    def output_batch_collate_fn(self):
        return self._mb_output.tuples_collate_fn

    @output_batch_collate_fn.setter
    def output_batch_collate_fn(self, batch_collate_fn: Callable[[List], Any]):
        self._mb_output.tuples_collate_fn = batch_collate_fn

    @property
    def output_batch_single_values_collate_fn(self):
        return self._mb_output.single_values_collate_fn

    @output_batch_single_values_collate_fn.setter
    def output_batch_single_values_collate_fn(
            self, single_values_collate_fn: Callable[[List], Any]):
        self._mb_output.single_values_collate_fn = single_values_collate_fn
    # --- END COLLATE FUNCTIONS (OUTPUT MB) ---

    # --- START LOCAL CONTEXT MANAGERS ---
    def use_local_input_batch(self, *args, **kwargs):
        return self._mbatch.use_local_value(*args, **kwargs)

    def use_local_output_batch(self, *args, **kwargs):
        return self._mb_output.use_local_value(*args, **kwargs)
    # --- END LOCAL CONTEXT MANAGERS ---


__all__ = [
    'DistributedMiniBatchStrategySupport'
]
