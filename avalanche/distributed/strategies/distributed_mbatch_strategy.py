from typing import Callable, List, Any, Optional, Union

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.collate_functions import \
    Collate, ClassificationCollate
from avalanche.distributed import CollateDistributedBatch
from avalanche.distributed.strategies import DistributedStrategySupport


class DistributedMiniBatchStrategySupport(DistributedStrategySupport):

    def __init__(self):
        super().__init__()

        default_collate_impl = ClassificationCollate()
        self._mbatch = CollateDistributedBatch(
            'mbatch',
            None,
            default_collate_impl.collate_fn,
            default_collate_impl.collate_single_value_fn
        )

        self._mb_output = CollateDistributedBatch(
            'mb_output',
            None,
            default_collate_impl.collate_fn,
            default_collate_impl.collate_single_value_fn
        )

        self._adapted_dataset: Optional[AvalancheDataset] = None
        self._collate_fn: Optional[Union[Collate, Callable]] = None

        self._use_local_contexts.append(self.use_local_input_batch)
        self._use_local_contexts.append(self.use_local_output_batch)

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

    # --- START - GET COLLATE FUNCTIONS FROM DATASET ---
    @property
    def collate_fn(self):
        """
        The collate function used to merge the values obtained from the
        dataset into a minibatch.

        This value is obtained from the adapted dataset directly.
        """
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, new_collate):
        self._collate_fn = new_collate

        if isinstance(new_collate, Collate):
            self.input_batch_collate_fn = new_collate.collate_fn
            self.input_batch_single_values_collate_fn = \
                new_collate.collate_single_value_fn
        else:
            self.input_batch_collate_fn = new_collate
            self.input_batch_single_values_collate_fn = None

    @property
    def adapted_dataset(self):
        return self._adapted_dataset

    @adapted_dataset.setter
    def adapted_dataset(self, dataset: Optional[AvalancheDataset]):
        # Every time a new dataset is set, the related collate
        # function is retrieved and set for sync-ing distributed
        # input/output minibatch fields.
        self._adapted_dataset = dataset
        if self._adapted_dataset is None:
            return

        new_collate = self._adapted_dataset.collate_fn
        if new_collate is None:
            return

        self.collate_fn = new_collate

    # --- END - GET COLLATE FUNCTIONS FROM DATASET ---


__all__ = [
    'DistributedMiniBatchStrategySupport'
]
