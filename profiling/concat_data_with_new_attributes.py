# see https://github.com/ContinualAI/avalanche/issues/1357

from os.path import expanduser
from typing import Optional

import torch
import time

from avalanche.benchmarks import SplitMNIST, SplitCIFAR100
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.training.storage_policy import (
    BalancedExemplarsBuffer,
    ReservoirSamplingBuffer,
)


class ClassBalancedBufferWithLogits(BalancedExemplarsBuffer):
    """
    ClassBalancedBuffer that also stores the logits
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: Optional[int] = None,
    ):
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes: set[int] = set()

    def update(self, dataset, add_attributes=True):
        logits = [torch.randn(1, 10) for _ in range(len(dataset))]
        if add_attributes:
            new_data_with_logits = make_avalanche_dataset(
                dataset,
                data_attributes=[
                    TensorDataAttribute(logits, name="logits", use_in_getitem=True)
                ],
            )
        else:
            new_data_with_logits = dataset
        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(dataset.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            subset = new_data_with_logits.subset(c_idxs)
            cl_datasets[c] = subset
        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                # Here it uses underlying dataset
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(None, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(None, class_to_len[class_id])


if __name__ == "__main__":
    benchmark = SplitMNIST(n_experiences=1)

    storage_policy = ClassBalancedBufferWithLogits(max_size=2000)
    dataset = benchmark.train_stream[0].dataset
    num_updates = 20
    for i in range(num_updates):
        print("Update ", i + 1)
        storage_policy.update(dataset, add_attributes=False)

    start = time.time()
    print("Buffer size: ", len(storage_policy.buffer))
    end = time.time()
    duration = end - start
    print("Buffer access duration: ", duration)
