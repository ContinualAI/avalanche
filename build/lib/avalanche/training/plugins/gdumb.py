from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

import torch
import tqdm
from torch import Tensor
from torch.utils.data import TensorDataset

from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class GDumbPlugin(StrategyPlugin):
    """
    A GDumb plugin. At each experience the model
    is trained with all and only the data of the external memory.
    The memory is updated at the end of each experience to add new classes or
    new examples of already encountered classes.
    In multitask scenarios, mem_size is the memory size for each task.
    This plugin can be combined with a Naive strategy to obtain the
    standard GDumb strategy.
    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """

    def __init__(self, mem_size: int = 200):
        super().__init__()
        self.mem_size = mem_size
        self.ext_mem: Dict[Any, Tuple[List[Tensor], List[Tensor]]] = {}
        # count occurrences for each class
        self.counter: Dict[Any, Dict[Any, int]] = {}

    def after_train_dataset_adaptation(
        self, strategy: "BaseStrategy", **kwargs
    ):
        """Before training we make sure to organize the memory following
        GDumb approach and updating the dataset accordingly.
        """

        # for each pattern, add it to the memory or not
        assert strategy.experience is not None
        dataset = strategy.experience.dataset
        pbar = tqdm.tqdm(
            dataset, desc="Exhausting dataset to create GDumb buffer"
        )
        for pattern, target, task_id in pbar:
            target = torch.as_tensor(target)
            target_value = target.item()

            if len(pattern.size()) == 1:
                pattern = pattern.unsqueeze(0)

            current_counter = self.counter.setdefault(task_id, defaultdict(int))
            current_mem = self.ext_mem.setdefault(task_id, ([], []))

            if current_counter == {}:
                # any positive (>0) number is ok
                patterns_per_class = 1
            else:
                patterns_per_class = int(
                    self.mem_size / len(current_counter.keys())
                )

            if (
                target_value not in current_counter
                or current_counter[target_value] < patterns_per_class
            ):
                # add new pattern into memory
                if sum(current_counter.values()) >= self.mem_size:
                    # full memory: replace item from most represented class
                    # with current pattern
                    to_remove = max(current_counter, key=current_counter.get)

                    dataset_size = len(current_mem[0])
                    for j in range(dataset_size):
                        if current_mem[1][j].item() == to_remove:
                            current_mem[0][j] = pattern
                            current_mem[1][j] = target
                            break
                    current_counter[to_remove] -= 1
                else:
                    # memory not full: add new pattern
                    current_mem[0].append(pattern)
                    current_mem[1].append(target)

                # Indicate that we've changed the number of stored instances of
                # this class.
                current_counter[target_value] += 1

        task_datasets: Dict[Any, TensorDataset] = {}
        for task_id, task_mem_tuple in self.ext_mem.items():
            patterns, targets = task_mem_tuple
            task_dataset = TensorDataset(
                torch.stack(patterns, dim=0), torch.stack(targets, dim=0)
            )
            task_datasets[task_id] = task_dataset

        adapted_dataset = AvalancheConcatDataset(task_datasets.values())
        strategy.adapted_dataset = adapted_dataset
