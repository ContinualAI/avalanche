from collections import defaultdict

import torch
from torch.utils.data import TensorDataset

from avalanche.training.plugins.strategy_plugin import StrategyPlugin


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

    def __init__(self, mem_size=200):

        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = defaultdict(lambda: None)
        # count occurrences for each class
        self.counter = defaultdict(lambda: defaultdict(int))

    def after_train_dataset_adaptation(self, strategy, **kwargs):
        """ Before training we make sure to organize the memory following
            GDumb approach and updating the dataset accordingly.
        """

        # for each pattern, add it to the memory or not
        dataset = strategy.experience.dataset
        current_counter = self.counter[strategy.experience.task_label]
        current_mem = self.ext_mem[strategy.experience.task_label]
        for i, (pattern, target_value, _) in enumerate(dataset):
            target = torch.tensor(target_value)
            if len(pattern.size()) == 1:
                pattern = pattern.unsqueeze(0)

            if current_counter == {}:
                # any positive (>0) number is ok
                patterns_per_class = 1
            else:
                patterns_per_class = int(
                    self.mem_size / len(current_counter.keys())
                )

            if target_value not in current_counter or \
                    current_counter[target_value] < patterns_per_class:
                # full memory: replace item from most represented class
                # with current pattern
                if sum(current_counter.values()) >= self.mem_size:
                    to_remove = max(current_counter, key=current_counter.get)
                    for j in range(len(current_mem.tensors[1])):
                        if current_mem.tensors[1][j].item() == to_remove:
                            current_mem.tensors[0][j] = pattern
                            current_mem.tensors[1][j] = target
                            break
                    current_counter[to_remove] -= 1
                else:
                    # memory not full: add new pattern
                    if current_mem is None:
                        current_mem = TensorDataset(
                            pattern, target.unsqueeze(0))
                    else:
                        current_mem = TensorDataset(
                            torch.cat([
                                pattern,
                                current_mem.tensors[0]], dim=0),

                            torch.cat([
                                target.unsqueeze(0),
                                current_mem.tensors[1]], dim=0)
                        )

                current_counter[target_value] += 1

        self.ext_mem[strategy.experience.task_label] = current_mem
        strategy.adapted_dataset = self.ext_mem
