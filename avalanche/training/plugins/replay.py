from torch.utils.data import random_split, ConcatDataset

from avalanche.training.plugins.strategy_plugin import StrategyPlugin


class ReplayPlugin(StrategyPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implements the "adapt_train_dataset" callback to add them to
    the training set.

    The :mem_size: attribute controls the number of patterns to be stored in
    the external memory. In multitask scenarios, mem_size is the memory size
    for each task. We assume the training set contains at least :mem_size: data
    points.
    """

    def __init__(self, mem_size=200):
        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = {}  # a Dict<task_id, Dataset>
        self.rm_add = None

    def adapt_train_dataset(self, strategy, **kwargs):
        """
        Expands the current training set with datapoint from
        the external memory before training.
        """
        curr_data = strategy.experience.dataset

        # Additional set of the current batch to be concatenated to the ext.
        # memory at the end of the training
        self.rm_add = None

        # how many patterns to save for next iter
        h = min(self.mem_size // (strategy.training_exp_counter + 1),
                len(curr_data))

        # We recover it using the random_split method and getting rid of the
        # second split.
        self.rm_add, _ = random_split(
            curr_data, [h, len(curr_data) - h]
        )

        if strategy.training_exp_counter > 0:
            # We update the train_dataset concatenating the external memory.
            # We assume the user will shuffle the data when creating the data
            # loader.
            for mem_task_id in self.ext_mem.keys():
                mem_data = self.ext_mem[mem_task_id]
                if mem_task_id in strategy.adapted_dataset:
                    cat_data = ConcatDataset([curr_data, mem_data])
                    strategy.adapted_dataset[mem_task_id] = cat_data
                else:
                    strategy.adapted_dataset[mem_task_id] = mem_data

    def after_training_exp(self, strategy, **kwargs):
        """ After training we update the external memory with the patterns of
         the current training batch/task. """
        curr_task_id = strategy.experience.task_label

        # replace patterns in random memory
        ext_mem = self.ext_mem
        if curr_task_id not in ext_mem:
            ext_mem[curr_task_id] = self.rm_add
        else:
            rem_len = len(ext_mem[curr_task_id]) - len(self.rm_add)
            _, saved_part = random_split(ext_mem[curr_task_id],
                                         [len(self.rm_add), rem_len]
                                         )
            ext_mem[curr_task_id] = ConcatDataset([saved_part, self.rm_add])
        self.ext_mem = ext_mem
