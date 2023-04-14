import os.path
from copy import copy

import dill
import torch

from avalanche.training.determinism.rng_manager import RNGManager


def maybe_load_checkpoint(strategy, fname, map_location=None):
    """Load the strategy state from a checkpoint file.

    The method returns the strategy with the state deserialized from the file
    and the index of the training experience to resume training.

    If the file does not exists, the method returns the strategy unmodified
    and the index 0. As a result, the method can be safely called even if no
    checkpoint has been previously created (e.g. during the first run).

    Example:

    ```
    strategy = Naive(model, opt, train_mb_size=128)
    strategy, initial_exp = maybe_load_checkpoint(strategy, fname)

    for exp in benchmark.train_stream[initial_exp:]:
        strategy.train(exp)
        save_checkpoint(strat, fname)
    ```

    :param strategy: strategy to load. It must be already initialized.
    :param fname: file name
    :param map_location: sets the location of the tensors after serialization.
        Same as `map_location` of `torch.load`, except that you can also pass
        a device object or a string (a proper
        map will be created accordingly). The recommended way to use this
        parameter is to pass the used reference device.
        In addition, all `torch.device` objects will be un-pickled using
        that map (this is not usually done by `torch.load`,
        but it is needed to properly manage things in Avalanche).
        Defaults to None, which means that no mapping will take place.
    :return: tuple <strategy, exp_counter>
        strategy after deserialization,
        index of the current experience to resume training.
    """
    if not os.path.exists(fname):
        return strategy, 0

    ckp = torch.load(fname, pickle_module=dill,
                     map_location=map_location)

    print(ckp)
    strategy.__dict__.update(ckp['strategy'].__dict__)
    exp_counter = ckp['exp_counter']
    return strategy, exp_counter


def save_checkpoint(strategy, fname, exclude=None):
    """Save the strategy state into a file.

    The file can be loaded using `maybe_load_checkpoint`.

    For efficiency, the user can specify some attributes to `exclude`.
    For example, if the optimizer is static and doesn't change during training,
    it can be safely excluded. These helps to speed up the serialization.

    WARNING: the method cannot be used inside the training and evaluation loops of the strategy.

    :param strategy: strategy to serialize.
    :param fname: name of the file.
    :param exclude: List[string] list of attributes to remove before the serialization.
    :return:
    """
    if exclude is None:
        exclude = []
    ended_experience_counter = strategy.clock.train_exp_counter

    strategy = copy(strategy)
    for attr in exclude:
        delattr(strategy, attr)

    checkpoint_data = {
        'strategy': strategy,
        'rng_manager': RNGManager,
        'exp_counter': ended_experience_counter
    }
    torch.save(checkpoint_data, fname, pickle_module=dill)
