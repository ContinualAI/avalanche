import os.path
import sys
import time
from copy import copy

import dill
import torch

from avalanche.training.determinism.rng_manager import RNGManager


def maybe_load_checkpoint(strategy, fname, map_location=None):
    if not os.path.exists(fname):
        return None, 0

    ckp = torch.load(fname, pickle_module=dill,
                     map_location=map_location)

    print(ckp)
    strategy.__dict__.update(ckp['strategy'].__dict__)
    exp_counter = ckp['exp_counter']
    return strategy, exp_counter


def save_checkpoint(strategy, fname, exclude=None):
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

    print('Saving checkpoint...')
    start_time = time.time()
    # torch.save(checkpoint_data, fname, pickle_module=dill)
    try:
        torch.save(checkpoint_data, fname)
    except Exception as e:
        print('Error saving checkpoint')
        print(e)
        the_type, the_value, the_traceback = sys.exc_info()
        print(the_type)
        print(the_value)
        print(the_traceback)
    else:
        end_time = time.time()
        print('Checkpoint duration:', end_time - start_time)
        print('Checkpoint', ended_experience_counter, 'saved!')