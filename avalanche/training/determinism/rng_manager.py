import random
from collections import OrderedDict
from typing import Any, Dict, Type

import numpy as np
import torch

from avalanche.training.determinism.cuda_rng import (
    cuda_rng_seed,
    cuda_rng_save_state,
    cuda_rng_load_state,
    cuda_rng_step,
    cpu_rng_seed,
)


class _Singleton(type):
    _instances: Dict[Type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _RNGManager:
    """
    A class used to manage a set of deterministic random number generators.

    The main goal of this class is to provide a simplified mechanism to
    improve determinism on the RNG side. This class includes a method to set
    a manual seed of all number generators and a method to register new
    generators.

    By default, Python (`random` module), NumPy, and PyTorch global generators
    are managed.
    """

    __metaclass__ = _Singleton

    RNG_DEF_REQUIRED_FIELDS = {"seed", "save_state", "load_state", "step"}

    def __init__(self):
        """
        Initializes this object, registering the default supported generators.
        """
        self.random_generators = OrderedDict()
        self._register_default_generators()

    def register_random_generator(self, name: str, rng_def: dict):
        """
        Register a new random number generator.

        Please note that Python's `random`, NumPy, and PyTorch global generators
        are already supported out-of-the-box and should not be re-registered.

        :param name: The name of the random number generator.
        :param rng_def: The definition of the random number generator.
            This must be a dictionary including the following fields:
            `seed`: a function that initializes the internal random
            number generator, it should accept an int (the seed);
            `save_state`: a function that returns an object that can be used to
            restore the state of the random number generator. The returned
            object should be pickleable; `load_state`: a function that sets
            the internal state of the  generator based on the object passed as
            an argument. That object is the one returned from a previous call to
            `save_state`; `step`: a function that advances the state of the
            random number generator. Its return value is ignored. To advance the
            state, it is recommended to generate a small amount of random
            data, like a float (to minimize the performance impact).
        """
        rng_def_keys = set(rng_def.keys())
        if not rng_def_keys.issubset(_RNGManager.RNG_DEF_REQUIRED_FIELDS):
            raise ValueError("Invalid random number generator definition")

        self.random_generators[name] = rng_def

    def _register_default_generators(self):
        self.register_random_generator(
            "torch",
            {
                "seed": cpu_rng_seed,
                "save_state": torch.random.get_rng_state,
                "load_state": torch.random.set_rng_state,
                "step": lambda: torch.rand(1),
            },
        )

        self.register_random_generator(
            "torch.cuda",
            {
                "seed": cuda_rng_seed,
                "save_state": cuda_rng_save_state,
                "load_state": cuda_rng_load_state,
                "step": cuda_rng_step,
            },
        )

        self.register_random_generator(
            "numpy",
            {
                "seed": np.random.seed,
                "save_state": np.random.get_state,
                "load_state": np.random.set_state,
                "step": lambda: np.random.rand(1),
            },
        )

        self.register_random_generator(
            "random",
            {
                "seed": random.seed,
                "save_state": random.getstate,
                "load_state": random.setstate,
                "step": random.random,
            },
        )

    def set_random_seeds(self, random_seed):
        """
        Set the initial seed of all number generators.

        :param random_seed: The initial seed. It should be a value compatible
            with all registered number generators. A native `int` value in
            range `[0, 2^32 - 1)` is usually acceptable.
        """

        for gen_name, gen_dict in self.random_generators.items():
            gen_dict["seed"](random_seed)

    def align_seeds(self):
        """
        Align random number generator seeds by using the next PyTorch generated
        integer value.
        """

        reference_seed = torch.randint(0, 2**32 - 1, (1,), dtype=torch.int64)

        seed = int(reference_seed)
        self.set_random_seeds(seed)

    def __getstate__(self):
        all_rngs_state = dict()
        for rng_name, rng_def in self.random_generators.items():
            rng_state = dict()
            rng_state["current_state"] = rng_def["save_state"]()
            all_rngs_state[rng_name] = rng_state
        return all_rngs_state

    def step_generators(self):
        for rng_name, rng_def in self.random_generators.items():
            rng_def["step"]()

    def __setstate__(self, rngs):
        # Note on the following:
        # here we load self.random_generators from the global singleton.
        # We have to do this because, while "rngs" could contain the un-pickled
        # rng functions (load_state, save_state, step), those functions
        # are incorrectly pickled and un-pickled.
        # In other words, they cannot be used to actually re-load the state
        # of the related number generator after closing and reopening
        # the process! This has been tested thanks to a generous expenditure of
        # man-hours.
        # On the other hand, RNGManager.random_generators will contain the
        # "correct" (of the current process) functions linked to the global
        # generators. User will need to register custom generators before
        # un-pickling _RNGManager objects (usually by loading a checkpoint).
        # Also consider that this object will probably not be used...
        # the goal of this __setstate__ is to load the state of the
        # global number generators registered in the singleton.
        self.random_generators = RNGManager.random_generators
        for rng_name, rng_def in self.random_generators.items():
            loaded_state = rngs[rng_name]["current_state"]
            rng_def["load_state"](loaded_state)

    def _replace_generators(self, generators):
        """
        For internal use only.
        """
        self.random_generators = generators


RNGManager = _RNGManager()


__all__ = ["RNGManager"]
