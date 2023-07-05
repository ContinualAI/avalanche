import warnings

import torch
from torch import default_generator


def cuda_rng_seed(seed):
    """
    Seeds the state of the cuda RNGs (all devices)
    """
    torch.cuda.manual_seed_all(seed)


def cpu_rng_seed(seed):
    """
    Seeds the state of the cpu RNG alone.

    This is different from `torch.random.manual_seed`, which also initializes
    the cuda RNGs.
    """
    default_generator.manual_seed(seed)


def cuda_rng_save_state():
    """
    Saves the state of the cuda RNGs.
    """
    return torch.cuda.get_rng_state_all()


def cuda_rng_load_state(rng_state):
    """
    Loads the state of cuda RNGs.

    :param rng_state: The list of RNG states (one per cuda device).
    """
    n_devices = torch.cuda.device_count()
    n_states = len(rng_state)
    if n_states < n_devices:
        warnings.warn(
            "Problem when reloading the state of torch.cuda RNGs: the given"
            "checkpoint contain a number of RNG states less than the "
            "number of currently available cuda devices "
            f"(got {n_states}, expected {n_devices}). "
            f"The RNG of cuda devices with ID >= {n_states} will not be "
            f"initialized!"
        )

    # The standard situation is n_devices == n_states: just re-load the state
    # of the RNGs of currently available GPUs. This reasoning also applies if
    # n_states > n_devices. However, this is a bad fit for n_states < n_devices
    # (and because of this, we show the warning above).
    for device_id, rng_state in enumerate(rng_state[:n_devices]):
        torch.cuda.set_rng_state(rng_state, f"cuda:{device_id}")


def cuda_rng_step():
    for device_id in range(torch.cuda.device_count()):
        torch.rand(1, device=f"cuda:{device_id}")


__all__ = [
    "cuda_rng_seed",
    "cpu_rng_seed",
    "cuda_rng_save_state",
    "cuda_rng_load_state",
    "cuda_rng_step",
]
