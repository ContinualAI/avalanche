from typing import (
    Callable,
    Union,
    Tuple,
    Optional,
)

import random
from avalanche.benchmarks.utils.data import AvalancheDataset
from .generic_scenario import EagerCLStream, CLScenario, make_stream
from .dataset_scenario import (
    LazyTrainValSplitter,
    DatasetExperience,
    split_validation_random,
)
from .supervised import with_classes_timeline


def benchmark_with_validation_stream(
    benchmark: CLScenario,
    validation_size: Union[int, float] = 0.5,
    shuffle: bool = False,
    seed: Optional[int] = None,
    split_strategy: Optional[
        Callable[[AvalancheDataset], Tuple[AvalancheDataset, AvalancheDataset]]
    ] = None,
) -> CLScenario:
    """Helper to obtain a benchmark with a validation stream.

    This generator accepts an existing benchmark instance and returns a version
    of it in which the train stream has been split into training and validation
    streams.

    Each train/validation experience will be by splitting the original training
    experiences. Patterns selected for the validation experience will be removed
    from the training experiences.

    The default splitting strategy is a random split as implemented by `split_validation_random`.
    If you want to use class balancing you can use `split_validation_class_balanced`, or
    use a custom `split_strategy`, as shown in the following example::

        validation_size = 0.2
        foo = lambda exp: split_dataset_class_balanced(validation_size, exp)
        bm = benchmark_with_validation_stream(bm, custom_split_strategy=foo)

    :param benchmark: The benchmark to split.
    :param validation_size: The size of the validation experience, as an int
        or a float between 0 and 1. Ignored if `custom_split_strategy` is used.
    :param shuffle: If True, patterns will be allocated to the validation
        stream randomly. This will use the default PyTorch random number
        generator at its current state. Defaults to False. Ignored if
        `custom_split_strategy` is used. If False, the first instances will be
        allocated to the training  dataset by leaving the last ones to the
        validation dataset.
    :param split_strategy: A function that implements a custom splitting
        strategy. The function must accept an AvalancheDataset and return a tuple
        containing the new train and validation dataset. By default, the splitting
        strategy will split the data according to `validation_size` and `shuffle`).
        A good starting to understand the mechanism is to look at the
        implementation of the standard splitting function
        :func:`random_validation_split_strategy`.

    :return: A benchmark instance in which the validation stream has been added.
    """

    if split_strategy is None:
        if seed is None:
            seed = random.randint(0, 1000000)

        # functools.partial is a more compact option
        # However, MyPy does not understand what a partial is -_-
        def random_validation_split_strategy_wrapper(data):
            return split_validation_random(validation_size, shuffle, seed, data)

        split_strategy = random_validation_split_strategy_wrapper
    else:
        split_strategy = split_strategy

    stream = benchmark.streams["train"]
    if isinstance(stream, EagerCLStream):  # eager split
        train_exps, valid_exps = [], []

        exp: DatasetExperience
        for exp in stream:
            train_data, valid_data = split_strategy(exp.dataset)
            train_exps.append(DatasetExperience(dataset=train_data))
            valid_exps.append(DatasetExperience(dataset=valid_data))
    else:  # Lazy splitting (based on a generator)
        split_generator = LazyTrainValSplitter(split_strategy, stream)
        train_exps = (DatasetExperience(dataset=a) for a, _ in split_generator)
        valid_exps = (DatasetExperience(dataset=b) for _, b in split_generator)

    train_stream = make_stream(name="train", exps=train_exps)
    valid_stream = make_stream(name="valid", exps=valid_exps)
    other_streams = benchmark.streams

    # don't drop classes-timeline for compatibility with old API
    e0 = next(iter(train_stream))

    if hasattr(e0, "dataset") and hasattr(e0.dataset, "targets"):
        train_stream = with_classes_timeline(train_stream)
        valid_stream = with_classes_timeline(valid_stream)

    del other_streams["train"]
    return CLScenario(
        streams=[train_stream, valid_stream] + list(other_streams.values())
    )


__all__ = ["benchmark_with_validation_stream"]
