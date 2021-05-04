from torchvision import transforms

import ctrl
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheTensorDataset


def CTrL(stream_name: str, seed: int = None):
    """
    Gives access to the Continual Transfer Learning benchmark streams
    introduced in https://arxiv.org/abs/2012.12631.
    :param stream_name: Name of the test stream to generate. Must be one of
    `s_plus`, `s_minus`, `s_in`, `s_out` and `s_pl`.
    :param seed: The seed to use to generate the streams.
    :return: A scenario containing 3 streams: train, val and test.
    """
    stream = ctrl.get_stream(stream_name, seed)

    # Train, val and test experiences
    exps = [[], [], []]
    for t in stream:
        trans = transforms.Normalize(t.statistics['mean'],
                                     t.statistics['std'])
        for split, exp in zip(t.datasets, exps):
            samples, labels = split.tensors
            task_labels = [t.id] * samples.size(0)
            dataset = AvalancheTensorDataset(samples, labels.squeeze(1),
                                             task_labels=task_labels,
                                             transform=trans)
            exp.append(dataset)

    return dataset_benchmark(
        train_datasets=exps[0],
        test_datasets=exps[2],
        other_streams_datasets=dict(valid=exps[1]),
    )
