import torch
from avalanche.benchmarks import GenericCLScenario, dataset_benchmark
from avalanche.benchmarks.utils import AvalancheTensorDataset
from torchvision import transforms

import ctrl


def CTrL(stream_name):
    stream = ctrl.get_stream(stream_name)

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
