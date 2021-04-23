import torch
from avalanche.benchmarks import GenericCLScenario, dataset_benchmark
from avalanche.benchmarks.utils import AvalancheTensorDataset
from torchvision import transforms

import ctrl


def CTrL(stream_name):
    stream = ctrl.get_stream(stream_name)

    exps = [[], [], []]
    norms = []
    for t in stream:
        for split, exp in zip(t.datasets, exps):
            samples, labels = split.tensors
            # samples -= torch.tensor(t.statistics['mean']).view(1, 3, 1, 1)
            # samples /= torch.tensor(t.statistics['std']).view(1, 3, 1, 1)

            task_labels = [t.id] * samples.size(0)
            dataset = AvalancheTensorDataset(samples, labels.squeeze(1),
                                             task_labels=task_labels)
            exp.append(dataset)
        norms.append(transforms.Normalize(t.statistics['mean'],
                                          t.statistics['std']))
    return dataset_benchmark(
        train_datasets=exps[0],
        test_datasets=exps[2],
        other_streams_datasets=dict(valid=exps[1]),
    )
