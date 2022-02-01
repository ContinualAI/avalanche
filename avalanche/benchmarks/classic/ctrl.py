################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 22-06-2021                                                             #
# Author(s): Tom Veniat                                                        #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import random
import sys
from pathlib import Path

import torchvision.transforms.functional as F
from torchvision import transforms
from tqdm import tqdm

import ctrl
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import (
    AvalancheTensorDataset,
    common_paths_root,
    AvalancheDataset,
    PathsDataset,
)


def CTrL(
    stream_name: str,
    save_to_disk: bool = False,
    path: Path = default_dataset_location(""),
    seed: int = None,
    n_tasks: int = None,
):
    """
    Gives access to the Continual Transfer Learning benchmark streams
    introduced in https://arxiv.org/abs/2012.12631.
    :param stream_name: Name of the test stream to generate. Must be one of
    `s_plus`, `s_minus`, `s_in`, `s_out` and `s_pl`.
    :param save_to_disk:  Whether to save each stream on the disk or load
    everything in memory. Setting it to `True` will save memory but takes more
    time on the first generation using the corresponding seed.
    :param path: The path under which the generated stream will be saved if
    save_to_disk is True.
    :param seed: The seed to use to generate the streams. If no seed is given,
    a random one will be used to make sure that the generated stream can
    be reproduced.
    :param n_tasks: The number of tasks to generate. This parameter is only
    relevant for the `s_long` stream, as all other streams have a fixed number
    of tasks.
    :return: A scenario containing 3 streams: train, val and test.
    """
    seed = seed or random.randint(0, sys.maxsize)
    if stream_name != "s_long" and n_tasks is not None:
        raise ValueError(
            "The n_tasks parameter can only be used with the "
            f'"s_long" stream, asked {n_tasks} for {stream_name}'
        )
    elif stream_name == "s_long" and n_tasks is None:
        n_tasks = 100

    stream = ctrl.get_stream(stream_name, seed)

    if save_to_disk:
        folder = path / "ctrl" / stream_name / f"seed_{seed}"

    # Train, val and test experiences
    exps = [[], [], []]
    for t_id, t in enumerate(
        tqdm(stream, desc=f"Loading {stream_name}"),
    ):
        trans = transforms.Normalize(t.statistics["mean"], t.statistics["std"])
        for split, split_name, exp in zip(t.datasets, t.split_names, exps):
            samples, labels = split.tensors
            task_labels = [t.id] * samples.size(0)
            if save_to_disk:
                exp_folder = folder / f"exp_{t_id}" / split_name
                exp_folder.mkdir(parents=True, exist_ok=True)
                files = []
                for i, (sample, label) in enumerate(zip(samples, labels)):
                    sample_path = exp_folder / f"sample_{i}.png"
                    if not sample_path.exists():
                        F.to_pil_image(sample).save(sample_path)
                    files.append((sample_path, label.item()))

                common_root, exp_paths_list = common_paths_root(files)
                paths_dataset = PathsDataset(common_root, exp_paths_list)
                dataset = AvalancheDataset(
                    paths_dataset,
                    task_labels=task_labels,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), trans]
                    ),
                )
            else:
                dataset = AvalancheTensorDataset(
                    samples,
                    labels.squeeze(1),
                    task_labels=task_labels,
                    transform=trans,
                )
            exp.append(dataset)
        if stream_name == "s_long" and t_id == n_tasks - 1:
            break

    return dataset_benchmark(
        train_datasets=exps[0],
        test_datasets=exps[2],
        other_streams_datasets=dict(val=exps[1]),
    )


__all__ = ["CTrL"]
