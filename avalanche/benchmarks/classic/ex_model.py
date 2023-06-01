################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-04-2022                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""Ex-Model Continual Learning benchmarks as defined in
https://arxiv.org/abs/2112.06511"""
import urllib

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomCrop,
    RandomRotation,
    ToTensor,
    CenterCrop,
    Normalize,
    Resize,
)
from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10
from avalanche.benchmarks.classic.cmnist import SplitMNIST
from avalanche.benchmarks.classic.core50 import CORe50
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location

from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.models import LeNet5, SlimResNet18
from torchvision.transforms import Compose
from avalanche.evaluation.metrics import TaskAwareAccuracy
from avalanche.benchmarks import ExModelCLScenario, nc_benchmark
import copy

import urllib.request

# seeds from the original ex-model paper. Necessary to get the same class
# splits.
SEED_BENCHMARK_RUNS = [1234, 2345, 3456, 5678, 6789]


def _load_expert_models(scenario_name, base_model, run_id, len_stream):
    """Load ExML experts.

    If necessary, the model are automatically downloaded.
    """
    # base_dir = f'/raid/carta/EXML_CLVISION_PRETRAINED_EXPERTS/{scenario_name}'
    base_dir = default_dataset_location(
        f"EXML_CLVISION22_PRETRAINED_EXPERTS/{scenario_name}/run{run_id}"
    )

    weburl = (
        f"http://131.114.50.174/data/EXML_CLVISION22_PRETRAINED_EXPERTS"
        f"/{scenario_name}/run{run_id}"
    )

    experts_stream = []
    for i in range(len_stream):
        fname_i = f"{base_dir}/model_e{i}.pth"
        weburl_i = f"{weburl}/model_e{i}.pth"

        if not os.path.exists(fname_i):
            os.makedirs(base_dir, exist_ok=True)
            print(f"Downloading expert model {i}")
            urllib.request.urlretrieve(weburl_i, fname_i)

        model = copy.deepcopy(base_model)
        state_d = torch.load(fname_i)
        model.load_state_dict(state_d)
        model.to("cpu").eval()
        experts_stream.append(model)
    return experts_stream


def check_experts_accuracy(exml_benchmark):
    """Sanity check. Compute experts accuracy on the train stream."""
    print(
        type(exml_benchmark).__name__,
        "testing expert models on the original train stream",
    )
    for i, exp in enumerate(exml_benchmark.expert_models_stream):
        model = exp.expert_model
        model.to("cuda")
        acc = TaskAwareAccuracy()

        train_data = exml_benchmark.original_benchmark.train_stream[i].dataset
        for x, y, t in DataLoader(
            train_data, batch_size=256, pin_memory=True, num_workers=4
        ):
            x, y, t = x.to("cuda"), y.to("cuda"), t.to("cuda")
            y_pred = model(x)
            acc.update(y_pred, y, t)
        print(f"(i={i}) Original model accuracy: {acc.result()}")
        model.to("cpu")


class ExMLMNIST(ExModelCLScenario):
    """ExML scenario on MNIST data.

    The pretrained models and class splits are taken from
    https://arxiv.org/abs/2112.06511
    """

    def __init__(self, scenario="split", run_id=0):
        """Init.

        :param scenario: If 'split', use a class-incremental scenario with 5
            experiences (2 classes each). If 'joint', use a single experience
            with all the classes. This should be used only as a baseline since
            it is not a continual scenario.
        :param run_id: an integer in [0, 4]. Each run uses a different set of
            expert models and data splits.
        """
        assert scenario in {
            "split",
            "joint",
        }, "`scenario` argument must be one of {'split', 'joint'}."

        CURR_SEED = SEED_BENCHMARK_RUNS[run_id]

        transforms = Compose([Resize(32), Normalize((0.1307,), (0.3081,))])
        if scenario == "split":
            benchmark = SplitMNIST(
                n_experiences=5,
                return_task_id=False,
                seed=CURR_SEED,
                train_transform=transforms,
                eval_transform=transforms,
            )
        elif scenario == "joint":
            benchmark = SplitMNIST(
                n_experiences=1,
                return_task_id=False,
                seed=CURR_SEED,
                train_transform=transforms,
                eval_transform=transforms,
            )
        else:
            assert False, "Should never get here."

        ll = len(benchmark.train_stream)
        base_model = LeNet5(10, 1)
        experts = _load_expert_models(f"{scenario}_mnist", base_model, run_id, ll)
        super().__init__(benchmark, experts)


class ExMLCoRE50(ExModelCLScenario):
    """ExML scenario on CoRE50.

    The pretrained models and class splits are taken from
    https://arxiv.org/abs/2112.06511
    """

    def __init__(self, scenario="ni", run_id=0):
        """Init.

        :param scenario: The desired CoRE50 scenario. Supports 'nc', 'ni', and
            'joint', which is the scenario with a single experience.
        :param run_id: an integer in [0, 4]. Each run uses a different set of
            expert models and data splits.
        """

        assert scenario in {
            "ni",
            "joint",
            "nc",
        }, "`scenario` argument must be one of {'ni', 'joint', 'nc'}."

        core50_normalization = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        core50_train_transforms = Compose(
            [
                RandomHorizontalFlip(p=0.5),
                RandomCrop(size=128, padding=1),
                RandomRotation(15),
                ToTensor(),
                core50_normalization,
            ]
        )
        core50_eval_transforms = Compose(
            [CenterCrop(size=128), ToTensor(), core50_normalization]
        )

        if scenario == "ni":
            benchmark = CORe50(
                scenario="ni",
                train_transform=core50_train_transforms,
                eval_transform=core50_eval_transforms,
                run=run_id,
            )
        elif scenario == "nc":
            benchmark = CORe50(
                scenario="nc",
                train_transform=core50_train_transforms,
                eval_transform=core50_eval_transforms,
                run=run_id,
            )
        elif scenario == "joint":
            core50nc = CORe50(scenario="nc")
            train_cat = concat_datasets([e.dataset for e in core50nc.train_stream])
            test_cat = concat_datasets([e.dataset for e in core50nc.test_stream])
            benchmark = nc_benchmark(
                train_cat, test_cat, n_experiences=1, task_labels=False
            )
        else:
            assert False, "Should never get here."

        ll = len(benchmark.train_stream)
        base_model = mobilenet_v2()
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_model.last_channel, 50),
        )
        experts = _load_expert_models(f"{scenario}_core50", base_model, run_id, ll)
        super().__init__(benchmark, experts)


class ExMLCIFAR10(ExModelCLScenario):
    """ExML scenario on CIFAR10.

    The pretrained models and class splits are taken from
    https://arxiv.org/abs/2112.06511
    """

    def __init__(self, scenario="split", run_id=0):
        """Init.

        :param scenario: If 'split', use a class-incremental scenario with 5
            experiences (2 classes each). If 'joint', use a single experience
            with all the classes. This should be used only as a baseline since
            it is not a continual scenario.
        :param run_id: an integer in [0, 4]. Each run uses a different set of
            expert models and data splits.
        """

        assert scenario in {
            "split",
            "joint",
        }, "`scenario` argument must be one of {'split', 'joint'}."

        CURR_SEED = SEED_BENCHMARK_RUNS[run_id]

        if scenario == "split":
            benchmark = SplitCIFAR10(
                n_experiences=5, return_task_id=False, seed=CURR_SEED
            )
        elif scenario == "joint":
            benchmark = SplitCIFAR10(
                n_experiences=1, return_task_id=False, seed=CURR_SEED
            )
        else:
            assert False, "Should never get here."

        ll = len(benchmark.train_stream)
        base_model = SlimResNet18(10)
        experts = _load_expert_models(f"{scenario}_cifar10", base_model, run_id, ll)
        super().__init__(benchmark, experts)


if __name__ == "__main__":
    """Sanity checks and automatic download test."""
    check_experts_accuracy(ExMLMNIST(scenario="split"))
    check_experts_accuracy(ExMLMNIST(scenario="joint"))
    # check_experts_accuracy(ExMLCIFAR10(scenario='split'))
    # check_experts_accuracy(ExMLCIFAR10(scenario='joint'))
    # check_experts_accuracy(ExMLCoRE50(scenario='ni'))
    # check_experts_accuracy(ExMLCoRE50(scenario='nc'))
    # check_experts_accuracy(ExMLCoRE50(scenario='joint'))


__all__ = ["ExMLMNIST", "ExMLCIFAR10", "ExMLCoRE50", "check_experts_accuracy"]
