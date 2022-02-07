from os.path import expanduser

import torch

from avalanche.benchmarks.datasets import CIFAR100
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    ExperienceAccuracy,
    StreamAccuracy,
    EpochAccuracy,
)
from avalanche.logging.interactive_logging import InteractiveLogger
import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.training.supervised.icarl import ICaRL


def get_dataset_per_pixel_mean(dataset):
    result = None
    patterns_count = 0

    for img_pattern, _ in dataset:
        if result is None:
            result = torch.zeros_like(img_pattern, dtype=torch.float)

        result += img_pattern
        patterns_count += 1

    if result is None:
        result = torch.empty(0, dtype=torch.float)
    else:
        result = result / patterns_count

    return result


def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ]
    else:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t


def run_experiment(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    per_pixel_mean = get_dataset_per_pixel_mean(
        CIFAR100(
            expanduser("~") + "/.avalanche/data/cifar100/",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    )

    transforms_group = dict(
        eval=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                ]
            ),
            None,
        ),
        train=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                    icarl_cifar100_augment_data,
                ]
            ),
            None,
        ),
    )

    train_set = CIFAR100(
        expanduser("~") + "/.avalanche/data/cifar100/",
        train=True,
        download=True,
    )
    test_set = CIFAR100(
        expanduser("~") + "/.avalanche/data/cifar100/",
        train=False,
        download=True,
    )

    train_set = AvalancheDataset(
        train_set,
        transform_groups=transforms_group,
        initial_transform_group="train",
    )
    test_set = AvalancheDataset(
        test_set,
        transform_groups=transforms_group,
        initial_transform_group="eval",
    )

    scenario = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=config.nb_exp,
        task_labels=False,
        seed=config.seed,
        shuffle=False,
        fixed_class_order=config.fixed_class_order,
    )

    evaluator = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[InteractiveLogger()],
    )

    model: IcarlNet = make_icarl_net(num_classes=100)
    model.apply(initialize_icarl_net)

    optim = SGD(
        model.parameters(),
        lr=config.lr_base,
        weight_decay=config.wght_decay,
        momentum=0.9,
    )
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, config.lr_milestones, gamma=1.0 / config.lr_factor)
    )

    strategy = ICaRL(
        model.feature_extractor,
        model.classifier,
        optim,
        config.memory_size,
        buffer_transform=transforms.Compose([icarl_cifar100_augment_data]),
        fixed_memory=True,
        train_mb_size=config.batch_size,
        train_epochs=config.epochs,
        eval_mb_size=config.batch_size,
        plugins=[sched],
        device=device,
        evaluator=evaluator,
    )

    for i, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: i + 1]
        strategy.train(exp, num_workers=4)
        strategy.eval(eval_exps, num_workers=4)


class Config(dict):
    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == "__main__":
    config = Config()

    config.batch_size = 128
    config.nb_exp = 10
    config.memory_size = 2000
    config.epochs = 70
    config.lr_base = 2.0
    config.lr_milestones = [49, 63]
    config.lr_factor = 5.0
    config.wght_decay = 0.00001
    config.fixed_class_order = [
        87,
        0,
        52,
        58,
        44,
        91,
        68,
        97,
        51,
        15,
        94,
        92,
        10,
        72,
        49,
        78,
        61,
        14,
        8,
        86,
        84,
        96,
        18,
        24,
        32,
        45,
        88,
        11,
        4,
        67,
        69,
        66,
        77,
        47,
        79,
        93,
        29,
        50,
        57,
        83,
        17,
        81,
        41,
        12,
        37,
        59,
        25,
        20,
        80,
        73,
        1,
        28,
        6,
        46,
        62,
        82,
        53,
        9,
        31,
        75,
        38,
        63,
        33,
        74,
        27,
        22,
        36,
        3,
        16,
        21,
        60,
        19,
        70,
        90,
        89,
        43,
        5,
        42,
        65,
        76,
        40,
        30,
        23,
        85,
        2,
        95,
        56,
        48,
        71,
        64,
        98,
        13,
        99,
        7,
        34,
        55,
        54,
        26,
        35,
        39,
    ]
    config.seed = 2222

    run_experiment(config)
