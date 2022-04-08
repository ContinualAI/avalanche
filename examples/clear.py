################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 04-06-2022                                                             #
# Author: Jia Shi                                                              #
# E-mail: jiashi@andrew.cmu.edu                                                #
# Website: https://clear-benchmark.github.io                                   #
################################################################################


from torch.optim import SGD
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics
)
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
import sys
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch
from avalanche.benchmarks.classic.clear import CLEAR
from avalanche.benchmarks.datasets.clear import clear10_data
from avalanche.benchmarks.datasets.clear import (
    CLEARImage,
    CLEARFeature,
    SEED_LIST,
    CLEAR_FEATURE_TYPES
)


def make_scheduler(optimizer, step_size, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler


_CLEAR_DATA_MODULE = {
    'clear10' : clear10_data
}

NUM_CLASSES = {
    'clear10' : 11
}
CLEAR_FEATURE_TYPES = {
    'clear10' : ['moco_b0', 'moco_imagenet', 'byol_imagenet', 'imagenet']
}

CLEAR_FEATURE_SHAPE = {
    'imagenet': 2048
}


SPLIT_OPTIONS = ['all', 'train', 'test']

SEED_LIST = [0, 1, 2, 3, 4]  # Available seeds for train:test split

EVALUATION_PROTOCOLS = ['iid', 'streaming']


HYPER_PARAMETER = {
    'image': {
        'batch_size' : 256,
        'step_schedular_decay' : 30,
        'schedular_step' : 0.1,
        'start_lr' : 0.01,
        'weight_decay' : 1e-5,
        'momentum' : 0.9
    },
    'feature': {
        'batch_size' : 512,
        'step_schedular_decay' : 60,
        'schedular_step' : 0.1,
        'start_lr' : 1,
        'weight_decay' : 0,
        'momentum' : 0.9
    }
}


data_name = 'clear10'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

root = ".."

# log to Tensorboard
tb_logger = TensorboardLogger(root)

# log to text file
text_logger = TextLogger(open(f'{root}/log.txt', 'w'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=NUM_CLASSES[data_name],
                             save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=True,
                       epoch=True,
                       experience=True,
                       stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

num_epoch = 70
for eval_mode in EVALUATION_PROTOCOLS:
    for mode in ['feature', 'image']:
        if eval_mode == 'streaming':
            seed = None
        else:
            seed = 0
        if mode == 'image':
            scenario = CLEAR(
                evaluation_protocol=eval_mode,
                feature_type=None,
                seed=seed,
                train_transform=train_transform,
                eval_transform=test_transform,
                dataset_root=f"{root}/avalanche_datasets/clear10"
            )
            model = torchvision.models.__dict__['resnet18'](pretrained=True)
        elif mode == 'feature':
            scenario = CLEAR(
                            evaluation_protocol=eval_mode,
                            feature_type=CLEAR_FEATURE_TYPES[data_name][-1],
                            seed=seed,
                            dataset_root=f"{root}/avalanche_datasets/clear10"
                        )
            # feature size for imagenet is 2048
            model = nn.Linear(
                CLEAR_FEATURE_SHAPE['imagenet'],
                NUM_CLASSES[data_name]
            )
            
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if(torch.cuda.is_available()):
            model = model.cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = SGD(
            model.parameters(),
            lr=HYPER_PARAMETER[mode]['start_lr'], 
            weight_decay=HYPER_PARAMETER[mode]['weight_decay'],
            momentum=HYPER_PARAMETER[mode]['momentum']
        )
        scheduler = make_scheduler(
            optimizer,
            HYPER_PARAMETER[mode]['step_schedular_decay'],
            HYPER_PARAMETER[mode]['schedular_step']
        )

        plugin_list = [LRSchedulerPlugin(scheduler)]
        cl_strategy = Naive(
            model, optimizer, CrossEntropyLoss(), 
            train_mb_size=HYPER_PARAMETER[mode]['batch_size'], 
            train_epochs=num_epoch,
            eval_mb_size=HYPER_PARAMETER[mode]['batch_size'],
            evaluator=eval_plugin,
            device=device,
            plugins=plugin_list
        )

        # TRAINING LOOP
        print('Starting experiment...')
        results = []
        print("Current input mode : ", mode)
        print("Current eval mode : ", eval_mode)
        for experience in scenario.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            res = cl_strategy.train(experience)
            print('Training completed')
            print(
                'Computing accuracy on the whole test set with'
                f' {eval_mode} evaluation protocols'
            )
            results.append(cl_strategy.eval(scenario.test_stream))
