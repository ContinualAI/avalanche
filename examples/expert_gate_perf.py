import os
import torch
from torch import nn

from torchvision import transforms, datasets
import torchvision.models as models

from torch.optim import SGD, lr_scheduler
from torch.nn import CrossEntropyLoss

from avalanche.models import ExpertGate
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators import paths_benchmark
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    gpu_usage_metrics,
)
from avalanche.training.supervised import ExpertGateStrategy, Naive

"""
This example tests ExpertGate on the benchmarks from the original paper. 
Please note (for now) that you will have to manually ensure that these 
benchmarks exist at the appropriate locations. It is meant to demonstrate
how one would set up this strategy.

Locations:
~/datasets/flowers102
~/datasets/cub2002011
~/datasets/scenes

For an example with no manual set up, please take a look at `expert_gate.py`
which trains with the SplitMNIST dataset.
"""


def main():
    # Set up benchmarks
    benchmark_setup()

    # Models to train
    train_and_eval_expertgate()
    # train_and_eval_single_alexnet()


##################
# MODEL TRAINING #
##################

def train_and_eval_expertgate():
    '''
    Train an Expert Gate model with SGD.
    '''
    # Set up experiment
    model_type = "expertgate"
    experiment_setup(run_name=model_type)

    # Set up pretrained AlexNet
    model = ExpertGate(shape=(3, 227, 227), device=device)

    # Set up training modules
    optimizer = SGD(model.parameters(), lr=0.001,
                    momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler_plugin = LRSchedulerPlugin(scheduler=scheduler)

    # Set up strategy
    strategy = ExpertGateStrategy(
        model,
        optimizer,
        device=device,
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=32,
        ae_train_mb_size=32,
        ae_train_epochs=1, 
        ae_lr=5e-4,
        plugins=[scheduler_plugin],
        evaluator=eval_plugin,
    )

    # Train on scenarios
    train_on_scenario(strategy, model_type)


def train_and_eval_single_alexnet():
    '''
    Train a single Alexnet with SGD and Naive fine-tuning strategy. It relies on
    switching out the final classification layer.
    '''

    # Set up experiment
    model_type = "single_alexnet"
    experiment_setup(model_type)

    # Set up pretrained AlexNet
    model = (models.__dict__["alexnet"]
             (weights='AlexNet_Weights.IMAGENET1K_V1')
             .to(device))
    original_classifier_input_dim = model._modules['classifier'][-1].in_features

    # Set up training modules
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001,
                    momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler_plugin = LRSchedulerPlugin(scheduler=scheduler)
    strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=32, train_epochs=50, eval_mb_size=32, device=device,
        evaluator=eval_plugin, plugins=[scheduler_plugin]
        )

    # Set up final layers
    layer_dict[scenes_loc] = nn.Linear(
        original_classifier_input_dim, scenes_classes)
    layer_dict[birds_loc] = nn.Linear(
        original_classifier_input_dim, birds_classes)
    layer_dict[flowers_loc] = nn.Linear(
        original_classifier_input_dim, flowers_classes)

    # Train on scenes
    train_on_scenario(strategy, model_type)

####################
# TRAIN-TEST LOGIC #
####################


def train_on_scenario(strategy, model_type):
    # Train on scenes
    train_on_benchmark(strategy, scenes_loc, model_type)
    eval_on_benchmark(strategy, scenes_loc, model_type)

    # Train on birds
    train_on_benchmark(strategy, birds_loc, model_type)
    eval_on_benchmark(strategy, scenes_loc, model_type)
    eval_on_benchmark(strategy, birds_loc, model_type)

    # Train on flowers
    train_on_benchmark(strategy, flowers_loc, model_type)
    eval_on_benchmark(strategy, scenes_loc, model_type)
    eval_on_benchmark(strategy, birds_loc, model_type)
    eval_on_benchmark(strategy, flowers_loc, model_type)


def train_on_benchmark(strategy, task_id, model_type):

    # Get scenario
    scenario = scenarios[task_id]

    # Replace layer for single alexnet
    if (model_type == "single_alexnet"):
        strategy.model._modules['classifier'][-1] = layer_dict[task_id]

    # Train loop for benchmark
    for experience in (scenario.train_stream):
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        print()
        print(f'Task {t} batch {exp_id}')
        print(f'This batch contains {len(training_dataset)} patterns')
        print(f'Current Classes: {experience.classes_in_this_experience}')
        strategy.train(experience)

    # Update dictionary of final layers for single_alexnet
    if (model_type == "single_alexnet"):
        layer_dict[task_id] = strategy.model._modules['classifier'][-1]


def eval_on_benchmark(strategy, task_id, model_type):
    # Get scenario
    scenario = scenarios[task_id]

    # Replace layer
    if (model_type == "single_alexnet"):
        strategy.model._modules['classifier'][-1] = layer_dict[task_id]

    # Evaluation loop for birds scenario
    for experience in (scenario.test_stream):
        strategy.eval(experience)

#####################
# EXPERIMENT SETUPS #
#####################


def experiment_setup(run_name):
    # Set device
    global device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    interactive_logger = InteractiveLogger()

    # Comment out to avoid WandB
    # wandb_logger = WandBLogger(
    #     project_name="expertgate3", run_name=run_name
    # )

    global eval_plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=True,
            epoch_running=False,
            experience=False,
            stream=True,
        ),
        loss_metrics(
            minibatch=False,
            epoch=True,
            epoch_running=False,
            experience=False,
            stream=True,
        ),
        gpu_usage_metrics(
            0,
            every=0.5,
            minibatch=False,
            epoch=True,
            experience=False,
            stream=False,
        ),
        # Comment out to avoid WandB
        # loggers=[interactive_logger, wandb_logger],
        loggers=[interactive_logger]
    )

###################
# BENCHMARK SETUP #
###################


def benchmark_setup():

    # Define the data transform for AlexNet
    global AlexTransform 
    AlexTransform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor()])

    # Initialize empty list of tasks
    # Each task is a separate dataset 
    # (Oxford Flowers, Caltech Birds, MIT Scenes)
    global scenarios
    scenarios = []

    global scenes_loc, scenes_classes
    scenes_loc = 0
    scenes_classes = 67

    global birds_loc, birds_classes
    birds_loc = 1
    birds_classes = 200

    global flowers_loc, flowers_classes
    flowers_loc = 2
    flowers_classes = 102

    global layer_dict
    layer_dict = {}

    # Add scenes
    scenes_bc = build_scenes_benchmark(task_label=scenes_loc)
    scenarios.append(scenes_bc)

    # Add birds
    birds_bc = build_birds_benchmark(task_label=birds_loc)
    scenarios.append(birds_bc)

    # Add flowers
    flowers_bc = build_flowers_benchmark(task_label=flowers_loc)
    scenarios.append(flowers_bc)


def build_flowers_benchmark(task_label):
    trainset = datasets.Flowers102(
        root="~/datasets/flowers102",
        split="train",
        download=True)
    train_paths = trainset._image_files
    train_labels = trainset._labels

    testset = datasets.Flowers102(
        root="~/datasets/flowers102",
        split="test",
        download=True)
    test_paths = testset._image_files
    test_labels = testset._labels

    # Initialize train/test lists
    train_experiences = []
    test_experiences = []

    for p, l in zip(train_paths, train_labels):
        instance_tuple = (str(p), l)
        train_experiences.append(instance_tuple)

    for p, l in zip(test_paths, test_labels):
        instance_tuple = (str(p), l)
        test_experiences.append(instance_tuple)

    print()
    print("Train images: ", len(train_experiences))
    print("Test images: ", len(test_experiences))

    scenario = paths_benchmark(
        [train_experiences],
        [test_experiences],
        task_labels=[task_label],
        complete_test_set_only=True,
        train_transform=AlexTransform,
        eval_transform=AlexTransform,
    )
    print("Generated the 'Oxford Flowers 102' with the label",
          scenario.task_labels[0][0])

    return scenario


def build_scenes_benchmark(task_label):
    # Setup directory paths
    scenes_dir = os.path.join(os.path.expanduser("~"), "datasets/scenes")
    images_dir = os.path.join(scenes_dir, "images")

    # Initialize train/test lists
    train_experiences = []
    test_experiences = []

    # Load train file
    with open(os.path.join(scenes_dir, "train.txt"), "r") as train_f:
        train_paths = train_f.read().splitlines()

    # Load test file
    with open(os.path.join(scenes_dir, "test.txt"), "r") as test_f:
        test_paths = test_f.read().splitlines()

    # Get classes
    classes = sorted(os.listdir(images_dir))

    for label, rel_dir in enumerate(classes):
        # Grab all images for a class
        filenames_list = [f for f in 
                          os.listdir(os.path.join(images_dir, rel_dir)) 
                          if not f.startswith('.')]

        # Iterate through each file
        train_experience_path = []
        test_experience_path = []
        for name in filenames_list:
            rel_img_path = os.path.join(rel_dir, name)
            instance_tuple = (os.path.join(images_dir, rel_img_path), label)

            # Bin the file into train or test depending on whether it shows up 
            # in the train.txt or test.txt
            if (rel_img_path in train_paths):
                train_experience_path.append(instance_tuple)
            elif (rel_img_path in test_paths):
                test_experience_path.append(instance_tuple)

        # Merge with larger array
        train_experiences = [*train_experiences, *train_experience_path]
        test_experiences = [*test_experiences, *test_experience_path]

    print()
    print("Train images: ", len(train_experiences))
    print("Test images: ", len(test_experiences))

    # Generate scenario
    scenario = paths_benchmark(
        [train_experiences],
        [test_experiences],
        task_labels=[task_label],
        complete_test_set_only=True,
        train_transform=AlexTransform,
        eval_transform=AlexTransform,
    )
    print("Generated the 'MIT Scenes' benchmark with the label",
          scenario.task_labels[0][0])

    return scenario


def build_birds_benchmark(task_label):

    # Setup directory paths
    cub200_dir = os.path.join(os.path.expanduser("~"), "datasets/cub2002011")
    images_dir = os.path.join(cub200_dir, "images")

    # Initialize train/test lists
    train_experiences = []
    test_experiences = []

    # Load classes file
    classes = []
    with open(os.path.join(cub200_dir, "classes.txt"), "r") as class_f:
        for line in class_f:
            (k, v) = line.split()
            classes.append(v)

    # Load images file
    images_dict = {}
    with open(os.path.join(cub200_dir, "images.txt"), "r") as images_f:
        for line in images_f:
            (v, k) = line.split()
            images_dict[k] = int(v)

    # Load train test split file
    train_split_dict = {}
    with open(os.path.join(cub200_dir, "train_test_split.txt"), "r") as split_f:
        for line in split_f:
            (k, v) = line.split()
            train_split_dict[int(k)] = int(v)

    # Iterate through all classes
    for label, rel_dir in enumerate(classes):

        # Grab all images for a class
        filenames_list = [f for f in 
                          os.listdir(os.path.join(images_dir, rel_dir)) 
                          if not f.startswith('.')]

        # Iterate through each file
        train_experience_path = []
        test_experience_path = []
        for name in filenames_list:
            rel_img_path = os.path.join(rel_dir, name)
            instance_tuple = (os.path.join(images_dir, rel_img_path), label)

            # Bin the file into train or test depending on whether it shows up 
            # in the train.txt or not
            img_id = images_dict[rel_img_path]
            if (train_split_dict[img_id]):
                train_experience_path.append(instance_tuple)
            else:
                test_experience_path.append(instance_tuple)

        # Merge with larger array
        train_experiences = [*train_experiences, *train_experience_path]
        test_experiences = [*test_experiences, *test_experience_path]

    print()
    print("Train images: ", len(train_experiences))
    print("Test images: ", len(test_experiences))

    # Generate scenario
    scenario = paths_benchmark(
        [train_experiences],
        [test_experiences],
        task_labels=[task_label],
        complete_test_set_only=True,
        train_transform=AlexTransform,
        eval_transform=AlexTransform,
    )
    print("Generated the 'Caltech Birds 200' benchmark with the label",
          scenario.task_labels[0][0])

    return scenario


if __name__ == '__main__':
    main()
