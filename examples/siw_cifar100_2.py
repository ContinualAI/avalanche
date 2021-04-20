from avalanche.benchmarks.classic import SplitCIFAR100
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, \
    AGEMPlugin, SIWPlugin, EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import accuracy_metrics
import torchvision
from avalanche.benchmarks.generators import filelist_scenario, \
    dataset_scenario, tensor_scenario, paths_scenario
from torchvision.transforms import Compose, CenterCrop, Normalize, \
    Scale, Resize, ToTensor, ToPILImage
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse


def main(args):
    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available()
                          and args.cuda >= 0 else "cpu")
    print(f'Using device: {device}')
    #############################################
    model = torchvision.models.resnet18(num_classes=100).to(device)

    siw = SIWPlugin(model, siw_layer_name=args.siw_layer_name,
                    batch_size=args.siw_batch_size,
                    num_workers=args.siw_num_workers)

    # log to text file
    text_logger = TextLogger(open('/scratch_global/eden/avalanche/log.txt', 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger]
    )

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = CrossEntropyLoss()
    strategy = Naive(model, optimizer, criterion, plugins=[siw],
                     device=device, train_epochs=args.epochs, evaluator=eval_plugin)

    normalize = transforms.Normalize(mean=[0.5356, 0.4898, 0.4255],
                                     std=[0.2007, 0.1999, 0.1992])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    # # scenario
    # scenario = SplitCIFAR100(n_experiences=10, return_task_id=False,
    #                          seed=1234, train_transform=train_transform,
    #                          eval_transform=test_transform)

    # scenario
    scenario = filelist_scenario(
        root="",
        train_file_lists=["/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch1",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch2",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch3",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch4",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch5",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch6",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch7",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch8",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch9",
                          "/scratch_global/eden/images_list_files/cifar100/s10/separated/train/batch10"],

        test_file_lists=["/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch1",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch2",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch3",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch4",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch5",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch6",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch7",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch8",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch9",
                         "/scratch_global/eden/images_list_files/cifar100/s10/accumulated/test/batch10"],

        task_labels=list(range(0, 10)),
        complete_test_set_only=False,
        train_transform=train_transform,
        eval_transform=test_transform,
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for i, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        strategy.train(experience)
        print('Training completed')
        print('Computing accuracy on the test set')
        res = strategy.eval(scenario.test_stream[i])
        results.append(res)

    print('Results = ' + str(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--siw_batch_size', type=int, default=128,
                        help='Batch size used to extract scores.')
    parser.add_argument('--siw_num_workers', type=int, default=8,
                        help='Number of workers used to extract scores.')
    parser.add_argument('--siw_layer_name', type=str, default='fc',
                        help='Name of the last fully connected layer.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify GPU id to use. Use CPU if -1.')
    args = parser.parse_args()

    main(args)
