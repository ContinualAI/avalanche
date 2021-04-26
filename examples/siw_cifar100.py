from avalanche.benchmarks.classic import SplitCIFAR100
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.strategies import Naive
from avalanche.training.plugins import SIWPlugin,\
    EvaluationPlugin, StrategyPlugin
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import accuracy_metrics
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import argparse
from torch.optim import lr_scheduler


class LRSchedulerPlugin(StrategyPlugin):
    def __init__(self, lr_scheduler):
        super().__init__()
        self.lr_scheduler = lr_scheduler

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self.lr_scheduler.step(strategy.loss.cpu().data.numpy())
        lr = strategy.optimizer.param_groups[0]['lr']
        print(f"\nlr = {lr}")


class SetIncrementalHyperParams(StrategyPlugin):
    def __init__(self, inc_exp_epochs, inc_exp_patience, first_exp_lr,
                 lr_decay):
        super().__init__()
        self.inc_exp_epochs = inc_exp_epochs
        self.inc_exp_patience = inc_exp_patience
        self.first_exp_lr = first_exp_lr
        self.lr_decay = lr_decay

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if strategy.experience.current_experience > 0:  # incremental update
            strategy.train_epochs = self.inc_exp_epochs
            strategy.optimizer.param_groups[0]['lr'] = \
                self.first_exp_lr / strategy.experience.current_experience
            strategy.scheduler = LRSchedulerPlugin(
                lr_scheduler.ReduceLROnPlateau(strategy.optimizer,
                                               patience=self.inc_exp_patience,
                                               factor=self.lr_decay))


def main(args):
    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available()
                          and args.cuda >= 0 else "cpu")
    print(f'Using device: {device}')
    #############################################
    model = torchvision.models.resnet18(num_classes=100).to(device)

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True,
                         stream=True),
        loggers=[interactive_logger]
    )

    optimizer = SGD(model.parameters(), lr=args.first_exp_lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()
    scheduler = LRSchedulerPlugin(
        lr_scheduler.ReduceLROnPlateau(optimizer,
                                       patience=args.first_exp_patience,
                                       factor=args.lr_decay))
    incremental_params = SetIncrementalHyperParams(args.inc_exp_epochs,
                                                   args.inc_exp_patience,
                                                   args.first_exp_lr,
                                                   args.lr_decay)

    siw = SIWPlugin(model, siw_layer_name=args.siw_layer_name,
                    batch_size=args.eval_batch_size,
                    num_workers=args.num_workers)

    strategy = Naive(model, optimizer, criterion,
                     device=device, train_epochs=args.first_exp_epochs,
                     evaluator=eval_plugin,
                     plugins=[siw, scheduler, incremental_params],
                     train_mb_size=args.train_batch_size,
                     eval_mb_size=args.eval_batch_size)

    normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])

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

    # scenario
    scenario = SplitCIFAR100(n_experiences=10, return_task_id=False,
                             fixed_class_order=range(0, 100),
                             train_transform=train_transform,
                             eval_transform=test_transform)
    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for i, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        strategy.train(experience, num_workers=args.num_workers)
        print('Training completed')
        print('Computing accuracy on the test set')
        res = strategy.eval(scenario.test_stream[:i + 1],
                            num_workers=args.num_workers)
        results.append(res)

    print('Results = ' + str(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_exp_lr', type=float, default=0.1,
                        help='Learning rate for the first experience.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='LR decay')
    parser.add_argument('--first_exp_patience', type=int, default=60,
                        help='Patience in the first experience')
    parser.add_argument('--inc_exp_patience', type=int, default=15,
                        help='Patience in the incremental experiences')
    parser.add_argument('--first_exp_epochs', type=int, default=300,
                        help='Number of epochs in the first experience.')
    parser.add_argument('--inc_exp_epochs', type=int, default=70,
                        help='Number of epochs in each incremental experience.')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='Training batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers used to extract scores.')
    parser.add_argument('--siw_layer_name', type=str, default='fc',
                        help='Name of the last fully connected layer.')
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify GPU id to use. Use CPU if -1.')
    args = parser.parse_args()

    main(args)
