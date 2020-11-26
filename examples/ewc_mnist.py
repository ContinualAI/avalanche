import torch
import argparse
from avalanche.benchmarks import PermutedMNIST, SplitMNIST
from avalanche.training.strategies import EWC
from avalanche.extras import SimpleMLP

"""
This example tests EWC on Split MNIST and Permuted MNIST.
It is possible to choose, among other options, between EWC with separate
penalties and online EWC with a single penalty.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str,
                    choices=['pmnist', 'smnist'], default='smnist',
                    help='Choose between Permuted MNIST, Split MNIST.')
parser.add_argument('--ewc_mode', type=str, choices=['separate', 'online'],
                    default='separate', help='Choose between EWC and online.')                    
parser.add_argument('--ewc_lambda', type=float, default=0.4,
                    help='Penalty hyperparameter for EWC')
parser.add_argument('--decay_factor', type=float, default=0.1,
                    help='Decay factor for importance when ewc_mode is online.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--hs', type=int, default=256, help='MLP hidden size.')
parser.add_argument('--epochs', type=int, default=10, 
                    help='Number of training epochs.')
parser.add_argument('--minibatch_size', type=int, default=128,
                    help='Minibatch size.')
parser.add_argument('--permutations', type=int, default=5,
                    help='Number of steps in Permuted MNIST.')
parser.add_argument('--cuda', type=int, default=-1,
                    help='Specify GPU id to use. Use CPU if -1.')
args = parser.parse_args()


model = SimpleMLP(hidden_size=args.hs)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# check if selected GPU is available or use CPU
assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
if args.cuda >= 0:
    assert torch.cuda.device_count() > args.cuda, \
           f"{args.cuda + 1} GPU needed. Found {torch.cuda.device_count()}."
device = 'cpu' if args.cuda == -1 else f'cuda:{args.cuda}'
print(f'Using device: {device}')

# create strategy
strategy = EWC(model, optimizer, criterion, args.ewc_lambda,
               args.ewc_mode, decay_factor=args.decay_factor,
               train_epochs=args.epochs, device=device,
               train_mb_size=args.minibatch_size)

# create scenario
if args.scenario == 'pmnist':
    scenario = PermutedMNIST(incremental_steps=args.permutations)
elif args.scenario == 'smnist':
    scenario = SplitMNIST(incremental_steps=5, return_task_id=False)
else:
    raise ValueError("Wrong scenario name. Allowed pmnist, smnist.")

# train on the selected scenario with the chosen strategy
print('Starting experiment...')
results = []
for train_batch_info in scenario.train_stream:
    print("Start training on step ", train_batch_info.current_step)

    strategy.train(train_batch_info, num_workers=4)
    print("End training on step ", train_batch_info.current_step)
    print('Computing accuracy on the test set')
    results.append(strategy.test(scenario.test_stream[:]))
