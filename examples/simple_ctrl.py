################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
In this simple example we show all the different ways you can use MNIST with
Avalanche.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import argparse

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST, \
    SplitMNIST
from avalanche.benchmarks.classic.ctrl import CTrL
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.training.strategies import Naive

def main(args):
    # Device config
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # model
    # model = SimpleMLP(num_classes=11, input_size=3*32*32)
    model = SimpleCNN(num_classes=10)

    scenario = CTrL(stream_name='s_plus')

    # Than we can extract the parallel train and test streams
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Continual learning strategy with default logger
    cl_strategy = Naive(
        model, optimizer, criterion, train_mb_size=32, train_epochs=200,
        eval_mb_size=32, device=device)

    # train and test loop
    results = []
    transfer_mat = []
    for train_task in train_stream:
        print("Current Classes: ", train_task.classes_in_this_experience)
        res = cl_strategy.train(train_task)
        print('')
        print(f'Train res: {res}')
        results.append(cl_strategy.eval(test_stream)) 
        print(f'Val res: {results}')
        
        transfer_mat.append([results[-1][f'Top1_Acc_Exp/eval_phase/test_stream/Task00{i}/Exp00{i}'] for i in range(len(train_stream))])
        print(torch.tensor(transfer_mat))
        print((torch.tensor(transfer_mat)*1000).round()/1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist_type', type=str, default='split',
                        choices=['rotated', 'permuted', 'split'],
                        help='Choose between MNIST variations: '
                             'rotated, permuted or split.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()

    main(args)
