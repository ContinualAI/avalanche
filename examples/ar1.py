################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 08-02-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
This is a simple example on how to use the AR1 strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

from avalanche.benchmarks import SplitCIFAR10
from avalanche.training.strategies import AR1


def main(args):
    # Device config
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose([
        Resize(224),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        Resize(224),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # ---------

    # --- SCENARIO CREATION
    scenario = SplitCIFAR10(5, train_transform=train_transform,
                            test_transform=test_transform)
    # ---------

    # CREATE THE STRATEGY INSTANCE
    cl_strategy = AR1(criterion=CrossEntropyLoss(), device=device)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for step in scenario.train_stream:
        print("Start of step: ", step.current_step)
        print("Current Classes: ", step.classes_in_this_step)

        cl_strategy.train(step, num_workers=4)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()
    main(args)
