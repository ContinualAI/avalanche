################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-04-2021                                                             #
# Author(s): Tyler Hayes                                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Deep SLDA strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.classic import CORe50
from avalanche.training.strategies.deep_slda import StreamingLDA


def main(args):
    # Device config
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")
    print('device ', device)
    # ---------

    # --- TRANSFORMATIONS
    _mu = [0.485, 0.456, 0.406]  # imagenet normalization
    _std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mu,
                             std=_std)
    ])
    # ---------

    # --- SCENARIO CREATION
    scenario = CORe50(root=args.dataset_dir, scenario=args.scenario,
                      train_transform=transform, eval_transform=transform)
    test_data_loader = DataLoader(scenario.test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=8)
    # ---------

    # CREATE THE STRATEGY INSTANCE
    cl_strategy = StreamingLDA(args.feature_size, args.n_classes,
                               test_batch_size=args.batch_size,
                               shrinkage_param=args.shrinkage,
                               streaming_update_sigma=args.plastic_cov,
                               arch=args.arch,
                               imagenet_pretrained=args.imagenet_pretrained,
                               device=device)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for i, batch in enumerate(scenario.train_stream):
        print("\n----------- Batch {0}/{1} -------------".format(i + 1, len(
            scenario.train_stream)))
        train_loader = DataLoader(batch.dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=8)

        # fit SLDA model to batch (one sample at a time)
        cl_strategy.train_model(train_loader)

        # evaluate model on test data
        test_acc, preds = cl_strategy.evaluate_model(test_data_loader)

        print("------------------------------------------")
        print("Test Accuracy: %0.3f" % test_acc)
        print("------------------------------------------")

        # update results
        results.append(test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLDA Example with ResNet-18 on CORe50')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')

    parser.add_argument('--n_classes', type=int, default=50)
    parser.add_argument('--scenario', type=str, default="nc",
                        choices=['ni', 'nc', 'nic', 'nicv2_79', 'nicv2_196',
                                 'nicv2_391'])
    parser.add_argument('--dataset_dir', type=str,
                        default='/media/tyler/Data/datasets/core50/')

    # deep slda model parameters
    parser.add_argument('--arch', type=str, default='resnet18', choices=[
        'resnet18'])  # to change this, need to modify creation of `self.feature_extraction_wrapper'
    # `avalanche.training.strategies.deep_slda import StreamingLDA'
    parser.add_argument('--imagenet_pretrained', type=bool,
                        default=True)  # initialize backbone with imagenet pre-trained weights
    parser.add_argument('--feature_size', type=int,
                        default=512)  # feature size before output layer (512 for resnet-18)
    parser.add_argument('--shrinkage', type=float,
                        default=1e-4)  # shrinkage value
    parser.add_argument('--plastic_cov', type=bool,
                        default=True)  # plastic covariance matrix
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()
    main(args)
