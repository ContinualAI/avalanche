################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 22-06-2021                                                             #
# Author(s): Tom Veniat                                                        #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
In this example we show a simple way to use the ctrl benchmark presented
in https://arxiv.org/abs/2012.12631.
The training procedure will report the Transfer metric as defined in
eq.3 in the article for all streams but the long one, for which
the average accuracy after training on the whole stream is reported.
 """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from copy import deepcopy

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic.ctrl import CTrL
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleCNN
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.training.supervised import Naive


def main(args):
    # Device config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # Intialize the model, stream and training strategy
    model = SimpleCNN(num_classes=10)
    if args.stream != "s_long":
        model_init = deepcopy(model)

    scenario = CTrL(
        stream_name=args.stream, save_to_disk=args.save, path=args.path, seed=10
    )

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream
    val_stream = scenario.val_stream

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    logger = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=False, experience=True, stream=True
        ),
        loggers=[InteractiveLogger()],
    )

    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        device=device,
        train_epochs=args.max_epochs,
        eval_mb_size=128,
        evaluator=logger,
        plugins=[EarlyStoppingPlugin(50, "val_stream")],
        eval_every=5,
    )

    # train and test loop
    for train_task, val_task in zip(train_stream, val_stream):
        cl_strategy.train(train_task, eval_streams=[val_task])
        cl_strategy.eval(test_stream)

    transfer_mat = []
    for tid in range(len(train_stream)):
        transfer_mat.append(
            logger.all_metric_results[
                "Top1_Acc_Exp/eval_phase/test_stream/" f"Task00{tid}/Exp00{tid}"
            ][1]
        )

    if args.stream == "s_long":
        res = logger.last_metric_results[
            "Top1_Acc_Stream/eval_phase/" "test_stream"
        ]
        print(f"Average accuracy on S_long : {res}")
    else:
        optimizer = SGD(model_init.parameters(), lr=0.001, momentum=0.9)
        cl_strategy = Naive(
            model_init,
            optimizer,
            criterion,
            train_mb_size=32,
            device=device,
            train_epochs=args.max_epochs,
            eval_mb_size=128,
            plugins=[EarlyStoppingPlugin(50, "val_stream")],
            eval_every=5,
        )

        cl_strategy.train(train_stream[-1])
        res = cl_strategy.eval([test_stream[-1]])

        acc_last_stream = transfer_mat[-1][-1]
        acc_last_only = res[
            "Top1_Acc_Exp/eval_phase/test_stream/" "Task005/Exp-01"
        ]
        transfer_value = acc_last_stream - acc_last_only

        print(
            f"Accuracy on probe task after training on the whole "
            f"stream: {acc_last_stream}"
        )
        print(
            f"Accuracy on probe task after trained "
            f"independently: {acc_last_only}"
        )
        print(f"T({args.stream})={transfer_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream",
        type=str,
        default="s_plus",
        choices=["s_plus", "s_minus", "s_in", "s_out", "s_pl", "s_long"],
        help="Select the CTrL Stream to train on: [s_plus], "
        "s_minus, s_in, s_out, s_pl or s_long.",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Whether to save the generated experiences to"
        " disk or load them all in memory.",
    )
    parser.add_argument(
        "--path", type=str, help="Path used to save the generated stream."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="The maximum number of training epochs for each "
        "task. Default to 200.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
