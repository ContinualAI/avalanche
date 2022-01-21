################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example shows how to use Standalone metrics like `Accuracy`, `Loss`,
`ConfusionMatrix` and others. Please, refer to `eval_plugin` example to
dig deeper into the use of Plugin metrics, which are already integrated
in the Avalanche training and evaluation loops.
"""


# import a standalone metric
import torch
from avalanche.evaluation.metrics import Accuracy

# all standalone metrics
from avalanche.evaluation.metrics import Accuracy

# , Loss, # Loss
# Forgetting,  # Forgetting
# ConfusionMatrix, # Confusion Matrix
# CPUUsage, # CPU Usage
# DiskUsage, # Disk Usage
# MaxGPU, # Max GPU Usage
# MAC,  # Multiply and Accumulate
# MaxRAM, # Max RAM Usage
# ElapsedTime # Timing metrics

# create an instance of the standalone Accuracy metric
# initial accuracy is 0
acc_metric = Accuracy()
print("Initial Accuracy: ", acc_metric.result())  # output 0

# update method allows to keep the running average accuracy
# result method returns the current average accuracy
real_y = torch.tensor([1, 2]).long()
predicted_y = torch.tensor([1, 0]).float()
acc_metric.update(real_y, predicted_y)
acc = acc_metric.result()
print("Average Accuracy: ", acc)  # output 0.5

# you can continue to update the metric with new values
predicted_y = torch.tensor([1, 2]).float()
acc_metric.update(real_y, predicted_y)
acc = acc_metric.result()
print("Average Accuracy: ", acc)  # output 0.75

# reset accuracy to 0
acc_metric.reset()
print("After reset: ", acc_metric.result())  # output 0
