#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Common metrics for CL. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import numpy as np
import os
import psutil
from .utils import bytes2human
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import PIL.Image
from torchvision.transforms import ToTensor
import io
import queue
import subprocess
import threading
import time


class MAC:
    """
        Multiply-and-accumulate metric. Approximately measure the computational
        cost of a model in a hardware-independent way by computing the number
        of multiplications. Currently supports only Linear or Conv2d modules.
        Other operations are ignored.
    """
    def __init__(self):
        self.hooks = []
        self._compute_cost = 0

    def compute(self, model, dummy_input):
        for mod in model.modules():
            if self.is_recognized_module(mod):
                def foo(a, b, c):
                    return self.update_compute_cost(a, b, c)
                handle = mod.register_forward_hook(foo)
                self.hooks.append(handle)

        self._compute_cost = 0
        model(dummy_input)  # trigger forward hooks

        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        return self._compute_cost

    def update_compute_cost(self, module, input, output):
        modname = module.__class__.__name__
        if modname == 'Linear':
            self._compute_cost += input[0].shape[1] * output.shape[1]
        elif modname == 'Conv2d':
            n, cout, hout, wout = output.shape  # Batch, Channels, Height, Width
            ksize = module.kernel_size[0] * module.kernel_size[1]
            self._compute_cost += cout * hout * wout * (ksize)
        print(self._compute_cost)

    def is_recognized_module(self, mod):
        modname = mod.__class__.__name__
        return modname == 'Linear' or modname == 'Conv2d'


class GPUUsage:
    """
        GPU usage metric measured as average usage percentage over time.

        :param gpu_id: GPU device ID
        :param every: time delay (in seconds) between measurements
    """

    def __init__(self, gpu_id, every=10):
        # 'nvidia-smi --loop=1 --query-gpu=utilization.gpu --format=csv'
        cmd = ['nvidia-smi', f'--loop={every}', '--query-gpu=utilization.gpu',
               '--format=csv', f'--id={gpu_id}']
        # something long running
        try:
            self.p = subprocess.Popen(cmd, bufsize=1, stdout=subprocess.PIPE)
        except NotADirectoryError:
            raise ValueError('No GPU available: nvidia-smi command not found.')

        self.lines_queue = queue.Queue()
        self.read_thread = threading.Thread(target=GPUUsage.push_lines,
                                            args=(self,), daemon=True)
        self.read_thread.start()

        self.n_measurements = 0
        self.avg_usage = 0

    def compute(self, t):
        """
        Compute CPU usage measured in seconds.

        :param t: task id
        :return: float: average GPU usage
        """
        while not self.lines_queue.empty():
            line = self.lines_queue.get()
            if line[0] == 'u':  # skip first line 'utilization.gpu [%]'
                continue
            usage = int(line.strip()[:-1])
            self.n_measurements += 1
            self.avg_usage += usage

        if self.n_measurements > 0:
            self.avg_usage /= float(self.n_measurements)
        print(f"Train Task {t} - average GPU usage: {self.avg_usage}%")

        return self.avg_usage

    def push_lines(self):
        while True:
            line = self.p.stdout.readline()
            self.lines_queue.put(line.decode('ascii'))

    def close(self):
        self.p.terminate()


class CPUUsage:
    """
        CPU usage metric measured in seconds.
    """

    def compute(self, t):
        """
        Compute CPU usage measured in seconds.

        :param t: task id
        :return: tuple (float, float): (user CPU time, system CPU time)
        """
        p = psutil.Process(os.getpid())
        times = p.cpu_times()
        user, sys = times.user, times.system
        print("Train Task {:} - CPU usage: user {} system {}"
              .format(t, user, sys))
        return user, sys


class ACC(object):

    def __init__(self, num_class=None):
        """
        Accuracy metrics should be called for each test set

        :param num_class (int, optional): number of classes in the test_set
            (useful in case the test_set does not cover all the classes
            in the train_set).
        """

        self.num_class = num_class

    def compute(self, y, y_hat):
        """
        :param y (tensor list or tensor): true labels for each mini-batch
        :param y_hat (tensor list or tensor): predicted labels for each
            mini-batch

        :return acc (float): average accuracy for the test set
        :return accs (float list): accuracy for each class in the training set            
        """
        
        assert type(y) == type(y_hat), "Predicted and target labels must be \
                both list (of tensors) or tensors"
        
        # manage list of tensors by default
        if not (isinstance(y, list) or isinstance(y, tuple)):
            y = [y]
            y_hat = [y_hat]

        if self.num_class is None:
            num_class = int(max([torch.max(el).item() + 1 for el in y]))
        else:
            num_class = self.num_class

        hits_per_class = [0] * num_class
        pattern_per_class = [0] * num_class

        correct_cnt = 0.

        for true_y, y_pred in zip(y, y_hat):
            
            correct_cnt += (true_y == y_pred).sum().float()

            for label in true_y:
                pattern_per_class[int(label)] += 1

            for i, pred in enumerate(y_pred):
                if pred == true_y[i]:
                    hits_per_class[int(pred)] += 1

        accs = np.zeros(len(hits_per_class), dtype=np.float)
        hits_per_class = np.asarray(hits_per_class)
        pattern_per_class = np.asarray(pattern_per_class).astype(float)

        # np.divide prevents the true divide warning from showing up
        # when one or more elements of pattern_per_class are zero
        # Also, those elements will be 0 instead of NaN
        np.divide(hits_per_class, pattern_per_class,
                  where=pattern_per_class != 0, out=accs)
        accs = torch.from_numpy(accs)

        acc = correct_cnt / float(y_hat[0].size(0) * len(y_hat))

        return acc, accs


class CF(object):

    def __init__(self, num_class=None):
        """
        Catastrophic Forgetting metric.
        """

        self.best_acc = {}
        self.acc_metric = ACC(num_class=num_class)

    def compute(self, y, y_hat, train_t, test_t):
        """
        :param y (tensor list or tensor): true labels for each mini-batch
        :param y_hat (tensor list or tensor): predicted labels for each
            mini-batch
        """

        acc, accs = self.acc_metric.compute(y, y_hat)
        if train_t not in self.best_acc.keys() and train_t == test_t:
            self.best_acc[train_t] = acc

        if test_t not in self.best_acc.keys():
            cf = np.NAN
        else:
            cf = self.best_acc[test_t] - acc

        print("Task {:} - CF: {:.4f}"
              .format(test_t, cf))

        return cf


class RAMU(object):

    def __init__(self):
        """
        RAM Usage metric.
        """

    def compute(self, t):

        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss  # in bytes

        print("Train Task {:} - MU: {:.3f} GB"
              .format(t, mem / (1024 * 1024 * 1024)))

        return mem / (1024 * 1024 * 1024)


class DiskUsage(object):

    def __init__(self, path_to_monitor=None, disk_io=False):
        """
        :param path_to_monitor (string): a valid path to folder.
                If None, the current working directory is used.
        :param disk_io: True to enable monitoring of I/O operations on disk.
                WARNING: Reports are system-wide, grouping all disks.
        """

        if path_to_monitor is not None:
            self.path_to_monitor = path_to_monitor
        else:
            self.path_to_monitor = os.getcwd()

        self.disk_io = disk_io

    def compute(self, t):
        """
        :param t: task id

        :return usage, io (tuple): io is None if disk_io is False
        """

        usage = psutil.disk_usage(self.path_to_monitor)

        total, used, free, percent = \
            bytes2human(usage.total), \
            bytes2human(usage.used), \
            bytes2human(usage.free), \
            usage.percent

        print("Disk usage for {:}".format(self.path_to_monitor))
        print("Task {:} - disk percent: {:}%, \
                disk total: {:}, \
                disk used: {:}, \
                disk free: {:}"
              .format(t, percent, total, used, free))

        if self.disk_io:
            io = psutil.disk_io_counters()
            read_count, write_count = \
                io.read_count, \
                io.write_count
            read_bytes, write_bytes = \
                bytes2human(io.read_bytes), \
                bytes2human(io.write_bytes)

            print("Task {:} - read count: {:}, \
                write count: {:}, \
                bytes read: {:}, \
                bytes written: {:}"
                  .format(t, read_count, write_count, read_bytes, write_bytes))
        else:
            io = None

        return usage, io


class CM(object):

    def __init__(self, num_class=None):
        """
        Confusion Matrix computation
        """
        self.num_class = num_class

    def compute(self, y, y_hat, normalize=False):
        """
        :param y (tensor or tensors list): true labels for each minibatch
        :param y_hat (tensor or tensors list): predicted labels for each
            minibatch
        """

        assert type(y) == type(y_hat), "Predicted and target labels must be \
                both list (of tensors) or tensors"
        
        # manage list of tensors by default
        if not (isinstance(y, list) or isinstance(y, tuple)):
            y = [y]
            y_hat = [y_hat]

        if self.num_class is None:
            num_class = int(max([torch.max(el).item() + 1 for el in y]))    
        else:
            num_class = self.num_class

        cmap = plt.cm.Blues

        cm = np.zeros((num_class, num_class))
        for i, (el, el_hat) in enumerate(zip(y, y_hat)):
            # Compute confusion matrix
            cm += confusion_matrix(
                el.numpy(), el_hat.numpy(), 
                labels=list(range(num_class)))                

        # Only use the labels that appear in the data
        classes = [str(i) for i in range(num_class)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots()
        im = ax.matshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=None,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='jpg', dpi=50)
        plt.close(fig)

        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        return image


class TimeUsage:

    """
        Time usage metric measured in seconds.
    """

    def __init__(self):
        self._start_time = time.perf_counter()

    def compute(self, t):
        elapsed_time = time.perf_counter() - self._start_time
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
