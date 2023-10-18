################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module conveniently wraps TorchAudio Datasets for using a clean and
comprehensive Avalanche API."""
import os

try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
except ImportError:
    import warnings

    warnings.warn(
        "TorchAudio package is required to load SpeechCommands. "
        "You can install it as extra dependency with "
        "`pip install avalanche-lib[extra]`"
    )
    SPEECHCOMMANDS = object

from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.datasets import default_dataset_location
import torch


def speech_commands_collate(batch):
    tensors, targets, t_labels = [], [], []
    for waveform, label, rate, sid, uid, t_label in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]
        t_labels += [torch.tensor(t_label)]
    tensors = [item.t() for item in tensors]
    tensors_padded = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.0
    )

    if len(tensors_padded.size()) == 2:  # no MFCC, add feature dimension
        tensors_padded = tensors_padded.unsqueeze(-1)
    targets = torch.stack(targets)
    t_labels = torch.stack(t_labels)
    return [tensors_padded, targets, t_labels]


class SpeechCommandsData(SPEECHCOMMANDS):
    def __init__(self, root, url, download, subset, mfcc_preprocessing):
        os.makedirs(root, exist_ok=True)
        super().__init__(root=root, download=download, subset=subset, url=url)
        self.labels_names = [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]
        self.mfcc_preprocessing = mfcc_preprocessing

    def __getitem__(self, item):
        wave, rate, label, speaker_id, ut_number = super().__getitem__(item)
        label = self.labels_names.index(label)
        wave = wave.squeeze(0)  # (T,)
        if self.mfcc_preprocessing is not None:
            assert rate == self.mfcc_preprocessing.sample_rate
            # (T, MFCC)
            wave = self.mfcc_preprocessing(wave).permute(1, 0)
        return wave, label, rate, speaker_id, ut_number


def SpeechCommands(
    root=default_dataset_location("speech_commands"),
    url="speech_commands_v0.02",
    download=True,
    subset=None,
    mfcc_preprocessing=None,
):
    """
    root: dataset root location
    url: version name of the dataset
    download: automatically download the dataset, if not present
    subset: one of 'training', 'validation', 'testing'
    mfcc_preprocessing: an optional torchaudio.transforms.MFCC instance
        to preprocess each audio. Warning: this may slow down the execution
        since preprocessing is applied on-the-fly each time a sample is
        retrieved from the dataset.
    """
    dataset = SpeechCommandsData(
        root=root,
        download=download,
        subset=subset,
        url=url,
        mfcc_preprocessing=mfcc_preprocessing,
    )
    labels = [datapoint[1] for datapoint in dataset]
    return _make_taskaware_classification_dataset(
        dataset, collate_fn=speech_commands_collate, targets=labels
    )


__all__ = ["SpeechCommands"]
