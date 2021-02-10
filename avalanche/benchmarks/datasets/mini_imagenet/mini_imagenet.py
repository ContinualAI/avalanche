# The dataset has been adapted from:
# https://github.com/yaoyao-liu/mini-imagenet-tools
# with the following license:
################################################################################
# MIT License
#
# Copyright (c) 2019 Yaoyao Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

# For the Avalanche adaptation, see the accompanying LICENSE file.
################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 10-02-2020                                                             #
# Author: Lorenzo Pellegrini                                                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

import csv
import glob
from pathlib import Path
from typing import Union, List, Tuple
from typing_extensions import Literal

import PIL
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Resize


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MiniImageNetDataset(Dataset):
    def __init__(self, imagenet_path: Union[str, Path],
                 split: Literal['train', 'val', 'test'] = 'train',
                 resize_to: Union[int, Tuple[int, int]] = 84):
        self.imagenet_path = MiniImageNetDataset.get_train_path(imagenet_path)
        self.split = split
        self.resize_to = resize_to
        self.image_paths: List[str] = []
        self.targets: List[int] = []

        if not self.imagenet_path.exists():
            raise ValueError('The provided directory does not exist.')

        if self.split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split. Valid values are: "train", "val", '
                             '"test"')

        self.prepare_dataset()
        super().__init__()

    @staticmethod
    def get_train_path(root_path: Union[str, Path]):
        root_path = Path(root_path)
        if (root_path / 'train').exists():
            return root_path / 'train'
        return root_path

    def prepare_dataset(self):
        # Read the CSV containing the file list for this split
        filename = './csv_files/' + self.split + '.csv'
        images = dict()
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader, None)  # Skip header

            for row in csv_reader:
                if row[1] in images.keys():
                    images[row[1]].append(row[0])
                else:
                    images[row[1]] = [row[0]]

        label_map = dict()
        for numerical_label, text_label in enumerate(sorted(images.keys())):
            label_map[text_label] = numerical_label

        for cls in images.keys():
            cls_numerical_label = label_map[cls]
            lst_files = []
            for file in glob.glob(str(self.imagenet_path / cls /
                                      ("*" + cls + "*"))):
                lst_files.append(file)

            lst_index = [int(i[i.rfind('_') + 1:i.rfind('.')]) for i in
                         lst_files]
            index_sorted = sorted(range(len(lst_index)),
                                  key=lst_index.__getitem__)

            index_selected = [int(i[i.index('.') - 4:i.index('.')]) for
                              i in images[cls]]
            selected_images = np.array(index_sorted)[
                np.array(index_selected) - 1]
            for i in np.arange(len(selected_images)):
                self.image_paths.append(lst_files[selected_images[i]])
                self.targets.append(cls_numerical_label)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        img = pil_loader(self.image_paths[item])
        # TODO: the original loader from yaoyao-liu uses cv2.INTER_AREA
        img = Resize(self.resize_to, interpolation=PIL.Image.BILINEAR)(img)
        return img, self.targets[item]


__all__ = [
    'MiniImageNetDataset'
]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Creating training dataset')
    train_dataset = MiniImageNetDataset('/ssd2/datasets/imagenet',
                                        split='train')
    print('Creating validation dataset')
    val_dataset = MiniImageNetDataset('/ssd2/datasets/imagenet',
                                      split='val')
    print('Creating test dataset')
    test_dataset = MiniImageNetDataset('/ssd2/datasets/imagenet',
                                       split='test')

    print('Training patterns:', len(train_dataset))
    print('Validation patterns:', len(val_dataset))
    print('Test patterns:', len(test_dataset))

    for img, label in train_dataset:
        plt.imshow(img)
        plt.show()
        print(img)
        print(label)
        break

    for img, label in val_dataset:
        plt.imshow(img)
        plt.show()
        print(img)
        print(label)
        break

    for img, label in test_dataset:
        plt.imshow(img)
        plt.show()
        print(img)
        print(label)
        break
