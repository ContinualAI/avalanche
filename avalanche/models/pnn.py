import torch
import torch.nn.functional as F
from torch import nn

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import MultiTaskModule, DynamicModule
from avalanche.models import MultiHeadClassifier


class LinearAdapter(nn.Module):
    def __init__(self, in_features, out_features_per_column, num_prev_modules):
        super().__init__()
        # Eq. 1 - lateral connections
        # one layer for each previous column. Empty for the first task.
        self.lat_layers = nn.ModuleList([])
        for _ in range(num_prev_modules):
            m = nn.Linear(in_features, out_features_per_column)
            self.lat_layers.append(m)

    def forward(self, x):
        hs = []
        for ii, lat in enumerate(self.lat_layers):
            hs.append(lat(x[ii]))
        return sum(hs)


class PNNColumn(nn.Module):
    def __init__(self, in_features, out_features_per_column, num_prev_modules):
        """ Progressive Neural Network column.

        :param in_features:
        :param out_features_per_column:
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.num_prev_modules = num_prev_modules

        # Eq. 1 - input to hidden connections
        # one module for each task
        self.itoh = nn.Linear(in_features, out_features_per_column)
        self.adapter = LinearAdapter(in_features, out_features_per_column,
                                     num_prev_modules)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        hs = self.adapter(x)
        hs += self.itoh(x[-1])
        return hs


class PNNLayer(MultiTaskModule, DynamicModule):
    def __init__(self, in_features, out_features_per_column):
        """ Progressive Neural Network layer.

        :param in_features:
        :param out_features_per_column:
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column

        # convert from task label to module list order
        self.task_to_module_idx = {}
        self.columns = nn.ModuleList([
            PNNColumn(in_features, out_features_per_column, 0)])

    @property
    def num_columns(self):
        return len(self.columns)

    def train_adaptation(self, dataset: AvalancheDataset):
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]
        else:
            task_labels = set(task_labels)
        assert len(task_labels) == 1, \
            "PNN assumes a single task for each experience. Please use a " \
            "compatible benchmark."
        # extract task label from set
        task_label = next(iter(task_labels))
        assert task_label not in self.task_to_module_idx, \
            "A new experience is using a previously seen task label. This is " \
            "not compatible with PNN, which assumes different task labels for" \
            " each training experience."

        if len(self.task_to_module_idx) == 0:
            # we have already initialized the first column.
            # No need to call add_column here.
            self.task_to_module_idx[task_label] = 0
        else:
            self.task_to_module_idx[task_label] = self.num_columns
            self.add_column()

    def add_column(self):
        # Freeze old parameters
        for param in self.parameters():
            param.requires_grad = False
        self.columns.append(PNNColumn(self.in_features,
                                      self.out_features_per_column,
                                      self.num_columns))

    def forward_single_task(self, x, task_label):
        col_idx = self.task_to_module_idx[task_label]
        hs = []
        for ii in range(col_idx + 1):
            hs.append(self.columns[ii](x[:ii+1]))
        return hs


class PNN(MultiTaskModule):
    def __init__(self, num_layers=1, in_features=784,
                 hidden_features_per_column=100):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features_per_columns = hidden_features_per_column

        self.layers = nn.ModuleList()
        self.layers.append(PNNLayer(in_features, hidden_features_per_column))
        for _ in range(num_layers - 1):
            lay = PNNLayer(hidden_features_per_column,
                           hidden_features_per_column)
            self.layers.append(lay)
        self.classifier = MultiHeadClassifier(hidden_features_per_column)

    def forward_single_task(self, x, task_label):
        x = x.contiguous()
        x = x.view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns
        col_idx = self.layers[-1].task_to_module_idx[task_label]

        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay(x, task_label)]
        return self.classifier(x[col_idx], task_label)
