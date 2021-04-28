"""
A set of internal utils used by nc and ni scenario.
"""
from avalanche.benchmarks.utils import AvalancheDataset


def train_eval_transforms(dataset_train, dataset_test):
    """
    Internal utility used to create the transform groups from a couple of
    train and test datasets.

    :param dataset_train: The training dataset.
    :param dataset_test: The test dataset.
    :return: The transformations groups.
    """

    if isinstance(dataset_train, AvalancheDataset):
        train_group = dataset_train.get_transforms('train')
    else:
        train_group = (
            getattr(dataset_train, 'transform', None),
            getattr(dataset_train, 'target_transform', None)
        )

    if isinstance(dataset_test, AvalancheDataset):
        eval_group = dataset_test.get_transforms('eval')
    else:
        eval_group = (
            getattr(dataset_test, 'transform', None),
            getattr(dataset_test, 'target_transform', None)
        )

    return dict(
        train=train_group,
        eval=eval_group
    )


__all__ = [
    'train_eval_transforms'
]
