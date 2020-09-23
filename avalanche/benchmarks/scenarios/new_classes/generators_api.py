################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Sequence, Optional, Dict, List, Union, Any

from .scenario_creation import create_nc_single_dataset_sit_scenario, \
    create_nc_single_dataset_multi_task_scenario, \
    create_nc_multi_dataset_sit_scenario, \
    create_nc_multi_dataset_multi_task_scenario

from .nc_scenario import NCMultiTaskScenario, NCSingleTaskScenario

from avalanche.training.utils import IDatasetWithTargets


def NCScenario(
        train_dataset: Sequence[IDatasetWithTargets],
        test_dataset: Sequence[IDatasetWithTargets],
        n_steps: int,
        multi_task=True,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        per_step_classes: Optional[Dict[int, int]] = None,
        classes_ids_from_zero_in_each_step: bool = True,
        remap_class_ids: bool = False,
        per_batch_classes: Optional[Dict[int, int]] = None,
        one_dataset_per_batch: bool = False,
        reproducibility_data: Optional[Dict[str, Any]] = None) \
        -> Union[NCMultiTaskScenario, NCSingleTaskScenario]:

    """

    :param train_dataset:
    :param test_dataset:
    :param n_steps: no for MT Multi dataset
    :param multi_task:
    :param shuffle:
    :param seed:
    :param fixed_class_order: Only valid for Scenario built from single a dataset.
    :param per_step_classes:
    :param classes_ids_from_zero_in_each_step: valid only if multi-task is true.
    :param remap_class_ids: works only for single dataset SIT scenarios.
    :param per_batch_classes: not working for MT multi dataset.
    :param one_dataset_per_batch: only for Multi dataset SIT.
    :param reproducibility_data:
    :return:
    """

    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        # we are in multi-datasets setting

        if multi_task:
            scenario = create_nc_multi_dataset_multi_task_scenario(
                train_dataset_list=train_dataset,
                test_dataset_list=test_dataset,
                shuffle=shuffle, seed=seed,
                classes_ids_from_zero_in_each_task=
                classes_ids_from_zero_in_each_step,
                reproducibility_data=reproducibility_data
            )
        else:
            scenario = create_nc_multi_dataset_sit_scenario(
                train_dataset_list=train_dataset,
                test_dataset_list=test_dataset,
                n_batches=n_steps, shuffle=shuffle,
                seed=seed, per_batch_classes=per_batch_classes,
                one_dataset_per_batch=one_dataset_per_batch,
                reproducibility_data=reproducibility_data
            )

    else:
        # we are working with a single input dataset

        if multi_task:
            scenario = create_nc_single_dataset_multi_task_scenario(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_tasks=n_steps,
                seed=seed,
                fixed_class_order=fixed_class_order,
                per_task_classes=per_step_classes,
                classes_ids_from_zero_in_each_task=
                classes_ids_from_zero_in_each_step,
                reproducibility_data=reproducibility_data
            )
        else:
            scenario = create_nc_single_dataset_sit_scenario(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_batches=n_steps, shuffle=shuffle,
                seed=seed, fixed_class_order=fixed_class_order,
                per_batch_classes=per_step_classes,
                remap_class_ids=remap_class_ids,
                reproducibility_data=reproducibility_data
            )

    return scenario
