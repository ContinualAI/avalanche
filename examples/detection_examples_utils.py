import torch

from avalanche.benchmarks import StreamUserDef
from avalanche.benchmarks.scenarios.detection_scenario import (
    DetectionScenario,
)

from avalanche.benchmarks.utils.collate_functions import detection_collate_fn
from avalanche.benchmarks.utils.detection_dataset import (
    detection_subset,
    make_detection_dataset,
)


def split_detection_benchmark(
    n_experiences: int,
    train_dataset,
    test_dataset,
    n_classes: int,
    train_transform=None,
    eval_transform=None,
    shuffle=True,
):
    """
    Creates an example object detection/segmentation benchmark.

    This is a generator for toy benchmarks and should be used only to
    show how a detection benchmark can be created. It was not meant to be
    used for research purposes!

    :param n_experiences: The number of train experiences to create.
    :param train_dataset: The training dataset.
    :param test_dataset: The test dataset.
    :param n_classes: The number of categories (excluding the background).
    :param train_transform: The train transformation.
    :param eval_transform: The eval transformation.
    :param shuffle: If True, the dataset will be split randomly
    :return: A :class:`DetectionScenario` instance.
    """

    transform_groups = dict(
        train=(train_transform, None),
        eval=(eval_transform, None),
    )

    exp_n_imgs = len(train_dataset) // n_experiences
    remaining = len(train_dataset) % n_experiences

    # Note: in future versions of Avalanche, the make_classification_dataset
    # function will be replaced with a more specific function for object
    # detection datasets.
    train_dataset_avl = make_detection_dataset(
        train_dataset,
        transform_groups=transform_groups,
        initial_transform_group="train",
        collate_fn=detection_collate_fn,
    )
    test_dataset_avl = make_detection_dataset(
        test_dataset,
        transform_groups=transform_groups,
        initial_transform_group="eval",
        collate_fn=detection_collate_fn,
    )

    exp_sz = [exp_n_imgs for _ in range(n_experiences)]
    for exp_id in range(n_experiences):
        if remaining == 0:
            break

        exp_sz[exp_id] += 1
        remaining -= 1

    train_indices = [i for i in range(len(train_dataset_avl))]
    if shuffle:
        train_indices = torch.as_tensor(train_indices)[
            torch.randperm(len(train_indices))
        ].tolist()

    train_exps_datasets = []
    last_slice_idx = 0
    for exp_id in range(n_experiences):
        n_imgs = exp_sz[exp_id]
        idx_range = train_indices[last_slice_idx : last_slice_idx + n_imgs]
        train_exps_datasets.append(
            detection_subset(train_dataset_avl, indices=idx_range)
        )
        last_slice_idx += n_imgs

    train_def = StreamUserDef(
        exps_data=train_exps_datasets,
        exps_task_labels=[0 for _ in range(len(train_exps_datasets))],
        origin_dataset=train_dataset,
        is_lazy=False,
    )

    test_def = StreamUserDef(
        exps_data=[test_dataset_avl],
        exps_task_labels=[0],
        origin_dataset=test_dataset,
        is_lazy=False,
    )

    return DetectionScenario(
        n_classes=n_classes,
        stream_definitions={"train": train_def, "test": test_def},
        complete_test_set_only=True,
    )


__all__ = ["split_detection_benchmark"]
