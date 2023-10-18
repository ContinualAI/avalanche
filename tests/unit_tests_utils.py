import copy
import itertools
from os.path import expanduser

import os
import random
import torch
from PIL.Image import Image
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import _make_taskaware_tensor_classification_dataset
from avalanche.benchmarks.utils.detection_dataset import (
    make_detection_dataset,
)


# Environment variable used to skip some expensive tests that are very unlikely
# to break unless you touch their code directly (e.g. datasets).
FAST_TEST = False
if "FAST_TEST" in os.environ:
    FAST_TEST = os.environ["FAST_TEST"].lower() == "true"

# Environment variable used to update the metric pickles providing the ground
# truth for metric tests. If you change the metrics (names, x values, y
# values, ...) you may need to update them.
UPDATE_METRICS = False
if "UPDATE_METRICS" in os.environ:
    UPDATE_METRICS = os.environ["UPDATE_METRICS"].lower() == "true"

# print(f"UPDATE_METRICS: {UPDATE_METRICS}")


def is_github_action():
    """Check whether we are running in a Github action.

    We want to avoid some expensive operations (such as downloading data)
    inside the CI pipeline.
    """
    return "GITHUB_ACTION" in os.environ


def common_setups():
    # adapt_dataset_urls()
    pass


def load_benchmark(use_task_labels=False, fast_test=True):
    """
    Returns a NC Benchmark from a fake dataset of 10 classes, 5 experiences,
    2 classes per experience.
    """
    if fast_test:
        my_nc_benchmark = get_fast_benchmark(use_task_labels)
    else:
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
            transform=Compose([ToTensor()]),
        )

        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
            transform=Compose([ToTensor()]),
        )
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=use_task_labels, seed=1234
        )

    return my_nc_benchmark


def load_image_data():
    mnist_train = MNIST(
        root=default_dataset_location("mnist"),
        train=True,
        download=True,
        transform=Compose([ToTensor()]),
    )
    mnist_test = MNIST(
        root=default_dataset_location("mnist"),
        train=False,
        download=True,
        transform=Compose([ToTensor()]),
    )
    return mnist_train, mnist_test


image_data = None


def dummy_image_dataset():
    """Returns a PyTorch image dataset of 10 classes."""
    global image_data

    if image_data is None:
        image_data = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
        )
    return image_data


def dummy_tensor_dataset():
    """Returns a PyTorch image dataset of 10 classes."""
    x = torch.rand(32, 10)
    y = torch.rand(32, 10)
    return TensorDataset(x, y)


def get_fast_benchmark(
    use_task_labels=False,
    shuffle=True,
    n_samples_per_class=100,
    n_classes=10,
    n_features=6,
    seed=None,
    train_transform=None,
    eval_transform=None,
):
    train, test = dummy_classification_datasets(
        n_classes, n_features, n_samples_per_class, seed
    )
    my_nc_benchmark = nc_benchmark(
        train,
        test,
        5,
        task_labels=use_task_labels,
        shuffle=shuffle,
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=seed,
    )
    return my_nc_benchmark


def dummy_classification_datasets(
    n_classes=10, n_features=7, n_samples_per_class=20, seed=42
):
    dataset = make_classification(
        n_samples=n_classes * n_samples_per_class,
        n_classes=n_classes,
        n_features=n_features,
        n_informative=6,
        n_redundant=0,
        random_state=seed,
    )
    X = torch.from_numpy(dataset[0]).float()
    y = torch.from_numpy(dataset[1]).long()
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.6, shuffle=True, stratify=y, random_state=seed
    )
    train = _make_taskaware_tensor_classification_dataset(train_X, train_y)
    test = _make_taskaware_tensor_classification_dataset(test_X, test_y)
    return train, test


class DummyImageDataset(Dataset):
    def __init__(self, n_elements=10000, n_classes=100):
        assert n_elements >= n_classes

        super().__init__()
        self.targets = list(range(n_classes))
        self.targets += [
            random.randint(0, n_classes - 1) for _ in range(n_elements - n_classes)
        ]

    def __getitem__(self, index):
        return (
            Image(),
            self.targets[index],
        )

    def __len__(self):
        return len(self.targets)


def load_experience_train_eval(experience, batch_size=32, num_workers=0):
    for x, y, t in DataLoader(
        experience.dataset.train(),
        batch_size=batch_size,
        num_workers=num_workers,
    ):
        break

    for x, y, t in DataLoader(
        experience.dataset.eval(),
        batch_size=batch_size,
        num_workers=num_workers,
    ):
        break


def get_device():
    if "USE_GPU" in os.environ:
        use_gpu = os.environ["USE_GPU"].lower() in ["true"]
    else:
        use_gpu = False
    print("Test on GPU:", use_gpu)
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return device


def set_deterministic_run(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class _DummyDetectionDataset:
    """
    A dataset that makes a defensive copy of the
    targets before returning them.

    Alas, many detection transformations, including the
    ones in the torchvision repository, modify bounding boxes
    (and other elements) in place.
    Luckly, images seem to be never modified in place.
    """

    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], copy.deepcopy(self.targets[index])


def get_fast_detection_datasets(
    n_images=30,
    max_elements_per_image=10,
    n_samples_per_class=20,
    n_classes=10,
    seed=None,
    image_size=64,
    n_test_images=5,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    assert n_images * max_elements_per_image >= n_samples_per_class * n_classes
    assert n_test_images < n_images
    assert n_test_images > 0

    base_n_per_images = (n_samples_per_class * n_classes) // n_images
    additional_elements = (n_samples_per_class * n_classes) % n_images
    to_allocate = np.full(n_images, base_n_per_images)
    to_allocate[:additional_elements] += 1
    np.random.shuffle(to_allocate)
    classes_elements = np.repeat(np.arange(n_classes), n_samples_per_class)
    np.random.shuffle(classes_elements)

    import matplotlib.colors as mcolors

    forms = ["ellipse", "rectangle", "line", "arc"]
    colors = list(mcolors.TABLEAU_COLORS.values())
    combs = list(itertools.product(forms, colors))
    random.shuffle(combs)

    generated_images = []
    generated_targets = []
    for img_idx in range(n_images):
        n_to_allocate = to_allocate[img_idx]
        base_alloc_idx = to_allocate[:img_idx].sum()
        classes_to_instantiate = classes_elements[
            base_alloc_idx : base_alloc_idx + n_to_allocate
        ]

        _, _, clusters = make_blobs(
            n_to_allocate,
            n_features=2,
            centers=n_to_allocate,
            center_box=(0, image_size - 1),
            random_state=seed,
            return_centers=True,
        )

        from PIL import Image as ImageApi
        from PIL import ImageDraw

        im = ImageApi.new("RGB", (image_size, image_size))
        draw = ImageDraw.Draw(im)

        target = {
            "boxes": torch.zeros((n_to_allocate, 4), dtype=torch.float32),
            "labels": torch.zeros((n_to_allocate,), dtype=torch.long),
            "image_id": torch.full((1,), img_idx, dtype=torch.long),
            "area": torch.zeros((n_to_allocate,), dtype=torch.float32),
            "iscrowd": torch.zeros((n_to_allocate,), dtype=torch.long),
        }

        obj_sizes = np.random.uniform(
            low=image_size * 0.1 * 0.95,
            high=image_size * 0.1 * 1.05,
            size=(n_to_allocate,),
        )
        for center_idx, center in enumerate(clusters):
            obj_size = float(obj_sizes[center_idx])
            class_to_gen = classes_to_instantiate[center_idx]

            class_form, class_color = combs[class_to_gen]

            left = center[0] - obj_size
            top = center[1] - obj_size
            right = center[0] + obj_size
            bottom = center[1] + obj_size
            ltrb = (left, top, right, bottom)
            if class_form == "ellipse":
                draw.ellipse(ltrb, fill=class_color)
            elif class_form == "rectangle":
                draw.rectangle(ltrb, fill=class_color)
            elif class_form == "line":
                draw.line(ltrb, fill=class_color, width=max(1, int(obj_size * 0.25)))
            elif class_form == "arc":
                draw.arc(ltrb, fill=class_color, start=45, end=200)
            else:
                raise RuntimeError("Unsupported form")

            target["boxes"][center_idx] = torch.as_tensor(ltrb)
            target["labels"][center_idx] = class_to_gen
            target["area"][center_idx] = obj_size**2

        generated_images.append(np.array(im))
        generated_targets.append(target)
        im.close()

    test_indices = set(
        np.random.choice(n_images, n_test_images, replace=False).tolist()
    )
    train_images = [x for i, x in enumerate(generated_images) if i not in test_indices]
    test_images = [x for i, x in enumerate(generated_images) if i in test_indices]

    train_targets = [
        x for i, x in enumerate(generated_targets) if i not in test_indices
    ]
    test_targets = [x for i, x in enumerate(generated_targets) if i in test_indices]

    return make_detection_dataset(
        _DummyDetectionDataset(train_images, train_targets),
        targets=train_targets,
        task_labels=0,
    ), make_detection_dataset(
        _DummyDetectionDataset(test_images, test_targets),
        targets=test_targets,
        task_labels=0,
    )


__all__ = [
    "common_setups",
    "load_benchmark",
    "get_fast_benchmark",
    "load_experience_train_eval",
    "get_device",
    "set_deterministic_run",
    "get_fast_detection_datasets",
]
