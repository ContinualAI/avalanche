################################################################################
# Copyright (c) 2022 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 18-02-2022                                                             #
# Author: Lorenzo Pellegrini                                                   #
#                                                                              #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" LVIS PyTorch Object Detection Dataset """

from pathlib import Path
import dill
from typing import Optional, Union, List, Sequence, TypedDict

import torch
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
    DownloadableDataset,
    default_dataset_location,
)
from avalanche.benchmarks.datasets.lvis_dataset.lvis_data import lvis_archives
from avalanche.checkpointing import constructor_based_serialization

try:
    from lvis import LVIS
except ImportError:
    raise ModuleNotFoundError(
        "LVIS not found, if you want to use detection "
        "please install avalanche with the detection "
        "dependencies: "
        "pip install avalanche-lib[detection]"
    )


class LvisDataset(DownloadableDataset):
    """LVIS PyTorch Object Detection Dataset"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        train=True,
        transform=None,
        loader=default_loader,
        download=True,
        lvis_api: Optional[LVIS] = None,
        img_ids: Optional[List[int]] = None,
    ):
        """
        Creates an instance of the LVIS dataset.

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            "lvis" will be used.
        :param train: If True, the training set will be returned. If False,
            the test set will be returned.
        :param transform: The transformation to apply to (img, annotations)
            values.
        :param loader: The image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        :param lvis_api: An instance of the LVIS class (from the lvis-api) to
            use. Defaults to None, which means that annotations will be loaded
            from the annotation json found in the root directory.
        :param img_ids: A list representing a subset of images to use. Defaults
            to None, which means that the dataset will contain all images
            in the LVIS dataset.
        """

        if root is None:
            root = default_dataset_location("lvis")

        self.train = train  # training set or test set
        self.transform = transform
        self.loader = loader
        self.bbox_crop = True
        self.img_ids: List[int] = img_ids  # type: ignore

        self.targets: LVISDetectionTargets = None  # type: ignore
        self.lvis_api: LVIS = lvis_api  # type: ignore

        super(LvisDataset, self).__init__(root, download=download, verbose=True)

        self._load_dataset()

    def _download_dataset(self) -> None:
        data2download = lvis_archives

        for name, url, checksum in data2download:
            if self.verbose:
                print("Downloading " + name + "...")

            result_file = self._download_file(url, name, checksum)
            if self.verbose:
                print("Download completed. Extracting...")

            self._extract_archive(result_file)
            if self.verbose:
                print("Extraction completed!")

    def _load_metadata(self) -> bool:
        must_load_api = self.lvis_api is None
        must_load_img_ids = self.img_ids is None
        try:
            # Load metadata
            if must_load_api:
                if self.train:
                    ann_json_path = str(self.root / "lvis_v1_train.json")
                else:
                    ann_json_path = str(self.root / "lvis_v1_val.json")

                self.lvis_api = LVIS(ann_json_path)

            lvis_api = self.lvis_api
            if must_load_img_ids:
                self.img_ids = list(sorted(lvis_api.get_img_ids()))

            self.targets = LVISDetectionTargets(lvis_api, self.img_ids)

            # Try loading an image
            if len(self.img_ids) > 0:
                img_id = self.img_ids[0]
                img_dict: LVISImgEntry = self.lvis_api.load_imgs(ids=[img_id])[0]
                assert self._load_img(img_dict) is not None
        except BaseException:
            if must_load_api:
                self.lvis_api = None  # type: ignore
            if must_load_img_ids:
                self.img_ids = None  # type: ignore

            self.targets = None  # type: ignore
            raise

        return True

    def _download_error_message(self) -> str:
        return (
            "[LVIS] Error downloading the dataset. Consider "
            "downloading it manually at: https://www.lvisdataset.org/dataset"
            " and placing it in: " + str(self.root)
        )

    def __getitem__(self, index):
        """
        Loads an instance given its index.

        :param index: The index of the instance to retrieve.

        :return: a (sample, target) tuple where the target is a
            torchvision-style annotation for object detection
            https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        """
        img_id = self.img_ids[index]
        img_dict: LVISImgEntry = self.lvis_api.load_imgs(ids=[img_id])[0]
        annotation_dicts: LVISImgTargets = self.targets[index]

        # Transform from LVIS dictionary to torchvision-style target
        num_objs = annotation_dicts["bbox"].shape[0]

        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = annotation_dicts["bbox"][i][0]
            ymin = annotation_dicts["bbox"][i][1]
            xmax = xmin + annotation_dicts["bbox"][i][2]
            ymax = ymin + annotation_dicts["bbox"][i][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation_dicts["category_id"][i])

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(annotation_dicts["area"][i])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        img = self._load_img(img_dict)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def _load_img(self, img_dict: "LVISImgEntry"):
        coco_url = img_dict["coco_url"]
        splitted_url = coco_url.split("/")
        img_path = splitted_url[-2] + "/" + splitted_url[-1]
        final_path = self.root / img_path  # <root>/train2017/<img_id>.jpg
        return self.loader(str(final_path))


@dill.register(LvisDataset)
def checkpoint_LvisDataset(pickler, obj: LvisDataset):
    constructor_based_serialization(
        pickler,
        obj,
        LvisDataset,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            train=obj.train,
            transform=obj.transform,
            loader=obj.loader,
            lvis_api=obj.lvis_api,
            img_ids=obj.img_ids,
        ),
    )


class LVISImgEntry(TypedDict):
    id: int
    date_captured: str
    neg_category_ids: List[int]
    license: int
    height: int
    width: int
    flickr_url: str
    coco_url: str
    not_exhaustive_category_ids: List[int]


class LVISAnnotationEntry(TypedDict):
    id: int
    area: float
    segmentation: List[List[float]]
    image_id: int
    bbox: List[int]
    category_id: int


class LVISImgTargets(TypedDict):
    id: torch.Tensor
    area: torch.Tensor
    segmentation: List[List[List[float]]]
    image_id: torch.Tensor
    bbox: torch.Tensor
    category_id: torch.Tensor
    labels: torch.Tensor


class LVISDetectionTargets(Sequence[List[LVISImgTargets]]):
    def __init__(self, lvis_api: LVIS, img_ids: Optional[List[int]] = None):
        super(LVISDetectionTargets, self).__init__()
        self.lvis_api = lvis_api
        if img_ids is None:
            img_ids = list(sorted(lvis_api.get_img_ids()))

        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        annotation_ids = self.lvis_api.get_ann_ids(img_ids=[img_id])
        annotation_dicts: List[LVISAnnotationEntry] = self.lvis_api.load_anns(
            annotation_ids
        )

        n_annotations = len(annotation_dicts)

        category_tensor = torch.empty((n_annotations,), dtype=torch.long)
        target_dict: LVISImgTargets = {
            "bbox": torch.empty((n_annotations, 4), dtype=torch.float32),
            "category_id": category_tensor,
            "id": torch.empty((n_annotations,), dtype=torch.long),
            "area": torch.empty((n_annotations,), dtype=torch.float32),
            "image_id": torch.full((1,), img_id, dtype=torch.long),
            "segmentation": [],
            "labels": category_tensor,  # Alias of category_id
        }

        for ann_idx, annotation in enumerate(annotation_dicts):
            target_dict["bbox"][ann_idx] = torch.as_tensor(annotation["bbox"])
            target_dict["category_id"][ann_idx] = annotation["category_id"]
            target_dict["id"][ann_idx] = annotation["id"]
            target_dict["area"][ann_idx] = annotation["area"]
            target_dict["segmentation"].append(annotation["segmentation"])

        return target_dict


def _test_to_tensor(a, b):
    return ToTensor()(a), b


def _detection_collate_fn(batch):
    return tuple(zip(*batch))


def _plot_detection_sample(img: Image.Image, target):
    from matplotlib import patches
    import matplotlib.pyplot as plt

    plt.gca().imshow(img)
    for box in target["boxes"]:
        box = box.tolist()

        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rect)


if __name__ == "__main__":
    # this little example script can be used to visualize the first image
    # loaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = LvisDataset(transform=_test_to_tensor)
    test_data = LvisDataset(transform=_test_to_tensor, train=False)
    print("train size: ", len(train_data))
    print("Test size: ", len(test_data))
    dataloader = DataLoader(train_data, batch_size=1, collate_fn=_detection_collate_fn)

    n_to_show = 5
    for instance_idx, batch_data in enumerate(dataloader):
        x, y = batch_data
        x = x[0]
        y = y[0]
        _plot_detection_sample(transforms.ToPILImage()(x), y)
        plt.show()
        print("X image shape", x.shape)
        print("N annotations:", len(y["boxes"]))
        if (instance_idx + 1) >= n_to_show:
            break

__all__ = [
    "LvisDataset",
    "LVISImgEntry",
    "LVISAnnotationEntry",
    "LVISImgTargets",
    "LVISDetectionTargets",
]
