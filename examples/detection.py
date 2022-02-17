################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-02-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to run detection benchmarks.
"""

import logging
# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
logging.basicConfig(level=logging.NOTSET)

import matplotlib

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset
from examples.tvdetection.engine import train_one_epoch, evaluate

matplotlib.use('Agg')

import argparse
from pathlib import Path
from typing import Union, List, Sequence, TypeVar, Callable

import matplotlib.pyplot as plt
import torch
from PIL import Image
from lvis_api import LVIS
from matplotlib import patches
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from avalanche.benchmarks import GenericScenarioStream, Experience, \
    TScenario, TScenarioStream, GenericCLScenario, StreamUserDef, \
    TStreamsUserDict, GenericExperience
from typing import TypedDict
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    # TODO: implement support for multi-parameter transforms in AvalancheDataset
    train_transform = ToTensor()
    test_transform = ToTensor()
    # ---------

    # --- SCENARIO CREATION
    torch.random.manual_seed(1234)
    n_exps = 100  # Keep it high to run a short exp
    # Dataset download at: https://www.lvisdataset.org/dataset
    benchmark = split_lvis(
        '/ssd1/datasets/coco',
        '/ssd1/datasets/coco/lvis_v1_train.json',
        '/ssd1/datasets/coco/lvis_v1_val.json',
        n_exps,
        train_transform=train_transform,
        eval_transform=test_transform)

    # ---------

    # MODEL CREATION
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # Just tune the box predictor
    for p in model.parameters():
        p.requires_grad = False

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = benchmark.n_classes + 1  # N classes + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    # TODO: integrate into a strategy
    # cl_strategy = Naive(
    #     model,
    #     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #     CrossEntropyLoss(),
    #     train_mb_size=100,
    #     train_epochs=4,
    #     eval_mb_size=100,
    #     device=device,
    # )

    # TRAINING LOOP
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)

        print('Dataset contains', len(experience.dataset), 'instances')

        params = [p for p in model.parameters() if p.requires_grad]

        print('Learnable parameters:')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)

        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        if Path('model_checkpoint.pth').exists():
            checkpoint = torch.load('model_checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.to(device)
        else:
            data_loader = DataLoader(
                experience.dataset, batch_size=5, shuffle=True, drop_last=True,
                num_workers=4,
                collate_fn=detection_collate_fn
            )

            for epoch in range(1):
                train_one_epoch(model, optimizer, data_loader, device, epoch,
                                print_freq=10)
            # cl_strategy.train(experience)

            if not Path('model_checkpoint.pth').exists():
                torch.save({
                    'epoch': 0,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'model_checkpoint.pth')
        print("Training completed")

        # Just run the eval on a small set (otherwise it takes ages to complete)
        mini_test = AvalancheSubset(benchmark.test_stream[0].dataset,
                                    indices=list(range(1000)))
        data_loader = DataLoader(
            mini_test, batch_size=5, shuffle=False, drop_last=False,
            num_workers=4,
            collate_fn=detection_collate_fn
        )

        print("Computing accuracy on the whole test set")

        # TODO: integrate metrics
        # results.append(cl_strategy.eval(scenario.test_stream))
        evaluate(model, data_loader, device=device)
        break


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


class LVISDetectionTargets(Sequence[List[LVISAnnotationEntry]]):
    def __init__(
            self,
            lvis_api: LVIS,
            img_ids: List[int] = None):
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
        annotation_dicts: List[LVISAnnotationEntry] = \
            self.lvis_api.load_anns(annotation_ids)
        return annotation_dicts


class LvisDataset(Dataset):
    def __init__(
            self,
            lvis_api: LVIS,
            rel_path: Union[str, Path],
            img_ids: List[int] = None,
            transforms=None):
        super(LvisDataset, self).__init__()
        self.lvis_api = lvis_api
        self.rel_path = Path(rel_path)
        if img_ids is None:
            img_ids = list(sorted(lvis_api.get_img_ids()))

        self.img_ids = img_ids
        self.transforms = transforms
        self.targets = LVISDetectionTargets(self.lvis_api, self.img_ids)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_dict: LVISImgEntry = self.lvis_api.load_imgs(ids=[img_id])[0]
        coco_url = img_dict['coco_url']
        splitted_url = coco_url.split('/')
        img_path = splitted_url[-2] + '/' + splitted_url[-1]
        final_path = self.rel_path / img_path
        annotation_dicts = self.targets[index]

        num_objs = len(annotation_dicts)

        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = annotation_dicts[i]['bbox'][0]
            ymin = annotation_dicts[i]['bbox'][1]
            xmax = xmin + annotation_dicts[i]['bbox'][2]
            ymax = ymin + annotation_dicts[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation_dicts[i]['category_id'])

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(annotation_dicts[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        img = Image.open(str(final_path)).convert("RGB")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))


def split_lvis(imgs_path, train_json_path, val_json_path, n_experiences,
               train_transform=None, eval_transform=None,
               shuffle=True):
    train_api = LVIS(train_json_path)
    val_api = LVIS(val_json_path)

    transform_groups = dict(
        train=(train_transform, None),
        eval=(eval_transform, None),
    )

    all_cat_ids = set(train_api.get_cat_ids())
    all_cat_ids.union(val_api.get_cat_ids())

    train_dataset = LvisDataset(train_api, imgs_path, img_ids=None)

    val_dataset = LvisDataset(val_api, imgs_path, img_ids=None)

    exp_n_imgs = len(train_dataset) // n_experiences
    remaining = len(train_dataset) % n_experiences

    train_dataset_avl = AvalancheDataset(
        train_dataset,
        transform_groups=transform_groups,
        initial_transform_group='train')
    val_dataset_avl = AvalancheDataset(
        val_dataset,
        transform_groups=transform_groups,
        initial_transform_group='eval')

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
        idx_range = train_indices[last_slice_idx:last_slice_idx+n_imgs]
        train_exps_datasets.append(
            AvalancheSubset(
                train_dataset_avl,
                indices=idx_range))
        last_slice_idx += n_imgs

    train_def = StreamUserDef(
        exps_data=train_exps_datasets,
        exps_task_labels=[0 for _ in range(len(train_exps_datasets))],
        origin_dataset=train_dataset,
        is_lazy=False
    )

    val_def = StreamUserDef(
        exps_data=[val_dataset_avl],
        exps_task_labels=[0],
        origin_dataset=val_dataset,
        is_lazy=False
    )

    return DetectionCLScenario(
        n_classes=len(all_cat_ids),
        stream_definitions={
            'train': train_def,
            'test': val_def
        },
        complete_test_set_only=True,
        experience_factory=det_exp_factory
    )


def det_exp_factory(stream: GenericScenarioStream, exp_id: int):
    return DetectionExperience(stream, exp_id)


TDetectionExperience = TypeVar("TDetectionExperience",
                               bound=GenericExperience)


class DetectionExperience(
    Experience[TScenario, TScenarioStream]
):
    def __init__(
        self: TDetectionExperience,
        origin_stream: TScenarioStream,
        current_experience: int,
    ):
        self.origin_stream: TScenarioStream = origin_stream
        self.benchmark: TScenario = origin_stream.benchmark
        self.current_experience: int = current_experience

        self.dataset: AvalancheDataset = (
            origin_stream.benchmark.stream_definitions[
                origin_stream.name
            ].exps_data[current_experience]
        )

    def _get_stream_def(self):
        return self.benchmark.stream_definitions[self.origin_stream.name]

    @property
    def task_labels(self) -> List[int]:
        stream_def = self._get_stream_def()
        return list(stream_def.exps_task_labels[self.current_experience])

    @property
    def task_label(self) -> int:
        if len(self.task_labels) != 1:
            raise ValueError(
                "The task_label property can only be accessed "
                "when the experience contains a single task label"
            )

        return self.task_labels[0]


class DetectionCLScenario(GenericCLScenario[TDetectionExperience]):
    def __init__(
            self,
            n_classes: int,
            *,
            stream_definitions: TStreamsUserDict,
            complete_test_set_only: bool = False,
            experience_factory: Callable[
                ["GenericScenarioStream", int], TDetectionExperience
            ] = None):
        if experience_factory is None:
            experience_factory = DetectionExperience

        super(DetectionCLScenario, self).__init__(
            stream_definitions=stream_definitions,
            complete_test_set_only=complete_test_set_only,
            experience_factory=experience_factory
        )

        self.n_classes = n_classes


def plot_sample(img: Image.Image, target):
    plt.gca().imshow(img)
    for box in target['boxes']:
        box = box.tolist()

        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none')
        plt.gca().add_patch(rect)

    plt.savefig('my_img.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
