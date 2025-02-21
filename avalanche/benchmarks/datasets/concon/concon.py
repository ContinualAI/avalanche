from pathlib import Path
from typing import Union, Optional

from PIL import Image
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
    SimpleDownloadableDataset,
    default_dataset_location,
)


class ConConDataset(SimpleDownloadableDataset):
    """
    ConConDataset represents a continual learning task with two classes: positive and negative.
    All data instances are images based on the CLEVR framework. A ground truth rule can be used
    to determine the binary class affiliation of any image. The dataset is designed to be used
    in a continual learning setting with three sequential tasks, each confounded by a task-specific
    confounder. The challenge arises from the fact that task-specific confounders change across tasks.
    There are two dataset variants:

    - Disjoint: Task-specific confounders never appear in other tasks.
    - Strict: Task-specific confounders may appear in other tasks as random features in both positive
      and negative samples.
    - Unconfounded: No task-specific confounders.

    Reference: 
    Busch, Florian Peter, et al. "Where is the Truth? The Risk of Getting Confounded in a Continual World." 
    arXiv preprint arXiv:2402.06434 (2024).

    Args:
        variant (str): The variant of the dataset, must be one of 'strict', 'disjoint', 'unconfounded'.
        scenario (int): The scenario number, must be between 0 and 2.
        root (str or Path): The root directory where the dataset will be stored. If None, the default
            avalanche dataset location will be used.
        train (bool): If True, use the training set, otherwise use the test set.
        download (bool): If True, download the dataset.
        transform: A function/transform that takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for data augmentation.
    """
    
    urls = {
        "strict": "https://zenodo.org/records/10630482/files/case_strict_main.zip",
        "disjoint": "https://zenodo.org/records/10630482/files/case_disjoint_main.zip",
        "unconfounded": "https://zenodo.org/records/10630482/files/unconfounded.zip"
    }

    def __init__(self,
                 variant: str,
                 scenario: int,
                 root: Optional[Union[str, Path]] = None,
                 train: bool = True,
                 download: bool = True,
                 transform = None,
                 ):
        assert variant in ["strict", "disjoint", "unconfounded"], "Invalid variant, must be one of 'strict', 'disjoint', 'unconf'"
        assert scenario in range(
            0, 3), "Invalid scenario, must be between 0 and 2"
        assert variant != "unconfounded" or scenario == 0, "Unconfounded scenario only has one variant"

        if root is None:
            root = default_dataset_location("concon")
                 
        self.root = Path(root)
            
        url = self.urls[variant]
        
        super(ConConDataset, self).__init__(
            self.root, url, None, download=download, verbose=True
        )
        
        if variant == "strict":
            self.variant = "case_strict_main"
        elif variant == "disjoint":
            self.variant = "case_disjoint_main"
        else:
            self.variant = variant
                    
        self.scenario = scenario
        self.train = train
        self.transform = transform
        self._load_dataset()
                
    def _load_metadata(self) -> bool:
        root = self.root / self.variant
        
        if self.train:
            images_dir = root / "train"
        else:
            images_dir = root / "test"

        images_dir = images_dir / "images" / f"t{self.scenario}"

        self.image_paths = []
        self.targets = []

        for class_id, class_dir in enumerate(images_dir.iterdir()):
            for image_path in class_dir.iterdir():
                self.image_paths.append(image_path)
                self.targets.append(class_id)
                
        return True

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        target = self.targets[idx]
        return image, target
    
    
if __name__ == "__main__":
    # this little example script can be used to visualize the first image
    # loaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = ConConDataset("strict", 0, "data_debug/concon", transform=ToTensor())
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(transforms.ToPILImage()(torch.squeeze(x)))
        plt.show()
        print(x.shape)
        print(y.shape)
        break


__all__ = ["ConConDataset"]
