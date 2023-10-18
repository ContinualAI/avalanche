from torchvision.transforms import ToPILImage, ToTensor
from avalanche.benchmarks.scenarios.deprecated.dataset_scenario import (
    DatasetScenario,
    DatasetStream,
)

from avalanche.benchmarks.utils.classification_dataset import (
    TaskAwareClassificationDataset,
)


def check_vision_benchmark(
    benchmark_instance: DatasetScenario, show_without_transforms=True
):
    from matplotlib import pyplot as plt
    from torch.utils.data.dataloader import DataLoader

    dataset: TaskAwareClassificationDataset
    train_stream: DatasetStream = benchmark_instance.train_stream

    print(
        "The benchmark instance contains",
        len(train_stream),
        "training experiences.",
    )

    for exp in train_stream:
        dataset, t = exp.dataset, exp.task_label
        if show_without_transforms:
            dataset = dataset.replace_current_transform_group(ToTensor())

        dl = DataLoader(dataset, batch_size=300)

        print("Train experience", exp.current_experience)
        for mb in dl:
            x, y, *other = mb
            print("X tensor:", x.shape)
            print("Y tensor:", y.shape)
            if len(other) > 0:
                print("T tensor:", other[0].shape)
            img = ToPILImage()(x[0])
            plt.title("Experience: " + str(exp.current_experience))
            plt.imshow(img)
            plt.show()
            break  # Show only an image for each experience


__all__ = ["check_vision_benchmark"]
