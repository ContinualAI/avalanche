from torchvision.transforms import ToPILImage, ToTensor

from avalanche.benchmarks.utils import AvalancheDataset


def check_vision_benchmark(benchmark_instance, show_without_transforms=True):
    from matplotlib import pyplot as plt
    from torch.utils.data.dataloader import DataLoader

    dataset: AvalancheDataset

    print(
        "The benchmark instance contains",
        len(benchmark_instance.train_stream),
        "training experiences.",
    )

    for i, exp in enumerate(benchmark_instance.train_stream):
        dataset, t = exp.dataset, exp.task_label
        if show_without_transforms:
            dataset = dataset.replace_transforms(ToTensor(), None)

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
