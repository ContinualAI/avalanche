"""
Simple script used to (manually) check if the FFCV pipeline returns
the expected outputs. This script can be used to inspect the output
of a decoding pipeline.

It is recommended to start with the automatic translation pipeline,
which Avalanche tries to put toghether when `prepare_ffcv_datasets`
has no `decoder_def` parameter. If you are not happy with the
automatic pipeline, then start putting your custom pipeline toghether
by folliwing the FFCV tutorials!
"""

#%%
import time
from matplotlib import pyplot as plt

import torch
from avalanche.benchmarks.classic.ccifar100 import SplitCIFAR100
from avalanche.benchmarks.classic.ctiny_imagenet import SplitTinyImageNet
from avalanche.benchmarks.utils.ffcv_support import prepare_ffcv_datasets
from avalanche.benchmarks.utils.ffcv_support.ffcv_components import HybridFfcvLoader
from avalanche.training.determinism.rng_manager import RNGManager

from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torch.utils.data import DataLoader


#%%
def main(cuda: int):
    # --- CONFIG
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    )
    RNGManager.set_random_seeds(1234)

    # Define here the transformations to check
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # benchmark = SplitCIFAR100(
    #     5,
    #     seed=4321,
    #     shuffle=True,
    #     train_transform=train_transform,
    #     eval_transform=eval_transform
    # )
    # write_dir='./ffcv_manual_test_cifar100'

    benchmark = SplitTinyImageNet()
    write_dir='./ffcv_manual_test_tiny_imagenet'
    
    # It is recommended to start with `None`, so that Avalanche can try
    # putting a pipeline together automatically by translating common
    # transformations to FFCV.
    # If you encounter issues or the output is not what you expect, then
    # it is recommended to start from the pipeline printed by Avalanche
    # and change it following the guides in the FFCV website and repo.
    custom_decoder_pipeline = None

    num_workers = 8

    print('Preparing FFCV datasets...')
    prepare_ffcv_datasets(
        benchmark=benchmark,
        write_dir=write_dir,
        device=device,
        ffcv_parameters=dict(num_workers=num_workers),
        decoder_def=custom_decoder_pipeline,
        print_summary=True  # Leave to True to get important info!
    )
    print('FFCV datasets ready')

    # Create the FFCV Loader
    start_time = time.time()
    ffcv_data_loader = HybridFfcvLoader(
        benchmark.train_stream[0].dataset,
        batch_sampler=None,
        batch_size=12,
        ffcv_loader_parameters=dict(
            num_workers=num_workers,
            drop_last=True
        ),
        device=device,
        persistent_workers=False,
        print_ffcv_summary=True,
        start_immediately=False
    )
    end_time = time.time()
    print('Loader creation took', end_time - start_time, 'seconds')

    # Also load the same data using a PyTorch DataLoader
    # Note: data will be different when using random augmentations!
    pytorch_loader = DataLoader(
        benchmark.train_stream[0].dataset,
        batch_size=12,
        drop_last=True
    )

    start_time = time.time()
    for i, (ffcv_batch, torch_batch) in enumerate(zip(ffcv_data_loader, pytorch_loader)):
        print(f'Batch {i} composition (FFCV vs PyTorch)')
        for element in ffcv_batch:
            print(element.shape, 'vs', element.shape)

        n_to_show = 3
        for idx in range(n_to_show):
            as_img_ffcv = to_pil_image(ffcv_batch[0][idx])
            as_img_torch = to_pil_image(torch_batch[0][idx])

            f, axarr = plt.subplots(1, 2)
            f.suptitle(
                f'Label: {ffcv_batch[1][idx].item()}/{torch_batch[1][idx].item()}, '
                f'Task label: {ffcv_batch[2][idx].item()}/{torch_batch[2][idx].item()}')

            axarr[0].set_title('FFCV')
            axarr[0].imshow(as_img_ffcv)
            axarr[1].set_title('PyTorch')
            axarr[1].imshow(as_img_torch)
            
            plt.show()
            f.clear()

        # ---------------------------------------------
        # Checks to verify that ffcv == pytorch
        # Note: when using certain transformations such as Normalize,
        # having `almost_same` True is usually sufficient even if
        # `all_same` is False.
        all_same = True
        almost_same = True
        correct_device = True

        for f, t in zip(ffcv_batch, torch_batch):
            print(f.shape, t.shape)
            correct_device = correct_device and f.device == device
            f = f.cpu()
            t = t.cpu()
            all_same = all_same and torch.equal(f, t)

            if f.dtype.is_floating_point:
                almost_same = almost_same and (
                    torch.sum(torch.abs(f - t) > 1e-6).item() == 0
                )
        
        print('all_same', all_same)
        print('almost_same', almost_same)
        print('correct_device', correct_device)
        # ---------------------------------------------
        
        # Keep this break if it is sufficient to analyze only the first batch
        break 

        # Print batch separator
        print('.' * 40)

    end_time = time.time()
    print('Loop time:', end_time - start_time, 'seconds')


# When running on VSCode (with Python extension), you will notice additional
# controls such as "Run Cell", "Run Above", ...
# The recommended way to use this script
# is to first "Run Above" and then "Run Cell".
#%%
main(0)

# %%
