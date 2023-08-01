# Avalanche-FFCV examples

This folder contains some examples that can be used to get started with the [FFCV](https://ffcv.io/) data loading mechanism in Avalanche.

Avalanche currently supports the FFCV data loading mechanism for virtually all benchmark types. However, automatic support is given only for **classification** and **regression** tasks due to the complex encoder/decoder definitions in FFCV.

## Examples list

- `ffcv_enable.py`: the main example, shows how to enable FFCV in Avalanche.
- `ffcv_enable_rgb_compress.py`: shows how to use the jpg/mixed image encoding.
- `ffcv_io_manual_test.py`: a template you can use to manually setup the decoder pipeline.
- `ffcv_try_speed.py`: a benchmarking script to compare FFCV to PyTorch.

