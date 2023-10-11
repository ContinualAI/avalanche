"""
Implementation of the CenterCrop transformation for FFCV
"""

from typing import Callable, Tuple
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.pipeline.state import State
from ffcv.pipeline.allocation_query import AllocationQuery
import numpy as np
from dataclasses import replace
from ffcv.fields.rgb_image import IMAGE_MODES
from ffcv.pipeline.compiler import Compiler
from ffcv.libffcv import imdecode


def get_center_crop_torchvision_alike(
    image_height, image_width, output_size, img, out_buffer
):
    crop_height = output_size[0]
    crop_width = output_size[1]

    padding_h = (crop_height - image_height) // 2 if crop_height > image_height else 0
    padding_w = (crop_width - image_width) // 2 if crop_width > image_width else 0

    crop_t = (
        int(round((image_height - crop_height) / 2.0))
        if image_height > crop_height
        else 0
    )
    crop_l = (
        int(round((image_width - crop_width) / 2.0)) if image_width > crop_width else 0
    )
    crop_height_effective = min(crop_height, image_height)
    crop_width_effective = min(crop_width, image_width)

    # print(image_height, image_width, crop_height, crop_width, padding_h, padding_w, crop_t, crop_l, crop_height_effective, crop_width_effective)
    # print(f'From ({crop_t} : {crop_t+crop_height_effective}, {crop_l} : {crop_l+crop_width_effective}) to '
    #       f'{padding_h} : {padding_h+crop_height_effective}, {padding_w} : {padding_w+crop_width_effective}')

    if crop_height_effective != crop_height or crop_width_effective != crop_width:
        out_buffer[:] = 0  # Set padding color
    out_buffer[
        padding_h : padding_h + crop_height_effective,
        padding_w : padding_w + crop_width_effective,
    ] = img[
        crop_t : crop_t + crop_height_effective, crop_l : crop_l + crop_width_effective
    ]

    return out_buffer


class CenterCropRGBImageDecoderTVAlike(SimpleRGBImageDecoder):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a center crop operation.

    It supports both variable and constant resolution datasets.

    Differently from the original CenterCropRGBImageDecoder from FFCV,
    this operates like torchvision CenterCrop.
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, AllocationQuery]:
        widths = self.metadata["width"]
        heights = self.metadata["height"]
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype("<u1")

        return (
            replace(previous_state, jit_mode=True, shape=output_shape, dtype=my_dtype),
            (
                AllocationQuery(output_shape, my_dtype),
                AllocationQuery(
                    (self.max_height * self.max_width * np.uint64(3),), my_dtype
                ),
            ),
        )

    def generate_code(self) -> Callable:
        jpg = IMAGE_MODES["jpg"]

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        c_crop = Compiler.compile(self.get_crop_generator)
        output_size = self.output_size

        def decode(batch_indices, my_storage, metadata, storage_state):
            destination, temp_storage = my_storage
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field["data_ptr"], storage_state)
                height = np.uint32(field["height"])
                width = np.uint32(field["width"])

                if field["mode"] == jpg:
                    temp_buffer = temp_storage[dst_ix]
                    imdecode_c(
                        image_data,
                        temp_buffer,
                        height,
                        width,
                        height,
                        width,
                        0,
                        0,
                        1,
                        1,
                        False,
                        False,
                    )
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)
                else:
                    temp_buffer = image_data.reshape(height, width, 3)

                c_crop(height, width, output_size, temp_buffer, destination[dst_ix])

            return destination[: len(batch_indices)]

        decode.is_parallel = True
        return decode

    @property
    def get_crop_generator(self):
        return get_center_crop_torchvision_alike


__all__ = ["CenterCropRGBImageDecoderTVAlike"]
