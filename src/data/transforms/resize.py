import numpy as np
import cv2


class Resize:
    """
    interpolation_mode:
        0: INTER_NEAREST
        1: INTER_LINEAR
        2: INTER_CUBIC
        3: INTER_AREA
        4: INTER_LANCZOS4
        5: INTER_LINEAR_EXACT
        6: INTER_MAX
        7: WARP_FILL_OUTLIERS
        8: WARP_INVERSE_MAP
    """

    def __init__(
        self,
        key_list,
        resize_shape=None,
        scale_factor=None,
        interpolation_mode=0,
        is_save_original_data=False,
        is_remain_original_data=False,
    ):
        self.key_list = key_list
        self.resize_shape = resize_shape
        self.scale_factor = scale_factor
        self.interpolation_mode = interpolation_mode
        self.is_save_original_data = is_save_original_data
        self.is_remain_original_data = is_remain_original_data

    def transform(self, data_info: dict):
        for key in self.key_list:
            data = data_info[key]
            resized_data = self.resize(data)
            if self.is_save_original_data:
                data_info[f'ori_{key}'] = data
            data_info[key] = resized_data
        return data_info

    def resize(self, data):
        original_shape = np.array(data.shape[1::-1])
        if self.scale_factor is not None:
            resize_shape = original_shape * self.scale_factor
            resize_shape = resize_shape.astype(np.uint)
        else:
            resize_shape = self.resize_shape
        resized_data = cv2.resize(
            data, resize_shape, interpolation=self.interpolation_mode
        )
        if self.is_remain_original_data:
            resized_data = cv2.resize(
                resized_data, original_shape, interpolation=self.interpolation_mode
            )
        return resized_data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
