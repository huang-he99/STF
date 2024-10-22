import numpy as np
from src.utils.patch import cal_padding_img_pixel_num_hw
from torch.nn.modules.utils import _pair, _quadruple


class Pad:
    def __init__(self, key_list, patch_size, patch_stride, is_drop_last=False):
        self.key_list = key_list
        self.patch_size = _pair(patch_size)
        self.patch_stride = _pair(patch_stride)
        self.is_drop_last = is_drop_last

    def transform(self, data_info: dict):
        for key in self.key_list:
            data = data_info[key]
            img_size = data_info[f'{key}_shape'][:2]
            data_info['ori_img_size'] = img_size
            data = self.pad(data, img_size)
            data_info[key] = data
        return data_info

    def pad(self, data, img_size):
        padding_img_pixel_num = cal_padding_img_pixel_num_hw(
            img_size, self.patch_size, self.patch_stride, self.is_drop_last
        )
        data = np.pad(data, (*padding_img_pixel_num, (0, 0)), mode='reflect')
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)
