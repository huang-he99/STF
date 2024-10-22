import cv2
import numpy as np


class Flip:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        flip_mode = np.random.randint(0, 2)
        is_flip = np.random.randint(0, 2)
        for key in self.key_list:
            data = data_info[key]
            data = self.flip(data, is_flip, flip_mode)
            data_info[key] = data
        data_info['flip_mode'] = flip_mode if is_flip else -1
        return data_info

    def flip(self, data, is_flip, flip_mode):
        if is_flip:
            data = cv2.flip(data, flip_mode)
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
