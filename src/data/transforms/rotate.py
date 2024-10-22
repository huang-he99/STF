import numpy as np
import cv2


class Rotate:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        rotate_mode = np.random.randint(0, 3)
        is_rotate = np.random.randint(0, 2)
        for key in self.key_list:
            data = data_info[key]
            data = self.rotate(data, is_rotate, rotate_mode)
            data_info[key] = data
        data_info['rotate_mode'] = rotate_mode if is_rotate else -1
        return data_info

    def rotate(self, data, is_rotate, rotate_mode):
        if is_rotate:
            data = cv2.rotate(data, rotate_mode)
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
