import numpy as np


# 1
class RescaleToZeroOne:
    normalize_mode = 1

    def __init__(self, key_list, data_range=[0, 255]):
        self.key_list = key_list
        self.data_range = data_range
        self.min = data_range[0]
        self.max = data_range[1]

    def transform(self, data_info: dict):
        for key in self.key_list:
            data = data_info[key]
            data = self.rescale_to_zero_one(data)
            data_info[key] = data
            data_info['normalize_scale'] = self.max
            data_info['normalize_mode'] = self.normalize_mode
        return data_info

    def rescale_to_zero_one(self, data):
        data = data.astype(np.float32)
        data = (data - self.min) / (self.max - self.min)
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'


# 2
class RescaleToMinusOneOne:
    normalize_mode = 2

    def __init__(self, key_list, data_range=[0, 255]):
        self.key_list = key_list
        self.data_range = data_range
        self.min = data_range[0]
        self.max = data_range[1]

    def transform(self, data_info: dict):
        for key in self.key_list:
            data = data_info[key]
            data = self.rescale_to_minus_one_one(data)
            data_info[key] = data
            data_info['normalize_scale'] = self.max
            data_info['normalize_mode'] = self.normalize_mode
        return data_info

    def rescale_to_minus_one_one(self, data):
        data = data.astype(np.float32)
        data = (data - self.min) / (self.max - self.min)
        data = 2 * data - 1
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
