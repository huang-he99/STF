import numpy as np
import torch


class Format:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        for key in self.key_list:
            data = data_info[key]
            data = self.to_tensor(data)
            data_info[key] = data
        return data_info

    def to_tensor(self, data):
        data = torch.from_numpy(data.transpose(2, 0, 1)).contiguous()
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
