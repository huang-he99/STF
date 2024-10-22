import numpy as np


class Nan2Zero:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        for key in self.key_list:
            data = data_info[key]
            data = self.nan2zero(data)
            data_info[key] = data

        return data_info

    def nan2zero(self, data):
        data = np.where(np.isnan(data), 0, data)
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
