import tifffile
import numpy as np
import scipy.io as sio


class LoadData:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        for key in self.key_list:
            data_path = data_info[f'{key}_path']
            data = self.load_data(data_path)
            data_info[key] = data
            data_info[f'{key}_shape'] = data.shape
        return data_info

    def load_data(self, data_path: str):
        data = tifffile.imread(data_path)
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)


# SPSTFM
class LoadDictionarySparistyMatrix:
    def __init__(self, key_list, np_key_list):
        self.key_list = key_list
        self.np_key_list = np_key_list

    def transform(self, data_info: dict):
        for key in self.key_list:
            data_path = data_info[f'{key}_path']
            data = self.load_data(data_path)
            data_info[''] = data
            for np_key in self.np_key_list:
                data_info[np_key] = data[np_key]
        return data_info

    def load_data(self, data_path: str):
        data = sio.loadmat(data_path)

        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)
