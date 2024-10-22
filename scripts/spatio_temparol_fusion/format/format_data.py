"""
description: 
    CHW -> HWC
return {*}
"""
import numpy as np


def format_data(data, band):
    if data.dtype == np.int16:
        data[data > 10000] = 10000
        data[data < 0] = 0
    shape = data.shape
    if shape[0] == band:  # CHW
        data = np.transpose(data, (1, 2, 0))
    return data


if __name__ == '__main__':
    pass
