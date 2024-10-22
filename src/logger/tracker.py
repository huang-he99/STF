import pandas as pd
import torch
import numpy as np

eps = 1e-16


class Tracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average', 'now']
        )
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]
        self._data.now[key] = value

    def avg(self, key):
        return self._data.average[key]

    @property
    def results(self):
        return dict(self._data.average)

    @property
    def now(self):
        return dict(self._data.now)
