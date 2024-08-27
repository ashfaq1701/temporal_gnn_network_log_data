import numpy as np


class CustomMaxScaler:

    def __init__(self):
        self.max = None

    def fit(self, data):
        self.max = np.max(np.abs(data))

    def transform(self, data):
        return data / self.max

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
