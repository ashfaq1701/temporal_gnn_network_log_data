import numpy as np


class CustomMaxScaler:

    def __init__(self, scaling_factor=None):
        self.max = None
        self.scaling_factor = scaling_factor or 1.0

    def fit(self, data):
        self.max = np.max(np.abs(data))

    def transform(self, data):
        return (data / self.max) * self.scaling_factor

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
