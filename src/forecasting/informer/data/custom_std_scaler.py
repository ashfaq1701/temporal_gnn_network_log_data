import numpy as np


class CustomStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

        if np.isnan(self.std):
            self.std = 0

    def transform(self, data):
        if self.mean is None:
            raise ValueError('Scaler is not fitted')

        if self.std != 0:
            scaled = (data - self.mean) / self.std
        else:
            scaled = data - self.mean

        return scaled

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if self.mean is None:
            raise ValueError('Scaler is not fitted')

        return data * self.std + self.mean
