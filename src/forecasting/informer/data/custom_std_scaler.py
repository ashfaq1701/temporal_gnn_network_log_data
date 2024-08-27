import numpy as np


class CustomStandardScaler:
    def __init__(self, per_feature):
        self.per_feature = per_feature
        self.mean = None
        self.std = None

    def fit(self, data):
        if not self.per_feature:
            self.mean = data.mean()
            self.std = data.std()
            if np.isnan(self.std):
                self.std = 0
        else:
            self.mean = data.mean(0)
            self.std = data.std(0)
            self.std[self.std == 0] = 1

    def transform(self, data):
        if self.mean is None:
            raise ValueError('Scaler is not fitted')

        if not self.per_feature:
            scaled = self._transform_combined(data)
        else:
            scaled = self._transform_per_feature(data)

        return scaled

    def _transform_per_feature(self, data):
        if self.std is not None:
            scaled = (data - self.mean) / self.std
        else:
            scaled = data - self.mean

        return scaled

    def _transform_combined(self, data):
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
