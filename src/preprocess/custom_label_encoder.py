class CustomLabelEncoder:
    def __init__(self):
        self.label_to_int = {}
        self.int_to_label = {}
        self.classes_ = []
        self.is_fitted = False

    def fit(self, labels):
        unique_labels = sorted(set(labels))
        if 'Undefined' in unique_labels:
            unique_labels.remove('Undefined')
        unique_labels = ['Undefined'] + unique_labels

        self.classes_ = unique_labels
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
        self.is_fitted = True
        return self

    def transform(self, labels):
        if not self.is_fitted:
            raise ValueError("CustomLabelEncoder is not fitted yet. Call 'fit' first.")

        return [self.label_to_int[label] for label in labels]

    def inverse_transform(self, indices):
        if not self.is_fitted:
            raise ValueError("CustomLabelEncoder is not fitted yet. Call 'fit' first.")

        return [self.int_to_label[idx] for idx in indices]

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)
