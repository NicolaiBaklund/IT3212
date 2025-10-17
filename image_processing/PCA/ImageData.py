import numpy as np
class ImageDataset:
    def __init__(self, data_matrix):
        self.data = data_matrix
        self.num_samples, self.num_features = data_matrix.shape
        self.mean = np.mean(data_matrix, axis=0)
        self.centered_data = data_matrix - self.mean