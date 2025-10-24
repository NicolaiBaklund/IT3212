import numpy as np
class ImageDataset:
    def __init__(self, data_matrix, img_h: int, img_w: int):
        self.data = data_matrix
        self.num_samples, self.num_features = data_matrix.shape
        self.image_height = img_h
        self.image_width = img_w
        self.normalize()
        self.mean = np.mean(self.data, axis=0)
        self.centered_data = self.data - self.mean
    def __getitem__(self, idx):
        """Indexing: int -> 1D ndarray (single sample); slice/list/mask -> ndarray of rows (2D)."""
        result = np.asarray(self.data[idx])  # keep views when possible
        if np.isscalar(idx) or isinstance(idx, (int, np.integer)):
            return result
        if result.ndim == 1:
            return result.reshape(1, -1)
        return result
    def normalize(self):
        """Normalize pixel values to [0,1] range."""
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        self.data = (self.data - min_val) / (max_val - min_val)
        self.mean = np.mean(self.data, axis=0)
        self.centered_data = self.data - self.mean
