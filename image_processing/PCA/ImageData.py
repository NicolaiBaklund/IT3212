import numpy as np
class ImageDataset:
    def __init__(self, data_matrix, img_h: int, img_w: int):
        self.data = data_matrix
        self.num_samples, self.num_features = data_matrix.shape
        self.mean = np.mean(data_matrix, axis=0)
        self.centered_data = data_matrix - self.mean
        self.image_height = img_h
        self.image_width = img_w
    def __getitem__(self, idx):
        """Indexing: int -> 1D ndarray (single sample); slice/list/mask -> ndarray of rows (2D)."""
        result = np.asarray(self.data[idx])  # keep views when possible
        if np.isscalar(idx) or isinstance(idx, (int, np.integer)):
            return result
        if result.ndim == 1:
            return result.reshape(1, -1)
        return result
