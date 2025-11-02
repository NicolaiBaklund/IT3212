import numpy as np
from ImageData import ImageDataset

class PCA(ImageDataset):
    """PCA estimator that performs principal component analysis on image data.
    Holds the learned principal components and dataset statistics (mean, centered data).
    """

    def __init__(self, n_components):
        """Initialize the PCA object.
        Parameters:
            n_components (int): The number of principal components to compute and retain.
        """
        self.data: np.ndarray = None # Data matrix (num_samples, num_features) # type: ignore
        self.num_samples: int = None # Number of samples (images) # type: ignore
        self.num_features: int = None # Number of features (pixels) # type: ignore
        self.mean: np.ndarray = None # Mean image vector # type: ignore
        self.centered_data: np.ndarray = None # Centered data matrix # type: ignore
        self.cov_matrix: np.ndarray = None # Covariance matrix # type: ignore
        self.eigenvalues: np.ndarray = None  # Eigenvalues of the covariance matrix # type: ignore
        self.components: np.ndarray = None  # Principal components (eigenvectors) # type: ignore
        self.n_components: int = n_components

    def fit(self, dataset: ImageDataset):
        """Fit the PCA model using an ImageDataset instance.
        Computes the covariance matrix, eigenvalues, and eigenvectors, then stores the top components.
        If num_samples < num_features, uses dual PCA approach for efficiency. Covariance matrix = XX^t/(n-1) instead of X^tX/(n-1).
        Parameters:
            dataset (ImageDataset): An object providing .data, .num_samples, .num_features, .mean, and .centered_data.
        Returns:
            None
        """
        # Center the data
        self.data = dataset.data
        self.num_samples = dataset.num_samples
        self.num_features = dataset.num_features
        self.mean = dataset.mean
        self.centered_data = dataset.centered_data

        # Dual PCA approach if num_samples < num_features
        if self.num_samples < self.num_features:
            print("Using dual PCA approach.")
            print("Computing smaller covariance matrix...")
            print("If dimensions of ImageDataset are too large, this may take some time...")
            C = (self.centered_data @ self.centered_data.T) / (self.num_samples - 1)
            print("Computing eigenvalues and eigenvectors of the smaller covariance matrix...")
            eigvals_small, U = np.linalg.eigh(C)

            # sort descending
            idx = np.argsort(eigvals_small)[::-1]
            eigvals_small = eigvals_small[idx]
            U = U[:, idx]

            # keep only positive / non-tiny eigenvalues
            mask = eigvals_small > 1e-12
            eigvals_keep = eigvals_small[mask]
            U_keep = U[:, mask]

            # map back to feature space (each column becomes one eigenvector)
            V_cols = []
            for i in range(eigvals_keep.shape[0]):
                v = self.centered_data.T @ U_keep[:, i]
                v /= np.sqrt((self.num_samples - 1) * eigvals_keep[i])  # correct scaling
                # optional: v /= np.linalg.norm(v)  # if you want to re-normalize
                V_cols.append(v)

            V = np.column_stack(V_cols)  # shape (num_features, r_kept)
            print(f"Computed {V.shape[1]} principal components from {self.num_samples} samples.")
            # select requested number of components
            r = min(self.n_components, V.shape[1])
            self.components = V[:, :r]                  # (num_features, r)
            self.eigenvalues = eigvals_keep[:r]         # align 1:1 with components

            # (optional) explained variance ratio
            total_var = eigvals_keep.sum()
            self.explained_variance_ratio_ = self.eigenvalues / total_var



        else:
            print("SVD is not yet implemented...")

        

    def transform(self, X):
        """Project input data X onto the learned principal components.
        Parameters:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).
        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """Fit the PCA model on X (an ImageDataset) and return the transformed data.
        Parameters:
            X (ImageDataset): Dataset to fit and transform.
        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X.data if isinstance(X, ImageDataset) else X)
    
    def reconstruct(self, X_transformed):
        """Reconstruct data from its PCA-transformed representation back to the original feature space.
        Parameters:
            X_transformed (np.ndarray): Data in the reduced PCA space of shape (n_samples, n_components).
        Returns:
            np.ndarray: Reconstructed data of shape (n_samples, n_features).
        """
        return np.dot(X_transformed, self.components.T) + self.mean
    
    def transform_and_reconstruct(self, X):
        """Transform input data and then reconstruct it back to the original space.
        Parameters:
            X (np.ndarray): Original data of shape (n_samples, n_features).
        Returns:
            tuple: (X_transformed, X_reconstructed) where X_transformed has shape (n_samples, n_components)
                   and X_reconstructed has shape (n_samples, n_features).
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.reconstruct(X_transformed)
        return X_transformed, X_reconstructed

    def explained_variance(self):
        """Return the eigenvalues (variances) and the explained-variance ratio.
        Returns:
            tuple: (eigenvalues, explained_variance_ratio)
        """
        return self.eigenvalues, self.explained_variance_ratio_
    
    def transform_and_reconstruct_n(self, X, n_components):
        """Reconstruct data using only the top n_components principal components.
        Parameters:
            X (np.ndarray): Original data of shape (n_samples, n_features).
            n_components (int): Number of principal components to use for reconstruction.
        Returns:
            X_reconstructed (np.ndarray): Reconstructed data of shape (n_samples, n_features).
            X_transformed (np.ndarray): The PCA-transformed data of shape (n_samples, n_components).
            mean (np.ndarray): The mean image vector used for centering.
        """
        if n_components > self.n_components:
            raise ValueError(f"n_components ({n_components}) cannot be greater than fitted components ({self.n_components}).")
        X_centered = X - self.mean
        components_subset = self.components[:, :n_components]
        X_transformed = np.dot(X_centered, components_subset)
        X_reconstructed = np.dot(X_transformed, components_subset.T) + self.mean
        return X_reconstructed, X_transformed, self.mean
    
    def reconstruct_n(self, X_transformed, n_components):
        """Reconstruct data from its PCA-transformed representation using only the top n_components principal components.
        Parameters:
            X_transformed (np.ndarray): Data in the reduced PCA space of shape (n_samples, n_components).
            n_components (int): Number of principal components to use for reconstruction.
        Returns:
            X_reconstructed (np.ndarray): Reconstructed data of shape (n_samples, n_features).

        """
        if n_components > self.n_components:
            raise ValueError(f"n_components ({n_components}) cannot be greater than fitted components ({self.n_components}).")
        components_subset = self.components[:, :n_components]
        X_reconstructed = np.dot(X_transformed[:, :n_components], components_subset.T) + self.mean
        return X_reconstructed 