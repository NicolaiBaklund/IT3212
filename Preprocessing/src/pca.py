import os, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------
# Data loading


def load_grayscale_faces(root_dir, pattern="*_aligned*.png"):
    """
    Recursively loads grayscale PNGs under root_dir/**/pattern.
    Returns:
      X: (n_samples, n_pixels) float32 in [0,1]
      imgs: list of 2D float32 arrays (H, W)
      H, W: image height and width
      paths: list of file paths in the same order as rows of X
    """
    files = sorted(glob.glob(os.path.join(root_dir, "**", pattern), recursive=True))
    if not files:
        raise FileNotFoundError(f"No images found with pattern {pattern} under {root_dir}")

    imgs = []
    for fp in files:
        img = Image.open(fp).convert("L")  # force grayscale
        imgs.append(np.asarray(img, dtype=np.float32) / 255.0)

    # Ensure all same size
    H, W = imgs[0].shape
    for i, im in enumerate(imgs):
        if im.shape != (H, W):
            raise ValueError(f"Image size mismatch at {files[i]}: {im.shape} != {(H, W)}")

    X = np.stack([im.reshape(-1) for im in imgs], axis=0)  # (n_samples, n_pixels)
    return X, imgs, H, W, files

# --------------------------
# PCA (simple, via SVD)
# --------------------------
def pca(X, k=None):
    """
    PCA using SVD on centered data.
    Args:
      X: (n_samples, n_features)
      k: number of components (None -> keep all)
    Returns:
      mean: (n_features,)
      components: (k, n_features) top eigenvectors (eigenfaces)
      explained_var: (k,) variances of each component
      scores: (n_samples, k) projections (weights for each sample)
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if k is None or k > Vt.shape[0]:
        k = Vt.shape[0]

    components = Vt[:k, :]                   # rows are eigenvectors (eigenfaces)
    explained_var_all = (S**2) / (n_samples - 1)
    explained_var = explained_var_all[:k]
    scores = Xc @ components.T               # projections/weights

    return mean.ravel(), components, explained_var, scores

# --------------------------
# Visualization helpers
# --------------------------
def show_eigenfaces(components, H, W, n_show=8):
    n_show = min(n_show, components.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(1.8*n_show, 2.2))
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        ef = components[i].reshape(H, W)
        # normalize for display
        ef_disp = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        axes[i].imshow(ef_disp, cmap="gray", interpolation="nearest")
        axes[i].set_title(f"PC {i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

def reconstruct(mean, components, weights):
    """
    Reconstruct from mean + linear combo of components with given weights.
    """
    return mean + weights @ components

def show_reconstruction_pipeline(X_row, mean, components, H, W, num_terms=8):
    """
    Show:
      - original face
      - mean face
      - first n eigenfaces
      - reconstruction as mean + sum(w_i * eigenface_i)
      - incremental partial sums (optional)
    """
    k = components.shape[0]
    num_terms = min(num_terms, k)

    # Compute weights for this face
    x_c = X_row - mean
    weights = x_c @ components.T  # (k,)

    # Full reconstruction using the first num_terms
    w_use = weights[:num_terms]
    rec = reconstruct(mean, components[:num_terms, :], w_use)

    # Display original vs mean vs reconstruction
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, img, title in zip(
        axes,
        [X_row.reshape(H, W), mean.reshape(H, W), rec.reshape(H, W)],
        ["Original", "Mean", f"Reconstruction\n({num_terms} PCs)"],
    ):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Show the first num_terms eigenfaces with their weights
    fig, axes = plt.subplots(2, num_terms, figsize=(1.6*num_terms, 3.6))
    for i in range(num_terms):
        ef = components[i].reshape(H, W)
        ef_disp = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        axes[0, i].imshow(ef_disp, cmap="gray", interpolation="nearest")
        axes[0, i].set_title(f"PC {i+1}")
        axes[0, i].axis("off")
        axes[1, i].text(0.5, 0.5, f"w{i+1}={weights[i]:.3f}", ha="center", va="center")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Eigenfaces", rotation=90, size=10)
    axes[1, 0].set_ylabel("Weights", rotation=90, size=10)
    plt.tight_layout()
    plt.show()

    # Optional: incremental partial sums to illustrate "mean + weighted eigenfaces"
    cols = min(num_terms, 6)
    rows = int(np.ceil(num_terms / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 2.0*rows))
    axes = np.atleast_2d(axes)
    partial = mean.copy()
    for i in range(num_terms):
        partial = partial + weights[i] * components[i]
        r, c = divmod(i, cols)
        axes[r, c].imshow(np.clip(partial.reshape(H, W), 0, 1), cmap="gray", interpolation="nearest")
        axes[r, c].set_title(f"Mean + Σ w_j e_j\nj=1..{i+1}")
        axes[r, c].axis("off")
    # Hide any unused axes
    for j in range(num_terms, rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.show()

# --------------------------
# Run it on your data
# --------------------------

def plot_face_mean_eigenfaces_individual(image, mean, components, H=None, W=None, num_terms=8, prefix=""):
    """
    Instead of one combined plot, make separate figures:
      - original image
      - mean face
      - each eigenface with its weight
    """
    img = np.asarray(image, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    comps = np.asarray(components, dtype=np.float32)

    # Infer shape
    if img.ndim == 2:
        H_img, W_img = img.shape
        H, W = H or H_img, W or W_img
        img_flat = img.ravel()
    elif img.ndim == 1:
        if H is None or W is None:
            raise ValueError("If 'image' is 1D, please provide H and W.")
        if img.size != H*W:
            raise ValueError(f"image size {img.size} doesn't match H*W={H*W}.")
        img_flat = img
    else:
        raise ValueError("image must be 1D or 2D array.")

    # Compute weights
    x_centered = img_flat - mean
    weights = x_centered @ comps.T

    num_terms = int(min(num_terms, comps.shape[0]))

    # Helper for plotting one image
    def show_single(arr, title="", normalize=False):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        a = np.asarray(arr, dtype=np.float32).reshape(H, W)
        if normalize:
            mn, mx = a.min(), a.max()
            a = (a - mn) / (mx - mn + 1e-8)
        ax.imshow(np.clip(a, 0, 1), cmap="gray", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Plot original
    show_single(img_flat, title=f"{prefix}Original")

    # Plot mean
    show_single(mean, title=f"{prefix}Mean")

    # Plot eigenfaces individually
    for i in range(num_terms):
        ef = comps[i]
        ef_disp = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        show_single(ef_disp, title=f"{prefix}Eigenface {i}\n w{i}", normalize=False)

