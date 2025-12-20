"""
Functions to visualize images grouped by cluster labels.

This module provides functions to display images from the dataset
organized by their cluster assignments from clustering analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2

from preprocess4 import DEFAULT_OUTPUT_FILE, DEFAULT_DATASET_FOLDER


def load_image_paths_from_csv(
    csv_path: Path = DEFAULT_OUTPUT_FILE,
    dataset_folder: Path = DEFAULT_DATASET_FOLDER,
    emotions: List[str] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load image paths from CSV and return DataFrame with paths and labels.
    
    Args:
        csv_path: Path to the CSV file with features
        dataset_folder: Path to the dataset folder containing images
        emotions: Optional list of emotions to filter (if None, uses all)
    
    Returns:
        Tuple of (df_with_paths, indices) where:
        - df_with_paths: DataFrame with Image_Path column added/reconstructed
        - indices: Array of indices for selected samples
    """
    df = pd.read_csv(csv_path)
    
    # Reconstruct image paths if Image_Path column doesn't exist
    if 'Image_Path' not in df.columns:
        # Reconstruct from Person_ID and Image_Name
        # Handle both string and numeric Person_ID
        df['Image_Path'] = df.apply(
            lambda row: f"{row['Person_ID']}/{row['Image_Name']}", 
            axis=1
        )
        print("Note: Reconstructed Image_Path from Person_ID and Image_Name")
    
    # Filter by emotions if specified
    if emotions:
        emotions_normalized = [e.lower() for e in emotions]
        df['Label_normalized'] = df['Label'].str.lower()
        mask = df['Label_normalized'].isin(emotions_normalized)
        df = df[mask].copy()
        indices = np.where(mask)[0]
    else:
        indices = np.arange(len(df))
    
    return df, indices


def get_image_paths_for_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    dataset_folder: Path = DEFAULT_DATASET_FOLDER,
    max_images_per_cluster: Optional[int] = 10
) -> Dict[int, List[Tuple[Path, str, str]]]:
    """
    Get image paths organized by cluster labels.
    
    Args:
        df: DataFrame with Image_Path, Label, Person_ID columns
        cluster_labels: Array of cluster labels (same length as df)
        dataset_folder: Path to dataset folder
        max_images_per_cluster: Maximum number of images to show per cluster.
                               If None, shows all images in each cluster.
    
    Returns:
        Dictionary mapping cluster_id -> list of (image_path, true_label, person_id) tuples
    """
    cluster_images = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            cluster_name = "Outliers"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Limit number of images (only if max_images_per_cluster is specified)
        if max_images_per_cluster is not None and len(cluster_indices) > max_images_per_cluster:
            np.random.seed(42)
            cluster_indices = np.random.choice(
                cluster_indices, 
                size=max_images_per_cluster, 
                replace=False
            )
        
        # Get image paths and metadata
        images_info = []
        for idx in cluster_indices:
            if idx < len(df):
                row = df.iloc[idx]
                image_path = dataset_folder / row['Image_Path']
                true_label = row['Label']
                person_id = row['Person_ID']
                images_info.append((image_path, true_label, person_id))
        
        cluster_images[cluster_id] = images_info
    
    return cluster_images


def visualize_cluster_images(
    cluster_images: Dict[int, List[Tuple[Path, str, str]]],
    title: str = "Images by Cluster",
    max_cols: int = 5
):
    """
    Display images organized by cluster in a grid layout.
    
    Args:
        cluster_images: Dictionary from get_image_paths_for_clusters()
        title: Title for the visualization
        max_cols: Maximum number of columns in the grid
    """
    n_clusters = len(cluster_images)
    
    # Calculate total number of images
    total_images = sum(len(images) for images in cluster_images.values())
    if total_images == 0:
        print("No images to display.")
        return
    
    # Create one figure per cluster for better organization
    for cluster_id, images in sorted(cluster_images.items()):
        if len(images) == 0:
            continue
            
        if cluster_id == -1:
            cluster_title = "Outliers"
        else:
            cluster_title = f"Cluster {cluster_id}"
        
        n_images = len(images)
        n_cols = min(max_cols, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_images == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f"{cluster_title} ({n_images} images)", 
                     fontsize=14, fontweight='bold')
        
        for img_idx, (image_path, true_label, person_id) in enumerate(images):
            ax = axes[img_idx] if n_images > 1 else axes[0]
            
            # Load and display image
            if image_path.exists():
                img = cv2.imread(str(image_path))
                if img is not None:
                    # Convert BGR to RGB for matplotlib
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                else:
                    ax.text(0.5, 0.5, 'Image\nnot found', 
                           ha='center', va='center', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Path not found:\n{image_path.name}', 
                       ha='center', va='center', fontsize=8)
            
            ax.axis('off')
            # Add label as title
            ax.set_title(f"{true_label}\n(P{person_id})", fontsize=9, pad=2)
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def visualize_clusters_with_images(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    csv_path: Path = DEFAULT_OUTPUT_FILE,
    dataset_folder: Path = DEFAULT_DATASET_FOLDER,
    method_name: str = "Clustering",
    max_images_per_cluster: Optional[int] = 10,
    selected_indices: np.ndarray = None
):
    """
    Main function to visualize images grouped by cluster assignments.
    
    Args:
        cluster_labels: Array of cluster labels from clustering algorithm
        true_labels: Array of true emotion labels
        csv_path: Path to CSV file with features
        dataset_folder: Path to dataset folder
        method_name: Name of clustering method (for title)
        max_images_per_cluster: Maximum images to show per cluster. If None, shows all images.
        selected_indices: Optional array of indices in original CSV that were used
                         (from load_and_prepare_data). If provided, uses these exact indices.
    """
    # Load full CSV
    df_full = pd.read_csv(csv_path)
    
    # Reconstruct image paths if Image_Path column doesn't exist
    if 'Image_Path' not in df_full.columns:
        df_full['Image_Path'] = df_full.apply(
            lambda row: f"{row['Person_ID']}/{row['Image_Name']}", 
            axis=1
        )
    
    # Use selected_indices if provided (from load_and_prepare_data)
    if selected_indices is not None:
        if len(selected_indices) != len(cluster_labels):
            raise ValueError(
                f"Mismatch: selected_indices ({len(selected_indices)}) != "
                f"cluster_labels ({len(cluster_labels)})"
            )
        # Use the exact indices that were used for clustering
        df = df_full.iloc[selected_indices].copy().reset_index(drop=True)
        print(f"Using provided selected_indices: {len(selected_indices)} samples")
    else:
        # Fallback: try to match by length (less reliable)
        if len(df_full) != len(cluster_labels):
            print(f"Warning: Mismatch between cluster_labels ({len(cluster_labels)}) "
                  f"and CSV data ({len(df_full)}). Using first {len(cluster_labels)} rows.")
            print("  Recommendation: Pass selected_indices from load_and_prepare_data()")
            df = df_full.iloc[:len(cluster_labels)].copy().reset_index(drop=True)
        else:
            df = df_full.copy()
    
    # Get images organized by cluster
    cluster_images = get_image_paths_for_clusters(
        df, cluster_labels, dataset_folder, max_images_per_cluster
    )
    
    # Display
    title = f"{method_name} - Images by Cluster Assignment"
    visualize_cluster_images(cluster_images, title=title)
    
    # Print summary statistics
    print(f"\n{method_name} - Cluster Summary:")
    print("=" * 60)
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = cluster_labels == cluster_id
        n_samples = cluster_mask.sum()
        true_labels_cluster = true_labels[cluster_mask]
        
        if cluster_id == -1:
            cluster_name = "Outliers"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        print(f"\n{cluster_name}: {n_samples} samples")
        if len(true_labels_cluster) > 0:
            label_counts = pd.Series(true_labels_cluster).value_counts()
            print("  True label distribution:")
            for label, count in label_counts.items():
                print(f"    {label}: {count} ({100*count/n_samples:.1f}%)")


def compare_clusters_vs_true_labels(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    csv_path: Path = DEFAULT_OUTPUT_FILE,
    dataset_folder: Path = DEFAULT_DATASET_FOLDER,
    method_name: str = "Clustering",
    max_images: int = 20,
    selected_indices: np.ndarray = None
):
    """
    Visualize images showing both cluster assignment and true label.
    Useful for understanding clustering mistakes.
    
    Args:
        cluster_labels: Array of cluster labels
        true_labels: Array of true emotion labels
        csv_path: Path to CSV file
        dataset_folder: Path to dataset folder
        method_name: Name of clustering method
        max_images: Maximum number of images to display
        selected_indices: Optional array of indices in original CSV that were used
    """
    # Load full CSV
    df_full = pd.read_csv(csv_path)
    
    # Reconstruct image paths if needed
    if 'Image_Path' not in df_full.columns:
        df_full['Image_Path'] = df_full.apply(
            lambda row: f"{row['Person_ID']}/{row['Image_Name']}", 
            axis=1
        )
    
    # Use selected_indices if provided
    if selected_indices is not None:
        if len(selected_indices) != len(cluster_labels):
            raise ValueError(
                f"Mismatch: selected_indices ({len(selected_indices)}) != "
                f"cluster_labels ({len(cluster_labels)})"
            )
        df = df_full.iloc[selected_indices].copy().reset_index(drop=True)
    else:
        if len(df_full) != len(cluster_labels):
            df = df_full.iloc[:len(cluster_labels)].copy().reset_index(drop=True)
        else:
            df = df_full.copy()
    
    # Create comparison: show images where cluster != true label (if applicable)
    # Or show sample from each cluster
    fig, axes = plt.subplots(2, max_images // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    # Sample images from different clusters
    unique_clusters = sorted(set(cluster_labels))
    images_per_cluster = max_images // len(unique_clusters) if len(unique_clusters) > 0 else max_images
    
    img_idx = 0
    for cluster_id in unique_clusters:
        if img_idx >= max_images:
            break
        
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            sample_indices = cluster_indices[:images_per_cluster]
            
            for idx in sample_indices:
                if img_idx >= max_images:
                    break
                
                row = df.iloc[idx]
                image_path = dataset_folder / row['Image_Path']
                true_label = row['Label']
                
                ax = axes[img_idx]
                
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        if len(img.shape) == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                
                cluster_name = "Outlier" if cluster_id == -1 else f"C{cluster_id}"
                ax.set_title(
                    f"Cluster: {cluster_name}\nTrue: {true_label}",
                    fontsize=9
                )
                ax.axis('off')
                img_idx += 1
    
    # Hide unused subplots
    for i in range(img_idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"{method_name} - Sample Images with Cluster and True Labels", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

