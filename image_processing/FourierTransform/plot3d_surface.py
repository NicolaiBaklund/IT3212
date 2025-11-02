import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_3d_surface_plot(in_dir, out_dir=None):
    """
    Creates 3D surface plots from grayscale images and saves them as PNG files.
    
    Args:
        in_dir (str): Path to directory containing grayscale images
        out_dir (str, optional): Path to directory where 3D surface plots will be saved.
                                If None, returns the 3D plot data instead of saving.
    
    Returns:
        dict or None: If out_dir is None, returns dict with plot data for first image.
                     If out_dir is provided, saves plots and returns None.
    """
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(in_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        return None
    
    # If out_dir is provided, save plots
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        
        # Process each image
        for image_file in image_files:
            # Load grayscale image
            image_path = os.path.join(in_dir, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Create 3D surface plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create coordinate grids
            height, width = img.shape
            x = np.arange(width)
            y = np.arange(height)
            X, Y = np.meshgrid(x, y)
            
            # Normalize image values to 0-1 range for better visualization
            Z = img.astype(np.float32) / 255.0
            
            # Create surface plot
            surface = ax.plot_surface(X, Y, Z, cmap='gray', alpha=0.8, linewidth=0, antialiased=True)
            
            # Set labels and title
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            ax.set_zlabel('Intensity')
            ax.set_title(f'3D Surface Plot: {os.path.splitext(image_file)[0]}')
            
            # Set view angle for better visualization
            ax.view_init(elev=30, azim=45)
            
            # Remove axes ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            # Save the plot
            output_filename = f"3d_surface_{os.path.splitext(image_file)[0]}.png"
            output_path = os.path.join(out_dir, output_filename)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
    
    # If out_dir is None, return 3D plot data for the first image
    else:
        image_file = image_files[0]
        image_path = os.path.join(in_dir, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        # Create coordinate grids
        height, width = img.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Normalize image values to 0-1 range for better visualization
        Z = img.astype(np.float32) / 255.0
        
        return {
            'X': X,
            'Y': Y,
            'Z': Z,
            'title': f'3D Surface Plot: {os.path.splitext(image_file)[0]}'
        }


def create_3d_surface_plot_from_image(image, title="3D Surface Plot"):
    """
    Creates 3D surface plot data from a single grayscale image.
    
    Args:
        image (numpy.ndarray): Input grayscale image
        title (str): Title for the 3D surface plot
        
    Returns:
        dict: Dictionary containing 'X', 'Y', 'Z', and 'title' for the 3D plot
    """
    # Create coordinate grids
    height, width = image.shape
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Normalize image values to 0-1 range for better visualization
    Z = image.astype(np.float32) / 255.0
    
    return {
        'X': X,
        'Y': Y,
        'Z': Z,
        'title': title
    }



