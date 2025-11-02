import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from plot3d_surface import create_3d_surface_plot_from_image


def load_grayscale_image(image_path):
    """
    Load a grayscale image from the specified path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Grayscale image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    print(f"Loaded: {os.path.basename(image_path)} ({img.shape[0]}x{img.shape[1]})")
    
    return img


def apply_2d_dft(image):
    """
    Apply 2D Discrete Fourier Transform to a grayscale image.
    
    Args:
        image (numpy.ndarray): Input grayscale image
        
    Returns:
        tuple: (magnitude_spectrum, phase_spectrum, complex_fft)
    """
    # Convert to float32 for better precision
    img_float = image.astype(np.float32)
    
    # Apply 2D FFT
    fft_result = np.fft.fft2(img_float)
    
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Calculate magnitude and phase
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    return magnitude, phase, fft_shifted


def visualize_dft_results(original_image, magnitude_spectrum, image_3d=None):
    """
    Visualize the original image, 3D surface plot, and its frequency spectrum (magnitude).
    
    Args:
        original_image (numpy.ndarray): Original grayscale image
        magnitude_spectrum (numpy.ndarray): Magnitude spectrum from DFT
        image_3d (dict, optional): 3D surface plot data containing 'X', 'Y', 'Z', and 'title'
    """
    # Create figure with subplots - 3 columns if 3D image provided, 2 otherwise
    if image_3d is not None:
        fig = plt.figure(figsize=(18, 5))
        
        # Original image (left)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original Grayscale Image')
        ax1.axis('off')
        
        # 3D surface plot (middle)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surface = ax2.plot_surface(image_3d['X'], image_3d['Y'], image_3d['Z'], 
                                 cmap='gray', alpha=0.8, linewidth=0, antialiased=True)
        ax2.set_title(image_3d['title'])
        ax2.set_xlabel('Width (pixels)')
        ax2.set_ylabel('Height (pixels)')
        ax2.set_zlabel('Intensity')
        ax2.view_init(elev=30, azim=45)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        
        # Frequency spectrum (right)
        ax3 = fig.add_subplot(1, 3, 3)
        magnitude_log = np.log(magnitude_spectrum + 1)  # Add 1 to avoid log(0)
        im = ax3.imshow(magnitude_log, cmap='hot')
        ax3.set_title('Frequency Spectrum (Magnitude)')
        ax3.axis('off')
        
        # Add colorbar for magnitude spectrum
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
    else:
        # Create figure with subplots (original 2-column layout)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Grayscale Image')
        axes[0].axis('off')
        
        # Plot magnitude spectrum (log scale for better visualization)
        magnitude_log = np.log(magnitude_spectrum + 1)  # Add 1 to avoid log(0)
        im = axes[1].imshow(magnitude_log, cmap='hot')
        axes[1].set_title('Frequency Spectrum (Magnitude)')
        axes[1].axis('off')
        
        # Add colorbar for magnitude spectrum
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def print_dft_analysis(magnitude_spectrum, phase_spectrum):
    """
    Print concise analysis of the DFT results.
    
    Args:
        magnitude_spectrum (numpy.ndarray): Magnitude spectrum
        phase_spectrum (numpy.ndarray): Phase spectrum
    """
    # Basic statistics
    print(f"Magnitude: [{magnitude_spectrum.min():.0f}, {magnitude_spectrum.max():.0f}], Mean: {magnitude_spectrum.mean():.0f}")
    
    # Energy distribution
    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    total_energy = np.sum(magnitude_spectrum**2)
    dc_energy = magnitude_spectrum[center_y, center_x]**2
    dc_percentage = 100 * dc_energy / total_energy
    print(f"DC component: {dc_percentage:.1f}% of total energy")


def run_task1(image_path):
    """
    Complete execution of Task 1: Load grayscale image and apply 2D DFT.
    
    Args:
        image_path (str): Path to the input grayscale image
        
    Returns:
        tuple: (original_image, magnitude_spectrum, phase_spectrum, fft_result)
    """
    # Step 1: Load grayscale image
    original_image = load_grayscale_image(image_path)
    
    # Step 2: Apply 2D DFT
    magnitude_spectrum, phase_spectrum, fft_result = apply_2d_dft(original_image)
    
    # Step 3: Print concise analysis
    #print_dft_analysis(magnitude_spectrum, phase_spectrum)
    
    # Step 4: Create 3D surface plot from the original image
    image_3d = create_3d_surface_plot_from_image(original_image)
    
    # Step 5: Visualize results with 3D surface plot
    visualize_dft_results(original_image, magnitude_spectrum, image_3d)
    
    return original_image, magnitude_spectrum, phase_spectrum, fft_result
