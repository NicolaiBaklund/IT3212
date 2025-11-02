import numpy as np
import matplotlib.pyplot as plt
from task1 import load_grayscale_image, apply_2d_dft
from reconstruct_image import reconstruct_from_fft


def create_lowpass_filter_mask(image_shape, cutoff_frequency):
    """
    Create a low-pass filter mask with a circular region around the center.
    
    Args:
        image_shape (tuple): Shape of the image (height, width)
        cutoff_frequency (float): Radius of the circular region (cutoff frequency)
        
    Returns:
        numpy.ndarray: Binary mask with ones in the circular region and zeros elsewhere
    """
    height, width = image_shape
    center_y, center_x = height // 2, width // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create binary mask: 1 if distance <= cutoff_frequency, 0 otherwise
    mask = (distance <= cutoff_frequency).astype(np.float32)
    
    return mask


def create_highpass_filter_mask(image_shape, cutoff_frequency):
    """
    Create a high-pass filter mask with a circular region around the center.
    
    Args:
        image_shape (tuple): Shape of the image (height, width)
        cutoff_frequency (float): Radius of the circular region (cutoff frequency)
        
    Returns:
        numpy.ndarray: Binary mask with zeros in the circular region and ones elsewhere
    """
    height, width = image_shape
    center_y, center_x = height // 2, width // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create binary mask: 0 if distance <= cutoff_frequency, 1 otherwise
    mask = (distance > cutoff_frequency).astype(np.float32)
    
    return mask


def apply_filter_and_visualize(original_image, cutoff_frequency, high_pass_filter=False):
    """
    Apply low-pass or high-pass filter to an image and visualize the results.
    
    Args:
        original_image (numpy.ndarray): Original grayscale image
        cutoff_frequency (float): Cutoff frequency for the filter
        high_pass_filter (bool): If True, apply high-pass filter; if False, apply low-pass filter
        
    Returns:
        tuple: (filtered_image, fft_result, filter_mask)
    """
    # Apply 2D DFT to get frequency domain representation
    magnitude_spectrum, phase_spectrum, fft_result = apply_2d_dft(original_image)
    
    # Create filter mask based on filter type
    if high_pass_filter:
        filter_mask = create_highpass_filter_mask(original_image.shape, cutoff_frequency)
        filter_type = "High-pass"
    else:
        filter_mask = create_lowpass_filter_mask(original_image.shape, cutoff_frequency)
        filter_type = "Low-pass"
    
    # Apply the filter to the FFT result
    filtered_fft = fft_result * filter_mask
    
    # Calculate filtered magnitude spectrum for visualization
    filtered_magnitude = np.abs(filtered_fft)
    
    # Reconstruct the filtered image
    filtered_image = reconstruct_from_fft(filtered_fft)
    
    # Create visualization with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image (left)
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Filtered frequency spectrum (middle) - using same style as task1.py lines 90-95
    filtered_magnitude_log = np.log(filtered_magnitude + 1)  # Add 1 to avoid log(0)
    im = axes[1].imshow(filtered_magnitude_log, cmap='hot')
    axes[1].set_title(f'{filter_type} Frequency Spectrum\n(Cutoff: {cutoff_frequency})')
    axes[1].axis('off')
    
    # Add colorbar for magnitude spectrum
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Reconstructed filtered image (right)
    axes[2].imshow(filtered_image, cmap='gray')
    axes[2].set_title(f'Reconstructed {filter_type} Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print information about the filtering
    print(f"{filter_type} filter applied with cutoff frequency: {cutoff_frequency}")
    print(f"Filter mask shape: {filter_mask.shape}")
    print(f"Percentage of frequencies passed: {np.sum(filter_mask) / filter_mask.size * 100:.1f}%")
    
    return filtered_image, fft_result, filter_mask


def run_task2(image_path, cutoff_frequency=30):
    """
    Complete execution of Task 2: Apply low-pass filter to an image and visualize results.
    
    Args:
        image_path (str): Path to the input grayscale image
        cutoff_frequency (float): Cutoff frequency for the low-pass filter (default: 30)
        
    Returns:
        tuple: (original_image, filtered_image, fft_result, filter_mask)
    """
    print(f"Running Task 2 with image: {image_path}")
    print(f"Cutoff frequency: {cutoff_frequency}")
    print("-" * 50)
    
    # Load the original image
    original_image = load_grayscale_image(image_path)
    
    # Apply low-pass filter and visualize (high_pass_filter=False)
    filtered_image, fft_result, filter_mask = apply_filter_and_visualize(
        original_image, cutoff_frequency, high_pass_filter=False
    )
    
    return original_image, filtered_image, fft_result, filter_mask


def run_task3(image_path, cutoff_frequency=30):
    """
    Complete execution of Task 3: Apply high-pass filter to an image and visualize results.
    
    Args:
        image_path (str): Path to the input grayscale image
        cutoff_frequency (float): Cutoff frequency for the high-pass filter (default: 30)
        
    Returns:
        tuple: (original_image, filtered_image, fft_result, filter_mask)
    """
    print(f"Running Task 3 with image: {image_path}")
    print(f"Cutoff frequency: {cutoff_frequency}")
    print("-" * 50)
    
    # Load the original image
    original_image = load_grayscale_image(image_path)
    
    # Apply high-pass filter and visualize (high_pass_filter=True)
    filtered_image, fft_result, filter_mask = apply_filter_and_visualize(
        original_image, cutoff_frequency, high_pass_filter=True
    )
    
    return original_image, filtered_image, fft_result, filter_mask


def add_images(image1, image2, scale_factor=1.0):
    """
    Add two images together with optional scaling.
    
    Args:
        image1 (numpy.ndarray): First image
        image2 (numpy.ndarray): Second image to add (will be scaled)
        scale_factor (float): Scaling factor for the second image (default: 1.0)
        
    Returns:
        numpy.ndarray: Sum of the images with proper clipping to [0, 255]
    """
    # Scale the second image
    scaled_image2 = image2 * scale_factor
    
    # Add the images
    result = image1 + scaled_image2
    
    # Clip values to valid range [0, 255] and convert to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def test_different_cutoff_frequencies(image_path, cutoff_frequencies=[10, 30, 50, 80], high_pass_filter=False):
    """
    Test the low-pass or high-pass filter with different cutoff frequencies for comparison.
    
    Args:
        image_path (str): Path to the input grayscale image
        cutoff_frequencies (list): List of cutoff frequencies to test
        high_pass_filter (bool): If True, test high-pass filter; if False, test low-pass filter
    """
    filter_type = "High-pass" if high_pass_filter else "Low-pass"
    print(f"Testing different cutoff frequencies for {filter_type} filter...")
    
    # Load the original image
    original_image = load_grayscale_image(image_path)
    
    # Apply DFT to get original spectrum
    magnitude_spectrum, phase_spectrum, fft_result = apply_2d_dft(original_image)
    
    # Create subplot grid (add 1 for original, no filter)
    # For high-pass filter, we'll show 3 rows: spectrum, filtered image, and edge enhanced image
    n_frequencies = len(cutoff_frequencies) + 1
    n_rows = 3 if high_pass_filter else 2
    fig, axes = plt.subplots(n_rows, n_frequencies, figsize=(4*n_frequencies, 3*n_rows))
    
    # First plot: Original image and spectrum (no filter)
    # Original spectrum (top)
    original_magnitude_log = np.log(magnitude_spectrum + 1)
    axes[0, 0].imshow(original_magnitude_log, cmap='hot')
    axes[0, 0].set_title('Original Spectrum')
    axes[0, 0].axis('off')

    # Original image (second row)
    axes[1, 0].imshow(original_image, cmap='gray')
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    # For high-pass filter, show original image again in third row (no edge enhancement for original)
    if high_pass_filter:
        axes[2, 0].imshow(original_image, cmap='gray')
        axes[2, 0].set_title('Original (No Enhancement)')
        axes[2, 0].axis('off')
    
    
    # Plot filtered results
    for i, cutoff_freq in enumerate(cutoff_frequencies):
        # Apply filter based on type
        if high_pass_filter:
            filter_mask = create_highpass_filter_mask(original_image.shape, cutoff_freq)
        else:
            filter_mask = create_lowpass_filter_mask(original_image.shape, cutoff_freq)
            
        filtered_fft = fft_result * filter_mask
        filtered_magnitude = np.abs(filtered_fft)
        filtered_image = reconstruct_from_fft(filtered_fft)
        
        # Plot filtered magnitude spectrum
        filtered_magnitude_log = np.log(filtered_magnitude + 1)
        axes[0, i+1].imshow(filtered_magnitude_log, cmap='hot')
        axes[0, i+1].set_title(f'Cutoff: {cutoff_freq}')
        axes[0, i+1].axis('off')
        
        # Plot reconstructed image
        axes[1, i+1].imshow(filtered_image, cmap='gray')
        axes[1, i+1].set_title(f'Reconstructed')
        axes[1, i+1].axis('off')
        
        # For high-pass filter, create and plot edge enhanced image
        if high_pass_filter:
            # Scale the high-pass filtered result before adding to original
            # Use a moderate scaling factor to enhance edges without over-saturating
            scale_factor = 1.5  # Adjust this value to control edge enhancement strength
            edge_enhanced = add_images(original_image, filtered_image, scale_factor)
            
            axes[2, i+1].imshow(edge_enhanced, cmap='gray')
            axes[2, i+1].set_title(f'Edge Enhanced\n(scale: {scale_factor})')
            axes[2, i+1].axis('off')
    
    plt.suptitle(f'{filter_type} Filter with Different Cutoff Frequencies', fontsize=16)
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
    # Also save the plot to a file for environments where display might not work
    output_path = f"../../data/{filter_type.lower()}_filter_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    image_path = "../../data/BilderFourier/4.jpg"
    
    # Run Task 2 with low-pass filter
    print("=== Task 2: Low-pass Filter ===")
    original, filtered, fft, mask = run_task2(image_path, cutoff_frequency=30)
    
    # Run Task 3 with high-pass filter
    print("\n=== Task 3: High-pass Filter ===")
    original, filtered, fft, mask = run_task3(image_path, cutoff_frequency=30)
    
    # Test different cutoff frequencies for low-pass filter
    print("\n=== Testing Different Cutoff Frequencies (Low-pass) ===")
    test_different_cutoff_frequencies(image_path, [10, 30, 50, 80], high_pass_filter=False)
    
    # Test different cutoff frequencies for high-pass filter
    print("\n=== Testing Different Cutoff Frequencies (High-pass) ===")
    test_different_cutoff_frequencies(image_path, [10, 30, 50, 80], high_pass_filter=True)


