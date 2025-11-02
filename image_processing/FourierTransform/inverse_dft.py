import numpy as np
import matplotlib.pyplot as plt


def inverse_dft_to_image(fft_object):
    """
    Apply inverse DFT to an FFT object and return the reconstructed grayscale image.
    
    Args:
        fft_object (numpy.ndarray): Complex FFT result (can be shifted or unshifted)
        
    Returns:
        numpy.ndarray: Reconstructed grayscale image as numpy array
    """
    # Check if input is valid
    if not isinstance(fft_object, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if not np.iscomplexobj(fft_object):
        raise ValueError("Input must be a complex array (FFT result)")
    
    # If the FFT is shifted (zero frequency at center), we need to unshift it first
    # We can detect this by checking if the maximum magnitude is at the center
    center_y, center_x = fft_object.shape[0] // 2, fft_object.shape[1] // 2
    magnitude = np.abs(fft_object)
    
    # Check if the maximum is at the center (indicating it's shifted)
    max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    is_shifted = (max_idx[0] == center_y and max_idx[1] == center_x)
    
    # Unshift if necessary
    if is_shifted:
        fft_unshifted = np.fft.ifftshift(fft_object)
    else:
        fft_unshifted = fft_object
    
    # Apply inverse 2D FFT
    reconstructed = np.fft.ifft2(fft_unshifted)
    
    # Take the real part and ensure it's in the correct range
    reconstructed_real = np.real(reconstructed)
    
    # Normalize to 0-255 range and convert to uint8
    # First, ensure all values are non-negative
    reconstructed_real = np.maximum(reconstructed_real, 0)
    
    # Normalize to 0-255 range
    if reconstructed_real.max() > reconstructed_real.min():
        reconstructed_normalized = ((reconstructed_real - reconstructed_real.min()) / 
                                   (reconstructed_real.max() - reconstructed_real.min()) * 255)
    else:
        reconstructed_normalized = np.zeros_like(reconstructed_real)
    
    # Convert to uint8
    reconstructed_image = reconstructed_normalized.astype(np.uint8)
    
    return reconstructed_image


def plot_reconstructed_image(reconstructed_image, title="Reconstructed Image"):
    """
    Plot the reconstructed image.
    
    Args:
        reconstructed_image (numpy.ndarray): Reconstructed grayscale image
        title (str): Title for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_original_and_reconstructed(original_image, reconstructed_image):
    """
    Compare the original image with the reconstructed image side by side.
    
    Args:
        original_image (numpy.ndarray): Original grayscale image
        reconstructed_image (numpy.ndarray): Reconstructed grayscale image
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print reconstruction error
    mse = np.mean((original_image.astype(float) - reconstructed_image.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    print(f"Reconstruction Error:")
    print(f"  MSE: {mse:.2f}")
    print(f"  PSNR: {psnr:.2f} dB")


def test_inverse_dft(image_path):
    """
    Test the inverse DFT function with a sample image.
    
    Args:
        image_path (str): Path to the input image
    """
    from task1 import load_grayscale_image, apply_2d_dft
    
    # Load original image
    original_image = load_grayscale_image(image_path)
    
    # Apply forward DFT
    magnitude_spectrum, phase_spectrum, fft_result = apply_2d_dft(original_image)
    
    # Apply inverse DFT
    reconstructed_image = inverse_dft_to_image(fft_result)
    
    # Compare original and reconstructed
    compare_original_and_reconstructed(original_image, reconstructed_image)
    
    return original_image, reconstructed_image, fft_result


if __name__ == "__main__":
    # Example usage
    import os
    
    # Test with a sample image
    sample_image_path = "../../data/BilderFourier/4.jpg"
    if os.path.exists(sample_image_path):
        print("Testing inverse DFT with sample image...")
        original, reconstructed, fft = test_inverse_dft(sample_image_path)
    else:
        print(f"Sample image not found at {sample_image_path}")
        print("Please provide a valid image path to test the function.")
