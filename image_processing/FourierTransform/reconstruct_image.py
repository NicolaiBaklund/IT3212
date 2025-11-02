import numpy as np
import matplotlib.pyplot as plt


def reconstruct_from_fft(fft_object):
    """
    Reconstruct a grayscale image from an FFT object using inverse DFT.
    
    Args:
        fft_object (numpy.ndarray): Complex FFT result (shifted or unshifted)
        
    Returns:
        numpy.ndarray: Reconstructed grayscale image (uint8, 0-255)
    """
    # Check if input is valid
    if not isinstance(fft_object, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if not np.iscomplexobj(fft_object):
        raise ValueError("Input must be a complex array (FFT result)")
    
    # Detect if FFT is shifted by checking if maximum magnitude is at center
    center_y, center_x = fft_object.shape[0] // 2, fft_object.shape[1] // 2
    magnitude = np.abs(fft_object)
    max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    is_shifted = (max_idx[0] == center_y and max_idx[1] == center_x)
    
    # Unshift if necessary
    if is_shifted:
        fft_unshifted = np.fft.ifftshift(fft_object)
    else:
        fft_unshifted = fft_object
    
    # Apply inverse 2D FFT
    reconstructed = np.fft.ifft2(fft_unshifted)
    
    # Take real part and normalize to 0-255 range
    reconstructed_real = np.real(reconstructed)
    reconstructed_real = np.maximum(reconstructed_real, 0)  # Ensure non-negative
    
    # Normalize to 0-255 range
    if reconstructed_real.max() > reconstructed_real.min():
        reconstructed_normalized = ((reconstructed_real - reconstructed_real.min()) / 
                                   (reconstructed_real.max() - reconstructed_real.min()) * 255)
    else:
        reconstructed_normalized = np.zeros_like(reconstructed_real)
    
    return reconstructed_normalized.astype(np.uint8)


def plot_image(image, title="Image", figsize=(6, 5)):
    """
    Plot a grayscale image.
    
    Args:
        image (numpy.ndarray): Grayscale image to plot
        title (str): Title for the plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



