import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def process_image(image_path, intensity_coefficient=0.5, blur_coefficient=5):
    """
    Process an image by reducing its intensity and applying blur.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    intensity_coefficient : float, optional (default=0.5)
        Coefficient to reduce image intensity (0.0 = black, 1.0 = original)
    blur_coefficient : int, optional (default=5)
        Kernel size for Gaussian blur (must be odd number, higher = more blur)
    
    Returns:
    --------
    tuple
        (processed_image, original_image) as numpy arrays
    """
    
    # Load the image
    try:
        # Try loading with OpenCV first (BGR format)
        original_image = cv2.imread(image_path)
        if original_image is None:
            # If OpenCV fails, try with PIL
            pil_image = Image.open(image_path)
            original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Could not load image from {image_path}: {str(e)}")
    
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to float for processing
    processed_image = original_image.astype(np.float32)
    
    # Reduce intensity by multiplying with coefficient
    processed_image = processed_image * intensity_coefficient
    
    # Ensure values are within valid range [0, 255]
    processed_image = np.clip(processed_image, 0, 255)
    
    # Apply Gaussian blur
    # Ensure blur_coefficient is odd
    if blur_coefficient % 2 == 0:
        blur_coefficient += 1
    
    # Apply Gaussian blur
    processed_image = cv2.GaussianBlur(processed_image, (blur_coefficient, blur_coefficient), 0)
    
    # Convert back to uint8
    processed_image = processed_image.astype(np.uint8)
    
    return processed_image, original_image


def display_comparison(original_image, processed_image, title="Image Processing Comparison"):
    """
    Display original and processed images side by side.
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original image array
    processed_image : numpy.ndarray
        Processed image array
    title : str, optional
        Title for the plot
    """
    # Convert BGR to RGB for matplotlib display
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(processed_rgb)
    axes[1].set_title('Processed Image')
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_processed_image(processed_image, output_path):
    """
    Save the processed image to a file.
    
    Parameters:
    -----------
    processed_image : numpy.ndarray
        Processed image array
    output_path : str
        Path where to save the processed image
    """
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to: {output_path}")


