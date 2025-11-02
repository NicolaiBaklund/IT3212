import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def create_grating_pattern(width=200, height=200, frequency=10, angle=0, contrast=255):
    """
    Create a grating pattern (parallel lines) in grayscale.
    
    Args:
        width (int): Image width
        height (int): Image height
        frequency (float): Spatial frequency of the grating
        angle (float): Angle of the grating in degrees
        contrast (int): Maximum contrast (0-255)
        
    Returns:
        numpy.ndarray: Grayscale grating pattern
    """
    print(f"Creating grating pattern: {width}x{height}, freq={frequency}, angle={angle}°")
    
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Create grating pattern
    # Rotate coordinates
    X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    
    # Create sinusoidal grating
    grating = np.sin(2 * np.pi * frequency * X / width)
    
    # Convert to 0-255 range
    grating = ((grating + 1) / 2 * contrast).astype(np.uint8)
    
    return grating


def create_circle_pattern(width=200, height=200, center=None, radius=50, fill_value=255, background_value=0):
    """
    Create a circle pattern in grayscale.
    
    Args:
        width (int): Image width
        height (int): Image height
        center (tuple): Center coordinates (x, y), if None uses image center
        radius (int): Circle radius
        fill_value (int): Circle fill value (0-255)
        background_value (int): Background value (0-255)
        
    Returns:
        numpy.ndarray: Grayscale circle pattern
    """
    print(f"Creating circle pattern: {width}x{height}, radius={radius}")
    
    # Create image with background
    image = np.full((height, width), background_value, dtype=np.uint8)
    
    # Set center if not provided
    if center is None:
        center = (width // 2, height // 2)
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Create circle mask
    circle_mask = distance <= radius
    
    # Fill circle
    image[circle_mask] = fill_value
    
    return image


def create_rectangle_pattern(width=200, height=200, rect_center=None, rect_width=80, rect_height=60, 
                           fill_value=255, background_value=0):
    """
    Create a rectangle pattern in grayscale.
    
    Args:
        width (int): Image width
        height (int): Image height
        rect_center (tuple): Rectangle center coordinates (x, y), if None uses image center
        rect_width (int): Rectangle width
        rect_height (int): Rectangle height
        fill_value (int): Rectangle fill value (0-255)
        background_value (int): Background value (0-255)
        
    Returns:
        numpy.ndarray: Grayscale rectangle pattern
    """
    print(f"Creating rectangle pattern: {width}x{height}, rect={rect_width}x{rect_height}")
    
    # Create image with background
    image = np.full((height, width), background_value, dtype=np.uint8)
    
    # Set center if not provided
    if rect_center is None:
        rect_center = (width // 2, height // 2)
    
    # Calculate rectangle bounds
    x1 = max(0, rect_center[0] - rect_width // 2)
    x2 = min(width, rect_center[0] + rect_width // 2)
    y1 = max(0, rect_center[1] - rect_height // 2)
    y2 = min(height, rect_center[1] + rect_height // 2)
    
    # Fill rectangle
    image[y1:y2, x1:x2] = fill_value
    
    return image


def create_noisy_dots_pattern(width=200, height=200, num_dots=10000, dot_size_range=(0, 0), 
                             dot_value=255, background_value=0, noise_level=0.8):
    """
    Create a pattern with noisy dots scattered across the image.
    
    Args:
        width (int): Image width
        height (int): Image height
        num_dots (int): Number of dots to create
        dot_size_range (tuple): Range of dot sizes (min, max)
        dot_value (int): Dot value (0-255)
        background_value (int): Background value (0-255)
        noise_level (float): Amount of random noise (0-1)
        
    Returns:
        numpy.ndarray: Grayscale noisy dots pattern
    """
    print(f"Creating noisy dots pattern: {width}x{height}, {num_dots} dots")
    
    # Create image with background
    image = np.full((height, width), background_value, dtype=np.uint8)
    
    # Add random noise to background
    noise = np.random.normal(0, noise_level * 255, (height, width))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Generate random dot positions and sizes
    """np.random.seed(42)  # For reproducible results
    x_positions = np.random.randint(0, width, num_dots)
    y_positions = np.random.randint(0, height, num_dots)
    dot_sizes = np.random.randint(dot_size_range[0], dot_size_range[1] + 1, num_dots)
    
    # Create dots
    for i in range(num_dots):
        x, y, size = x_positions[i], y_positions[i], dot_sizes[i]
        
        # Create circular dot
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        
        # Add dot
        dot_mask = distance <= size
        image[dot_mask] = dot_value
    """
    return image


def create_rectangle_over_noisy_dots(width=200, height=200, rect_center=None, rect_width=100, rect_height=80,
                                   num_dots=150, dot_size_range=(3, 10), noise_level=0.8):
    """
    Create a pattern with noisy dots and overlay a white rectangle on top.
    
    Args:
        width (int): Image width
        height (int): Image height
        rect_center (tuple): Rectangle center coordinates (x, y), if None uses image center
        rect_width (int): Rectangle width
        rect_height (int): Rectangle height
        num_dots (int): Number of dots to create in the noisy pattern
        dot_size_range (tuple): Range of dot sizes (min, max)
        noise_level (float): Amount of random noise (0-1)
        
    Returns:
        numpy.ndarray: Grayscale image with rectangle overlaid on noisy dots
    """
    print(f"Creating rectangle over noisy dots: {width}x{height}, rect={rect_width}x{rect_height}")
    
    # First create the noisy dots pattern
    noisy_image = create_noisy_dots_pattern(width, height, num_dots, dot_size_range, 
                                          dot_value=255, background_value=0, noise_level=noise_level)
    
    # Create the rectangle pattern
    rectangle = create_rectangle_pattern(width, height, rect_center, rect_width, rect_height,
                                       fill_value=255, background_value=0)
    
    # Overlay rectangle on noisy image
    # Where rectangle is white (255), use rectangle value
    # Where rectangle is black (0), keep the noisy image
    result = np.where(rectangle == 255, rectangle, noisy_image)
    
    return result


def generate_all_patterns(output_dir="data/FourierGen", image_size=200):
    """
    Generate all synthetic patterns and save them to files.
    
    Args:
        output_dir (str): Directory to save generated images
        image_size (int): Size of generated images (width=height)
        
    Returns:
        list: List of generated image file paths
    """
    print(f"Generating synthetic images in {output_dir}/")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # 1. Grating pattern (horizontal)
    grating_h = create_grating_pattern(image_size, image_size, frequency=8, angle=0)
    grating_path = os.path.join(output_dir, "grating_horizontal.jpg")
    cv2.imwrite(grating_path, grating_h)
    generated_files.append(grating_path)
    print(f"Saved: {grating_path}")
    
    # 2. Grating pattern (high frequency diagonal)
    grating_d = create_grating_pattern(image_size, image_size, frequency=30, angle=45)
    grating_high_freq_diag_path = os.path.join(output_dir, "grating_high_freq_diagonal.jpg")
    cv2.imwrite(grating_high_freq_diag_path, grating_d)
    generated_files.append(grating_high_freq_diag_path)
    print(f"Saved: {grating_high_freq_diag_path}")
    
    # 3. Circle pattern
    circle = create_circle_pattern(image_size, image_size, radius=60)
    circle_path = os.path.join(output_dir, "circle.jpg")
    cv2.imwrite(circle_path, circle)
    generated_files.append(circle_path)
    print(f"Saved: {circle_path}")
    
    # 4. Rectangle pattern
    rectangle = create_rectangle_pattern(image_size, image_size, rect_width=100, rect_height=80)
    rectangle_path = os.path.join(output_dir, "rectangle.jpg")
    cv2.imwrite(rectangle_path, rectangle)
    generated_files.append(rectangle_path)
    print(f"Saved: {rectangle_path}")
    
    # 5. Noisy dots pattern
    dots = create_noisy_dots_pattern(image_size, image_size, num_dots=150, dot_size_range=(3, 10))
    dots_path = os.path.join(output_dir, "noisy_dots.jpg")
    cv2.imwrite(dots_path, dots)
    generated_files.append(dots_path)
    print(f"Saved: {dots_path}")
    
    # 6. Rectangle over noisy dots pattern
    rect_over_dots = create_rectangle_over_noisy_dots(image_size, image_size, 
                                                    rect_width=100, rect_height=80,
                                                    num_dots=150, dot_size_range=(3, 10))
    rect_over_dots_path = os.path.join(output_dir, "rectangle_over_noisy_dots.jpg")
    cv2.imwrite(rect_over_dots_path, rect_over_dots)
    generated_files.append(rect_over_dots_path)
    print(f"Saved: {rect_over_dots_path}")
    
    print(f"\nGenerated {len(generated_files)} synthetic images!")
    return generated_files


def display_generated_images(image_paths, title="Generated Synthetic Images"):
    """
    Display all generated images in a grid.
    
    Args:
        image_paths (list): List of image file paths
        title (str): Title for the display
    """
    print(f"Displaying {len(image_paths)} generated images...")
    
    # Create subplot grid
    n_images = len(image_paths)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for i, image_path in enumerate(image_paths):
        if i < len(axes):
            # Load and display image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(os.path.basename(image_path))
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'Could not load\n{os.path.basename(image_path)}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Error: {os.path.basename(image_path)}')
                axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def run_image_generation():
    """
    Main function to generate and display all synthetic patterns.
    """
    print("SYNTHETIC IMAGE GENERATION")
    print("="*50)
    
    # Generate all patterns
    generated_files = generate_all_patterns()
    
    # Display the generated images
    display_generated_images(generated_files)
    
    print("\nImage generation completed successfully!")
    return generated_files


if __name__ == "__main__":
    # Run the image generation when script is executed directly
    run_image_generation()
