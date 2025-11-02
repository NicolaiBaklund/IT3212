# Intel Image Classification Dataset - Contour Detection Pipeline

This project implements a comprehensive contour detection pipeline for the Intel Image Classification dataset using OpenCV and Python.

## Features

- **Configurable Parameters**: Easy-to-modify configuration section for experimentation
- **Multi-Category Support**: Processes images from all Intel dataset categories (buildings, forest, glacier, mountain, sea, street)
- **Random Color Visualization**: Each detected contour is drawn with a unique random color
- **Side-by-Side Comparison**: Displays original and processed images for easy comparison
- **Comprehensive Documentation**: Detailed explanations of methodology and parameter choices

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script from the `contur` directory:

```bash
python contour_detection_pipeline.py
```

### Configuration

The script includes a comprehensive configuration section at the top. Key parameters you can adjust:

```python
config = {
    # Gaussian Blur Parameters
    'GAUSSIAN_BLUR_KERNEL_SIZE': (5, 5),  # Kernel size for noise reduction
    
    # Canny Edge Detection Parameters
    'CANNY_THRESHOLD_1': 50,              # Lower threshold for weak edges
    'CANNY_THRESHOLD_2': 150,             # Upper threshold for strong edges
    
    # Processing Parameters
    'SAMPLE_IMAGES_PER_CATEGORY': 2,      # Number of samples per category
    'INPUT_DIRECTORY': '../../../data/seg_train/seg_train',  # Dataset path
    'OUTPUT_DIRECTORY': 'output_contours',                # Results directory
}
```

### Parameter Tuning Guide

**Gaussian Blur Kernel Size:**
- `(3, 3)`: Less noise reduction, more detailed edges
- `(5, 5)`: Balanced approach (recommended)
- `(7, 7)`: More aggressive noise reduction, smoother edges

**Canny Thresholds:**
- Lower `threshold_1`: Detects more weak edges, potentially more contours
- Higher `threshold_1`: Fewer weak edges, cleaner results
- Lower `threshold_2`: More edges classified as "strong"
- Higher `threshold_2`: More selective contour detection

## Output

The script will:
1. Display side-by-side comparisons of original and processed images
2. Save contour-detected images to the `output_contours` directory
3. Print processing statistics for each image

## Methodology

The pipeline follows these steps:

1. **Image Loading**: Load color images from the dataset
2. **Grayscale Conversion**: Convert to single-channel for contour detection
3. **Gaussian Blur**: Reduce noise while preserving important edges
4. **Canny Edge Detection**: Detect edges using dual-threshold approach
5. **Contour Finding**: Extract contours from the edge map
6. **Visualization**: Draw contours with random colors on original image

## Dataset Structure

The script expects the Intel Image Classification dataset in the following structure:
```
seg_train/
├── buildings/
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/
```

## Dependencies

- `opencv-python`: Computer vision operations
- `numpy`: Numerical computations
- `matplotlib`: Image visualization
- `pathlib2`: Path handling utilities

## Troubleshooting

**"Input directory not found" error:**
- Verify the `INPUT_DIRECTORY` path in the config
- Ensure the dataset is properly extracted

**No contours detected:**
- Try lowering the Canny thresholds
- Reduce the Gaussian blur kernel size
- Check if the image has sufficient contrast

**Memory issues with large images:**
- Reduce `SAMPLE_IMAGES_PER_CATEGORY`
- Process images in smaller batches

## License

This project is for educational and research purposes.