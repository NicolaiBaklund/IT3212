import numpy as np
import matplotlib.pyplot as plt
from task1 import load_grayscale_image, apply_2d_dft
from reconstruct_image import reconstruct_from_fft


def compress_fourier_coefficients(fft_result, keep_percentage):
    """
    Compress Fourier coefficients by keeping only the largest magnitude coefficients.
    
    Args:
        fft_result (numpy.ndarray): Complex FFT result
        keep_percentage (float): Percentage of coefficients to keep (0.0 to 1.0)
        
    Returns:
        numpy.ndarray: Compressed FFT result with small coefficients zeroed out
    """
    # Get magnitudes of all coefficients
    magnitudes = np.abs(fft_result)
    
    # Flatten the magnitude array for sorting
    flat_magnitudes = magnitudes.flatten()
    
    # Sort magnitudes in ascending order
    sorted_magnitudes = np.sort(flat_magnitudes)
    
    # Calculate threshold based on keep_percentage
    num_coefficients = len(sorted_magnitudes)
    num_to_keep = int(num_coefficients * keep_percentage)
    
    if num_to_keep == 0:
        threshold = sorted_magnitudes[-1] + 1  # Keep nothing
    else:
        threshold = sorted_magnitudes[-num_to_keep]
    
    # Create mask for coefficients above threshold
    mask = magnitudes >= threshold
    
    # Apply mask to FFT result
    compressed_fft = fft_result * mask
    
    return compressed_fft, mask, threshold


def apply_compression_and_visualize(original_image, keep_percentage):
    """
    Apply Fourier coefficient compression to an image and visualize the results.
    
    Args:
        original_image (numpy.ndarray): Original grayscale image
        keep_percentage (float): Percentage of coefficients to keep (0.0 to 1.0)
        
    Returns:
        tuple: (compressed_image, fft_result, compression_mask, threshold)
    """
    # Apply 2D DFT to get frequency domain representation
    magnitude_spectrum, phase_spectrum, fft_result = apply_2d_dft(original_image)
    
    # Apply compression
    compressed_fft, compression_mask, threshold = compress_fourier_coefficients(fft_result, keep_percentage)
    
    # Calculate compressed magnitude spectrum for visualization
    compressed_magnitude = np.abs(compressed_fft)
    
    # Reconstruct the compressed image
    compressed_image = reconstruct_from_fft(compressed_fft)
    
    # Create visualization with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image (left)
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Compressed frequency spectrum (middle)
    compressed_magnitude_log = np.log(compressed_magnitude + 1)  # Add 1 to avoid log(0)
    im = axes[1].imshow(compressed_magnitude_log, cmap='hot')
    axes[1].set_title(f'Compressed Frequency Spectrum\n(Keep: {keep_percentage*100:.1f}%)')
    axes[1].axis('off')
    
    # Add colorbar for magnitude spectrum
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Reconstructed compressed image (right)
    axes[2].imshow(compressed_image, cmap='gray')
    axes[2].set_title(f'Compressed Image\n({keep_percentage*100:.1f}% coefficients)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print information about the compression
    print(f"Compression applied with {keep_percentage*100:.1f}% of coefficients kept")
    print(f"Compression mask shape: {compression_mask.shape}")
    print(f"Percentage of coefficients kept: {np.sum(compression_mask) / compression_mask.size * 100:.1f}%")
    print(f"Threshold magnitude: {threshold:.2f}")
    
    return compressed_image, fft_result, compression_mask, threshold


def run_task4(image_path, keep_percentage=0.5):
    """
    Complete execution of Task 4: Apply Fourier coefficient compression to an image.
    
    Args:
        image_path (str): Path to the input grayscale image
        keep_percentage (float): Percentage of coefficients to keep (default: 0.5)
        
    Returns:
        tuple: (original_image, compressed_image, fft_result, compression_mask, threshold)
    """
    print(f"Running Task 4 with image: {image_path}")
    print(f"Keep percentage: {keep_percentage*100:.1f}%")
    print("-" * 50)
    
    # Load the original image
    original_image = load_grayscale_image(image_path)
    
    # Apply compression and visualize
    compressed_image, fft_result, compression_mask, threshold = apply_compression_and_visualize(
        original_image, keep_percentage
    )
    
    return original_image, compressed_image, fft_result, compression_mask, threshold


def calculate_mse_percentage(original_image, compressed_image):
    """
    Calculate the MSE between original and compressed images as a percentage.
    
    Args:
        original_image (numpy.ndarray): Original image
        compressed_image (numpy.ndarray): Compressed image
        
    Returns:
        float: MSE as a percentage of the maximum possible error
    """
    # Calculate MSE
    mse = np.mean((original_image.astype(float) - compressed_image.astype(float)) ** 2)
    
    # Calculate maximum possible MSE (when all pixels are maximally different)
    # For 8-bit images, max difference per pixel is 255, so max MSE is 255^2
    
    # Convert to percentage
    mse_percentage = mse
    return mse_percentage


def calculate_mse(imageA, imageB):
    """
    Beregner den "rå" Mean Squared Error mellom to bilder.
    """
    # Antar at bildene er numpy arrays med samme dimensjoner
    # og normalisert (f.eks. 0-255).
    # Konverter til float for å unngå overflow-problemer ved subtraksjon
    imageA = imageA.astype(np.float64)
    imageB = imageB.astype(np.float64)
    
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]) # Dele på totalt antall piksler
    return err

# --- NY FUNKSJON ---
def calculate_rmse(imageA, imageB):
    """
    Beregner Root Mean Squared Error (RMSE) mellom to bilder.
    """
    mse = calculate_mse(imageA, imageB)
    return np.sqrt(mse)
# --- SLUTT NY FUNKSJON ---


# --- OPPDATERT FUNKSJONSDEFINISJON ---
def test_different_compression_ratios(image_path, 
                                      compression_ratios=[0.1, 0.2, 0.5, 0.8], 
                                      num_metric_points=50):
    """
    Test Fourier coefficient compression with different compression ratios for comparison.
    
    Args:
        image_path (str): Path to the input grayscale image
        compression_ratios (list): List of compression ratios for VISUAL examples.
        num_metric_points (int): Number of data points to calculate for the RMSE plot.
    """
    print(f"Testing {len(compression_ratios)} visual compression ratios...")
    
    # Load the original image
    original_image = load_grayscale_image(image_path)
    
    # Apply DFT to get original spectrum
    magnitude_spectrum, phase_spectrum, fft_result = apply_2d_dft(original_image)
    
    # --- Denne delen er uendret og bygger det visuelle plottet ---
    # Create subplot grid
    n_ratios = len(compression_ratios) + 1  # +1 for original
    fig, axes = plt.subplots(2, n_ratios, figsize=(4*n_ratios, 8))
    
    # First plot: Original image and spectrum (no compression)
    original_magnitude_log = np.log(magnitude_spectrum + 1)
    axes[0, 0].imshow(original_magnitude_log, cmap='hot')
    axes[0, 0].set_title('Original Spectrum')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(original_image, cmap='gray')
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    # Store MSE values for each *visual* compression ratio
    mse_percentage_values = [] # For bilde-titlene
    raw_mse_values = []        # For print-seksjonen
    
    # Plot compressed results
    for i, keep_ratio in enumerate(compression_ratios):
        # Apply compression
        compressed_fft, compression_mask, threshold = compress_fourier_coefficients(fft_result, keep_ratio)
        compressed_magnitude = np.abs(compressed_fft)
        compressed_image = reconstruct_from_fft(compressed_fft)
        
        # Beregn begge MSE-verdiene
        mse_percentage = calculate_mse_percentage(original_image, compressed_image)
        raw_mse = calculate_mse(original_image, compressed_image)
        
        mse_percentage_values.append(mse_percentage)
        raw_mse_values.append(raw_mse)
        
        # Plot compressed magnitude spectrum
        compressed_magnitude_log = np.log(compressed_magnitude + 1)
        axes[0, i+1].imshow(compressed_magnitude_log, cmap='hot')
        axes[0, i+1].set_title(f'Keep: {keep_ratio*100:.0f}%')
        axes[0, i+1].axis('off')
        
        # Plot reconstructed compressed image
        axes[1, i+1].imshow(compressed_image, cmap='gray')
        axes[1, i+1].set_title(f'Compressed Image\nMSE: {mse_percentage:.2f}%')
        axes[1, i+1].axis('off')
    
    plt.suptitle('Fourier Coefficient Compression with Different Ratios', fontsize=16)
    plt.tight_layout()
    
    output_path = "../../data/fourier_compression_comparison.png"
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    except Exception as e:
        print(f"Could not save comparison plot: {e}")

    # Display the comparison plot
    plt.show()
    
    # --- NY SEKSJON: Beregn data for høyoppløselig RMSE-plott ---
    
    print(f"\nCalculating {num_metric_points} data points for RMSE plot...")
    
    # Bruk linspace for å få jevne punkter (unngå 0.0, start litt over)
    metric_keep_ratios = np.linspace(0.001, 1.0, num_metric_points)
    plot_cutoff_percentages = (1.0 - metric_keep_ratios) * 100
    
    # Beholdere for resultater
    plot_rmse_values = []
    
    for keep_ratio in metric_keep_ratios:
        # Trenger ikke lagre hele fft-resultatet, bare bildet
        compressed_fft, _, _ = compress_fourier_coefficients(fft_result, keep_ratio)
        compressed_image = reconstruct_from_fft(compressed_fft)
        
        # Beregn RMSE
        rmse = calculate_rmse(original_image, compressed_image)
        plot_rmse_values.append(rmse)

    print("...Calculation complete.")

    # --- ENDRINGER I PLOTTE-SEKSJON ---
    
    # Lag et nytt plot for RMSE vs. Avkuttede komponenter
    plt.figure(figsize=(8, 5)) # Lag en ny figur
    
    # Plotter RMSE (plot_rmse_values)
    plt.plot(plot_cutoff_percentages, plot_rmse_values, marker='o', markersize=2, linestyle='--')
    
    # Sett titler og etiketter
    plt.title('RMSE vs. Percentage of Cutoff Components')
    plt.xlabel('Percentage of Cutoff Components (%)')
    plt.ylabel('Root Mean Squared Error (RMSE)') # Endret etikett
    
    plt.gca().invert_xaxis() 
    plt.grid(True)
    plt.tight_layout()
    
    # Lagre RMSE-plottet til en fil
    rmse_plot_path = "../../data/fourier_rmse_vs_cutoff.png" # Nytt filnavn
    try:
        plt.savefig(rmse_plot_path, dpi=150, bbox_inches='tight')
        print(f"RMSE plot saved to: {rmse_plot_path}")
    except Exception as e:
        print(f"Could not save RMSE plot: {e}")

    # Vis RMSE-plottet
    plt.show()
    
    # --- ENDRING I PRINT-SEKSJON ---

    # Print MSE/RMSE analysis for de visuelle eksemplene
    print("\nMSE/RMSE Analysis (for visual examples):")
    print("-" * 75)
    print(f"{'Keep %':<10} {'Raw MSE':<12} {'RMSE':<12} {'Pct MSE (%)':<15}")
    print("-" * 75)
    
    # Oppdatert zip og print-statement
    for keep_ratio, mse_pct, raw_mse in zip(compression_ratios, mse_percentage_values, raw_mse_values):
        rmse = np.sqrt(raw_mse) # Beregn RMSE fra den lagrede rå MSE
        print(f"{keep_ratio*100:<10.0f} {raw_mse:<12.2f} {rmse:<12.2f} {mse_pct:<15.2f}")
    
    # Returnerer listene fra de visuelle eksemplene
    return raw_mse_values, mse_percentage_values

def calculate_compression_metrics(original_image, compressed_image, compression_mask):
    """
    Calculate compression metrics including compression ratio and reconstruction quality.
    
    Args:
        original_image (numpy.ndarray): Original image
        compressed_image (numpy.ndarray): Compressed image
        compression_mask (numpy.ndarray): Mask showing which coefficients were kept
        
    Returns:
        dict: Dictionary containing compression metrics
    """
    # Calculate compression ratio
    total_coefficients = compression_mask.size
    kept_coefficients = np.sum(compression_mask)
    compression_ratio = kept_coefficients / total_coefficients
    
    # Calculate reconstruction quality metrics
    mse = np.mean((original_image.astype(float) - compressed_image.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Calculate energy retention
    original_energy = np.sum(original_image.astype(float) ** 2)
    compressed_energy = np.sum(compressed_image.astype(float) ** 2)
    energy_retention = compressed_energy / original_energy if original_energy > 0 else 0
    
    metrics = {
        'compression_ratio': compression_ratio,
        'coefficients_kept': int(kept_coefficients),
        'coefficients_total': int(total_coefficients),
        'mse': mse,
        'psnr': psnr,
        'energy_retention': energy_retention
    }
    
    return metrics


def print_compression_analysis(metrics):
    """
    Print detailed analysis of compression results.
    
    Args:
        metrics (dict): Compression metrics dictionary
    """
    print(f"\nCompression Analysis:")
    print(f"  Coefficients kept: {metrics['coefficients_kept']}/{metrics['coefficients_total']} ({metrics['compression_ratio']*100:.1f}%)")
    print(f"  MSE: {metrics['mse']:.2f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  Energy retention: {metrics['energy_retention']*100:.1f}%")


if __name__ == "__main__":
    # Example usage
    image_path = "../../data/BilderFourier/4.jpg"
    
    # Run Task 4 with 50% compression
    print("=== Task 4: Fourier Coefficient Compression ===")
    original, compressed, fft, mask, threshold = run_task4(image_path, keep_percentage=0.5)
    
    # Calculate and print compression metrics
    metrics = calculate_compression_metrics(original, compressed, mask)
    print_compression_analysis(metrics)
    
    # Test different compression ratios
    print("\n=== Testing Different Compression Ratios ===")
    test_different_compression_ratios(image_path, [0.05, 0.1, 0.2, 0.5, 0.8])
