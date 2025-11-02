import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def convert_images_to_grayscale(in_dir, out_dir):
    """
    Går gjennom alle bildefiler i 'in_dir' ved hjelp av OpenCV,
    konverterer dem til gråskala, og lagrer dem i 'out_dir'.
    """
    
    # 1. Opprett output-mappen hvis den ikke finnes
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starter konvertering (med CV2) fra: {in_dir}")
    print(f"Lagrer gråskalabilder til: {out_dir}")
    
    # 2. Gå gjennom hver fil i input-mappen
    for filename in os.listdir(in_dir):
        in_path = os.path.join(in_dir, filename)
        
        # Sjekk om stien er en fil
        if os.path.isfile(in_path):
            try:
                # 3. Les bildet direkte inn som gråskala
                # Dette er den mest effektive måten med cv2
                gray_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                
                # 4. Sjekk om bildet ble lastet riktig
                # cv2.imread returnerer 'None' hvis den feiler, istedenfor å kaste en feil
                if gray_img is not None:
                    
                    # 5. Lag den fullstendige filstien for output-bildet
                    out_path = os.path.join(out_dir, filename)
                    
                    # 6. Lagre det nye gråskalabildet med cv2
                    cv2.imwrite(out_path, gray_img)
                else:
                    # Dette skjer hvis filen ikke er et bildeformat cv2 kjenner igjen
                    print(f"Kunne ikke lese '{filename}'. Hopper over.")
                    
            except Exception as e:
                print(f"En feil oppstod med filen '{filename}'. Feil: {e}")
                
    print("\nKonvertering fullført!")


def display_grayscale_images(grayscale_dir, num_images=5):
    """
    Display grayscale images from the specified directory
    """
    # Get list of image files
    image_files = [f for f in os.listdir(grayscale_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit to num_images
    image_files = image_files[:num_images]
    
    if not image_files:
        print("No image files found in the directory")
        return
    
    # Create subplot grid
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 3))
    if len(image_files) == 1:
        axes = [axes]
    
    for i, filename in enumerate(image_files):
        # Read the grayscale image
        img_path = os.path.join(grayscale_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Grayscale: {filename}')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'Could not load\n{filename}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Error: {filename}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
