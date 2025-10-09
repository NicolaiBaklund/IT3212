from alignement import align_face_bgr
import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

def crop_and_resize_face(
    aligned_img: np.ndarray,
    target_size: int,
    x_ratio: float = 0.18,
    y_ratio: float = 0.10,
    width_ratio: float = 0.64,
    height_ratio: float = 0.75
) -> np.ndarray:
    """
    Beskjærer et justert bilde for å isolere ansiktet, og endrer
    deretter størrelsen tilbake til den opprinnelige mål-størrelsen.

    Args:
        aligned_img: Det innkommende bildet, antatt å være justert og kvadratisk.
        target_size: Den ønskede output-størrelsen (bredde og høyde).
        x_ratio: Startposisjon for X som en andel av bildets bredde.
        y_ratio: Startposisjon for Y som en andel av bildets høyde.
        width_ratio: Bredden på beskjæringen som en andel av total bredde.
        height_ratio: Høyden på beskjæringen som en andel av total høyde.

    Returns:
        Det ferdig beskjærte og størrelsesendrede ansiktsbildet.
    """
    # 1. Beregn de absolutte pikselkoordinatene for beskjæringen
    x1 = int(target_size * x_ratio)
    y1 = int(target_size * y_ratio)
    x2 = int(x1 + target_size * width_ratio)
    y2 = int(y1 + target_size * height_ratio)

    # 2. Utfør selve beskjæringen
    # Bruker max(0,...) og min(target_size,...) som en sikkerhet
    cropped_face = aligned_img[
        max(0, y1):min(target_size, y2),
        max(0, x1):min(target_size, x2)
    ]
    
    # 3. Endre størrelsen tilbake til target_size for å sikre konsistent input til modellen
    resized_face = cv2.resize(
        cropped_face, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    
    return resized_face


def process_single_image(
    in_path: str,
    out_path: str,
    detector: MTCNN,
    target_size: int,
    min_conf: float,
    not_aligned=True,
    do_crop=True,
    do_grayscale=True,
    do_clahe=True,
    do_contrast_min=True
):
    """
    Leser, justerer, etterbehandler og lagrer ett enkelt bilde.
    Returnerer en feilmeldingstuppel ved feil, ellers None.
    """
    img = cv2.imread(in_path)
    if img is None:
        return (in_path, 'read_error')

    if not_aligned:
        aligned, ok, info = align_face_bgr(img, detector, target_size=target_size, min_conf=min_conf)
        if not ok:
            conf = info.get('confidence', 0)
            return (in_path, f"detect_or_align_failed(conf={conf:.2f})")
    else:
        aligned = img

    to_save = aligned

    # Beskjær og endre størrelse for å fokusere på ansiktet
    if do_crop:
        to_save = crop_and_resize_face(to_save, target_size)

    # Konverter til gråtoner
    if do_grayscale:
        to_save = cv2.cvtColor(to_save, cv2.COLOR_BGR2GRAY)

    # CLAHE
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        to_save = clahe.apply(to_save)

    # Minimere kontrast/belysningsvariasjoner
    if do_contrast_min:
        blurred = cv2.GaussianBlur(to_save, (99, 99), 0)
        to_save = cv2.divide(to_save, blurred, scale=128)

    # Lagre bildet
    cv2.imwrite(out_path, to_save)
    return None # Ingen feil

# ===================================================================
# STEG 3: HOVEDFUNKSJON SOM GÅR GJENNOM MAPPER
# Denne er den nye "dirigenten" som bruker os.walk.
# ===================================================================

def process_folder_recursively(
    in_dir: str,
    out_dir: str,
    target_size: int = 256,
    not_aligned=False,
    do_crop: bool = True,
    do_grayscale: bool = True,
    do_clahe: bool = True,
    do_contrast_min: bool = True,
    min_conf: float = 0.85
):
    """
    Går rekursivt gjennom 'in_dir', finner ansikter i bilder,
    justerer dem og lagrer dem i 'out_dir' med bevart mappestruktur.
    """
    detector = MTCNN()
    failed_files = []
    
    print("Starter prosessering...")
    for root, _, files in os.walk(in_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                in_path = os.path.join(root, fname)
                print(f"Behandler: {in_path}")

                # Lager en speilet mappestruktur i output-mappen
                rel_dir = os.path.relpath(root, in_dir)
                out_subdir = os.path.join(out_dir, rel_dir)
                os.makedirs(out_subdir, exist_ok=True)

                # Definerer filsti for output
                base_name = os.path.splitext(fname)[0]
                out_path = os.path.join(out_subdir, f'{base_name}_aligned_{target_size}.png')

                # Prosesserer bildet og fanger opp eventuelle feil
                error = process_single_image(
                    in_path, out_path, detector,
                    target_size, min_conf, not_aligned=not_aligned,
                    do_crop=do_crop,
                    do_grayscale=do_grayscale,
                    do_clahe=do_clahe,
                    do_contrast_min=do_contrast_min
                )
                if error:
                    failed_files.append(error)

    print(f"\nProsessering ferdig. {len(failed_files)} bilder feilet.")

    #Helt til slutt normaliser bildene før de lagres som filer sendt til netverk 

    return failed_files
