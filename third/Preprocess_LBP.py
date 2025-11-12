from __future__ import annotations
"""Loop gjennom alle bilder: Gå gjennom alle bildene i hele datasettet.



Del 1: Din jobb (Preprocessing + LBP)
Du skal nå lage tre lister

Start med tre tomme lister:

all_features = []

all_labels = []

all_person_ids = [] (Dette er den nye, viktige listen)

Instantiate one MTCNN() object.

Lag en hovedmappe for å se på bildene: "data/mood_detection/images_processed/"


Gå gjennom alle de 19 personene:


for person in ...

Lag en undermappe for hver person: "data/mood_detection/images_processed/person_id/"

person_id = ... (f.eks. "person_5")

Gå gjennom de 8 følelsene (og eventuelle flere bilder) for den personen:

for bilde, label in ...

Full prosessering: For hvert bilde, prosessen er slik:

Steg A: Prosesser originalbildet

Bruk funksjonen process_single_image 
fra preprocess-helper-func.py til å prosessere bildet.

Lagre det prosesserte bildet i undermappen for personen. 
Bildene er normalisert til 0-1, derav må vi konvertere tilbake til 0-255.


LBP: Kjør LBP-funksjonen din for å 
få den endelige 1D-vektoren. 
Bruk lbp_vector funksjonen fra preprocess-helper-func.py


Steg B: Prosesser det speilede bildet

Speil 2D-bildet: bilde_speilet = np.fliplr(bilde)

Lagre det speilede bildet

Kjør LBP-funksjonen på bilde_speilet -> Få vektor_speilet.


Steg C: Lagre begge resultatene

all_features.append(vektor_original)

all_labels.append(label)

all_person_ids.append(person_id)

all_features.append(vektor_speilet)

all_labels.append(label)

all_person_ids.append(person_id)



Lagre til disk:

Når loopen er ferdig, vil du ha (for eksempel) 19 personer * 8 bilder * 2 (aug) = 304 rader.

1. Konverter features og grupper til NumPy-arrays.

X = np.array(all_features)

y = np.array(all_labels)

2. Konverter labels
encoder = LabelEncoder()
y_numeric = encoder.fit_transform(all_labels) # Gjør om ['happy', 'sad'] -> [0, 1]

3. Hent ut "nøkkelen" (class_names)

# 4. Lagre alt i én fil
np.savez_compressed(
    'ferdig_lbp_data.npz', 
    features=X, 
    labels=y_numeric, 
    groups=groups,
    class_names=class_names  # Lagrer "fasiten"
)
"""

"""
Slik lagres dataene til ferdig_lbp_data.npz:
    Indeks (Rad)    X (features)        y_numeric   groups (person-ID)  class_names (klassename)
    0               [vektor_P1_original]     0      'person_1'             'happy'
    1               [vektor_P1_speilet]      0      'person_1'             'happy'
    2               [vektor_P2_original]     1      'person_2'             'sad'
    3               [vektor_P2_speilet]      1      'person_2'             'sad'
    4               [vektor_P3_original]     2      'person_3'             'surprised'
    5               [vektor_P3_speilet]      2      'person_3'             'surprised'
"""

###Kode i preprocess_LBP bruk det du har lært fra ablobation study. 
###Print ut bilder riktig fra ablobation study

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
try:
    NumpyArrayMemoryError = np.core._exceptions._ArrayMemoryError  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    NumpyArrayMemoryError = MemoryError
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder

from preprocess_helper_func import lbp_vector, process_single_image


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = _BASE_DIR / "data" / "facial_emotion_recognition"
DEFAULT_ALIGNED_IMAGE_ROOT = _BASE_DIR / "data" / "images_aligned"
DEFAULT_OUTPUT_ROOT = _BASE_DIR / "data" / "images_processed_lbp"
DEFAULT_DATASET_PATH = _BASE_DIR / "third" / "ferdig_lbp_data.npz"

# Crop parameters from ablation study (best performing: crop base)
CROP_BASE = (0.18, 0.10, 0.64, 0.75)  # (x_ratio, y_ratio, width_ratio, height_ratio)
RESIZE_SHAPE = (256, 256)


@dataclass
class FailureRecord:
    image_path: Path
    reason: str


def _iter_person_dirs(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry


def _iter_images(person_dir: Path) -> Iterable[Path]:
    for path in sorted(person_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _infer_label(image_path: Path, person_dir: Path) -> str:
    """Bruk mappenavnet rett over bildet som label."""
    if image_path.parent == person_dir:
        return image_path.stem
    return image_path.parent.name


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Konverter 0-1 float eller annet format til uint8 [0,255]."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.dtype == np.uint16:
        return (image / 257.0).astype(np.uint8)
    return image.astype(np.uint8)


def _crop_by_ratio(image: np.ndarray, crop: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop image by ratio parameters (x_ratio, y_ratio, width_ratio, height_ratio)."""
    h, w = image.shape
    x_ratio, y_ratio, width_ratio, height_ratio = crop

    x_start = int(round(x_ratio * w))
    y_start = int(round(y_ratio * h))
    x_end = int(round((x_ratio + width_ratio) * w))
    y_end = int(round((y_ratio + height_ratio) * h))

    x_start = max(0, min(x_start, w - 1))
    y_start = max(0, min(y_start, h - 1))
    x_end = max(x_start + 1, min(x_end, w))
    y_end = max(y_start + 1, min(y_end, h))

    return image[y_start:y_end, x_start:x_end]


def _iter_aligned_entries(root: Path) -> Iterable[Tuple[Path, str, str]]:
    """Iterate over aligned images in the aligned image root directory."""
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        for path in sorted(person_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                label = path.stem
                if label.endswith("_aligned"):
                    label = label[: -len("_aligned")]
                yield path, person_id, label


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)




def preprocess_dataset(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    target_size: int = 256,
    aligned_image_root: Path = DEFAULT_ALIGNED_IMAGE_ROOT, #Sett denne til None hvis du ikke har aligna bilder
) -> Dict[str, np.ndarray]:
    """
    Hovedfunksjonen: loop gjennom datasettet, preprocess og bygg LBP-features.
    
    Pipeline: Alignment (or load from aligned) → Cropping → Grayscale → Normalization → LBP
    
    If aligned_image_root exists and is not empty, uses pre-aligned images instead of
    aligning with process_single_image.
    """
    use_aligned = aligned_image_root.exists() and any(aligned_image_root.iterdir())
    
    if use_aligned:
        print(f"Using pre-aligned images from: {aligned_image_root}")
    else:
        if not input_root.exists():
            raise FileNotFoundError(f"Inngangsmappe finnes ikke: {input_root}")
        print(f"Aligning images from: {input_root}")
        detector = MTCNN()

    all_features: List[np.ndarray] = []
    all_labels: List[str] = []
    all_person_ids: List[str] = []
    failures: List[FailureRecord] = []

    if use_aligned:
        # Use pre-aligned images
        for aligned_path, person_id, label in _iter_aligned_entries(aligned_image_root):
            try:
                # Load aligned image (already grayscale, normalized 0-1)
                aligned = cv2.imread(str(aligned_path), cv2.IMREAD_GRAYSCALE)
                if aligned is None:
                    failures.append(FailureRecord(aligned_path, "read_error"))
                    continue
                
                # Convert to float32 [0, 1] if needed
                if aligned.dtype == np.uint8:
                    aligned = aligned.astype(np.float32) / 255.0
                else:
                    aligned = aligned.astype(np.float32)
                
                # Apply cropping
                cropped = _crop_by_ratio(aligned, CROP_BASE)
                
                # Resize back to target size
                resized = cv2.resize(cropped, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
                resized = resized.astype(np.float32)
                
                # Convert to uint8 for LBP (LBP expects uint8)
                processed_uint8 = _ensure_uint8(resized)
                
                # Save processed image
                person_output_dir = output_root / person_id
                person_output_dir.mkdir(parents=True, exist_ok=True)
                base_name = aligned_path.stem
                processed_path = person_output_dir / f"{base_name}_processed.png"
                _save_image(processed_path, processed_uint8)
                
                # Extract LBP features
                feature = lbp_vector(processed_uint8)
                
                all_features.append(feature)
                all_labels.append(label)
                all_person_ids.append(person_id)
                
            except Exception as e:
                failures.append(FailureRecord(aligned_path, f"processing_error: {str(e)}"))
                continue
    else:
        # Align images on the fly
        for person_dir in _iter_person_dirs(input_root):
            person_id = person_dir.name
            person_output_dir = output_root / person_id
            person_output_dir.mkdir(parents=True, exist_ok=True)

            for image_path in _iter_images(person_dir):
                label = _infer_label(image_path, person_dir)
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    failures.append(FailureRecord(image_path, "read_error"))
                    continue

                try:
                    # Align, grayscale, normalize (returns float32 [0, 1])
                    aligned = process_single_image(
                        image=image_bgr,
                        target_size=target_size,
                        normalize_0_1=True,
                        detector=detector,
                    )
                except (MemoryError, NumpyArrayMemoryError):
                    failures.append(FailureRecord(image_path, "detector_oom"))
                    continue

                if aligned is None:
                    failures.append(FailureRecord(image_path, "detect_or_align_failed"))
                    continue

                # Apply cropping
                cropped = _crop_by_ratio(aligned, CROP_BASE)
                
                # Resize back to target size
                resized = cv2.resize(cropped, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
                resized = resized.astype(np.float32)
                
                # Convert to uint8 for LBP (LBP expects uint8)
                processed_uint8 = _ensure_uint8(resized)

                base_name = image_path.stem
                processed_path = person_output_dir / f"{base_name}_processed.png"
                _save_image(processed_path, processed_uint8)

                # Extract LBP features (no augmentation)
                feature = lbp_vector(processed_uint8)

                all_features.append(feature)
                all_labels.append(label)
                all_person_ids.append(person_id)

    if not all_features:
        raise RuntimeError("Ingen features generert. Sjekk datasett-stier og detektorens konfigurasjon.")

    X = np.vstack(all_features)
    groups = np.array(all_person_ids)

    encoder = LabelEncoder()
    y_numeric = encoder.fit_transform(all_labels)
    class_names = encoder.classes_

    np.savez_compressed(
        dataset_path,
        features=X,
        labels=y_numeric,
        groups=groups,
        class_names=class_names,
    )

    print(f"Lagre dataset til {dataset_path}")
    print(f"Antall samples: {len(all_features)}")
    if failures:
        print(f"Feilede bilder: {len(failures)}")
        for failure in failures[:10]:
            print(f"  - {failure.reason}: {failure.image_path}")
        if len(failures) > 10:
            print(f"  ... og {len(failures) - 10} flere.")

    return {
        "features": X,
        "labels": y_numeric,
        "groups": groups,
        "class_names": class_names,
        "failures": failures,
    }


if __name__ == "__main__":
    preprocess_dataset()


"""
#Hva du må gi
Du må ha en folder med bilder som skal preprocesses.
Eventtuelt kan du gi inn en folder med allerede aligna bilder.
Se over funksjonen preprocess_dataset() for å se hva du må gi inn.

#Hva du får ut
Ved å kjøre funksjonen preprocess_dataset() 
vil du få ut en fil som heter ferdig_lbp_data.npz.
Den inneholder features, labels, groups og class_names.


#Hva den gjør
Den bygger på vår abloation study og grayscaler, corpper, normalisering og LBP-funksjonen.
på bilden """