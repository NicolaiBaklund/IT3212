# pip install mtcnn opencv-python numpy
import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor

# ===== Canonical target layout =====
# Ratios chosen for frontal-ish faces; keep these fixed for your whole dataset.
def target_landmarks(target_size: int) -> np.ndarray:
    W = H = target_size
    pts = np.array([
        (0.35 * W, 0.38 * H),  # left eye
        (0.65 * W, 0.38 * H),  # right eye
        (0.50 * W, 0.55 * H),  # nose tip
        (0.40 * W, 0.75 * H),  # left mouth
        (0.60 * W, 0.75 * H),  # right mouth
    ], dtype=np.float32)
    return pts

def _pick_largest_face(detections):
    if not detections:
        return None
    # MTCNN returns {'box':[x,y,w,h], 'keypoints':{...}, 'confidence':...}
    return max(detections, key=lambda d: d['box'][2] * d['box'][3])

def align_face_bgr(
    img_bgr: np.ndarray,
    detector: MTCNN = None,  # type: ignore
    target_size: int = 256,
    min_conf: float = 0.85,
    border_mode: int = cv2.BORDER_REFLECT
):
    """
    Returns (aligned_bgr, success, info_dict).
    - aligned_bgr: aligned 256x256 (or target_size) BGR image if success else None
    - success: bool
    - info_dict: {'confidence': float, 'landmarks_src': np.ndarray(5,2) or None}
    """
    if detector is None:
        detector = MTCNN()  # lazy init

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)
    face = _pick_largest_face(detections)
    if face is None or face.get('confidence', 0.0) < min_conf:
        return None, False, {'confidence': face.get('confidence', 0.0) if face else 0.0, 'landmarks_src': None}

    kps = face['keypoints']  # dict: left_eye, right_eye, nose, mouth_left, mouth_right
    src = np.array([
        kps['left_eye'],
        kps['right_eye'],
        kps['nose'],
        kps['mouth_left'],
        kps['mouth_right'],
    ], dtype=np.float32)

    dst = target_landmarks(target_size)

    # Similarity transform (scale+rot+trans). estimateAffinePartial2D handles it.
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None, False, {'confidence': face['confidence'], 'landmarks_src': src}

    aligned = cv2.warpAffine(
        img_bgr, M, (target_size, target_size),
        flags=cv2.INTER_LINEAR, borderMode=border_mode
    )
    return aligned, True, {'confidence': face['confidence'], 'landmarks_src': src}

# ===================================================================

def process_folder(
    in_dir: str,
    out_dir: str,
    target_size: int = 256,
    grayscale_after: bool = False,
    normalize_0_1: bool = False,
    min_conf: float = 0.85
):
    os.makedirs(out_dir, exist_ok=True)
    detector = MTCNN()
    failed = []

    for root, _, files in os.walk(in_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
                in_path = os.path.join(root, fname)
                print(f"Processing: {in_path}", flush=True)
                rel_dir = os.path.relpath(root, in_dir)
                out_subdir = os.path.join(out_dir, rel_dir)
                os.makedirs(out_subdir, exist_ok=True)
                out_path = os.path.join(out_subdir, os.path.splitext(fname)[0] + f'_aligned_{target_size}.png')

                img = cv2.imread(in_path)
                if img is None:
                    failed.append((in_path, 'read_error'))
                    continue

                aligned, ok, info = align_face_bgr(img, detector, target_size=target_size, min_conf=min_conf)
                if not ok:
                    failed.append((in_path, f"detect_or_align_failed(conf={info.get('confidence',0):.2f})"))
                    continue

                # Post steps (optional): grayscale/normalize AFTER alignment
                to_save = aligned
                if grayscale_after:
                    to_save = cv2.cvtColor(to_save, cv2.COLOR_BGR2GRAY) #type: ignore

                if normalize_0_1:
                    # Save float PNG (or scale back to 0-255 uint8 as needed)
                    to_save = to_save.astype(np.float32) / 255.0 #type: ignore

                # Ensure type is uint8 for standard PNG/JPG
                if to_save.dtype != np.uint8: #type: ignore
                    to_save = np.clip(to_save * 255.0, 0, 255).astype(np.uint8) if normalize_0_1 else to_save.astype(np.uint8) #type: ignore

                cv2.imwrite(out_path, to_save) #type: ignore


    return failed