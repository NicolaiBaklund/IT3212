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
    detector: MTCNN = None,
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