"""
risk_score.py — Rule-based visual exposure scoring for DeepFake Mirror.

Evaluates simple, interpretable visual qualities of a face photo:
- Face clarity / sharpness (Laplacian variance)
- Face angle / frontality (eye alignment and symmetry)
- Lighting condition (overall brightness and left-vs-right balance)

Returns a risk level ("Low", "Medium", or "High") and a plain-English explanation.
Rule-based only — does not use an ML model for scoring.
"""

import os
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace


class RiskResult(tuple):
    """
    Tuple subclass supporting:
    - Unpacking: risk_level, reason = get_risk_score(img)
    - Attribute access: result.risk_level, result.reason, result.score, result.factors
    - Dict key access: result["risk_level"], result["score"], result["factors"]
    """
    def __new__(cls, risk_level: str, reason: str, score: int = 0, factors: dict = None):
        instance = super().__new__(cls, (risk_level, reason))
        instance.risk_level = risk_level
        instance.reason = reason
        instance.score = score
        instance.factors = factors or {}
        return instance

    def __getitem__(self, item):
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return (
            f"RiskResult(risk_level={self.risk_level!r}, score={self.score}, "
            f"reason={self.reason!r})"
        )


# Canonical sentence library from ui-text.md Section 3.3
FACTOR_SENTENCES = {
    "clarity": {
        "clear": "The face appears clear.",
        "partial": "Some facial details are unclear.",
        "poor": "The face is difficult to distinguish.",
    },
    "angle": {
        "frontal": "The face is mostly front-facing.",
        "partial": "The face is partially turned.",
        "profile": "The face is viewed mostly from the side.",
    },
    "lighting": {
        "even": "Lighting is even.",
        "mixed": "Lighting is uneven.",
        "poor": "The face is poorly lit.",
    },
}

BAND_REASONS = {
    "Low": "The face is less clearly exposed in this image. This does not mean the photo is safe from misuse.",
    "Medium": "Some facial details are visible and may be useful to automated face tools.",
    "High": "The face is clearly visible, making it easier for automated face tools to analyze.",
}


def get_risk_score(image_path: str) -> RiskResult:
    """
    Evaluates visual exposure qualities of a face image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        RiskResult: Contains risk_level ('Low', 'Medium', or 'High'),
                    score (0-100), plain-English explanation reason,
                    and detailed factors.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If file is too large, corrupted, has no face, or has multiple faces.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check file size limit (15MB)
    if os.path.getsize(image_path) > 15 * 1024 * 1024:
        raise ValueError("This image is too large. Please use a smaller file.")

    # Validate image readability with PIL
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError("We couldn't open this image. Try a different photo.") from e

    # Check image format
    try:
        with Image.open(image_path) as img:
            fmt = (img.format or "").upper()
            if fmt not in ["JPEG", "JPG", "PNG"]:
                raise ValueError("This file type is not supported. Please choose a JPG or PNG image.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("We couldn't open this image. Try a different photo.") from e

    # Detect face and landmarks using DeepFace extract_faces
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",
            enforce_detection=True,
        )
    except Exception as exc:
        raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.") from exc

    if not faces or len(faces) == 0:
        raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.")

    if len(faces) > 1:
        raise ValueError("We found more than one face. Please choose a photo of one person.")

    face_info = faces[0]
    if face_info.get("confidence", 0) == 0:
        raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.")

    facial_area = face_info.get("facial_area", {})
    x = facial_area.get("x", 0)
    y = facial_area.get("y", 0)
    w = facial_area.get("w", 0)
    h = facial_area.get("h", 0)
    left_eye = facial_area.get("left_eye")
    right_eye = facial_area.get("right_eye")

    if w < 50 or h < 50:
        raise ValueError("The face is too small to analyze reliably. Try a closer photo.")

    # Read image using OpenCV for rule-based image processing checks
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise ValueError("We couldn't open this image. Try a different photo.")

    img_h, img_w = bgr_img.shape[:2]

    # Crop the face region safely
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    if (x2 - x1) <= 10 or (y2 - y1) <= 10:
        face_crop = bgr_img
    else:
        face_crop = bgr_img[y1:y2, x1:x2]

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------------------
    # Factor 1: Clarity (Laplacian variance of face crop)
    # ----------------------------------------------------
    lap_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    if lap_var >= 100.0:
        clarity_state = "clear"
        clarity_pts = 40
    elif lap_var >= 30.0:
        clarity_state = "partial"
        clarity_pts = 20
    else:
        clarity_state = "poor"
        clarity_pts = 5

    # ----------------------------------------------------
    # Factor 2: Face Angle (Eye alignment & symmetry)
    # ----------------------------------------------------
    if left_eye is not None and right_eye is not None:
        dx = abs(left_eye[0] - right_eye[0])
        dy = abs(left_eye[1] - right_eye[1])
        tilt = dy / (dx + 1e-5)
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        face_center_x = x + (w / 2.0)
        yaw_offset = abs(eye_center_x - face_center_x) / (w / 2.0 + 1e-5)

        if yaw_offset <= 0.25 and tilt <= 0.25:
            angle_state = "frontal"
            angle_pts = 35
        elif yaw_offset <= 0.45 and tilt <= 0.45:
            angle_state = "partial"
            angle_pts = 20
        else:
            angle_state = "profile"
            angle_pts = 10
    else:
        angle_state = "profile"
        angle_pts = 10

    # ----------------------------------------------------
    # Factor 3: Lighting (Mean brightness & L-R balance)
    # ----------------------------------------------------
    mean_bright = float(np.mean(gray_face))
    half_w = gray_face.shape[1] // 2
    if half_w > 0:
        left_b = float(np.mean(gray_face[:, :half_w]))
        right_b = float(np.mean(gray_face[:, half_w:]))
        lr_diff = abs(left_b - right_b)
    else:
        lr_diff = 0.0

    if mean_bright < 65.0 or mean_bright > 215.0:
        lighting_state = "poor"
        lighting_pts = 5
    elif lr_diff > 45.0:
        lighting_state = "mixed"
        lighting_pts = 15
    else:
        lighting_state = "even"
        lighting_pts = 25

    # ----------------------------------------------------
    # Compute deterministic total score (0 to 100)
    # ----------------------------------------------------
    total_score = max(0, min(100, clarity_pts + angle_pts + lighting_pts))

    # Categorize into Low / Medium / High
    if total_score < 35:
        risk_level = "Low"
    elif total_score < 70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    clarity_sentence = FACTOR_SENTENCES["clarity"][clarity_state]
    angle_sentence = FACTOR_SENTENCES["angle"][angle_state]
    lighting_sentence = FACTOR_SENTENCES["lighting"][lighting_state]
    band_reason = BAND_REASONS[risk_level]

    reason = f"{band_reason} {clarity_sentence} {angle_sentence} {lighting_sentence}"

    factors_dict = {
        "clarity": {"state": clarity_state, "sentence": clarity_sentence, "variance": round(lap_var, 1)},
        "angle": {"state": angle_state, "sentence": angle_sentence},
        "lighting": {"state": lighting_state, "sentence": lighting_sentence, "brightness": round(mean_bright, 1)},
    }

    return RiskResult(
        risk_level=risk_level,
        reason=reason,
        score=total_score,
        factors=factors_dict,
    )
