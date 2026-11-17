"""
face_tools.py — Face detection and embedding extraction for DeepFake Mirror.

Uses the DeepFace library to detect a face in a given photo and return its
face embedding as a list of numerical values.
"""

import os
from PIL import Image
from deepface import DeepFace


def get_face_embedding(image_path: str, model_name: str = "VGG-Face") -> list:
    """
    Detects a face in the provided image file and extracts its embedding vector.

    Args:
        image_path (str): Path to the image file (JPG or PNG).
        model_name (str): Face recognition model to use (default: 'VGG-Face').

    Returns:
        list: A list of floating-point numbers representing the facial embedding.

    Raises:
        FileNotFoundError: If the specified image_path does not exist.
        ValueError: If no face is detected, multiple faces exist, face is too small,
                    or the image cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Check file size limit (15MB)
    if os.path.getsize(image_path) > 15 * 1024 * 1024:
        raise ValueError("This image is too large. Please use a smaller file.")

    # Verify that the image can be opened and decoded
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError("We couldn't open this image. Try a different photo.") from e

    # Check image format explicitly
    try:
        with Image.open(image_path) as img:
            fmt = (img.format or "").upper()
            if fmt not in ["JPEG", "JPG", "PNG"]:
                raise ValueError("This file type is not supported. Please choose a JPG or PNG image.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("We couldn't open this image. Try a different photo.") from e

    # First check face count to handle multiple faces explicitly
    try:
        detected_faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",
            enforce_detection=True,
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if "face could not be detected" in exc_str or "facenotdetected" in exc.__class__.__name__.lower():
            raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.") from exc
        raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.") from exc

    if not detected_faces or len(detected_faces) == 0:
        raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.")

    if len(detected_faces) > 1:
        raise ValueError("We found more than one face. Please choose a photo of one person.")

    # Check if face is too small (< 50x50 px)
    fa = detected_faces[0].get("facial_area", {})
    if fa.get("w", 0) < 50 or fa.get("h", 0) < 50:
        raise ValueError("The face is too small to analyze reliably. Try a closer photo.")

    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=True,
        )

        if not results:
            raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.")

        embedding = results[0].get("embedding")
        if embedding is None:
            raise ValueError("We couldn't complete the analysis. Please try another photo.")

        return list(embedding)

    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        exc_name = exc.__class__.__name__
        exc_str = str(exc).lower()

        if "facenotdetected" in exc_name.lower() or "face could not be detected" in exc_str or "no face" in exc_str:
            raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.") from exc

        raise ValueError("We couldn't complete the analysis. Please try another photo.") from exc
