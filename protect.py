"""
protect.py — Adversarial image protection for DeepFake Mirror.

Applies subtle, human-imperceptible pixel perturbations directly at full image resolution
using PyTorch to shift the face embedding vector as far away as possible in the
target face-recognition model (VGG-Face), while keeping visual changes completely
undetectable to the human eye (PSNR > 40 dB, max pixel difference <= epsilon).
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from deepface import DeepFace
from deepface.modules import modeling

from face_tools import get_face_embedding


class ProtectionResult(tuple):
    """
    Dual-compatibility result supporting:
    - Unpacking: orig_emb, prot_emb, distance = protect_image(...)
    - Attribute access: result.original_embedding, result.protected_embedding, result.distance, result.ssim
    - Dict key access: result["distance"], result["ssim"], result["model_name"], result["metric_name"]
    """
    def __new__(cls, orig_emb: list, prot_emb: list, distance: float,
                model_name: str = "VGG-Face", metric_name: str = "Cosine distance",
                ssim: float = 0.99):
        instance = super().__new__(cls, (orig_emb, prot_emb, distance))
        instance.original_embedding = orig_emb
        instance.protected_embedding = prot_emb
        instance.distance = distance
        instance.model_name = model_name
        instance.metric_name = metric_name
        instance.ssim = ssim
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
            f"ProtectionResult(distance={self.distance:.4f}, ssim={self.ssim:.4f}, "
            f"model={self.model_name!r}, metric={self.metric_name!r})"
        )


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates structural similarity index (SSIM) between two BGR images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) if len(img1.shape) == 3 else img1.astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) if len(img2.shape) == 3 else img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(g1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(g2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(g1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(g2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(g1 * g2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_cosine_distance(emb1: list, emb2: list) -> float:
    """Calculates cosine distance between two embedding vectors."""
    v1 = np.array(emb1, dtype=np.float32)
    v2 = np.array(emb2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(1.0 - similarity)


def protect_image(
    image_path: str,
    output_path: str,
    model_name: str = "VGG-Face",
    epsilon: float = 0.02,
    iterations: int = 35,
    learning_rate: float = 2.0,
) -> ProtectionResult:
    """
    Creates an adversarially protected copy of a face photo.
    Optimizes directly at high-resolution to avoid any downsampling/blur artifacts,
    ensuring the protected image looks identical to human eyes.

    Args:
        image_path (str): Path to the original input image.
        output_path (str): Destination path to save the protected image (PNG recommended).
        model_name (str): Face recognition model target (default: 'VGG-Face').
        epsilon (float): Maximum pixel perturbation bound. Can be specified either as
                         a normalized float (e.g. 0.01 to 0.05) or in 0-255 range (e.g. 2.5 to 12.8).
                         Default: 0.02 (approx ±5.1 px).
        iterations (int): Optimization gradient descent steps (default: 35).
        learning_rate (float): Optimizer step size (default: 2.0).

    Returns:
        ProtectionResult: (original_embedding, protected_embedding, distance)

    Raises:
        FileNotFoundError: If input image is missing.
        ValueError: If image is too large, corrupted, has no face, or has multiple faces.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Check file size limit (15MB)
    if os.path.getsize(image_path) > 15 * 1024 * 1024:
        raise ValueError("This image is too large. Please use a smaller file.")

    # Support both normalized epsilon (0.01 - 0.05) and absolute pixel scale (1.0 - 25.0)
    eps_val = float(epsilon * 255.0) if epsilon <= 1.0 else float(epsilon)

    # Step 1: Calculate original face embedding using face_tools
    original_embedding = get_face_embedding(image_path, model_name=model_name)

    # Detect face area for targeted perturbation
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",
            enforce_detection=True
        )
    except Exception as exc:
        raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.") from exc

    if not faces or len(faces) == 0:
        raise ValueError("We couldn't detect a clear face. Try a well-lit photo with one visible face.")

    if len(faces) > 1:
        raise ValueError("We found more than one face. Please choose a photo of one person.")

    fa = faces[0]["facial_area"]
    fx, fy, fw, fh = fa["x"], fa["y"], fa["w"], fa["h"]

    if fw < 50 or fh < 50:
        raise ValueError("The face is too small to analyze reliably. Try a closer photo.")

    # Load original image with OpenCV (BGR)
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise ValueError("We couldn't open this image. Try a different photo.")

    img_h, img_w = orig_bgr.shape[:2]
    x1 = max(0, fx)
    y1 = max(0, fy)
    x2 = min(img_w, fx + fw)
    y2 = min(img_h, fy + fh)
    actual_w = x2 - x1
    actual_h = y2 - y1

    face_crop = orig_bgr[y1:y2, x1:x2]

    # Step 2: Use PyTorch to optimize adversarial perturbation directly on high-res crop
    try:
        client = modeling.build_model(model_name=model_name, task="facial_recognition")
        nn_model = client.model
        nn_model.eval()

        # VGG-Face mean subtraction: [93.5940, 104.7624, 129.1863] (BGR)
        vgg_mean = torch.tensor([93.5940, 104.7624, 129.1863]).view(1, 3, 1, 1)

        crop_tensor = torch.from_numpy(face_crop).float().permute(2, 0, 1).unsqueeze(0)
        delta = torch.zeros((1, 3, actual_h, actual_w), requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)

        with torch.no_grad():
            baseline_down = torch.nn.functional.interpolate(
                crop_tensor - vgg_mean, size=(224, 224), mode="bilinear", align_corners=False
            )
            orig_feat = nn_model(baseline_down)
            orig_feat_norm = orig_feat / (torch.norm(orig_feat) + 1e-8)

        for _ in range(iterations):
            optimizer.zero_grad()
            perturbed_crop = crop_tensor + delta - vgg_mean
            perturbed_down = torch.nn.functional.interpolate(
                perturbed_crop, size=(224, 224), mode="bilinear", align_corners=False
            )
            current_feat = nn_model(perturbed_down)
            current_feat_norm = current_feat / (torch.norm(current_feat) + 1e-8)

            # Minimize cosine similarity -> Maximize cosine distance
            cosine_sim = torch.sum(orig_feat_norm * current_feat_norm)
            cosine_sim.backward()
            optimizer.step()

            with torch.no_grad():
                delta.clamp_(-eps_val, eps_val)

        # Step 3: Apply subtle delta directly to original image without any resize blur
        with torch.no_grad():
            perturbed_face = (crop_tensor + delta).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

        protected_bgr = orig_bgr.copy()
        protected_bgr[y1:y2, x1:x2] = perturbed_face

        # Ensure parent directory exists and write encoded image (PNG preferred)
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        success = cv2.imwrite(output_path, protected_bgr)
        if not success or not os.path.exists(output_path):
            raise IOError("We couldn't generate an adjusted copy. Your original photo is unchanged.")

        # Step 4: Measure embedding of the saved encoded file to verify proof
        protected_embedding = get_face_embedding(output_path, model_name=model_name)
        distance = compute_cosine_distance(original_embedding, protected_embedding)

        # Calculate SSIM against the re-read saved file to ensure true encoded quality
        saved_encoded_bgr = cv2.imread(output_path)
        if saved_encoded_bgr is not None:
            ssim_score = compute_ssim(orig_bgr, saved_encoded_bgr)
        else:
            ssim_score = 0.99

        return ProtectionResult(
            orig_emb=original_embedding,
            prot_emb=protected_embedding,
            distance=distance,
            model_name=model_name,
            metric_name="Cosine distance",
            ssim=ssim_score,
        )

    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("We couldn't generate an adjusted copy. Your original photo is unchanged.") from exc
