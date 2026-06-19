"""
Image Tampering Detector for ForeSight.
Runs five forensic checks on a document image and returns
heatmaps + flags in the standard ForeSight flag format.
"""

import cv2
import numpy as np
import logging
import base64
import os
from PIL import Image, ImageChops, ImageEnhance
from scipy import ndimage
from skimage.filters import gaussian
from skimage.util import img_as_float
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)


def _load_image(image_path: str) -> tuple[np.ndarray, Image.Image]:
    """
    Load an image from disk and return both:
    - A NumPy BGR array for OpenCV operations
    - A PIL Image object for PIL operations (ELA needs PIL)
    
    For PDFs, this function is not called directly — the OCR engine
    already converts pages to images. Tamper detection works on the
    rendered page images, not the raw PDF bytes.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load with PIL first (handles more formats reliably)
    pil_img = Image.open(image_path).convert("RGB")
    
    # Convert to NumPy BGR for OpenCV
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return cv_img, pil_img


def _create_heatmap_overlay(
    original_cv: np.ndarray,
    anomaly_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.45,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> str:
    """
    Blend an anomaly map over the original image and return
    the result as a base64-encoded PNG string.
    Only colors regions where there is an actual anomaly, keeping
    clean regions completely transparent.
    
    Parameters
    ----------
    original_cv  : BGR NumPy array (the original document image)
    anomaly_map  : Grayscale NumPy array (float or uint8, same spatial size)
    colormap     : OpenCV colormap constant (JET = blue→red, HOT = black→red)
    alpha        : Opacity of the heatmap layer (0.0 = invisible, 1.0 = opaque)
    min_val      : Optional minimum value to start scaling (values below this are 0)
    max_val      : Optional maximum value where scaling saturates (values above this are 255)
    
    Returns
    -------
    str  : base64-encoded PNG — paste directly into st.image()
    """
    # Normalize anomaly map to 0-255 using provided or auto limits
    norm = anomaly_map.astype(np.float32)
    if min_val is None:
        min_val = norm.min()
    if max_val is None:
        max_val = max(norm.max(), min_val + 1e-5)
    
    # Scale and clamp to [0, 255]
    norm = np.clip((norm - min_val) / (max_val - min_val), 0.0, 1.0) * 255
    norm = norm.astype(np.uint8)
    
    # Resize to match original if needed
    if norm.shape[:2] != original_cv.shape[:2]:
        norm = cv2.resize(norm, (original_cv.shape[1], original_cv.shape[0]))
    
    # Apply colormap → produces a BGR colour image
    heatmap_colored = cv2.applyColorMap(norm, colormap)
    
    # Create a dynamic blending mask:
    # - Subtle base tint (alpha=0.18) for clean areas (intensity = 0.0)
    # - Strong highlight (alpha=0.65) for maximum anomaly (intensity = 1.0)
    intensity = norm.astype(np.float32) / 255.0
    alpha_base = 0.18
    alpha_max = 0.65
    blend_mask = alpha_base + intensity * (alpha_max - alpha_base)
    blend_mask = np.expand_dims(blend_mask, axis=2)
    
    # Blend with original using the dynamic mask
    overlay = original_cv.astype(np.float32) * (1.0 - blend_mask) + heatmap_colored.astype(np.float32) * blend_mask
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Encode to base64 PNG
    _, buffer = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64


def _check_ela(pil_img: Image.Image, cv_img: np.ndarray) -> dict:
    """
    Error Level Analysis — detects regions compressed at a different
    quality level than the rest of the image.
    
    How it works:
    1. Save the image to a buffer at quality=90
    2. Reload the saved version
    3. Compute per-pixel absolute difference (original vs resaved)
    4. Amplify the difference map for visibility
    5. Flag if any region's mean ELA value exceeds the threshold
    """
    ELA_QUALITY = 90
    ELA_AMPLIFY = 10        # multiply difference for visual clarity
    FLAG_THRESHOLD = 12.0   # mean ELA value above this → suspicious
    HIGH_THRESHOLD = 20.0   # above this → high severity
    
    # Step 1: Save at controlled quality into a buffer
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=ELA_QUALITY)
    buffer.seek(0)
    
    # Step 2: Reload
    resaved = Image.open(buffer).convert("RGB")
    
    # Step 3: Compute difference
    diff = ImageChops.difference(pil_img, resaved)
    
    # Step 4: Amplify and convert to numpy
    diff_array = np.array(diff).astype(np.float32) * ELA_AMPLIFY
    diff_array = np.clip(diff_array, 0, 255)
    
    # Convert to grayscale anomaly map (mean across RGB channels)
    ela_map = diff_array.mean(axis=2)
    
    # Step 5: Compute statistics
    mean_ela = float(ela_map.mean())
    max_ela = float(ela_map.max())
    
    # Identify suspicious regions (top 5% brightest pixels)
    threshold_95 = float(np.percentile(ela_map, 95))
    suspicious_ratio = float((ela_map > threshold_95).mean())
    
    # Build flag
    flags = []
    if mean_ela > HIGH_THRESHOLD:
        flags.append({
            "check": "tampering_ela",
            "severity": "high",
            "message": (
                f"ELA analysis detected strong compression inconsistencies "
                f"(mean ELA: {mean_ela:.1f}, threshold: {HIGH_THRESHOLD}). "
                f"Multiple regions appear to have been inserted from another source."
            ),
            "evidence": {
                "mean_ela_value": round(mean_ela, 2),
                "max_ela_value": round(max_ela, 2),
                "suspicious_region_ratio": f"{suspicious_ratio:.1%}",
            },
        })
    elif mean_ela > FLAG_THRESHOLD:
        flags.append({
            "check": "tampering_ela",
            "severity": "medium",
            "message": (
                f"ELA analysis detected moderate compression inconsistencies "
                f"(mean ELA: {mean_ela:.1f}). Some regions may have been edited."
            ),
            "evidence": {
                "mean_ela_value": round(mean_ela, 2),
                "max_ela_value": round(max_ela, 2),
                "suspicious_region_ratio": f"{suspicious_ratio:.1%}",
            },
        })
    
    heatmap_b64 = _create_heatmap_overlay(
        cv_img, ela_map, cv2.COLORMAP_JET, min_val=4.0, max_val=20.0
    )
    
    return {
        "check": "ela",
        "label": "JPEG Error Level Analysis",
        "heatmap_b64": heatmap_b64,
        "mean_value": round(mean_ela, 2),
        "flags": flags,
        "description": (
            "Highlights regions with compression inconsistencies. "
            "Red/bright areas indicate pixels that may have been inserted or edited."
        ),
    }


def _check_blur_inconsistency(cv_img: np.ndarray) -> dict:
    """
    Detects regions with anomalous sharpness relative to the rest of the image.
    
    Authentic document photos have consistent focus. A pasted region
    often comes from a different source with different blur characteristics.
    
    Method: Laplacian variance in a sliding window.
    High variance = sharp. Low variance = blurry.
    Flag regions that deviate significantly from the image median.
    """
    WINDOW_SIZE = 32        # pixels — size of each analysis block
    DEVIATION_FACTOR = 2.5  # flag blocks more than 2.5x from median variance
    FLAG_RATIO = 0.08       # flag if more than 8% of blocks are anomalous
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Build a block-level variance map
    variance_map = np.zeros_like(gray, dtype=np.float32)
    step = WINDOW_SIZE // 2  # 50% overlap for smoother map
    
    variances = []
    blocks = []
    
    for y in range(0, h - WINDOW_SIZE, step):
        for x in range(0, w - WINDOW_SIZE, step):
            block = laplacian[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
            var = float(block.var())
            variances.append(var)
            blocks.append((y, x, var))
    
    if not variances:
        return {"check": "blur", "label": "Blur Inconsistency",
                "heatmap_b64": None, "flags": [], "mean_value": 0}
    
    median_var = float(np.median(variances))
    
    # Paint variance values onto the map
    for y, x, var in blocks:
        variance_map[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] = var
    
    # Compute deviation from median
    if median_var > 0:
        deviation_map = np.abs(variance_map - median_var) / median_var
    else:
        deviation_map = np.zeros_like(variance_map)
    
    anomaly_mask = deviation_map > DEVIATION_FACTOR
    anomaly_ratio = float(anomaly_mask.mean())
    
    flags = []
    if anomaly_ratio > FLAG_RATIO:
        severity = "high" if anomaly_ratio > 0.15 else "medium"
        flags.append({
            "check": "tampering_blur",
            "severity": severity,
            "message": (
                f"Blur inconsistency detected: {anomaly_ratio:.1%} of the document "
                f"has anomalous sharpness levels. This may indicate compositing "
                f"from multiple sources."
            ),
            "evidence": {
                "anomalous_region_ratio": f"{anomaly_ratio:.1%}",
                "median_block_variance": round(median_var, 2),
                "deviation_threshold": DEVIATION_FACTOR,
            },
        })
    
    heatmap_b64 = _create_heatmap_overlay(
        cv_img, deviation_map, cv2.COLORMAP_JET, min_val=1.0, max_val=4.0
    )
    
    return {
        "check": "blur",
        "label": "Blur Inconsistency",
        "heatmap_b64": heatmap_b64,
        "mean_value": round(anomaly_ratio * 100, 1),
        "flags": flags,
        "description": (
            "Highlights regions with anomalous sharpness. "
            "Areas that are unusually sharp or blurry relative to the document "
            "may have been inserted from a different source."
        ),
    }



def _check_noise_inconsistency(cv_img: np.ndarray) -> dict:
    """
    Detects regions with anomalous sensor noise patterns.
    
    Every image has a characteristic noise floor from the capture device.
    Edited regions often have different noise (too clean = digitally generated,
    too noisy = from a different camera).
    
    Method:
    1. Apply a strong Gaussian blur to isolate the low-frequency content
    2. Subtract from the original to isolate high-frequency noise
    3. Compute local standard deviation of the noise residual in blocks
    4. Flag blocks that deviate significantly from the image-wide noise floor
    """
    BLOCK_SIZE = 16
    SIGMA_THRESHOLD = 2.2   # flag blocks more than 2.2 std devs from mean
    FLAG_RATIO = 0.06
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Isolate noise residual (original minus smoothed)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    noise_residual = gray - smoothed
    
    h, w = noise_residual.shape
    noise_map = np.zeros_like(noise_residual)
    
    block_stds = []
    block_positions = []
    
    for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
            block = noise_residual[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            std = float(block.std())
            block_stds.append(std)
            block_positions.append((y, x, std))
    
    if not block_stds:
        return {"check": "noise", "label": "Noise Inconsistency",
                "heatmap_b64": None, "flags": [], "mean_value": 0}
    
    global_mean = float(np.mean(block_stds))
    global_std = float(np.std(block_stds))
    
    anomaly_blocks = 0
    for y, x, std in block_positions:
        if global_std > 0:
            z_score = abs(std - global_mean) / global_std
        else:
            z_score = 0
        noise_map[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = z_score
        if z_score > SIGMA_THRESHOLD:
            anomaly_blocks += 1
    
    anomaly_ratio = anomaly_blocks / max(len(block_positions), 1)
    
    flags = []
    if anomaly_ratio > FLAG_RATIO:
        severity = "high" if anomaly_ratio > 0.12 else "medium"
        flags.append({
            "check": "tampering_noise",
            "severity": severity,
            "message": (
                f"Noise inconsistency detected in {anomaly_ratio:.1%} of blocks. "
                f"Regions with anomalous noise patterns may have originated "
                f"from a different image source."
            ),
            "evidence": {
                "anomalous_block_ratio": f"{anomaly_ratio:.1%}",
                "global_noise_mean": round(global_mean, 3),
                "global_noise_std": round(global_std, 3),
            },
        })
    
    heatmap_b64 = _create_heatmap_overlay(
        cv_img, noise_map, cv2.COLORMAP_JET, min_val=1.5, max_val=4.0
    )
    
    return {
        "check": "noise",
        "label": "Noise Inconsistency",
        "heatmap_b64": heatmap_b64,
        "mean_value": round(anomaly_ratio * 100, 1),
        "flags": flags,
        "description": (
            "Highlights blocks with anomalous sensor noise levels. "
            "Too-clean or too-noisy regions relative to the document baseline "
            "may indicate compositing or digital insertion."
        ),
    }



def _check_pixel_artifacts(cv_img: np.ndarray) -> dict:
    """
    Detects localized JPEG blocking artifacts and pixel-level anomalies.
    
    JPEG compression divides images into 8x8 blocks. Tampered regions often
    have dramatically different DCT quantization from surrounding areas,
    creating visible block boundaries at insertion edges.
    
    Method:
    1. Convert to grayscale
    2. Compute the image gradient (Sobel)
    3. Align gradient to 8x8 JPEG block boundaries
    4. Compare inter-block boundary strength vs intra-block gradient
    5. Unusually strong block boundaries = possible edit boundary
    """
    BLOCK_SIZE = 8
    BOUNDARY_RATIO_THRESHOLD = 1.8
    FLAG_RATIO = 0.10
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # Compute gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    artifact_map = np.zeros_like(gray)
    suspicious_blocks = 0
    total_blocks = 0
    
    for y in range(0, h - BLOCK_SIZE * 2, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE * 2, BLOCK_SIZE):
            # Gradient inside the block
            inner = gradient_mag[y+1:y+BLOCK_SIZE-1, x+1:x+BLOCK_SIZE-1]
            inner_mean = float(inner.mean()) + 1e-6
            
            # Gradient on the block boundary
            top = gradient_mag[y, x:x+BLOCK_SIZE]
            bottom = gradient_mag[y+BLOCK_SIZE-1, x:x+BLOCK_SIZE]
            left = gradient_mag[y:y+BLOCK_SIZE, x]
            right = gradient_mag[y:y+BLOCK_SIZE, x+BLOCK_SIZE-1]
            boundary_mean = float(
                np.concatenate([top, bottom, left, right]).mean()
            ) + 1e-6
            
            ratio = boundary_mean / inner_mean
            artifact_map[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = ratio
            
            total_blocks += 1
            if ratio > BOUNDARY_RATIO_THRESHOLD:
                suspicious_blocks += 1
    
    anomaly_ratio = suspicious_blocks / max(total_blocks, 1)
    
    flags = []
    if anomaly_ratio > FLAG_RATIO:
        severity = "high" if anomaly_ratio > 0.20 else "low"
        flags.append({
            "check": "tampering_artifacts",
            "severity": severity,
            "message": (
                f"Pixel artifact analysis detected suspicious block boundaries "
                f"in {anomaly_ratio:.1%} of 8×8 regions. This pattern is "
                f"consistent with JPEG editing artifacts at tampered boundaries."
            ),
            "evidence": {
                "suspicious_block_ratio": f"{anomaly_ratio:.1%}",
                "boundary_ratio_threshold": BOUNDARY_RATIO_THRESHOLD,
            },
        })
    
    heatmap_b64 = _create_heatmap_overlay(
        cv_img, artifact_map, cv2.COLORMAP_JET, min_val=1.2, max_val=2.5
    )
    
    return {
        "check": "artifacts",
        "label": "Pixel Artifact Analysis",
        "heatmap_b64": heatmap_b64,
        "mean_value": round(anomaly_ratio * 100, 1),
        "flags": flags,
        "description": (
            "Detects anomalous JPEG block boundaries that appear at "
            "editing seams. High values at regular 8×8 grid positions "
            "indicate possible content replacement."
        ),
    }



def _check_copy_paste(cv_img: np.ndarray) -> dict:
    """
    Detects copy-pasted regions within the same image.
    
    If a region was cloned from another part of the document
    (e.g., a stamp duplicated, a number copied), ORB feature matching
    within the same image will find high-similarity keypoint clusters
    at two different spatial locations with a consistent geometric transform.
    
    Note: This has the highest false-positive rate of all five checks.
    Use severity=low unless the match cluster is very strong.
    """
    MIN_MATCH_COUNT = 10        # minimum ORB matches to consider suspicious
    SPATIAL_MIN_DISTANCE = 50   # matched regions must be at least 50px apart
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # ORB detector — fast, no patent issues, good for document features
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    copy_paste_map = np.zeros(gray.shape, dtype=np.float32)
    flags = []
    
    if descriptors is None or len(keypoints) < MIN_MATCH_COUNT * 2:
        return {
            "check": "copy_paste",
            "label": "Copy-Paste Detection",
            "heatmap_b64": _create_heatmap_overlay(
                cv_img, copy_paste_map, cv2.COLORMAP_JET, min_val=1.0, max_val=255.0
            ),
            "mean_value": 0,
            "flags": [],
            "description": "Not enough keypoints found for copy-paste analysis.",
        }
    
    # BFMatcher with Hamming distance for binary ORB descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Self-match: find similar descriptors within the same image
    matches = bf.knnMatch(descriptors, descriptors, k=3)
    
    suspicious_pairs = []
    for match_group in matches:
        for m in match_group:
            # Skip self-matches (same keypoint)
            if m.queryIdx == m.trainIdx:
                continue
            if m.distance > 40:  # only strong matches
                continue
            
            pt1 = keypoints[m.queryIdx].pt
            pt2 = keypoints[m.trainIdx].pt
            
            # Only flag if the matching points are spatially separated
            dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            if dist > SPATIAL_MIN_DISTANCE:
                suspicious_pairs.append((pt1, pt2, m.distance))
    
    # Mark suspicious regions on the map
    for pt1, pt2, dist in suspicious_pairs:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        radius = 15
        
        # Draw circles at both matched locations
        cv2.circle(copy_paste_map, (x1, y1), radius, 255 - dist * 4, -1)
        cv2.circle(copy_paste_map, (x2, y2), radius, 255 - dist * 4, -1)
    
    n_suspicious = len(suspicious_pairs)
    
    if n_suspicious >= MIN_MATCH_COUNT:
        flags.append({
            "check": "tampering_copy_paste",
            "severity": "medium",
            "message": (
                f"Copy-paste detection found {n_suspicious} suspicious keypoint "
                f"matches within the same document. This may indicate a region "
                f"was duplicated or cloned from elsewhere in the document."
            ),
            "evidence": {
                "suspicious_match_count": n_suspicious,
                "min_threshold": MIN_MATCH_COUNT,
            },
        })
    
    heatmap_b64 = _create_heatmap_overlay(
        cv_img, copy_paste_map, cv2.COLORMAP_JET, min_val=1.0, max_val=255.0
    )
    
    return {
        "check": "copy_paste",
        "label": "Copy-Paste Detection",
        "heatmap_b64": heatmap_b64,
        "mean_value": n_suspicious,
        "flags": flags,
        "description": (
            "Detects regions that appear to be duplicated within the same document. "
            "Bright spots indicate keypoint clusters that match another "
            "spatial location in the image."
        ),
    }




def detect_tampering(image_path: str) -> dict:
    """
    Run all five tampering checks on a document image.
    
    Parameters
    ----------
    image_path : str
        Path to a JPEG, PNG, or TIFF image file.
        For PDFs, pass the path of a rendered page image
        (the OCR engine already produces these as temp files).
    
    Returns
    -------
    dict
        {
            "checks":       list[dict]  — one result dict per check,
            "flags":        list[dict]  — all flags across all checks,
            "heatmaps":     dict        — { check_name: base64_png },
            "summary":      str         — human readable summary,
            "risk_level":   str         — "clean" / "suspicious" / "high_risk"
        }
    """
    logger.info("Starting tampering analysis on: %s", image_path)
    
    try:
        cv_img, pil_img = _load_image(image_path)
    except Exception as exc:
        logger.error("Failed to load image: %s", exc)
        return {
            "checks": [],
            "flags": [],
            "heatmaps": {},
            "summary": f"Image could not be loaded: {exc}",
            "risk_level": "unknown",
        }
    
    # Run all five checks
    results = []
    all_flags = []
    heatmaps = {}
    
    checks_to_run = [
        ("ELA",       lambda: _check_ela(pil_img, cv_img)),
        ("Blur",      lambda: _check_blur_inconsistency(cv_img)),
        ("Noise",     lambda: _check_noise_inconsistency(cv_img)),
        ("Artifacts", lambda: _check_pixel_artifacts(cv_img)),
        ("CopyPaste", lambda: _check_copy_paste(cv_img)),
    ]
    
    for check_name, check_fn in checks_to_run:
        try:
            result = check_fn()
            results.append(result)
            all_flags.extend(result.get("flags", []))
            if result.get("heatmap_b64"):
                heatmaps[result["check"]] = result["heatmap_b64"]
            logger.info("%s check: %d flag(s)", check_name, len(result.get("flags", [])))
        except Exception as exc:
            logger.error("%s check failed: %s", check_name, exc)
    
    # Determine overall risk level
    high_flags = sum(1 for f in all_flags if f.get("severity") == "high")
    medium_flags = sum(1 for f in all_flags if f.get("severity") == "medium")
    
    if high_flags >= 2:
        risk_level = "high_risk"
    elif high_flags == 1 or medium_flags >= 2:
        risk_level = "suspicious"
    elif medium_flags == 1 or len(all_flags) > 0:
        risk_level = "low_suspicion"
    else:
        risk_level = "clean"
    
    # Build summary
    if all_flags:
        summary = (
            f"Tampering analysis detected {len(all_flags)} indicator(s): "
            f"{high_flags} high, {medium_flags} medium severity. "
            f"Overall assessment: {risk_level.replace('_', ' ').title()}."
        )
    else:
        summary = "No tampering indicators detected. Document appears unmodified."
    
    logger.info("Tampering analysis complete — risk level: %s", risk_level)
    
    return {
        "checks": results,
        "flags": all_flags,
        "heatmaps": heatmaps,
        "summary": summary,
        "risk_level": risk_level,
    }



def detect_tampering_from_pdf_page(pdf_path: str, page_number: int = 0) -> dict:
    """
    Render a single PDF page to an image and run tampering analysis.
    
    Parameters
    ----------
    pdf_path    : str — path to the PDF file
    page_number : int — zero-indexed page number (default: first page)
    """
    import fitz
    import tempfile
    
    try:
        doc = fitz.open(pdf_path)
        if page_number >= len(doc):
            page_number = 0
        
        page = doc[page_number]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI, fast but adequate
        pix = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()
        
        # Save rendered page to a temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        pix.save(tmp_path)
        
        # Run tampering analysis on the rendered image
        result = detect_tampering(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return result
        
    except Exception as exc:
        logger.error("PDF page render failed: %s", exc)
        return {
            "checks": [], "flags": [], "heatmaps": {},
            "summary": f"Could not analyse PDF page: {exc}",
            "risk_level": "unknown",
        }


