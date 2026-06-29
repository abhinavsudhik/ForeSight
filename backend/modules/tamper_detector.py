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
import re
from PIL import Image, ImageChops, ImageEnhance
from scipy import ndimage
from skimage.filters import gaussian
from skimage.util import img_as_float
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)

from backend.modules.fft_halftone import analyze_halftone
from backend.modules.font_weight import analyze_font_weight


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


def _is_bank_related_document(document_type: str) -> bool:
    """
    Determines if a document type is bank-related.
    """
    if not document_type:
        return False
    doc_type_lower = document_type.lower()
    bank_types = {
        "bank_statement",
        "cheque",
        "demand_draft",
        "deposit_receipt",
        "account_opening_form",
        "loan_application_form",
        "nach_ecs_mandate",
        "bank_account",
    }
    return doc_type_lower in bank_types or "bank" in doc_type_lower


def _check_halftone_runner(image_path: str, document_type: str = None) -> dict:
    """
    Runner wrapper for FFT Halftone Analysis.
    Loads and runs the halftone analysis, generates a log magnitude spectrum visualization.
    """
    if not _is_bank_related_document(document_type):
        logger.info("FFT Halftone Analysis skipped: document type '%s' is not bank-related", document_type)
        return None

    halftone_res = analyze_halftone(image_path)
    
    # Load image and compute log magnitude spectrum for UI visualization
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    heatmap_b64 = None
    if img is not None:
        h, w = img.shape
        start_y = int(h * 0.25)
        end_y = int(h * 0.75)
        start_x = int(w * 0.25)
        end_x = int(w * 0.75)
        patch = img[start_y:end_y, start_x:end_x]
        if patch.size > 0:
            f_coef = np.fft.fft2(patch)
            f_shift = np.fft.fftshift(f_coef)
            magnitude = np.log(np.abs(f_shift) + 1.0)
            cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            magnitude[cy - 5 : cy + 5, cx - 5 : cx + 5] = 0.0
            
            mag_min, mag_max = magnitude.min(), magnitude.max()
            if mag_max > mag_min:
                mag_norm = ((magnitude - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
            else:
                mag_norm = np.zeros_like(magnitude, dtype=np.uint8)
                
            mag_norm_resized = cv2.resize(mag_norm, (300, 300))
            mag_color = cv2.applyColorMap(mag_norm_resized, cv2.COLORMAP_JET)
            _, buffer = cv2.imencode(".png", mag_color)
            heatmap_b64 = base64.b64encode(buffer).decode("utf-8")
            
    flags = []
    if halftone_res["risk"] != "low":
        severity = "high" if halftone_res["risk"] == "high" else "medium"
        flags.append({
            "check": "tampering_halftone",
            "severity": severity,
            "message": halftone_res["interpretation"],
            "evidence": {
                "peak_sharpness_ratio": halftone_res["peak_sharpness_ratio"],
                "strong_peak_count": halftone_res["strong_peak_count"],
                "likely_professional_print": halftone_res["likely_professional_print"],
            }
        })
        
    return {
        "check": "halftone",
        "label": "FFT Halftone Analysis",
        "heatmap_b64": heatmap_b64,
        "mean_value": halftone_res["peak_sharpness_ratio"],
        "flags": flags,
        "penalty_weight": halftone_res["penalty_weight"],
        "description": (
            "Analyzes spatial frequency peaks to detect printing techniques. "
            "Sharp frequency peaks indicate professional offset printing, "
            "while diffuse frequencies suggest home inkjet or digital editing."
        )
    }


def _check_font_weight_runner(image_path: str, ocr_word_boxes: list) -> dict:
    """
    Runner wrapper for Font Weight Inconsistency analysis.
    Runs the font weight check, draws red bounding boxes around outliers for visualization.
    """
    font_res = analyze_font_weight(image_path, ocr_word_boxes)
    
    img = cv2.imread(image_path)
    heatmap_b64 = None
    if img is not None:
        vis_img = img.copy()
        for word_info in font_res.get("outlier_words", []):
            box = word_info.get("box")
            if box:
                bx, by, bw, bh = box
                cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.putText(vis_img, word_info["word"], (bx, max(15, by - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
        _, buffer = cv2.imencode(".png", vis_img)
        heatmap_b64 = base64.b64encode(buffer).decode("utf-8")
        
    flags = []
    if font_res["risk"] != "low":
        severity = "high" if font_res["risk"] == "high" else "medium"
        flags.append({
            "check": "tampering_font_weight",
            "severity": severity,
            "message": f"Font weight inconsistency detected: {font_res['outlier_count']} outlier word(s) found.",
            "evidence": {
                "mean_stroke_width": font_res["mean_stroke_width"],
                "std_deviation": font_res["std_deviation"],
                "outlier_count": font_res["outlier_count"],
                "outlier_words": font_res["outlier_words"][:10],
            }
        })
        
    return {
        "check": "font_weight",
        "label": "Font Weight Inconsistency",
        "heatmap_b64": heatmap_b64,
        "mean_value": float(font_res["outlier_count"]),
        "flags": flags,
        "penalty_weight": font_res["penalty_weight"],
        "description": (
            "Scans document for words with anomalous line thickness/font weight. "
            "Highlights outlier words in red, suggesting insertions or font swaps."
        )
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
    DEVIATION_FACTOR = 3.8  # flag blocks more than 3.8× from median variance (adjusted up from 3.5)
    FLAG_RATIO = 0.30       # flag if more than 30% of blocks are anomalous (adjusted up from 0.25)
    FLAT_THRESHOLD = 15.0   # exclude flat/blank blocks from skewing the median
    
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
    
    # Filter out flat background blocks to get a true text/foreground median
    non_flat_variances = [v for v in variances if v >= FLAT_THRESHOLD]
    if not non_flat_variances:
        median_var = float(np.median(variances))
    else:
        median_var = float(np.median(non_flat_variances))
    
    # Paint variance values onto the map
    for y, x, var in blocks:
        variance_map[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] = var
    
    # Compute deviation from median using a symmetric ratio for textured blocks
    deviation_map = np.ones_like(variance_map)
    for y, x, var in blocks:
        if var >= FLAT_THRESHOLD and median_var >= FLAT_THRESHOLD:
            ratio = var / median_var
            deviation = max(ratio, 1.0 / ratio)
            deviation_map[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] = deviation
        else:
            deviation_map[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] = 1.0
    
    anomaly_mask = deviation_map > DEVIATION_FACTOR
    anomaly_ratio = float(anomaly_mask.mean())
    
    flags = []
    if anomaly_ratio > FLAG_RATIO:
        severity = "high" if anomaly_ratio > 0.40 else "medium"
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



def _detect_faces(gray_img: np.ndarray) -> list:
    """
    Detect faces using multiple Haar cascades and parameter fallbacks
    to handle low-quality, rotated, or high-noise ID photos.
    """
    import os
    
    cascades = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_profileface.xml"
    ]
    
    detected_faces = []
    
    # Try with original image first, then with slight blur to remove halftone scan noise, then histogram equalized
    preprocessed_images = [
        gray_img,
        cv2.GaussianBlur(gray_img, (3, 3), 0),
        cv2.equalizeHist(gray_img)
    ]
    
    for cascade_name in cascades:
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
        if not os.path.exists(cascade_path):
            continue
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        for img in preprocessed_images:
            # Try different scaleFactors and minNeighbors
            for scale in [1.05, 1.03, 1.08]:
                for neighbors in [3, 2, 4]:
                    try:
                        faces = face_cascade.detectMultiScale(
                            img,
                            scaleFactor=scale,
                            minNeighbors=neighbors,
                            minSize=(24, 24)
                        )
                        if len(faces) > 0:
                            # Convert to list of tuples and return immediately upon first success
                            for face in faces:
                                detected_faces.append(tuple(face))
                            return detected_faces
                    except Exception as exc:
                        logger.warning("Error running Haar cascade %s: %s", cascade_name, exc)
                        
    return detected_faces


def _check_noise_inconsistency(cv_img: np.ndarray) -> dict:
    """
    Detects regions with anomalous sensor noise patterns.
    
    Every image has a characteristic noise floor from the capture device.
    Edited regions often have different noise (too clean = digitally generated,
    too noisy = from a different camera).
    
    Method (two-scale with spatial clustering):
    1. Extract noise residuals at two scales:
       - Fine (sigma=1.5): captures sensor-level noise differences
       - Medium (sigma=5.0): captures texture-level inconsistencies
    2. Compute local std-dev in 8×8 blocks (finer than before for small regions)
    3. Flag blocks deviating >1.8 sigma from global mean
    4. NEW: Spatial clustering — check if anomalous blocks form a contiguous
       region (10+ adjacent blocks). Scattered anomalies = natural; clustered
       anomalies = tampering. This avoids false positives while catching
       coherent tampered regions like pasted photos.
    """
    BLOCK_SIZE = 8          # finer granularity
    SIGMA_THRESHOLD = 2.2   # flag blocks more than 2.2 std devs from mean (adjusted up from 2.0)
    FLAG_RATIO = 0.10       # flag if more than 10% of blocks are anomalous (adjusted up from 0.08)
    CLUSTER_MIN_BLOCKS = 75 # minimum adjacent anomalous blocks to count as a cluster (adjusted up from 60)
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # --- Two-scale noise extraction ---
    smoothed_fine = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    smoothed_medium = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0)
    noise_fine = gray - smoothed_fine
    noise_medium = smoothed_fine - smoothed_medium
    
    # Combine both residuals (weighted)
    noise_residual = np.sqrt(noise_fine**2 * 0.7 + noise_medium**2 * 0.3)
    
    noise_map = np.zeros_like(noise_residual)
    
    # Build grid dimensions for spatial clustering
    grid_h = (h - BLOCK_SIZE) // BLOCK_SIZE
    grid_w = (w - BLOCK_SIZE) // BLOCK_SIZE
    
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
    
    # Build anomaly grid for spatial clustering
    anomaly_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    anomaly_blocks = 0
    idx = 0
    
    for y, x, std in block_positions:
        if global_std > 0:
            z_score = abs(std - global_mean) / global_std
        else:
            z_score = 0
        noise_map[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = z_score
        
        gy = y // BLOCK_SIZE
        gx = x // BLOCK_SIZE
        if z_score > SIGMA_THRESHOLD:
            anomaly_blocks += 1
            if gy < grid_h and gx < grid_w:
                anomaly_grid[gy, gx] = 1
        idx += 1
    
    anomaly_ratio = anomaly_blocks / max(len(block_positions), 1)
    
    # --- Spatial clustering: find connected components of anomalous blocks ---
    labeled_clusters, num_clusters = ndimage.label(anomaly_grid)
    largest_cluster_size = 0
    cluster_ys, cluster_xs = [], []
    if num_clusters > 0:
        cluster_sizes = ndimage.sum(anomaly_grid, labeled_clusters,
                                     range(1, num_clusters + 1))
        largest_cluster_size = int(max(cluster_sizes))
        largest_label = int(np.argmax(cluster_sizes)) + 1
        cluster_mask = (labeled_clusters == largest_label)
        cluster_ys, cluster_xs = np.where(cluster_mask)
    
    # Check if the cluster shape is thin/ribbon-like (e.g. barcode, watermark text line, or border line)
    # vs a solid 2D shape (photo or paper patch).
    is_thin_ribbon = False
    if largest_cluster_size >= CLUSTER_MIN_BLOCKS and len(cluster_ys) > 0:
        h_blocks = int(cluster_ys.max() - cluster_ys.min()) + 1
        w_blocks = int(cluster_xs.max() - cluster_xs.min()) + 1
        if min(h_blocks, w_blocks) < 6:
            is_thin_ribbon = True
            logger.info(
                "Noise cluster shape is thin ribbon (%dx%d blocks) — suppressing as watermark or text line",
                w_blocks, h_blocks
            )
            
    # --- Face-aware cluster suppression ---
    # On physical ID documents, the printed photo area naturally has different noise.
    # Fix: detect faces using robust helper. Pre-process the face search image with a Gaussian blur to
    # remove scanner halftone grids, and search with finer scale.
    face_overlaps_cluster = False
    if largest_cluster_size >= CLUSTER_MIN_BLOCKS and not is_thin_ribbon:
        gray_uint8 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Use robust helper to detect faces
        faces = _detect_faces(gray_uint8)
        
        if len(faces) > 0 and len(cluster_ys) > 0:
            for (ffx, ffy, ffw, ffh) in faces:
                # Expand face bbox by 60% to cover the full ID photo area
                expand = 0.6
                efx = max(0, int(ffx - ffw * expand))
                efy = max(0, int(ffy - ffh * expand))
                efw = min(w, int(ffx + ffw + ffw * expand)) - efx
                efh = min(h, int(ffy + ffh + ffh * expand)) - efy
                
                # Convert expanded face bbox to grid coordinates
                g_fx1 = efx // BLOCK_SIZE
                g_fy1 = efy // BLOCK_SIZE
                g_fx2 = (efx + efw) // BLOCK_SIZE
                g_fy2 = (efy + efh) // BLOCK_SIZE
                
                # Count how many of the cluster's blocks are within the face bbox
                y_start = max(0, g_fy1)
                y_end = min(grid_h, g_fy2 + 1)
                x_start = max(0, g_fx1)
                x_end = min(grid_w, g_fx2 + 1)
                
                cluster_blocks_in_face = np.sum(cluster_mask[y_start:y_end, x_start:x_end])
                face_blocks_count = (y_end - y_start) * (x_end - x_start)
                
                if face_blocks_count > 0:
                    overlap_ratio_cluster = cluster_blocks_in_face / largest_cluster_size
                    overlap_ratio_face = cluster_blocks_in_face / face_blocks_count
                    
                    # If 25%+ of the cluster lies in the face area OR if 25%+ of the face area is filled with the cluster blocks
                    if overlap_ratio_cluster > 0.25 or overlap_ratio_face > 0.25:
                        face_overlaps_cluster = True
                        logger.info(
                            "Noise cluster (size=%d) overlaps with detected face region (cluster overlap: %.1f%%, face overlap: %.1f%%) — suppressing as natural ID card photo region (not tampering)",
                            largest_cluster_size, overlap_ratio_cluster * 100, overlap_ratio_face * 100
                        )
                        break
    
    has_coherent_cluster = (
        largest_cluster_size >= CLUSTER_MIN_BLOCKS and not face_overlaps_cluster and not is_thin_ribbon
    )
    
    flags = []
    if anomaly_ratio > FLAG_RATIO or has_coherent_cluster:
        # Clustered anomalies get higher severity — they indicate a coherent
        # tampered region rather than natural noise variation
        is_critical_escalation = has_coherent_cluster and largest_cluster_size >= 150
        
        if is_critical_escalation:
            severity = "high"
            # Do not display cluster details in tampering_noise message to prevent redundancy
            cluster_msg = ""
        elif has_coherent_cluster:
            severity = "high" if largest_cluster_size >= 90 else "medium"
            cluster_msg = (
                f" A contiguous cluster of {largest_cluster_size} anomalous blocks "
                f"was detected, indicating a coherent region with different "
                f"noise characteristics (likely pasted from another source)."
            )
        else:
            severity = "high" if anomaly_ratio > 0.12 else "medium"
            cluster_msg = ""
        
        flags.append({
            "check": "tampering_noise",
            "severity": severity,
            "message": (
                f"Noise inconsistency detected in {anomaly_ratio:.1%} of blocks. "
                f"Regions with anomalous noise patterns may have originated "
                f"from a different image source."
                + cluster_msg
            ),
            "evidence": {
                "anomalous_block_ratio": f"{anomaly_ratio:.1%}",
                "global_noise_mean": round(global_mean, 3),
                "global_noise_std": round(global_std, 3),
                "largest_cluster_blocks": largest_cluster_size,
                "num_clusters": num_clusters,
            },
        })
        
        # Critical cluster escalation: a massive contiguous cluster of
        # anomalous blocks is very strong evidence of a pasted region.
        # Emit a second, separate flag to trigger the hard-kill rule
        # (2+ high tampering flags → 20% additional score reduction).
        if is_critical_escalation:
            flags.append({
                "check": "tampering_noise_cluster",
                "severity": "high",
                "message": (
                    f"CRITICAL: Massive contiguous noise cluster of "
                    f"{largest_cluster_size} blocks detected. This is very "
                    f"strong evidence that a large region was pasted from "
                    f"a different image source (e.g., photo replacement)."
                ),
                "evidence": {
                    "cluster_size_blocks": largest_cluster_size,
                    "cluster_area_percent": f"{largest_cluster_size * BLOCK_SIZE * BLOCK_SIZE / (h * w) * 100:.1f}%",
                },
            })
    
    heatmap_b64 = _create_heatmap_overlay(
        cv_img, noise_map, cv2.COLORMAP_JET, min_val=1.0, max_val=3.5
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
            "may indicate compositing or digital insertion. Uses two-scale "
            "noise extraction and spatial clustering to distinguish natural "
            "variation from coherent tampered regions."
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
    BOUNDARY_RATIO_THRESHOLD = 2.3  # adjusted up from 2.1
    FLAG_RATIO = 0.17  # adjusted up from 0.14
    # NOTE: PDF→PNG rendering at 144 DPI creates legitimate block artifacts
    # at text edges. Previous thresholds (1.8 / 10%) flagged clean rendered PDFs.
    
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
        severity = "high" if anomaly_ratio > 0.32 else "medium"
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
    MIN_MATCH_COUNT = 60        # minimum ORB matches to consider suspicious
    CLUSTER_MIN_MATCHES = 50    # minimum matches in a single displacement cluster to flag (adjusted up from 45)
    SPATIAL_MIN_DISTANCE = 100  # matched regions must be at least 100px apart
    MAX_MATCH_DISTANCE = 22     # only very strong descriptor matches (was 25)
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # ORB detector — fast, no patent issues, good for document features
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    copy_paste_map = np.zeros(gray.shape, dtype=np.float32)
    flags = []
    
    if descriptors is None or len(keypoints) < CLUSTER_MIN_MATCHES * 2:
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
            if m.distance > MAX_MATCH_DISTANCE:  # only very strong matches
                continue
            
            pt1 = keypoints[m.queryIdx].pt
            pt2 = keypoints[m.trainIdx].pt
            
            # Only flag if the matching points are spatially separated
            dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            if dist > SPATIAL_MIN_DISTANCE:
                suspicious_pairs.append((pt1, pt2, m.distance))
    
    # Cluster matches by spatial translation displacement vector (dx, dy)
    displacement_clusters = []
    for i, pair1 in enumerate(suspicious_pairs):
        pt1_1, pt2_1, dist_1 = pair1
        # Sort coordinates to ensure consistent displacement vector direction
        p1, p2 = (pt1_1, pt2_1) if pt1_1[0] < pt2_1[0] else (pt2_1, pt1_1)
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        
        cluster = [pair1]
        for j, pair2 in enumerate(suspicious_pairs):
            if i == j:
                continue
            pt1_2, pt2_2, dist_2 = pair2
            pa, pb = (pt1_2, pt2_2) if pt1_2[0] < pt2_2[0] else (pt2_2, pt1_2)
            dx2 = pb[0] - pa[0]
            dy2 = pb[1] - pa[1]
            
            # Group pairs with identical/similar displacement vectors (within 20px)
            if np.sqrt((dx1 - dx2)**2 + (dy1 - dy2)**2) < 20.0:
                cluster.append(pair2)
        displacement_clusters.append(cluster)
        
    largest_cluster = max(displacement_clusters, key=len) if displacement_clusters else []
    n_suspicious = len(largest_cluster)
    
    # Mark suspicious regions on the map
    for pt1, pt2, dist in largest_cluster:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        radius = 15
        
        # Draw circles at both matched locations
        cv2.circle(copy_paste_map, (x1, y1), radius, 255 - dist * 4, -1)
        cv2.circle(copy_paste_map, (x2, y2), radius, 255 - dist * 4, -1)
    
    if n_suspicious >= CLUSTER_MIN_MATCHES:
        flags.append({
            "check": "tampering_copy_paste",
            "severity": "medium",
            "message": (
                f"Copy-paste detection found a cluster of {n_suspicious} suspicious keypoint "
                f"matches with consistent spatial displacement. This indicates a region "
                f"was duplicated or cloned from elsewhere in the document."
            ),
            "evidence": {
                "suspicious_match_count": n_suspicious,
                "min_threshold": CLUSTER_MIN_MATCHES,
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



def _check_face_region_tampering(cv_img: np.ndarray) -> dict:
    """
    Face-region-aware tampering analysis for identity documents.
    
    On identity documents (Aadhaar, PAN, passport), the photo area is the
    #1 tampering target. This check specifically analyzes the face region
    against the rest of the document to detect photo replacement.
    
    Method:
    1. Detect face using Haar cascade
    2. Expand bounding box by 40% to capture paste-boundary seams
    3. Compare face region vs background on five metrics:
       a. Noise level ratio — different camera = different noise floor
       b. ELA intensity ratio — re-compressed paste = higher ELA locally
       c. Edge discontinuity — paste boundary creates unnatural edges
       d. Color statistics mismatch — different lighting = different luminance stats
       e. JPEG compression quality — different source = different quantization
    4. Flag if 2+ sub-metrics are anomalous (medium), 3+ (high)
    
    Gracefully returns no flags if no face is detected (non-ID documents).
    """
    # Threshold rationale:
    # - PHYSICAL ID cards: printed photo inherently differs from text in noise
    #   (~2-3×) and luminance (~0.3-0.4), but has smooth printed edges and
    #   consistent JPEG compression (same camera captured the whole card).
    # - DIGITAL PASTE: the pasted photo may have similar or lower noise ratio
    #   (both digital sources), BUT creates sharp cut edges, different JPEG
    #   quality levels, and compression blocking artifacts.
    # Key distinguishers: edge sharpness and compression quality are the most
    # reliable because they're high for digital paste but low for physical cards.
    NOISE_RATIO_THRESHOLD = 3.4    # adjusted up from 3.2 to catch subtler noise inconsistencies
    EDGE_STRENGTH_THRESHOLD = 53.0 # adjusted up from 50.0 to detect cut-and-paste seams
    LUMINANCE_DIFF_THRESHOLD = 0.48 # adjusted down from 0.50
    COMPRESSION_RATIO_THRESHOLD = 3.0 # adjusted down from 3.5 to detect quantization discrepancies
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Step 1: Detect face using robust helper
    faces = _detect_faces(gray)
    
    face_map = np.zeros(gray.shape, dtype=np.float32)
    
    if len(faces) == 0:
        # No face detected — not an ID photo or face too small/obscured
        return None
    
    # Use the largest detected face (most likely the ID photo)
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    
    # Step 2: Expand bounding box by 40% to capture paste seams
    expand = 0.40
    ex = max(0, int(fx - fw * expand))
    ey = max(0, int(fy - fh * expand))
    ew = min(w, int(fx + fw * (1 + expand))) - ex
    eh = min(h, int(fy + fh * (1 + expand))) - ey
    
    # Create face mask
    face_mask = np.zeros((h, w), dtype=bool)
    face_mask[ey:ey+eh, ex:ex+ew] = True
    bg_mask = ~face_mask
    
    anomaly_count = 0
    sub_metrics = {}
    
    # --- Metric A: Noise level ratio ---
    gray_f = gray.astype(np.float32)
    smoothed = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.5)
    noise_residual = gray_f - smoothed
    
    face_noise_std = float(noise_residual[face_mask].std()) if face_mask.any() else 0
    bg_noise_std = float(noise_residual[bg_mask].std()) if bg_mask.any() else 1e-6
    # Enforce noise floor to avoid extremely large ratios from flat scanned backgrounds
    bg_noise_std_clamped = max(bg_noise_std, 1.2)
    noise_ratio = face_noise_std / bg_noise_std_clamped
    
    sub_metrics["noise_ratio"] = round(noise_ratio, 2)
    if noise_ratio > NOISE_RATIO_THRESHOLD:
        anomaly_count += 1
    
    # --- Metric B: Edge discontinuity at face boundary ---
    sobel_x = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Extract gradient values along the face boundary (3-pixel band)
    boundary_mask = np.zeros((h, w), dtype=bool)
    band = 3
    boundary_mask[max(0,ey-band):ey+band, ex:ex+ew] = True
    boundary_mask[max(0,ey+eh-band):min(h,ey+eh+band), ex:ex+ew] = True
    boundary_mask[ey:ey+eh, max(0,ex-band):ex+band] = True
    boundary_mask[ey:ey+eh, max(0,ex+ew-band):min(w,ex+ew+band)] = True
    
    boundary_gradient = float(gradient_mag[boundary_mask].mean()) if boundary_mask.any() else 0
    
    sub_metrics["boundary_gradient"] = round(boundary_gradient, 2)
    if boundary_gradient > EDGE_STRENGTH_THRESHOLD:
        anomaly_count += 1
    
    # --- Metric C: Color statistics mismatch ---
    face_lum_std = float(gray_f[face_mask].std()) if face_mask.any() else 0
    bg_lum_std = float(gray_f[bg_mask].std()) if bg_mask.any() else 1e-6
    max_std = max(face_lum_std, bg_lum_std, 1e-6)
    lum_diff = abs(face_lum_std - bg_lum_std) / max_std
    
    sub_metrics["luminance_diff"] = round(lum_diff, 3)
    if lum_diff > LUMINANCE_DIFF_THRESHOLD:
        anomaly_count += 1
    
    # --- Metric D: JPEG compression quality difference ---
    face_blocks_var = []
    bg_blocks_var = []
    for by in range(0, h - 8, 8):
        for bx in range(0, w - 8, 8):
            block = gray_f[by:by+8, bx:bx+8]
            dct_block = cv2.dct(block - block.mean())
            hf_var = float(dct_block[4:, 4:].var())
            if face_mask[by + 4, bx + 4]:
                face_blocks_var.append(hf_var)
            else:
                bg_blocks_var.append(hf_var)
    
    if face_blocks_var and bg_blocks_var:
        face_compression = float(np.mean(face_blocks_var))
        bg_compression = float(np.mean(bg_blocks_var))
        # Enforce a minimum compression variance floor to prevent clean background division explosions
        compression_ratio = face_compression / max(bg_compression, 8.0)
    else:
        compression_ratio = 1.0
    
    sub_metrics["compression_ratio"] = round(compression_ratio, 2)
    if compression_ratio > COMPRESSION_RATIO_THRESHOLD:
        anomaly_count += 1
    
    # --- Paint the face map for heatmap ---
    face_map[face_mask] = float(anomaly_count) / 4.0 * 255
    face_map[boundary_mask] = min(boundary_gradient / EDGE_STRENGTH_THRESHOLD, 1.0) * 255
    
    # --- Build flags ---
    flags = []
    if anomaly_count >= 2:
        severity = "high" if anomaly_count >= 3 else "medium"
        triggered = []
        if noise_ratio > NOISE_RATIO_THRESHOLD:
            triggered.append(f"noise ({noise_ratio:.1f}×)")
        if boundary_gradient > EDGE_STRENGTH_THRESHOLD:
            triggered.append(f"edge ({boundary_gradient:.0f})")
        if compression_ratio > COMPRESSION_RATIO_THRESHOLD:
            triggered.append(f"compression ({compression_ratio:.1f}×)")
        if lum_diff > LUMINANCE_DIFF_THRESHOLD:
            triggered.append(f"luminance ({lum_diff:.2f})")
        
        flags.append({
            "check": "tampering_face_region",
            "severity": severity,
            "message": (
                f"Face region analysis detected {anomaly_count}/4 tampering indicators "
                f"({', '.join(triggered)}). The photo area shows characteristics "
                f"inconsistent with the rest of the document, suggesting the "
                f"photo may have been replaced or digitally inserted."
            ),
            "evidence": {
                "anomaly_count": f"{anomaly_count}/4",
                "face_bbox": f"({fx},{fy},{fw},{fh})",
                **sub_metrics,
            },
        })
    
    heatmap_b64 = _create_heatmap_overlay(
        cv_img, face_map, cv2.COLORMAP_JET, min_val=0, max_val=200.0
    )
    
    return {
        "check": "face_region",
        "label": "Face Region Analysis",
        "heatmap_b64": heatmap_b64,
        "mean_value": anomaly_count,
        "flags": flags,
        "description": (
            "Analyzes the face/photo region of identity documents against the "
            "rest of the document. Compares noise levels, compression artifacts, "
            "edge continuity, color statistics, and JPEG compression quality "
            "to detect photo replacement."
        ),
    }


def is_high_value_field(text: str) -> bool:
    """
    Check if a text block corresponds to a high-value field (amount, account, date, PAN, Aadhaar).
    """
    cleaned = text.strip()
    if not cleaned:
        return False
    
    cleaned_upper = cleaned.upper()
    
    # 1. PAN Pattern (e.g. ABCDE1234F)
    if re.match(r'^[A-Z]{5}\d{4}[A-Z]$', cleaned_upper):
        return True
        
    # 2. Aadhaar Pattern (12 digits, can be spaced or hyphenated)
    if re.match(r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}$', cleaned):
        return True
        
    # 3. Date Patterns (e.g., DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD)
    if re.match(r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$', cleaned):
        return True
        
    # 4. Account Numbers (9 to 18 digits)
    if re.match(r'^\d{9,18}$', cleaned):
        return True
        
    # 5. Amounts (e.g. ₹50,000, 12345.67, $100.00)
    # Remove common currency symbols and commas
    amt_cleaned = re.sub(r'[₹\$\€\£\,]', '', cleaned)
    if re.match(r'^\d+(\.\d+)?$', amt_cleaned):
        try:
            # Check if it has 3+ digits or contains a decimal point
            if len(re.sub(r'\D', '', cleaned)) >= 3 or '.' in amt_cleaned:
                return True
        except Exception:
            pass
            
    return False


def check_background_pattern_disruption(image: np.ndarray, ocr_bboxes: list) -> dict:
    """
    Background Pattern Disruption analysis check.
    Detects editing in high-value numeric fields by analyzing local background texture variance 
    and Gabor energy against the global document background.
    """
    if ocr_bboxes is None or len(ocr_bboxes) == 0:
        return {
            "check": "background_pattern_disruption",
            "label": "Background Pattern Disruption",
            "status": "skipped",
            "penalty_applied": 0.0,
            "penalty_weight": 0.0,
            "flagged_fields": [],
            "global_variance": 0.0,
            "skipped": True,
            "skip_reason": "no_ocr_data",
            "heatmap_b64": None,
            "mean_value": 0.0,
            "flags": [],
            "description": "Background pattern disruption check was skipped: No OCR data available."
        }
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]
    
    # 1. Separate high-value boxes and construct a global OCR exclusion mask
    high_value_boxes = []
    ocr_mask = np.zeros(gray.shape, dtype=np.uint8)
    
    for item in ocr_bboxes:
        if isinstance(item, dict):
            text = item.get("text", "")
            box = item.get("box", [0, 0, 0, 0])
            if len(box) == 4:
                x, y, w, h = box
            else:
                continue
        elif isinstance(item, (tuple, list)) and len(item) == 5:
            x, y, w, h, text = item
        else:
            continue
            
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Exclude all OCR words from global background sampling
        ocr_mask[max(0, y):min(h_img, y+h), max(0, x):min(w_img, x+w)] = 255
        
        # Categorize text field
        if is_high_value_field(text):
            # Try to identify field name from format
            cleaned = text.strip()
            cleaned_upper = cleaned.upper()
            if re.match(r'^[A-Z]{5}\d{4}[A-Z]$', cleaned_upper):
                field_name = "pan"
            elif re.match(r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}$', cleaned):
                field_name = "aadhaar"
            elif re.match(r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$', cleaned):
                field_name = "date"
            elif re.match(r'^\d{9,18}$', cleaned):
                field_name = "account_number"
            else:
                field_name = "amount"
                
            high_value_boxes.append({
                "field_name": field_name,
                "bbox": [x, y, w, h],
                "text": text
            })
            
    if not high_value_boxes:
        return {
            "check": "background_pattern_disruption",
            "label": "Background Pattern Disruption",
            "status": "skipped",
            "penalty_applied": 0.0,
            "penalty_weight": 0.0,
            "flagged_fields": [],
            "global_variance": 0.0,
            "skipped": True,
            "skip_reason": "no_high_value_fields",
            "heatmap_b64": None,
            "mean_value": 0.0,
            "flags": [],
            "description": "Background pattern disruption check was skipped: No high-value numeric fields found to analyze."
        }
        
    # 2. Global background segmentation
    # Run adaptive thresholding globally to separate text stroke foreground from background
    global_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30
    )
    # Background pixels are: threshold == 0 AND outside any OCR box
    # We dilate the text mask to remove text edge transitions from background variance
    global_text_mask = (global_thresh == 255) | (ocr_mask == 255)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_global_text = cv2.dilate(global_text_mask.astype(np.uint8), dilation_kernel)
    global_bg_mask = (dilated_global_text == 0)
    
    # Fallback if too few pixels in global_bg_mask
    if np.sum(global_bg_mask) < 1000:
        global_bg_mask = (global_thresh == 0)
        
    # 3. Global background texture variance
    global_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    global_bg_vals = global_laplacian[global_bg_mask]
    
    if len(global_bg_vals) > 0:
        global_variance = float(np.var(global_bg_vals))
    else:
        global_variance = 0.0
        
    # If the document has no background pattern at all (below noise floor threshold of 5.0)
    if global_variance < 5.0:
        return {
            "check": "background_pattern_disruption",
            "label": "Background Pattern Disruption",
            "status": "skipped",
            "penalty_applied": 0.0,
            "penalty_weight": 0.0,
            "flagged_fields": [],
            "global_variance": round(global_variance, 3),
            "skipped": True,
            "skip_reason": "no_background_pattern",
            "heatmap_b64": None,
            "mean_value": 0.0,
            "flags": [],
            "description": "Background pattern disruption check was skipped: Document has no background security pattern (variance is below noise floor)."
        }
        
    # 4. Gabor filter bank on global image
    # Orientations: 0, 45, 90, 135 degrees; frequency 0.3 -> lambda = 1.0 / 0.3
    gabor_kernels = []
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for theta in orientations:
        kernel = cv2.getGaborKernel(
            ksize=(15, 15),
            sigma=2.0,
            theta=theta,
            lambd=1.0/0.3,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_64F
        )
        gabor_kernels.append(kernel)
        
    gray_f = gray.astype(np.float64)
    gabor_energy_images = []
    global_gabor_energies = []
    
    for kernel in gabor_kernels:
        response = cv2.filter2D(gray_f, cv2.CV_64F, kernel)
        energy_img = response ** 2
        gabor_energy_images.append(energy_img)
        
        # Calculate global Gabor energy on the global background mask
        global_vals = energy_img[global_bg_mask]
        global_energy = float(np.mean(global_vals)) if len(global_vals) > 0 else 0.0
        global_gabor_energies.append(global_energy)
        
    # Determine dominant orientations where global background Gabor energy has signal (> 5.0)
    dominant_orientations = [i for i, energy in enumerate(global_gabor_energies) if energy > 5.0]
    
    # 5. Local Field Analysis
    flagged_fields = []
    clean_fields = []
    expansion = 10
    
    for field in high_value_boxes:
        field_name = field["field_name"]
        x, y, w, h = field["bbox"]
        
        # Expand region by 8-12 (we use 10) pixels on all sides
        ex0 = max(0, x - expansion)
        ey0 = max(0, y - expansion)
        ex1 = min(w_img, x + w + expansion)
        ey1 = min(h_img, y + h + expansion)
        
        if (ex1 - ex0) < 11 or (ey1 - ey0) < 11:
            continue
            
        patch_gray = gray[ey0:ey1, ex0:ex1]
        
        # Local adaptive thresholding to isolate local background
        local_thresh = cv2.adaptiveThreshold(
            patch_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30
        )
        # Dilate local text mask
        dilated_local_text = cv2.dilate(local_thresh.astype(np.uint8), dilation_kernel)
        local_bg_mask = (dilated_local_text == 0)
        
        # A. Laplacian Variance check
        local_laplacian = global_laplacian[ey0:ey1, ex0:ex1]
        local_bg_vals = local_laplacian[local_bg_mask]
        
        if len(local_bg_vals) > 0:
            local_variance = float(np.var(local_bg_vals))
        else:
            local_variance = 0.0
            
        ratio = local_variance / global_variance if global_variance > 0 else 1.0
        
        triggered_check = None
        trigger_ratio = ratio
        
        if ratio < 0.15 or ratio > 4.0:
            triggered_check = "laplacian"
            trigger_ratio = ratio
            
        # B. Gabor energy checks on dominant orientations
        g_ratios = []
        if not triggered_check and dominant_orientations:
            for idx in dominant_orientations:
                local_energy_vals = gabor_energy_images[idx][ey0:ey1, ex0:ex1][local_bg_mask]
                local_energy = float(np.mean(local_energy_vals)) if len(local_energy_vals) > 0 else 0.0
                global_energy = global_gabor_energies[idx]
                
                g_ratio = local_energy / global_energy if global_energy > 0 else 1.0
                g_ratios.append(round(g_ratio, 3))
                if g_ratio < 0.30:  # Guilloche pattern disrupted/erased locally
                    triggered_check = "gabor"
                    trigger_ratio = g_ratio
                    break
        field_result = {
            "field_name": field_name,
            "bbox": [x, y, w, h],
            "local_variance": round(local_variance, 3),
            "global_variance": round(global_variance, 3),
            "ratio": round(trigger_ratio, 3),
        }
        
        if triggered_check:
            field_result["trigger"] = triggered_check
            flagged_fields.append(field_result)
        else:
            clean_fields.append(field_result)
            
    # 6. Visualize results: Draw rectangles on image
    vis_img = image.copy()
    for field in flagged_fields:
        x, y, w, h = field["bbox"]
        # Red box for flagged
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{field['field_name']}: {field['ratio']:.2f} ({field['trigger']})"
        cv2.putText(vis_img, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
    for field in clean_fields:
        x, y, w, h = field["bbox"]
        # Green box for clean
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        label = f"{field['field_name']}: {field['ratio']:.2f}"
        cv2.putText(vis_img, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
    # Encode visualization image
    _, buffer = cv2.imencode(".png", vis_img)
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")
    
    is_flagged = len(flagged_fields) > 0
    status = "flagged" if is_flagged else "clean"
    penalty_weight = 0.65 if is_flagged else 0.0
    
    flags = []
    if is_flagged:
        flagged_details = ", ".join([f"{f['field_name']} (ratio: {f['ratio']:.3f})" for f in flagged_fields])
        flags.append({
            "check": "background_pattern_disruption",
            "severity": "high",
            "message": f"Background pattern disruption detected on: {flagged_details}.",
            "evidence": {
                "flagged_fields": flagged_fields
            }
        })
        
    return {
        "check": "background_pattern_disruption",
        "label": "Background Pattern Disruption",
        "status": status,
        "penalty_applied": penalty_weight,
        "penalty_weight": penalty_weight,
        "flagged_fields": flagged_fields,
        "global_variance": round(global_variance, 3),
        "skipped": False,
        "skip_reason": None,
        "heatmap_b64": heatmap_b64,
        "mean_value": len(flagged_fields),
        "flags": flags,
        "description": (
            "Detects edits behind high-value numeric fields by analyzing local background "
            "texture variance and Gabor energy against the global document background. "
            "Erased or disrupted textures are flagged as tampered."
        )
    }


def detect_tampering(image_path: str, ocr_word_boxes: list = None, document_type: str = None) -> dict:
    """
    Run all visual tampering checks on a document image.
    
    Parameters
    ----------
    image_path : str
        Path to a JPEG, PNG, or TIFF image file.
    ocr_word_boxes : list, optional
        List of (x, y, w, h, text) tuples from OCR. Passed to font weight check.
    document_type : str, optional
        The classification type of the document (e.g., 'bank_statement'). Used
        to conditionally enable/disable document-specific checks.
    
    Returns
    -------
    dict
        {
            "checks":          list[dict]  — one result dict per check,
            "flags":           list[dict]  — all flags across all checks,
            "heatmaps":        dict        — { check_name: base64_png },
            "summary":         str         — human readable summary,
            "risk_level":      str         — "clean" / "suspicious" / "high_risk",
            "penalty_weight":  float       — highest penalty weight found across all sub-checks
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
            "penalty_weight": 0.0,
        }
    
    # Run all sub-checks
    results = []
    all_flags = []
    heatmaps = {}
    
    checks_to_run = [
        ("Halftone",   lambda: _check_halftone_runner(image_path, document_type)),
        ("FontWeight", lambda: _check_font_weight_runner(image_path, ocr_word_boxes)),
        ("BackgroundPatternDisruption", lambda: check_background_pattern_disruption(cv_img, ocr_word_boxes)),
        ("Blur",       lambda: _check_blur_inconsistency(cv_img)),
        ("Noise",      lambda: _check_noise_inconsistency(cv_img)),
        ("Artifacts",  lambda: _check_pixel_artifacts(cv_img)),
        ("CopyPaste",  lambda: _check_copy_paste(cv_img)),
        ("FaceRegion", lambda: _check_face_region_tampering(cv_img)),
    ]
    
    for check_name, check_fn in checks_to_run:
        try:
            result = check_fn()
            if result is None:
                logger.info("%s check: skipped (no face detected/applicable context)", check_name)
                continue
            results.append(result)
            all_flags.extend(result.get("flags", []))
            if result.get("heatmap_b64"):
                heatmaps[result["check"]] = result["heatmap_b64"]
            logger.info("%s check: %d flag(s)", check_name, len(result.get("flags", [])))
        except Exception as exc:
            logger.error("%s check failed: %s", check_name, exc)
    
    # Determine overall risk level based on flags
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
    
    # Determine the overall highest penalty weight across all checks
    # For checks without explicit penalty_weight, map severity of their flags
    all_penalty_weights = [0.0]
    for r in results:
        if "penalty_weight" in r:
            all_penalty_weights.append(r["penalty_weight"])
        elif r.get("flags"):
            severities = [f.get("severity", "low").lower() for f in r["flags"]]
            if "high" in severities:
                all_penalty_weights.append(0.60)
            elif "medium" in severities:
                all_penalty_weights.append(0.35)
            else:
                all_penalty_weights.append(0.10)
                
    highest_penalty_weight = max(all_penalty_weights)
    
    logger.info("Tampering analysis complete — risk level: %s, penalty weight: %.2f", 
                risk_level, highest_penalty_weight)
    
    return {
        "checks": results,
        "flags": all_flags,
        "heatmaps": heatmaps,
        "summary": summary,
        "risk_level": risk_level,
        "penalty_weight": highest_penalty_weight,
    }


def detect_tampering_from_pdf_page(pdf_path: str, page_number: int = 0, ocr_word_boxes: list = None, document_type: str = None) -> dict:
    """
    Render a single PDF page to an image and run tampering analysis.
    
    Parameters
    ----------
    pdf_path        : str — path to the PDF file
    page_number     : int — zero-indexed page number (default: first page)
    ocr_word_boxes  : list, optional — list of word bounding boxes
    document_type   : str, optional — the document classification label
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
        result = detect_tampering(tmp_path, ocr_word_boxes, document_type)
        
        # Clean up
        os.unlink(tmp_path)
        
        return result
        
    except Exception as exc:
        logger.error("PDF page render failed: %s", exc)
        return {
            "checks": [],
            "flags": [],
            "heatmaps": {},
            "summary": f"Could not analyse PDF page: {exc}",
            "risk_level": "unknown",
            "penalty_weight": 0.0,
        }


