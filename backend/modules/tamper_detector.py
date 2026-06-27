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
    6. NEW: Regional ELA analysis — check if any quadrant has significantly
       higher ELA than others (catches localized edits like photo swaps)
    """
    ELA_QUALITY = 90
    ELA_AMPLIFY = 10        # multiply difference for visual clarity
    FLAG_THRESHOLD = 13.5   # mean ELA value above this → suspicious (adjusted up from 12.0)
    HIGH_THRESHOLD = 22.0   # above this → high severity (adjusted up from 20.0)
    REGIONAL_RATIO = 3.5    # if any quadrant's mean > 3.5× lowest → localized edit (adjusted up from 3.0)
    # NOTE: Photographed physical ID cards have natural ELA variation from
    # JPEG compression of mixed content (photo, text, QR). Previous thresholds
    # (10.0 / 17.0 / 2.2) caused false positives on genuine cards.
    
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
    
    # Step 6: Regional ELA analysis — divide into quadrants
    h, w = ela_map.shape
    mid_h, mid_w = h // 2, w // 2
    quadrants = [
        ela_map[:mid_h, :mid_w],      # top-left
        ela_map[:mid_h, mid_w:],       # top-right
        ela_map[mid_h:, :mid_w],       # bottom-left
        ela_map[mid_h:, mid_w:],       # bottom-right
    ]
    quadrant_means = [float(q.mean()) for q in quadrants]
    min_quadrant = max(min(quadrant_means), 0.1)  # avoid division by zero
    max_quadrant = max(quadrant_means)
    regional_ratio = max_quadrant / min_quadrant
    
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
    
    # Regional ELA flag — catches localized edits that global mean misses
    if regional_ratio > REGIONAL_RATIO and not flags:
        severity = "high" if regional_ratio > 4.0 else "medium"
        flags.append({
            "check": "tampering_ela",
            "severity": severity,
            "message": (
                f"Regional ELA analysis detected localized compression inconsistency. "
                f"One region has {regional_ratio:.1f}× higher ELA than the cleanest region, "
                f"suggesting a localized edit (e.g., photo replacement or text insertion)."
            ),
            "evidence": {
                "mean_ela_value": round(mean_ela, 2),
                "regional_ratio": round(regional_ratio, 2),
                "quadrant_means": [round(q, 2) for q in quadrant_means],
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
    ELA_RATIO_THRESHOLD = 2.4      # adjusted up from 2.2 to be more sensitive to photo ELA variations
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
    
    # --- Metric B: ELA intensity ratio ---
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")
    diff = np.array(ImageChops.difference(pil_img, resaved)).astype(np.float32).mean(axis=2)
    
    face_ela_mean = float(diff[face_mask].mean()) if face_mask.any() else 0
    bg_ela_mean = float(diff[bg_mask].mean()) if bg_mask.any() else 1e-6
    ela_ratio = face_ela_mean / max(bg_ela_mean, 1e-6)
    
    sub_metrics["ela_ratio"] = round(ela_ratio, 2)
    if ela_ratio > ELA_RATIO_THRESHOLD:
        anomaly_count += 1
    
    # --- Metric C: Edge discontinuity at face boundary ---
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
    
    # --- Metric D: Color statistics mismatch ---
    face_lum_std = float(gray_f[face_mask].std()) if face_mask.any() else 0
    bg_lum_std = float(gray_f[bg_mask].std()) if bg_mask.any() else 1e-6
    max_std = max(face_lum_std, bg_lum_std, 1e-6)
    lum_diff = abs(face_lum_std - bg_lum_std) / max_std
    
    sub_metrics["luminance_diff"] = round(lum_diff, 3)
    if lum_diff > LUMINANCE_DIFF_THRESHOLD:
        anomaly_count += 1
    
    # --- Metric E: JPEG compression quality difference ---
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
    face_map[face_mask] = float(anomaly_count) / 5.0 * 255
    face_map[boundary_mask] = min(boundary_gradient / EDGE_STRENGTH_THRESHOLD, 1.0) * 255
    
    # --- Build flags ---
    flags = []
    if anomaly_count >= 2:
        severity = "high" if anomaly_count >= 3 else "medium"
        triggered = []
        if noise_ratio > NOISE_RATIO_THRESHOLD:
            triggered.append(f"noise ({noise_ratio:.1f}×)")
        if ela_ratio > ELA_RATIO_THRESHOLD:
            triggered.append(f"ELA ({ela_ratio:.1f}×)")
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
                f"Face region analysis detected {anomaly_count}/5 tampering indicators "
                f"({', '.join(triggered)}). The photo area shows characteristics "
                f"inconsistent with the rest of the document, suggesting the "
                f"photo may have been replaced or digitally inserted."
            ),
            "evidence": {
                "anomaly_count": f"{anomaly_count}/5",
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
        ("ELA",        lambda: _check_ela(pil_img, cv_img)),
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


