"""
FFT Halftone Analysis for ForeSight.
Analyzes the spatial frequency of the document to identify halftone print signatures.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_halftone(image_path: str) -> dict:
    """
    Analyze the halftone structure of the document image using 2D Fast Fourier Transform (FFT).
    
    Professional offset lithography produces regular rosette patterns at high spatial
    frequencies (resulting in sharp peaks in the Fourier magnitude spectrum),
    whereas inkjet printing uses irregular/stochastic dot distributions (resulting in a
    diffuse/flat spectrum).
    
    This signal is STRONGER when a document has been photocopied then edited — the photocopy
    adds a second-generation halftone and the edited region has a different (third-generation)
    frequency signature.
    
    Parameters
    ----------
    image_path : str
        Path to the document image file.
        
    Returns
    -------
    dict
        {
            "peak_sharpness_ratio": float,
            "strong_peak_count": int,
            "likely_professional_print": bool,
            "risk": "low" | "medium" | "high",
            "penalty_weight": float,
            "interpretation": str
        }
    """
    logger.info("Running FFT halftone analysis on: %s", image_path)
    
    # Load image as grayscale using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("Failed to load image for FFT halftone analysis: %s", image_path)
        return {
            "peak_sharpness_ratio": 0.0,
            "strong_peak_count": 0,
            "likely_professional_print": False,
            "risk": "high",
            "penalty_weight": 0.55,
            "interpretation": f"Error: Image could not be loaded from {image_path}."
        }
        
    h, w = img.shape
    
    # Crop to text-heavy region: take the middle 50% of height and width (avoid blank margins)
    start_y = int(h * 0.25)
    end_y = int(h * 0.75)
    start_x = int(w * 0.25)
    end_x = int(w * 0.75)
    patch = img[start_y:end_y, start_x:end_x]
    
    # Handle extremely small image edge case
    if patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
        return {
            "peak_sharpness_ratio": 0.0,
            "strong_peak_count": 0,
            "likely_professional_print": False,
            "risk": "medium",
            "penalty_weight": 0.30,
            "interpretation": "Image patch size is too small for frequency analysis."
        }
        
    # Run np.fft.fft2 on the patch
    f_coef = np.fft.fft2(patch)
    
    # Apply fftshift and compute log magnitude spectrum
    f_shift = np.fft.fftshift(f_coef)
    magnitude = np.log(np.abs(f_shift) + 1.0)
    
    # Zero out a 10x10 region at the DC component centre (center frequency is at shape // 2)
    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    magnitude[cy - 5 : cy + 5, cx - 5 : cx + 5] = 0.0
    
    # Compute peak_sharpness_ratio: max(magnitude) / mean(magnitude)
    max_mag = float(np.max(magnitude))
    mean_mag = float(np.mean(magnitude))
    
    # Avoid division by zero
    peak_sharpness_ratio = max_mag / mean_mag if mean_mag > 0 else 0.0
    
    # Count strong_peaks: pixels above the 99th percentile
    percentile_99 = float(np.percentile(magnitude, 99))
    strong_peak_count = int(np.sum(magnitude > percentile_99))
    
    # Professional print signature: True if ratio > 8.0 AND peaks > 20
    likely_professional_print = (peak_sharpness_ratio > 8.0) and (strong_peak_count > 20)
    
    # Risk logic:
    # - ratio < 4.0 AND peaks < 10: risk "high", weight 0.55
    #   (inkjet/recreated print signature)
    # - ratio 4.0-8.0: risk "medium", weight 0.30
    # - ratio > 8.0: risk "low", weight 0.0
    # Fallback/Else: If ratio < 4.0 but peaks >= 10, or peaks >= 10 but ratio <= 8.0, map to medium risk.
    if peak_sharpness_ratio < 4.0 and strong_peak_count < 10:
        risk = "high"
        penalty_weight = 0.55
        interpretation = (
            "High risk: FFT halftone analysis indicates inkjet or digital-recreation print signature "
            "(sharpness ratio: {:.2f}, strong peak count: {}). The regular frequency rosette patterns "
            "typical of professional offset lithography are absent."
        ).format(peak_sharpness_ratio, strong_peak_count)
    elif peak_sharpness_ratio > 8.0:
        risk = "low"
        penalty_weight = 0.0
        interpretation = (
            "Low risk: FFT halftone analysis indicates professional offset lithography "
            "(sharpness ratio: {:.2f}, strong peak count: {}). Regular rosette patterns detected."
        ).format(peak_sharpness_ratio, strong_peak_count)
    else:
        risk = "medium"
        penalty_weight = 0.30
        interpretation = (
            "Medium risk: FFT halftone analysis indicates moderate peak sharpness "
            "(sharpness ratio: {:.2f}, strong peak count: {}). This may be due to "
            "inkjet printing, scanning degradation, or a photocopy reprint."
        ).format(peak_sharpness_ratio, strong_peak_count)
        
    return {
        "peak_sharpness_ratio": round(peak_sharpness_ratio, 4),
        "strong_peak_count": strong_peak_count,
        "likely_professional_print": likely_professional_print,
        "risk": risk,
        "penalty_weight": penalty_weight,
        "interpretation": interpretation
    }
